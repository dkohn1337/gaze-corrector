"""Threaded capture → process → output pipeline."""

import threading
import time
from queue import Queue, Full

import cv2
import numpy as np

import config
from face_detector import FaceDetector
from gaze_estimator import estimate_gaze, GazeInfo
from gaze_corrector import correct_gaze
from behavior_fsm import BehaviorFSM, State
from smoothing import LandmarkSmoother, CorrectionSmoother, EMAFilter
from virtual_camera import VirtualCameraOutput


class DropOldestQueue:
    """Thread-safe bounded queue that drops the oldest item on overflow."""

    def __init__(self, maxsize: int):
        self._queue = Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            self._queue.put_nowait(item)

    def get(self, timeout: float = 1.0):
        return self._queue.get(timeout=timeout)

    def empty(self):
        return self._queue.empty()


class Pipeline:
    """Three-thread pipeline: capture → process → output."""

    def __init__(self, *,
                 enable_vcam: bool = True,
                 enable_preview: bool = False,
                 correction_strength: float = config.DEFAULT_CORRECTION_STRENGTH,
                 use_piecewise: bool = False,
                 camera_index: int = config.CAMERA_INDEX):
        self._enable_vcam = enable_vcam
        self._enable_preview = enable_preview
        self._correction_strength = correction_strength
        self._use_piecewise = use_piecewise
        self._camera_index = camera_index

        self._running = False
        self._enabled = True  # Correction enabled toggle

        # Queues
        self._capture_q = DropOldestQueue(config.QUEUE_MAX_SIZE)
        self._output_q = DropOldestQueue(config.QUEUE_MAX_SIZE)

        # Threads
        self._capture_thread = None
        self._process_thread = None
        self._output_thread = None

        # Correction smoothing for output display
        self._correction_smoother = CorrectionSmoother(config.CORRECTION_EMA_ALPHA)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def correction_strength(self) -> float:
        return self._correction_strength

    @correction_strength.setter
    def correction_strength(self, value: float):
        self._correction_strength = max(0.0, min(1.0, value))

    def start(self):
        """Start the pipeline threads."""
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture",
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="process",
        )
        self._output_thread = threading.Thread(
            target=self._output_loop, daemon=True, name="output",
        )

        self._capture_thread.start()
        self._process_thread.start()
        self._output_thread.start()

    def stop(self):
        """Signal all threads to stop."""
        self._running = False

    def wait(self):
        """Wait for all threads to finish."""
        for t in [self._capture_thread, self._process_thread, self._output_thread]:
            if t is not None:
                t.join(timeout=5.0)

    def _capture_loop(self):
        """Capture frames from the webcam."""
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            self._running = False
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue
                self._capture_q.put(frame)
        finally:
            cap.release()

    def _process_loop(self):
        """Process frames: face detection, gaze estimation, correction."""
        detector = FaceDetector()
        fsm = BehaviorFSM()
        landmark_smoother = LandmarkSmoother(config.LANDMARK_EMA_ALPHA)
        l_corr_smoother = EMAFilter(config.CORRECTION_EMA_ALPHA)
        r_corr_smoother = EMAFilter(config.CORRECTION_EMA_ALPHA)

        try:
            while self._running:
                try:
                    frame = self._capture_q.get(timeout=0.5)
                except Exception:
                    continue

                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect face
                raw_landmarks = detector.detect(frame_rgb)

                if raw_landmarks is None or not self._enabled:
                    # No face or correction disabled: pass through
                    self._output_q.put((frame, None, None, 0.0))
                    continue

                # Smooth landmarks
                landmarks = landmark_smoother.smooth(raw_landmarks)

                # Estimate gaze
                gaze = estimate_gaze(landmarks, w, h)

                # Smooth correction vectors
                gaze.left_correction = l_corr_smoother.update(gaze.left_correction)
                gaze.right_correction = r_corr_smoother.update(gaze.right_correction)

                # FSM update
                raw_blend = fsm.update(gaze)
                blend = self._correction_smoother.smooth(raw_blend)

                # Apply gaze correction
                corrected = correct_gaze(
                    frame, landmarks, gaze,
                    self._correction_strength, blend,
                    self._use_piecewise,
                )

                self._output_q.put((corrected, landmarks, gaze, blend))
        finally:
            detector.close()

    def _output_loop(self):
        """Output frames to virtual camera and/or preview window."""
        vcam = None
        if self._enable_vcam:
            try:
                vcam = VirtualCameraOutput()
                vcam.start()
            except Exception as e:
                print(f"WARNING: Could not start virtual camera: {e}")
                print("         Running without virtual camera output.")
                vcam = None

        try:
            while self._running:
                try:
                    frame, landmarks, gaze, blend = self._output_q.get(timeout=0.5)
                except Exception:
                    continue

                # Send to virtual camera
                if vcam is not None:
                    vcam.send(frame)

                # Preview window
                if self._enable_preview:
                    preview = frame.copy()
                    self._draw_debug(preview, landmarks, gaze, blend)
                    cv2.imshow(config.PREVIEW_WINDOW_NAME, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self._running = False
                        break
        finally:
            if vcam is not None:
                vcam.stop()
            if self._enable_preview:
                cv2.destroyAllWindows()

    def _draw_debug(self, frame: np.ndarray,
                    landmarks: np.ndarray | None,
                    gaze: GazeInfo | None,
                    blend: float):
        """Draw debug overlays on the preview frame."""
        h, w = frame.shape[:2]

        if landmarks is not None:
            # Draw iris markers
            for idx in config.LEFT_IRIS_INDICES + config.RIGHT_IRIS_INDICES:
                pt = tuple(landmarks[idx, :2].astype(int))
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # Draw eye contours
            for contour in [config.LEFT_EYE_CONTOUR, config.RIGHT_EYE_CONTOUR]:
                pts = landmarks[contour, :2].astype(int)
                cv2.polylines(frame, [pts], True, (255, 200, 0), 1)

        if gaze is not None:
            # Text overlay
            info_lines = [
                f"Gaze angle: {gaze.gaze_angle:.1f} deg",
                f"Head yaw: {gaze.yaw:.1f}  pitch: {gaze.pitch:.1f}",
                f"Blend: {blend:.2f}",
                f"Strength: {self._correction_strength:.0%}",
                f"Enabled: {self._enabled}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            # Correction vectors
            for contour, correction in [
                (config.LEFT_EYE_CONTOUR, gaze.left_correction),
                (config.RIGHT_EYE_CONTOUR, gaze.right_correction),
            ]:
                eye_center = landmarks[contour, :2].mean(axis=0).astype(int)
                end_pt = (eye_center + correction * 3).astype(int)
                cv2.arrowedLine(frame, tuple(eye_center), tuple(end_pt),
                                (0, 0, 255), 2, tipLength=0.3)
