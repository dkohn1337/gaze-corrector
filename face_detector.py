"""MediaPipe FaceMesh wrapper returning 478 landmarks (468 face + 10 iris)."""

import numpy as np
import mediapipe as mp
import config


class FaceDetector:
    """Wraps MediaPipe FaceMesh for face and iris landmark detection."""

    def __init__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Detect face landmarks in an RGB frame.

        Args:
            frame_rgb: RGB image (H, W, 3), uint8.

        Returns:
            Landmarks as (478, 3) float array with (x, y, z) in pixel coords,
            or None if no face detected.
        """
        h, w = frame_rgb.shape[:2]
        results = self._mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        landmarks = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in face.landmark],
            dtype=np.float64,
        )
        return landmarks

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
