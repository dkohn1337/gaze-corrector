"""pyvirtualcam output wrapper for OBS Virtual Camera."""

import numpy as np
import pyvirtualcam

import config


class VirtualCameraOutput:
    """Manages the virtual camera output via pyvirtualcam."""

    def __init__(self, width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT,
                 fps: int = config.CAMERA_FPS):
        self._width = width
        self._height = height
        self._fps = fps
        self._cam = None

    def start(self):
        """Open the virtual camera."""
        self._cam = pyvirtualcam.Camera(
            width=self._width,
            height=self._height,
            fps=self._fps,
            print_fps=False,
        )
        print(f"Virtual camera started: {self._cam.device}")

    def send(self, frame_bgr: np.ndarray):
        """Send a BGR frame to the virtual camera (converts to RGB)."""
        if self._cam is None:
            return
        # pyvirtualcam expects RGB on macOS
        frame_rgb = frame_bgr[:, :, ::-1]
        # Resize if needed
        if (frame_rgb.shape[1] != self._width or
                frame_rgb.shape[0] != self._height):
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self._width, self._height))
        self._cam.send(frame_rgb)

    def stop(self):
        """Close the virtual camera."""
        if self._cam is not None:
            self._cam.close()
            self._cam = None
            print("Virtual camera stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
