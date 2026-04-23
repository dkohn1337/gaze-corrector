"""Iris offset calculation and head pose estimation via solvePnP."""

from dataclasses import dataclass
import numpy as np
import cv2
import config


@dataclass
class GazeInfo:
    """Gaze estimation results for a single frame."""
    # Per-eye iris offset normalized by eye dimensions, range ~ [-1, 1]
    left_iris_offset: np.ndarray   # (2,) [horizontal, vertical]
    right_iris_offset: np.ndarray  # (2,) [horizontal, vertical]

    # Head pose in degrees
    yaw: float
    pitch: float
    roll: float

    # Combined gaze angle (magnitude of iris offset, degrees approx)
    gaze_angle: float

    # Correction vectors (how much to shift iris to look at camera)
    left_correction: np.ndarray   # (2,) pixel shift
    right_correction: np.ndarray  # (2,) pixel shift


def _eye_geometry(landmarks: np.ndarray, contour_indices: list[int],
                  iris_center_idx: int) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Compute eye center, dimensions, and iris offset.

    Returns:
        (eye_center_2d, eye_width, eye_height, iris_offset_normalized)
    """
    contour = landmarks[contour_indices, :2]
    eye_center = contour.mean(axis=0)

    x_coords = contour[:, 0]
    y_coords = contour[:, 1]
    eye_width = x_coords.max() - x_coords.min()
    eye_height = y_coords.max() - y_coords.min()

    iris_center = landmarks[iris_center_idx, :2]
    offset = iris_center - eye_center

    # Normalize by eye dimensions
    norm_offset = np.array([
        offset[0] / max(eye_width, 1.0),
        offset[1] / max(eye_height, 1.0),
    ])

    return eye_center, eye_width, eye_height, norm_offset


def _estimate_head_pose(landmarks: np.ndarray,
                        frame_width: int,
                        frame_height: int) -> tuple[float, float, float]:
    """Estimate head yaw/pitch/roll using solvePnP."""
    image_points = landmarks[config.HEAD_POSE_LANDMARKS, :2].astype(np.float64)
    model_points = np.array(config.MODEL_POINTS_3D, dtype=np.float64)

    # Approximate camera matrix
    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix to Euler angles
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = 0.0

    return yaw, pitch, roll


def estimate_gaze(landmarks: np.ndarray,
                  frame_width: int,
                  frame_height: int) -> GazeInfo:
    """Compute full gaze information from face landmarks."""
    # Eye geometry
    l_center, l_w, l_h, l_offset = _eye_geometry(
        landmarks, config.LEFT_EYE_CONTOUR, config.LEFT_IRIS_CENTER,
    )
    r_center, r_w, r_h, r_offset = _eye_geometry(
        landmarks, config.RIGHT_EYE_CONTOUR, config.RIGHT_IRIS_CENTER,
    )

    # Head pose
    yaw, pitch, roll = _estimate_head_pose(landmarks, frame_width, frame_height)

    # Gaze angle: approximate from average iris offset magnitude
    avg_offset = (np.abs(l_offset) + np.abs(r_offset)) / 2.0
    # Convert normalized offset to rough degrees (empirical: full offset ~ 30°)
    gaze_angle = float(np.linalg.norm(avg_offset) * 30.0)

    # Correction vectors: shift iris toward eye center
    # Target is (0, 0) normalized offset; dampen vertical
    l_correction = np.array([
        -l_offset[0] * l_w,
        -l_offset[1] * l_h * config.VERTICAL_DAMPING,
    ])
    r_correction = np.array([
        -r_offset[0] * r_w,
        -r_offset[1] * r_h * config.VERTICAL_DAMPING,
    ])

    return GazeInfo(
        left_iris_offset=l_offset,
        right_iris_offset=r_offset,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        gaze_angle=gaze_angle,
        left_correction=l_correction,
        right_correction=r_correction,
    )
