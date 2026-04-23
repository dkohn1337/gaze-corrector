"""Eye-region warping to redirect gaze toward the camera."""

import cv2
import numpy as np

import config
from gaze_estimator import GazeInfo


def _get_eye_roi(frame: np.ndarray, landmarks: np.ndarray,
                 contour_indices: list[int]) -> tuple[np.ndarray, int, int, int, int]:
    """Extract padded eye ROI bounding box.

    Returns:
        (roi, x1, y1, x2, y2) — the cropped region and its coords in the frame.
    """
    h, w = frame.shape[:2]
    contour = landmarks[contour_indices, :2]

    x_min, y_min = contour.min(axis=0)
    x_max, y_max = contour.max(axis=0)
    eye_w = x_max - x_min
    eye_h = y_max - y_min

    pad_x = eye_w * config.EYE_ROI_PAD_HORIZONTAL
    pad_y = eye_h * config.EYE_ROI_PAD_VERTICAL

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    return frame[y1:y2, x1:x2].copy(), x1, y1, x2, y2


def _build_affine_warp(roi: np.ndarray,
                       landmarks: np.ndarray,
                       iris_indices: list[int],
                       contour_indices: list[int],
                       correction: np.ndarray,
                       strength: float,
                       roi_x: int, roi_y: int) -> np.ndarray:
    """Apply a 3-point affine warp to shift the iris region.

    Uses eye corners as anchors and iris center as the moved point.
    """
    roi_h, roi_w = roi.shape[:2]

    # Key points in ROI coordinates
    iris_center = landmarks[iris_indices[0], :2] - np.array([roi_x, roi_y])

    # Use first and last contour points as anchors (roughly left/right eye corners)
    anchor1 = landmarks[contour_indices[0], :2] - np.array([roi_x, roi_y])
    anchor2 = landmarks[contour_indices[8], :2] - np.array([roi_x, roi_y])  # Opposite corner

    src_pts = np.float32([anchor1, anchor2, iris_center])
    shifted_iris = iris_center + correction * strength
    dst_pts = np.float32([anchor1, anchor2, shifted_iris])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(roi, M, (roi_w, roi_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    return warped


def _build_piecewise_warp(roi: np.ndarray,
                          landmarks: np.ndarray,
                          iris_indices: list[int],
                          contour_indices: list[int],
                          correction: np.ndarray,
                          strength: float,
                          roi_x: int, roi_y: int) -> np.ndarray:
    """Piecewise affine warp via Delaunay triangulation.

    Eye contour points stay anchored; iris points shift by correction.
    """
    roi_h, roi_w = roi.shape[:2]
    offset = np.array([roi_x, roi_y])

    # Source control points: contour (anchored) + iris (to be shifted)
    contour_pts = landmarks[contour_indices, :2] - offset
    iris_pts = landmarks[iris_indices, :2] - offset

    # Add ROI corner points as anchors
    corners = np.array([
        [0, 0], [roi_w - 1, 0],
        [0, roi_h - 1], [roi_w - 1, roi_h - 1],
    ], dtype=np.float64)

    src_points = np.vstack([contour_pts, iris_pts, corners])

    # Destination: contour and corners stay, iris shifts
    shifted_iris = iris_pts + correction * strength
    dst_points = np.vstack([contour_pts, shifted_iris, corners])

    # Delaunay triangulation on source points
    rect = (0, 0, roi_w, roi_h)
    subdiv = cv2.Subdiv2D(rect)

    # Clamp points to rect
    src_clamped = src_points.copy()
    src_clamped[:, 0] = np.clip(src_clamped[:, 0], 0, roi_w - 1)
    src_clamped[:, 1] = np.clip(src_clamped[:, 1], 0, roi_h - 1)

    for pt in src_clamped:
        try:
            subdiv.insert((float(pt[0]), float(pt[1])))
        except cv2.error:
            continue

    triangles = subdiv.getTriangleList()
    warped = np.zeros_like(roi)

    for tri in triangles:
        pts_src = np.array([[tri[0], tri[1]], [tri[2], tri[3]], [tri[4], tri[5]]])

        # Skip triangles outside ROI
        if (pts_src[:, 0].min() < -1 or pts_src[:, 0].max() > roi_w or
                pts_src[:, 1].min() < -1 or pts_src[:, 1].max() > roi_h):
            continue

        # Find matching destination points
        pts_dst = np.zeros_like(pts_src)
        valid = True
        for i, pt in enumerate(pts_src):
            dists = np.linalg.norm(src_clamped - pt, axis=1)
            idx = np.argmin(dists)
            if dists[idx] > 2.0:  # No matching control point
                valid = False
                break
            pts_dst[i] = dst_points[idx]

        if not valid:
            continue

        # Warp triangle
        r_src = cv2.boundingRect(np.float32([pts_src]))
        r_dst = cv2.boundingRect(np.float32([pts_dst]))

        # Crop triangles to their bounding rects
        tri_src_cropped = pts_src - np.array([r_src[0], r_src[1]])
        tri_dst_cropped = pts_dst - np.array([r_dst[0], r_dst[1]])

        # Extract source region
        x, y, w, h = r_src
        x, y = max(0, x), max(0, y)
        w = min(w, roi_w - x)
        h = min(h, roi_h - y)
        if w <= 0 or h <= 0:
            continue
        src_crop = roi[y:y + h, x:x + w]

        if src_crop.size == 0:
            continue

        M = cv2.getAffineTransform(
            np.float32(tri_src_cropped),
            np.float32(tri_dst_cropped),
        )
        dst_crop = cv2.warpAffine(
            src_crop, M, (r_dst[2], r_dst[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Create mask for the triangle
        mask = np.zeros((r_dst[3], r_dst[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri_dst_cropped), 255)

        # Paste into warped image
        dx, dy = r_dst[0], r_dst[1]
        dx, dy = max(0, dx), max(0, dy)
        dw = min(r_dst[2], roi_w - dx)
        dh = min(r_dst[3], roi_h - dy)
        if dw <= 0 or dh <= 0:
            continue

        mask_region = mask[:dh, :dw]
        dst_region = dst_crop[:dh, :dw]
        mask_bool = mask_region > 0
        if mask_bool.any():
            warped_region = warped[dy:dy + dh, dx:dx + dw]
            for c in range(3):
                warped_region[:, :, c] = np.where(
                    mask_bool, dst_region[:, :, c], warped_region[:, :, c],
                )

    return warped


def _blend_roi(frame: np.ndarray, warped_roi: np.ndarray,
               landmarks: np.ndarray, contour_indices: list[int],
               x1: int, y1: int, x2: int, y2: int,
               blend_factor: float) -> np.ndarray:
    """Blend warped eye ROI back into the frame with feathered edges."""
    if blend_factor <= 0.0:
        return frame

    roi_h = y2 - y1
    roi_w = x2 - x1
    offset = np.array([x1, y1])

    # Build convex hull mask from eye contour
    contour_pts = landmarks[contour_indices, :2] - offset
    contour_pts = np.clip(contour_pts, 0, [roi_w - 1, roi_h - 1]).astype(np.int32)
    hull = cv2.convexHull(contour_pts)

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Feather edges
    ksize = config.BLEND_FEATHER_KSIZE
    if ksize % 2 == 0:
        ksize += 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask_float = (mask.astype(np.float32) / 255.0) * blend_factor

    # Blend
    original_roi = frame[y1:y2, x1:x2].astype(np.float32)
    warped_f = warped_roi.astype(np.float32)
    mask_3 = mask_float[:, :, np.newaxis]

    blended = original_roi * (1.0 - mask_3) + warped_f * mask_3
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    return frame


def correct_gaze(frame: np.ndarray,
                 landmarks: np.ndarray,
                 gaze: GazeInfo,
                 strength: float,
                 blend_factor: float,
                 use_piecewise: bool = False) -> np.ndarray:
    """Apply gaze correction to both eyes.

    Args:
        frame: BGR image to modify in-place.
        landmarks: (478, 3) smoothed landmarks.
        gaze: Gaze estimation results with correction vectors.
        strength: Correction strength (0.0 to 1.0).
        blend_factor: FSM blend factor (0.0 = off, 1.0 = full).
        use_piecewise: Use piecewise affine (higher quality, slower).

    Returns:
        Modified frame.
    """
    effective_blend = blend_factor * strength
    if effective_blend < 0.01:
        return frame

    result = frame.copy()

    warp_fn = _build_piecewise_warp if use_piecewise else _build_affine_warp

    # Process each eye
    for contour_idx, iris_idx, correction in [
        (config.LEFT_EYE_CONTOUR, config.LEFT_IRIS_INDICES, gaze.left_correction),
        (config.RIGHT_EYE_CONTOUR, config.RIGHT_IRIS_INDICES, gaze.right_correction),
    ]:
        roi, x1, y1, x2, y2 = _get_eye_roi(result, landmarks, contour_idx)
        if roi.size == 0:
            continue

        warped = warp_fn(
            roi, landmarks, iris_idx, contour_idx,
            correction, strength, x1, y1,
        )

        result = _blend_roi(
            result, warped, landmarks, contour_idx,
            x1, y1, x2, y2, blend_factor,
        )

    return result
