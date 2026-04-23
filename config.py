"""All tunable constants for the gaze correction system."""

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# --- Pipeline ---
QUEUE_MAX_SIZE = 2  # Bounded queue, drop-oldest

# --- MediaPipe FaceMesh ---
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True  # Enables iris landmarks (468-477)

# --- Landmark indices ---
# Iris
LEFT_IRIS_CENTER = 468
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Eye contours (16 points each)
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Head pose estimation (6 key points)
HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

# 3D model points for solvePnP (generic face model, in mm)
# Nose tip, right eye corner, left eye corner, right mouth, left mouth, chin
MODEL_POINTS_3D = [
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (-225.0, 170.0, -135.0),   # Right eye right corner (landmark 33)
    (225.0, 170.0, -135.0),    # Left eye left corner (landmark 263)
    (-150.0, -150.0, -125.0),  # Right mouth corner (landmark 61)
    (150.0, -150.0, -125.0),   # Left mouth corner (landmark 291)
    (0.0, -330.0, -65.0),      # Chin (landmark 199)
]

# --- Gaze Estimation ---
VERTICAL_DAMPING = 0.5  # Reduce vertical correction (looking down at screen is normal)

# --- Gaze Correction ---
DEFAULT_CORRECTION_STRENGTH = 0.7  # 70% default
EYE_ROI_PAD_HORIZONTAL = 0.3  # 30% horizontal padding around eye
EYE_ROI_PAD_VERTICAL = 0.5    # 50% vertical padding around eye
BLEND_FEATHER_KSIZE = 15       # Gaussian blur kernel for feathered blending

# --- Behavior FSM ---
# Gaze angle thresholds (degrees)
ENGAGE_THRESHOLD = 15.0      # Gaze < 15° → start re-engaging
DISENGAGE_THRESHOLD = 25.0   # Gaze > 25° → start disengaging

# Head pose thresholds (degrees)
HEAD_YAW_THRESHOLD = 20.0    # Head yaw > 20° → force disengage
HEAD_PITCH_THRESHOLD = 15.0  # Head pitch > 15° → force disengage

# Transition durations (seconds)
DISENGAGE_DURATION = 0.4  # Fade out over 0.4s
RE_ENGAGE_DURATION = 0.2  # Fade in over 0.2s

# --- Smoothing ---
LANDMARK_EMA_ALPHA = 0.6   # Higher = more responsive, less smooth
CORRECTION_EMA_ALPHA = 0.3  # Lower = smoother correction transitions

# --- Preview ---
PREVIEW_WINDOW_NAME = "Gaze Corrector Preview"
PREVIEW_SCALE = 1.0  # Scale factor for preview window

# --- Strength presets ---
STRENGTH_PRESETS = {
    "30%": 0.3,
    "50%": 0.5,
    "70%": 0.7,
    "85%": 0.85,
    "100%": 1.0,
}
