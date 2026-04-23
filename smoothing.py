"""EMA filters for landmark jitter and correction fade smoothing."""

import numpy as np


class EMAFilter:
    """Exponential Moving Average filter for numpy arrays or scalars."""

    def __init__(self, alpha: float):
        """
        Args:
            alpha: Smoothing factor in (0, 1]. Higher = more responsive.
        """
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        """Apply EMA and return smoothed value."""
        if self._value is None:
            self._value = np.array(new_value, dtype=np.float64) if isinstance(new_value, np.ndarray) else new_value
        else:
            self._value = self.alpha * new_value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self):
        return self._value


class LandmarkSmoother:
    """EMA smoother for MediaPipe landmark arrays (478, 3)."""

    def __init__(self, alpha: float):
        self._filter = EMAFilter(alpha)

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        return self._filter.update(landmarks)

    def reset(self):
        self._filter.reset()


class CorrectionSmoother:
    """EMA smoother for the correction blend factor (scalar 0-1)."""

    def __init__(self, alpha: float):
        self._filter = EMAFilter(alpha)

    def smooth(self, blend_factor: float) -> float:
        return float(self._filter.update(blend_factor))

    def reset(self):
        self._filter.reset()
