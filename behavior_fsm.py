"""Natural behavior state machine with 4 states and hysteresis."""

import time
from enum import Enum, auto

import config
from gaze_estimator import GazeInfo


class State(Enum):
    ENGAGED = auto()
    DISENGAGING = auto()
    DISENGAGED = auto()
    RE_ENGAGING = auto()


class BehaviorFSM:
    """Determines whether gaze correction should be active based on user behavior.

    Uses hysteresis thresholds and timed transitions to avoid flickering.
    """

    def __init__(self):
        self.state = State.ENGAGED
        self._transition_start: float = 0.0

    def _should_force_disengage(self, gaze: GazeInfo) -> bool:
        """Head pose override: always disengage for large head turns."""
        return (abs(gaze.yaw) > config.HEAD_YAW_THRESHOLD or
                abs(gaze.pitch) > config.HEAD_PITCH_THRESHOLD)

    def _gaze_is_away(self, gaze: GazeInfo) -> bool:
        return gaze.gaze_angle > config.DISENGAGE_THRESHOLD

    def _gaze_is_near(self, gaze: GazeInfo) -> bool:
        return gaze.gaze_angle < config.ENGAGE_THRESHOLD

    def update(self, gaze: GazeInfo) -> float:
        """Update FSM and return correction blend factor (0.0 = off, 1.0 = full).

        Args:
            gaze: Current frame's gaze estimation.

        Returns:
            Blend factor in [0.0, 1.0].
        """
        now = time.monotonic()
        force_away = self._should_force_disengage(gaze)

        if self.state == State.ENGAGED:
            if force_away or self._gaze_is_away(gaze):
                self.state = State.DISENGAGING
                self._transition_start = now
                return 1.0
            return 1.0

        elif self.state == State.DISENGAGING:
            elapsed = now - self._transition_start
            if not force_away and self._gaze_is_near(gaze):
                # Returned quickly, go back to engaged
                self.state = State.ENGAGED
                return 1.0
            if elapsed >= config.DISENGAGE_DURATION:
                self.state = State.DISENGAGED
                return 0.0
            # Linear fade out
            return max(0.0, 1.0 - elapsed / config.DISENGAGE_DURATION)

        elif self.state == State.DISENGAGED:
            if not force_away and self._gaze_is_near(gaze):
                self.state = State.RE_ENGAGING
                self._transition_start = now
                return 0.0
            return 0.0

        elif self.state == State.RE_ENGAGING:
            elapsed = now - self._transition_start
            if force_away or self._gaze_is_away(gaze):
                self.state = State.DISENGAGED
                return 0.0
            if elapsed >= config.RE_ENGAGE_DURATION:
                self.state = State.ENGAGED
                return 1.0
            # Linear fade in
            return min(1.0, elapsed / config.RE_ENGAGE_DURATION)

        return 0.0  # Fallback
