"""
rep_counter.py — Repetition Counting via State Machine
========================================================
Tracks movement phases (e.g. standing → down → standing) and counts
repetitions for each exercise type.

Supported exercises: squat, push-up, lunge, plank (hold timer).
"""

import time
import numpy as np
from typing import Dict, Tuple

from src.angle_calculator import knee_angle, elbow_angle, hip_angle


class RepCounter:
    """
    Count exercise repetitions using a simple state machine.

    States for dynamic exercises (squat, push-up, lunge):
        'up'   — starting / top position
        'down' — bottom position

    Transition from 'up' → 'down' → 'up' increments the counter by 1.

    For plank (a static hold), we track hold duration instead.
    """

    def __init__(self):
        self.count: int = 0
        self.state: str = "up"      # 'up' or 'down'
        self.exercise: str = ""
        self._state_frames: int = 0
        self._last_rep_at: float = 0.0

        # Thresholds (angle in degrees) — STRICT
        # Squat: must reach <90° at bottom, must stand to >165° at top
        # Pushup: must reach <85° at bottom (chest near floor), >160° at top
        # Lunge: front knee must reach <90° at bottom, >160° at top
        self._thresholds: Dict[str, Dict[str, float]] = {
            "squat": {"down": 90,  "up": 165},
            "pushup": {"down": 85,  "up": 160},
            "lunge": {"down": 90,  "up": 160},
        }

        # Require 4 consecutive frames in the new state before committing —
        # eliminates jitter and brief angle spikes from counting as real reps.
        self._min_state_frames: Dict[str, int] = {
            "squat": 4,
            "pushup": 4,
            "lunge": 4,
        }
        # Minimum real-time gap between reps to guard against double-counting
        self._min_rep_interval: Dict[str, float] = {
            "squat": 1.0,
            "pushup": 0.8,
            "lunge": 1.0,
        }

        # Plank hold tracking
        self._plank_start: float = 0.0
        self._plank_holding: bool = False
        self.plank_duration: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, exercise: str = ""):
        """Reset counter to zero and optionally set a new exercise."""
        self.count = 0
        self.state = "up"
        self.exercise = exercise.lower().replace("-", "").replace(" ", "")
        self._state_frames = 0
        self._last_rep_at = 0.0
        self._plank_start = 0.0
        self._plank_holding = False
        self.plank_duration = 0.0

    def update(self, landmarks: np.ndarray, exercise: str = "") -> Tuple[int, str]:
        """
        Update the rep counter with new landmarks.

        Args:
            landmarks: (33, 4) array from PoseDetector.
            exercise: Exercise name (used if counter not yet assigned).

        Returns:
            (rep_count, current_state) tuple.
        """
        if exercise:
            ex = exercise.lower().replace("-", "").replace(" ", "")
            if ex != self.exercise:
                self.reset(ex)

        if self.exercise == "plank":
            return self._update_plank(landmarks)

        return self._update_dynamic(landmarks)

    def get_status(self) -> str:
        """Return a human-readable status string."""
        if self.exercise == "plank":
            return f"Hold: {self.plank_duration:.1f}s"
        return f"Reps: {self.count} | Phase: {self.state}"

    # ------------------------------------------------------------------
    # Dynamic exercises
    # ------------------------------------------------------------------

    def _update_dynamic(self, lm: np.ndarray) -> Tuple[int, str]:
        """State-machine update for squat / push-up / lunge."""
        angle = self._get_key_angle(lm)
        thresholds = self._thresholds.get(self.exercise, {"down": 100, "up": 160})
        min_frames = self._min_state_frames.get(self.exercise, 2)
        min_interval = self._min_rep_interval.get(self.exercise, 0.65)
        now = time.time()

        if self.state == "up" and angle < thresholds["down"]:
            self._state_frames += 1
            if self._state_frames >= min_frames:
                self.state = "down"
                self._state_frames = 0
        elif self.state == "up":
            self._state_frames = 0
        elif self.state == "down" and angle > thresholds["up"]:
            self._state_frames += 1
            if self._state_frames >= min_frames and (now - self._last_rep_at) >= min_interval:
                self.state = "up"
                self.count += 1
                self._last_rep_at = now
                self._state_frames = 0
        elif self.state == "down":
            self._state_frames = 0

        return self.count, self.state

    def _get_key_angle(self, lm: np.ndarray) -> float:
        """Return the primary angle used for rep counting."""
        if self.exercise == "pushup":
            # Use elbow angle
            return (elbow_angle(lm, "left") + elbow_angle(lm, "right")) / 2.0
        if self.exercise == "lunge":
            # Use the bent "front" knee rather than averaging both legs.
            return min(knee_angle(lm, "left"), knee_angle(lm, "right"))
        else:
            # Squat: use the average knee angle.
            return (knee_angle(lm, "left") + knee_angle(lm, "right")) / 2.0

    # ------------------------------------------------------------------
    # Plank (static hold)
    # ------------------------------------------------------------------

    def _update_plank(self, lm: np.ndarray) -> Tuple[int, str]:
        """
        Track plank hold duration.

        A hold is considered active when the average hip angle
        (shoulder-hip-ankle alignment) is > 150°.
        """
        l_hip = hip_angle(lm, "left")
        r_hip = hip_angle(lm, "right")
        avg = (l_hip + r_hip) / 2.0

        # Strict plank: body must be flat (155°–180°). Butt-in-air or hip-sag
        # does NOT count as a valid plank hold.
        is_holding = 155 < avg <= 180

        if is_holding and not self._plank_holding:
            # Start hold
            self._plank_holding = True
            self._plank_start = time.time()
        elif is_holding and self._plank_holding:
            self.plank_duration = time.time() - self._plank_start
        elif not is_holding and self._plank_holding:
            # End hold
            self._plank_holding = False
            self.count += 1  # count completed holds

        return self.count, "holding" if self._plank_holding else "resting"
