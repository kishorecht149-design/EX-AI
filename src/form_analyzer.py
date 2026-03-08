"""
form_analyzer.py — Rule-Based Form Error Detection
=====================================================
Provides exercise-specific posture checks and returns corrective
feedback messages.

Supported exercises:
    - Squat
    - Push-up
    - Lunge
    - Plank
"""

import numpy as np
from typing import List, Dict

from src.angle_calculator import (
    knee_angle, hip_angle, elbow_angle, shoulder_angle,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)


class FormAnalyzer:
    """Analyse exercise form and produce feedback messages."""

    def __init__(self):
        """Initialise thresholds for each exercise."""
        # Squat thresholds
        self.squat_knee_min = 70
        self.squat_knee_max = 110
        self.squat_hip_min = 60
        self.squat_back_angle_min = 150  # shoulder-hip-knee should be roughly straight

        # Push-up thresholds
        self.pushup_elbow_min = 70
        self.pushup_elbow_max = 160
        self.pushup_body_alignment_min = 160  # shoulder-hip-ankle nearly straight

        # Lunge thresholds
        self.lunge_front_knee_min = 75
        self.lunge_front_knee_max = 105
        self.lunge_back_knee_max = 120

        # Plank thresholds
        self.plank_body_alignment_min = 160
        self.plank_elbow_min = 70
        self.plank_elbow_max = 110

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, exercise: str, landmarks: np.ndarray) -> List[str]:
        """
        Analyse form for a given exercise.

        Args:
            exercise: One of 'squat', 'pushup', 'lunge', 'plank'.
            landmarks: (33, 4) landmark array.

        Returns:
            List of feedback strings. Empty list = good form.
        """
        exercise = exercise.lower().replace("-", "").replace(" ", "")
        if exercise in ("squat",):
            return self._check_squat(landmarks)
        elif exercise in ("pushup", "push_up"):
            return self._check_pushup(landmarks)
        elif exercise in ("lunge",):
            return self._check_lunge(landmarks)
        elif exercise in ("plank",):
            return self._check_plank(landmarks)
        return ["Unknown exercise"]

    def get_form_score(self, exercise: str, landmarks: np.ndarray) -> float:
        """
        Return a simple 0–100 form quality score.

        100 = perfect form, fewer points per feedback issue.
        """
        issues = self.analyze(exercise, landmarks)
        score = max(0, 100 - len(issues) * 25)
        return float(score)

    # ------------------------------------------------------------------
    # Exercise-specific checks
    # ------------------------------------------------------------------

    def _check_squat(self, lm: np.ndarray) -> List[str]:
        feedback: List[str] = []

        l_knee = knee_angle(lm, "left")
        r_knee = knee_angle(lm, "right")
        avg_knee = (l_knee + r_knee) / 2.0

        l_hip = hip_angle(lm, "left")
        r_hip = hip_angle(lm, "right")
        avg_hip = (l_hip + r_hip) / 2.0

        # Check depth
        if avg_knee > self.squat_knee_max:
            feedback.append("Go deeper — bend your knees more")
        elif avg_knee < self.squat_knee_min:
            feedback.append("Don't go too deep — ease up on knee bend")

        # Check back straightness (hip angle)
        if avg_hip < self.squat_hip_min:
            feedback.append("Keep your back straight — avoid leaning forward")

        # Check knee alignment (knees shouldn't cave inward)
        l_knee_x = lm[LEFT_KNEE][0]
        l_ankle_x = lm[LEFT_ANKLE][0]
        r_knee_x = lm[RIGHT_KNEE][0]
        r_ankle_x = lm[RIGHT_ANKLE][0]
        if l_knee_x < l_ankle_x - 0.03:
            feedback.append("Push your left knee outward — avoid caving in")
        if r_knee_x > r_ankle_x + 0.03:
            feedback.append("Push your right knee outward — avoid caving in")

        return feedback

    def _check_pushup(self, lm: np.ndarray) -> List[str]:
        feedback: List[str] = []

        l_elbow = elbow_angle(lm, "left")
        r_elbow = elbow_angle(lm, "right")
        avg_elbow = (l_elbow + r_elbow) / 2.0

        # Body alignment: shoulder-hip-ankle should be straight
        body_angle_l = self._three_point_angle(
            lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_ANKLE]
        )
        body_angle_r = self._three_point_angle(
            lm[RIGHT_SHOULDER], lm[RIGHT_HIP], lm[RIGHT_ANKLE]
        )
        avg_body = (body_angle_l + body_angle_r) / 2.0

        if avg_elbow > self.pushup_elbow_max:
            feedback.append("Bend your elbows more — lower your chest")
        elif avg_elbow < self.pushup_elbow_min:
            feedback.append("Extend your arms a bit — don't go too low")

        if avg_body < self.pushup_body_alignment_min:
            feedback.append("Keep your body straight — don't sag or pike your hips")

        return feedback

    def _check_lunge(self, lm: np.ndarray) -> List[str]:
        feedback: List[str] = []

        # We'll check both legs and pick the front leg as the one with smaller knee angle
        l_knee = knee_angle(lm, "left")
        r_knee = knee_angle(lm, "right")

        # Front leg has the deeper bend
        if l_knee < r_knee:
            front_knee, back_knee = l_knee, r_knee
            front_side, back_side = "left", "right"
        else:
            front_knee, back_knee = r_knee, l_knee
            front_side, back_side = "right", "left"

        if front_knee < self.lunge_front_knee_min:
            feedback.append(f"Don't over-bend your {front_side} knee")
        elif front_knee > self.lunge_front_knee_max:
            feedback.append(f"Bend your {front_side} knee to about 90°")

        # Check torso — should be roughly upright
        l_hip_a = hip_angle(lm, "left")
        r_hip_a = hip_angle(lm, "right")
        avg_hip = (l_hip_a + r_hip_a) / 2.0
        if avg_hip < 140:
            feedback.append("Keep your torso upright — don't lean forward")

        return feedback

    def _check_plank(self, lm: np.ndarray) -> List[str]:
        feedback: List[str] = []

        # Body alignment: shoulder-hip-ankle
        body_angle_l = self._three_point_angle(
            lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_ANKLE]
        )
        body_angle_r = self._three_point_angle(
            lm[RIGHT_SHOULDER], lm[RIGHT_HIP], lm[RIGHT_ANKLE]
        )
        avg_body = (body_angle_l + body_angle_r) / 2.0

        if avg_body < self.plank_body_alignment_min:
            # Determine if hips are too high or too low
            mid_hip_y = (lm[LEFT_HIP][1] + lm[RIGHT_HIP][1]) / 2.0
            mid_shoulder_y = (lm[LEFT_SHOULDER][1] + lm[RIGHT_SHOULDER][1]) / 2.0
            if mid_hip_y < mid_shoulder_y:
                feedback.append("Lower your hips — your butt is too high")
            else:
                feedback.append("Raise your hips — don't let them sag")

        # Elbow angle (for forearm plank)
        l_elbow = elbow_angle(lm, "left")
        r_elbow = elbow_angle(lm, "right")
        avg_elbow = (l_elbow + r_elbow) / 2.0
        if avg_elbow > self.plank_elbow_max + 30:
            # Arms too straight → might be a high plank which is fine
            pass  # high plank is acceptable
        elif avg_elbow < self.plank_elbow_min - 10:
            feedback.append("Adjust your elbow position — keep them under your shoulders")

        return feedback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _three_point_angle(a, b, c) -> float:
        """Calculate angle at b from points a, b, c (uses only x, y)."""
        from src.angle_calculator import calculate_angle
        return calculate_angle(a, b, c)
