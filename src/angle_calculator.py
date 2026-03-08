"""
angle_calculator.py — Joint Angle Computation
===============================================
Provides a generic ``calculate_angle(a, b, c)`` function and convenience
helpers for common joints (knee, hip, elbow, shoulder).

All angle functions accept points as (x, y) or (x, y, z) sequences.
Angles are returned in **degrees** (0–180).
"""

import numpy as np
from typing import Sequence, Union

# Type alias for a 2D or 3D point
Point = Union[Sequence[float], np.ndarray]


# ======================================================================
# Core function
# ======================================================================

def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """
    Calculate the angle ∠ABC at vertex **b** formed by rays BA and BC.

    Works for 2D (x, y) and 3D (x, y, z) points. If 3D points are given
    only the first two components are used (projection onto the image plane)
    to keep things consistent with 2D video analysis.

    Args:
        a: First point  (e.g. hip).
        b: Vertex point  (e.g. knee).
        c: Third point   (e.g. ankle).

    Returns:
        Angle in degrees [0, 180].
    """
    a = np.array(a[:2], dtype=np.float64)
    b = np.array(b[:2], dtype=np.float64)
    c = np.array(c[:2], dtype=np.float64)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


# ======================================================================
# MediaPipe landmark indices (for convenience)
# ======================================================================
# Shoulders
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
# Elbows
LEFT_ELBOW  = 13
RIGHT_ELBOW = 14
# Wrists
LEFT_WRIST  = 15
RIGHT_WRIST = 16
# Hips
LEFT_HIP  = 23
RIGHT_HIP = 24
# Knees
LEFT_KNEE  = 25
RIGHT_KNEE = 26
# Ankles
LEFT_ANKLE  = 27
RIGHT_ANKLE = 28


# ======================================================================
# Convenience functions — accept (33, 4) landmark array
# ======================================================================

def knee_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """
    Calculate the knee angle (hip → knee → ankle).

    Args:
        landmarks: (33, 4) landmark array from PoseDetector.
        side: 'left' or 'right'.

    Returns:
        Knee angle in degrees.
    """
    if side == "left":
        return calculate_angle(
            landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]
        )
    return calculate_angle(
        landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]
    )


def hip_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """
    Calculate the hip angle (shoulder → hip → knee).

    Args:
        landmarks: (33, 4) landmark array.
        side: 'left' or 'right'.

    Returns:
        Hip angle in degrees.
    """
    if side == "left":
        return calculate_angle(
            landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_KNEE]
        )
    return calculate_angle(
        landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE]
    )


def elbow_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """
    Calculate the elbow angle (shoulder → elbow → wrist).

    Args:
        landmarks: (33, 4) landmark array.
        side: 'left' or 'right'.

    Returns:
        Elbow angle in degrees.
    """
    if side == "left":
        return calculate_angle(
            landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]
        )
    return calculate_angle(
        landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]
    )


def shoulder_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """
    Calculate the shoulder angle (elbow → shoulder → hip).

    Args:
        landmarks: (33, 4) landmark array.
        side: 'left' or 'right'.

    Returns:
        Shoulder angle in degrees.
    """
    if side == "left":
        return calculate_angle(
            landmarks[LEFT_ELBOW], landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP]
        )
    return calculate_angle(
        landmarks[RIGHT_ELBOW], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP]
    )


def all_angles(landmarks: np.ndarray) -> dict:
    """
    Compute all key joint angles for both sides of the body.

    Args:
        landmarks: (33, 4) landmark array.

    Returns:
        Dictionary with angle names as keys and degree values.
    """
    return {
        "left_knee":      knee_angle(landmarks, "left"),
        "right_knee":     knee_angle(landmarks, "right"),
        "left_hip":       hip_angle(landmarks, "left"),
        "right_hip":      hip_angle(landmarks, "right"),
        "left_elbow":     elbow_angle(landmarks, "left"),
        "right_elbow":    elbow_angle(landmarks, "right"),
        "left_shoulder":  shoulder_angle(landmarks, "left"),
        "right_shoulder": shoulder_angle(landmarks, "right"),
    }
