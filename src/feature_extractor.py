"""
feature_extractor.py — Keypoint-to-Feature-Vector Conversion
==============================================================
Converts raw MediaPipe landmarks (33 × 4) into a flat feature vector
suitable for machine-learning classification.

Feature groups:
    1. **Joint angles** (8 angles, both sides)
    2. **Normalised coordinates** (33 × 2 = 66 values, hip-centred)
    3. **Relative distances** (selected limb-length ratios, 6 values)

Total feature dimension: 8 + 66 + 6 = **80**.
"""

import numpy as np
from typing import Optional, List

from src.angle_calculator import (
    all_angles,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)


# ======================================================================
# Utility helpers
# ======================================================================

def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def _normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Centre landmarks around the mid-hip point and scale by torso length.

    This makes features invariant to the person's position in the frame
    and (approximately) their distance from the camera.

    Args:
        landmarks: (33, 4) array.

    Returns:
        (33, 2) normalised (x, y) array.
    """
    xy = landmarks[:, :2].copy()

    # Centre: midpoint of left and right hip
    centre = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2.0
    xy -= centre

    # Scale: distance between left shoulder and left hip (torso length)
    torso = _distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP])
    if torso < 1e-6:
        torso = 1.0  # fallback to avoid division by zero
    xy /= torso

    return xy


# ======================================================================
# Main extractor
# ======================================================================

def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert a (33, 4) landmark array into an 80-dimensional feature vector.

    Feature layout:
        [0:8]   — joint angles (left/right knee, hip, elbow, shoulder)
        [8:74]  — normalised (x, y) for all 33 landmarks (66 values)
        [74:80] — relative distance ratios (6 values)

    Args:
        landmarks: (33, 4) from PoseDetector.get_landmarks().

    Returns:
        1-D numpy array of shape (80,).
    """
    # --- 1. Joint angles (8) ---
    angles = all_angles(landmarks)
    angle_features = np.array([
        angles["left_knee"],
        angles["right_knee"],
        angles["left_hip"],
        angles["right_hip"],
        angles["left_elbow"],
        angles["right_elbow"],
        angles["left_shoulder"],
        angles["right_shoulder"],
    ])

    # --- 2. Normalised coordinates (66) ---
    norm_xy = _normalise_landmarks(landmarks)
    coord_features = norm_xy.flatten()  # shape (66,)

    # --- 3. Relative distances (6) ---
    # Ratios of limb lengths relative to torso length
    torso = _distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP])
    if torso < 1e-6:
        torso = 1.0

    distances = np.array([
        _distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW]) / torso,
        _distance(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]) / torso,
        _distance(landmarks[LEFT_HIP], landmarks[LEFT_KNEE]) / torso,
        _distance(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]) / torso,
        _distance(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP]) / torso,
        _distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]) / torso,
    ])

    return np.concatenate([angle_features, coord_features, distances])



# MediaPipe landmark names (inlined to avoid importing PoseDetector/MediaPipe
# at generation time — keeps the synthetic generator dependency-free).
_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def get_feature_names() -> List[str]:
    """
    Return human-readable names for each of the 80 features.

    Useful for inspecting feature importance in tree-based models.
    """
    names: List[str] = []

    # Angle names
    for side in ("left", "right"):
        for joint in ("knee", "hip", "elbow", "shoulder"):
            names.append(f"angle_{side}_{joint}")

    # Coordinate names
    for lm_name in _LANDMARK_NAMES:
        names.append(f"norm_x_{lm_name}")
        names.append(f"norm_y_{lm_name}")

    # Distance ratio names
    names += [
        "dist_l_upper_arm", "dist_l_forearm",
        "dist_l_thigh", "dist_l_shin",
        "dist_r_torso", "dist_shoulder_width",
    ]
    return names
