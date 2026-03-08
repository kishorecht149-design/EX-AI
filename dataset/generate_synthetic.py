"""
generate_synthetic.py — Synthetic Keypoint Data Generator
===========================================================
Creates realistic synthetic pose-keypoint data for all 8 exercise
classes so the full training pipeline can run without real videos.

The generator produces feature vectors that mimic the statistical
distribution of each exercise/form combination by:
    1. Defining a "base pose" per exercise (normalised landmark coords).
    2. Adjusting joint angles to simulate correct vs. incorrect form.
    3. Adding Gaussian noise for natural variation.

Usage:
    python dataset/generate_synthetic.py
    python dataset/generate_synthetic.py --samples 500
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extractor import get_feature_names

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed_keypoints")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "keypoints.csv")


# ======================================================================
# Synthetic base profiles
# ======================================================================

def _make_angle_features(knee_l, knee_r, hip_l, hip_r,
                         elbow_l, elbow_r, shoulder_l, shoulder_r):
    """Return the 8 angle features."""
    return np.array([knee_l, knee_r, hip_l, hip_r,
                     elbow_l, elbow_r, shoulder_l, shoulder_r])


def _make_coord_features(rng: np.random.Generator):
    """
    Generate 66 normalised coordinate features.
    Values are centred around 0 with small spread, mimicking hip-centred
    normalised coordinates.
    """
    return rng.normal(0.0, 0.15, size=66)


def _make_distance_features(rng: np.random.Generator):
    """
    Generate 6 relative distance ratio features.
    Typical values are around 0.5–1.2 of torso length.
    """
    base = np.array([0.55, 0.50, 0.85, 0.80, 1.0, 0.75])
    noise = rng.normal(0.0, 0.05, size=6)
    return base + noise


# ======================================================================
# Exercise profiles
# ======================================================================

EXERCISE_PROFILES = {
    # (exercise, form): dict of mean angles ± std
    "squat_correct": {
        "knee_mean": 90, "knee_std": 8,
        "hip_mean": 85, "hip_std": 8,
        "elbow_mean": 165, "elbow_std": 5,
        "shoulder_mean": 40, "shoulder_std": 5,
    },
    "squat_incorrect": {
        "knee_mean": 130, "knee_std": 12,   # knees not bending enough
        "hip_mean": 55, "hip_std": 10,      # leaning too far forward
        "elbow_mean": 160, "elbow_std": 8,
        "shoulder_mean": 45, "shoulder_std": 8,
    },
    "pushup_correct": {
        "knee_mean": 170, "knee_std": 5,
        "hip_mean": 170, "hip_std": 5,
        "elbow_mean": 90, "elbow_std": 10,
        "shoulder_mean": 70, "shoulder_std": 8,
    },
    "pushup_incorrect": {
        "knee_mean": 165, "knee_std": 8,
        "hip_mean": 140, "hip_std": 12,     # hips sagging or piked
        "elbow_mean": 135, "elbow_std": 15, # not going low enough
        "shoulder_mean": 55, "shoulder_std": 12,
    },
    "lunge_correct": {
        "knee_mean": 92, "knee_std": 8,
        "hip_mean": 160, "hip_std": 6,
        "elbow_mean": 165, "elbow_std": 5,
        "shoulder_mean": 35, "shoulder_std": 5,
    },
    "lunge_incorrect": {
        "knee_mean": 60, "knee_std": 10,    # over-bending
        "hip_mean": 120, "hip_std": 15,     # leaning forward
        "elbow_mean": 160, "elbow_std": 8,
        "shoulder_mean": 40, "shoulder_std": 8,
    },
    "plank_correct": {
        "knee_mean": 175, "knee_std": 3,
        "hip_mean": 172, "hip_std": 4,
        "elbow_mean": 88, "elbow_std": 6,   # forearm plank
        "shoulder_mean": 85, "shoulder_std": 5,
    },
    "plank_incorrect": {
        "knee_mean": 170, "knee_std": 6,
        "hip_mean": 140, "hip_std": 12,     # hips too high or low
        "elbow_mean": 95, "elbow_std": 10,
        "shoulder_mean": 80, "shoulder_std": 10,
    },
}


def generate_sample(label: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a single synthetic feature vector for the given class."""
    profile = EXERCISE_PROFILES[label]

    knee = rng.normal(profile["knee_mean"], profile["knee_std"])
    hip = rng.normal(profile["hip_mean"], profile["hip_std"])
    elbow = rng.normal(profile["elbow_mean"], profile["elbow_std"])
    shoulder = rng.normal(profile["shoulder_mean"], profile["shoulder_std"])

    # Slight asymmetry between left/right
    angles = _make_angle_features(
        knee + rng.normal(0, 2), knee + rng.normal(0, 2),
        hip + rng.normal(0, 2), hip + rng.normal(0, 2),
        elbow + rng.normal(0, 2), elbow + rng.normal(0, 2),
        shoulder + rng.normal(0, 2), shoulder + rng.normal(0, 2),
    )

    coords = _make_coord_features(rng)
    distances = _make_distance_features(rng)

    return np.concatenate([angles, coords, distances])


# ======================================================================
# Main generation
# ======================================================================

def generate_dataset(samples_per_class: int = 300, seed: int = 42):
    """
    Generate a synthetic dataset and save as CSV.

    Args:
        samples_per_class: Number of samples per exercise/form class.
        seed: Random seed for reproducibility.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(seed)
    feature_names = get_feature_names()

    rows = []
    labels = []

    for label in EXERCISE_PROFILES:
        for _ in range(samples_per_class):
            vec = generate_sample(label, rng)
            rows.append(vec)
            labels.append(label)

    data = np.array(rows)
    df = pd.DataFrame(data, columns=feature_names)
    df["label"] = labels

    df.to_csv(OUTPUT_CSV, index=False)

    total = len(df)
    print(f"[OK] Generated {total} synthetic samples ({samples_per_class}/class)")
    print(f"   Saved to: {OUTPUT_CSV}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Classes:  {df['label'].nunique()}")
    print()
    print(df["label"].value_counts().to_string())


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic exercise data")
    parser.add_argument("--samples", type=int, default=300,
                        help="Samples per class (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    generate_dataset(args.samples, args.seed)
