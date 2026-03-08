"""
prepare_dataset.py — Video-to-Keypoint Dataset Pipeline
=========================================================
Reads exercise videos organised in category folders, runs MediaPipe
pose detection on each frame, extracts 33 body keypoints, computes
joint angles and features, and saves everything as a CSV suitable
for model training.

Expected folder layout under ``dataset/raw_videos/``:
    squat_correct/
    squat_incorrect/
    pushup_correct/
    pushup_incorrect/
    lunge_correct/
    lunge_incorrect/
    plank_correct/
    plank_incorrect/

Usage:
    python dataset/prepare_dataset.py
"""

import os
import sys
import csv
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure the project root is on the path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.pose_detector import PoseDetector
from src.feature_extractor import extract_features, get_feature_names


# ======================================================================
# Configuration
# ======================================================================
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw_videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed_keypoints")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "keypoints.csv")

# Supported exercise categories (folder names)
CATEGORIES = [
    "squat_correct", "squat_incorrect",
    "pushup_correct", "pushup_incorrect",
    "lunge_correct", "lunge_incorrect",
    "plank_correct", "plank_incorrect",
]

# Video extensions to look for
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

# Sample every N-th frame to reduce redundancy
FRAME_SKIP = 5


# ======================================================================
# Pipeline
# ======================================================================

def collect_videos():
    """Discover all video files grouped by category."""
    videos = {}
    for cat in CATEGORIES:
        cat_dir = os.path.join(RAW_VIDEO_DIR, cat)
        if not os.path.isdir(cat_dir):
            print(f"[SKIP] Category folder not found: {cat_dir}")
            continue
        files = []
        for ext in VIDEO_EXTENSIONS:
            files.extend(glob.glob(os.path.join(cat_dir, ext)))
        videos[cat] = sorted(files)
        print(f"  {cat}: {len(files)} video(s)")
    return videos


def process_videos():
    """Main processing loop: videos → features → CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Scanning for videos...")
    video_map = collect_videos()

    total_videos = sum(len(v) for v in video_map.values())
    if total_videos == 0:
        print("\n[!] No videos found. Please place exercise videos in:")
        print(f"    {RAW_VIDEO_DIR}/<category>/")
        print("    Categories:", ", ".join(CATEGORIES))
        print("\nAlternatively, run generate_synthetic.py to create synthetic data.")
        return

    feature_names = get_feature_names()
    header = feature_names + ["label"]

    all_rows = []
    detector = PoseDetector(static_image_mode=True, model_complexity=1)

    for category, files in video_map.items():
        for video_path in tqdm(files, desc=category):
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for efficiency
                if frame_idx % FRAME_SKIP != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1

                landmarks = detector.get_landmarks(frame)
                if landmarks is None:
                    continue

                features = extract_features(landmarks)
                row = list(features) + [category]
                all_rows.append(row)

            cap.release()

    detector.close()

    # Save CSV
    df = pd.DataFrame(all_rows, columns=header)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df)} samples to {OUTPUT_CSV}")
    print(f"   Features per sample: {len(feature_names)}")
    print(f"   Classes: {df['label'].nunique()}")
    print(df["label"].value_counts().to_string())


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    process_videos()
