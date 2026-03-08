"""
run_webcam_demo.py — Webcam / Video Demo Entry Point
======================================================
Launches the real-time exercise detection and form correction overlay.

Usage:
    python demo/run_webcam_demo.py            # webcam (default)
    python demo/run_webcam_demo.py video.mp4  # video file
    python demo/run_webcam_demo.py 1          # camera index 1
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.realtime_feedback import run_feedback


def main():
    """Parse CLI args and launch the feedback pipeline."""
    # Default source: webcam 0
    source = 0

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            source = int(arg)  # camera index
        except ValueError:
            source = arg       # file path

    # Optional model path override
    model_path = None
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    print("=" * 50)
    print("  AI Exercise Detection — Real-Time Demo")
    print("=" * 50)
    print(f"  Source     : {source}")
    print(f"  Model      : {model_path or 'default (models/exercise_model.pkl)'}")
    print(f"  Controls   : Press 'q' to quit")
    print("=" * 50)

    run_feedback(source=source, model_path=model_path)


if __name__ == "__main__":
    main()
