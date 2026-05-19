"""
runtime_assets.py — Ensure deploy-time model assets exist
=========================================================
Bootstraps required model files for hosted environments where large binary
artifacts are intentionally not committed to the repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_CSV = PROJECT_ROOT / "dataset" / "processed_keypoints" / "keypoints.csv"
MODEL_PATH = MODELS_DIR / "exercise_model.pkl"
POSE_TASK_PATH = MODELS_DIR / "pose_landmarker_lite.task"
TRAIN_SCRIPT = PROJECT_ROOT / "train_model.py"
SYNTHETIC_SCRIPT = PROJECT_ROOT / "dataset" / "generate_synthetic.py"
POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

_bootstrap_lock = threading.Lock()
_bootstrap_complete = False


def _run_command(args: list[str]) -> None:
    """Run a subprocess and raise a readable error if it fails."""
    completed = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def ensure_runtime_assets() -> None:
    """
    Ensure required runtime artifacts exist.

    On hosted platforms like Render, the repository may not include the trained
    model binary. In that case we generate a fallback synthetic dataset and
    train a model during the first boot.
    """
    global _bootstrap_complete

    if _bootstrap_complete:
        return

    with _bootstrap_lock:
        if _bootstrap_complete:
            return

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not POSE_TASK_PATH.exists():
            print(f"[BOOTSTRAP] Downloading pose task model to {POSE_TASK_PATH}")
            urllib.request.urlretrieve(POSE_TASK_URL, POSE_TASK_PATH)

        if not MODEL_PATH.exists():
            if not DATASET_CSV.exists():
                print("[BOOTSTRAP] No dataset found. Generating synthetic training data...")
                _run_command([sys.executable, str(SYNTHETIC_SCRIPT)])

            print("[BOOTSTRAP] Training fallback exercise classifier...")
            _run_command(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--dataset",
                    str(DATASET_CSV),
                    "--model-out",
                    str(MODEL_PATH),
                    "--report-out",
                    str(MODELS_DIR / "training_report.json"),
                    "--confusion-out",
                    str(MODELS_DIR / "confusion_matrix.csv"),
                    "--n-estimators",
                    os.environ.get("AI_EX_BOOTSTRAP_ESTIMATORS", "120"),
                ]
            )

        _bootstrap_complete = True
