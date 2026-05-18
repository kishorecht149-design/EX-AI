"""
train_model.py — Train a model from real or synthetic workout features
======================================================================
Builds a reproducible training pipeline around the extracted keypoint
dataset and prefers grouping by source video when real workout metadata
is available.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "processed_keypoints" / "keypoints.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "exercise_model.pkl"
DEFAULT_REPORT = PROJECT_ROOT / "models" / "training_report.json"
DEFAULT_CONFUSION = PROJECT_ROOT / "models" / "confusion_matrix.csv"

METADATA_COLUMNS = {
    "label",
    "exercise",
    "form",
    "source_video",
    "video_id",
    "frame_idx",
}


def split_dataset(df: pd.DataFrame, test_size: float, random_state: int):
    """Split by source video when possible, else fall back to stratified rows."""
    if "video_id" in df.columns and df["video_id"].nunique() > df["label"].nunique():
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(df, y=df["label"], groups=df["video_id"]))
        split_kind = "grouped_by_video"
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(df, df["label"]))
        split_kind = "stratified_rows"

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), split_kind


def train_model(
    dataset_path: Path,
    model_path: Path,
    report_path: Path,
    confusion_path: Path,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> None:
    """Train and save the classifier plus evaluation artifacts."""
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    train_df, test_df, split_kind = split_dataset(df, test_size, random_state)

    feature_columns = [col for col in df.columns if col not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns found in the dataset.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=list(model.classes_))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    payload = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "split_kind": split_kind,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_videos": int(train_df["video_id"].nunique()) if "video_id" in train_df else None,
        "test_videos": int(test_df["video_id"].nunique()) if "video_id" in test_df else None,
        "classes": list(model.classes_),
        "metrics": report,
    }
    report_path.write_text(json.dumps(payload, indent=2))

    pd.DataFrame(matrix, index=model.classes_, columns=model.classes_).to_csv(confusion_path)

    print("=" * 72)
    print("Real-data training complete")
    print("=" * 72)
    print(f"Dataset        : {dataset_path}")
    print(f"Split strategy : {split_kind}")
    print(f"Train rows     : {len(train_df)}")
    print(f"Test rows      : {len(test_df)}")
    if "video_id" in train_df.columns:
        print(f"Train videos   : {train_df['video_id'].nunique()}")
        print(f"Test videos    : {test_df['video_id'].nunique()}")
    print(f"Model saved    : {model_path}")
    print(f"Report saved   : {report_path}")
    print(f"Confusion CSV  : {confusion_path}")
    print()
    print(report_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the exercise classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--confusion-out", type=Path, default=DEFAULT_CONFUSION)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        model_path=args.model_out,
        report_path=args.report_out,
        confusion_path=args.confusion_out,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
