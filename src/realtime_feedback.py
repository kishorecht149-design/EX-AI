"""
realtime_feedback.py — Real-Time Exercise Feedback Pipeline
=============================================================
Reads a webcam or video file, runs pose detection, classifies the
exercise, checks form, counts reps, and overlays everything on the video.

Usage:
    python -m src.realtime_feedback            # webcam
    python -m src.realtime_feedback video.mp4   # video file
"""

import sys
import cv2
import numpy as np
from typing import Optional

from src.pose_detector import PoseDetector
from src.feature_extractor import extract_features
from src.exercise_classifier import ExerciseClassifier
from src.form_analyzer import FormAnalyzer
from src.rep_counter import RepCounter


# ======================================================================
# Drawing helpers
# ======================================================================

def _draw_text(frame, text: str, pos: tuple, scale=0.7, color=(255, 255, 255),
               thickness=2, bg_color=(0, 0, 0)):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - h - 6), (x + w + 4, y + 6), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_overlay(frame, exercise: str, form: str, feedback: list,
                  rep_status: str, confidence: float):
    """Draw the full HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent panel on the left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 40 + 30 * (4 + len(feedback))), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 30
    _draw_text(frame, f"Exercise: {exercise.upper()}", (10, y),
               scale=0.8, color=(0, 255, 200))
    y += 35
    _draw_text(frame, f"Form: {form}", (10, y),
               scale=0.7, color=(0, 255, 0) if form == "correct" else (0, 100, 255))
    y += 30
    _draw_text(frame, f"Confidence: {confidence:.0%}", (10, y),
               scale=0.6, color=(200, 200, 200))
    y += 30
    _draw_text(frame, rep_status, (10, y), scale=0.7, color=(255, 255, 0))
    y += 35

    # Feedback messages
    for msg in feedback:
        _draw_text(frame, f"! {msg}", (10, y), scale=0.55, color=(0, 140, 255))
        y += 28


# ======================================================================
# Main pipeline
# ======================================================================

def run_feedback(source=0, model_path: Optional[str] = None):
    """
    Run the real-time feedback pipeline.

    Args:
        source: 0 for webcam, or path to a video file.
        model_path: Optional custom path for the trained model.
    """
    # Initialise components
    detector = PoseDetector(static_image_mode=False)
    try:
        classifier = ExerciseClassifier(model_path)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Running without classifier — form analysis only.")
        classifier = None

    form_analyzer = FormAnalyzer()
    rep_counter = RepCounter()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{source}'")
        return

    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect pose
        results = detector.detect(frame)
        landmarks = None
        if results.pose_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility]
                 for lm in results.pose_landmarks[0]]
            )

        # Default values
        exercise_name = "detecting..."
        form_label = "—"
        confidence = 0.0
        feedback_msgs = []
        rep_status = "Reps: 0"

        if landmarks is not None:
            # Extract features & classify
            features = extract_features(landmarks)

            if classifier is not None:
                label, confidence = classifier.predict_proba(features)
                exercise_name, form_label = ExerciseClassifier.parse_label(label)
            else:
                exercise_name = "unknown"
                form_label = "—"

            # Form analysis (rule-based, works independently of the classifier)
            if exercise_name not in ("unknown", "detecting..."):
                feedback_msgs = form_analyzer.analyze(exercise_name, landmarks)
                rep_counter.update(landmarks, exercise_name)
                rep_status = rep_counter.get_status()
            else:
                rep_status = "Reps: —"

        # Draw skeleton
        annotated = detector.draw_landmarks(frame, results)

        # Draw HUD overlay
        _draw_overlay(annotated, exercise_name, form_label,
                      feedback_msgs, rep_status, confidence)

        cv2.imshow("AI Exercise Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Session ended.")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    video_source = 0  # default: webcam
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        # Try to interpret as integer (camera index)
        try:
            video_source = int(video_source)
        except ValueError:
            pass  # keep as string (file path)

    run_feedback(source=video_source)
