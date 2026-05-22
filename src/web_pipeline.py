"""
web_pipeline.py — Shared analysis helpers for web uploads and live monitoring
=============================================================================
Builds a reusable layer around pose detection, classification, form analysis,
and rep counting so image, video, and live camera inputs can be handled from
the web UI.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import Counter
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

import cv2
import numpy as np

from src.exercise_classifier import ExerciseClassifier
from src.feature_extractor import extract_features
from src.form_analyzer import FormAnalyzer
from src.pose_detector import PoseDetector
from src.rep_counter import RepCounter


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""

    exercise: str
    form: str
    confidence: float
    feedback: list[str]
    annotated_frame: np.ndarray
    rep_status: str = "Reps: —"


@dataclass
class VideoAnalysis:
    """Aggregate analysis result for a video."""

    detected_frames: int
    sampled_frames: int
    exercise: str
    form: str
    confidence: float
    common_feedback: list[str]
    preview_frame: Optional[np.ndarray]


@dataclass
class LiveMetrics:
    """Current live-monitoring state."""

    exercise: str = "detecting..."
    form: str = "unknown"
    confidence: float = 0.0
    feedback: list[str] | None = None
    rep_status: str = "Reps: 0"
    pose_detected: bool = False
    session_active: bool = False
    selected_exercise: str = "auto"
    form_score: float = 0.0
    proper_reps: int = 0
    needs_work_reps: int = 0
    session_seconds: float = 0.0
    status_badge: str = "Waiting"

    def __post_init__(self) -> None:
        if self.feedback is None:
            self.feedback = []


def _draw_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with a background block for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - h - 6), (x + w + 4, y + 6), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_live_overlay(frame: np.ndarray, metrics: LiveMetrics) -> np.ndarray:
    """Render exercise, form, reps, and feedback on top of a frame."""
    overlay = frame.copy()
    panel_height = 190 + 28 * len(metrics.feedback)
    cv2.rectangle(overlay, (0, 0), (460, panel_height), (62, 42, 99), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    y = 28
    _draw_text(frame, f"Exercise: {metrics.exercise.upper()}", (12, y), 0.75, (238, 225, 255))
    y += 34
    session_text = "Session: ACTIVE" if metrics.session_active else "Session: READY"
    session_color = (154, 255, 191) if metrics.session_active else (220, 220, 220)
    _draw_text(frame, session_text, (12, y), 0.6, session_color)
    y += 28
    form_color = (110, 255, 170) if metrics.form == "correct" else (110, 180, 255)
    _draw_text(frame, f"Form: {metrics.form}", (12, y), 0.68, form_color)
    y += 30
    _draw_text(frame, f"Confidence: {metrics.confidence:.0%}", (12, y), 0.62, (244, 239, 255))
    y += 28
    _draw_text(frame, metrics.rep_status, (12, y), 0.68, (255, 231, 155))
    y += 34
    _draw_text(frame, f"Form score: {metrics.form_score:.0f}/100", (12, y), 0.55, (222, 210, 255))
    y += 26
    _draw_text(frame, f"Good reps: {metrics.proper_reps} | Needs work: {metrics.needs_work_reps}", (12, y), 0.52, (244, 239, 255))
    y += 30

    if metrics.feedback:
        for msg in metrics.feedback[:4]:
            _draw_text(frame, f"! {msg}", (12, y), 0.54, (255, 205, 180))
            y += 26
    elif metrics.pose_detected:
        _draw_text(frame, "No major form issues detected", (12, y), 0.54, (190, 245, 190))
    else:
        _draw_text(frame, "Move into frame so the full body is visible", (12, y), 0.54, (255, 205, 180))

    return frame


class ExerciseAnalyzer:
    """Reusable analyzer for uploaded images and videos."""

    def __init__(self, model_path: Optional[str] = None):
        self.detector_error: Optional[str] = None
        self.detector = None
        self.classifier = ExerciseClassifier(model_path)
        self.form_analyzer = FormAnalyzer()
        try:
            self.detector = PoseDetector(static_image_mode=False)
        except Exception as exc:
            self.detector_error = str(exc)

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    def ensure_ready(self) -> None:
        """Raise a clear error when pose detection is unavailable."""
        if self.detector is None:
            raise RuntimeError(
                "Pose detector is unavailable in this runtime. "
                f"Underlying error: {self.detector_error}"
            )

    def analyze_frame(self, frame: np.ndarray) -> Optional[FrameAnalysis]:
        """Analyze a single BGR frame."""
        self.ensure_ready()
        results = self.detector.detect(frame)
        landmarks = self.detector.landmarks_from_results(results)
        annotated = self.detector.draw_landmarks(frame, results)

        if landmarks is None:
            return None

        features = extract_features(landmarks)
        label, confidence = self.classifier.predict_proba(features)
        exercise, form = ExerciseClassifier.parse_label(label)
        feedback = self.form_analyzer.analyze(exercise, landmarks)

        return FrameAnalysis(
            exercise=exercise,
            form=form,
            confidence=confidence,
            feedback=feedback,
            annotated_frame=annotated,
            rep_status="Reps: —",
        )

    def analyze_image_bytes(self, image_bytes: bytes) -> Optional[FrameAnalysis]:
        """Analyze an uploaded image."""
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return self.analyze_frame(frame)

    def analyze_video_file(
        self, video_bytes: bytes, sample_every: int = 10
    ) -> VideoAnalysis:
        """Analyze sampled frames from an uploaded video."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        sampled_frames = 0
        analyses: list[FrameAnalysis] = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    sampled_frames += 1
                    analysis = self.analyze_frame(frame)
                    if analysis is not None:
                        analyses.append(analysis)

                frame_idx += 1
        finally:
            cap.release()
            os.unlink(video_path)

        if not analyses:
            return VideoAnalysis(
                detected_frames=0,
                sampled_frames=sampled_frames,
                exercise="not detected",
                form="unknown",
                confidence=0.0,
                common_feedback=[],
                preview_frame=None,
            )

        labels = [f"{item.exercise}_{item.form}" for item in analyses]
        label_counts = Counter(labels)
        top_label, top_count = label_counts.most_common(1)[0]
        top_exercise, top_form = ExerciseClassifier.parse_label(top_label)

        matching = [item for item in analyses if f"{item.exercise}_{item.form}" == top_label]
        avg_confidence = sum(item.confidence for item in matching) / len(matching)

        feedback_counts = Counter(
            msg for item in matching for msg in item.feedback
        )
        common_feedback = [msg for msg, _ in feedback_counts.most_common(3)]
        preview = max(matching, key=lambda item: item.confidence).annotated_frame

        return VideoAnalysis(
            detected_frames=top_count,
            sampled_frames=sampled_frames,
            exercise=top_exercise,
            form=top_form,
            confidence=avg_confidence,
            common_feedback=common_feedback,
            preview_frame=preview,
        )


class LiveExerciseMonitor:
    """Real-time analyzer for browser camera frames."""

    def __init__(self, model_path: Optional[str] = None):
        self.detector_error: Optional[str] = None
        self.detector = None
        self.classifier = ExerciseClassifier(model_path)
        self.form_analyzer = FormAnalyzer()
        self.rep_counter = RepCounter()
        self.metrics = LiveMetrics()
        self._lock = threading.Lock()
        self.selected_exercise = "auto"
        self.session_active = False
        self._probability_history: deque[dict[str, float]] = deque(maxlen=8)
        self._feedback_history: deque[list[str]] = deque(maxlen=5)
        self._form_score_history: deque[float] = deque(maxlen=5)
        self._proper_reps = 0
        self._needs_work_reps = 0
        self._session_started_at: float | None = None
        try:
            self.detector = PoseDetector(static_image_mode=False)
        except Exception as exc:
            self.detector_error = str(exc)

    @staticmethod
    def _normalise_exercise_name(exercise: str) -> str:
        return exercise.lower().replace("-", "").replace(" ", "")

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    def configure(self, selected_exercise: str, session_active: bool) -> None:
        """Update selected exercise and session state without losing counts."""
        selected = self._normalise_exercise_name(selected_exercise)

        with self._lock:
            previous_selected = self.selected_exercise
            previous_active = self.session_active
            self.selected_exercise = selected
            self.session_active = session_active

        if selected != previous_selected:
            if selected != "auto":
                self.rep_counter.reset(selected)
            elif not session_active:
                self.rep_counter.reset("")
            self._proper_reps = 0
            self._needs_work_reps = 0
            self._probability_history.clear()
            self._feedback_history.clear()
            self._form_score_history.clear()

        if session_active and not previous_active:
            base_exercise = "" if selected == "auto" else selected
            self.rep_counter.reset(base_exercise)
            self._proper_reps = 0
            self._needs_work_reps = 0
            self._session_started_at = time.time()
            self._probability_history.clear()
            self._feedback_history.clear()
            self._form_score_history.clear()
        elif not session_active and previous_active:
            self._session_started_at = None

    def end_session(self) -> None:
        """Freeze the latest count and stop incrementing until restarted."""
        with self._lock:
            self.session_active = False
            self._session_started_at = None

    def reset_session(self) -> None:
        """Explicitly zero the counter for the selected exercise."""
        selected = self.selected_exercise
        self.rep_counter.reset("" if selected == "auto" else selected)
        self._proper_reps = 0
        self._needs_work_reps = 0
        self._session_started_at = time.time() if self.session_active else None
        self._probability_history.clear()
        self._feedback_history.clear()
        self._form_score_history.clear()
        with self._lock:
            self.metrics.rep_status = self.rep_counter.get_status()

    def get_metrics(self) -> LiveMetrics:
        """Return a copy of the current metrics."""
        with self._lock:
            return LiveMetrics(
                exercise=self.metrics.exercise,
                form=self.metrics.form,
                confidence=self.metrics.confidence,
                feedback=list(self.metrics.feedback),
                rep_status=self.metrics.rep_status,
                pose_detected=self.metrics.pose_detected,
                session_active=self.metrics.session_active,
                selected_exercise=self.metrics.selected_exercise,
                form_score=self.metrics.form_score,
                proper_reps=self.metrics.proper_reps,
                needs_work_reps=self.metrics.needs_work_reps,
                session_seconds=self.metrics.session_seconds,
                status_badge=self.metrics.status_badge,
            )

    def _resolve_selected_prediction(
        self, probabilities: dict[str, float], selected_exercise: str
    ) -> tuple[str, str, float]:
        """Map class probabilities to a chosen exercise and right/wrong form."""
        correct_key = f"{selected_exercise}_correct"
        incorrect_key = f"{selected_exercise}_incorrect"
        correct_score = probabilities.get(correct_key, 0.0)
        incorrect_score = probabilities.get(incorrect_key, 0.0)
        total_score = correct_score + incorrect_score
        form = "correct" if correct_score >= incorrect_score else "incorrect"
        return selected_exercise, form, total_score

    def _smooth_probabilities(self, probabilities: dict[str, float]) -> dict[str, float]:
        self._probability_history.append(probabilities)
        labels = self.classifier.classes
        history_len = len(self._probability_history)
        return {
            label: sum(snapshot.get(label, 0.0) for snapshot in self._probability_history) / history_len
            for label in labels
        }

    def _stabilize_feedback(self, feedback: list[str]) -> list[str]:
        self._feedback_history.append(feedback)
        feedback_counts = Counter(item for batch in self._feedback_history for item in batch)
        return [msg for msg, count in feedback_counts.items() if count >= max(2, len(self._feedback_history) // 2)]

    @staticmethod
    def _status_from_score(score: float, confidence: float, pose_detected: bool) -> str:
        if not pose_detected:
            return "Need pose"
        if confidence < 0.45:
            return "Low confidence"
        if score >= 90:
            return "Excellent"
        if score >= 75:
            return "Strong"
        if score >= 55:
            return "Adjust"
        return "Correct form"

    def process_landmarks(self, landmarks_list: list, selected_exercise: str, session_active: bool) -> LiveMetrics:
        """Analyze pre-extracted landmarks directly from the client side."""
        landmarks = np.array(landmarks_list)
        
        selected = self._normalise_exercise_name(selected_exercise)
        with self._lock:
            # Dynamically configure if selected exercise or session state changed
            if selected != self.selected_exercise or session_active != self.session_active:
                self.configure(selected_exercise, session_active)
        
        # Core pipeline matching process_frame
        features = extract_features(landmarks)
        probabilities = self._smooth_probabilities(
            self.classifier.get_probabilities(features)
        )

        if self.selected_exercise != "auto":
            exercise, form, confidence = self._resolve_selected_prediction(
                probabilities, self.selected_exercise
            )
        else:
            label = max(probabilities, key=probabilities.get)
            confidence = probabilities[label]
            exercise, form = ExerciseClassifier.parse_label(label)

        raw_feedback = self.form_analyzer.analyze(exercise, landmarks)
        feedback = self._stabilize_feedback(raw_feedback)
        form_score = self.form_analyzer.get_form_score(exercise, landmarks)
        self._form_score_history.append(form_score)
        smoothed_form_score = sum(self._form_score_history) / len(self._form_score_history)
        status_badge = self._status_from_score(smoothed_form_score, confidence, True)

        previous_count = self.rep_counter.count
        if self.session_active:
            count_exercise = exercise if self.selected_exercise == "auto" else self.selected_exercise
            self.rep_counter.update(landmarks, count_exercise)
            if self.rep_counter.count > previous_count:
                if smoothed_form_score >= 75:
                    self._proper_reps += 1
                else:
                    self._needs_work_reps += 1

        session_seconds = (
            time.time() - self._session_started_at
            if self.session_active and self._session_started_at is not None
            else 0.0
        )

        metrics = LiveMetrics(
            exercise=exercise,
            form=form,
            confidence=confidence,
            feedback=feedback,
            rep_status=self.rep_counter.get_status(),
            pose_detected=True,
            session_active=self.session_active,
            selected_exercise=self.selected_exercise,
            form_score=smoothed_form_score,
            proper_reps=self._proper_reps,
            needs_work_reps=self._needs_work_reps,
            session_seconds=session_seconds,
            status_badge=status_badge,
        )

        with self._lock:
            self.metrics = metrics

        return metrics

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze and annotate a live BGR frame."""
        if self.detector is None:
            metrics = LiveMetrics(
                exercise="unavailable",
                form="unknown",
                confidence=0.0,
                feedback=[
                    "Pose detector could not start in this runtime.",
                    "Try a local GUI session or a deployment/runtime with camera-safe OpenGL support.",
                ],
                rep_status=self.rep_counter.get_status(),
                pose_detected=False,
                session_active=self.session_active,
                selected_exercise=self.selected_exercise,
                proper_reps=self._proper_reps,
                needs_work_reps=self._needs_work_reps,
                status_badge="Detector unavailable",
            )
            with self._lock:
                self.metrics = metrics
            return draw_live_overlay(frame.copy(), metrics)

        results = self.detector.detect(frame)
        landmarks = self.detector.landmarks_from_results(results)
        annotated = self.detector.draw_landmarks(frame, results)

        with self._lock:
            selected_exercise = self.selected_exercise
            session_active = self.session_active

        metrics = LiveMetrics(
            session_active=session_active,
            selected_exercise=selected_exercise,
        )

        if landmarks is not None:
            features = extract_features(landmarks)
            probabilities = self._smooth_probabilities(
                self.classifier.get_probabilities(features)
            )

            if selected_exercise != "auto":
                exercise, form, confidence = self._resolve_selected_prediction(
                    probabilities, selected_exercise
                )
            else:
                label = max(probabilities, key=probabilities.get)
                confidence = probabilities[label]
                exercise, form = ExerciseClassifier.parse_label(label)

            raw_feedback = self.form_analyzer.analyze(exercise, landmarks)
            feedback = self._stabilize_feedback(raw_feedback)
            form_score = self.form_analyzer.get_form_score(exercise, landmarks)
            self._form_score_history.append(form_score)
            smoothed_form_score = sum(self._form_score_history) / len(self._form_score_history)
            status_badge = self._status_from_score(smoothed_form_score, confidence, True)

            previous_count = self.rep_counter.count
            if session_active:
                count_exercise = exercise if selected_exercise == "auto" else selected_exercise
                self.rep_counter.update(landmarks, count_exercise)
                if self.rep_counter.count > previous_count:
                    if smoothed_form_score >= 75:
                        self._proper_reps += 1
                    else:
                        self._needs_work_reps += 1

            session_seconds = (
                time.time() - self._session_started_at
                if session_active and self._session_started_at is not None
                else 0.0
            )

            metrics = LiveMetrics(
                exercise=exercise,
                form=form,
                confidence=confidence,
                feedback=feedback,
                rep_status=self.rep_counter.get_status(),
                pose_detected=True,
                session_active=session_active,
                selected_exercise=selected_exercise,
                form_score=smoothed_form_score,
                proper_reps=self._proper_reps,
                needs_work_reps=self._needs_work_reps,
                session_seconds=session_seconds,
                status_badge=status_badge,
            )
        else:
            metrics.rep_status = self.rep_counter.get_status()
            metrics.proper_reps = self._proper_reps
            metrics.needs_work_reps = self._needs_work_reps
            metrics.session_seconds = (
                time.time() - self._session_started_at
                if session_active and self._session_started_at is not None
                else 0.0
            )
            metrics.status_badge = "Need pose"

        with self._lock:
            self.metrics = metrics

        return draw_live_overlay(annotated, metrics)
