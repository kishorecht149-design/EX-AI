"""
exercise_classifier.py — Exercise Classification Wrapper
==========================================================
Loads a trained scikit-learn model (``models/exercise_model.pkl``) and
exposes a simple ``predict(features)`` method.

Supports 8 classes:
    squat_correct, squat_incorrect,
    pushup_correct, pushup_incorrect,
    lunge_correct, lunge_incorrect,
    plank_correct, plank_incorrect
"""

import os
import joblib
import numpy as np
from typing import Optional, Tuple

# Default model path (relative to repository root)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "exercise_model.pkl",
)


class ExerciseClassifier:
    """Load and use a trained exercise classification model."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Load the model from disk.

        Args:
            model_path: Path to the pickled sklearn model.
                        Defaults to ``models/exercise_model.pkl``.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first (see notebooks/model_training.ipynb)."
            )

        self.model = joblib.load(self.model_path)
        self.classes = list(self.model.classes_)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> str:
        """
        Predict the exercise class for a single feature vector.

        Args:
            features: 1-D array of shape (80,) from ``extract_features()``.

        Returns:
            Predicted class label, e.g. ``"squat_correct"``.
        """
        features = np.array(features).reshape(1, -1)
        return str(self.model.predict(features)[0])

    def predict_proba(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict with confidence score.

        Args:
            features: 1-D feature vector.

        Returns:
            (predicted_label, confidence) tuple.
        """
        features = np.array(features).reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        return self.classes[idx], float(proba[idx])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict for a batch of feature vectors.

        Args:
            X: 2-D array of shape (N, 80).

        Returns:
            1-D array of predicted labels.
        """
        return self.model.predict(X)

    @staticmethod
    def parse_label(label: str) -> Tuple[str, str]:
        """
        Split a combined label into (exercise, form).

        Example:
            >>> ExerciseClassifier.parse_label("squat_correct")
            ('squat', 'correct')
        """
        parts = label.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return label, "unknown"
