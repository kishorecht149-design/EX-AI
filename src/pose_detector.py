"""
pose_detector.py — MediaPipe Pose Detection Module
====================================================
Wraps Google MediaPipe PoseLandmarker (Tasks API, v0.10.20+) to detect
33 body landmarks from images, video files, or live webcam frames.

Each landmark provides (x, y, z, visibility) in normalised coordinates.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List

# New MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pose_landmarker_lite.task",
)


class PoseDetector:
    """Detect human body pose using MediaPipe PoseLandmarker (Tasks API)."""

    # Landmark names for reference (MediaPipe Pose has 33 landmarks)
    LANDMARK_NAMES: List[str] = [
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

    # Pose connections for skeleton drawing (pairs of landmark indices)
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        static_image_mode: bool = False,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialise the pose detector.

        Args:
            model_path: Path to the .task model file. Uses default if None.
            static_image_mode: If True, treats each image independently (IMAGE mode).
                Set True for batch processing, False for video.
            num_poses: Maximum number of poses to detect.
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.static_image_mode = static_image_mode

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"PoseLandmarker model not found at {self.model_path}. "
                "Download it with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', "
                "'models/pose_landmarker_lite.task')\""
            )

        running_mode = VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray):
        """
        Run pose detection on a single BGR frame.

        Args:
            frame: BGR image (OpenCV format).

        Returns:
            MediaPipe PoseLandmarkerResult object.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.static_image_mode:
            results = self.landmarker.detect(mp_image)
        else:
            self._frame_timestamp_ms += 33  # ~30 FPS
            results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        return results

    def get_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 33 landmarks as an (33, 4) numpy array.

        Each row: [x, y, z, visibility].
        Returns None if no pose is detected.

        Args:
            frame: BGR image.

        Returns:
            np.ndarray of shape (33, 4) or None.
        """
        results = self.detect(frame)
        if not results.pose_landmarks:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks[0]]
        )
        return landmarks

    def get_landmarks_xy(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return landmarks as pixel coordinates (x_px, y_px) for drawing.

        Args:
            frame: BGR image.

        Returns:
            np.ndarray of shape (33, 2) with pixel coords, or None.
        """
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            return None
        h, w, _ = frame.shape
        xy = landmarks[:, :2].copy()
        xy[:, 0] *= w
        xy[:, 1] *= h
        return xy.astype(int)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def draw_landmarks(self, frame: np.ndarray, results=None) -> np.ndarray:
        """
        Draw the pose skeleton on the frame.

        Args:
            frame: BGR image.
            results: Optional pre-computed results. If None, detection runs again.

        Returns:
            Frame with landmarks drawn.
        """
        if results is None:
            results = self.detect(frame)

        annotated = frame.copy()
        if not results.pose_landmarks:
            return annotated

        h, w, _ = annotated.shape
        landmarks = results.pose_landmarks[0]

        # Draw connections
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        return annotated

    # ------------------------------------------------------------------
    # Video / webcam helpers
    # ------------------------------------------------------------------
    def process_video(self, video_path: str, callback=None) -> List[Optional[np.ndarray]]:
        """
        Process every frame of a video file and return landmarks per frame.

        Args:
            video_path: Path to the video file.
            callback: Optional function(frame, landmarks) called per frame.

        Returns:
            List of landmark arrays (33, 4) or None entries.
        """
        cap = cv2.VideoCapture(video_path)
        all_landmarks: List[Optional[np.ndarray]] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.get_landmarks(frame)
            all_landmarks.append(landmarks)

            if callback:
                callback(frame, landmarks)

        cap.release()
        return all_landmarks

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
