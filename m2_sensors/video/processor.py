"""
M2 Video Intelligence Module
- Eye contact detection via MediaPipe Face Mesh
- Facial expression / emotion recognition via DeepFace
- Posture analysis via MediaPipe Pose
- Outputs: engagement score, stress indicators, eye contact ratio
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# MediaPipe imports
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# DeepFace for emotion
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[Video] DeepFace not available — emotion analysis disabled")


@dataclass
class VideoFrameResult:
    eye_contact: bool                  # Is candidate looking at camera?
    eye_contact_score: float           # Rolling average (0–100)
    emotion: str                       # Dominant emotion
    emotion_scores: Dict[str, float]   # All emotion probabilities
    engagement_score: float            # 0–100
    stress_score: float                # 0–100
    posture_score: float               # 0–100 (upright = high)
    face_detected: bool
    gaze_direction: str                # "center", "left", "right", "up", "down"
    timestamp: float = field(default_factory=time.time)


class VideoProcessor:
    """
    Processes webcam frames for:
    - Eye contact (MediaPipe Face Mesh + iris landmarks)
    - Facial expression / emotion (DeepFace)
    - Posture (MediaPipe Pose)
    """

    def __init__(self, emotion_every_n_frames: int = 15):
        # How often to run the heavier DeepFace emotion analysis
        self.emotion_every_n = emotion_every_n_frames
        self._frame_count = 0

        # Rolling averages
        self._eye_contact_history: List[bool] = []
        self._engagement_history: List[float] = []
        self._stress_history: List[float] = []
        self._history_len = 30  # rolling window

        # Last emotion result (reuse between DeepFace calls)
        self._last_emotion = "neutral"
        self._last_emotion_scores: Dict[str, float] = {}

        # MediaPipe models
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Iris landmark indices (MediaPipe)
        # Left iris: 468–472, Right iris: 473–477
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        # Eye corner landmarks
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]

    # ─────────────────────── Eye Contact ──────────────────────────

    def _get_iris_center(self, landmarks, iris_indices, img_w, img_h):
        points = [(landmarks[i].x * img_w, landmarks[i].y * img_h)
                  for i in iris_indices if i < len(landmarks)]
        if not points:
            return None
        return np.mean(points, axis=0)

    def _get_eye_center(self, landmarks, corner_indices, img_w, img_h):
        points = [(landmarks[i].x * img_w, landmarks[i].y * img_h)
                  for i in corner_indices if i < len(landmarks)]
        if not points:
            return None
        return np.mean(points, axis=0)

    def _compute_gaze(self, landmarks, img_w, img_h) -> Tuple[bool, str, float]:
        """
        Returns (is_eye_contact, gaze_direction, gaze_offset_ratio)
        Uses iris position relative to eye corners.
        """
        lm = landmarks.landmark

        l_iris = self._get_iris_center(lm, self.LEFT_IRIS, img_w, img_h)
        r_iris = self._get_iris_center(lm, self.RIGHT_IRIS, img_w, img_h)
        l_eye = self._get_eye_center(lm, self.LEFT_EYE_CORNERS, img_w, img_h)
        r_eye = self._get_eye_center(lm, self.RIGHT_EYE_CORNERS, img_w, img_h)

        if l_iris is None or r_iris is None or l_eye is None or r_eye is None:
            return False, "unknown", 1.0

        # Horizontal gaze ratio for each eye
        # eye_width = distance between corners
        def horizontal_ratio(iris, corners_lm, indices):
            left_pt = np.array([lm[indices[0]].x * img_w, lm[indices[0]].y * img_h])
            right_pt = np.array([lm[indices[1]].x * img_w, lm[indices[1]].y * img_h])
            eye_width = np.linalg.norm(right_pt - left_pt)
            if eye_width < 1:
                return 0.5
            offset = iris[0] - left_pt[0]
            return float(offset / eye_width)  # 0 = far left, 1 = far right, 0.5 = center

        l_ratio = horizontal_ratio(l_iris, None, self.LEFT_EYE_CORNERS)
        r_ratio = horizontal_ratio(r_iris, None, self.RIGHT_EYE_CORNERS)
        avg_ratio = (l_ratio + r_ratio) / 2

        # Vertical: compare iris y to eye center y
        l_iris_y = l_iris[1]
        r_iris_y = r_iris[1]
        l_eye_y = l_eye[1]
        r_eye_y = r_eye[1]
        avg_vertical = ((l_iris_y - l_eye_y) + (r_iris_y - r_eye_y)) / 2

        # Determine gaze direction
        if avg_ratio < 0.35:
            direction = "left"
        elif avg_ratio > 0.65:
            direction = "right"
        elif avg_vertical < -3:
            direction = "up"
        elif avg_vertical > 3:
            direction = "down"
        else:
            direction = "center"

        # Eye contact = looking center
        center_offset = abs(avg_ratio - 0.5)
        is_contact = direction == "center"

        return is_contact, direction, center_offset

    # ─────────────────────── Emotion ──────────────────────────────

    def _analyze_emotion(self, frame: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Run DeepFace emotion analysis on frame."""
        if not DEEPFACE_AVAILABLE:
            return "neutral", {}
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                result = result[0]
            emotion = result.get("dominant_emotion", "neutral")
            scores = result.get("emotion", {})
            return emotion, {k: round(float(v), 2) for k, v in scores.items()}
        except Exception:
            return self._last_emotion, self._last_emotion_scores

    # ─────────────────────── Posture ──────────────────────────────

    def _analyze_posture(self, frame: np.ndarray) -> float:
        """
        Returns posture score 0–100.
        Uses shoulder alignment and head position from MediaPipe Pose.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return 70.0  # neutral default

        lm = result.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Key landmarks
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_EAR = 7
        RIGHT_EAR = 8
        NOSE = 0

        try:
            ls = np.array([lm[LEFT_SHOULDER].x * w, lm[LEFT_SHOULDER].y * h])
            rs = np.array([lm[RIGHT_SHOULDER].x * w, lm[RIGHT_SHOULDER].y * h])
            le = np.array([lm[LEFT_EAR].x * w, lm[LEFT_EAR].y * h])
            re = np.array([lm[RIGHT_EAR].x * w, lm[RIGHT_EAR].y * h])
            nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])
        except IndexError:
            return 70.0

        score = 100.0

        # Shoulder tilt (level = good)
        shoulder_tilt = abs(ls[1] - rs[1])
        score -= min(shoulder_tilt / 10, 20)

        # Head forward posture (nose should be between shoulders horizontally)
        shoulder_center_x = (ls[0] + rs[0]) / 2
        head_offset = abs(nose[0] - shoulder_center_x) / max(abs(ls[0] - rs[0]), 1)
        score -= min(head_offset * 30, 20)

        # Ear-to-shoulder vertical alignment (ears should be above shoulders)
        ear_mid_y = (le[1] + re[1]) / 2
        shoulder_mid_y = (ls[1] + rs[1]) / 2
        if ear_mid_y > shoulder_mid_y:  # ears below shoulders = slouching
            score -= 20

        return round(max(0.0, min(100.0, score)), 1)

    # ─────────────────────── Scoring ──────────────────────────────

    def _compute_engagement(self, eye_contact: bool, emotion: str, posture_score: float) -> float:
        score = 50.0

        # Eye contact is the biggest factor
        if eye_contact:
            score += 25.0
        else:
            score -= 10.0

        # Emotion contribution
        positive_emotions = {"happy": 20, "surprise": 10, "neutral": 5}
        negative_emotions = {"sad": -15, "angry": -20, "fear": -15, "disgust": -10}
        score += positive_emotions.get(emotion, 0)
        score += negative_emotions.get(emotion, 0)

        # Posture
        score += (posture_score - 70) * 0.3

        return round(max(0.0, min(100.0, score)), 1)

    def _compute_stress(self, emotion: str, emotion_scores: Dict[str, float], eye_contact: bool) -> float:
        score = 20.0  # baseline low stress

        # Stress-related emotions
        stress_emotions = {"fear", "angry", "sad", "disgust"}
        for em in stress_emotions:
            score += emotion_scores.get(em, 0) * 0.5

        # Darting eyes = stress
        if not eye_contact:
            score += 15.0

        # Low eye contact rolling average = stress
        recent_contact = sum(self._eye_contact_history[-10:]) / max(len(self._eye_contact_history[-10:]), 1)
        if recent_contact < 0.3:
            score += 15.0

        return round(max(0.0, min(100.0, score)), 1)

    # ─────────────────────── Main Process ─────────────────────────

    def process_frame(self, frame: np.ndarray) -> VideoFrameResult:
        """
        Process a single BGR frame.
        Returns VideoFrameResult with all scores.
        """
        self._frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Face Mesh ---
        face_result = self._face_mesh.process(rgb)
        face_detected = face_result.multi_face_landmarks is not None

        eye_contact = False
        gaze_direction = "unknown"

        if face_detected:
            landmarks = face_result.multi_face_landmarks[0]
            eye_contact, gaze_direction, _ = self._compute_gaze(landmarks, w, h)

        # Update eye contact history
        self._eye_contact_history.append(eye_contact)
        if len(self._eye_contact_history) > self._history_len:
            self._eye_contact_history.pop(0)

        eye_contact_score = (
            sum(self._eye_contact_history) / len(self._eye_contact_history) * 100
        )

        # --- Emotion (every N frames) ---
        if self._frame_count % self.emotion_every_n == 0 and face_detected:
            self._last_emotion, self._last_emotion_scores = self._analyze_emotion(frame)

        emotion = self._last_emotion
        emotion_scores = self._last_emotion_scores

        # --- Posture ---
        posture_score = self._analyze_posture(frame)

        # --- Engagement & Stress ---
        engagement = self._compute_engagement(eye_contact, emotion, posture_score)
        stress = self._compute_stress(emotion, emotion_scores, eye_contact)

        return VideoFrameResult(
            eye_contact=eye_contact,
            eye_contact_score=round(eye_contact_score, 1),
            emotion=emotion,
            emotion_scores=emotion_scores,
            engagement_score=engagement,
            stress_score=stress,
            posture_score=posture_score,
            face_detected=face_detected,
            gaze_direction=gaze_direction,
            timestamp=time.time()
        )

    def annotate_frame(self, frame: np.ndarray, result: VideoFrameResult) -> np.ndarray:
        """Draw debug overlay on frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        color = (0, 255, 0) if result.eye_contact else (0, 0, 255)
        cv2.putText(annotated, f"Eye Contact: {'YES' if result.eye_contact else 'NO'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Gaze: {result.gaze_direction}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated, f"Emotion: {result.emotion}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(annotated, f"Engagement: {result.engagement_score:.0f}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)
        cv2.putText(annotated, f"Posture: {result.posture_score:.0f}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(annotated, f"Stress: {result.stress_score:.0f}",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        return annotated

    def close(self):
        self._face_mesh.close()
        self._pose.close()
