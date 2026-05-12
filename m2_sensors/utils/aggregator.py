"""
M2 Session Aggregator
- Accumulates audio and video results across entire interview
- Computes session-level metrics and trends
- Generates final M2 report for M1 (coaching agent)
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from audio.processor import AudioAnalysisResult
from video.processor import VideoFrameResult


@dataclass
class SessionMetrics:
    """Aggregated metrics for the full interview session."""

    # Audio
    avg_confidence_score: float = 0.0
    avg_communication_clarity: float = 0.0
    avg_speech_rate_wpm: float = 0.0
    total_hesitations: int = 0
    avg_pause_ratio: float = 0.0
    confidence_trend: str = "stable"   # "improving", "declining", "stable"
    full_transcript: str = ""

    # Video
    avg_eye_contact_score: float = 0.0
    avg_engagement_score: float = 0.0
    avg_stress_score: float = 0.0
    avg_posture_score: float = 0.0
    dominant_emotion: str = "neutral"
    eye_contact_percentage: float = 0.0

    # Combined
    overall_confidence: float = 0.0
    overall_engagement: float = 0.0
    nervousness_level: str = "low"     # "low", "moderate", "high"

    # Coaching flags
    coaching_flags: List[str] = field(default_factory=list)

    session_duration_seconds: float = 0.0
    audio_chunk_count: int = 0
    video_frame_count: int = 0


class SessionAggregator:
    """
    Maintains running state across the interview session.
    Call add_audio_result() and add_video_result() as they arrive.
    Call get_session_metrics() at any point for current aggregated state.
    """

    def __init__(self):
        self._audio_results: List[AudioAnalysisResult] = []
        self._video_results: List[VideoFrameResult] = []
        self._start_time = time.time()

    def add_audio_result(self, result: AudioAnalysisResult):
        self._audio_results.append(result)

    def add_video_result(self, result: VideoFrameResult):
        self._video_results.append(result)

    def _avg(self, values: List[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    def _confidence_trend(self) -> str:
        """Compute whether confidence is improving or declining."""
        scores = [r.confidence_score for r in self._audio_results]
        if len(scores) < 3:
            return "stable"
        first_half = self._avg(scores[:len(scores)//2])
        second_half = self._avg(scores[len(scores)//2:])
        delta = second_half - first_half
        if delta > 5:
            return "improving"
        elif delta < -5:
            return "declining"
        return "stable"

    def _dominant_emotion(self) -> str:
        """Most frequent dominant emotion over session."""
        from collections import Counter
        emotions = [r.emotion for r in self._video_results if r.face_detected]
        if not emotions:
            return "neutral"
        return Counter(emotions).most_common(1)[0][0]

    def _compute_coaching_flags(self, metrics: SessionMetrics) -> List[str]:
        flags = []

        if metrics.avg_confidence_score < 50:
            flags.append("Low vocal confidence detected — practice speaking more slowly and deliberately")
        if metrics.avg_communication_clarity < 55:
            flags.append(f"High filler word usage ({metrics.total_hesitations} instances) — reduce 'um', 'uh', 'like'")
        if metrics.avg_speech_rate_wpm > 190:
            flags.append("Speaking too fast — slow down to improve clarity (target: 120–160 WPM)")
        elif metrics.avg_speech_rate_wpm > 0 and metrics.avg_speech_rate_wpm < 80:
            flags.append("Speaking too slowly — try to be more direct and confident")
        if metrics.avg_eye_contact_score < 40:
            flags.append("Limited eye contact with camera — maintain gaze toward the interviewer")
        if metrics.avg_posture_score < 60:
            flags.append("Posture issues detected — sit upright with shoulders level")
        if metrics.avg_stress_score > 65:
            flags.append("High stress indicators — practice deep breathing before and during responses")
        if metrics.avg_pause_ratio > 0.6:
            flags.append("Excessive pausing — structure your answers with STAR method to reduce silences")
        if metrics.confidence_trend == "declining":
            flags.append("Confidence declined during interview — fatigue or difficulty with questions noted")
        elif metrics.confidence_trend == "improving":
            flags.append("Great job — confidence improved as the interview progressed")

        if not flags:
            flags.append("Strong performance — keep maintaining your communication style")

        return flags

    def get_session_metrics(self) -> SessionMetrics:
        """Compute and return full session metrics."""
        metrics = SessionMetrics()
        duration = time.time() - self._start_time
        metrics.session_duration_seconds = round(duration, 1)
        metrics.audio_chunk_count = len(self._audio_results)
        metrics.video_frame_count = len(self._video_results)

        # Audio metrics
        if self._audio_results:
            metrics.avg_confidence_score = self._avg([r.confidence_score for r in self._audio_results])
            metrics.avg_communication_clarity = self._avg([r.communication_clarity_score for r in self._audio_results])
            metrics.avg_speech_rate_wpm = self._avg([r.speech_rate_wpm for r in self._audio_results if r.speech_rate_wpm > 0])
            metrics.total_hesitations = sum(r.hesitation_count for r in self._audio_results)
            metrics.avg_pause_ratio = self._avg([r.pause_ratio for r in self._audio_results])
            metrics.confidence_trend = self._confidence_trend()
            metrics.full_transcript = " ".join(
                r.transcript for r in self._audio_results if r.transcript
            ).strip()

        # Video metrics
        if self._video_results:
            metrics.avg_eye_contact_score = self._avg([r.eye_contact_score for r in self._video_results])
            metrics.avg_engagement_score = self._avg([r.engagement_score for r in self._video_results])
            metrics.avg_stress_score = self._avg([r.stress_score for r in self._video_results])
            metrics.avg_posture_score = self._avg([r.posture_score for r in self._video_results])
            metrics.dominant_emotion = self._dominant_emotion()
            contact_frames = sum(1 for r in self._video_results if r.eye_contact)
            metrics.eye_contact_percentage = round(contact_frames / len(self._video_results) * 100, 1)

        # Combined scores
        audio_conf = metrics.avg_confidence_score
        video_eng = metrics.avg_engagement_score
        metrics.overall_confidence = round((audio_conf * 0.6 + (100 - metrics.avg_stress_score) * 0.4), 1)
        metrics.overall_engagement = round((video_eng * 0.7 + metrics.avg_eye_contact_score * 0.3), 1)

        # Nervousness level
        if metrics.avg_stress_score > 65 or metrics.avg_confidence_score < 40:
            metrics.nervousness_level = "high"
        elif metrics.avg_stress_score > 40 or metrics.avg_confidence_score < 60:
            metrics.nervousness_level = "moderate"
        else:
            metrics.nervousness_level = "low"

        # Coaching flags
        metrics.coaching_flags = self._compute_coaching_flags(metrics)

        return metrics

    def to_dict(self) -> dict:
        return asdict(self.get_session_metrics())

    def reset(self):
        """Reset for a new session."""
        self._audio_results.clear()
        self._video_results.clear()
        self._start_time = time.time()
