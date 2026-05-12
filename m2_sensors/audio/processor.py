"""
M2 Audio Intelligence Module
- Speech-to-Text using OpenAI Whisper
- Audio feature analysis using Librosa
- Outputs: transcript, confidence score, communication clarity score
"""

import os
import io
import time
import tempfile
import numpy as np
import soundfile as sf
import librosa
import whisper
from dataclasses import dataclass, asdict
from typing import Optional

# Load whisper model once at startup (use 'base' for speed, 'small' for accuracy)
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_size = os.getenv("WHISPER_MODEL", "base")
        print(f"[Audio] Loading Whisper model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


@dataclass
class AudioAnalysisResult:
    transcript: str
    confidence_score: float          # 0–100: how confident the speaker sounds
    communication_clarity_score: float  # 0–100: clarity of speech
    hesitation_count: int            # number of filler words / long pauses
    speech_rate_wpm: float           # words per minute
    pitch_mean: float                # mean fundamental frequency
    pitch_variation: float           # std dev of pitch (monotone vs expressive)
    pause_ratio: float               # ratio of silence to total duration
    energy_mean: float               # mean RMS energy
    duration_seconds: float
    timestamp: float


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe audio bytes using Whisper."""
    model = get_whisper_model()
    
    # Save to temp file (Whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        # Write raw audio bytes as WAV
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(tmp_path, audio_array, sample_rate)
    
    try:
        result = model.transcribe(tmp_path, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def analyze_audio_features(audio_bytes: bytes, sample_rate: int = 16000) -> dict:
    """
    Analyze acoustic features from raw audio bytes.
    Returns pitch, energy, pause ratio, speech rate, etc.
    """
    # Convert bytes to numpy float array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio_array) / sample_rate

    if duration < 0.5:
        return _empty_features(duration)

    # --- Pitch Analysis (F0 via PYIN) ---
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_array,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        f0_voiced = f0[voiced_flag]
        pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_variation = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_variation = 0.0

    # --- Energy Analysis ---
    rms = librosa.feature.rms(y=audio_array)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # --- Pause / Silence Detection ---
    # Frames with RMS below threshold are considered silence
    silence_threshold = max(0.01, energy_mean * 0.2)
    silence_frames = np.sum(rms < silence_threshold)
    pause_ratio = float(silence_frames / len(rms)) if len(rms) > 0 else 0.0

    # --- Speech Rate Estimation (via zero crossing rate as rough proxy) ---
    # Actual WPM computed from transcript word count / duration
    zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
    zcr_mean = float(np.mean(zcr))

    return {
        "pitch_mean": pitch_mean,
        "pitch_variation": pitch_variation,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "pause_ratio": pause_ratio,
        "zcr_mean": zcr_mean,
        "duration": duration
    }


def _empty_features(duration: float) -> dict:
    return {
        "pitch_mean": 0.0, "pitch_variation": 0.0,
        "energy_mean": 0.0, "energy_std": 0.0,
        "pause_ratio": 1.0, "zcr_mean": 0.0,
        "duration": duration
    }


FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "literally", "actually", "so", "right", "okay", "hmm"
}


def count_hesitations(transcript: str) -> int:
    """Count filler words in transcript."""
    words = transcript.lower().split()
    count = sum(1 for w in words if w in FILLER_WORDS)
    return count


def compute_speech_rate(transcript: str, duration_seconds: float) -> float:
    """Compute words per minute."""
    if duration_seconds < 0.1:
        return 0.0
    word_count = len(transcript.split())
    return round((word_count / duration_seconds) * 60, 1)


def compute_confidence_score(features: dict, hesitation_count: int, transcript: str) -> float:
    """
    Heuristic confidence score (0–100) based on:
    - Pitch variation (expressive = more confident, but too high = nervous)
    - Pause ratio (too many pauses = less confident)
    - Energy (louder = more confident)
    - Hesitation frequency
    - Speech rate (too slow or too fast = less confident)
    """
    score = 70.0  # baseline

    # Pause ratio penalty (ideal: 20-40% pauses)
    pause_ratio = features.get("pause_ratio", 0.5)
    if pause_ratio > 0.6:
        score -= (pause_ratio - 0.6) * 60  # heavy penalty for too many pauses
    elif pause_ratio < 0.1:
        score -= 5  # slight penalty for non-stop rambling

    # Pitch variation bonus (monotone = low confidence)
    pitch_var = features.get("pitch_variation", 0)
    if pitch_var > 0:
        if pitch_var < 10:
            score -= 10  # monotone
        elif 20 <= pitch_var <= 60:
            score += 10  # natural variation
        elif pitch_var > 80:
            score -= 5  # overly erratic

    # Energy bonus
    energy = features.get("energy_mean", 0)
    if energy < 0.02:
        score -= 10  # too quiet
    elif energy > 0.05:
        score += 5  # good vocal energy

    # Hesitation penalty
    duration = features.get("duration", 1)
    hesitation_rate = hesitation_count / max(duration / 60, 0.1)  # per minute
    score -= min(hesitation_rate * 3, 20)

    # Speech rate penalty (ideal: 120–160 WPM)
    wpm = compute_speech_rate(transcript, duration)
    if wpm > 0:
        if wpm < 80:
            score -= 10
        elif wpm > 200:
            score -= 10
        elif 120 <= wpm <= 160:
            score += 5

    return round(max(0.0, min(100.0, score)), 1)


def compute_communication_clarity(features: dict, hesitation_count: int, transcript: str) -> float:
    """
    Communication clarity score (0–100) based on:
    - Sentence coherence (word count / filler ratio)
    - Pause ratio (structured pauses are good)
    - Energy consistency
    """
    score = 70.0

    duration = features.get("duration", 1)
    word_count = len(transcript.split())

    # Too short responses
    if word_count < 5 and duration > 3:
        score -= 20

    # Filler word ratio
    filler_ratio = hesitation_count / max(word_count, 1)
    score -= min(filler_ratio * 100, 25)

    # Energy consistency (high std = varied energy = more dynamic / clear)
    energy_std = features.get("energy_std", 0)
    if energy_std > 0.02:
        score += 5

    # Reasonable pause ratio for emphasis (10-35% ideal)
    pause_ratio = features.get("pause_ratio", 0.5)
    if 0.10 <= pause_ratio <= 0.35:
        score += 10
    elif pause_ratio > 0.55:
        score -= 15

    return round(max(0.0, min(100.0, score)), 1)


def analyze_audio_chunk(audio_bytes: bytes, sample_rate: int = 16000) -> AudioAnalysisResult:
    """
    Full pipeline: raw audio bytes → AudioAnalysisResult
    Call this from the API endpoint.
    """
    start_time = time.time()

    # Step 1: Transcribe
    try:
        transcript = transcribe_audio(audio_bytes, sample_rate)
    except Exception as e:
        print(f"[Audio] Transcription error: {e}")
        transcript = ""

    # Step 2: Extract acoustic features
    try:
        features = analyze_audio_features(audio_bytes, sample_rate)
    except Exception as e:
        print(f"[Audio] Feature extraction error: {e}")
        features = _empty_features(0)

    # Step 3: Compute derived metrics
    hesitation_count = count_hesitations(transcript)
    duration = features.get("duration", 0)
    speech_rate = compute_speech_rate(transcript, duration)
    confidence = compute_confidence_score(features, hesitation_count, transcript)
    clarity = compute_communication_clarity(features, hesitation_count, transcript)

    return AudioAnalysisResult(
        transcript=transcript,
        confidence_score=confidence,
        communication_clarity_score=clarity,
        hesitation_count=hesitation_count,
        speech_rate_wpm=speech_rate,
        pitch_mean=features.get("pitch_mean", 0.0),
        pitch_variation=features.get("pitch_variation", 0.0),
        pause_ratio=features.get("pause_ratio", 0.0),
        energy_mean=features.get("energy_mean", 0.0),
        duration_seconds=duration,
        timestamp=start_time
    )
