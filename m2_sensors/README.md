# M2 — Sensors Module
### Audio + Video Intelligence for Mock Interview Agent

---

## What M2 Does

M2 is the **sensory layer** of the interview platform. It listens and watches the candidate in real time:

| Input | Analysis | Output |
|---|---|---|
| 🎙️ Microphone | Whisper STT + Librosa | Transcript, Confidence Score, Clarity Score, Speech Rate, Hesitations |
| 📷 Webcam | MediaPipe + DeepFace | Eye Contact Score, Emotion, Engagement Score, Stress Score, Posture Score |

All scores are sent to:
- **M1 (AI Brain)** — to adapt question difficulty in real time
- **M3 (Frontend)** — to display live meters in the interview dashboard

---

## Architecture

```
Microphone
    │
    ▼
AudioRecorder (sounddevice, 16kHz, 5s chunks)
    │
    ▼
Whisper STT ──────────────────────► Transcript
    +
Librosa Feature Extraction ───────► Pitch, Energy, Pause Ratio
    │
    ▼
Scoring Engine ───────────────────► Confidence Score (0-100)
                                    Communication Clarity (0-100)
                                    Speech Rate (WPM)
                                    Hesitation Count

Webcam
    │
    ▼
VideoCapture (OpenCV, 15fps)
    │
    ▼
MediaPipe Face Mesh ──────────────► Eye Contact (iris tracking)
    +                               Gaze Direction
MediaPipe Pose ───────────────────► Posture Score
    +
DeepFace (every ~1s) ─────────────► Emotion (7 classes)
    │
    ▼
Scoring Engine ───────────────────► Engagement Score (0-100)
                                    Stress Score (0-100)
                                    Eye Contact % 

SessionAggregator ────────────────► Full session metrics + coaching flags
FastAPI Server ───────────────────► REST + WebSocket APIs for M1 & M3
```

---

## Setup

### Prerequisites
- Python 3.10+
- Webcam + Microphone
- `ffmpeg` installed (for Whisper)

```bash
# Install ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg

# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Windows)
choco install ffmpeg
```

### Install

```bash
cd m2_sensors

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> First run will download the Whisper `base` model (~140MB). Set `WHISPER_MODEL=small` for better accuracy (~460MB).

---

## Running

### Option 1: Start API Server (for M3 integration)

```bash
python start_server.py
```

Server runs at `http://localhost:8001`

- API docs: `http://localhost:8001/docs`
- WebSocket audio: `ws://localhost:8001/ws/audio`
- WebSocket video: `ws://localhost:8001/ws/video`

### Option 2: Local CLI Test (mic + webcam directly)

```bash
# Basic run (60 seconds)
python run_local.py

# With video preview window
python run_local.py --preview

# Run for 2 minutes
python run_local.py --duration 120

# Audio only (no webcam)
python run_local.py --audio-only

# Video only (no mic)
python run_local.py --video-only
```

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/audio/analyze` | Upload audio chunk → scores |
| POST | `/video/analyze` | Upload JPEG frame → scores |
| GET | `/session/metrics` | Full session aggregated scores |
| POST | `/session/reset` | Reset for new interview |

### WebSocket Endpoints

#### `ws://localhost:8001/ws/audio`
- **Client → Server**: Raw PCM int16 bytes (16kHz mono)
- **Server → Client**: JSON with scores

```json
{
  "type": "audio_result",
  "transcript": "I have 3 years of experience with Python...",
  "confidence_score": 72.5,
  "communication_clarity_score": 68.0,
  "hesitation_count": 2,
  "speech_rate_wpm": 145.0,
  "pause_ratio": 0.28
}
```

#### `ws://localhost:8001/ws/video`
- **Client → Server**: JPEG-encoded frame bytes
- **Server → Client**: JSON with scores

```json
{
  "type": "video_result",
  "face_detected": true,
  "eye_contact": true,
  "eye_contact_score": 78.3,
  "gaze_direction": "center",
  "emotion": "neutral",
  "engagement_score": 81.0,
  "stress_score": 24.5,
  "posture_score": 88.0
}
```

#### `ws://localhost:8001/session/metrics`
Full session report (returned on GET /session/metrics):

```json
{
  "avg_confidence_score": 68.4,
  "avg_communication_clarity": 71.2,
  "avg_speech_rate_wpm": 138.0,
  "total_hesitations": 12,
  "confidence_trend": "improving",
  "full_transcript": "...",
  "avg_eye_contact_score": 74.0,
  "avg_engagement_score": 77.5,
  "avg_stress_score": 31.2,
  "avg_posture_score": 82.1,
  "dominant_emotion": "neutral",
  "eye_contact_percentage": 71.0,
  "overall_confidence": 75.0,
  "overall_engagement": 75.7,
  "nervousness_level": "low",
  "coaching_flags": [
    "Strong performance — keep maintaining your communication style"
  ]
}
```

---

## How M3 (Frontend) Connects

```javascript
// Audio WebSocket
const audioWS = new WebSocket("ws://localhost:8001/ws/audio");

// Send audio every 5 seconds from MediaRecorder
mediaRecorder.ondataavailable = async (e) => {
  const buffer = await e.data.arrayBuffer();
  audioWS.send(buffer);  // raw PCM bytes
};

audioWS.onmessage = (e) => {
  const scores = JSON.parse(e.data);
  updateConfidenceMeter(scores.confidence_score);
  updateTranscript(scores.transcript);
};

// Video WebSocket
const videoWS = new WebSocket("ws://localhost:8001/ws/video");

// Send frames from canvas every 100ms
setInterval(() => {
  ctx.drawImage(videoElement, 0, 0);
  canvas.toBlob((blob) => {
    blob.arrayBuffer().then(buf => videoWS.send(buf));
  }, "image/jpeg", 0.7);
}, 100);

videoWS.onmessage = (e) => {
  const scores = JSON.parse(e.data);
  updateEngagementMeter(scores.engagement_score);
  updateEyeContactIndicator(scores.eye_contact);
};
```

## How M1 (AI Brain) Uses M2

M1 polls `/session/metrics` every 30 seconds or listens to WS events to:
- Get `avg_confidence_score` → if < 50, switch to easier questions
- Get `confidence_trend` → if "declining", add encouragement
- Get `full_transcript` → use as candidate's answer text for technical scoring
- Get `nervousness_level` → if "high", agent provides calming prompts

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `M2_PORT` | `8001` | Server port |
| `M2_HOST` | `0.0.0.0` | Server host |
| `WHISPER_MODEL` | `base` | Whisper model size: tiny/base/small/medium |

---

## Scoring Logic Summary

### Confidence Score (0–100)
- **+** Natural pitch variation (not monotone)
- **+** Good vocal energy
- **+** Ideal speech rate (120–160 WPM)
- **−** High pause ratio (>60% silence)
- **−** Many filler words (um, uh, like...)
- **−** Very quiet or very loud voice

### Communication Clarity (0–100)
- **+** Structured pauses (10–35%)
- **+** Consistent energy
- **−** High filler word ratio
- **−** Very short responses

### Engagement Score (0–100)
- **+** Eye contact with camera
- **+** Positive emotions (happy, surprise)
- **+** Good posture
- **−** Looking away
- **−** Sad/angry/fearful expressions

### Stress Score (0–100)
- **+** Fear/anger/sad emotions detected
- **+** Darting eyes (no eye contact)
- **+** Low rolling eye contact average

---

## File Structure

```
m2_sensors/
├── audio/
│   ├── processor.py     # Whisper STT + Librosa analysis + scoring
│   └── recorder.py      # Microphone capture (local mode)
├── video/
│   ├── processor.py     # MediaPipe + DeepFace analysis + scoring
│   └── capture.py       # Webcam capture loop
├── utils/
│   └── aggregator.py    # Session-level metrics + coaching flags
├── api/
│   └── main.py          # FastAPI server (REST + WebSocket)
├── run_local.py         # CLI test with real mic + webcam
├── start_server.py      # Start API server
└── requirements.txt
```
