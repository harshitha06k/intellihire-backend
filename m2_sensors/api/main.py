"""
M2 Sensors — FastAPI Server
Exposes:
  POST /audio/analyze        — Upload audio chunk → scores
  POST /video/analyze        — Upload JPEG frame → scores
  WS   /ws/audio             — Stream audio bytes → live scores
  WS   /ws/video             — Stream JPEG frames → live scores
  GET  /session/metrics      — Current session aggregated scores
  POST /session/reset        — Reset session
  GET  /health               — Health check
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from audio.processor import analyze_audio_chunk, AudioAnalysisResult
from video.capture import process_web_frame, get_web_processor
from utils.aggregator import SessionAggregator

# ─────────────────────── App Setup ────────────────────────────────

session = SessionAggregator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n✅ M2 Sensors Server starting...")
    print("   Audio: Whisper STT + Librosa")
    print("   WebSocket endpoints: /ws/audio  /ws/video")
    yield
    print("\n🛑 M2 Sensors Server shutting down")

app = FastAPI(
    title="M2 Sensors API",
    description="Audio + Video Intelligence for Mock Interview Agent",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow M3 frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────── Health ────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "module": "M2-Sensors", "timestamp": time.time()}


# ─────────────────────── REST Audio ────────────────────────────────

@app.post("/audio/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    sample_rate: int = 16000
):
    """
    Upload a WAV/PCM audio chunk.
    Returns: transcript, confidence score, clarity score, hesitations, speech rate.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) < 100:
        raise HTTPException(400, "Audio chunk too small")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, analyze_audio_chunk, audio_bytes, sample_rate
    )

    session.add_audio_result(result)

    return {
        "transcript": result.transcript,
        "confidence_score": result.confidence_score,
        "communication_clarity_score": result.communication_clarity_score,
        "hesitation_count": result.hesitation_count,
        "speech_rate_wpm": result.speech_rate_wpm,
        "pitch_mean": result.pitch_mean,
        "pitch_variation": result.pitch_variation,
        "pause_ratio": result.pause_ratio,
        "duration_seconds": result.duration_seconds,
        "timestamp": result.timestamp
    }


# ─────────────────────── REST Video ────────────────────────────────

@app.post("/video/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a single JPEG frame.
    Returns: eye_contact, emotion, engagement, stress, posture scores.
    """
    frame_bytes = await file.read()
    if len(frame_bytes) < 100:
        raise HTTPException(400, "Frame too small")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_web_frame, frame_bytes)

    if result is None:
        raise HTTPException(422, "Could not decode frame")

    session.add_video_result(result)

    return {
        "face_detected": result.face_detected,
        "eye_contact": result.eye_contact,
        "eye_contact_score": result.eye_contact_score,
        "gaze_direction": result.gaze_direction,
        "emotion": result.emotion,
        "emotion_scores": result.emotion_scores,
        "engagement_score": result.engagement_score,
        "stress_score": result.stress_score,
        "posture_score": result.posture_score,
        "timestamp": result.timestamp
    }


# ─────────────────────── Session ───────────────────────────────────

@app.get("/session/metrics")
async def get_session_metrics():
    """Get aggregated session metrics (full interview stats)."""
    return session.to_dict()


@app.post("/session/reset")
async def reset_session():
    """Reset session for new interview."""
    session.reset()
    return {"status": "reset", "timestamp": time.time()}


# ─────────────────────── WebSocket Audio ───────────────────────────

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Client sends: raw PCM int16 audio bytes (16kHz, mono)
                  every ~5 seconds
    Server sends: JSON with scores after each chunk
    
    Protocol:
      Client → binary audio bytes
      Server → JSON result
    """
    await websocket.accept()
    print("[WS/audio] Client connected")

    try:
        while True:
            # Receive audio bytes from browser
            audio_bytes = await websocket.receive_bytes()

            if len(audio_bytes) < 200:
                await websocket.send_json({"error": "chunk_too_small"})
                continue

            # Analyze in thread pool (CPU-intensive)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, analyze_audio_chunk, audio_bytes, 16000
            )

            session.add_audio_result(result)

            # Send result back
            await websocket.send_json({
                "type": "audio_result",
                "transcript": result.transcript,
                "confidence_score": result.confidence_score,
                "communication_clarity_score": result.communication_clarity_score,
                "hesitation_count": result.hesitation_count,
                "speech_rate_wpm": result.speech_rate_wpm,
                "pause_ratio": result.pause_ratio,
                "duration_seconds": result.duration_seconds,
                "timestamp": result.timestamp
            })

    except WebSocketDisconnect:
        print("[WS/audio] Client disconnected")
    except Exception as e:
        print(f"[WS/audio] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ─────────────────────── WebSocket Video ───────────────────────────

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    
    Client sends: JPEG-encoded frames (every ~100ms)
    Server sends: JSON with video scores
    
    Browser captures frames via canvas.toBlob("image/jpeg")
    """
    await websocket.accept()
    print("[WS/video] Client connected")

    processor = get_web_processor()
    frame_count = 0
    SEND_EVERY = 3  # Only send results every N frames (reduce bandwidth)

    try:
        while True:
            data = await websocket.receive_bytes()
            frame_count += 1

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_web_frame, data)

            if result is None:
                continue

            session.add_video_result(result)

            # Send every N frames to avoid flooding
            if frame_count % SEND_EVERY == 0:
                await websocket.send_json({
                    "type": "video_result",
                    "face_detected": result.face_detected,
                    "eye_contact": result.eye_contact,
                    "eye_contact_score": result.eye_contact_score,
                    "gaze_direction": result.gaze_direction,
                    "emotion": result.emotion,
                    "emotion_scores": result.emotion_scores,
                    "engagement_score": result.engagement_score,
                    "stress_score": result.stress_score,
                    "posture_score": result.posture_score,
                    "timestamp": result.timestamp
                })

    except WebSocketDisconnect:
        print("[WS/video] Client disconnected")
    except Exception as e:
        print(f"[WS/video] Error: {e}")


# ─────────────────────── Combined WS ───────────────────────────────

@app.websocket("/ws/combined")
async def ws_combined(websocket: WebSocket):
    """
    Single combined WebSocket for both audio and video.
    
    Client sends JSON: { "type": "audio"|"video", "data": <base64-encoded bytes> }
    Server sends JSON results + periodic session metrics
    """
    await websocket.accept()
    print("[WS/combined] Client connected")
    last_metrics_push = time.time()

    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            msg_type = payload.get("type")
            data_b64 = payload.get("data", "")

            import base64
            raw_bytes = base64.b64decode(data_b64)

            loop = asyncio.get_event_loop()

            if msg_type == "audio":
                result = await loop.run_in_executor(None, analyze_audio_chunk, raw_bytes, 16000)
                session.add_audio_result(result)
                await websocket.send_json({
                    "type": "audio_result",
                    "transcript": result.transcript,
                    "confidence_score": result.confidence_score,
                    "communication_clarity_score": result.communication_clarity_score,
                    "hesitation_count": result.hesitation_count,
                    "speech_rate_wpm": result.speech_rate_wpm,
                })

            elif msg_type == "video":
                result = await loop.run_in_executor(None, process_web_frame, raw_bytes)
                if result:
                    session.add_video_result(result)
                    await websocket.send_json({
                        "type": "video_result",
                        "eye_contact": result.eye_contact,
                        "eye_contact_score": result.eye_contact_score,
                        "emotion": result.emotion,
                        "engagement_score": result.engagement_score,
                        "stress_score": result.stress_score,
                        "posture_score": result.posture_score,
                    })

            # Push session metrics every 10 seconds
            if time.time() - last_metrics_push > 10:
                metrics = session.get_session_metrics()
                await websocket.send_json({
                    "type": "session_metrics",
                    **session.to_dict()
                })
                last_metrics_push = time.time()

    except WebSocketDisconnect:
        print("[WS/combined] Client disconnected")
    except Exception as e:
        print(f"[WS/combined] Error: {e}")


# ─────────────────────── Entry Point ───────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("M2_PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
