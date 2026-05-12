"""
M2 Local Test Runner
Run this to test M2 with real webcam + microphone input.
Starts both audio and video pipelines and prints live scores.

Usage:
    python run_local.py                    # Default camera + mic
    python run_local.py --camera 1         # Different camera
    python run_local.py --preview          # Show annotated video window
    python run_local.py --audio-only       # Skip video
    python run_local.py --video-only       # Skip audio
"""

import sys
import time
import asyncio
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audio.recorder import AudioRecorder, list_audio_devices
from audio.processor import analyze_audio_chunk
from video.capture import VideoCapture
from utils.aggregator import SessionAggregator


def print_separator():
    print("\n" + "─" * 60)


def audio_loop(recorder: AudioRecorder, aggregator: SessionAggregator, stop_event: threading.Event):
    """Runs audio recording + analysis in a loop."""
    print("[Audio] Starting... speak into your microphone")
    recorder.start()

    while not stop_event.is_set():
        chunk = recorder.get_chunk(timeout=1.0)
        if chunk is None:
            continue

        try:
            result = analyze_audio_chunk(chunk, 16000)
            aggregator.add_audio_result(result)

            print_separator()
            print(f"🎙️  AUDIO ANALYSIS")
            print(f"   Transcript    : {result.transcript or '[no speech detected]'}")
            print(f"   Confidence    : {result.confidence_score:.1f}/100")
            print(f"   Clarity       : {result.communication_clarity_score:.1f}/100")
            print(f"   Hesitations   : {result.hesitation_count}")
            print(f"   Speech Rate   : {result.speech_rate_wpm:.0f} WPM")
            print(f"   Pause Ratio   : {result.pause_ratio:.0%}")
            print(f"   Pitch Var     : {result.pitch_variation:.1f} Hz")
        except Exception as e:
            print(f"[Audio] Error: {e}")

    recorder.stop()


def video_loop(capture: VideoCapture, aggregator: SessionAggregator, stop_event: threading.Event):
    """Runs video capture + analysis in a loop."""
    print("[Video] Starting webcam...")
    capture.start()

    last_print = time.time()
    PRINT_INTERVAL = 2.0  # print every 2 seconds

    while not stop_event.is_set():
        result = capture.get_result(timeout=0.5)
        if result is None:
            continue

        aggregator.add_video_result(result)

        # Only print every few seconds (frames arrive at 15fps)
        if time.time() - last_print >= PRINT_INTERVAL:
            last_print = time.time()
            eye_str = "✅" if result.eye_contact else "❌"
            print_separator()
            print(f"📷  VIDEO ANALYSIS")
            print(f"   Face Detected : {'Yes' if result.face_detected else 'No'}")
            print(f"   Eye Contact   : {eye_str} ({result.eye_contact_score:.0f}% rolling)")
            print(f"   Gaze          : {result.gaze_direction}")
            print(f"   Emotion       : {result.emotion}")
            print(f"   Engagement    : {result.engagement_score:.1f}/100")
            print(f"   Stress        : {result.stress_score:.1f}/100")
            print(f"   Posture       : {result.posture_score:.1f}/100")

    capture.stop()


def print_final_report(aggregator: SessionAggregator):
    """Print the final session report."""
    metrics = aggregator.get_session_metrics()
    print("\n" + "═" * 60)
    print("📊  FINAL SESSION REPORT")
    print("═" * 60)
    print(f"\n⏱️  Duration: {metrics.session_duration_seconds:.0f}s")
    print(f"🎙️  Audio chunks: {metrics.audio_chunk_count}")
    print(f"📷  Video frames: {metrics.video_frame_count}")

    print("\n── Audio ─────────────────────────────────────────────────")
    print(f"   Avg Confidence      : {metrics.avg_confidence_score:.1f}/100")
    print(f"   Avg Clarity         : {metrics.avg_communication_clarity:.1f}/100")
    print(f"   Avg Speech Rate     : {metrics.avg_speech_rate_wpm:.0f} WPM")
    print(f"   Total Hesitations   : {metrics.total_hesitations}")
    print(f"   Confidence Trend    : {metrics.confidence_trend}")
    print(f"\n   Full Transcript:\n   {metrics.full_transcript or '[no transcript]'}")

    print("\n── Video ─────────────────────────────────────────────────")
    print(f"   Avg Eye Contact     : {metrics.avg_eye_contact_score:.1f}/100")
    print(f"   Eye Contact %       : {metrics.eye_contact_percentage:.0f}%")
    print(f"   Avg Engagement      : {metrics.avg_engagement_score:.1f}/100")
    print(f"   Avg Stress          : {metrics.avg_stress_score:.1f}/100")
    print(f"   Avg Posture         : {metrics.avg_posture_score:.1f}/100")
    print(f"   Dominant Emotion    : {metrics.dominant_emotion}")

    print("\n── Overall ───────────────────────────────────────────────")
    print(f"   Overall Confidence  : {metrics.overall_confidence:.1f}/100")
    print(f"   Overall Engagement  : {metrics.overall_engagement:.1f}/100")
    print(f"   Nervousness Level   : {metrics.nervousness_level.upper()}")

    print("\n── Coaching Flags ────────────────────────────────────────")
    for i, flag in enumerate(metrics.coaching_flags, 1):
        print(f"   {i}. {flag}")

    print("\n" + "═" * 60)


def main():
    parser = argparse.ArgumentParser(description="M2 Sensors Local Test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="Show annotated webcam window")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--duration", type=int, default=60, help="Run for N seconds (0=forever)")
    args = parser.parse_args()

    list_audio_devices()

    aggregator = SessionAggregator()
    stop_event = threading.Event()
    threads = []

    # Start audio thread
    if not args.video_only:
        recorder = AudioRecorder(chunk_duration=5)
        t_audio = threading.Thread(
            target=audio_loop, args=(recorder, aggregator, stop_event), daemon=True
        )
        threads.append(t_audio)
        t_audio.start()

    # Start video thread
    if not args.audio_only:
        capture = VideoCapture(camera_index=args.camera, fps=15, show_preview=args.preview)
        t_video = threading.Thread(
            target=video_loop, args=(capture, aggregator, stop_event), daemon=True
        )
        threads.append(t_video)
        t_video.start()

    print(f"\n🚀 M2 Sensors running — press Ctrl+C to stop")
    if args.duration > 0:
        print(f"   Auto-stop in {args.duration}s\n")

    try:
        if args.duration > 0:
            time.sleep(args.duration)
            stop_event.set()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        stop_event.set()

    for t in threads:
        t.join(timeout=5)

    print_final_report(aggregator)


if __name__ == "__main__":
    main()
