"""
M2 Audio Recorder
- Captures microphone audio in chunks
- Sends to processor for real-time analysis
- Used in local/CLI mode; in web mode, audio arrives via WebSocket
"""

import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Callable, Optional

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds per analysis chunk
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION


class AudioRecorder:
    """
    Continuously records from microphone.
    Fires callback with audio bytes every CHUNK_DURATION seconds.
    """

    def __init__(self, chunk_duration: int = CHUNK_DURATION, sample_rate: int = SAMPLE_RATE):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_samples = sample_rate * chunk_duration
        self._buffer = []
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio frame."""
        if status:
            print(f"[Recorder] Status: {status}")
        self._buffer.extend(indata[:, 0].tolist())

        # Emit chunk when buffer is full
        while len(self._buffer) >= self.chunk_samples:
            chunk = np.array(self._buffer[:self.chunk_samples])
            self._buffer = self._buffer[self.chunk_samples:]
            # Convert to int16 bytes
            audio_bytes = (chunk * 32768).astype(np.int16).tobytes()
            self._queue.put(audio_bytes)

    def start(self):
        """Start recording."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        )
        self._stream.start()
        print(f"[Recorder] Started — {self.chunk_duration}s chunks at {self.sample_rate}Hz")

    def stop(self):
        """Stop recording."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        print("[Recorder] Stopped")

    def get_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """Get next audio chunk (blocking)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def get_chunk_async(self, timeout: float = 6.0) -> Optional[bytes]:
        """Async-compatible chunk getter."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_chunk, timeout)


def list_audio_devices():
    """Print available audio input devices."""
    devices = sd.query_devices()
    print("\n[Recorder] Available audio devices:")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"  [{i}] {d['name']} (inputs: {d['max_input_channels']})")
    print(f"  Default input: {sd.default.device[0]}\n")
