"""
M2 Video Capture
- Continuously captures webcam frames
- Sends to VideoProcessor
- Broadcasts results over a shared queue for WebSocket
"""

import asyncio
import queue
import threading
import time
import cv2
import numpy as np
from typing import Optional, Callable
from .processor import VideoProcessor, VideoFrameResult


class VideoCapture:
    """
    Reads webcam frames, processes with VideoProcessor.
    Pushes VideoFrameResult to a queue.
    """

    def __init__(self, camera_index: int = 0, fps: int = 15, show_preview: bool = False):
        self.camera_index = camera_index
        self.fps = fps
        self.show_preview = show_preview
        self._running = False
        self._queue: queue.Queue = queue.Queue(maxsize=30)
        self._thread: Optional[threading.Thread] = None
        self._processor = VideoProcessor(emotion_every_n_frames=fps)  # emotion once per second
        self._latest_result: Optional[VideoFrameResult] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Video] Started capture from camera {self.camera_index}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._processor.close()
        print("[Video] Stopped")

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[Video] ERROR: Cannot open camera {self.camera_index}")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FPS, self.fps)
        frame_interval = 1.0 / self.fps

        while self._running:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[Video] Frame read failed — retrying...")
                time.sleep(0.1)
                continue

            try:
                result = self._processor.process_frame(frame)
                self._latest_result = result

                # Push to queue (drop oldest if full)
                try:
                    self._queue.put_nowait(result)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(result)

                if self.show_preview:
                    annotated = self._processor.annotate_frame(frame, result)
                    cv2.imshow("M2 Video Preview", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._running = False
                        break

            except Exception as e:
                print(f"[Video] Processing error: {e}")

            # Throttle to target FPS
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()

    def get_latest(self) -> Optional[VideoFrameResult]:
        """Get most recent result without blocking."""
        return self._latest_result

    def get_result(self, timeout: float = 0.5) -> Optional[VideoFrameResult]:
        """Get next result from queue (blocking)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def get_result_async(self) -> Optional[VideoFrameResult]:
        """Async wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_result, 0.5)


def process_video_frame_bytes(frame_bytes: bytes) -> Optional[VideoFrameResult]:
    """
    Process a single JPEG frame from web client.
    Used when video frames arrive via WebSocket from browser.
    """
    processor = VideoProcessor.__new__(VideoProcessor)
    VideoProcessor.__init__(processor)

    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    result = processor.process_frame(frame)
    processor.close()
    return result


# Singleton processor for web frame processing (stateful — maintains history)
_web_processor: Optional[VideoProcessor] = None

def get_web_processor() -> VideoProcessor:
    global _web_processor
    if _web_processor is None:
        _web_processor = VideoProcessor(emotion_every_n_frames=15)
    return _web_processor

def process_web_frame(frame_bytes: bytes) -> Optional[VideoFrameResult]:
    """Process JPEG bytes from browser WebSocket — uses stateful singleton."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return get_web_processor().process_frame(frame)
