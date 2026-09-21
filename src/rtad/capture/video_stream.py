"""Threaded webcam reader so capture never blocks inference."""

import threading
import time

import cv2


class VideoStream:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")

        self._frame = None
        self._timestamp = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self) -> "VideoStream":
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame
                self._timestamp = time.perf_counter()

    def read(self):
        """Return (frame, capture_timestamp) for the most recent frame, or (None, 0)."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._timestamp

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
