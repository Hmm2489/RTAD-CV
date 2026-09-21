"""Eye-based features: eye aspect ratio (EAR), blinks, PERCLOS."""

import numpy as np


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR (Soukupová & Čech, 2016) for 6 eye points ordered [p1..p6]."""
    p1, p2, p3, p4, p5, p6 = pts[:, :2]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2.0 * horizontal + 1e-6))


class BlinkTracker:
    """Tracks blinks and PERCLOS over a rolling window of frames."""

    def __init__(self, ear_threshold: float, window_frames: int):
        self.ear_threshold = ear_threshold
        self.window_frames = window_frames
        self._closed_history: list[bool] = []
        self._was_closed = False
        self.blink_count = 0

    def update(self, ear: float) -> None:
        closed = ear < self.ear_threshold
        if self._was_closed and not closed:
            self.blink_count += 1
        self._was_closed = closed
        self._closed_history.append(closed)
        if len(self._closed_history) > self.window_frames:
            self._closed_history.pop(0)

    @property
    def perclos(self) -> float:
        """Fraction of recent frames with eyes closed."""
        if not self._closed_history:
            return 0.0
        return sum(self._closed_history) / len(self._closed_history)

    # TODO: blink rate (blinks/min) and mean blink duration
