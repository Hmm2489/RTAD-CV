import numpy as np

from rtad.features.eye import BlinkTracker, eye_aspect_ratio
from rtad.features.gaze import iris_offset


def _eye(open_height: float) -> np.ndarray:
    # [outer, top1, top2, inner, bottom2, bottom1]
    return np.array(
        [[0, 0], [3, -open_height], [7, -open_height], [10, 0], [7, open_height], [3, open_height]],
        dtype=np.float32,
    )


def test_ear_open_greater_than_closed():
    assert eye_aspect_ratio(_eye(3.0)) > eye_aspect_ratio(_eye(0.5))


def test_ear_closed_near_zero():
    assert eye_aspect_ratio(_eye(0.0)) < 1e-3


def test_iris_centered_has_zero_offset():
    dx, dy = iris_offset(_eye(3.0), np.array([5.0, 0.0]))
    assert abs(dx) < 1e-6 and abs(dy) < 1e-6


def test_perclos():
    tracker = BlinkTracker(ear_threshold=0.2, window_frames=10)
    for ear in [0.3] * 5 + [0.1] * 5:
        tracker.update(ear)
    assert tracker.perclos == 0.5
