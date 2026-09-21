"""Gaze direction from iris position relative to the eye corners."""

import numpy as np


def iris_offset(eye_pts: np.ndarray, iris_center: np.ndarray) -> tuple[float, float]:
    """Return (horizontal, vertical) iris offset in [-1, 1]-ish, 0 = looking straight.

    eye_pts: 6 eye points in EAR order [outer, top1, top2, inner, bottom2, bottom1].
    """
    outer, inner = eye_pts[0, :2], eye_pts[3, :2]
    top = (eye_pts[1, :2] + eye_pts[2, :2]) / 2
    bottom = (eye_pts[4, :2] + eye_pts[5, :2]) / 2
    center = (outer + inner) / 2

    half_width = np.linalg.norm(inner - outer) / 2 + 1e-6
    half_height = np.linalg.norm(bottom - top) / 2 + 1e-6

    dx = (iris_center[0] - center[0]) / half_width
    dy = (iris_center[1] - center[1]) / half_height
    return float(dx), float(dy)

# TODO: combine iris offset with head pose for a head-compensated gaze estimate
