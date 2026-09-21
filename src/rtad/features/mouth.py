"""Mouth-based features: mouth aspect ratio (MAR) for yawn detection."""

import numpy as np


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR for 6 mouth points ordered [left, top1, top2, right, bottom2, bottom1]."""
    p1, p2, p3, p4, p5, p6 = pts[:, :2]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2.0 * horizontal + 1e-6))
