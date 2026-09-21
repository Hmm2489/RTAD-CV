"""Head pose (yaw, pitch, roll) via solvePnP against a generic 3D face model."""

import cv2
import numpy as np

# Generic 3D model points (mm) matching landmarks.indices.HEAD_POSE order
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -63.6, -12.5),  # chin
        (-43.3, 32.7, -26.0),  # left eye outer corner
        (43.3, 32.7, -26.0),  # right eye outer corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)


def estimate_head_pose(image_points: np.ndarray, frame_size: tuple[int, int]) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees. image_points: (6, 2) pixel coords."""
    h, w = frame_size
    focal = w
    camera_matrix = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points.astype(np.float64), camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    # TODO: verify sign conventions / wrap-around against recorded data
    return float(yaw), float(pitch), float(roll)
