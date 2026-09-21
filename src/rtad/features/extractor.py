"""Turns a landmark array into a fixed-length feature vector per frame."""

import numpy as np

from rtad.features.eye import BlinkTracker, eye_aspect_ratio
from rtad.features.gaze import iris_offset
from rtad.features.head_pose import estimate_head_pose
from rtad.features.mouth import mouth_aspect_ratio
from rtad.landmarks import indices as idx

FEATURE_NAMES = [
    "ear_left",
    "ear_right",
    "ear_mean",
    "perclos",
    "mar",
    "yaw",
    "pitch",
    "roll",
    "gaze_x",
    "gaze_y",
]


class FeatureExtractor:
    def __init__(self, ear_closed: float = 0.21, window_frames: int = 90):
        self.blinks = BlinkTracker(ear_closed, window_frames)

    def extract(self, landmarks: np.ndarray, frame_size: tuple[int, int]) -> dict[str, float]:
        left_eye = landmarks[idx.LEFT_EYE]
        right_eye = landmarks[idx.RIGHT_EYE]

        ear_l = eye_aspect_ratio(left_eye)
        ear_r = eye_aspect_ratio(right_eye)
        ear = (ear_l + ear_r) / 2
        self.blinks.update(ear)

        mar = mouth_aspect_ratio(landmarks[idx.MOUTH])
        yaw, pitch, roll = estimate_head_pose(landmarks[idx.HEAD_POSE, :2], frame_size)

        gx_l, gy_l = iris_offset(left_eye, landmarks[idx.LEFT_IRIS_CENTER])
        gx_r, gy_r = iris_offset(right_eye, landmarks[idx.RIGHT_IRIS_CENTER])

        return {
            "ear_left": ear_l,
            "ear_right": ear_r,
            "ear_mean": ear,
            "perclos": self.blinks.perclos,
            "mar": mar,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "gaze_x": (gx_l + gx_r) / 2,
            "gaze_y": (gy_l + gy_r) / 2,
        }

    @staticmethod
    def to_vector(features: dict[str, float]) -> np.ndarray:
        return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
