"""Wrapper around MediaPipe FaceLandmarker (Tasks API) returning pixel-space landmarks.

MediaPipe >= 1.0 removed the legacy `mp.solutions.face_mesh`; FaceLandmarker runs the same
FaceMesh model and returns 478 landmarks (468 mesh + 10 iris). Get the model with
`python scripts/download_model.py`.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


class FaceMeshDetector:
    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> np.ndarray | None:
        """Return a (478, 3) array of landmarks (x, y in pixels, z relative) or None if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps
        if timestamp_ms is None:
            timestamp_ms = int(time.perf_counter() * 1000)
        timestamp_ms = max(timestamp_ms, self._last_ts_ms + 1)
        self._last_ts_ms = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        return np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()
