"""OpenCV drawing helpers for the live demo."""

import cv2
import numpy as np

from rtad.classifier.base import AttentionState, Prediction
from rtad.landmarks import indices as idx

STATE_COLORS = {
    AttentionState.ATTENTIVE: (80, 200, 80),
    AttentionState.DISTRACTED: (0, 165, 255),
    AttentionState.DROWSY: (60, 60, 230),
    AttentionState.NO_FACE: (160, 160, 160),
}


def draw_overlay(
    frame: np.ndarray,
    pred: Prediction,
    features: dict[str, float] | None,
    landmarks: np.ndarray | None,
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    color = STATE_COLORS[pred.state]

    if landmarks is not None:
        for i in idx.LEFT_EYE + idx.RIGHT_EYE + idx.MOUTH + [idx.LEFT_IRIS_CENTER, idx.RIGHT_IRIS_CENTER]:
            x, y = landmarks[i, :2].astype(int)
            cv2.circle(frame, (x, y), 1, color, -1)

    cv2.putText(frame, pred.state.value.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"{fps:5.1f} fps  {latency_ms:5.1f} ms", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if features:
        y = 80
        for k in ("ear_mean", "perclos", "mar", "yaw", "pitch", "gaze_x"):
            cv2.putText(frame, f"{k}: {features[k]:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            y += 18

    return frame
