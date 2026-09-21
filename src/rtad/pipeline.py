"""End-to-end per-frame pipeline: frame -> landmarks -> features -> smoothed prediction."""

from dataclasses import dataclass

import numpy as np

from rtad.classifier import AttentionState, MLClassifier, Prediction, RuleBasedClassifier, TemporalSmoother
from rtad.features import FeatureExtractor
from rtad.landmarks import FaceMeshDetector
from rtad.utils import LatencyTracker


@dataclass
class FrameResult:
    prediction: Prediction
    features: dict[str, float] | None
    landmarks: np.ndarray | None


class AttentionPipeline:
    def __init__(self, cfg: dict):
        self.detector = FaceMeshDetector(**cfg["face_mesh"])
        window_frames = int(cfg["features"]["window_seconds"] * cfg["camera"]["fps"])
        self.extractor = FeatureExtractor(cfg["thresholds"]["ear_closed"], window_frames)

        if cfg["classifier"]["type"] == "ml":
            self.classifier = MLClassifier(cfg["classifier"]["model_path"])
        else:
            self.classifier = RuleBasedClassifier(cfg["thresholds"])

        self.smoother = TemporalSmoother(**cfg["smoothing"])
        self.latency = LatencyTracker()

    def process(self, frame: np.ndarray) -> FrameResult:
        with self.latency.measure("total"):
            with self.latency.measure("landmarks"):
                landmarks = self.detector.detect(frame)

            if landmarks is None:
                return FrameResult(Prediction(AttentionState.NO_FACE), None, None)

            with self.latency.measure("features"):
                features = self.extractor.extract(landmarks, frame.shape[:2])

            with self.latency.measure("classify"):
                pred = self.smoother.update(self.classifier.predict(features))

        return FrameResult(pred, features, landmarks)

    def close(self) -> None:
        self.detector.close()
