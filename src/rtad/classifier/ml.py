"""Lightweight learned classifier (scikit-learn) over per-frame / windowed features."""

from pathlib import Path

import joblib
import numpy as np

from rtad.classifier.base import AttentionState, Prediction
from rtad.features.extractor import FEATURE_NAMES


class MLClassifier:
    def __init__(self, model_path: str | Path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle.get("feature_names", FEATURE_NAMES)

    def predict(self, f: dict[str, float]) -> Prediction:
        x = np.array([[f[n] for n in self.feature_names]], dtype=np.float32)
        proba = self.model.predict_proba(x)[0]
        scores = {str(c): float(p) for c, p in zip(self.model.classes_, proba)}
        best = max(scores, key=scores.get)
        return Prediction(AttentionState(best), scores)
