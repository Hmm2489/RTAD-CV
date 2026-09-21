"""Threshold-based baseline. Useful before any labelled data exists."""

from rtad.classifier.base import AttentionState, Prediction


class RuleBasedClassifier:
    def __init__(self, thresholds: dict):
        self.t = thresholds

    def predict(self, f: dict[str, float]) -> Prediction:
        if f["perclos"] > self.t["perclos_drowsy"] or f["mar"] > self.t["mar_yawn"]:
            state = AttentionState.DROWSY
        elif (
            abs(f["yaw"]) > self.t["yaw_deg"]
            or abs(f["pitch"]) > self.t["pitch_deg"]
            or abs(f["gaze_x"]) > self.t["gaze_offset"]
        ):
            state = AttentionState.DISTRACTED
        else:
            state = AttentionState.ATTENTIVE
        # TODO: produce soft scores instead of one-hot so smoothing has something to work with
        return Prediction(state, {s.value: float(s == state) for s in AttentionState if s != AttentionState.NO_FACE})
