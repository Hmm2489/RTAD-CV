"""Temporal smoothing: EMA over class scores + hysteresis on state switches."""

from rtad.classifier.base import AttentionState, Prediction


class TemporalSmoother:
    def __init__(self, ema_alpha: float = 0.3, min_state_frames: int = 10):
        self.alpha = ema_alpha
        self.min_state_frames = min_state_frames
        self._scores: dict[str, float] = {}
        self._state = AttentionState.ATTENTIVE
        self._candidate = self._state
        self._candidate_frames = 0

    def update(self, pred: Prediction) -> Prediction:
        for k, v in pred.scores.items():
            self._scores[k] = self.alpha * v + (1 - self.alpha) * self._scores.get(k, v)

        if not self._scores:
            return pred

        top = AttentionState(max(self._scores, key=self._scores.get))
        if top == self._state:
            self._candidate_frames = 0
        elif top == self._candidate:
            self._candidate_frames += 1
            if self._candidate_frames >= self.min_state_frames:
                self._state = top
                self._candidate_frames = 0
        else:
            self._candidate = top
            self._candidate_frames = 1

        return Prediction(self._state, dict(self._scores))
