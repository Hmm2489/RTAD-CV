from rtad.classifier import AttentionState, Prediction, RuleBasedClassifier, TemporalSmoother

THRESHOLDS = {"perclos_drowsy": 0.3, "mar_yawn": 0.6, "yaw_deg": 25, "pitch_deg": 20, "gaze_offset": 0.25}
NEUTRAL = {"perclos": 0.0, "mar": 0.2, "yaw": 0.0, "pitch": 0.0, "gaze_x": 0.0}


def test_rules_attentive():
    assert RuleBasedClassifier(THRESHOLDS).predict(NEUTRAL).state == AttentionState.ATTENTIVE


def test_rules_distracted():
    assert RuleBasedClassifier(THRESHOLDS).predict({**NEUTRAL, "yaw": 40}).state == AttentionState.DISTRACTED


def test_rules_drowsy():
    assert RuleBasedClassifier(THRESHOLDS).predict({**NEUTRAL, "perclos": 0.5}).state == AttentionState.DROWSY


def test_smoother_requires_sustained_change():
    smoother = TemporalSmoother(ema_alpha=1.0, min_state_frames=3)
    drowsy = Prediction(AttentionState.DROWSY, {"attentive": 0.0, "distracted": 0.0, "drowsy": 1.0})
    states = [smoother.update(drowsy).state for _ in range(4)]
    assert states[0] == AttentionState.ATTENTIVE
    assert states[-1] == AttentionState.DROWSY
