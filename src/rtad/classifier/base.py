"""Shared classifier types."""

from dataclasses import dataclass, field
from enum import Enum


class AttentionState(str, Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    NO_FACE = "no_face"


@dataclass
class Prediction:
    state: AttentionState
    scores: dict[str, float] = field(default_factory=dict)  # per-class probability / score
