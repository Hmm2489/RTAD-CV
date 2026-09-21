"""Per-stage latency measurement for the evaluation report."""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class LatencyTracker:
    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        yield
        self.samples[stage].append((time.perf_counter() - start) * 1000)

    def last(self, stage: str) -> float:
        return self.samples[stage][-1] if self.samples[stage] else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        """Return {stage: {mean, p50, p95, max}} in milliseconds."""
        return {
            stage: {
                "mean": float(np.mean(v)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "max": float(np.max(v)),
            }
            for stage, v in self.samples.items()
            if v
        }
