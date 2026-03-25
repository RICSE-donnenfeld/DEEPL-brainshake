"""
Lightweight ThresholdClassifier
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ...data_handling.extract_features import FeatureDict


@dataclass
class ThresholdClassifier:
    """Basic rule-based classifier using std and range thresholds."""

    std_threshold: float = 70.0
    range_threshold: float = 220.0
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    require_both: bool = False

    def predict(self, features: FeatureDict) -> int:
        std_val = features.get("std", 0.0)
        range_val = features.get("range", 0.0)
        min_val = features.get("min", 0.0)
        max_val = features.get("max", 0.0)
        std_pass = std_val >= self.std_threshold
        range_pass = range_val >= self.range_threshold
        min_pass = self.min_threshold is not None and min_val <= self.min_threshold
        max_pass = self.max_threshold is not None and max_val >= self.max_threshold

        primary_pass = (
            std_pass and range_pass if self.require_both else std_pass or range_pass
        )
        return int(primary_pass or min_pass or max_pass)

    def predict_batch(self, features: Sequence[FeatureDict]) -> List[int]:
        return [self.predict(f) for f in features]

    def evaluate(
        self,
        features: Sequence[FeatureDict],
        labels: Iterable[int],
    ) -> float:
        preds = self.predict_batch(features)
        label_list = list(labels)
        if len(preds) != len(label_list):
            raise ValueError("Features and labels must be the same length")
        correct = sum(1 for p, l in zip(preds, label_list) if p == l)
        return correct / len(label_list)

    def describe(self) -> str:
        condition = "AND" if self.require_both else "OR"
        parts = [
            f"std>={self.std_threshold}",
            f"range>={self.range_threshold}",
        ]
        if self.min_threshold is not None:
            parts.append(f"min<={self.min_threshold}")
        if self.max_threshold is not None:
            parts.append(f"max>={self.max_threshold}")
        return f"ThresholdClassifier({', '.join(parts)}, operator={condition})"
