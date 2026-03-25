"""
Lightweight ThresholdClassifier
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..data_handling.extract_features import FeatureDict


@dataclass
class ThresholdClassifier:
    """Basic rule-based classifier using std and range thresholds."""

    std_threshold: float = 70.0
    range_threshold: float = 220.0
    require_both: bool = False

    def predict(self, features: FeatureDict) -> int:
        std_val = features.get("std", 0.0)
        range_val = features.get("range", 0.0)
        std_pass = std_val >= self.std_threshold
        range_pass = range_val >= self.range_threshold

        if self.require_both:
            return int(std_pass and range_pass)
        return int(std_pass or range_pass)

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
        return (
            f"ThresholdClassifier(std>={self.std_threshold}, range>={self.range_threshold}, "
            f"operator={condition})"
        )
