"""Random Forest classifier that operates on feature dictionaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ...data_handling.extract_features import FeatureDict


@dataclass
class RandomForestSignalClassifier:
    """Wrapper around sklearn RandomForest for EEG feature dictionaries."""

    n_estimators: int = 100
    max_depth: int | None = None
    random_state: int = 42
    class_weight: str | dict | None = "balanced"
    feature_order: Sequence[str] = field(
        default_factory=lambda: [
            "mean",
            "std",
            "min",
            "max",
            "range",
            "peak_to_peak",
            "std_range_ratio",
            "range_std_sum",
        ]
    )
    classifier: RandomForestClassifier = field(init=False)

    def __post_init__(self) -> None:
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

    def _vectorize(self, features: FeatureDict) -> np.ndarray:
        return np.array(
            [features.get(key, 0.0) for key in self.feature_order],
            dtype=np.float32,
        )

    def _prepare_matrix(self, features: Iterable[FeatureDict]) -> np.ndarray:
        return np.vstack([self._vectorize(f) for f in features])

    def fit(self, features: Iterable[FeatureDict], labels: Iterable[int]) -> None:
        matrix = self._prepare_matrix(features)
        self.classifier.fit(matrix, list(labels))

    def predict(self, features: Iterable[FeatureDict]) -> List[int]:
        matrix = self._prepare_matrix(features)
        return list(self.classifier.predict(matrix))

    def predict_proba(self, features: Iterable[FeatureDict]) -> np.ndarray:
        matrix = self._prepare_matrix(features)
        return self.classifier.predict_proba(matrix)

    def describe(self) -> str:
        importances = self.classifier.feature_importances_
        pairs = ", ".join(
            f"{name}={importance:.3f}"
            for name, importance in zip(self.feature_order, importances)
        )
        return (
            f"RandomForest(n_est={self.n_estimators}, max_depth={self.max_depth}, "
            f"features=[{pairs}])"
        )
