from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence
import numpy as np
from sklearn.svm import SVC
from ...data_handling.extract_features import FeatureDict

@dataclass
class SVMSeizureClassifier:
    """SVM classifier optimized for statistical EEG features."""
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    class_weight: str | dict | None = "balanced"
    # Order must match the one used in the feature extraction module
    feature_order: Sequence[str] = field(
        default_factory=lambda: [
            "mean", "std", "min", "max", "range", "peak_to_peak", "std_range_ratio", "range_std_sum"
        ]
    )
    classifier: SVC = field(init=False)

    def __post_init__(self) -> None:
        self.classifier = SVC(
            kernel=self.kernel, 
            C=self.C, 
            gamma=self.gamma,
            class_weight=self.class_weight, 
            probability=True
        )

    def _vectorize(self, features: FeatureDict) -> np.ndarray:
        """Convert feature dictionary to a numerical vector."""
        return np.array([features.get(k, 0.0) for k in self.feature_order], dtype=np.float32)

    def fit(self, features: Iterable[FeatureDict], labels: Iterable[int]) -> None:
        """Train the SVM model."""
        matrix = np.vstack([self._vectorize(f) for f in features])
        self.classifier.fit(matrix, list(labels))

    def predict(self, features: Iterable[FeatureDict]) -> List[int]:
        """Make predictions on new signal windows."""
        matrix = np.vstack([self._vectorize(f) for f in features])
        return list(self.classifier.predict(matrix))