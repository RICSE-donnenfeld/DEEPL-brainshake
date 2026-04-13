"""Random forest signal classifier helpers."""

from .model import RandomForestSignalClassifier


def evaluate_dataset(*args, **kwargs):
    from .evaluate import evaluate_dataset as _evaluate_dataset

    return _evaluate_dataset(*args, **kwargs)


__all__ = ["RandomForestSignalClassifier", "evaluate_dataset"]
