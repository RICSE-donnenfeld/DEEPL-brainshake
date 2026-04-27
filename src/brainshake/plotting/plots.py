"""Utilities to chart benchmark outputs from the models.

Supports both legacy benchmark JSONs (top-level fold keys like "accuracy") and
newer benchmark JSONs that include nested confusion/metric blocks, e.g.
"folds[].metrics.recall" and "aggregate.metrics.balanced_accuracy".
"""

from __future__ import annotations

import glob
import json
import argparse
import math
from pathlib import Path
from typing import Iterable, Mapping, cast, Any, Optional, List

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = REPO_ROOT / "out" / "plots"


def _load_benchmark(path: Path) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compute_from_confusion(
    confusion: Mapping[str, object], metric: str
) -> Optional[float]:
    tp = _as_float(confusion.get("tp"))
    fp = _as_float(confusion.get("fp"))
    tn = _as_float(confusion.get("tn"))
    fn = _as_float(confusion.get("fn"))
    if tp is None or fp is None or tn is None or fn is None:
        return None

    total = tp + fp + tn + fn
    if metric == "accuracy":
        return (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    if metric == "precision":
        return precision
    if metric == "recall":
        return recall
    if metric == "specificity":
        return specificity
    if metric == "f1":
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if metric == "balanced_accuracy":
        return 0.5 * (recall + specificity)
    return None


def _extract_fold_metric(fold: Mapping[str, object], metric: str) -> Optional[float]:
    direct = _as_float(fold.get(metric))
    if direct is not None:
        return direct

    metrics = fold.get("metrics")
    if isinstance(metrics, Mapping):
        nested = _as_float(cast(Mapping[str, object], metrics).get(metric))
        if nested is not None:
            return nested

    confusion = fold.get("confusion")
    if isinstance(confusion, Mapping):
        computed = _compute_from_confusion(cast(Mapping[str, object], confusion), metric)
        if computed is not None:
            return float(computed)

    return None


def _extract_aggregate_metric(data: Mapping[str, object], metric: str) -> Optional[float]:
    # Legacy
    if metric == "accuracy":
        avg = _as_float(data.get("average_accuracy"))
        if avg is not None:
            return avg

    aggregate = data.get("aggregate")
    if isinstance(aggregate, Mapping):
        agg = cast(Mapping[str, object], aggregate)
        agg_metrics = agg.get("metrics")
        if isinstance(agg_metrics, Mapping):
            nested = _as_float(cast(Mapping[str, object], agg_metrics).get(metric))
            if nested is not None:
                return nested
        agg_conf = agg.get("confusion")
        if isinstance(agg_conf, Mapping):
            computed = _compute_from_confusion(cast(Mapping[str, object], agg_conf), metric)
            if computed is not None:
                return float(computed)

    # Fallback: mean over folds (ignoring missing values)
    folds = cast(List[Mapping[str, object]], data.get("folds") or [])
    values: list[float] = []
    for fold in folds:
        val = _extract_fold_metric(fold, metric)
        if val is not None:
            values.append(float(val))
    if values:
        return float(sum(values) / len(values))
    return None


def _plot_fold_metric(
    benchmarks: Mapping[str, Mapping[str, object]],
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    any_series = False
    for label, data in benchmarks.items():
        folds = cast(List[Mapping[str, object]], data.get("folds") or [])
        if not folds:
            continue
        x_values: list[int] = []
        y_values: list[float] = []
        for idx, fold in enumerate(folds):
            fold_idx = fold.get("fold")
            if isinstance(fold_idx, (int, float)):
                x_values.append(int(fold_idx))
            else:
                x_values.append(idx)

            metric_value = _extract_fold_metric(fold, metric)
            y_values.append(float(metric_value) if metric_value is not None else float("nan"))

        if all(math.isnan(value) for value in y_values):
            continue

        any_series = True
        ax.plot(x_values, y_values, marker="o", label=label)

    if not any_series:
        plt.close(fig)
        return
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} per Fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_average_metric(
    benchmarks: Mapping[str, Mapping[str, object]],
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    labels = []
    values = []
    for label, data in benchmarks.items():
        avg = _extract_aggregate_metric(data, metric)
        if avg is not None:
            labels.append(label)
            values.append(float(avg))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Average {ylabel} by Benchmark")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_benchmark_plots(
    benchmark_files: Iterable[Path], *, output_dir: Optional[Path] = None
) -> Mapping[str, Path]:
    output_dir = output_dir or DEFAULT_PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks: dict[str, Mapping[str, object]] = {}
    for path in benchmark_files:
        if not path.exists():
            continue
        benchmarks[path.stem] = _load_benchmark(path)

    results: dict[str, Path] = {}
    if benchmarks:
        accuracy_path = output_dir / "accuracy_by_fold.png"
        _plot_fold_metric(benchmarks, "accuracy", accuracy_path, "Accuracy")
        results["accuracy_by_fold"] = accuracy_path

        loss_path = output_dir / "loss_by_fold.png"
        _plot_fold_metric(benchmarks, "loss", loss_path, "Loss")
        results["loss_by_fold"] = loss_path

        avg_path = output_dir / "average_accuracy.png"
        _plot_average_metric(benchmarks, "accuracy", avg_path, "Accuracy")
        results["average_accuracy"] = avg_path

        balanced_path = output_dir / "balanced_accuracy_by_fold.png"
        _plot_fold_metric(
            benchmarks, "balanced_accuracy", balanced_path, "Balanced Accuracy"
        )
        results["balanced_accuracy_by_fold"] = balanced_path

        avg_balanced_path = output_dir / "average_balanced_accuracy.png"
        _plot_average_metric(
            benchmarks, "balanced_accuracy", avg_balanced_path, "Balanced Accuracy"
        )
        results["average_balanced_accuracy"] = avg_balanced_path

        recall_path = output_dir / "recall_by_fold.png"
        _plot_fold_metric(benchmarks, "recall", recall_path, "Recall")
        results["recall_by_fold"] = recall_path

        avg_recall_path = output_dir / "average_recall.png"
        _plot_average_metric(benchmarks, "recall", avg_recall_path, "Recall")
        results["average_recall"] = avg_recall_path
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from Brainshake benchmark JSON files."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_ROOT / "out" / "benchmarks",
        help="Directory containing benchmark JSON files (default: out/benchmarks under repo root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory to write plot PNG files (default: out/plots under repo root).",
    )
    args = parser.parse_args()

    benchmark_paths = [Path(path) for path in glob.glob(str(args.benchmark_dir / "*"))]
    build_benchmark_plots(benchmark_files=benchmark_paths, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
