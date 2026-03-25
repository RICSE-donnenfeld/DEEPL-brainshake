"""Utilities to chart benchmark outputs from the models."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, Mapping, cast

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = REPO_ROOT / "out" / "plots"


def _load_benchmark(path: Path) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_fold_metric(
    benchmarks: Mapping[str, Mapping[str, object]],
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data in benchmarks.items():
        folds = cast(list[Mapping[str, object]], data.get("folds") or [])
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

            metric_value = fold.get(metric)
            if isinstance(metric_value, (int, float)):
                y_values.append(float(metric_value))
            else:
                y_values.append(0.0)
        ax.plot(x_values, y_values, marker="o", label=label)
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} per Fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_average_accuracy(
    benchmarks: Mapping[str, Mapping[str, object]], output_path: Path
) -> None:
    labels = []
    values = []
    for label, data in benchmarks.items():
        avg = data.get("average_accuracy")
        if isinstance(avg, (int, float)):
            labels.append(label)
            values.append(float(avg))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Average Accuracy by Benchmark")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_benchmark_plots(
    benchmark_files: Iterable[Path], *, output_dir: Path | None = None
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
        _plot_average_accuracy(benchmarks, avg_path)
        results["average_accuracy"] = avg_path
    return results


def main():
    benchmark_paths = [
        Path(path) for path in glob.glob(f"{REPO_ROOT}/out/benchmarks/*")
    ]
    build_benchmark_plots(benchmark_files=benchmark_paths, output_dir=DEFAULT_PLOT_DIR)


if __name__ == "__main__":
    main()
