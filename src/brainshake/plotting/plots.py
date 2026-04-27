"""Utilities to chart benchmark outputs from the models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, cast

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = REPO_ROOT / "out" / "plots"
BENCH_DIR = REPO_ROOT / "out" / "benchmarks"


def _load_benchmark(path: Path) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_fold_metric(
    benchmarks: Mapping[str, Mapping[str, object]],
    metric: str,
    output_path: Path,
    ylabel: str,
    title: str | None = None,
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
    ax.set_title(title or f"{ylabel} per Fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "")


def build_main_comparison_plots(output_dir: Path | None = None) -> dict[str, Path]:
    """Build the key comparison plots for the report: accuracy per fold and avg accuracy bar chart."""
    output_dir = output_dir or DEFAULT_PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    main_keys = {
        "Threshold": BENCH_DIR / "threshold.json",
        "Random Forest": BENCH_DIR / "randomforest.json",
        "CNN (patient)": BENCH_DIR / "cnn_patient.json",
        "LSTM-std (seizure)": BENCH_DIR / "lstm_std_seizure.json",
    }
    benchmarks: dict[str, Mapping[str, object]] = {}
    for label, path in main_keys.items():
        if path.exists():
            benchmarks[label] = _load_benchmark(path)

    results: dict[str, Path] = {}
    if benchmarks:
        p1 = output_dir / "accuracy_by_fold.png"
        _plot_fold_metric(benchmarks, "accuracy", p1, "Accuracy", "Accuracy per Fold — Main Comparison")
        results["accuracy_by_fold"] = p1

        p2 = output_dir / "average_accuracy.png"
        _plot_average_accuracy(benchmarks, p2)
        results["average_accuracy"] = p2

    return results


def build_lstm_pool_comparison(output_dir: Path | None = None) -> dict[str, Path]:
    """Build the LSTM pooling comparison plot."""
    output_dir = output_dir or DEFAULT_PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_keys = {
        "std": BENCH_DIR / "lstm_std_seizure.json",
        "mean": BENCH_DIR / "lstm_mean_seizure.json",
        "mean_std": BENCH_DIR / "lstm_mean_std_seizure.json",
        "conv_proj": BENCH_DIR / "lstm_conv_proj_seizure.json",
    }
    benchmarks: dict[str, Mapping[str, object]] = {}
    for label, path in pool_keys.items():
        if path.exists():
            benchmarks[label] = _load_benchmark(path)

    results: dict[str, Path] = {}
    if benchmarks:
        p = output_dir / "lstm_pool_comparison.png"
        _plot_fold_metric(benchmarks, "accuracy", p, "Accuracy", "LSTM Pooling Comparison (seizure-level)")
        results["lstm_pool_comparison"] = p

        avg_labels = []
        avg_values = []
        avg_f1s = []
        for label, data in benchmarks.items():
            avg = data.get("average_accuracy")
            if isinstance(avg, (int, float)):
                avg_labels.append(label)
                avg_values.append(float(avg))
                f1 = data.get("average_f1", 0.0)
                avg_f1s.append(float(f1) if isinstance(f1, (int, float)) else 0.0)

        if avg_labels:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(avg_labels))
            width = 0.35
            bars1 = ax.bar(x - width / 2, avg_values, width, label="Accuracy", color="#4c72b0")
            bars2 = ax.bar(x + width / 2, avg_f1s, width, label="F1", color="#dd8452")
            ax.set_xticks(x)
            ax.set_xticklabels(avg_labels)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("LSTM Pooling — Average Accuracy & F1 (seizure-level)")
            ax.legend()
            for bar_group in (bars1, bars2):
                for bar in bar_group:
                    h = bar.get_height()
                    if h > 0.05:
                        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
            fig.tight_layout()
            p2 = output_dir / "lstm_pool_avg.png"
            fig.savefig(p2, dpi=150)
            plt.close(fig)
            results["lstm_pool_avg"] = p2

    return results


def build_cnn_levels_comparison(output_dir: Path | None = None) -> dict[str, Path]:
    """Build the CNN split-level comparison plot."""
    output_dir = output_dir or DEFAULT_PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    level_keys = {
        "patient": BENCH_DIR / "cnn_patient.json",
        "window": BENCH_DIR / "cnn_window.json",
        "seizure": BENCH_DIR / "cnn_seizure.json",
    }
    benchmarks: dict[str, Mapping[str, object]] = {}
    for label, path in level_keys.items():
        if path.exists():
            benchmarks[label] = _load_benchmark(path)

    results: dict[str, Path] = {}
    if benchmarks:
        p = output_dir / "cnn_levels_accuracy.png"
        _plot_fold_metric(benchmarks, "accuracy", p, "Accuracy", "CNN — Split-Level Comparison")
        results["cnn_levels_accuracy"] = p

    avg_labels = []
    avg_accs = []
    avg_f1s = []
    for label, data in benchmarks.items():
        avg = data.get("average_accuracy")
        f1 = data.get("average_f1", 0.0)
        if isinstance(avg, (int, float)):
            avg_labels.append(label)
            avg_accs.append(float(avg))
            avg_f1s.append(float(f1) if isinstance(f1, (int, float)) else 0.0)
    if avg_labels:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(avg_labels))
        width = 0.35
        bars1 = ax.bar(x - width / 2, avg_accs, width, label="Accuracy", color="#4c72b0")
        bars2 = ax.bar(x + width / 2, avg_f1s, width, label="F1", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(avg_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("CNN — Average Accuracy & F1 by Split Level")
        ax.legend()
        for bar_group in (bars1, bars2):
            for bar in bar_group:
                h = bar.get_height()
                if h > 0.05:
                    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        fig.tight_layout()
        p2 = output_dir / "cnn_levels_avg.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        results["cnn_levels_avg"] = p2

    return results


def build_nsl_comparison(output_dir: Path | None = None) -> dict[str, Path]:
    """Build the LSTM non-seizure-length sweep comparison."""
    output_dir = output_dir or DEFAULT_PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    nsl_keys = {
        "nsl=5": BENCH_DIR / "lstm_std_seizure_nsl5.json",
        "nsl=10": BENCH_DIR / "lstm_std_seizure_nsl10.json",
        "nsl=20": BENCH_DIR / "lstm_std_seizure_nsl20.json",
        "nsl=50": BENCH_DIR / "lstm_std_seizure_nsl50.json",
    }
    benchmarks: dict[str, Mapping[str, object]] = {}
    for label, path in nsl_keys.items():
        if path.exists():
            benchmarks[label] = _load_benchmark(path)

    results: dict[str, Path] = {}
    if benchmarks:
        p = output_dir / "lstm_nsl_comparison.png"
        _plot_fold_metric(benchmarks, "accuracy", p, "Accuracy", "LSTM — Non-Seizure Length Sweep (std, seizure-level)")
        results["lstm_nsl_comparison"] = p

    avg_labels = []
    avg_accs = []
    avg_f1s = []
    for label, data in benchmarks.items():
        avg = data.get("average_accuracy")
        f1 = data.get("average_f1", 0.0)
        if isinstance(avg, (int, float)):
            avg_labels.append(label)
            avg_accs.append(float(avg))
            avg_f1s.append(float(f1) if isinstance(f1, (int, float)) else 0.0)
    if avg_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(avg_labels, avg_accs, color="#4c72b0")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Average Accuracy")
        ax.set_title("LSTM — Average Accuracy by NSL (std, seizure-level)")
        for i, v in enumerate(avg_accs):
            if v > 0.05:
                ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        fig.tight_layout()
        p2 = output_dir / "lstm_nsl_avg.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        results["lstm_nsl_avg"] = p2

    return results


def build_all_plots(output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = output_dir or DEFAULT_PLOT_DIR
    results: dict[str, Path] = {}
    results.update(build_main_comparison_plots(output_dir))
    results.update(build_lstm_pool_comparison(output_dir))
    results.update(build_cnn_levels_comparison(output_dir))
    results.update(build_nsl_comparison(output_dir))
    return results


def main():
    build_all_plots()


if __name__ == "__main__":
    main()