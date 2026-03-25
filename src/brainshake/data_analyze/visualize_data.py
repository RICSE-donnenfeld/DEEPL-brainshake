import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("out/data_analyze")
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_summary(path: Path = SUMMARY_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    with path.open() as f:
        return json.load(f)


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )


def _sorted_comparisons(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    comparisons = summary.get("comparisons", [])
    return sorted(comparisons, key=lambda c: c.get("patient_id", 0))


def create_simple_comparison(summary: Dict[str, Any]) -> None:
    comparisons = _sorted_comparisons(summary)
    if not comparisons:
        print("No comparisons available to plot.")
        return

    patients = [f"Patient {int(comp['patient_id'])}" for comp in comparisons]
    x = np.arange(len(patients))
    width = 0.35

    non_std = [comp.get("non_seizure_avg_std", 0) for comp in comparisons]
    sz_std = [comp.get("seizure_avg_std", 0) for comp in comparisons]
    non_range = [comp.get("non_seizure_avg_range", 0) for comp in comparisons]
    sz_range = [comp.get("seizure_avg_range", 0) for comp in comparisons]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"non": "#2ecc71", "seizure": "#e74c3c"}

    ax_std = axes[0]
    bars_ns = ax_std.bar(
        x - width / 2,
        non_std,
        width,
        label="Non-Seizure",
        color=colors["non"],
        edgecolor="black",
    )
    bars_sz = ax_std.bar(
        x + width / 2,
        sz_std,
        width,
        label="Seizure",
        color=colors["seizure"],
        edgecolor="black",
    )
    ax_std.set_xticks(x)
    ax_std.set_xticklabels(patients, rotation=90, ha="center")
    ax_std.set_xlabel("Patient")
    ax_std.set_ylabel("Standard Deviation")
    ax_std.set_title("Non-Seizure vs Seizure Std")
    ax_std.legend(fontsize=10)
    ax_std.grid(True, alpha=0.3)
    # Annotation removed to reduce clutter for 24 patients

    ax_range = axes[1]
    bars_ns = ax_range.bar(
        x - width / 2,
        non_range,
        width,
        label="Non-Seizure",
        color=colors["non"],
        edgecolor="black",
    )
    bars_sz = ax_range.bar(
        x + width / 2,
        sz_range,
        width,
        label="Seizure",
        color=colors["seizure"],
        edgecolor="black",
    )
    ax_range.set_xticks(x)
    ax_range.set_xticklabels(patients, rotation=90, ha="center")
    ax_range.set_xlabel("Patient")
    ax_range.set_ylabel("Range")
    ax_range.set_title("Non-Seizure vs Seizure Range")
    ax_range.grid(True, alpha=0.3)
    # Annotation removed to reduce clutter for 24 patients

    fig.suptitle(
        "Seizure vs Non-Seizure Variability",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "simple_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: simple_comparison.png")


def create_metric_trends(summary: Dict[str, Any]) -> None:
    comparisons = _sorted_comparisons(summary)
    if not comparisons:
        print("No comparisons available to plot metric trends.")
        return

    metrics = ["mean", "std", "min", "max", "range"]

    def avg(prefix: str, metric: str) -> float:
        values = [comp.get(f"{prefix}_{metric}", 0) for comp in comparisons]
        clean = [float(v) for v in values if isinstance(v, (int, float))]
        return float(np.mean(clean)) if clean else 0.0

    non_vals = [avg("non_seizure_avg", metric) for metric in metrics]
    sz_vals = [avg("seizure_avg", metric) for metric in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics, non_vals, marker="o", label="Non-Seizure", color="#2ecc71")
    ax.plot(metrics, sz_vals, marker="o", label="Seizure", color="#e74c3c")
    ax.fill_between(metrics, non_vals, sz_vals, color="gray", alpha=0.1)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Avg Value")
    ax.set_title("Average Metric Trends for Seizure vs Non-Seizure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "metric_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: metric_trends.png")


def create_summary_table(summary: Dict[str, Any]) -> None:
    comparisons = _sorted_comparisons(summary)
    if not comparisons:
        print("No comparisons available to build table.")
        return

    data = [
        ["Patient", "Windows", "Seizure Std", "Non-Seizure Std", "Std Δ"],
    ]
    for comp in comparisons:
        patient_label = f"Patient {int(comp.get('patient_id', 0))}"
        windows = int(comp.get("windows", comp.get("n_windows", 0)))
        seizure_std = comp.get("seizure_avg_std", 0)
        non_std = comp.get("non_seizure_avg_std", 0)
        std_delta = seizure_std - non_std
        data.append(
            [
                patient_label,
                f"{windows:,}",
                f"{seizure_std:.2f}",
                f"{non_std:.2f}",
                f"{std_delta:.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 4 + len(comparisons) * 0.4))
    ax.axis("off")
    table = ax.table(
        cellText=data,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.25, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.2)

    for i in range(len(data)):
        for j in range(len(data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#3498db")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#fdfefe" if i % 2 else "#f2f4f4")

    ax.set_title("Patient-Level Summary", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "results_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results_table.png")


def main() -> Dict[str, Any]:
    print("=" * 60)
    print("CREATING VISUALIZATIONS FROM summary.json")
    print("=" * 60)

    summary = load_summary()
    create_simple_comparison(summary)
    create_metric_trends(summary)
    create_summary_table(summary)

    print("=" * 60)
    print("VISUALIZATIONS SAVED TO: outputs_real/")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    main()
