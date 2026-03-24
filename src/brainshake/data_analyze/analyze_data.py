import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple, Sequence
from numpy.typing import NDArray

DATA_DIR: Path = Path("data/Epilepsy")
OUTPUT_DIR: Path = Path("out")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_mean(eeg_signal: NDArray[Any]) -> NDArray[Any]:
    return np.mean(eeg_signal, axis=-1)


def compute_std(eeg_signal: NDArray[Any]) -> NDArray[Any]:
    return np.std(eeg_signal, axis=-1)


def compute_min(eeg_signal: NDArray[Any]) -> NDArray[Any]:
    return np.min(eeg_signal, axis=-1)


def compute_max(eeg_signal: NDArray[Any]) -> NDArray[Any]:
    return np.max(eeg_signal, axis=-1)


def compute_range(eeg_signal: NDArray[Any]) -> NDArray[Any]:
    return compute_max(eeg_signal) - compute_min(eeg_signal)


def extract_5_metrics(eeg_window: NDArray[Any]) -> Dict[str, float]:
    """
    Extract all 5 basic metrics from EEG window.

    Input:  EEG window of shape [21 channels, 128 samples]
    Output: Dictionary with 5 metrics
    """
    # Compute per-channel, then average across channels
    return {
        "mean": float(np.mean(compute_mean(eeg_window))),
        "std": float(np.mean(compute_std(eeg_window))),
        "min": float(np.mean(compute_min(eeg_window))),
        "max": float(np.mean(compute_max(eeg_window))),
        "range": float(np.mean(compute_range(eeg_window))),
    }


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_patient(
    patient_id: int,
) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
    """
    Load EEG data for one patient.

    Returns:
        eeg_data: shape [N_windows, 21, 128]
        labels: shape [N_windows], 0=non-seizure, 1=seizure
    """
    pid = f"chb{patient_id:02d}"

    npz_path = DATA_DIR / f"{pid}_seizure_EEGwindow_1.npz"
    meta_path = DATA_DIR / f"{pid}_seizure_metadata_1.parquet"

    if not npz_path.exists():
        print(f"  WARNING: Patient {patient_id} not found")
        return None, None

    # Load EEG windows
    data = np.load(npz_path, allow_pickle=True)
    eeg_data = data["EEG_win"].astype(np.float32)

    # Load labels
    metadata = pd.read_parquet(meta_path)
    labels = metadata["class"].to_numpy()

    return eeg_data, labels


def load_all_patients(
    patient_ids: Sequence[int],
) -> Tuple[List[NDArray[Any]], List[NDArray[Any]], List[Dict[str, Any]]]:
    """
    Load data for multiple patients.

    Returns:
        all_eeg: list of [N_windows, 21, 128]
        all_labels: list of [N_windows]
        patient_info: list of dicts with patient info
    """
    all_eeg = []
    all_labels = []
    patient_info = []

    for pid in patient_ids:
        eeg, labels = load_patient(pid)
        if eeg is not None and labels is not None:
            all_eeg.append(eeg)
            all_labels.append(labels)
            patient_info.append(
                {
                    "patient_id": pid,
                    "n_windows": len(eeg),
                    "n_seizure": int(np.sum(labels == 1)),
                    "n_non_seizure": int(np.sum(labels == 0)),
                }
            )

    return all_eeg, all_labels, patient_info


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def analyze_single_window(
    eeg_window: NDArray[Any], label: Optional[int] = None
) -> Dict[str, Any]:
    """Analyze one EEG window."""
    metrics: Dict[str, Any] = extract_5_metrics(eeg_window)
    metrics["label"] = label
    return metrics


def analyze_patient(
    patient_id: int,
) -> Optional[Tuple[List[Dict[str, Any]], NDArray[Any]]]:
    """Analyze all windows for one patient."""
    eeg_data, labels = load_patient(patient_id)

    if eeg_data is None or labels is None:
        return None

    print(f"\n  Patient {patient_id}:")
    print(f"    Total windows: {len(eeg_data)}")
    print(f"    Seizure: {np.sum(labels == 1)}, Non-seizure: {np.sum(labels == 0)}")

    # Compute metrics for all windows
    all_metrics = []
    for i in range(len(eeg_data)):
        m = extract_5_metrics(eeg_data[i])
        m["window_id"] = i
        m["label"] = int(labels[i])
        all_metrics.append(m)

    return all_metrics, labels


def compare_seizure_vs_nonseizure(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare average metrics between seizure and non-seizure windows."""
    seizure = [m for m in metrics_list if m["label"] == 1]
    non_seizure = [m for m in metrics_list if m["label"] == 0]

    result: Dict[str, Any] = {
        "n_seizure": len(seizure),
        "n_non_seizure": len(non_seizure),
    }

    if seizure:
        for key in ["mean", "std", "min", "max", "range"]:
            result[f"seizure_avg_{key}"] = np.mean([m[key] for m in seizure])

    if non_seizure:
        for key in ["mean", "std", "min", "max", "range"]:
            result[f"non_seizure_avg_{key}"] = np.mean([m[key] for m in non_seizure])

    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main() -> Dict[str, Any]:
    print("=" * 70)
    print("EEG EPILEPSY ANALYSIS - REAL PATIENT DATA")
    print("=" * 70)

    # Check available patients
    available = []
    for pid in range(1, 25):
        if (DATA_DIR / f"chb{pid:02d}_seizure_EEGwindow_1.npz").exists():
            available.append(pid)

    print(f"\nFound {len(available)} patients with data")
    print(f"Patients: {available[:5]}... (showing first 5)")

    # Analyze first 3 patients for simplicity
    analyze_patients = available[:3]
    print(f"\nAnalyzing patients: {analyze_patients}")

    # Collect all metrics
    all_patient_metrics = []
    all_comparisons = []

    for patient_id in analyze_patients:
        print(f"\n--- Patient {patient_id} ---")

        # Analyze patient
        patient_result = analyze_patient(patient_id)
        if patient_result is None:
            continue
        metrics_list, labels = patient_result

        all_patient_metrics.extend(metrics_list)

        # Compare seizure vs non-seizure
        comparison = compare_seizure_vs_nonseizure(metrics_list)
        comparison["patient_id"] = patient_id
        comparison["windows"] = len(metrics_list)
        all_comparisons.append(comparison)

        # Print comparison
        print("\n  Metric Comparison:")
        print(
            f"  {'Metric':<12} {'Non-Seizure':<15} {'Seizure':<15} {'Difference':<15}"
        )
        print(f"  {'-' * 57}")

        for key in ["mean", "std", "min", "max", "range"]:
            ns = comparison.get(f"non_seizure_avg_{key}", 0)
            sz = comparison.get(f"seizure_avg_{key}", 0)
            diff = sz - ns
            print(f"  {key:<12} {ns:<15.4f} {sz:<15.4f} {diff:<+15.4f}")

    # Save summary
    summary: Dict[str, Any] = {
        "patients_analyzed": analyze_patients,
        "total_windows": len(all_patient_metrics),
        "comparisons": all_comparisons,
        "metrics_mean_all": {
            key: np.mean([m[key] for m in all_patient_metrics])
            for key in ["mean", "std", "min", "max", "range"]
        },
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path.name}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE! Summary ready for visualization.")
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
