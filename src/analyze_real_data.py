"""
EEG Epilepsy Analysis - Simple & Clean
=====================================
Uses REAL patient data from CHB-MIT dataset.

5 Basic Metrics: mean, std, min, max, range

Run: python src/analyze_real_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/epilepsy_data/Epilepsy")
OUTPUT_DIR = Path("outputs_real")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 5 BASIC METRICS FUNCTIONS
# ============================================================================

def compute_mean(eeg_signal):
    """
    MEAN - Average value of signal.
    
    Formula: mean = sum(x) / count(x)
    
    What it means: The center/baseline of the signal.
    """
    return np.mean(eeg_signal, axis=-1)

def compute_std(eeg_signal):
    """
    STD (Standard Deviation) - How spread out the values are.
    
    Formula: std = sqrt(sum((x - mean)²) / n)
    
    What it means:
    - Low std = values clustered together
    - High std = values spread out
    """
    return np.std(eeg_signal, axis=-1)

def compute_min(eeg_signal):
    """
    MIN - Smallest value in signal.
    
    What it means: Deepest negative point.
    """
    return np.min(eeg_signal, axis=-1)

def compute_max(eeg_signal):
    """
    MAX - Largest value in signal.
    
    What it means: Highest positive point.
    """
    return np.max(eeg_signal, axis=-1)

def compute_range(eeg_signal):
    """
    RANGE - Distance from min to max.
    
    Formula: range = max - min
    
    What it means: Total amplitude of the signal.
    Key metric for seizure detection!
    """
    return compute_max(eeg_signal) - compute_min(eeg_signal)

def extract_5_metrics(eeg_window):
    """
    Extract all 5 basic metrics from EEG window.
    
    Input:  EEG window of shape [21 channels, 128 samples]
    Output: Dictionary with 5 metrics
    """
    # Compute per-channel, then average across channels
    return {
        'mean': float(np.mean(compute_mean(eeg_window))),
        'std': float(np.mean(compute_std(eeg_window))),
        'min': float(np.mean(compute_min(eeg_window))),
        'max': float(np.mean(compute_max(eeg_window))),
        'range': float(np.mean(compute_range(eeg_window))),
    }

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_patient(patient_id):
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
    labels = metadata["class"].values
    
    return eeg_data, labels

def load_all_patients(patient_ids):
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
        if eeg is not None:
            all_eeg.append(eeg)
            all_labels.append(labels)
            patient_info.append({
                'patient_id': pid,
                'n_windows': len(eeg),
                'n_seizure': int(np.sum(labels == 1)),
                'n_non_seizure': int(np.sum(labels == 0)),
            })
    
    return all_eeg, all_labels, patient_info

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_single_window(eeg_window, label=None):
    """Analyze one EEG window."""
    metrics = extract_5_metrics(eeg_window)
    metrics['label'] = label
    return metrics

def analyze_patient(patient_id):
    """Analyze all windows for one patient."""
    eeg_data, labels = load_patient(patient_id)
    
    if eeg_data is None:
        return None
    
    print(f"\n  Patient {patient_id}:")
    print(f"    Total windows: {len(eeg_data)}")
    print(f"    Seizure: {np.sum(labels==1)}, Non-seizure: {np.sum(labels==0)}")
    
    # Compute metrics for all windows
    all_metrics = []
    for i in range(len(eeg_data)):
        m = extract_5_metrics(eeg_data[i])
        m['window_id'] = i
        m['label'] = int(labels[i])
        all_metrics.append(m)
    
    return all_metrics, labels

def compare_seizure_vs_nonseizure(metrics_list):
    """Compare average metrics between seizure and non-seizure windows."""
    seizure = [m for m in metrics_list if m['label'] == 1]
    non_seizure = [m for m in metrics_list if m['label'] == 0]
    
    result = {
        'n_seizure': len(seizure),
        'n_non_seizure': len(non_seizure),
    }
    
    if seizure:
        for key in ['mean', 'std', 'min', 'max', 'range']:
            result[f'seizure_avg_{key}'] = np.mean([m[key] for m in seizure])
    
    if non_seizure:
        for key in ['mean', 'std', 'min', 'max', 'range']:
            result[f'non_seizure_avg_{key}'] = np.mean([m[key] for m in non_seizure])
    
    return result

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_eeg_window(eeg_window, label=None, save_path=None):
    """Plot all 21 EEG channels."""
    n_channels, n_samples = eeg_window.shape
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 14), sharex=True)
    
    for i in range(n_channels):
        axes[i].plot(eeg_window[i], linewidth=0.5, color='steelblue')
        axes[i].set_ylabel(f'Ch{i+1}', fontsize=7)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([eeg_window.min()-10, eeg_window.max()+10])
    
    axes[-1].set_xlabel('Time (samples)', fontsize=10)
    
    label_text = f"Seizure (Label=1)" if label == 1 else f"Non-Seizure (Label=0)" if label == 0 else "Unknown"
    fig.suptitle(f'EEG Signal - 21 Channels\n{label_text}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    Saved: {save_path.name}")
    else:
        plt.show()

def plot_metric_comparison(comparison_dict, save_path=None):
    """Plot bar chart comparing seizure vs non-seizure metrics."""
    metrics = ['mean', 'std', 'min', 'max', 'range']
    
    seizure_vals = [comparison_dict.get(f'seizure_avg_{m}', 0) for m in metrics]
    non_seizure_vals = [comparison_dict.get(f'non_seizure_avg_{m}', 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, non_seizure_vals, width, label='Non-Seizure', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, seizure_vals, width, label='Seizure', color='red', alpha=0.7)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Comparison: Seizure vs Non-Seizure Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    Saved: {save_path.name}")
    else:
        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("EEG EPILEPSY ANALYSIS - REAL PATIENT DATA")
    print("="*70)
    
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
        metrics_list, labels = analyze_patient(patient_id)
        if metrics_list is None:
            continue
        
        all_patient_metrics.extend(metrics_list)
        
        # Compare seizure vs non-seizure
        comparison = compare_seizure_vs_nonseizure(metrics_list)
        comparison['patient_id'] = patient_id
        all_comparisons.append(comparison)
        
        # Print comparison
        print(f"\n  Metric Comparison:")
        print(f"  {'Metric':<12} {'Non-Seizure':<15} {'Seizure':<15} {'Difference':<15}")
        print(f"  {'-'*57}")
        
        for key in ['mean', 'std', 'min', 'max', 'range']:
            ns = comparison.get(f'non_seizure_avg_{key}', 0)
            sz = comparison.get(f'seizure_avg_{key}', 0)
            diff = sz - ns
            print(f"  {key:<12} {ns:<15.4f} {sz:<15.4f} {diff:<+15.4f}")
    
    # Save summary
    summary = {
        'patients_analyzed': analyze_patients,
        'total_windows': len(all_patient_metrics),
        'comparisons': all_comparisons,
        'metrics_mean_all': {
            key: np.mean([m[key] for m in all_patient_metrics])
            for key in ['mean', 'std', 'min', 'max', 'range']
        }
    }
    
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path.name}")
    
    # Plot example windows
    print("\n--- Generating Visualizations ---")
    
    # Find a seizure and non-seizure window
    for patient_id in analyze_patients:
        eeg_data, labels = load_patient(patient_id)
        
        # Non-seizure example
        non_sz_idx = np.where(labels == 0)[0][0]
        plot_eeg_window(
            eeg_data[non_sz_idx], 
            label=0,
            save_path=OUTPUT_DIR / f"patient{patient_id}_nonseizure.png"
        )
        
        # Seizure example
        sz_idx = np.where(labels == 1)[0]
        if len(sz_idx) > 0:
            plot_eeg_window(
                eeg_data[sz_idx[0]], 
                label=1,
                save_path=OUTPUT_DIR / f"patient{patient_id}_seizure.png"
            )
    
    # Plot comparison
    if all_comparisons:
        plot_metric_comparison(all_comparisons[0], save_path=OUTPUT_DIR / "metric_comparison.png")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("="*70)
    
    return summary

if __name__ == "__main__":
    summary = main()
