from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


DATA_DIR = Path("data/Epilepsy")


def load_patient(patient_id: int):
    pid = f"chb{patient_id:02d}"
    npz_path = DATA_DIR / f"{pid}_seizure_EEGwindow_1.npz"
    meta_path = DATA_DIR / f"{pid}_seizure_metadata_1.parquet"

    npz_data = np.load(npz_path, allow_pickle=True)
    X = npz_data["EEG_win"]              # shape: [N, 21, 128]
    meta = pd.read_parquet(meta_path)    # contains 'class'

    return X, meta


def show_basic_info(patient_id: int):
    X, meta = load_patient(patient_id)

    print(f"\n=== Patient chb{patient_id:02d} ===")
    print("EEG shape:", X.shape)
    print("Single window shape:", X[0].shape)
    print("dtype:", X.dtype)
    print("\nMetadata columns:", list(meta.columns))
    print("\nClass distribution:")
    print(meta["class"].value_counts().sort_index())

    print("\nFirst rows of metadata:")
    print(meta.head())


def plot_window(patient_id: int, window_idx: int):
    X, meta = load_patient(patient_id)
    window = X[window_idx]   # shape [21, 128]
    label = meta.iloc[window_idx]["class"]

    plt.figure(figsize=(12, 8))

    offset = 0
    for ch in range(window.shape[0]):
        plt.plot(window[ch] + offset, linewidth=0.8)
        offset += 5  # vertical spacing between channels

    plt.title(
        f"Patient chb{patient_id:02d} | Window {window_idx} | "
        f"Class = {label}"
    )
    plt.xlabel("Time step")
    plt.ylabel("Channels (offset)")
    plt.tight_layout()
    plt.show()


def compare_nonseizure_vs_seizure(patient_id: int):
    X, meta = load_patient(patient_id)

    non_idx = meta.index[meta["class"] == 0][0]
    seiz_idx = meta.index[meta["class"] == 1][0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for ax, idx, title in [
        (axes[0], non_idx, "Non-seizure"),
        (axes[1], seiz_idx, "Seizure"),
    ]:
        window = X[idx]
        offset = 0
        for ch in range(window.shape[0]):
            ax.plot(window[ch] + offset, linewidth=0.8)
            offset += 5
        ax.set_title(f"{title} | window {idx}")

    axes[1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()


def summarize_all_patients():
    rows = []

    for patient_id in range(1, 25):
        print(f"Loading chb{patient_id:02d}...")
        try:
            X, meta = load_patient(patient_id)
            class_counts = meta["class"].value_counts().to_dict()

            rows.append({
                "patient": f"chb{patient_id:02d}",
                "n_windows": len(meta),
                "n_non_seizure": class_counts.get(0, 0),
                "n_seizure": class_counts.get(1, 0),
            })
        except Exception as e:
            print(f"Could not load chb{patient_id:02d}: {e}")

    df = pd.DataFrame(rows)
    print("\n=== Summary over all patients ===")
    print(df)

    plt.figure(figsize=(12, 5))
    plt.bar(df["patient"], df["n_windows"])
    plt.xticks(rotation=90)
    plt.title("Number of windows per patient")
    plt.tight_layout()
    plt.savefig("windows_per_patient.png")
    print("Saved windows_per_patient.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(df["patient"], df["n_seizure"])
    plt.xticks(rotation=90)
    plt.title("Number of seizure windows per patient")
    plt.tight_layout()
    plt.savefig("seizures_per_patient.png")
    print("Saved seizures_per_patient.png")
    plt.close()


if __name__ == "__main__":
    show_basic_info(1)
    plot_window(patient_id=1, window_idx=0)
    compare_nonseizure_vs_seizure(patient_id=1)
    summarize_all_patients()