# Epileptic Seizure Detection using EEG

![CI](https://github.com/RICSE-donnenfeld/DEEPL-brainshake/actions/workflows/ci.yml/badge.svg)

[Project report (PDF)](latex/main.pdf)

<!--toc:start-->

- [Epileptic Seizure Detection using EEG](#epileptic-seizure-detection-using-eeg)
  - [Getting started](#getting-started)
  - [Dataset](#dataset)
  - [Project structure](#project-structure)
  - [Usage](#usage)
  - [Next steps](#next-steps)
  <!--toc:end-->

Lightweight PyTorch tooling and documentation for experimenting with convolutional models on the CHB-MIT scalp EEG recordings.

## Getting started

1. Clone the repo and switch into the project:

   ```bash
   git clone <repo-url>
   cd DEEPL-brainshake
   ```

2. Recreate the provided environment and activate it:

   ```bash
   conda env create -f environment.yml
   conda activate epilepsy-dl
   ```

3. Install the package in editable mode so the scripts stay in sync with your edits:

   ```bash
   python -m pip install -e .
   ```

   If you prefer to run directly from the repository without installing, you can also
   use the module entry point from the repo root:

   ```bash
   python -m brainshake list
   ```

4. (Optional) Run the data loader to verify that the CHB-MIT files are accessible:

   ```bash
   python -m brainshake.data_handling.load_data
   ```

## Dataset

The project relies on the **CHB-MIT Scalp EEG Database**.
Download the full collection from <https://physionet.org/content/chbmit/1.0.0/>, unzip it locally, and move the seizure-specific files into `data/Epilepsy` so the loader can find `chbXX_seizure_EEGwindow_1.npz` and `chbXX_seizure_metadata_1.parquet` pairs.

Suggested layout:

```
data/
└── Epilepsy/
    ├── chb01_seizure_EEGwindow_1.npz
    ├── chb01_seizure_metadata_1.parquet
    ├── ...
    └── chb24_seizure_metadata_1.parquet
```

The loader in `src/brainshake/data.py` iterates over 24 patients, so both `.npz` and `.parquet` files must follow the naming schema above.

## Project structure

```
DEEPL-brainshake/
├── docs/                          # research notes, architectures, validation reports
│   ├── Architecture/              # RNN/LSTM/CNN explorations and code snippets
│   ├── Data Set Description/      # dataset summaries and text descriptions
│   ├── References/                # collected papers and external guides
│   └── Validation and Verification/# experimental designs and statistical analyses
├── data/                          # user-provided CHB-MIT exports
│   └── Epilepsy/
├── src/
│   └── brainshake/                 # python package containing helpers
├── environment.yml                # conda environment definition
├── pyproject.toml                 # package metadata + dependencies
└── README.md
```

## Usage

All workflows are exposed through a single CLI entry point so you always launch experiments in a consistent manner:

```bash
# after installing the package (editable or regular)
brainshake list
brainshake run analyze-data -- --help

# or run from the repository root without installation
python -m brainshake list
python -m brainshake run visualize-data
```

On Windows, `pip install -e .` may install `brainshake.exe` into your Python
Scripts directory which is not always on `PATH`. In that case, prefer the
portable module entry point:

```bash
python -m brainshake list
```

- `brainshake list` (or `python -m brainshake.cli list`) shows the available commands grouped by category.
- `brainshake list` (or `python -m brainshake list`) shows the available commands grouped by category.
- `brainshake run <workflow>` forwards the rest of the arguments to that module so you can still use the flags you expect (e.g., `brainshake run train-cnn -- --epochs 20`).
- `brainshake run analyze-data` runs the real-patient feature extractor that creates `out/data_analyze/summary.json`, followed by `brainshake run visualize-data` for the static plots, and `brainshake run plot-benchmarks` to build the benchmark charts after evaluations.

- **Reference material.** The `docs/` subtree still collects architecture sketches, channel-fusion experiments, and validation plans that can guide experiments.

### Full pipeline

`brainshake compile` (or `python -m brainshake compile`) runs every workflow top-to-bottom: data analysis, visualization, CNN training, each evaluation pipeline, and benchmark plotting. It feeds the CNN training/evaluation commands the production-level arguments (`-c train -e 30 --kfolds 5 --seed 2026` for training, `--epochs 20 --n-splits 5 --random-state 2026 --use-saved-models` for the CNN eval, `--n-splits 5 --n-estimators 250 --max-depth 12 --random-state 2026` for the random forest, etc.) so the whole dataset is exercised with enough epochs before plotting. Training now saves fold checkpoints under `out/models/cnn/cnn_fold_00N.pt` and the evaluation step reuses those saved models instead of retraining. Run it once per heavy experiment; use `brainshake run <command>` for quick iterations.

## Next steps

1. Flesh out `brainshake.train_cnn` with the desired convolutional architecture, data batching, and logging.
2. Add unit tests for `EEGDataset.k_fold` split modes (patient, window, seizure).
3. Add tests/benchmarks that validate training on a subset of the CHB-MIT windows.

- Windows/seizure/patient-level baselines
- CNN models
- LSTM
- Compare to existing literature
