# Epileptic Seizure Detection using EEG

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

4. (Optional) Run the data loader to verify that the CHB-MIT files are accessible:

   ```bash
   python -m brainshake.data -v
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

- **Inspect the dataset.** The `brainshake.data` entry point loads every `.npz` EEG window and logs success or failure with increasing verbosity. This is a good sanity check before training.
- **Train a CNN (under development).** `src/brainshake/train_cnn.py` is intended to orchestrate model, optimizer, and evaluation wiring once training logic is implemented. Run `python -m brainshake.train_cnn` when it is ready.
- **Reference material.** The `docs/` subtree collects architecture sketches, channel-fusion experiments, and validation plans that can guide experiments.

## Next steps

1. Flesh out `brainshake.train_cnn` with the desired convolutional architecture, data batching, and logging.
2. Add tests/benchmarks that validate training on a subset of the CHB-MIT windows.
3. Document reproducibility steps inside `docs/Validation and Verification` to capture the current evaluation plan.
