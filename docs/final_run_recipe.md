# Final run recipe (seed 2026)

This checklist freezes a **single reproducible run** (metrics + plots) and makes the report reference **exactly those outputs**.

## Prerequisites

- Dataset present under `data/Epilepsy/` as pairs:
  - `chbXX_seizure_EEGwindow_1.npz`
  - `chbXX_seizure_metadata_1.parquet`
- Use the project environment:
  - `conda env create -f environment.yml`
  - `conda activate epilepsy-dl`
  - `python -m pip install -e .`

### Notes for the UAB cluster / Linux hosts

- Many cluster images only provide `python3` (no `python` alias). In that case, replace `python` with `python3` in all commands below.
- Run commands from the **repo root** (the folder that contains `pyproject.toml` and `src/`).
- If you do *not* install the package (no `pip install -e .`), run with:

  ```bash
  export PYTHONPATH="$PWD/src"
  ```

  This also helps avoid import shadowing if you have a `~/brainshake/` directory in your home.

## Option A (recommended): explicit final pipeline

Run these commands from the repo root.

### 1) Data analysis + data plots (all patients)

```bash
python -m brainshake run analyze-data -- --all-patients
python -m brainshake run visualize-data -- --summary-path out/data_analyze/summary.json --output-dir out/data_analyze
```

Cluster equivalent (if `python` is missing):

```bash
python3 -m brainshake run analyze-data -- --all-patients
python3 -m brainshake run visualize-data -- --summary-path out/data_analyze/summary.json --output-dir out/data_analyze
```

Expected outputs:
- `out/data_analyze/summary.json`
- `out/data_analyze/simple_comparison.png`
- `out/data_analyze/metric_trends.png`

### 2) Train CNN (fixed seed, 5 folds)

Pick the epoch count you want to report (example: 30):

```bash
python -m brainshake run train-cnn -- -c train -e 30 --kfolds 5 --seed 2026 -vvv
```

Expected outputs:
- CNN fold checkpoints under `out/models/cnn/` (one per fold)

### 3) Evaluate all models (patient-level 5-fold CV)

```bash
python -m brainshake run evaluate-cnn -- --n-splits 5 --random-state 2026 --use-saved-models
python -m brainshake run evaluate-randomforest -- --n-splits 5 --n-estimators 250 --max-depth 12 --random-state 2026
python -m brainshake run evaluate-threshold -- --n-splits 5 --random-state 2026
```

Expected outputs (benchmarks):
- `out/benchmarks/cnn.json` (name may be `evaluate-cnn.json` depending on the module)
- `out/benchmarks/randomforest.json`
- `out/benchmarks/threshold.json`

### 4) Plot benchmarks (includes recall + balanced accuracy)

```bash
python -m brainshake run plot-benchmarks
```

Expected outputs (figures):
- `out/plots/average_accuracy.png`
- `out/plots/accuracy_by_fold.png`
- `out/plots/average_recall.png`
- `out/plots/recall_by_fold.png`
- `out/plots/average_balanced_accuracy.png`
- `out/plots/balanced_accuracy_by_fold.png`

### 5) Build the PDF

```bash
cd latex
latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex
```

Expected output:
- `latex/main.pdf`

## Freeze the run in git (recommended)

After the final run succeeds, commit at least:

- `latex/main.tex` (numbers/text updated)
- `out/plots/*.png` that are referenced by the report

Optional but useful:
- copy the benchmark JSONs into a tracked folder (so the exact metrics are preserved even if `out/benchmarks/` is ignored on some machines):

```bash
mkdir -p docs/final_results
cp out/benchmarks/*.json docs/final_results/
```

Then commit:

```bash
git add latex/main.tex out/plots/*.png docs/final_results/*.json
git commit -m "Freeze final results (seed 2026)"
```

## Option B: `brainshake compile`

`python -m brainshake compile` runs the full pipeline with **fixed seeds**, but note that it uses a short CNN training configuration intended for quick runs.
For the report-quality run, prefer Option A.
