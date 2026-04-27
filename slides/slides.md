---
marp: true
theme: uncover
paginate: true
size: 16:9
style: |
  section {
    background-color: #0f172a;
    background-image: radial-gradient(circle at 20% 20%, #1e293b 0%, #0f172a 100%);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }
  h1 {
    color: #38bdf8;
    font-size: 1.6rem;
    border-bottom: 1px solid #334155;
  }
  h2 {
    color: #94a3b8;
    font-size: 1.3rem;
  }
  table {
    font-size: 0.8rem;
    background: rgba(30, 41, 59, 0.5);
  }
  footer {
    color: #64748b;
  }
---

# Deep Learning for EEG Seizure Detection

### Project Report

<img src="../out/plots/average_accuracy.png" width="40%">

---

# Abstract

Four models compared under patient-level and seizure-level CV: **threshold**, **random forest**, **CNN**, and **LSTM**.
The CNN leads in per-window accuracy; the LSTM exploits temporal context within seizure episodes.

---

# Contents

- Introduction
- Dataset & Analysis
- Methodology (4 pipelines)
- Cross-validation strategies
- Results
- Conclusion & future work

---

# Introduction

Automated seizure detection aids clinical diagnosis. We compare a **threshold algorithm**, **random forest**, **CNN**, and **LSTM** — progressively richer models and split strategies.

---

<img src="../out/data_analyze/simple_comparison.png" width="45%"> <img src="../out/data_analyze/metric_trends.png" width="45%">

# Dataset

- **Source**: CHB-MIT (24 patients, pediatric)
- **Windows**: 1 s segments → ~571k samples, 3.4k seizures
- **Top-left**: Std & range by patient
- **Top-right**: Avg metrics across folds

---

# Methodology: 4 Pipelines

| Model | Input | CV level |
|---|---|---|
| Threshold | Raw signal stats | patient |
| Random Forest | Mean/std/range features | patient |
| CNN | Raw windows [21×128] | patient |
| **LSTM** | Mean-pooled episodes [seq, 21] | **seizure** |

---

# Cross-Validation Strategies

- **Patient-level**: whole patient → one fold (no leakage)
- **Window-level**: each window independently, ±4 seizure neighbors dropped from val
- **Seizure-level**: contiguous seizure episode → one fold; non-seizure always in train

---

# LSTM Pipeline Detail

1. `EEGDataset.k_fold(level="seizure")` splits seizure episodes as atomic units
2. `SeizureEpisodeDataset` groups windows into temporal sequences
   - Seizure episodes stay intact (variable length)
   - Non-seizure windows → chunks of 10
3. `pad_collate` pads variable-length batches, produces `lengths` tensor
4. `SeizureLSTM(input=21, hidden=128, layers=2)` → per-window logits `[batch, seq, 2]`
5. `episode_loss` masks padding (`ignore_index=-1`) + class weights

---

<img src="../out/plots/threshold.png" width="30%"> <img src="../out/plots/randomforest.png" width="30%"> <img src="../out/plots/cnn.png" width="30%">

# Pipeline Diagrams

Threshold → Random Forest → CNN (LSTM not shown but follows episode grouping)

---

# Hyperparameters

| Parameter | RF | CNN | LSTM |
|---|---|---|---|
| Trees/filters/layers | 100 | [32,64,128] | 2 LSTM |
| Hidden size | — | — | 128 |
| Learning rate | — | 1e-3 | 1e-3 |
| Batch size | — | 64 | 32 |
| Epochs | — | 50 | 20 |
| Dropout | — | 0.3 | 0.3 |
| CV level | patient | patient | seizure |

---

<img src="../out/plots/accuracy_by_fold.png" width="45%"> <img src="../out/plots/average_accuracy.png" width="45%">

# Results

- **CNN** outperforms threshold and RF baselines (>80% across folds)
- **LSTM** exploits temporal context within seizure episodes
- Seizure-level CV preserves episode integrity; direct comparison pending matching splits

---

# Conclusion

- CNN is the top per-window classifier under patient-level CV
- LSTM adds a sequential perspective via episode-level processing
- Future work: compare CNN vs LSTM under matching seizure-level splits, explore multichannel + longer temporal context

---

# Thank You

**Any Questions?**