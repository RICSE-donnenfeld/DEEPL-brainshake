# SeizureLSTM — Episode-Level LSTM for EEG Seizure Detection

## Overview

The LSTM pipeline treats EEG seizure detection as a **sequential** problem: instead of classifying each 1 s window in isolation (as the CNN does), it groups contiguous, same-label windows from the same patient into **episodes** and feeds the resulting variable-length sequences to a recurrent model. This lets the network exploit temporal context — the progression from pre-ictal → ictal → post-ictal — that a per-window classifier ignores.

## Architecture

```
Input:  [batch, seq_len, 21]   (mean-pooled across the 128 time-steps per window)
        ↓
    nn.LSTM(input_size=21, hidden_size=128, num_layers=2, batch_first=True)
        ↓
    nn.Linear(128 → 2)  applied at every time-step
        ↓
Output: [batch, seq_len, 2]    (per-window logits)
```

- **Mean pooling**: each 21×128 window is averaged over the time axis to produce a 21-d feature vector per window.
- **Packed sequences**: variable-length episodes are padded and packed so the LSTM only processes real timesteps.
- **Class weighting**: `CrossEntropyLoss` is weighted by inverse class frequency (`ignore_index=-1` masks padding).

## Data pipeline: seizure-level splits → episodes

```
EEGDataset.k_fold(level="seizure")
         │
         ▼
  train_indices / val_indices
         │
         ▼
SeizureEpisodeDataset(indices)
  ├─ contiguous same-patient, same-label windows → seizure episodes
  └─ non-seizure windows → chunked into segments of length `non_seizure_len`
         │
         ▼
  pad_collate → padded [batch, max_len, 21] + lengths tensor
         │
         ▼
SeizureLSTM (packed LSTM → per-timestep logits)
         │
         ▼
episode_loss() — masks padding (-1), weighted CrossEntropy
```

### k_fold split levels

| Level     | How indices are grouped                                                 | Leakage safeguard                         |
| --------- | ----------------------------------------------------------------------- | ----------------------------------------- |
| `patient` | Whole patient → one fold                                                | No patient spans train/val                |
| `window`  | Each window assigned round-robin; ±4 seizure neighbors dropped from val | No augmented seizure copies leak into val |
| `seizure` | Contiguous seizure episode → one fold; ±`context` non-seizure windows around each episode included in val (default 20) | No partial seizure spans train/val; val has realistic class mix |

### SeizureEpisodeDataset details

- **Seizure episodes**: contiguous windows with `label == 1` within the same patient are kept as a single variable-length sequence.
- **Non-seizure segments**: background windows are chunked into fixed-length segments (default `non_seizure_len = 10`) so they don't produce extremely long sequences.
- **Feature extraction**: `std` pool (default) takes per-channel standard deviation (preserves amplitude for zero-centered oscillating EEG); other options: `mean` (risky — averages to ~0), `mean_std` (42-d), `none` (2688-d, full waveform), `conv_proj` (Conv1d projects raw [21,128] to 32 channels — fast alternative to `none`).

## Usage

```bash
# Seizure-level k-fold evaluation (default)
brainshake run evaluate-lstm -- --n-splits 5 --level seizure --epochs 20

# Patient-level k-fold
brainshake run evaluate-lstm -- --n-splits 5 --level patient --epochs 20

# Quick smoke test on 2 patients
brainshake run evaluate-lstm -- --n-splits 2 --patient-ids 1 2 --epochs 5
```

Results are saved to `out/benchmarks/lstm{suffix}.json`.

## Hyperparameters

| Parameter         | Default                |
| ----------------- | ---------------------- |
| `hidden_size`     | 128                    |
| `num_layers`      | 2                      |
| `dropout`         | 0.3                    |
| `lr`              | 1e-3                   |
| `batch_size`      | 32                     |
| `non_seizure_len` | 10                     |
| `pool`            | std (21-d)             |
| `context`         | 20 (windows around each seizure episode in val) |
| `pool`            | mean (21-d per window) |

## Files

| File          | Purpose                                      |
| ------------- | -------------------------------------------- |
| `model.py`    | `SeizureLSTM` module + `episode_loss` helper |
| `dataset.py`  | `SeizureEpisodeDataset` + `pad_collate`      |
| `evaluate.py` | Full k-fold train/eval loop + CLI            |

