# Slide Speaker Notes

## Slide 1 — Title: Deep Learning for EEG Seizure Detection

Welcome everyone. Today we're presenting our work on automatic seizure detection in EEG signals using deep learning. We compared four approaches of increasing complexity — from simple threshold rules to an LSTM that models temporal episodes. The EEG snippet on screen shows what a seizure onset looks like: the sudden burst of rhythmic activity is what we want our models to catch automatically.

---

## Slide 2 — Contents

Here's the roadmap. We'll start with motivation, go through the dataset, walk through each pipeline, discuss cross-validation strategies, then dive into LSTM-specific choices like pooling and context, and finish with results and conclusions.

---

## Slide 3 — Introduction

Epileptic seizures produce abnormal electrical patterns visible in EEG recordings. Manual review is tedious, so automatic detection is clinically valuable. Our goal is to compare models from simplest to most sophisticated and see how far each gets.

---

## Slide 4 — Introduction - Models

We start with a threshold baseline using hand-crafted rules, then a random forest on hand-crafted features, then a CNN that learns features end-to-end, and finally an LSTM that exploits the temporal structure of seizure episodes. The key question: does temporal modelling help?

---

## Slide 5 — Dataset: CHB-MIT

We use the CHB-MIT scalp EEG dataset: 24 pediatric patients, about 572k one-second windows. Each window is 21 channels by 128 samples. Seizure prevalence is only 15%, so class imbalance is a real concern.

---

## Slide 6 — Dataset: Feature Trends

These plots show that standard deviation and range clearly separate seizure from non-seizure windows. The mean, on the other hand, hovers around zero for both classes — confirming that mean alone is a risky feature, but it can still carry some signal when combined with others.

---

## Slide 7 — Preprocessing & Data Flow

The pipeline is straightforward: windows were already segmented in the dataset, so we reuse them as-is. We don't apply per-window normalization — we want the amplitude information preserved. Class-weighted cross-entropy compensates for the imbalance.

---

## Slide 8 — Overview: 4 Pipelines

This diagram gives the big picture. All models share the same data input but differ in how they process it. The threshold and RF use hand-crafted statistics. The CNN processes raw windows through convolutions. The LSTM groups windows into temporal episodes.

---

## Slide 9 — Pipeline 1: Threshold Classifier

The simplest approach: compute stats like standard deviation and range per window, compare against a threshold. The threshold is set at the midpoint between the mean seizure and non-seizure values for each fold. It serves as a sanity check — if a model can't beat this, something is wrong.

---

## Slide 10 — Pipeline 2: Random Forest

We extract 8 features per window — mean, std, min, max, range, peak-to-peak, std/range ratio, and range+std. 200 trees, patient-level 5-fold CV. It does better than the threshold on some patients but struggles overall with 69% accuracy.

---

## Slide 11 — Pipeline 3: CNN

The CNN takes raw 21×128 windows. Three Conv1d blocks extract features, followed by a fully classified head. Only 2 epochs of training with Adam and class-weighted loss. It reaches 86% accuracy on patient-level CV, but the F1 is only 0.55 — good precision but poor recall on harder patients.

---

## Slide 12 — Pipeline 4: LSTM

The LSTM is our most sophisticated model. It groups windows into temporal episodes — sequences of consecutive windows from the same patient. This lets the model exploit the sequential nature of seizures. We support several per-window pooling options to reduce the 21×128 input. The conv_proj mode learns its own projection instead of hand-picking a pooling strategy.

---

## Slide 13 — Cross-Validation Strategies

This is critical. How you split the data matters as much as the model. Patient-level: hold out entire patients — no leakage. Window-level: round-robin by window — but the class ratios per fold become degenerate. Seizure-level: hold out entire seizure episodes — used by the LSTM, no leakage, but requires adding context windows around seizures for meaningful evaluation.

---

## Slide 14 — Seizure-Level k-fold: Context

Without context, the validation set for seizure-level CV would be 100% seizure windows — you couldn't measure specificity. We include ±20 non-seizure windows around each seizure episode, giving a realistic 40-60% non-seizure mix in validation. Training still sees all other windows.

---

## Slide 15 — LSTM: Pooling Options

Since EEG oscillates around zero, mean pooling collapses the amplitude information. Standard deviation preserves it. We tested five options: std is the default, mean is risky, mean_std concatenates both, conv_proj learns a projection, and none flattens the entire window into 2688 dimensions.

---

## Slide 16 — LSTM: Architecture

The LSTM takes packed variable-length sequences, passes them through 2 LSTM layers with 128 hidden units and 0.3 dropout, then classifies each timestep into seizure or non-seizure. Packed sequences handle the variable episode lengths efficiently.

---

## Slide 17 — LSTM: conv_proj Mode

In conv_proj mode, we replace hand-crafted pooling with a learned 1D convolution that projects the raw window from 21×128 down to 32 dimensions before the LSTM. This is more expressive than std (21-d) but much faster than none (2688-d).

---

## Slide 18 — Hyperparameters

Summary table of all hyperparameters. Note: we deliberately keep training short at 2 epochs to avoid overfitting. All models use class-weighted loss. The LSTM uses seizure-level CV with context=20, while simpler models use patient-level.

---

## Slide 19 — Results: Accuracy per Fold

This bar chart shows accuracy across the 5 folds for each model under patient-level CV. Notice the high variance — some patients are inherently harder. CNN ranges from 79% to 93%. The threshold actually beats the random forest on most folds.

---

## Slide 20 — Results: Main Comparison

The headline result: LSTM with seizure-level CV achieves 97.2% accuracy and 0.985 F1. Compare that to CNN at 86.3% / 0.554 under the same patient-level protocol. The LSTM also reaches 99.5% accuracy under patient-level CV. But note — these aren't directly comparable because seizure-level CV includes context windows in validation. The key takeaway is that temporal modelling makes a big difference.

---

## Slide 21 — Results: Window-Level CV Degeneracy

Window-level CV looks great on paper — 97%+ accuracy — but F1 is 0. The model just learns to predict "non-seizure" for everything because the round-robin split creates degenerate class distributions per fold. Window-level CV is unsuitable for this imbalanced dataset.

---

## Slide 22 — Results: CNN Split-Level Comparison

CNN across the three CV levels. Patient-level gives 86% accuracy but F1 is only 0.55. Window-level is degenerate. Seizure-level gives lower accuracy (80%) but much higher F1 (0.89) — the model is more precise about what it calls a seizure.

---

## Slide 23 — Results: LSTM Pooling

The pooling comparison for LSTM under seizure-level CV. Mean actually slightly outperforms std (97.8% vs 97.2%). This is surprising because mean should collapse to zero — the DC offset must shift slightly during seizures. Mean_std is unstable: 2 out of 5 folds collapse to predicting one class, likely because the feature scale mismatch between mean and std components creates gradient instability with only 2 epochs of training. Conv_proj at 93% needs more training epochs.

---

## Slide 24 — Results: Non-Seizure Length Sweep

NSL controls how many consecutive non-seizure windows we chunk together. NSL=5 gives the best result at 98.8% accuracy. NSL=10 is slightly worse. NSL=20 and above causes training collapse — the long non-seizure sequences dominate and the model fails to learn. Keep non-seizure segments short.

---

## Slide 25 — Key Observations

Summarizing: LSTM with seizure-level CV dominates. CNN at patient-level has good accuracy but poor recall on some patients. Window-level CV is degenerate for all models. Mean and std pooling both work well for LSTM, but mean_std is unstable. And NSL must be kept small.

---

## Slide 26 — Conclusion

LSTM seizure-level is the clear winner. CNN at patient-level is decent but has high variance. Mean pooling slightly outperforms std — surprising but it just means the seizure DC offset carries some signal. NSL above 20 kills training. Conv_proj is promising but undertrained at 2 epochs.

---

## Slide 27 — Follow-up Work

Two main directions: first, removing pre-ictal and post-ictal windows — these transition windows are ambiguous and might be hurting precision. Second, trying longer temporal context. Currently we use context=20 (±20 windows around seizures), but longer context could help the LSTM learn better pre-seizure patterns. Also worth running conv_proj for more epochs.

---

## Slide 28 — Thank You

Thank you for your attention. Happy to take any questions.