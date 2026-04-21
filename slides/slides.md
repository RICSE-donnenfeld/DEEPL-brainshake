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

![w:600](../out/plots/average_accuracy.png)

---

# Abstract

A one-dimensional CNN outperforms threshold and random forest baselines for EEG seizure detection under patient-level cross-validation.

---

# Contents

- Introduction
- Dataset Description & Analysis
- Methodology
- Implementation Details
- Experimental Design
- Results
- Discussion / Conclusion

---

# Introduction

Automated seizure detection aids clinical diagnosis. We compare a **threshold algorithm**, **random forest**, and **CNN** to assess model expressivity.

---

![bg left:50%](../out/data_analyze/simple_comparison.png)
![bg left:50%](../out/data_analyze/metric_trends.png)

# Dataset Description

- **Source**: CHB-MIT (24 patients)
- **Windows**: 1s segments (~571k samples)
- **Visuals**:
  - Std/Range (Top Left)
  - Metric Trends (Bottom Left)

---

# Methodology: Pipeline

- **Threshold**: Raw signal statistics.
- **Random Forest**: Handcrafted time-domain features.
- **CNN**: 3x `Conv1d` → `BN` → `ReLU` → `Pool`.

---

# Methodology: Diagrams

![h:150](../out/plots/threshold.png)
![h:150](../out/plots/randomforest.png)
![h:150](../out/plots/cnn.png)

---

# Experimental Design

| Parameter       | Random Forest | CNN           |
| --------------- | ------------- | ------------- |
| Trees / filters | 100           | [32, 64, 128] |
| Kernel sizes    | -             | [5, 5, 3]     |
| Learning rate   | -             | $10^{-3}$     |
| Batch size      | -             | 64            |
| Epochs          | -             | 50            |
| Dropout         | -             | 0.3           |

---

![bg right:50%](../out/plots/accuracy_by_fold.png)
![bg right:50%](../out/plots/average_accuracy.png)

# Results

- **CNN** outperforms baselines.
- **Stability**: Performance >80% across all patient folds.
- **Capture**: Captures temporal patterns missed by RF.

---

# Conclusion

- End-to-end CNN is the superior architecture for patient-level CV.
- Future work: Multichannel spatial context.

---

# Thank You

**Questions?**
Refer to `latex/main.pdf` for the full technical story.
