# Epileptic Seizure Detection using EEG

Deep Learning project for detecting epileptic seizures using the CHB-MIT EEG dataset.

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd DEEPL-brainshake
```

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate epilepsy-dl
```

## Project Structure

```
DEEPL-brainshake/
│
├── Architecture/                 # neural network architectures
├── Data Set Description/         # dataset documentation
├── References/                   # research papers
├── Validation and Verification/  # validation methodology
│
├── environment.yml               # conda environment
├── README.md
└── .gitignore
```

## Dataset

The project uses the **CHB-MIT Scalp EEG Database**.

Download it from:

<https://physionet.org/content/chbmit/1.0.0/>

The dataset should be placed in:

```
data/
```

## Goal

Detect epileptic seizures from multi-channel EEG recordings using different deep learning architectures
