#!/bin/bash
#SBATCH -J bs_lstm_splits
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/brainshake
#SBATCH -t 0-06:00
#SBATCH -p dcca40
#SBATCH --mem 16000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail
source .venv/bin/activate
mkdir -p slurm_io

echo "=== LSTM split-level + pool grid — $(date) ==="

# --- LSTM × 3 split levels (pool=std, context=20) ---
for LEVEL in patient window seizure; do
	echo ">>> LSTM pool=std level=${LEVEL}"
	brainshake run evaluate-lstm -- \
		--epochs 2 \
		--n-splits 5 \
		--random-state 2026 \
		--level "${LEVEL}" \
		--pool std \
		--context 20 \
		--suffix "_std_${LEVEL}"
done

# --- LSTM pool sweep (seizure-level, context=20) ---
for POOL in mean std mean_std conv_proj; do
	echo ">>> LSTM seizure pool=${POOL}"
	brainshake run evaluate-lstm -- \
		--epochs 2 \
		--n-splits 5 \
		--random-state 2026 \
		--level seizure \
		--pool "${POOL}" \
		--context 20 \
		--suffix "_${POOL}_seizure"
done

echo "=== LSTM split-level + pool grid done — $(date) ==="
