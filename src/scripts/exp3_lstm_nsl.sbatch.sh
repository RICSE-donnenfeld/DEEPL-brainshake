#!/bin/bash
#SBATCH -J bs_lstm_nsl
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/brainshake
#SBATCH -t 0-08:00
#SBATCH -p dcca40
#SBATCH --mem 16000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail
source .venv/bin/activate
mkdir -p slurm_io

echo "=== LSTM non-seizure-length sweep — $(date) ==="

for NSL in 5 10 20 50; do
	echo ">>> LSTM seizure pool=std nsl=${NSL}"
	brainshake run evaluate-lstm -- \
		--epochs 2 \
		--n-splits 5 \
		--random-state 2026 \
		--level seizure \
		--pool std \
		--context 20 \
		--non-seizure-len "${NSL}" \
		--suffix "_std_seizure_nsl${NSL}"
done

echo "=== LSTM nsl sweep done — $(date) ==="
