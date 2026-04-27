#!/bin/bash
#SBATCH -J bs_cnn
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/brainshake
#SBATCH -t 0-04:00
#SBATCH -p dcca40
#SBATCH --mem 16000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail
source .venv/bin/activate
mkdir -p slurm_io

echo "=== CNN split-level grid — $(date) ==="

for LEVEL in patient window seizure; do
	echo ">>> CNN level=${LEVEL}"
	brainshake run evaluate-cnn -- \
		--epochs 2 \
		--n-splits 5 \
		--random-state 2026 \
		--level "${LEVEL}" \
		--suffix "_${LEVEL}"
done

echo "=== CNN grid done — $(date) ==="
