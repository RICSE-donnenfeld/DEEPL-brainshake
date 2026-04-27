#!/bin/bash
#SBATCH -J brainshake_exp
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/brainshake
#SBATCH -t 1-00:00
#SBATCH -p dcca40
#SBATCH --mem 32000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail

source .venv/bin/activate

mkdir -p slurm_io

echo "============================================================"
echo "  Brainshake — Full Experiment Grid"
echo "  Started: $(date)"
echo "============================================================"

# ------------------------------------------------------------------
# 1. CNN × 3 split levels
# ------------------------------------------------------------------
for LEVEL in patient window seizure; do
    echo ""
    echo ">>> CNN  level=${LEVEL}"
    brainshake run evaluate-cnn -- \
        --epochs 30 \
        --n-splits 5 \
        --random-state 2026 \
        --level "${LEVEL}" \
        --suffix "_${LEVEL}"
done

# ------------------------------------------------------------------
# 2. LSTM × 3 split levels  (pool=std, default settings)
# ------------------------------------------------------------------
for LEVEL in patient window seizure; do
    echo ""
    echo ">>> LSTM level=${LEVEL} pool=std"
    brainshake run evaluate-lstm -- \
        --epochs 30 \
        --n-splits 5 \
        --random-state 2026 \
        --level "${LEVEL}" \
        --pool std \
        --suffix "_std_${LEVEL}"
done

# ------------------------------------------------------------------
# 3. LSTM pooling sweep  (seizure-level only)
# ------------------------------------------------------------------
for POOL in mean std mean_std none; do
    echo ""
    echo ">>> LSTM seizure pool=${POOL}"
    brainshake run evaluate-lstm -- \
        --epochs 30 \
        --n-splits 5 \
        --random-state 2026 \
        --level seizure \
        --pool "${POOL}" \
        --suffix "_${POOL}_seizure"
done

# ------------------------------------------------------------------
# 4. LSTM hidden-size sweep  (seizure-level, pool=std)
#    NOTE: hidden_size is not a CLI flag yet; modify
#    SeizureLSTM(input_size=...) call in evaluate.py to test
#    64 / 128 / 256 manually, or add --hidden-size to the CLI.
#    Below runs the default hidden_size=128 (already covered
#    in experiment 2). Uncomment and adjust evaluate.py when ready.
# ------------------------------------------------------------------
# for HS in 64 256; do
#     echo ""
#     echo ">>> LSTM seizure pool=std hidden_size=${HS}"
#     brainshake run evaluate-lstm -- \
#         --epochs 30 \
#         --n-splits 5 \
#         --random-state 2026 \
#         --level seizure \
#         --pool std \
#         --suffix "_std_seizure_h${HS}"
# done

# ------------------------------------------------------------------
# 5. LSTM non-seizure-episode length sweep  (seizure-level, pool=std)
# ------------------------------------------------------------------
for NSL in 5 10 20 50; do
    echo ""
    echo ">>> LSTM seizure pool=std non_seizure_len=${NSL}"
    brainshake run evaluate-lstm -- \
        --epochs 30 \
        --n-splits 5 \
        --random-state 2026 \
        --level seizure \
        --pool std \
        --non-seizure-len "${NSL}" \
        --suffix "_std_seizure_nsl${NSL}"
done

echo ""
echo "============================================================"
echo "  All experiments complete: $(date)"
echo "============================================================"