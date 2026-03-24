#!/bin/bash
#SBATCH -J brainshake
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /hhome/ricse04/brainshake
#SBATCH -t 0-02:00
#SBATCH -p dcca40
#SBATCH --mem 16000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail

source .venv/bin/activate

python -m src.brainshake.cnn -c train --epochs 10 --kfolds 5 --seed 67 -vvv
