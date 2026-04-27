#!/bin/bash
#SBATCH -J brainshake
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D .
#SBATCH -t 0-02:00
#SBATCH -p dcca40
#SBATCH --mem 16000
#SBATCH --gres gpu:1
#SBATCH -o slurm_io/%x_%u_%j.out
#SBATCH -e slurm_io/%x_%u_%j.err

set -euo pipefail

mkdir -p slurm_io

echo "=== brainshake slurm job start ==="
date
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "job_id=${SLURM_JOB_ID:-<none>}"
echo "job_nodelist=${SLURM_JOB_NODELIST:-<none>}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v micromamba >/dev/null 2>&1; then
	export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
	eval "$(micromamba shell hook -s bash)"
	micromamba activate "${BRAINSHAKE_ENV:-brainshake}"
elif [[ -x "./bin/micromamba" ]]; then
	export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
	eval "$(./bin/micromamba shell hook -s bash)"
	./bin/micromamba activate "${BRAINSHAKE_ENV:-brainshake}"
elif [[ -f ".venv/bin/activate" ]]; then
	# Optional local fallback
	source .venv/bin/activate
else
	echo "ERROR: No micromamba found and no .venv present to activate." >&2
	echo "Hint: install micromamba and create env 'brainshake' (or set BRAINSHAKE_ENV)." >&2
	exit 1
fi

echo "python=$(command -v python)"
python -V

echo "=== nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi
else
	echo "nvidia-smi not found"
fi

echo "=== torch cuda probe ==="
python - <<'PY'
import os

try:
		import torch

		print("torch_version", torch.__version__)
		print("torch_cuda_version", torch.version.cuda)
		print("cuda_available", torch.cuda.is_available())
		if torch.cuda.is_available():
				print("cuda_device_count", torch.cuda.device_count())
				for i in range(torch.cuda.device_count()):
						print(f"cuda_device_{i}", torch.cuda.get_device_name(i))
		else:
				print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
except Exception as e:
		print("torch_probe_error", repr(e))
PY

python -m brainshake compile

echo "=== brainshake slurm job end ==="
date

