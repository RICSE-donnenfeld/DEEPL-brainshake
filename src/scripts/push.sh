#!/bin/bash
set -euo pipefail

if [ ! -f pyproject.toml ]; then
  echo "Run at project root to push and sync to cluster" >&2
  exit 1
fi

rsync -avz \
  --exclude='HelicoDataSet/' \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".mypy_cache" \
  --exclude ".Trash-1000" \
  --exclude "__pycache__" \
  --exclude ".ruff_cache" \
  --exclude ".pytest_cache" \
  --exclude ".mypy_cache" \
  --exclude ".gitignore" \
  -e "ssh -p 55022" \
  . ricse04@158.109.75.52:/hhome/ricse04/brainshake/
