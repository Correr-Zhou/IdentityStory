#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

python main.py \
  --json_dir bench/ConsiStory-Human-single \
  --output_dir results/identitystory_single \
  --max_stories 1 \
  --prompt_indices 0 \
  --device cuda \
  --suffix identitystory_single
