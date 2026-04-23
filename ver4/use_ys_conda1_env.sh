#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="/home/a202192020/miniconda3"
ENV_NAME="ys_conda1_env"

if [[ ! -f "$CONDA_BASE/bin/activate" ]]; then
  echo "[ERROR] conda activate script not found: $CONDA_BASE/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/bin/activate" "$ENV_NAME"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  echo "[ERROR] failed to activate conda env: $ENV_NAME" >&2
  exit 1
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
