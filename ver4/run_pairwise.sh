#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/use_ys_conda1_env.sh"
"$PYTHON_BIN" -m src.raglab_exhaustive.compare_pairwise \
  --root "$ROOT_DIR" \
  --matrix configs/experiment_matrix.json \
  --judge-prompts configs/judge_prompts.yaml
