#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PDF_PATH="${1:-/data2/a202192020/4-1/ai_sys/2026_4-1_ai_system/2025 나에게  힘이 되는 복지서비스.pdf}"

cd "$ROOT_DIR"
source "$ROOT_DIR/use_ys_conda1_env.sh"
"$PYTHON_BIN" -m src.raglab_exhaustive.resume_failed \
  --root "$ROOT_DIR" \
  --pdf "$PDF_PATH" \
  --matrix configs/experiment_matrix.json \
  --prompts configs/generation_prompts.yaml \
  --weights configs/scoring_weights.yaml
