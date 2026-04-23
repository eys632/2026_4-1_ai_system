#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/use_ys_conda1_env.sh"
"$PYTHON_BIN" -m src.raglab_exhaustive.export_best_worst --root "$ROOT_DIR"
"$PYTHON_BIN" -m src.raglab_exhaustive.export_human_eval --root "$ROOT_DIR"
"$PYTHON_BIN" -m src.raglab_exhaustive.final_report --root "$ROOT_DIR"
