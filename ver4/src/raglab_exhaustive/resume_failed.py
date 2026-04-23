from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_matrix_config, load_yaml
from .orchestration import rerun_failed_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--matrix", default="configs/experiment_matrix.json")
    ap.add_argument("--prompts", default="configs/generation_prompts.yaml")
    ap.add_argument("--weights", default="configs/scoring_weights.yaml")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    matrix = load_matrix_config(root / args.matrix)
    prompts = load_yaml(root / args.prompts)
    weights = load_yaml(root / args.weights).get("weights", {})

    out = rerun_failed_runs(
        root=root,
        matrix=matrix,
        prompts=prompts,
        weights=weights,
        pdf_path=Path(args.pdf),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
