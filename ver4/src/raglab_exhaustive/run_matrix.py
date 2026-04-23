from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_matrix_config, load_yaml
from .io_utils import write_json
from .orchestration import run_full_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--matrix", default="configs/experiment_matrix.json")
    ap.add_argument("--prompts", default="configs/generation_prompts.yaml")
    ap.add_argument("--weights", default="configs/scoring_weights.yaml")
    ap.add_argument("--dataset", default=None, help="manual|auto|stress only")
    ap.add_argument("--rerun-failed-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    matrix = load_matrix_config((root / args.matrix).resolve())
    prompts = load_yaml((root / args.prompts).resolve())
    weights = load_yaml((root / args.weights).resolve()).get("weights", {})

    out = run_full_matrix(
        root=root,
        pdf_path=Path(args.pdf),
        matrix=matrix,
        prompts=prompts,
        weights=weights,
        only_dataset=args.dataset,
        rerun_failed_only=bool(args.rerun_failed_only),
    )

    write_json(root / "results" / "summaries" / "run_matrix_summary.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
