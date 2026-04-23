from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io_utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    df = pd.read_csv(root / "results" / "experiment_results.csv")

    retrieval_maps = {}
    for run_dir in sorted((root / "artifacts" / "runs").glob("run_*")):
        for row in read_jsonl(run_dir / "retrievals.jsonl"):
            retrieval_maps[(run_dir.name, row.get("question_id"))] = row

    rows = []
    for _, r in df.iterrows():
        run_key = str(r["run_id"])
        ret = retrieval_maps.get((run_key, r["question_id"]), {})
        rows.append(
            {
                "question_id": r["question_id"],
                "prompt_text": "",
                "run_id": r["run_id"],
                "loader": r["loader_name"],
                "cleaning": r["cleaning_name"],
                "chunking": r["chunking_name"],
                "representation": r["representation_name"],
                "retrieval": r["retrieval_name"],
                "post_retrieval": r["post_retrieval_name"],
                "generation": r["generation_name"],
                "answer_text": r["answer_text"],
                "top_evidence": str(ret.get("top_k", []))[:1200],
                "human_score_correctness": "",
                "human_score_groundedness": "",
                "human_score_completeness": "",
                "human_score_readability": "",
                "human_comment": "",
            }
        )

    out_csv = root / "eval" / "human_eval_template.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"{out_csv}")


if __name__ == "__main__":
    main()
