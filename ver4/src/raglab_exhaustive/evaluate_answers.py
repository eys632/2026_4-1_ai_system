from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import load_yaml
from .io_utils import read_jsonl
from .metrics import weighted_overall_score
from .orchestration import RESULT_COLUMNS


def _load_question_map(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        qid = r.get("question_id") or r.get("id")
        if not qid:
            continue
        out[str(qid)] = r
    return out


def _collect_rows(root: Path, weights: Dict[str, float]) -> List[Dict[str, Any]]:
    runs_dir = root / "artifacts" / "runs"
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(runs_dir.glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        stage = meta.get("stage_config", {})
        dataset_path = Path(meta.get("dataset_path", ""))
        qmap = _load_question_map(dataset_path) if dataset_path.exists() else {}

        ans_rows = {r["question_id"]: r for r in read_jsonl(run_dir / "answers.jsonl")}
        auto_rows = {r["question_id"]: r for r in read_jsonl(run_dir / "auto_scores.jsonl")}
        judge_rows = {r["question_id"]: r for r in read_jsonl(run_dir / "judge_scores.jsonl")}

        for qid, ans in ans_rows.items():
            auto = auto_rows.get(qid, {})
            judge = judge_rows.get(qid, {})
            q = qmap.get(qid, {})

            row = {
                "run_id": meta.get("run_id"),
                "dataset_name": meta.get("dataset_name"),
                "question_id": qid,
                "question_type": q.get("question_type", "unknown"),
                "loader_name": stage.get("loader"),
                "cleaning_name": stage.get("cleaning"),
                "chunking_name": stage.get("chunking"),
                "representation_name": stage.get("representation"),
                "retrieval_name": stage.get("retrieval"),
                "post_retrieval_name": stage.get("post_retrieval"),
                "generation_name": stage.get("generation"),
                "hyperparams_json": json.dumps(meta.get("hyperparams", {}), ensure_ascii=False),
                "answer_text": ans.get("answer_text", ""),
                "hit@1": auto.get("hit@1", 0.0),
                "hit@3": auto.get("hit@3", 0.0),
                "hit@5": auto.get("hit@5", 0.0),
                "mrr": auto.get("mrr", 0.0),
                "ndcg@5": auto.get("ndcg@5", 0.0),
                "recall@5": auto.get("recall@5", 0.0),
                "exact_match": auto.get("exact_match", 0.0),
                "token_f1": auto.get("token_f1", 0.0),
                "answer_similarity": auto.get("answer_similarity", 0.0),
                "citation_overlap": auto.get("citation_overlap", 0.0),
                "groundedness_heuristic": auto.get("groundedness_heuristic", 0.0),
                "abstention_correct": auto.get("abstention_correct", 0.0),
                "llm_judge_intent": judge.get("intent_fulfillment"),
                "llm_judge_correctness": judge.get("correctness"),
                "llm_judge_groundedness": judge.get("groundedness"),
                "llm_judge_completeness": judge.get("completeness"),
                "llm_judge_readability": judge.get("readability"),
                "llm_judge_abstention": judge.get("abstention_safety"),
                "llm_judge_overall": judge.get("overall_preference"),
                "llm_judge_comment": judge.get("comment", ""),
                "elapsed_sec": ans.get("generation_latency", 0.0),
                "success": True,
            }
            row["overall_score"] = weighted_overall_score(row, weights)
            rows.append(row)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--weights", default="configs/scoring_weights.yaml")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    weights = load_yaml(root / args.weights).get("weights", {})

    rows = _collect_rows(root, weights)

    out_csv = root / "results" / "experiment_results.csv"
    out_parquet = root / "results" / "experiment_results.parquet"

    if rows:
        df = pd.DataFrame(rows)
        for col in RESULT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[RESULT_COLUMNS]
        df.to_csv(out_csv, index=False)
        try:
            df.to_parquet(out_parquet, index=False)
        except Exception:
            pass

    print(json.dumps({"n_rows": len(rows), "csv": str(out_csv), "parquet": str(out_parquet)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
