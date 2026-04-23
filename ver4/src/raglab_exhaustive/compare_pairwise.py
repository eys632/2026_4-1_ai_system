from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import load_matrix_config, load_yaml
from .io_utils import read_jsonl, write_jsonl
from .llm_backends import LLMRequest, get_backend


JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_BLOCK_RE.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _load_answer_map(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for run_dir in sorted((root / "artifacts" / "runs").glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta.get("run_id")
        for row in read_jsonl(run_dir / "answers.jsonl"):
            qid = row.get("question_id")
            if run_id and qid:
                out[(run_id, qid)] = row
    return out


def _load_retrieval_map(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for run_dir in sorted((root / "artifacts" / "runs").glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta.get("run_id")
        for row in read_jsonl(run_dir / "retrievals.jsonl"):
            qid = row.get("question_id")
            if run_id and qid:
                out[(run_id, qid)] = row
    return out


def _build_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    pair_defs: List[Tuple[str, str, str]] = []

    run_overall = df.groupby("run_id")["overall_score"].mean().sort_values(ascending=False)
    if len(run_overall) >= 2:
        pair_defs.append(("overall_best_vs_worst", run_overall.index[0], run_overall.index[-1]))

    run_retrieval = df.groupby("run_id")["hit@5"].mean().sort_values(ascending=False)
    run_answer = (
        df.assign(answer_combo=(df["token_f1"] + df["answer_similarity"] + df["exact_match"]) / 3.0)
        .groupby("run_id")["answer_combo"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(run_retrieval) >= 1 and len(run_answer) >= 1:
        rb = run_retrieval.index[0]
        ab = run_answer.index[0]
        if rb != ab:
            pair_defs.append(("retrieval_best_vs_answer_best", rb, ab))

    return pair_defs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--matrix", default="configs/experiment_matrix.json")
    ap.add_argument("--judge-prompts", default="configs/judge_prompts.yaml")
    ap.add_argument("--max-per-pair", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    matrix = load_matrix_config(root / args.matrix)
    judge_prompts = load_yaml(root / args.judge_prompts)
    template = judge_prompts["pairwise_prompt"]["template"]

    df = pd.read_csv(root / "results" / "experiment_results.csv")
    backend = get_backend(matrix.judge_backend)

    ans_map = _load_answer_map(root)
    ret_map = _load_retrieval_map(root)

    rng = random.Random(args.seed)

    pair_defs = _build_pairs(df)
    out_rows: List[Dict[str, Any]] = []

    for scenario, run_a, run_b in pair_defs:
        qa = set(df[df["run_id"] == run_a]["question_id"].tolist())
        qb = set(df[df["run_id"] == run_b]["question_id"].tolist())
        shared = sorted(qa & qb)
        if len(shared) > args.max_per_pair:
            shared = rng.sample(shared, args.max_per_pair)

        for qid in shared:
            a = ans_map.get((run_a, qid), {})
            b = ans_map.get((run_b, qid), {})
            ra = ret_map.get((run_a, qid), {})
            rb = ret_map.get((run_b, qid), {})
            question = a.get("prompt_text") or b.get("prompt_text")

            prompt = template.format(
                question=question,
                answer_a=a.get("answer_text", ""),
                answer_b=b.get("answer_text", ""),
                evidence_a=json.dumps(ra.get("top_k", []), ensure_ascii=False),
                evidence_b=json.dumps(rb.get("top_k", []), ensure_ascii=False),
            )
            raw = backend.generate(LLMRequest(prompt=prompt, temperature=0.0, top_p=0.1, max_tokens=280))
            parsed = _parse_json(raw)

            pref = parsed.get("preferred", "A")
            preferred_run_id = run_a if str(pref).upper() == "A" else run_b
            row = {
                "scenario": scenario,
                "question_id": qid,
                "question": question,
                "run_a": run_a,
                "run_b": run_b,
                "preferred": pref,
                "preferred_run_id": preferred_run_id,
                "preferred_answer": a.get("answer_text", "") if pref == "A" else b.get("answer_text", ""),
                "pairwise_reason": parsed.get("reason", ""),
                "pairwise_dimension_scores": parsed.get("dimension_scores", {}),
                "raw_output": raw,
            }
            out_rows.append(row)

    out_path = root / "results" / "pairwise_scores.jsonl"
    write_jsonl(out_path, out_rows)

    # Also mirror pairwise output in each involved run folder for full traceability.
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for row in out_rows:
        for run_id in [row["run_a"], row["run_b"]]:
            by_run.setdefault(run_id, []).append(row)
    for run_id, rows in by_run.items():
        write_jsonl(root / "artifacts" / "runs" / run_id / "pairwise_scores.jsonl", rows)

    print(json.dumps({"n_pairwise": len(out_rows), "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
