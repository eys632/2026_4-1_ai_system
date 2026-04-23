from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .config import load_matrix_config, load_yaml
from .io_utils import read_jsonl, write_jsonl
from .llm_backends import LLMRequest, get_backend


JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _load_qmap(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    out = {}
    for r in rows:
        qid = r.get("question_id") or r.get("id")
        if not qid:
            continue
        out[str(qid)] = r
    return out


def _score_default() -> Dict[str, Any]:
    return {
        "intent_fulfillment": 1,
        "correctness": 1,
        "groundedness": 1,
        "completeness": 1,
        "readability": 1,
        "abstention_safety": 1,
        "overall_preference": 1,
        "comment": "judge parse fallback",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--matrix", default="configs/experiment_matrix.json")
    ap.add_argument("--judge-prompts", default="configs/judge_prompts.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    matrix = load_matrix_config(root / args.matrix)
    judge_cfg = load_yaml(root / args.judge_prompts)
    judge_template = judge_cfg["judge_answer_prompt"]["template"]

    backend = get_backend(matrix.judge_backend)

    runs_dir = root / "artifacts" / "runs"
    logs_path = root / "eval" / "llm_judge_logs" / f"judge_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    all_logs: List[Dict[str, Any]] = []

    for run_dir in sorted(runs_dir.glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        ans_rows = read_jsonl(run_dir / "answers.jsonl")
        ret_map = {r["question_id"]: r for r in read_jsonl(run_dir / "retrievals.jsonl")}
        judge_rows_existing = {r["question_id"]: r for r in read_jsonl(run_dir / "judge_scores.jsonl")}

        dataset_path = Path(meta.get("dataset_path", ""))
        qmap = _load_qmap(dataset_path) if dataset_path.exists() else {}

        out_rows: List[Dict[str, Any]] = []
        for ans in tqdm(ans_rows, desc=f"judge:{run_dir.name}", leave=False):
            qid = ans["question_id"]
            if qid in judge_rows_existing and not args.force:
                out_rows.append(judge_rows_existing[qid])
                continue

            q = qmap.get(qid, {})
            ret = ret_map.get(qid, {})

            gold_ref = {
                "gold_service": q.get("gold_service") or (q.get("gold") or {}).get("service"),
                "gold_field": q.get("gold_field") or (q.get("gold") or {}).get("field"),
                "gold_answer": q.get("gold_answer", ""),
                "gold_page": q.get("gold_page"),
                "requires_abstention": q.get("requires_abstention", False),
            }
            prompt = judge_template.format(
                question=ans.get("prompt_text", ""),
                gold_reference=json.dumps(gold_ref, ensure_ascii=False),
                generated_answer=ans.get("answer_text", ""),
                retrieved_evidence=json.dumps(ret.get("top_k", []), ensure_ascii=False),
                baseline_answer="",
            )

            raw = backend.generate(LLMRequest(prompt=prompt, temperature=0.0, top_p=0.1, max_tokens=350))
            parsed = _parse_json(raw)
            if not parsed:
                parsed = _score_default()

            row = {
                "question_id": qid,
                "intent_fulfillment": parsed.get("intent_fulfillment", 1),
                "correctness": parsed.get("correctness", 1),
                "groundedness": parsed.get("groundedness", 1),
                "completeness": parsed.get("completeness", 1),
                "readability": parsed.get("readability", 1),
                "abstention_safety": parsed.get("abstention_safety", 1),
                "overall_preference": parsed.get("overall_preference", 1),
                "comment": parsed.get("comment", ""),
            }
            out_rows.append(row)

            all_logs.append(
                {
                    "run_id": meta.get("run_id"),
                    "question_id": qid,
                    "prompt": prompt,
                    "raw_output": raw,
                    "parsed": row,
                }
            )

        write_jsonl(run_dir / "judge_scores.jsonl", out_rows)

    write_jsonl(logs_path, all_logs)
    print(json.dumps({"judge_logs": str(logs_path), "n_logs": len(all_logs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
