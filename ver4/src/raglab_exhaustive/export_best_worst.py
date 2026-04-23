from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .io_utils import read_jsonl, write_json, write_jsonl


def _load_run_meta_map(root: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for run_dir in sorted((root / "artifacts" / "runs").glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out[meta["run_id"]] = meta
    return out


def _load_ans_ret_maps(root: Path):
    ans = {}
    ret = {}
    for run_dir in sorted((root / "artifacts" / "runs").glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        run_id = json.loads(meta_path.read_text(encoding="utf-8"))["run_id"]
        for r in read_jsonl(run_dir / "answers.jsonl"):
            ans[(run_id, r["question_id"])] = r
        for r in read_jsonl(run_dir / "retrievals.jsonl"):
            ret[(run_id, r["question_id"])] = r
    return ans, ret


def _stage_signature(row: pd.Series) -> str:
    return "|".join(
        [
            row["loader_name"],
            row["cleaning_name"],
            row["chunking_name"],
            row["representation_name"],
            row["retrieval_name"],
            row["post_retrieval_name"],
            row["generation_name"],
        ]
    )


def _leaderboards(df: pd.DataFrame) -> Dict[str, Any]:
    by_run = df.groupby("run_id").agg(
        retrieval_score=("hit@5", "mean"),
        answer_score=("token_f1", "mean"),
        judge_score=("llm_judge_overall", "mean"),
        abstention_score=("abstention_correct", "mean"),
        latency=("elapsed_sec", "mean"),
        overall=("overall_score", "mean"),
    )

    by_run["judge_score"] = by_run["judge_score"].fillna(0.0)
    by_run["efficiency"] = by_run["overall"] / by_run["latency"].replace(0, 1e-6)

    return {
        "retrieval_best_config": by_run.sort_values("retrieval_score", ascending=False).head(1).reset_index().to_dict("records")[0],
        "answer_metric_best_config": by_run.sort_values("answer_score", ascending=False).head(1).reset_index().to_dict("records")[0],
        "judge_best_config": by_run.sort_values("judge_score", ascending=False).head(1).reset_index().to_dict("records")[0],
        "abstention_best_config": by_run.sort_values("abstention_score", ascending=False).head(1).reset_index().to_dict("records")[0],
        "latency_efficient_best_config": by_run.sort_values("efficiency", ascending=False).head(1).reset_index().to_dict("records")[0],
    }


def _question_level_best_worst(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, grp in df.groupby("question_id"):
        g = grp.sort_values("overall_score", ascending=False)
        best = g.iloc[0]
        worst = g.iloc[-1]

        gr = grp.sort_values(["hit@5", "mrr"], ascending=False)
        best_ret = gr.iloc[0]
        worst_ret = gr.iloc[-1]

        rows.append(
            {
                "question_id": qid,
                "best_run_id": best["run_id"],
                "worst_run_id": worst["run_id"],
                "best_retrieval_run_id": best_ret["run_id"],
                "worst_retrieval_run_id": worst_ret["run_id"],
                "best_overall": best["overall_score"],
                "worst_overall": worst["overall_score"],
            }
        )
    return pd.DataFrame(rows)


def _question_type_summaries(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    df = df.copy()
    df["stage_signature"] = df.apply(_stage_signature, axis=1)

    for qtype, grp in df.groupby("question_type"):
        by_sig = grp.groupby("stage_signature")["overall_score"].mean().sort_values(ascending=False)
        best_sig = by_sig.index[0]
        worst_sig = by_sig.index[-1]

        examples = grp.sort_values("overall_score", ascending=False).head(3)
        out.append(
            {
                "question_type": qtype,
                "best_config": best_sig,
                "worst_config": worst_sig,
                "representative_examples": examples[["question_id", "run_id", "answer_text", "overall_score"]].to_dict("records"),
            }
        )
    return out


def _make_examples(
    *,
    root: Path,
    df: pd.DataFrame,
    ans_map: Dict[Tuple[str, str], Dict[str, Any]],
    ret_map: Dict[Tuple[str, str], Dict[str, Any]],
    qlevel: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    best_rows: List[Dict[str, Any]] = []
    worst_rows: List[Dict[str, Any]] = []

    for _, row in qlevel.iterrows():
        qid = row["question_id"]
        best_run = row["best_run_id"]
        worst_run = row["worst_run_id"]

        best = df[(df["question_id"] == qid) & (df["run_id"] == best_run)].iloc[0]
        worst = df[(df["question_id"] == qid) & (df["run_id"] == worst_run)].iloc[0]

        best_ans = ans_map.get((best_run, qid), {})
        worst_ans = ans_map.get((worst_run, qid), {})

        best_ret = ret_map.get((best_run, qid), {})
        worst_ret = ret_map.get((worst_run, qid), {})

        rec = {
            "question_id": qid,
            "question": best_ans.get("prompt_text") or worst_ans.get("prompt_text"),
            "gold_reference": {
                "question_type": best.get("question_type"),
            },
            "best_answer": best_ans.get("answer_text", best.get("answer_text")),
            "worst_answer": worst_ans.get("answer_text", worst.get("answer_text")),
            "best_config": best_run,
            "worst_config": worst_run,
            "retrieval_evidence_best": best_ret.get("top_k", []),
            "retrieval_evidence_worst": worst_ret.get("top_k", []),
            "judge_summary": {
                "best": best.get("llm_judge_comment", ""),
                "worst": worst.get("llm_judge_comment", ""),
            },
            "selection_reason": "overall_score and judge/groundedness differences",
        }
        best_rows.append(rec)
        worst_rows.append(rec)

    best_rows = sorted(best_rows, key=lambda x: x["question_id"])[:120]
    worst_rows = sorted(worst_rows, key=lambda x: x["question_id"])[:120]
    return best_rows, worst_rows


def _render_presentation_cases(best_rows: List[Dict[str, Any]], pairwise: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Presentation Cases")
    lines.append("")
    lines.append("## Best/Worst Highlights")

    for rec in best_rows[:5]:
        lines.append(f"### {rec['question_id']}")
        lines.append(f"- Question: {rec['question']}")
        lines.append(f"- Best run: {rec['best_config']}")
        lines.append(f"- Worst run: {rec['worst_config']}")
        lines.append(f"- Best answer: {rec['best_answer']}")
        lines.append(f"- Worst answer: {rec['worst_answer']}")
        lines.append(f"- Why selected: {rec['selection_reason']}")
        lines.append("")

    lines.append("## Pairwise Judge Highlights")
    for rec in pairwise[:3]:
        lines.append(f"- [{rec.get('scenario')}] qid={rec.get('question_id')} preferred={rec.get('preferred_run_id')} reason={rec.get('pairwise_reason')}")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    df = pd.read_csv(root / "results" / "experiment_results.csv")

    leaderboards = _leaderboards(df)
    qlevel = _question_level_best_worst(df)
    qtype = _question_type_summaries(df)

    ans_map, ret_map = _load_ans_ret_maps(root)
    best_rows, worst_rows = _make_examples(root=root, df=df, ans_map=ans_map, ret_map=ret_map, qlevel=qlevel)

    pairwise_path = root / "results" / "pairwise_scores.jsonl"
    pairwise_rows = read_jsonl(pairwise_path) if pairwise_path.exists() else []

    write_json(root / "results" / "leaderboards" / "global_leaderboard.json", leaderboards)
    qlevel.to_csv(root / "results" / "leaderboards" / "question_level_best_worst.csv", index=False)
    write_json(root / "results" / "leaderboards" / "question_type_best_worst.json", {"rows": qtype})

    write_jsonl(root / "results" / "best_examples.jsonl", best_rows)
    write_jsonl(root / "results" / "worst_examples.jsonl", worst_rows)
    write_jsonl(root / "results" / "pairwise_examples.jsonl", pairwise_rows)

    md = _render_presentation_cases(best_rows, pairwise_rows)
    (root / "results" / "presentation_cases.md").write_text(md, encoding="utf-8")

    print(
        json.dumps(
            {
                "leaderboard": str(root / "results" / "leaderboards" / "global_leaderboard.json"),
                "best_examples": str(root / "results" / "best_examples.jsonl"),
                "worst_examples": str(root / "results" / "worst_examples.jsonl"),
                "pairwise_examples": str(root / "results" / "pairwise_examples.jsonl"),
                "presentation_cases": str(root / "results" / "presentation_cases.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
