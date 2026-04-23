from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .io_utils import read_jsonl


def _safe_row(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if df.empty:
        return {}
    return df.sort_values(col, ascending=False).iloc[0].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    df = pd.read_csv(root / "results" / "experiment_results.csv")

    by_run = df.groupby("run_id").agg(
        retrieval=("hit@5", "mean"),
        answer=("token_f1", "mean"),
        judge=("llm_judge_overall", "mean"),
        abstention=("abstention_correct", "mean"),
        latency=("elapsed_sec", "mean"),
        overall=("overall_score", "mean"),
    ).reset_index()
    by_run["judge"] = by_run["judge"].fillna(0.0)
    by_run["efficiency"] = by_run["overall"] / by_run["latency"].replace(0, 1e-6)

    retrieval_best = _safe_row(by_run, "retrieval")
    answer_best = _safe_row(by_run, "answer")
    judge_best = _safe_row(by_run, "judge")
    abstention_best = _safe_row(by_run, "abstention")
    efficiency_best = _safe_row(by_run, "efficiency")

    stage_effects = {}
    for stage_col in [
        "loader_name",
        "cleaning_name",
        "chunking_name",
        "representation_name",
        "retrieval_name",
        "post_retrieval_name",
        "generation_name",
    ]:
        stage_effects[stage_col] = (
            df.groupby(stage_col)["overall_score"].mean().sort_values(ascending=False).reset_index().to_dict("records")
        )

    pairwise_examples = read_jsonl(root / "results" / "pairwise_examples.jsonl")
    best_examples = read_jsonl(root / "results" / "best_examples.jsonl")
    worst_examples = read_jsonl(root / "results" / "worst_examples.jsonl")

    lines = []
    lines.append("# Final Report")
    lines.append("")
    lines.append("## Top Configurations")
    lines.append(f"- Retrieval best: {retrieval_best}")
    lines.append(f"- Answer best: {answer_best}")
    lines.append(f"- Judge best: {judge_best}")
    lines.append(f"- Abstention best: {abstention_best}")
    lines.append(f"- Latency-efficient best: {efficiency_best}")
    lines.append("")

    lines.append("## Ablation Summary")
    for k, v in stage_effects.items():
        lines.append(f"- {k}: {v[:3]}")
    lines.append("")

    lines.append("## Best/Worst Case Analysis")
    for rec in best_examples[:5]:
        lines.append(f"- BEST {rec.get('question_id')}: best={rec.get('best_config')} worst={rec.get('worst_config')}")
    for rec in worst_examples[:5]:
        lines.append(f"- WORST {rec.get('question_id')}: best={rec.get('best_config')} worst={rec.get('worst_config')}")
    lines.append("")

    lines.append("## Pairwise Analysis")
    for rec in pairwise_examples[:5]:
        lines.append(
            f"- {rec.get('scenario')} q={rec.get('question_id')} preferred={rec.get('preferred_run_id')} reason={rec.get('pairwise_reason')}"
        )

    report_md = "\n".join(lines) + "\n"
    (root / "docs" / "final_report.md").write_text(report_md, encoding="utf-8")
    (root / "results" / "summaries" / "final_report.md").write_text(report_md, encoding="utf-8")

    summary_json = {
        "retrieval_best": retrieval_best,
        "answer_best": answer_best,
        "judge_best": judge_best,
        "abstention_best": abstention_best,
        "efficiency_best": efficiency_best,
        "stage_effects": stage_effects,
    }
    (root / "results" / "summaries" / "final_report_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({"report": str(root / 'docs' / 'final_report.md')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
