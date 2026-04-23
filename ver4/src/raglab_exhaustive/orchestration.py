from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from .config import MatrixConfig, StageConfig, stable_hash
from .io_utils import append_csv, append_jsonl, ensure_dir, read_jsonl, setup_logger, try_write_parquet, write_json, write_jsonl
from .llm_backends import LLMRequest, get_backend
from .metrics import evaluate_answer, retrieval_metrics_from_hits, weighted_overall_score
from .questions import ensure_question_sets
from .stages import (
    Chunk,
    RetrievalItem,
    build_context,
    build_generation_prompt,
    build_representation,
    chunk_documents,
    clean_pages,
    extract_citations,
    extract_pages,
    infer_sections,
    load_chunks_jsonl,
    normalize_answer,
    post_retrieve,
    retrieve,
    save_chunks_jsonl,
    save_pages_jsonl,
    save_sections_jsonl,
)


RESULT_COLUMNS = [
    "run_id",
    "dataset_name",
    "question_id",
    "question_type",
    "loader_name",
    "cleaning_name",
    "chunking_name",
    "representation_name",
    "retrieval_name",
    "post_retrieval_name",
    "generation_name",
    "hyperparams_json",
    "answer_text",
    "hit@1",
    "hit@3",
    "hit@5",
    "mrr",
    "ndcg@5",
    "recall@5",
    "exact_match",
    "token_f1",
    "answer_similarity",
    "citation_overlap",
    "groundedness_heuristic",
    "abstention_correct",
    "llm_judge_intent",
    "llm_judge_correctness",
    "llm_judge_groundedness",
    "llm_judge_completeness",
    "llm_judge_readability",
    "llm_judge_abstention",
    "llm_judge_overall",
    "llm_judge_comment",
    "overall_score",
    "elapsed_sec",
    "success",
]


@dataclass
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def builds(self) -> Path:
        return self.artifacts / "builds"

    @property
    def runs(self) -> Path:
        return self.artifacts / "runs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"


@dataclass
class BuildResult:
    build_id: str
    build_dir: Path
    chunks: List[Chunk]
    sections_path: Path
    build_meta: Dict[str, Any]


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    ok: bool
    error: Optional[str]
    n_answers: int


def _default_hparams(matrix: MatrixConfig) -> Dict[str, Any]:
    d = dict(matrix.defaults)
    d.setdefault("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    d.setdefault("embed_device", "cuda")
    d.setdefault("top_k", matrix.top_k)
    d.setdefault("max_context_chars", 9000)
    return d


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out = []
    for r in rows:
        out.append(
            {
                "question_id": r.get("question_id") or r.get("id"),
                "prompt_text": r.get("prompt_text") or r.get("question"),
                "question_type": r.get("question_type") or "unknown",
                "answer_type": r.get("answer_type") or "fact",
                "gold_service": r.get("gold_service") or (r.get("gold") or {}).get("service"),
                "gold_field": r.get("gold_field") or (r.get("gold") or {}).get("field"),
                "gold_answer": r.get("gold_answer") or "",
                "gold_page": r.get("gold_page"),
                "gold_evidence_chunk_ids": r.get("gold_evidence_chunk_ids") or [],
                "requires_abstention": bool(r.get("requires_abstention", False)),
                "set_name": r.get("set_name") or "manual",
            }
        )
    return [r for r in out if r.get("question_id") and r.get("prompt_text")]


def _chunk_match_gold(chunk: Chunk, q: Dict[str, Any]) -> bool:
    m = chunk.meta or {}

    g_chunk_ids = {int(x) for x in (q.get("gold_evidence_chunk_ids") or []) if str(x).isdigit()}
    if g_chunk_ids and int(chunk.chunk_id) in g_chunk_ids:
        return True

    g_service = q.get("gold_service")
    if g_service and m.get("service") == g_service:
        g_field = q.get("gold_field")
        if not g_field or m.get("field") == g_field:
            return True

    g_page = q.get("gold_page")
    if g_page and m.get("page_start") and m.get("page_end"):
        if int(m.get("page_start")) <= int(g_page) <= int(m.get("page_end")):
            return True

    return False


def _retrieval_metric_per_question(retrieved: Sequence[RetrievalItem], q: Dict[str, Any], k: int) -> Dict[str, float]:
    hits_at_k: List[int] = []
    first_rank: Optional[int] = None

    for idx, item in enumerate(retrieved[:k], start=1):
        is_hit = 1 if _chunk_match_gold(item.chunk, q) else 0
        hits_at_k.append(is_hit)
        if is_hit and first_rank is None:
            first_rank = idx

    while len(hits_at_k) < k:
        hits_at_k.append(0)

    return retrieval_metrics_from_hits(hits_at_k=hits_at_k, first_hit_rank=first_rank, k=min(k, 5))


def _resolve_prompt_template(prompts: Dict[str, Any], generation_name: str) -> str:
    section = prompts.get("generation_prompts", {})
    if generation_name not in section:
        raise KeyError(f"generation prompt not found: {generation_name}")
    return str(section[generation_name]["template"])


def _ensure_build(paths: Paths, pdf_path: Path, stage: StageConfig, matrix: MatrixConfig, logger) -> BuildResult:
    hparams = _default_hparams(matrix)
    build_hash_payload = {
        "pdf": str(pdf_path),
        "loader": stage.loader,
        "cleaning": stage.cleaning,
        "chunking": stage.chunking,
        "representation": stage.representation,
        "embed_model": hparams["embed_model"],
        "embed_device": hparams["embed_device"],
    }
    build_id = stable_hash(build_hash_payload, "build")
    build_dir = paths.builds / build_id
    ensure_dir(build_dir)

    chunks_path = build_dir / "chunks.jsonl"
    meta_path = build_dir / "build_meta.json"
    sections_path = build_dir / "sections.jsonl"

    if chunks_path.exists() and meta_path.exists():
        chunks = load_chunks_jsonl(chunks_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return BuildResult(build_id=build_id, build_dir=build_dir, chunks=chunks, sections_path=sections_path, build_meta=meta)

    logger.info("[BUILD] %s start", build_id)
    t0 = time.perf_counter()

    raw_pages = extract_pages(pdf_path=pdf_path, loader_name=stage.loader, cache_dir=build_dir)
    cleaned_pages = clean_pages(raw_pages, method=stage.cleaning)
    sections = infer_sections(cleaned_pages)
    chunks = chunk_documents(cleaned_pages, sections, chunking=stage.chunking)

    save_pages_jsonl(cleaned_pages, build_dir / "pages.jsonl")
    save_sections_jsonl(sections, sections_path)
    save_chunks_jsonl(chunks, chunks_path)

    rep_meta = build_representation(
        representation=stage.representation,
        chunks=chunks,
        build_dir=build_dir,
        embed_model=hparams["embed_model"],
        embed_device=hparams["embed_device"],
    )

    elapsed = time.perf_counter() - t0
    build_meta = {
        "build_id": build_id,
        "pdf": str(pdf_path),
        "loader": stage.loader,
        "cleaning": stage.cleaning,
        "chunking": stage.chunking,
        "representation": stage.representation,
        "embed_model": hparams["embed_model"],
        "embed_device": hparams["embed_device"],
        "n_pages": len(cleaned_pages),
        "n_sections": len(sections),
        "n_chunks": len(chunks),
        "elapsed_sec": elapsed,
        **rep_meta,
    }
    write_json(meta_path, build_meta)
    logger.info("[BUILD] %s done (chunks=%d, %.1fs)", build_id, len(chunks), elapsed)
    return BuildResult(build_id=build_id, build_dir=build_dir, chunks=chunks, sections_path=sections_path, build_meta=build_meta)


def _load_generation_backend(matrix: MatrixConfig):
    return get_backend(matrix.generation_backend)


def _load_judge_placeholder() -> Dict[str, Any]:
    return {
        "llm_judge_intent": None,
        "llm_judge_correctness": None,
        "llm_judge_groundedness": None,
        "llm_judge_completeness": None,
        "llm_judge_readability": None,
        "llm_judge_abstention": None,
        "llm_judge_overall": None,
        "llm_judge_comment": "",
    }


def _run_one_question(
    *,
    stage: StageConfig,
    build: BuildResult,
    q: Dict[str, Any],
    matrix: MatrixConfig,
    prompts: Dict[str, Any],
    gen_backend,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[str]]:
    hparams = _default_hparams(matrix)
    qid = q["question_id"]
    prompt_text = q["prompt_text"]
    top_k = int(hparams["top_k"])

    retrieval_scored = retrieve(
        retrieval=stage.retrieval,
        query=prompt_text,
        chunks=build.chunks,
        build_dir=build.build_dir,
        top_k=top_k,
        embed_model=hparams["embed_model"],
        embed_device=hparams["embed_device"],
    )
    if not retrieval_scored:
        return {}, {}, {}, f"retrieval returned empty for retrieval={stage.retrieval}, representation={stage.representation}"

    post = post_retrieve(stage.post_retrieval, prompt_text, retrieval_scored, top_k=top_k)
    context_text = build_context(post, max_chars=int(hparams["max_context_chars"]))

    template = _resolve_prompt_template(prompts, stage.generation)
    prompt = build_generation_prompt(template=template, question=prompt_text, context=context_text)

    t0 = time.perf_counter()
    raw_answer = gen_backend.generate(
        LLMRequest(
            prompt=prompt,
            temperature=float(hparams.get("temperature", 0.0)),
            top_p=float(hparams.get("top_p", 0.9)),
            max_tokens=int(hparams.get("max_tokens", 512)),
        )
    )
    latency = time.perf_counter() - t0

    answer = normalize_answer(raw_answer)
    cited_ranks = extract_citations(answer)
    rank_to_chunk = {r.rank: r.chunk.chunk_id for r in post}
    cited_chunk_ids = [rank_to_chunk[r] for r in cited_ranks if r in rank_to_chunk]

    retrieval_json = {
        "question_id": qid,
        "prompt_text": prompt_text,
        "top_k": [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk.chunk_id,
                "service": (r.chunk.meta or {}).get("service"),
                "field": (r.chunk.meta or {}).get("field"),
                "page_start": (r.chunk.meta or {}).get("page_start"),
                "page_end": (r.chunk.meta or {}).get("page_end"),
            }
            for r in post
        ],
        "context_text": context_text,
        "evidence_chunk_ids": [r.chunk.chunk_id for r in post],
        "evidence_pages": [
            {
                "page_start": (r.chunk.meta or {}).get("page_start"),
                "page_end": (r.chunk.meta or {}).get("page_end"),
            }
            for r in post
        ],
    }

    answer_json = {
        "question_id": qid,
        "prompt_text": prompt_text,
        "answer_text": answer,
        "raw_answer_text": raw_answer,
        "context_text": context_text,
        "top_k_docs": retrieval_json["top_k"],
        "citations": cited_ranks,
        "citation_chunk_ids": cited_chunk_ids,
        "abstained": "문서에서 확인되지 않습니다" in answer,
        "generation_latency": latency,
    }

    ret_scores = _retrieval_metric_per_question(post, q, k=top_k)
    auto_scores = evaluate_answer(
        question={
            "gold_answer": q.get("gold_answer"),
            "requires_abstention": q.get("requires_abstention", False),
        },
        answer_text=answer,
        context_text=context_text,
        citation_chunk_ids=cited_chunk_ids,
        evidence_chunk_ids=q.get("gold_evidence_chunk_ids") or [],
    )

    auto_json = {
        "question_id": qid,
        **ret_scores,
        **auto_scores,
    }

    return answer_json, retrieval_json, auto_json, None


def run_single_config(
    *,
    paths: Paths,
    matrix: MatrixConfig,
    prompts: Dict[str, Any],
    weights: Dict[str, float],
    stage: StageConfig,
    dataset_name: str,
    dataset_path: Path,
    pdf_path: Path,
    rerun_failed_only: bool,
) -> RunResult:
    logger = setup_logger(paths.results / "logs" / "run_matrix.log")

    build = _ensure_build(paths, pdf_path, stage, matrix, logger)

    run_payload = {
        "build_id": build.build_id,
        "dataset_name": dataset_name,
        **stage.to_dict(),
    }
    run_id = stable_hash(run_payload, "run")
    run_dir = paths.runs / run_id
    ensure_dir(run_dir)

    run_meta_path = run_dir / "run_meta.json"
    answers_path = run_dir / "answers.jsonl"
    retrievals_path = run_dir / "retrievals.jsonl"
    auto_scores_path = run_dir / "auto_scores.jsonl"
    errors_path = run_dir / "errors.jsonl"
    checkpoint_path = run_dir / "checkpoint.json"

    existing_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}
    if existing_meta.get("status") == "completed" and not rerun_failed_only:
        return RunResult(run_id=run_id, run_dir=run_dir, ok=True, error=None, n_answers=len(read_jsonl(answers_path)))

    questions = _load_questions(dataset_path)
    if not questions:
        err = f"empty question set: {dataset_path}"
        append_jsonl(errors_path, {"run_id": run_id, "error": err})
        return RunResult(run_id=run_id, run_dir=run_dir, ok=False, error=err, n_answers=0)

    if not answers_path.exists() or not rerun_failed_only:
        write_jsonl(answers_path, [])
        write_jsonl(retrievals_path, [])
        write_jsonl(auto_scores_path, [])
        write_jsonl(errors_path, [])

    done_ids = set()
    if checkpoint_path.exists():
        done_ids = set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("done_question_ids", []))

    gen_backend = _load_generation_backend(matrix)

    run_meta = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "build_id": build.build_id,
        "build_dir": str(build.build_dir),
        "stage_config": stage.to_dict(),
        "hyperparams": _default_hparams(matrix),
        "model_backend": matrix.generation_backend.get("backend", "local_vllm"),
        "model_name_or_path": matrix.generation_backend.get("model_name_or_path"),
        "judge_model_name_or_path": matrix.judge_backend.get("model_name_or_path"),
        "generation_device": matrix.generation_backend.get("device", "cuda"),
        "judge_device": matrix.judge_backend.get("device", "cuda"),
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_meta_path, run_meta)

    result_rows: List[Dict[str, Any]] = []
    ok = True
    error_message: Optional[str] = None

    pbar = tqdm(questions, desc=f"{run_id}:{dataset_name}", leave=False)
    for q in pbar:
        qid = q["question_id"]
        if qid in done_ids:
            continue

        try:
            ans, ret, auto, err = _run_one_question(
                stage=stage,
                build=build,
                q=q,
                matrix=matrix,
                prompts=prompts,
                gen_backend=gen_backend,
            )
            if err:
                append_jsonl(errors_path, {"question_id": qid, "error": err, "trace": ""})
                ok = False
                continue

            append_jsonl(answers_path, ans)
            append_jsonl(retrievals_path, ret)
            append_jsonl(auto_scores_path, auto)

            row = {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "question_id": qid,
                "question_type": q.get("question_type"),
                "loader_name": stage.loader,
                "cleaning_name": stage.cleaning,
                "chunking_name": stage.chunking,
                "representation_name": stage.representation,
                "retrieval_name": stage.retrieval,
                "post_retrieval_name": stage.post_retrieval,
                "generation_name": stage.generation,
                "hyperparams_json": json.dumps(_default_hparams(matrix), ensure_ascii=False),
                "answer_text": ans["answer_text"],
                **auto,
                **_load_judge_placeholder(),
                "elapsed_sec": ans["generation_latency"],
                "success": True,
            }
            row["overall_score"] = weighted_overall_score(row, weights)
            result_rows.append(row)

            done_ids.add(qid)
            write_json(checkpoint_path, {"done_question_ids": sorted(done_ids)})
        except Exception as exc:
            ok = False
            err_text = str(exc)
            error_message = err_text
            append_jsonl(
                errors_path,
                {
                    "question_id": qid,
                    "error": err_text,
                    "trace": traceback.format_exc(),
                },
            )

    if result_rows:
        append_csv(paths.results / "experiment_results.csv", result_rows)
        try_write_parquet(paths.results / "experiment_results.csv", paths.results / "experiment_results.parquet")

    run_meta["status"] = "completed" if ok else "partial_failed"
    run_meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["n_questions"] = len(questions)
    run_meta["n_done"] = len(done_ids)
    write_json(run_meta_path, run_meta)

    append_jsonl(
        paths.results / "summaries" / "experiment_status_log.jsonl",
        {
            "run_id": run_id,
            "dataset": dataset_name,
            "status": run_meta["status"],
            "n_questions": len(questions),
            "n_done": len(done_ids),
            "stage_config": stage.to_dict(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    return RunResult(run_id=run_id, run_dir=run_dir, ok=ok, error=error_message, n_answers=len(done_ids))


def prepare_question_sets(paths: Paths, matrix: MatrixConfig, pdf_path: Path, logger) -> Dict[str, Path]:
    # Use a canonical build to discover service/field structure for auto/stress question generation.
    canonical = StageConfig(
        loader=matrix.stages["loader"][0],
        cleaning=matrix.stages["cleaning"][0],
        chunking="field_aware",
        representation="hybrid_dual",
        retrieval="hybrid",
        post_retrieval="none",
        generation=matrix.stages["generation"][0],
    )
    build = _ensure_build(paths, pdf_path, canonical, matrix, logger)

    sections_rows = read_jsonl(build.sections_path)
    from .stages import Section

    sections = [
        Section(
            section_id=int(r["section_id"]),
            service=r.get("service"),
            field=r.get("field"),
            major_category=r.get("major_category"),
            page_start=int(r.get("page_start", 1)),
            page_end=int(r.get("page_end", 1)),
            text=r.get("text", ""),
        )
        for r in sections_rows
    ]

    return ensure_question_sets(
        out_dir=paths.data / "questions",
        sections=sections,
        seed=int(matrix.seed),
        auto_n=int(matrix.defaults.get("auto_questions", 120)),
    )


def run_full_matrix(
    *,
    root: Path,
    pdf_path: Path,
    matrix: MatrixConfig,
    prompts: Dict[str, Any],
    weights: Dict[str, float],
    only_dataset: Optional[str] = None,
    rerun_failed_only: bool = False,
) -> Dict[str, Any]:
    paths = Paths(root=root)
    logger = setup_logger(paths.results / "logs" / "run_matrix.log")
    ensure_dir(paths.results)
    ensure_dir(paths.runs)
    ensure_dir(paths.builds)

    qsets = prepare_question_sets(paths, matrix, pdf_path, logger)

    combos = list(matrix.combinations())
    logger.info("Total stage combinations: %d", len(combos))

    run_summaries: List[Dict[str, Any]] = []

    for stage in combos:
        for ds_name in matrix.datasets:
            if only_dataset and ds_name != only_dataset:
                continue
            ds_path = qsets.get(ds_name)
            if not ds_path:
                continue

            result = run_single_config(
                paths=paths,
                matrix=matrix,
                prompts=prompts,
                weights=weights,
                stage=stage,
                dataset_name=ds_name,
                dataset_path=ds_path,
                pdf_path=pdf_path,
                rerun_failed_only=rerun_failed_only,
            )
            run_summaries.append(
                {
                    "run_id": result.run_id,
                    "dataset": ds_name,
                    "ok": result.ok,
                    "error": result.error,
                    "n_answers": result.n_answers,
                    **stage.to_dict(),
                }
            )

    write_json(paths.results / "summaries" / "run_summaries.json", {"runs": run_summaries})

    return {
        "n_stage_combinations": len(combos),
        "n_runs": len(run_summaries),
        "n_success": sum(1 for r in run_summaries if r["ok"]),
        "n_failed": sum(1 for r in run_summaries if not r["ok"]),
    }


def rerun_failed_runs(*, root: Path, matrix: MatrixConfig, prompts: Dict[str, Any], weights: Dict[str, float], pdf_path: Path) -> Dict[str, Any]:
    paths = Paths(root=root)
    status_rows = read_jsonl(paths.results / "summaries" / "experiment_status_log.jsonl")
    failed = [r for r in status_rows if r.get("status") in {"partial_failed"}]
    if not failed:
        return {"message": "no failed runs"}

    by_stage: List[StageConfig] = []
    datasets: List[str] = []
    for row in failed:
        st = row.get("stage_config") or {}
        by_stage.append(
            StageConfig(
                loader=st["loader"],
                cleaning=st["cleaning"],
                chunking=st["chunking"],
                representation=st["representation"],
                retrieval=st["retrieval"],
                post_retrieval=st["post_retrieval"],
                generation=st["generation"],
            )
        )
        datasets.append(str(row.get("dataset")))

    qsets = prepare_question_sets(paths, matrix, pdf_path, setup_logger(paths.results / "logs" / "run_matrix.log"))

    retried = 0
    for st, ds in zip(by_stage, datasets):
        run_single_config(
            paths=paths,
            matrix=matrix,
            prompts=prompts,
            weights=weights,
            stage=st,
            dataset_name=ds,
            dataset_path=qsets[ds],
            pdf_path=pdf_path,
            rerun_failed_only=True,
        )
        retried += 1

    return {"retried": retried}
