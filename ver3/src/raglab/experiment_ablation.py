import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .rag_lab import (
    PDF_DEFAULT,
    Chunk,
    _read_chunks_jsonl,
    _write_chunks_jsonl,
    build_faiss_index,
    build_tfidf,
    clean_page_text,
    extract_pages_via_pdftotext,
    extract_pages_via_pypdf,
    retrieve_dense,
    retrieve_sparse,
    save_pages_jsonl,
    chunk_fields,
    chunk_fixed,
)


def infer_field(question: str) -> Optional[str]:
    q = (question or "").replace(" ", "")
    if any(k in q for k in ["문의", "연락", "전화", "콜센터", "번호"]):
        return "문의"
    if any(k in q for k in ["신청", "방법", "어떻게", "절차", "제출", "접수"]):
        return "방법"
    if any(k in q for k in ["대상", "자격", "조건", "누가", "누구"]):
        return "대상"
    if any(k in q for k in ["내용", "지원", "혜택", "금액", "얼마", "본인부담"]):
        return "내용"
    return None


def load_questions(path: Path) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs


def _norm_key(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    # 서비스명에 들어간 공백/줄바꿈 차이로 평가가 깨지지 않도록 정규화
    return re.sub(r"\s+", "", s)


def is_hit(chunks: List[Chunk], gold: Dict[str, Any]) -> bool:
    g_service = _norm_key(gold.get("service"))
    g_field = _norm_key(gold.get("field"))
    if not g_service:
        return False

    for ch in chunks:
        m = ch.meta or {}
        if _norm_key(m.get("service")) != g_service:
            continue
        if g_field and _norm_key(m.get("field")) != g_field:
            continue
        return True
    return False


def reciprocal_rank(chunks: List[Chunk], gold: Dict[str, Any]) -> float:
    g_service = _norm_key(gold.get("service"))
    g_field = _norm_key(gold.get("field"))
    for i, ch in enumerate(chunks, start=1):
        m = ch.meta or {}
        if _norm_key(m.get("service")) != g_service:
            continue
        if g_field and _norm_key(m.get("field")) != g_field:
            continue
        return 1.0 / i
    return 0.0


@dataclass(frozen=True)
class BuildConfig:
    loader: str
    layout: bool
    chunking: str

    @property
    def name(self) -> str:
        if self.loader == "pdftotext":
            lay = "layout" if self.layout else "nolayout"
            return f"pdftotext_{lay}_{self.chunking}"
        return f"pypdf_{self.chunking}"


def build_index_if_needed(
    *,
    pdf_path: Path,
    cfg: BuildConfig,
    artifacts_dir: Path,
    embed_model_name: str,
    embed_device: str,
    embedder: SentenceTransformer,
) -> Dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    need = [
        artifacts_dir / "chunks.jsonl",
        artifacts_dir / "faiss.index",
        artifacts_dir / "tfidf.joblib",
        artifacts_dir / "build_meta.json",
    ]
    if all(p.exists() for p in need):
        return json.loads((artifacts_dir / "build_meta.json").read_text(encoding="utf-8"))

    if cfg.loader == "pdftotext":
        raw_pages = extract_pages_via_pdftotext(pdf_path, layout=cfg.layout, artifacts_dir=artifacts_dir)
    else:
        raw_pages = extract_pages_via_pypdf(pdf_path)

    pages = [clean_page_text(t) for t in raw_pages]
    save_pages_jsonl(pages, artifacts_dir / "pages.jsonl")

    if cfg.chunking == "fields":
        chunks = chunk_fields(pages)
    else:
        chunks = chunk_fixed(pages)

    _write_chunks_jsonl(chunks, artifacts_dir / "chunks.jsonl")

    texts = [c.text for c in chunks]
    emb = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(artifacts_dir / "embeddings.npy", emb)

    index = build_faiss_index(emb)
    import faiss

    faiss.write_index(index, str(artifacts_dir / "faiss.index"))

    vec, mat = build_tfidf(texts)
    import joblib

    joblib.dump({"vectorizer": vec, "matrix": mat}, artifacts_dir / "tfidf.joblib")

    meta = {
        "pdf": str(pdf_path),
        "loader": cfg.loader,
        "layout": bool(cfg.layout),
        "chunking": cfg.chunking,
        "embed_model": embed_model_name,
        "embed_device": embed_device,
        "n_chunks": len(chunks),
    }
    (artifacts_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def hybrid_topk(
    query: str,
    *,
    k: int,
    chunks_all: List[Chunk],
    embedder: SentenceTransformer,
    faiss_index,
    tfidf_vec,
    tfidf_mat,
) -> List[Tuple[float, Chunk]]:
    dense = retrieve_dense(query, embedder, faiss_index, chunks_all, k)
    sparse = retrieve_sparse(query, tfidf_vec, tfidf_mat, chunks_all, k)

    d_scores = {c.chunk_id: float(s) for s, c in dense}
    s_scores = {c.chunk_id: float(s) for s, c in sparse}

    def norm(m: Dict[int, float]) -> Dict[int, float]:
        if not m:
            return {}
        vals = np.array(list(m.values()), dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            return {kk: 1.0 for kk in m}
        return {kk: (vv - lo) / (hi - lo) for kk, vv in m.items()}

    d_n = norm(d_scores)
    s_n = norm(s_scores)
    all_ids = set(d_n) | set(s_n)
    by_id = {c.chunk_id: c for c in chunks_all}

    combined: List[Tuple[float, Chunk]] = []
    for cid in all_ids:
        combined.append((float(d_n.get(cid, 0.0) + s_n.get(cid, 0.0)), by_id[cid]))
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined


def postprocess(
    scored: List[Tuple[float, Chunk]],
    *,
    question: str,
    k: int,
    post: str,
) -> List[Chunk]:
    if not scored:
        return []
    candidates = [c for _, c in scored]

    if post == "none":
        return candidates[:k]

    top_service = (candidates[0].meta or {}).get("service")
    field_hint = infer_field(question) if post == "service_field" else None

    def ok(ch: Chunk) -> bool:
        m = ch.meta or {}
        if top_service and m.get("service") != top_service:
            return False
        if field_hint and m.get("field") != field_hint:
            return False
        return True

    filtered = [c for c in candidates if ok(c)]
    if len(filtered) >= k:
        return filtered[:k]

    # fallback: service only
    if top_service:
        filtered2 = [c for c in candidates if (c.meta or {}).get("service") == top_service]
        if filtered2:
            return filtered2[:k]

    return candidates[:k]


def generate_synthetic_questions(
    *,
    pdf_path: Path,
    out_path: Path,
    n: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(int(seed))

    # 빠른 파싱용: pdftotext(layout) + fields 청킹만 수행(임베딩/인덱스 X)
    raw_pages = extract_pages_via_pdftotext(pdf_path, layout=True)
    pages = [clean_page_text(t) for t in raw_pages]
    chunks = chunk_fields(pages)

    pairs: List[Tuple[str, str]] = []
    for ch in chunks:
        m = ch.meta or {}
        s = m.get("service")
        f = m.get("field")
        if s and f:
            pairs.append((str(s), str(f)))

    # unique
    pairs = sorted(set(pairs))
    if not pairs:
        return 0

    templates = {
        "대상": ["{s} 대상은 누구야?", "{s} 지원 자격 조건 알려줘"],
        "내용": ["{s} 지원 내용이 뭐야?", "{s} 혜택(지원) 내용 알려줘"],
        "방법": ["{s} 신청 방법 알려줘", "{s} 신청 절차가 어떻게 돼?"],
        "문의": ["{s} 문의 전화번호 알려줘", "{s} 어디에 연락하면 돼?"],
    }

    pick_idx = rng.integers(0, len(pairs), size=int(n))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, idx in enumerate(pick_idx.tolist(), start=1):
            service, field = pairs[int(idx)]
            tpls = templates.get(field)
            if not tpls:
                continue
            q = str(rng.choice(tpls)).format(s=service)
            rec = {
                "id": f"auto-{i:04d}",
                "question": q,
                "gold": {"service": service, "field": field},
                "source": "synthetic",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
    return wrote


def dense_scored_for_queries(
    *,
    queries: List[str],
    k: int,
    chunks_all: List[Chunk],
    embedder: SentenceTransformer,
    faiss_index,
) -> List[List[Tuple[float, Chunk]]]:
    q_emb = (
        embedder.encode(
            queries,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        .astype(np.float32)
        .copy()
    )

    scores, ids = faiss_index.search(q_emb, int(k))

    out: List[List[Tuple[float, Chunk]]] = []
    for row_s, row_i in zip(scores, ids):
        scored: List[Tuple[float, Chunk]] = []
        for s, i in zip(row_s.tolist(), row_i.tolist()):
            if int(i) < 0:
                continue
            scored.append((float(s), chunks_all[int(i)]))
        out.append(scored)
    return out


def sparse_scored_for_queries(
    *,
    queries: List[str],
    k: int,
    chunks_all: List[Chunk],
    tfidf_vec,
    tfidf_mat,
) -> List[List[Tuple[float, Chunk]]]:
    from sklearn.metrics.pairwise import cosine_similarity

    qv = tfidf_vec.transform(queries)
    sims = cosine_similarity(qv, tfidf_mat)

    out: List[List[Tuple[float, Chunk]]] = []
    n_docs = len(chunks_all)
    k2 = min(int(k), n_docs)

    for row in sims:
        if k2 >= n_docs:
            idx = np.argsort(-row)
        else:
            idx = np.argpartition(-row, k2)[:k2]
            idx = idx[np.argsort(-row[idx])]
        out.append([(float(row[int(i)]), chunks_all[int(i)]) for i in idx])

    return out


def hybrid_scored_from_dense_sparse(
    *,
    dense_lists: List[List[Tuple[float, Chunk]]],
    sparse_lists: List[List[Tuple[float, Chunk]]],
    chunks_all: List[Chunk],
) -> List[List[Tuple[float, Chunk]]]:
    by_id = {c.chunk_id: c for c in chunks_all}

    def norm(m: Dict[int, float]) -> Dict[int, float]:
        if not m:
            return {}
        vals = np.array(list(m.values()), dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            return {kk: 1.0 for kk in m}
        return {kk: (vv - lo) / (hi - lo) for kk, vv in m.items()}

    out: List[List[Tuple[float, Chunk]]] = []
    for dense, sparse in zip(dense_lists, sparse_lists):
        d_scores = {c.chunk_id: float(s) for s, c in dense}
        s_scores = {c.chunk_id: float(s) for s, c in sparse}

        d_n = norm(d_scores)
        s_n = norm(s_scores)
        all_ids = set(d_n) | set(s_n)

        combined: List[Tuple[float, Chunk]] = []
        for cid in all_ids:
            combined.append((float(d_n.get(cid, 0.0) + s_n.get(cid, 0.0)), by_id[cid]))
        combined.sort(key=lambda x: x[0], reverse=True)
        out.append(combined)

    return out


def eval_from_scored(
    *,
    questions: List[Dict[str, Any]],
    scored_lists: List[List[Tuple[float, Chunk]]],
    k: int,
    method: str,
    post: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hits: List[float] = []
    mrrs: List[float] = []
    details: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for q, scored in zip(questions, scored_lists):
        query = (q.get("question") or "").strip()
        gold = q.get("gold", {})

        top = postprocess(scored, question=query, k=int(k), post=post)

        h = is_hit(top, gold)
        rr = reciprocal_rank(top, gold)
        hits.append(1.0 if h else 0.0)
        mrrs.append(rr)

        top1 = top[0] if top else None
        top1m = (top1.meta or {}) if top1 else {}
        details.append(
            {
                "id": q.get("id"),
                "question": query,
                "gold_service": gold.get("service"),
                "gold_field": gold.get("field"),
                "hit": bool(h),
                "rr": float(rr),
                "top1_service": top1m.get("service"),
                "top1_field": top1m.get("field"),
            }
        )

    dt = time.perf_counter() - t0

    summary = {
        "n": len(questions),
        "k": int(k),
        "method": method,
        "post": post,
        "hit@k": float(np.mean(hits)) if hits else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "seconds": float(dt),
        "qps": float(len(questions) / dt) if dt > 0 else None,
    }
    return summary, details


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF_DEFAULT)
    ap.add_argument("--outdir", default="experiments")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--embed-device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--auto-questions", type=int, default=0, help="0이면 비활성. >0이면 템플릿 기반 질문을 생성해 같이 평가")
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformer(args.embed_model, device=args.embed_device)

    datasets: List[Tuple[str, Path]] = [("manual", Path(args.questions))]

    if int(args.auto_questions) > 0:
        auto_path = outdir / "questions_auto.jsonl"
        n_written = generate_synthetic_questions(
            pdf_path=pdf_path,
            out_path=auto_path,
            n=int(args.auto_questions),
            seed=int(args.seed),
        )
        if n_written > 0:
            datasets.append(("auto", auto_path))

    build_configs = [
        BuildConfig(loader="pdftotext", layout=True, chunking="fields"),
        BuildConfig(loader="pdftotext", layout=True, chunking="fixed"),
        BuildConfig(loader="pypdf", layout=False, chunking="fields"),
        BuildConfig(loader="pypdf", layout=False, chunking="fixed"),
    ]

    results_rows: List[Dict[str, Any]] = []
    details_map: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}

    run_id = time.strftime("%Y%m%d_%H%M%S")

    for cfg in build_configs:
        artifacts_dir = Path("artifacts") / f"exp_{cfg.name}"
        meta = build_index_if_needed(
            pdf_path=pdf_path,
            cfg=cfg,
            artifacts_dir=artifacts_dir,
            embed_model_name=args.embed_model,
            embed_device=args.embed_device,
            embedder=embedder,
        )

        chunks_all = _read_chunks_jsonl(artifacts_dir / "chunks.jsonl")

        import faiss

        faiss_index = faiss.read_index(str(artifacts_dir / "faiss.index"))

        import joblib

        tfidf = joblib.load(artifacts_dir / "tfidf.joblib")
        vec = tfidf["vectorizer"]
        mat = tfidf["matrix"]

        for ds_name, ds_path in datasets:
            if not ds_path.exists():
                continue
            questions = load_questions(ds_path)
            if not questions:
                continue

            queries = [(q.get("question") or "").strip() for q in questions]
            cand_k = max(int(args.k) * 5, 30)

            dense_lists = dense_scored_for_queries(
                queries=queries,
                k=cand_k,
                chunks_all=chunks_all,
                embedder=embedder,
                faiss_index=faiss_index,
            )
            sparse_lists = sparse_scored_for_queries(
                queries=queries,
                k=cand_k,
                chunks_all=chunks_all,
                tfidf_vec=vec,
                tfidf_mat=mat,
            )
            hybrid_lists = hybrid_scored_from_dense_sparse(
                dense_lists=dense_lists,
                sparse_lists=sparse_lists,
                chunks_all=chunks_all,
            )

            by_method = {
                "dense": dense_lists,
                "sparse": sparse_lists,
                "hybrid": hybrid_lists,
            }

            for method, scored_lists in by_method.items():
                for post in ["none", "service", "service_field"]:
                    summary, details = eval_from_scored(
                        questions=questions,
                        scored_lists=scored_lists,
                        k=int(args.k),
                        method=method,
                        post=post,
                    )
                    row = {
                        "run_id": run_id,
                        "dataset": ds_name,
                        "build": cfg.name,
                        "loader": cfg.loader,
                        "layout": bool(cfg.layout),
                        "chunking": cfg.chunking,
                        "method": method,
                        "post": post,
                        **summary,
                        "n_chunks": meta.get("n_chunks"),
                    }
                    results_rows.append(row)
                    details_map[(ds_name, cfg.name, method, post)] = details

    # rank overall
    def score_key(r: Dict[str, Any]) -> Tuple[float, float]:
        return (float(r.get("hit@k", 0.0)), float(r.get("mrr", 0.0)))

    results_sorted = sorted(results_rows, key=score_key, reverse=True)

    out_json = outdir / f"results_{run_id}.json"
    out_csv = outdir / f"results_{run_id}.csv"

    out_json.write_text(json.dumps(results_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results_sorted, out_csv)

    # write a compact TXT summary
    top_lines = []
    for r in results_sorted[:10]:
        top_lines.append(
            f"dataset={r['dataset']} build={r['build']} method={r['method']} post={r['post']} hit@{r['k']}={r['hit@k']:.3f} mrr={r['mrr']:.3f} (n={r['n']})"
        )
    (outdir / f"top10_{run_id}.txt").write_text("\n".join(top_lines) + "\n", encoding="utf-8")

    # best details per dataset
    for ds in sorted({r["dataset"] for r in results_sorted}):
        best = next((r for r in results_sorted if r["dataset"] == ds), None)
        if not best:
            continue
        det = details_map.get((ds, best["build"], best["method"], best["post"]))
        if not det:
            continue
        out_det = outdir / f"best_details_{ds}_{run_id}.jsonl"
        with out_det.open("w", encoding="utf-8") as f:
            for row in det:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n".join(top_lines))
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
