import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .rag_lab import Chunk, _read_chunks_jsonl, retrieve_dense, retrieve_sparse


def load_questions(path: Path) -> List[Dict]:
    qs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs


def is_hit(chunks: List[Chunk], gold: Dict) -> bool:
    g_service = gold.get("service")
    g_field = gold.get("field")
    if not g_service:
        return False

    for ch in chunks:
        m = ch.meta or {}
        if m.get("service") != g_service:
            continue
        if g_field and m.get("field") != g_field:
            continue
        return True
    return False


def reciprocal_rank(chunks: List[Chunk], gold: Dict) -> float:
    g_service = gold.get("service")
    g_field = gold.get("field")
    for i, ch in enumerate(chunks, start=1):
        m = ch.meta or {}
        if m.get("service") != g_service:
            continue
        if g_field and m.get("field") != g_field:
            continue
        return 1.0 / i
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", choices=["dense", "sparse", "hybrid"], default="hybrid")
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    chunks_all = _read_chunks_jsonl(Path("artifacts/chunks.jsonl"))

    import faiss

    index = faiss.read_index(str(Path("artifacts/faiss.index")))

    import joblib

    tfidf = joblib.load(Path("artifacts/tfidf.joblib"))
    vec = tfidf["vectorizer"]
    mat = tfidf["matrix"]

    embed_model = SentenceTransformer(args.embed_model, device=args.device)

    questions = load_questions(Path(args.questions))

    hits = []
    mrrs = []

    for q in questions:
        query = q["question"]
        gold = q.get("gold", {})

        if args.method == "dense":
            scored = retrieve_dense(query, embed_model, index, chunks_all, args.k)
            top = [c for _, c in scored]
        elif args.method == "sparse":
            scored = retrieve_sparse(query, vec, mat, chunks_all, args.k)
            top = [c for _, c in scored]
        else:
            dense = retrieve_dense(query, embed_model, index, chunks_all, args.k * 5)
            sparse = retrieve_sparse(query, vec, mat, chunks_all, args.k * 5)

            d_scores = {c.chunk_id: s for s, c in dense}
            s_scores = {c.chunk_id: s for s, c in sparse}

            def norm(m: Dict[int, float]) -> Dict[int, float]:
                if not m:
                    return {}
                vals = np.array(list(m.values()), dtype=np.float32)
                lo, hi = float(vals.min()), float(vals.max())
                if hi - lo < 1e-6:
                    return {k: 1.0 for k in m}
                return {k: (v - lo) / (hi - lo) for k, v in m.items()}

            d_n = norm(d_scores)
            s_n = norm(s_scores)
            all_ids = set(d_n) | set(s_n)
            by_id = {c.chunk_id: c for c in chunks_all}
            combined = [(float(d_n.get(cid, 0.0) + s_n.get(cid, 0.0)), by_id[cid]) for cid in all_ids]
            combined.sort(key=lambda x: x[0], reverse=True)
            top = [c for _, c in combined[: args.k]]

        hits.append(1.0 if is_hit(top, gold) else 0.0)
        mrrs.append(reciprocal_rank(top, gold))

    hit_at_k = float(np.mean(hits)) if hits else 0.0
    mrr = float(np.mean(mrrs)) if mrrs else 0.0

    out = {
        "n": len(questions),
        "method": args.method,
        "k": args.k,
        "hit@k": hit_at_k,
        "mrr": mrr,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
