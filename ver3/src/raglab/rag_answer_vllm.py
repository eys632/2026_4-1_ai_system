import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .rag_lab import Chunk, _read_chunks_jsonl, retrieve_dense, retrieve_sparse


def build_context(chunks: List[Chunk], max_chars: int = 8000) -> str:
    parts = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        m = ch.meta or {}
        pages = None
        if m.get("page_start") and m.get("page_end"):
            if m["page_start"] == m["page_end"]:
                pages = f"p{m['page_start']}"
            else:
                pages = f"p{m['page_start']}-{m['page_end']}"

        # 답변 포맷과 혼동되지 않도록, 근거 헤더는 bullet 스타일('-')을 절대 쓰지 않음
        meta = {
            "service": m.get("service"),
            "field": m.get("field"),
            "pages": pages,
        }
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if v)
        header = f"【근거 {i} | {meta_str}】" if meta_str else f"【근거 {i}】"

        block = f"{header}\n{ch.text.strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    return (
        "너는 '2025 나에게 힘이 되는 복지서비스' 책자 기반의 한국어 안내 챗봇이다.\n"
        "아래 '근거'에 있는 내용만 사용해 답하라.\n"
        "근거에 없는 내용은 절대 추측하지 말고, 출력은 딱 한 줄로: '문서에서 확인되지 않습니다.'\n"
        "출력 형식(엄격): bullet 줄만 출력한다. bullet 외 텍스트/설명/결론 금지.\n"
        "각 bullet은 반드시 '-'로 시작한다.\n"
        "질문이 '방법/신청/절차'면, 신청 절차를 단계별로 2~4개 bullet로 쓴다.\n"
        "bullet은 최대 4줄. 각 bullet 끝에는 근거 번호를 정확히 1개만 [근거 n]로 붙인다.\n"
        "예시:\n"
        "- ... [근거 1]\n"
        "- ... [근거 2]\n\n"
        f"질문: {question}\n\n"
        "근거:\n"
        f"{context}\n\n"
        "답변:\n"
    )


def infer_field(question: str) -> str | None:
    q = question.replace(" ", "")
    if any(k in q for k in ["문의", "연락", "전화", "콜센터", "번호"]):
        return "문의"
    if any(k in q for k in ["신청", "방법", "어떻게", "절차", "제출", "접수"]):
        return "방법"
    if any(k in q for k in ["대상", "자격", "조건", "누가", "누구"]):
        return "대상"
    if any(k in q for k in ["내용", "지원", "혜택", "금액", "얼마", "본인부담"]):
        return "내용"
    return None


def filter_by_service_and_field(chunks: List[Chunk], *, service: str | None, field: str | None, k: int) -> List[Chunk]:
    out: List[Chunk] = []
    seen = set()
    for ch in chunks:
        if ch.chunk_id in seen:
            continue
        m = ch.meta or {}
        if service and m.get("service") != service:
            continue
        if field and m.get("field") != field:
            continue
        out.append(ch)
        seen.add(ch.chunk_id)
        if len(out) >= k:
            break
    return out


def hybrid_topk(
    query: str,
    *,
    k: int,
    chunks_all: List[Chunk],
    embed_model: SentenceTransformer,
    faiss_index,
    tfidf_vec,
    tfidf_mat,
) -> List[Chunk]:
    dense = retrieve_dense(query, embed_model, faiss_index, chunks_all, k * 5)
    sparse = retrieve_sparse(query, tfidf_vec, tfidf_mat, chunks_all, k * 5)

    d_scores = {c.chunk_id: s for s, c in dense}
    s_scores = {c.chunk_id: s for s, c in sparse}

    def norm(m):
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
    return [c for _, c in combined[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--method", choices=["dense", "sparse", "hybrid"], default="hybrid")

    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--embed-device", default="cuda")

    ap.add_argument("--model", required=True, help="Qwen 로컬 경로 또는 HF 모델 id")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (A100 4장 쓰면 4)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-context-chars", type=int, default=8000)
    ap.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.75,
        help="vLLM이 점유하려는 GPU 메모리 비율(0~1). 다른 프로세스가 있으면 낮추세요.",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="모델 레포의 커스텀 코드를 신뢰하고 실행(필요할 때만 켜세요)",
    )

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)

    args = ap.parse_args()

    chunks_all = _read_chunks_jsonl(Path("artifacts/chunks.jsonl"))

    import faiss

    faiss_index = faiss.read_index(str(Path("artifacts/faiss.index")))

    import joblib

    tfidf = joblib.load(Path("artifacts/tfidf.joblib"))
    tfidf_vec = tfidf["vectorizer"]
    tfidf_mat = tfidf["matrix"]

    embed_model = SentenceTransformer(args.embed_model, device=args.embed_device)

    field_hint = infer_field(args.q)

    if args.method == "dense":
        scored = retrieve_dense(args.q, embed_model, faiss_index, chunks_all, max(args.k * 5, 30))
        candidates = [c for _, c in scored]
    elif args.method == "sparse":
        scored = retrieve_sparse(args.q, tfidf_vec, tfidf_mat, chunks_all, max(args.k * 5, 30))
        candidates = [c for _, c in scored]
    else:
        candidates = hybrid_topk(
            args.q,
            k=max(args.k * 5, 30),
            chunks_all=chunks_all,
            embed_model=embed_model,
            faiss_index=faiss_index,
            tfidf_vec=tfidf_vec,
            tfidf_mat=tfidf_mat,
        )

    top_service = (candidates[0].meta or {}).get("service") if candidates else None

    # 1) (서비스 + 필드)로 최대한 좁히기
    top_chunks = filter_by_service_and_field(candidates, service=top_service, field=field_hint, k=args.k)
    # 2) 필드로 너무 좁혀 비면, 서비스만으로
    if not top_chunks and top_service:
        top_chunks = filter_by_service_and_field(candidates, service=top_service, field=None, k=args.k)
    # 3) 그래도 비면 그냥 상위 k
    if not top_chunks:
        top_chunks = candidates[: args.k]

    context = build_context(top_chunks, max_chars=args.max_context_chars)
    prompt = build_prompt(args.q, context)

    # In-process vLLM (local GPU)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=float(args.gpu_mem_util),
        trust_remote_code=bool(args.trust_remote_code),
    )

    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    out = llm.generate([prompt], params)
    text = out[0].outputs[0].text

    # 데모/시현에서 형태가 흔들리지 않도록 bullet 형태로 맞춰 출력
    import re

    cite_re = re.compile(r"\[근거\s*\d+\]")

    def normalize_bullet(ln: str) -> str:
        ln2 = ln.strip()
        if not ln2.startswith("-"):
            ln2 = f"- {ln2.lstrip('-> ').strip()}"
        # 의미 없는 메타/헤더 bullet 제거
        if "major=" in ln2 or ln2.lstrip("- ").startswith("[근거"):
            return ""
        if not cite_re.search(ln2):
            ln2 = ln2.rstrip() + " [근거 1]"
        # 근거 표기가 여러 번 반복되면 첫 1회까지만 남김
        m = cite_re.search(ln2)
        if m:
            ln2 = ln2[: m.end()]
        return ln2

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    raw_bullets = [ln for ln in lines if ln.startswith("-")]

    out_lines = []
    if raw_bullets:
        for ln in raw_bullets:
            ln2 = normalize_bullet(ln)
            if ln2:
                out_lines.append(ln2)
            if len(out_lines) >= 4:
                break
    else:
        for ln in lines[:4]:
            ln2 = normalize_bullet(ln)
            if ln2:
                out_lines.append(ln2)

    print("\n".join(out_lines).strip())


if __name__ == "__main__":
    main()
