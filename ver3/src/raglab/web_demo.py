import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

from .rag_lab import Chunk, _read_chunks_jsonl, retrieve_dense, retrieve_sparse


QWEN_05B = "/home/a202192020/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
QWEN_15B = "/home/a202192020/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


CITE_RE = re.compile(r"\[근거\s*\d+\]")


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


def build_context(chunks: List[Chunk], max_chars: int = 8000) -> str:
    parts: List[str] = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        m = ch.meta or {}
        pages = None
        if m.get("page_start") and m.get("page_end"):
            if m["page_start"] == m["page_end"]:
                pages = f"p{m['page_start']}"
            else:
                pages = f"p{m['page_start']}-{m['page_end']}"
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


def normalize_answer(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    raw_bullets = [ln for ln in lines if ln.startswith("-")]

    def normalize_bullet(ln: str) -> str:
        ln2 = ln.strip()
        if not ln2.startswith("-"):
            ln2 = f"- {ln2.lstrip('-> ').strip()}"
        if "major=" in ln2 or ln2.lstrip("- ").startswith("[근거"):
            return ""
        if not CITE_RE.search(ln2):
            ln2 = ln2.rstrip() + " [근거 1]"
        m = CITE_RE.search(ln2)
        if m:
            ln2 = ln2[: m.end()]
        return ln2

    out_lines: List[str] = []
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

    return "\n".join(out_lines).strip()


@dataclass
class RetrievalResult:
    score: float
    chunk: Chunk


class RagDemoEngine:
    def __init__(
        self,
        *,
        artifacts_dir: Path,
        embed_model: str,
        embed_device: str,
        llm_model_path: str,
        llm_tp: int,
        llm_max_model_len: int,
        llm_gpu_mem_util: float,
        llm_enforce_eager: bool,
        trust_remote_code: bool,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.embed_model_name = embed_model
        self.embed_device = embed_device

        self.llm_model_path = llm_model_path
        self.llm_tp = llm_tp
        self.llm_max_model_len = llm_max_model_len
        self.llm_gpu_mem_util = llm_gpu_mem_util
        self.llm_enforce_eager = llm_enforce_eager
        self.trust_remote_code = trust_remote_code

        self._chunks: Optional[List[Chunk]] = None
        self._faiss = None
        self._index = None
        self._tfidf = None
        self._embedder: Optional[SentenceTransformer] = None
        self._llm = None

    def load(self) -> None:
        if self._chunks is None:
            self._chunks = _read_chunks_jsonl(self.artifacts_dir / "chunks.jsonl")

        if self._index is None:
            import faiss

            self._index = faiss.read_index(str(self.artifacts_dir / "faiss.index"))

        if self._tfidf is None:
            import joblib

            self._tfidf = joblib.load(self.artifacts_dir / "tfidf.joblib")

        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model_name, device=self.embed_device)

    @property
    def chunks(self) -> List[Chunk]:
        assert self._chunks is not None
        return self._chunks

    @property
    def index(self):
        assert self._index is not None
        return self._index

    @property
    def tfidf_vec(self):
        assert self._tfidf is not None
        return self._tfidf["vectorizer"]

    @property
    def tfidf_mat(self):
        assert self._tfidf is not None
        return self._tfidf["matrix"]

    @property
    def embedder(self) -> SentenceTransformer:
        assert self._embedder is not None
        return self._embedder

    def ensure_llm(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(
            model=self.llm_model_path,
            tensor_parallel_size=int(self.llm_tp),
            dtype="bfloat16",
            max_model_len=int(self.llm_max_model_len),
            gpu_memory_utilization=float(self.llm_gpu_mem_util),
            enforce_eager=bool(self.llm_enforce_eager),
            trust_remote_code=bool(self.trust_remote_code),
        )

    def retrieve(self, query: str, *, method: str, k: int) -> List[RetrievalResult]:
        if method == "dense":
            scored = retrieve_dense(query, self.embedder, self.index, self.chunks, max(k * 5, 30))
            return [RetrievalResult(score=float(s), chunk=c) for s, c in scored[:k]]
        if method == "sparse":
            scored = retrieve_sparse(query, self.tfidf_vec, self.tfidf_mat, self.chunks, max(k * 5, 30))
            return [RetrievalResult(score=float(s), chunk=c) for s, c in scored[:k]]

        # hybrid: normalize then sum
        dense = retrieve_dense(query, self.embedder, self.index, self.chunks, max(k * 5, 30))
        sparse = retrieve_sparse(query, self.tfidf_vec, self.tfidf_mat, self.chunks, max(k * 5, 30))

        d_scores = {c.chunk_id: float(s) for s, c in dense}
        s_scores = {c.chunk_id: float(s) for s, c in sparse}

        def norm(m: Dict[int, float]) -> Dict[int, float]:
            if not m:
                return {}
            vals = np.array(list(m.values()), dtype=np.float32)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-6:
                return {k: 1.0 for k in m}
            return {kk: (vv - lo) / (hi - lo) for kk, vv in m.items()}

        d_n = norm(d_scores)
        s_n = norm(s_scores)
        all_ids = set(d_n) | set(s_n)
        by_id = {c.chunk_id: c for c in self.chunks}

        combined: List[RetrievalResult] = []
        for cid in all_ids:
            combined.append(RetrievalResult(score=float(d_n.get(cid, 0.0) + s_n.get(cid, 0.0)), chunk=by_id[cid]))
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:k]

    def retrieve_filtered(self, query: str, *, method: str, k: int, use_field_routing: bool) -> List[RetrievalResult]:
        candidates = self.retrieve(query, method=method, k=max(k * 5, 30))

        if not candidates:
            return []

        top_service = (candidates[0].chunk.meta or {}).get("service")
        field_hint = infer_field(query) if use_field_routing else None

        def ok(r: RetrievalResult) -> bool:
            m = r.chunk.meta or {}
            if top_service and m.get("service") != top_service:
                return False
            if field_hint and m.get("field") != field_hint:
                return False
            return True

        filtered = [r for r in candidates if ok(r)]
        if filtered:
            return filtered[:k]

        # fallback: service only
        def ok2(r: RetrievalResult) -> bool:
            m = r.chunk.meta or {}
            if top_service and m.get("service") != top_service:
                return False
            return True

        filtered2 = [r for r in candidates if ok2(r)]
        if filtered2:
            return filtered2[:k]

        return candidates[:k]

    def generate(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> str:
        self.ensure_llm()
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        out = self._llm.generate([prompt], params)
        return out[0].outputs[0].text


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_debug_rows(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    rows = []
    for rank, r in enumerate(results, start=1):
        m = r.chunk.meta or {}
        pages = None
        if m.get("page_start") and m.get("page_end"):
            pages = f"p{m['page_start']}-{m['page_end']}" if m["page_start"] != m["page_end"] else f"p{m['page_start']}"
        rows.append(
            {
                "rank": rank,
                "score": round(float(r.score), 4),
                "service": m.get("service"),
                "field": m.get("field"),
                "pages": pages,
                "snippet": (r.chunk.text[:220].replace("\n", " ") + ("..." if len(r.chunk.text) > 220 else "")),
            }
        )
    return rows


def compute_retrieval_metrics(engine: RagDemoEngine, questions_path: Path, *, method: str, k: int) -> Dict[str, Any]:
    if not questions_path.exists():
        return {"error": f"questions not found: {str(questions_path)}"}

    questions = []
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        questions.append(json.loads(line))

    if not questions:
        return {"error": "questions is empty"}

    def is_hit(results: List[RetrievalResult], gold: Dict[str, Any]) -> bool:
        g_service = gold.get("service")
        g_field = gold.get("field")
        if not g_service:
            return False
        for r in results:
            m = r.chunk.meta or {}
            if m.get("service") != g_service:
                continue
            if g_field and m.get("field") != g_field:
                continue
            return True
        return False

    def rr(results: List[RetrievalResult], gold: Dict[str, Any]) -> float:
        g_service = gold.get("service")
        g_field = gold.get("field")
        if not g_service:
            return 0.0
        for i, r in enumerate(results, start=1):
            m = r.chunk.meta or {}
            if m.get("service") != g_service:
                continue
            if g_field and m.get("field") != g_field:
                continue
            return 1.0 / i
        return 0.0

    hits = []
    rrs = []
    details = []

    for q in questions:
        query = q.get("question", "")
        gold = q.get("gold", {})
        results = engine.retrieve(query, method=method, k=k)
        h = is_hit(results, gold)
        hits.append(1.0 if h else 0.0)
        rrs.append(rr(results, gold))
        details.append(
            {
                "id": q.get("id"),
                "question": query,
                "gold_service": gold.get("service"),
                "gold_field": gold.get("field"),
                "hit": bool(h),
                "top1_service": (results[0].chunk.meta or {}).get("service") if results else None,
                "top1_field": (results[0].chunk.meta or {}).get("field") if results else None,
            }
        )

    return {
        "n": len(questions),
        "method": method,
        "k": k,
        "hit@k": float(np.mean(hits)),
        "mrr": float(np.mean(rrs)),
        "details": details,
    }


def make_app(engine: RagDemoEngine) -> gr.Blocks:
    engine.load()

    build_meta = load_json(engine.artifacts_dir / "build_meta.json")
    pdf_probe = load_json(engine.artifacts_dir / "pdf_probe.json")

    # field distribution
    field_counts: Dict[str, int] = {}
    major_counts: Dict[str, int] = {}
    for ch in engine.chunks:
        m = ch.meta or {}
        f = m.get("field")
        if f:
            field_counts[f] = field_counts.get(f, 0) + 1
        maj = m.get("major_category")
        if maj:
            major_counts[maj] = major_counts.get(maj, 0) + 1

    def rag_answer(
        question: str,
        method: str,
        k: int,
        max_context_chars: int,
        use_field_routing: bool,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        q = (question or "").strip()
        if not q:
            return "", [], ""

        results = engine.retrieve_filtered(q, method=method, k=int(k), use_field_routing=bool(use_field_routing))
        ctx = build_context([r.chunk for r in results], max_chars=int(max_context_chars))
        prompt = build_prompt(q, ctx)

        raw = engine.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        ans = normalize_answer(raw)

        rows = build_debug_rows(results)
        return ans, rows, ctx

    def retrieval_only(query: str, method: str, k: int) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        results = engine.retrieve(q, method=method, k=int(k))
        return build_debug_rows(results)

    def run_eval(k: int) -> Tuple[str, List[Dict[str, Any]]]:
        qpath = Path("eval/questions.jsonl")
        outs = []
        for method in ["dense", "sparse", "hybrid"]:
            m = compute_retrieval_metrics(engine, qpath, method=method, k=int(k))
            outs.append({k: v for k, v in m.items() if k != "details"})
        details = compute_retrieval_metrics(engine, qpath, method="hybrid", k=int(k)).get("details", [])
        return json.dumps(outs, ensure_ascii=False, indent=2), details

    발표체크 = """
발표/시현 체크리스트(이 프로젝트 기준)

1) 데이터/문서 특성
- PDF 페이지 수, 텍스트 추출 가능 여부(스캔 페이지 유무)
- 문서가 '서비스 템플릿(대상/내용/방법/문의)' 반복 구조인지
- 표(테이블) 존재 여부 및 처리 방식

2) RAG 파이프라인(단계별)
- Loader: pdftotext(layout) vs pypdf 등 비교 포인트
- Cleaning: 푸터/제어문자 제거
- Chunking: 고정 길이 vs 필드 기반(대상/내용/방법/문의)
- Indexing: Dense(FAISS) / Sparse(TF-IDF) / Hybrid
- (선택) Rerank, Context 압축/중복제거
- Generation: '근거 외 추측 금지' + 근거 인용
- Post-retrieval: 문서에 없으면 "확인되지 않습니다" 정책

3) 성능평가(반드시 숫자)
- Retrieval: hit@k, MRR (dense/sparse/hybrid 비교)
- 질문 유형별(대상/방법/문의/내용)로 잘 되는지/약한지

4) 시현 시나리오
- 정상 질문 2개: (방법) (문의처) 같은 필드가 확실한 질문
- 표 기반 질문 1개: '본인부담률' 같은 테이블 질문
- 부재 질문 1개: 문서에 없는 내용을 물어 '확인되지 않습니다'가 나오는지

5) 가점(데이터 맞춤)
- 필드 기반 청킹(대상/내용/방법/문의)
- 질문-필드 라우팅(방법 질문이면 '방법' chunk 우선)
- 표 전용 chunk/요약(선택)
""".strip()

    with gr.Blocks(title="RAG 시현 (복지서비스 PDF)") as demo:
        gr.Markdown("# 데이터 기반 RAG 시현\nPDF: 2025 나에게 힘이 되는 복지서비스")

        with gr.Tabs():
            with gr.TabItem("질문 → 답변 (RAG)"):
                with gr.Row():
                    question = gr.Textbox(label="질문", placeholder="예) 실업급여 신청 방법은?", lines=2)

                with gr.Row():
                    method = gr.Dropdown(["hybrid", "dense", "sparse"], value="hybrid", label="Retrieval")
                    k = gr.Slider(1, 12, value=4, step=1, label="top-k")
                    max_context_chars = gr.Slider(1000, 14000, value=8000, step=500, label="컨텍스트 최대 글자수")

                with gr.Row():
                    use_field_routing = gr.Checkbox(value=True, label="필드 라우팅(방법/문의/대상/내용) 사용")

                with gr.Accordion("생성 파라미터(보통 그대로 둠)", open=False):
                    with gr.Row():
                        temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature")
                        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                        max_tokens = gr.Slider(64, 1024, value=220, step=16, label="max_tokens")

                run_btn = gr.Button("검색 + 답변 생성", variant="primary")

                answer = gr.Markdown(label="답변")
                debug_table = gr.Dataframe(
                    headers=["rank", "score", "service", "field", "pages", "snippet"],
                    label="Top-k 근거(요약)",
                    interactive=False,
                )
                context = gr.Textbox(label="실제 컨텍스트(근거 원문)", lines=18)

                run_btn.click(
                    rag_answer,
                    inputs=[question, method, k, max_context_chars, use_field_routing, temperature, top_p, max_tokens],
                    outputs=[answer, debug_table, context],
                )

            with gr.TabItem("검색 디버그"):
                with gr.Row():
                    q2 = gr.Textbox(label="검색어", placeholder="예) 건강보험 산정특례 문의", lines=2)
                with gr.Row():
                    method2 = gr.Dropdown(["hybrid", "dense", "sparse"], value="hybrid", label="Retrieval")
                    k2 = gr.Slider(1, 20, value=8, step=1, label="top-k")
                btn2 = gr.Button("검색만 실행")
                table2 = gr.Dataframe(
                    headers=["rank", "score", "service", "field", "pages", "snippet"],
                    label="검색 결과", interactive=False
                )
                btn2.click(retrieval_only, inputs=[q2, method2, k2], outputs=[table2])

            with gr.TabItem("PDF/데이터 특성"):
                gr.Markdown("## 빌드 메타데이터")
                gr.Code(json.dumps(build_meta, ensure_ascii=False, indent=2), language="json")

                gr.Markdown("## PDF 특성(프로브)")
                if pdf_probe:
                    gr.Code(json.dumps(pdf_probe, ensure_ascii=False, indent=2), language="json")
                else:
                    gr.Markdown("`artifacts/pdf_probe.json`이 없어서 여기엔 표시되지 않았습니다.")

                gr.Markdown("## 청킹 분포")
                gr.Code(json.dumps({"field_counts": field_counts, "major_counts": major_counts}, ensure_ascii=False, indent=2), language="json")

            with gr.TabItem("평가(리트리벌)"):
                gr.Markdown("eval/questions.jsonl 기반으로 hit@k / MRR을 비교합니다(작게 시작).")
                k_eval = gr.Slider(1, 15, value=5, step=1, label="k")
                btn_eval = gr.Button("평가 실행")
                metrics_out = gr.Code(language="json", label="요약 결과")
                details_out = gr.Dataframe(
                    headers=["id", "question", "gold_service", "gold_field", "hit", "top1_service", "top1_field"],
                    label="문항별 결과(hybrid)",
                    interactive=False,
                )
                btn_eval.click(run_eval, inputs=[k_eval], outputs=[metrics_out, details_out])

            with gr.TabItem("발표/시현 체크"):
                gr.Textbox(value=발표체크, label="체크리스트", lines=22)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")

    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--embed-device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--llm", default=QWEN_15B, choices=[QWEN_05B, QWEN_15B])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.75)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    ap.add_argument("--trust-remote-code", action="store_true", default=False)

    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", default=False)

    args = ap.parse_args()

    engine = RagDemoEngine(
        artifacts_dir=Path(args.artifacts),
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        llm_model_path=args.llm,
        llm_tp=args.tp,
        llm_max_model_len=args.max_model_len,
        llm_gpu_mem_util=args.gpu_mem_util,
        llm_enforce_eager=bool(args.enforce_eager),
        trust_remote_code=bool(args.trust_remote_code),
    )

    app = make_app(engine)
    app.launch(server_name=args.host, server_port=int(args.port), share=bool(args.share))


if __name__ == "__main__":
    main()
