from __future__ import annotations

import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

import gradio as gr

from jj_rag.config import load_settings
from jj_rag.rag import RAGSystem


def main() -> None:
    load_dotenv()
    settings = load_settings()

    # Lazy-load heavy models on first call, but index files must exist.
    rag_baseline = RAGSystem.load(settings, chunker_name="baseline")
    rag_llm = RAGSystem.load(settings, chunker_name="llm")

    def answer_both(question: str) -> tuple[str, str]:
        question = (question or "").strip()
        if not question:
            return "", ""

        # 기본 청킹 답변 생성 시간 측정
        start_baseline = time.time()
        a1 = rag_baseline.answer(question).answer
        elapsed_baseline = (time.time() - start_baseline) * 1000  # ms 단위
        
        # LLM 청킹 답변 생성 시간 측정
        start_llm = time.time()
        a2 = rag_llm.answer(question).answer
        elapsed_llm = (time.time() - start_llm) * 1000  # ms 단위
        
        # 답변에 소요 시간 정보 추가
        a1_with_time = f"⏱️ 소요 시간: {elapsed_baseline:.0f}ms\n\n{a1}"
        a2_with_time = f"⏱️ 소요 시간: {elapsed_llm:.0f}ms\n\n{a2}"
        
        return a1_with_time, a2_with_time

    with gr.Blocks(title="JJ RAG (baseline vs LLM chunking)") as demo:
        gr.Markdown("# 📊 JJ RAG 시스템\n## 청킹 방식 비교 분석\n동일한 질문에 대해 **기본 청킹 vs LLM 기반 청킹**의 답변 품질과 속도를 비교합니다.\n- 📄 데이터: 전주대학교 인공지능학과 웹페이지 + 2025 복지서비스 가이드 PDF")
        inp = gr.Textbox(label="💬 질문", placeholder="예) 인공지능학과의 학과 교육 모토가 뭐야?")
        btn = gr.Button("🚀 답변 생성")
        out1 = gr.Textbox(label="[기본 청킹] 고정 크기(900자) 청킹", lines=15)
        out2 = gr.Textbox(label="[LLM 청킹] 의미 기반 동적 청킹 ⭐", lines=15)

        btn.click(fn=answer_both, inputs=[inp], outputs=[out1, out2])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
