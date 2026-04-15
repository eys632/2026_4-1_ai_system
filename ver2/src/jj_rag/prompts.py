from __future__ import annotations


def build_rag_prompt(question: str, contexts: list[tuple[str, str, str]]) -> str:
    """contexts: list of (source, title, text)"""

    ctx_blocks = []
    for i, (source, title, text) in enumerate(contexts, start=1):
        ctx_blocks.append(f"[근거 {i}]\n- 출처: {source}\n- 제목: {title}\n- 내용:\n{text}")

    joined = "\n\n".join(ctx_blocks)

    return (
        "너는 검색 기반 질의응답(RAG) 도우미다.\n"
        "규칙:\n"
        "- 아래 '근거'에 있는 내용만 사용해 답해라. 근거에 없으면 '근거에서 확인할 수 없습니다'라고 말해라.\n"
        "- 근거들 중 질문과 직접 관련 없는 내용은 제외하고 답해라.\n"
        "- 근거에서 확인되는 지원/제도/혜택을 항목별로 간단히 정리해라.\n"
        "- 같은 항목이나 문장을 반복하지 마라.\n"
        "- 가능하면 각 항목을 (제도명/지원대상/지원내용/신청·문의) 순서로 짧게 정리해라.\n"
        "- 답변은 한국어로, 핵심 위주로 간결하게 작성해라.\n"
        "- 마지막 줄에 사용한 근거 번호를 (근거: 1,2) 형태로 적어라.\n\n"
        f"[질문]\n{question}\n\n"
        f"[근거]\n{joined}\n\n"
        "[답변]\n"
    )
