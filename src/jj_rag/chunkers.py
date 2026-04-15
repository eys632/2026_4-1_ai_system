from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from .llm import HFGenerator
from .text_utils import normalize_text
from .types import Chunk, Document


@dataclass
class BaselineChunker:
    chunk_chars: int = 900
    overlap: int = 150

    def chunk(self, doc: Document) -> list[Chunk]:
        text = normalize_text(doc.text)
        if not text:
            return []

        # Split by paragraphs first, then pack.
        paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0

        def flush() -> None:
            nonlocal buf, buf_len
            if buf:
                chunks.append("\n\n".join(buf).strip())
                buf = []
                buf_len = 0

        for p in paras:
            if buf_len + len(p) + 2 <= self.chunk_chars:
                buf.append(p)
                buf_len += len(p) + 2
            else:
                flush()
                if len(p) <= self.chunk_chars:
                    buf.append(p)
                    buf_len = len(p)
                else:
                    # Hard split long paragraph
                    start = 0
                    while start < len(p):
                        end = min(start + self.chunk_chars, len(p))
                        chunks.append(p[start:end].strip())
                        start = max(end - self.overlap, end)

        flush()

        # Add overlap by characters between adjacent chunks.
        overlapped: list[str] = []
        for i, c in enumerate(chunks):
            if i == 0 or self.overlap <= 0:
                overlapped.append(c)
                continue
            prev = overlapped[-1]
            prefix = prev[-self.overlap :]
            overlapped.append((prefix + "\n" + c).strip())

        out: list[Chunk] = []
        for i, c in enumerate(overlapped):
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::baseline::{i}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=c,
                    metadata={**doc.metadata, "chunker": "baseline", "chunk_index": i},
                )
            )
        return out


@dataclass
class LLMChunker:
    generator: HFGenerator
    min_chars: int = 450
    max_chars: int = 1200
    max_new_tokens: int = 700

    def _prompt(self, title: str, text: str) -> str:
        return (
            "너는 문서를 RAG용으로 '의미 단위'로 청킹하는 도구다.\n"
            "규칙:\n"
            f"- 각 청크는 대략 {self.min_chars}~{self.max_chars}자 사이가 되도록 하되, 문맥이 끊기면 안 된다.\n"
            "- 제목/소제목/목록 구조가 있으면 최대한 보존한다.\n"
            "- 출력은 반드시 JSON 배열만 출력한다. (설명/코드블록 금지)\n"
            "- 각 원소는 {\"title\": string, \"text\": string} 형태다.\n"
            "- text에는 원문 내용을 그대로(요약하지 말고) 포함한다.\n\n"
            f"[문서 제목]\n{title}\n\n"
            f"[문서 본문]\n{text}\n"
        )

    def _extract_json_array(self, s: str) -> Optional[list[dict]]:
        # Try to locate JSON array region.
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = s[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        if not isinstance(parsed, list):
            return None
        rows: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            rows.append({"title": title, "text": text})
        return rows or None

    def chunk(self, doc: Document) -> list[Chunk]:
        text = normalize_text(doc.text)
        if not text:
            return []

        prompt = self._prompt(doc.title, text)
        raw = self.generator.generate(prompt, max_new_tokens=self.max_new_tokens, temperature=0.0)
        rows = self._extract_json_array(raw)
        if rows is None:
            # Fallback to baseline-like split to avoid total failure.
            fallback = BaselineChunker(chunk_chars=self.max_chars, overlap=0)
            return [
                Chunk(
                    chunk_id=f"{c.chunk_id.replace('baseline', 'llm_fallback')}",
                    doc_id=c.doc_id,
                    source=c.source,
                    title=c.title,
                    text=c.text,
                    metadata={**c.metadata, "chunker": "llm_fallback"},
                )
                for c in fallback.chunk(doc)
            ]

        chunks: list[Chunk] = []
        for i, row in enumerate(rows):
            c_title = row.get("title") or doc.title
            c_text = normalize_text(row["text"])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::llm::{i}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=c_title,
                    text=c_text,
                    metadata={**doc.metadata, "chunker": "llm", "chunk_index": i},
                )
            )
        return chunks
