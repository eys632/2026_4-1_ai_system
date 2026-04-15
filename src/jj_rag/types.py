from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str  # url or file path
    title: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    contexts: list[RetrievalResult]
    model_id: str
