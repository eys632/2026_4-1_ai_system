from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .io_utils import read_jsonl, write_jsonl
from .types import Chunk, RetrievalResult


@dataclass
class FaissChunkIndex:
    dim: int
    index: faiss.Index
    chunks: list[Chunk]

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[Chunk]) -> "FaissChunkIndex":
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings/chunks length mismatch")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(dim=dim, index=index, chunks=chunks)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if query_emb.ndim != 1:
            raise ValueError("query_emb must be 1D")
        q = np.asarray(query_emb, dtype=np.float32)[None, :]
        scores, idxs = self.index.search(q, top_k)
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))

        rows = []
        for c in self.chunks:
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "title": c.title,
                    "text": c.text,
                    "metadata": c.metadata,
                }
            )
        write_jsonl(out_dir / "chunks.jsonl", rows)

        (out_dir / "meta.json").write_text(json.dumps({"dim": self.dim}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str | Path) -> "FaissChunkIndex":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
        dim = int(meta["dim"])
        index = faiss.read_index(str(in_dir / "index.faiss"))

        rows = read_jsonl(in_dir / "chunks.jsonl")
        chunks: list[Chunk] = []
        for r in rows:
            chunks.append(
                Chunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    source=r["source"],
                    title=r.get("title") or "",
                    text=r.get("text") or "",
                    metadata=r.get("metadata") or {},
                )
            )
        return cls(dim=dim, index=index, chunks=chunks)
