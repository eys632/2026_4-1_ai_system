from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from tqdm import tqdm

from jj_rag.chunkers import BaselineChunker, LLMChunker
from jj_rag.config import load_settings
from jj_rag.embeddings import E5Embedder
from jj_rag.io_utils import read_jsonl
from jj_rag.llm import HFGenerator
from jj_rag.types import Document
from jj_rag.vectorstore import FaissChunkIndex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunker", choices=["baseline", "llm"], required=True)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = load_settings()

    docs_path = settings.data_processed_dir / "documents.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"Run collect_data first. Missing: {docs_path}")

    rows = read_jsonl(docs_path)
    docs: list[Document] = []
    for r in rows:
        docs.append(
            Document(
                doc_id=r["doc_id"],
                source=r["source"],
                title=r.get("title") or "",
                text=r.get("text") or "",
                metadata=r.get("metadata") or {},
            )
        )

    if args.chunker == "baseline":
        chunker = BaselineChunker(
            chunk_chars=settings.baseline_chunk_chars,
            overlap=settings.baseline_chunk_overlap,
        )
    else:
        gen = HFGenerator(model_id=settings.chunking_model_id, device=settings.device)
        chunker = LLMChunker(
            generator=gen,
            min_chars=settings.llm_chunk_min_chars,
            max_chars=settings.llm_chunk_max_chars,
            max_new_tokens=900,
        )

    all_chunks = []
    for d in tqdm(docs, desc=f"Chunking ({args.chunker})"):
        all_chunks.extend(chunker.chunk(d))

    if not all_chunks:
        raise RuntimeError("No chunks created")

    embedder = E5Embedder(model_id=settings.embedding_model_id)
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_passages(texts, batch_size=32)

    index = FaissChunkIndex.build(embeddings, all_chunks)
    out_dir = settings.index_dir / args.chunker
    index.save(out_dir)

    stats = {
        "chunker": args.chunker,
        "documents": len(docs),
        "chunks": len(all_chunks),
        "index_dir": str(out_dir),
        "embed_model": settings.embedding_model_id,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ build_index complete")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
