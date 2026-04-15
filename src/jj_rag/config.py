from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    index_dir: Path

    embedding_model_id: str
    answer_model_id: str
    chunking_model_id: str

    device: str  # "auto" | "cpu" | "cuda"

    # Retrieval
    top_k: int

    # Baseline chunking
    baseline_chunk_chars: int
    baseline_chunk_overlap: int

    # LLM chunking
    llm_chunk_min_chars: int
    llm_chunk_max_chars: int

    # Generation
    max_new_tokens: int


def load_settings(project_root: str | Path | None = None) -> Settings:
    if project_root is None:
        project_root_path = Path(__file__).resolve().parents[2]
    else:
        project_root_path = Path(project_root).resolve()

    data_dir = Path(os.environ.get("JJ_RAG_DATA_DIR", project_root_path / "data")).resolve()
    index_dir = Path(os.environ.get("JJ_RAG_INDEX_DIR", project_root_path / "index")).resolve()

    return Settings(
        project_root=project_root_path,
        data_raw_dir=data_dir / "raw",
        data_processed_dir=data_dir / "processed",
        index_dir=index_dir,
        embedding_model_id=os.environ.get("JJ_RAG_EMBED_MODEL", "intfloat/multilingual-e5-small"),
        answer_model_id=os.environ.get("JJ_RAG_ANSWER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        chunking_model_id=os.environ.get("JJ_RAG_CHUNK_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        device=os.environ.get("JJ_RAG_DEVICE", "auto"),
        top_k=int(os.environ.get("JJ_RAG_TOP_K", "6")),
        baseline_chunk_chars=int(os.environ.get("JJ_RAG_BASE_CHUNK_CHARS", "900")),
        baseline_chunk_overlap=int(os.environ.get("JJ_RAG_BASE_CHUNK_OVERLAP", "150")),
        llm_chunk_min_chars=int(os.environ.get("JJ_RAG_LLM_CHUNK_MIN", "450")),
        llm_chunk_max_chars=int(os.environ.get("JJ_RAG_LLM_CHUNK_MAX", "1200")),
        max_new_tokens=int(os.environ.get("JJ_RAG_MAX_NEW_TOKENS", "512")),
    )
