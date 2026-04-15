from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class E5Embedder:
    model_id: str = "intfloat/multilingual-e5-small"

    _model: Optional[SentenceTransformer] = None

    def load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_id)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.load()
        assert self._model is not None
        prefixed = [f"passage: {t}" for t in texts]
        emb = self._model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self.load()
        assert self._model is not None
        emb = self._model.encode([f"query: {query}"], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)[0]
