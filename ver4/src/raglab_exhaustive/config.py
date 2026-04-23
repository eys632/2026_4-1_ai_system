from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass(frozen=True)
class StageConfig:
    loader: str
    cleaning: str
    chunking: str
    representation: str
    retrieval: str
    post_retrieval: str
    generation: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "loader": self.loader,
            "cleaning": self.cleaning,
            "chunking": self.chunking,
            "representation": self.representation,
            "retrieval": self.retrieval,
            "post_retrieval": self.post_retrieval,
            "generation": self.generation,
        }


@dataclass
class MatrixConfig:
    seed: int
    top_k: int
    stages: Dict[str, List[str]]
    defaults: Dict[str, Any]
    generation_backend: Dict[str, Any]
    judge_backend: Dict[str, Any]
    datasets: List[str]

    def combinations(self) -> Iterable[StageConfig]:
        keys = [
            "loader",
            "cleaning",
            "chunking",
            "representation",
            "retrieval",
            "post_retrieval",
            "generation",
        ]
        values = [self.stages[k] for k in keys]
        for combo in itertools.product(*values):
            yield StageConfig(**dict(zip(keys, combo)))

    def num_combinations(self) -> int:
        n = 1
        for key in [
            "loader",
            "cleaning",
            "chunking",
            "representation",
            "retrieval",
            "post_retrieval",
            "generation",
        ]:
            n *= len(self.stages[key])
        return n


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_matrix_config(path: Path) -> MatrixConfig:
    raw = load_json(path)
    return MatrixConfig(
        seed=int(raw.get("seed", 7)),
        top_k=int(raw.get("top_k", 5)),
        stages=raw["stages"],
        defaults=raw.get("defaults", {}),
        generation_backend=raw.get("generation_backend", {}),
        judge_backend=raw.get("judge_backend", {}),
        datasets=raw.get("datasets", ["manual", "auto", "stress"]),
    )


def stable_hash(payload: Dict[str, Any], prefix: str, n: int = 12) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{digest}"
