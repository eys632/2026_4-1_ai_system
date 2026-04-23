from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    columns: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in columns:
                columns.append(k)

    mode = "a" if exists else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def setup_logger(log_path: Path, name: str = "raglab_exhaustive") -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def safe_text(text: str) -> str:
    return text.replace("\x00", " ").strip()


def try_write_parquet(csv_path: Path, parquet_path: Path) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False

    if not csv_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
        ensure_dir(parquet_path.parent)
        df.to_parquet(parquet_path, index=False)
        return True
    except Exception:
        return False


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}
