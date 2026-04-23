from __future__ import annotations

import argparse
from pathlib import Path

from .config import StageConfig, load_matrix_config, load_yaml
from .orchestration import Paths, _ensure_build
from .io_utils import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--matrix", default="configs/experiment_matrix.json")
    ap.add_argument("--prompts", default="configs/generation_prompts.yaml")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    matrix = load_matrix_config((root / args.matrix).resolve())
    _ = load_yaml((root / args.prompts).resolve())

    logger = setup_logger(root / "results" / "logs" / "build_all.log")
    paths = Paths(root=root)

    unique_builds = set()
    for st in matrix.combinations():
        unique_builds.add((st.loader, st.cleaning, st.chunking, st.representation))

    logger.info("Unique build combinations: %d", len(unique_builds))

    for loader, cleaning, chunking, representation in sorted(unique_builds):
        stage = StageConfig(
            loader=loader,
            cleaning=cleaning,
            chunking=chunking,
            representation=representation,
            retrieval=matrix.stages["retrieval"][0],
            post_retrieval=matrix.stages["post_retrieval"][0],
            generation=matrix.stages["generation"][0],
        )
        _ensure_build(paths, Path(args.pdf), stage, matrix, logger)

    logger.info("Build precomputation done.")


if __name__ == "__main__":
    main()
