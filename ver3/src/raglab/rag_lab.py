import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_DEFAULT = "2025 나에게  힘이 되는 복지서비스.pdf"


FOOTER_RE = re.compile(r"^\s*[가-힣·\s]{2,30}지원\s+\d{1,4}\s*$")
BOOK_FOOTER_RE = re.compile(r"^\s*\d{1,4}\s+2025\s+나에게\s+힘이\s+되는\s+복지서비스\s*$")
MAJOR_CATEGORY_RE = re.compile(r"^\s*[가-힣·\s]{2,30}지원\s*$")
FIELD_RE = re.compile(r"^\s*(대상|내용|방법|문의)\b")
# Non-printable control chars (often introduced by PDF icon fonts) can break terminal output.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    meta: Dict[str, Any]


def _run_pdftotext(pdf_path: Path, out_txt: Path, layout: bool = True) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pdftotext",
        "-enc",
        "UTF-8",
    ]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(pdf_path), str(out_txt)])
    subprocess.run(cmd, check=True)


def extract_pages_via_pdftotext(pdf_path: Path, *, layout: bool = True) -> List[str]:
    tmp_txt = Path("artifacts") / (pdf_path.stem + (".layout.txt" if layout else ".txt"))
    if not tmp_txt.exists():
        _run_pdftotext(pdf_path, tmp_txt, layout=layout)

    text = tmp_txt.read_text(encoding="utf-8", errors="replace")
    pages = text.split("\f")

    # pdftotext commonly returns (n_pages + 1) segments with an empty tail.
    while pages and not pages[-1].strip():
        pages.pop()
    return pages


def extract_pages_via_pypdf(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def clean_page_text(page_text: str) -> str:
    lines = []
    for raw in page_text.splitlines():
        line = raw.rstrip("\n")
        line = CONTROL_RE.sub("", line)
        # PDF 아이콘 폰트가 텍스트로 깨진 경우가 많아, 눈에 보이는 화살표로 통일
        line = line.replace("è", "->")
        if not line.strip():
            lines.append("")
            continue
        if FOOTER_RE.match(line) or BOOK_FOOTER_RE.match(line):
            continue
        # remove pure page-number lines
        if re.fullmatch(r"\s*\d{1,4}\s*", line):
            continue
        lines.append(line)

    # collapse excessive blank lines
    out = []
    blank_run = 0
    for ln in lines:
        if ln.strip():
            blank_run = 0
            out.append(ln)
        else:
            blank_run += 1
            if blank_run <= 2:
                out.append("")

    return "\n".join(out).strip("\n")


def probe_pdf(pdf_path: Path) -> Dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)

    # quick char counts via pypdf (fast proxy for empty/graphic pages)
    char_counts: List[int] = []
    empty_pages: List[int] = []
    for i, p in enumerate(reader.pages, start=1):
        try:
            t = (p.extract_text() or "").strip()
        except Exception:
            t = ""
        c = len(t)
        char_counts.append(c)
        if c == 0:
            empty_pages.append(i)

    s = pd.Series(char_counts)
    stats = {
        "pages": n_pages,
        "empty_pages_count": len(empty_pages),
        "empty_pages": empty_pages,
        "chars_p50": int(s.quantile(0.50)),
        "chars_p90": int(s.quantile(0.90)),
        "chars_p99": int(s.quantile(0.99)),
    }
    return stats


def save_pages_jsonl(pages: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(pages, start=1):
            rec = {
                "page": i,
                "text": t,
                "chars": len(t),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def chunk_fixed(pages: List[str], *, chunk_chars: int = 1200, overlap: int = 200) -> List[Chunk]:
    full = "\n\n".join(p for p in pages if p.strip())
    full = full.strip()
    chunks: List[Chunk] = []
    i = 0
    cid = 1
    while i < len(full):
        j = min(len(full), i + chunk_chars)
        text = full[i:j]
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=text,
                meta={"chunking": "fixed", "start_char": i, "end_char": j},
            )
        )
        cid += 1
        if j == len(full):
            break
        i = max(0, j - overlap)
    return chunks


def chunk_fields(pages: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []

    current_major: Optional[str] = None
    current_service: Optional[str] = None
    current_field: Optional[str] = None
    buf: List[str] = []
    cid = 1

    field_start_page: Optional[int] = None
    last_page_seen: Optional[int] = None

    def flush() -> None:
        nonlocal cid, buf, current_field, field_start_page, last_page_seen
        if not current_service or not current_field:
            buf = []
            field_start_page = None
            return
        content = "\n".join(x for x in buf).strip()
        if not content:
            buf = []
            field_start_page = None
            return

        prefix = ""
        if current_major:
            prefix += f"[{current_major}] "
        prefix += f"{current_service} - {current_field}\n"

        meta = {
            "chunking": "fields",
            "major_category": current_major,
            "service": current_service,
            "field": current_field,
            "page_start": field_start_page,
            "page_end": last_page_seen,
        }
        chunks.append(Chunk(chunk_id=cid, text=prefix + content, meta=meta))
        cid += 1
        buf = []
        field_start_page = None

    prev_nonempty_line: Optional[str] = None

    for page_no, page_text in enumerate(pages, start=1):
        cleaned = clean_page_text(page_text)
        last_page_seen = page_no
        for line in cleaned.splitlines():
            if not line.strip():
                continue

            # major category headings like "생계 지원", "취업 지원"...
            if MAJOR_CATEGORY_RE.match(line) and len(line.strip()) <= 20:
                current_major = re.sub(r"\s+", " ", line.strip())
                prev_nonempty_line = line.strip()
                continue

            m = FIELD_RE.match(line)
            if m:
                field = m.group(1)
                # a new service typically starts at "대상"; infer service name from previous non-empty line.
                if field == "대상" and prev_nonempty_line and prev_nonempty_line not in {"대상", "내용", "방법", "문의"}:
                    # flush previous field/service
                    flush()
                    current_service = re.sub(r"\s+", " ", prev_nonempty_line.strip())
                    current_field = field
                    field_start_page = page_no
                    buf = []
                    continue

                # switch field within the same service
                if current_service:
                    flush()
                    current_field = field
                    field_start_page = page_no
                    buf = []
                    continue

            # normal content
            if current_service and current_field:
                buf.append(line.strip())

            prev_nonempty_line = line.strip()

    flush()
    return chunks


def build_embeddings(texts: List[str], model_name: str, *, device: str = "cuda") -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def build_faiss_index(emb: np.ndarray):
    import faiss

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index


def build_tfidf(texts: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1, 2),
        lowercase=False,
    )
    mat = vec.fit_transform(texts)
    return vec, mat


def retrieve_dense(query: str, embed_model: SentenceTransformer, index, chunks: List[Chunk], top_k: int) -> List[Tuple[float, Chunk]]:
    import faiss

    q = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, top_k)
    out: List[Tuple[float, Chunk]] = []
    for s, i in zip(scores[0].tolist(), ids[0].tolist()):
        if i < 0:
            continue
        out.append((float(s), chunks[i]))
    return out


def retrieve_sparse(query: str, vec, mat, chunks: List[Chunk], top_k: int) -> List[Tuple[float, Chunk]]:
    from sklearn.metrics.pairwise import cosine_similarity

    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()
    if top_k >= len(chunks):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
    return [(float(sims[i]), chunks[int(i)]) for i in idx]


def _write_chunks_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            rec = {
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "meta": ch.meta,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_chunks_jsonl(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(Chunk(chunk_id=int(rec["chunk_id"]), text=rec["text"], meta=rec["meta"]))
    # keep stable order for indexing
    chunks.sort(key=lambda c: c.chunk_id)
    return chunks


def cmd_probe(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)
    stats = probe_pdf(pdf_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)

    if args.loader == "pdftotext":
        raw_pages = extract_pages_via_pdftotext(pdf_path, layout=args.layout)
    else:
        raw_pages = extract_pages_via_pypdf(pdf_path)

    pages = [clean_page_text(t) for t in raw_pages]

    save_pages_jsonl(pages, Path("artifacts") / "pages.jsonl")

    if args.chunking == "fields":
        chunks = chunk_fields(pages)
    else:
        chunks = chunk_fixed(pages, chunk_chars=args.chunk_chars, overlap=args.overlap)

    _write_chunks_jsonl(chunks, Path("artifacts") / "chunks.jsonl")

    texts = [c.text for c in chunks]

    emb_model_name = args.embed_model
    emb = build_embeddings(texts, emb_model_name, device=args.device)
    np.save(Path("artifacts") / "embeddings.npy", emb)

    index = build_faiss_index(emb)
    import faiss

    faiss.write_index(index, str(Path("artifacts") / "faiss.index"))

    vec, mat = build_tfidf(texts)
    import joblib

    joblib.dump({"vectorizer": vec, "matrix": mat}, Path("artifacts") / "tfidf.joblib")

    meta = {
        "pdf": str(pdf_path),
        "loader": args.loader,
        "layout": bool(args.layout),
        "chunking": args.chunking,
        "embed_model": emb_model_name,
        "device": args.device,
        "n_chunks": len(chunks),
    }
    (Path("artifacts") / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"built chunks: {len(chunks)}")


def cmd_query(args: argparse.Namespace) -> None:
    chunks = _read_chunks_jsonl(Path("artifacts") / "chunks.jsonl")

    import faiss

    index = faiss.read_index(str(Path("artifacts") / "faiss.index"))

    import joblib

    tfidf = joblib.load(Path("artifacts") / "tfidf.joblib")
    vec = tfidf["vectorizer"]
    mat = tfidf["matrix"]

    embed_model = SentenceTransformer(args.embed_model, device=args.device)

    if args.method == "dense":
        results = retrieve_dense(args.q, embed_model, index, chunks, args.k)
    elif args.method == "sparse":
        results = retrieve_sparse(args.q, vec, mat, chunks, args.k)
    else:
        # simple hybrid: normalize each score list then sum
        dense = retrieve_dense(args.q, embed_model, index, chunks, args.k * 5)
        sparse = retrieve_sparse(args.q, vec, mat, chunks, args.k * 5)

        d_scores = {c.chunk_id: s for s, c in dense}
        s_scores = {c.chunk_id: s for s, c in sparse}

        def norm(m: Dict[int, float]) -> Dict[int, float]:
            if not m:
                return {}
            vals = np.array(list(m.values()), dtype=np.float32)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-6:
                return {k: 1.0 for k in m}
            return {k: (v - lo) / (hi - lo) for k, v in m.items()}

        d_n = norm(d_scores)
        s_n = norm(s_scores)
        all_ids = set(d_n) | set(s_n)
        combined = []
        by_id = {c.chunk_id: c for c in chunks}
        for cid in all_ids:
            combined.append((float(d_n.get(cid, 0.0) + s_n.get(cid, 0.0)), by_id[cid]))
        combined.sort(key=lambda x: x[0], reverse=True)
        results = combined[: args.k]

    for rank, (score, ch) in enumerate(results, start=1):
        meta = ch.meta
        major = meta.get("major_category")
        service = meta.get("service")
        field = meta.get("field")
        pages = None
        if meta.get("page_start") and meta.get("page_end"):
            pages = f"p{meta['page_start']}-{meta['page_end']}"
        print(f"\n[{rank}] score={score:.4f}")
        if any([major, service, field, pages]):
            print("meta:", {k: v for k, v in [("major", major), ("service", service), ("field", field), ("pages", pages)] if v})
        snippet = ch.text
        if len(snippet) > args.max_chars:
            snippet = snippet[: args.max_chars] + "..."
        print(snippet)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe")
    sp.add_argument("--pdf", default=PDF_DEFAULT)
    sp.set_defaults(func=cmd_probe)

    sb = sub.add_parser("build")
    sb.add_argument("--pdf", default=PDF_DEFAULT)
    sb.add_argument("--loader", choices=["pdftotext", "pypdf"], default="pdftotext")
    sb.add_argument("--layout", action="store_true", default=True)
    sb.add_argument("--no-layout", dest="layout", action="store_false")
    sb.add_argument("--chunking", choices=["fields", "fixed"], default="fields")
    sb.add_argument("--chunk-chars", type=int, default=1200)
    sb.add_argument("--overlap", type=int, default=200)
    sb.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sb.add_argument("--device", default="cuda")
    sb.set_defaults(func=cmd_build)

    sq = sub.add_parser("query")
    sq.add_argument("-q", required=True)
    sq.add_argument("-k", type=int, default=5)
    sq.add_argument("--method", choices=["dense", "sparse", "hybrid"], default="hybrid")
    sq.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sq.add_argument("--device", default="cuda")
    sq.add_argument("--max-chars", type=int, default=800)
    sq.set_defaults(func=cmd_query)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
