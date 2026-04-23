from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pypdf import PdfReader


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
PAGE_NO_RE = re.compile(r"^\s*\d{1,4}\s*$")
BOOK_FOOTER_RE = re.compile(r"^\s*\d{1,4}\s+2025\s+나에게\s+힘이\s+되는\s+복지서비스\s*$")
FIELD_RE = re.compile(r"^\s*(대상|내용|방법|문의)\b")
PHONE_RE = re.compile(r"(\d{2,4}-\d{3,4}-\d{4})")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:원|만원|천원)")
CITATION_RE = re.compile(r"\[근거\s*(\d+)\]")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    meta: Dict[str, Any]


@dataclass
class Section:
    section_id: int
    service: Optional[str]
    field: Optional[str]
    major_category: Optional[str]
    page_start: int
    page_end: int
    text: str


@dataclass
class RetrievalItem:
    rank: int
    score: float
    chunk: Chunk


_EMBEDDER_CACHE: Dict[str, Any] = {}


def _get_embedder(model_name: str, device: str = "cuda"):
    cache_key = f"{model_name}:{device}"
    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    _EMBEDDER_CACHE[cache_key] = model
    return model


def _run_pdftotext(pdf_path: Path, txt_path: Path, layout: bool = True) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["pdftotext", "-enc", "UTF-8"]
    if layout:
        cmd.append("-layout")
    cmd += [str(pdf_path), str(txt_path)]
    subprocess.run(cmd, check=True)


def extract_pages(pdf_path: Path, loader_name: str, cache_dir: Path) -> List[str]:
    if loader_name == "pdftotext_layout":
        out_txt = cache_dir / f"{pdf_path.stem}.layout.txt"
        if not out_txt.exists():
            _run_pdftotext(pdf_path, out_txt, layout=True)
        text = out_txt.read_text(encoding="utf-8", errors="replace")
        pages = text.split("\f")
        while pages and not pages[-1].strip():
            pages.pop()
        return pages

    if loader_name == "pymupdf":
        try:
            import fitz
        except Exception:
            # Graceful fallback when pymupdf is unavailable.
            loader_name = "pypdf"
        else:
            doc = fitz.open(str(pdf_path))
            return [page.get_text("text") or "" for page in doc]

    if loader_name == "pypdf":
        reader = PdfReader(str(pdf_path))
        pages: List[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages

    raise ValueError(f"Unsupported loader_name: {loader_name}")


def _normalize_line(line: str) -> str:
    line = CONTROL_RE.sub("", line)
    line = line.replace("è", "->")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def clean_pages(raw_pages: List[str], method: str) -> List[str]:
    if method not in {"minimal", "regex_rules", "field_layout_aware"}:
        raise ValueError(f"Unsupported cleaning method: {method}")

    cleaned_per_page: List[List[str]] = []
    all_lines_flat: List[str] = []

    for page_text in raw_pages:
        lines: List[str] = []
        for raw in page_text.splitlines():
            ln = raw.rstrip("\n")
            ln = CONTROL_RE.sub("", ln)
            if method in {"regex_rules", "field_layout_aware"}:
                if PAGE_NO_RE.match(ln) or BOOK_FOOTER_RE.match(ln):
                    continue
            lines.append(ln)
        cleaned_per_page.append(lines)
        all_lines_flat.extend([_normalize_line(x) for x in lines if _normalize_line(x)])

    if method == "field_layout_aware":
        freq: Dict[str, int] = {}
        for ln in all_lines_flat:
            freq[ln] = freq.get(ln, 0) + 1

        # Repeated boilerplate headers/footers often appear across many pages.
        repeated_noise = {ln for ln, cnt in freq.items() if cnt >= 12 and len(ln) <= 40}
    else:
        repeated_noise = set()

    out_pages: List[str] = []
    for lines in cleaned_per_page:
        out_lines: List[str] = []
        blank_run = 0
        for ln in lines:
            norm = _normalize_line(ln)
            if method == "field_layout_aware" and norm in repeated_noise:
                continue

            if not norm:
                blank_run += 1
                if blank_run <= 2:
                    out_lines.append("")
                continue

            blank_run = 0
            out_lines.append(ln.strip())

        out_pages.append("\n".join(out_lines).strip())

    return out_pages


def _is_major_heading(line: str) -> bool:
    s = re.sub(r"\s+", " ", line.strip())
    return bool(re.match(r"^[가-힣A-Za-z·\s]{2,30}지원$", s))


def infer_sections(cleaned_pages: List[str]) -> List[Section]:
    sections: List[Section] = []

    current_major: Optional[str] = None
    current_service: Optional[str] = None
    current_field: Optional[str] = None
    buffer: List[str] = []
    section_page_start = 1
    section_id = 1
    prev_nonempty: Optional[str] = None

    def flush(end_page: int) -> None:
        nonlocal section_id, buffer, section_page_start
        if not current_service or not current_field:
            buffer = []
            section_page_start = end_page
            return
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            section_page_start = end_page
            return
        sections.append(
            Section(
                section_id=section_id,
                service=current_service,
                field=current_field,
                major_category=current_major,
                page_start=section_page_start,
                page_end=end_page,
                text=content,
            )
        )
        section_id += 1
        buffer = []
        section_page_start = end_page

    for page_no, page_text in enumerate(cleaned_pages, start=1):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if _is_major_heading(line):
                current_major = re.sub(r"\s+", " ", line)
                prev_nonempty = line
                continue

            m = FIELD_RE.match(line)
            if m:
                field = m.group(1)
                if field == "대상" and prev_nonempty and prev_nonempty not in {"대상", "내용", "방법", "문의"}:
                    flush(page_no)
                    current_service = re.sub(r"\s+", " ", prev_nonempty)
                    current_field = field
                    section_page_start = page_no
                    continue

                if current_service:
                    flush(page_no)
                    current_field = field
                    section_page_start = page_no
                    continue

            if current_service and current_field:
                buffer.append(line)

            prev_nonempty = line

    flush(len(cleaned_pages))
    return sections


def chunk_fixed(cleaned_pages: List[str], chunk_chars: int = 1200, overlap: int = 200) -> List[Chunk]:
    full = "\n\n".join([p for p in cleaned_pages if p.strip()]).strip()
    chunks: List[Chunk] = []
    i = 0
    chunk_id = 1
    while i < len(full):
        j = min(len(full), i + chunk_chars)
        text = full[i:j]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                meta={
                    "chunking": "fixed_size",
                    "start_char": i,
                    "end_char": j,
                },
            )
        )
        chunk_id += 1
        if j >= len(full):
            break
        i = max(0, j - overlap)
    return chunks


def chunk_recursive(cleaned_pages: List[str], target_chars: int = 1100) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 1

    for page_no, page_text in enumerate(cleaned_pages, start=1):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        buf: List[str] = []
        size = 0

        for para in paragraphs:
            if size + len(para) <= target_chars:
                buf.append(para)
                size += len(para)
                continue

            if buf:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text="\n\n".join(buf),
                        meta={
                            "chunking": "recursive_paragraph",
                            "page_start": page_no,
                            "page_end": page_no,
                        },
                    )
                )
                chunk_id += 1
                buf = []
                size = 0

            if len(para) > target_chars:
                # Split long paragraphs into sentence-like segments.
                bits = re.split(r"(?<=[.!?다요])\s+", para)
                sub: List[str] = []
                sub_size = 0
                for bit in bits:
                    if not bit:
                        continue
                    if sub_size + len(bit) <= target_chars:
                        sub.append(bit)
                        sub_size += len(bit)
                    else:
                        if sub:
                            chunks.append(
                                Chunk(
                                    chunk_id=chunk_id,
                                    text=" ".join(sub),
                                    meta={
                                        "chunking": "recursive_paragraph",
                                        "page_start": page_no,
                                        "page_end": page_no,
                                    },
                                )
                            )
                            chunk_id += 1
                        sub = [bit]
                        sub_size = len(bit)
                if sub:
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=" ".join(sub),
                            meta={
                                "chunking": "recursive_paragraph",
                                "page_start": page_no,
                                "page_end": page_no,
                            },
                        )
                    )
                    chunk_id += 1
            else:
                buf = [para]
                size = len(para)

        if buf:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text="\n\n".join(buf),
                    meta={
                        "chunking": "recursive_paragraph",
                        "page_start": page_no,
                        "page_end": page_no,
                    },
                )
            )
            chunk_id += 1

    return chunks


def chunk_field_aware(sections: List[Section]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for idx, sec in enumerate(sections, start=1):
        header = f"[{sec.major_category}] {sec.service} - {sec.field}" if sec.major_category else f"{sec.service} - {sec.field}"
        chunks.append(
            Chunk(
                chunk_id=idx,
                text=f"{header}\n{sec.text}",
                meta={
                    "chunking": "field_aware",
                    "service": sec.service,
                    "field": sec.field,
                    "major_category": sec.major_category,
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                    "section_id": sec.section_id,
                },
            )
        )
    return chunks


def chunk_documents(cleaned_pages: List[str], sections: List[Section], chunking: str) -> List[Chunk]:
    if chunking == "fixed_size":
        return chunk_fixed(cleaned_pages)
    if chunking == "recursive_paragraph":
        return chunk_recursive(cleaned_pages)
    if chunking == "field_aware":
        return chunk_field_aware(sections)
    raise ValueError(f"Unsupported chunking: {chunking}")


def save_pages_jsonl(pages: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(pages, start=1):
            rec = {"page": i, "chars": len(text), "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_sections_jsonl(sections: List[Section], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sec in sections:
            f.write(
                json.dumps(
                    {
                        "section_id": sec.section_id,
                        "service": sec.service,
                        "field": sec.field,
                        "major_category": sec.major_category,
                        "page_start": sec.page_start,
                        "page_end": sec.page_end,
                        "text": sec.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_chunks_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"chunk_id": ch.chunk_id, "text": ch.text, "meta": ch.meta}, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(Chunk(chunk_id=int(rec["chunk_id"]), text=rec["text"], meta=rec.get("meta", {})))
    chunks.sort(key=lambda c: c.chunk_id)
    return chunks


def build_representation(
    representation: str,
    chunks: List[Chunk],
    build_dir: Path,
    embed_model: str,
    embed_device: str,
) -> Dict[str, Any]:
    texts = [c.text for c in chunks]
    out: Dict[str, Any] = {
        "representation": representation,
        "n_chunks": len(chunks),
        "has_dense": False,
        "has_sparse": False,
        "embed_model": embed_model,
        "embed_device": embed_device,
    }

    if representation in {"dense_minilm", "hybrid_dual"}:
        embedder = _get_embedder(embed_model, embed_device)
        emb = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.save(build_dir / "embeddings.npy", emb)

        import faiss

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(build_dir / "index_dense.faiss"))
        out["has_dense"] = True

    if representation in {"sparse_tfidf", "hybrid_dual"}:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=250_000, ngram_range=(1, 2), lowercase=False)
        mat = vec.fit_transform(texts)
        import joblib

        joblib.dump({"vectorizer": vec, "matrix": mat}, build_dir / "sparse.joblib")
        out["has_sparse"] = True

    return out


def _retrieve_dense(query: str, chunks: List[Chunk], build_dir: Path, embed_model: str, embed_device: str, top_k: int) -> List[Tuple[float, Chunk]]:
    import faiss

    index = faiss.read_index(str(build_dir / "index_dense.faiss"))
    embedder = _get_embedder(embed_model, embed_device)
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, int(top_k))
    out: List[Tuple[float, Chunk]] = []
    for s, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        out.append((float(s), chunks[int(idx)]))
    return out


def _retrieve_sparse(query: str, chunks: List[Chunk], build_dir: Path, top_k: int) -> List[Tuple[float, Chunk]]:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity

    sparse = joblib.load(build_dir / "sparse.joblib")
    vec = sparse["vectorizer"]
    mat = sparse["matrix"]

    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()
    if top_k >= len(chunks):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, int(top_k))[: int(top_k)]
        idx = idx[np.argsort(-sims[idx])]
    return [(float(sims[int(i)]), chunks[int(i)]) for i in idx]


def retrieve(
    retrieval: str,
    query: str,
    chunks: List[Chunk],
    build_dir: Path,
    top_k: int,
    embed_model: str,
    embed_device: str,
) -> List[Tuple[float, Chunk]]:
    cand_k = max(int(top_k) * 5, 30)

    if retrieval == "sparse":
        if not (build_dir / "sparse.joblib").exists():
            return []
        return _retrieve_sparse(query, chunks, build_dir, cand_k)

    if retrieval == "dense":
        if not (build_dir / "index_dense.faiss").exists():
            return []
        return _retrieve_dense(query, chunks, build_dir, embed_model, embed_device, cand_k)

    if retrieval == "hybrid":
        if not (build_dir / "sparse.joblib").exists() or not (build_dir / "index_dense.faiss").exists():
            return []
        dense = _retrieve_dense(query, chunks, build_dir, embed_model, embed_device, cand_k)
        sparse = _retrieve_sparse(query, chunks, build_dir, cand_k)

        d_scores = {c.chunk_id: s for s, c in dense}
        s_scores = {c.chunk_id: s for s, c in sparse}

        def norm(m: Dict[int, float]) -> Dict[int, float]:
            if not m:
                return {}
            vals = np.array(list(m.values()), dtype=np.float32)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-6:
                return {k: 1.0 for k in m}
            return {k: float((v - lo) / (hi - lo)) for k, v in m.items()}

        d_n = norm(d_scores)
        s_n = norm(s_scores)
        by_id = {c.chunk_id: c for c in chunks}
        all_ids = set(d_n) | set(s_n)
        combined = [(float(d_n.get(cid, 0.0) + s_n.get(cid, 0.0)), by_id[cid]) for cid in all_ids]
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined

    raise ValueError(f"Unsupported retrieval: {retrieval}")


def infer_field(question: str) -> Optional[str]:
    q = question.replace(" ", "")
    if any(k in q for k in ["문의", "연락", "전화", "콜센터", "번호"]):
        return "문의"
    if any(k in q for k in ["신청", "방법", "절차", "접수", "어떻게"]):
        return "방법"
    if any(k in q for k in ["대상", "자격", "조건", "누가", "누구"]):
        return "대상"
    if any(k in q for k in ["내용", "지원", "혜택", "금액", "본인부담", "얼마", "비율"]):
        return "내용"
    return None


def post_retrieve(post_name: str, query: str, scored: List[Tuple[float, Chunk]], top_k: int) -> List[RetrievalItem]:
    if not scored:
        return []

    candidates = [c for _, c in scored]
    top_service = (candidates[0].meta or {}).get("service")
    field_hint = infer_field(query)

    if post_name == "none":
        return [RetrievalItem(rank=i + 1, score=float(s), chunk=c) for i, (s, c) in enumerate(scored[:top_k])]

    if post_name == "metadata_field_filter":
        filtered: List[Tuple[float, Chunk]] = []
        for s, c in scored:
            m = c.meta or {}
            if top_service and m.get("service") != top_service:
                continue
            if field_hint and m.get("field") != field_hint:
                continue
            filtered.append((s, c))
        if not filtered:
            filtered = [(s, c) for s, c in scored if (c.meta or {}).get("service") == top_service] or scored
        return [RetrievalItem(rank=i + 1, score=float(s), chunk=c) for i, (s, c) in enumerate(filtered[:top_k])]

    if post_name == "heuristic_rerank":
        boosts: List[Tuple[float, Chunk]] = []
        for s, c in scored:
            bonus = 0.0
            m = c.meta or {}
            txt = c.text
            if top_service and m.get("service") == top_service:
                bonus += 0.25
            if field_hint and m.get("field") == field_hint:
                bonus += 0.20
            if PHONE_RE.search(query) and PHONE_RE.search(txt):
                bonus += 0.15
            if PERCENT_RE.search(query) and PERCENT_RE.search(txt):
                bonus += 0.15
            if MONEY_RE.search(query) and MONEY_RE.search(txt):
                bonus += 0.15
            if m.get("service") and str(m.get("service")) in query:
                bonus += 0.20
            boosts.append((float(s + bonus), c))
        boosts.sort(key=lambda x: x[0], reverse=True)
        return [RetrievalItem(rank=i + 1, score=float(s), chunk=c) for i, (s, c) in enumerate(boosts[:top_k])]

    raise ValueError(f"Unsupported post_name: {post_name}")


def build_context(retrieved: Sequence[RetrievalItem], max_chars: int = 9000) -> str:
    parts: List[str] = []
    total = 0
    for r in retrieved:
        m = r.chunk.meta or {}
        pages = None
        if m.get("page_start") and m.get("page_end"):
            if m["page_start"] == m["page_end"]:
                pages = f"p{m['page_start']}"
            else:
                pages = f"p{m['page_start']}-{m['page_end']}"
        block = (
            f"[EVID {r.rank} | chunk_id={r.chunk.chunk_id} | service={m.get('service')} | "
            f"field={m.get('field')} | pages={pages} | score={r.score:.4f}]\n{r.chunk.text.strip()}"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_generation_prompt(template: str, question: str, context: str) -> str:
    return template.format(question=question, context=context)


def normalize_answer(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "문서에서 확인되지 않습니다"

    if any("문서에서 확인되지 않습니다" in ln for ln in lines):
        return "문서에서 확인되지 않습니다"

    bullets = [ln for ln in lines if ln.startswith("-")]
    if not bullets:
        bullets = [f"- {ln}" for ln in lines[:4]]

    out: List[str] = []
    for ln in bullets[:6]:
        if not CITATION_RE.search(ln):
            ln = ln + " [근거 1]"
        out.append(ln)
    return "\n".join(out)


def extract_citations(answer_text: str) -> List[int]:
    ids: List[int] = []
    for m in CITATION_RE.finditer(answer_text or ""):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            continue
    return ids
