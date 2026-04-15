from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .text_utils import normalize_text, strip_common_footer_noise
from .types import Document


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def fetch_url(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Prefer likely content containers.
    candidates = []
    for selector in [
        "div#content",
        "div#cms-content",
        "div.sub-content",
        "div.contents",
        "div.cont",
        "article",
        "main",
    ]:
        node = soup.select_one(selector)
        if node is not None:
            candidates.append(node)

    node = candidates[0] if candidates else soup.body or soup

    # Remove scripts/styles/nav/footer-ish blocks
    for tag in node.find_all(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = node.get_text("\n", strip=True)
    text = normalize_text(text)
    text = strip_common_footer_noise(text)
    return text


def load_web_document(url: str, title: Optional[str] = None, raw_html_path: Optional[Path] = None) -> Document:
    html = fetch_url(url)
    if raw_html_path is not None:
        raw_html_path.parent.mkdir(parents=True, exist_ok=True)
        raw_html_path.write_text(html, encoding="utf-8")

    text = html_to_text(html)
    if not title:
        # Try to grab title tag
        soup = BeautifulSoup(html, "lxml")
        t = soup.find("title")
        title = (t.get_text(strip=True) if t else url)

    return Document(
        doc_id=_stable_id("web", url),
        source=url,
        title=title,
        text=text,
        metadata={"type": "web", "url": url},
    )


def load_pdf_document(pdf_path: str | Path, title: Optional[str] = None) -> Document:
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = normalize_text(t)
        if t:
            pages_text.append(f"[page {i+1}]\n{t}")

    full_text = "\n\n".join(pages_text).strip()
    if not title:
        title = pdf_path.stem

    return Document(
        doc_id=_stable_id("pdf", str(pdf_path.resolve())),
        source=str(pdf_path),
        title=title,
        text=full_text,
        metadata={"type": "pdf", "path": str(pdf_path)},
    )
