from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from tqdm import tqdm

from jj_rag.config import load_settings
from jj_rag.io_utils import write_jsonl
from jj_rag.loaders import load_pdf_document, load_web_document


def main() -> None:
    load_dotenv()
    settings = load_settings()

    web_urls = [
        ("학과안내-인사말", "https://ai.jj.ac.kr/ai/info/greeting.do"),
        ("학과안내-학과소개", "https://ai.jj.ac.kr/ai/info/intro.do"),
        ("학과안내-교수진소개", "https://ai.jj.ac.kr/ai/info/faculty.do"),
        ("학과안내-강의실습실", "https://ai.jj.ac.kr/ai/info/lab.do"),
        ("학과안내-행정안내", "https://ai.jj.ac.kr/ai/info/admin.do"),
    ]

    docs = []

    web_raw_dir = settings.data_raw_dir / "web"
    web_raw_dir.mkdir(parents=True, exist_ok=True)

    for title, url in tqdm(web_urls, desc="Collect web"):
        try:
            doc = load_web_document(url, title=title, raw_html_path=web_raw_dir / f"{title}.html")
            docs.append(
                {
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "title": doc.title,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
            )
        except Exception as e:
            print(f"⚠️  Skip {title} ({url}): {e}")

    pdf_path = settings.project_root.parent / "2025 나에게  힘이 되는 복지서비스.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_doc = load_pdf_document(pdf_path)
    docs.append(
        {
            "doc_id": pdf_doc.doc_id,
            "source": pdf_doc.source,
            "title": pdf_doc.title,
            "text": pdf_doc.text,
            "metadata": pdf_doc.metadata,
        }
    )

    out_path = settings.data_processed_dir / "documents.jsonl"
    write_jsonl(out_path, docs)

    summary = {
        "documents": len(docs),
        "processed_path": str(out_path),
        "raw_web_dir": str(web_raw_dir),
        "pdf_path": str(pdf_path),
    }
    (settings.data_processed_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("✅ collect_data complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
