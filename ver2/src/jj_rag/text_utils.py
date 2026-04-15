from __future__ import annotations

import re


_whitespace_re = re.compile(r"[\t\r\f\v ]+")
_many_newlines_re = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _whitespace_re.sub(" ", text)
    text = _many_newlines_re.sub("\n\n", text)
    return text.strip()


def strip_common_footer_noise(text: str) -> str:
    # Heuristic: remove repeated site footer chunks if present.
    for marker in ["COPYRIGHTS", "개인정보처리방침", "이메일무단수집거부", "홈페이지 관리 담당:"]:
        idx = text.find(marker)
        if idx != -1 and idx > 500:
            text = text[:idx].rstrip()
    return text
