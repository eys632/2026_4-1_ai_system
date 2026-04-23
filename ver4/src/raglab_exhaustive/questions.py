from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import write_jsonl
from .stages import Section


def _q(
    question_id: str,
    prompt_text: str,
    question_type: str,
    answer_type: str,
    *,
    gold_service: Optional[str] = None,
    gold_field: Optional[str] = None,
    gold_answer: Optional[str] = None,
    gold_page: Optional[int] = None,
    gold_evidence_chunk_ids: Optional[List[int]] = None,
    requires_abstention: bool = False,
    set_name: str = "manual",
) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "prompt_text": prompt_text,
        "question_type": question_type,
        "answer_type": answer_type,
        "gold_service": gold_service,
        "gold_field": gold_field,
        "gold_answer": gold_answer,
        "gold_page": gold_page,
        "gold_evidence_chunk_ids": gold_evidence_chunk_ids or [],
        "requires_abstention": requires_abstention,
        "set_name": set_name,
    }


def default_manual_questions() -> List[Dict[str, Any]]:
    return [
        _q("m001", "실업급여 신청 방법은 무엇인가요?", "방법", "fact", gold_service="실업급여", gold_field="방법"),
        _q("m002", "실업급여 지원 대상은 누구인가요?", "대상", "fact", gold_service="실업급여", gold_field="대상"),
        _q("m003", "건강보험 산정특례의 문의처는 어디인가요?", "문의", "phone", gold_service="건강보험 산정특례", gold_field="문의"),
        _q("m004", "건강보험 산정특례에서 암의 본인부담률은 얼마인가요?", "금액", "percentage", gold_service="건강보험 산정특례", gold_field="내용"),
        _q("m005", "맞춤형 기초생활보장제도의 지원 내용은 무엇인가요?", "내용", "fact", gold_service="맞춤형 기초생활보장제도", gold_field="내용"),
        _q("m006", "청년내일저축계좌 신청 절차를 단계별로 알려주세요.", "방법", "procedure", gold_field="방법"),
        _q("m007", "긴급복지지원제도는 어떤 조건에서 받을 수 있나요?", "조건", "fact", gold_field="대상"),
        _q("m008", "기초연금은 누가 받을 수 있나요?", "대상", "fact", gold_field="대상"),
        _q("m009", "장애인연금 신청 방법을 간단히 요약해 주세요.", "방법", "procedure", gold_field="방법"),
        _q("m010", "국민내일배움카드 문의 전화번호를 알려주세요.", "문의", "phone", gold_field="문의"),
        _q("m011", "보육료 지원의 지원 내용은 무엇인가요?", "내용", "fact", gold_field="내용"),
        _q("m012", "아이돌봄서비스 신청 방법은?", "방법", "procedure", gold_field="방법"),
        _q("m013", "한부모가족 지원사업의 대상 기준은 무엇인가요?", "대상", "fact", gold_field="대상"),
        _q("m014", "주거급여는 무엇을 지원하나요?", "내용", "fact", gold_field="내용"),
        _q("m015", "의료급여 신청은 어디에서 하나요?", "방법", "fact", gold_field="방법"),
        _q("m016", "노인일자리 사업 문의처를 알려주세요.", "문의", "phone", gold_field="문의"),
        _q("m017", "산모·신생아 건강관리 지원사업의 신청 절차는?", "방법", "procedure", gold_field="방법"),
        _q("m018", "청소년상담복지센터 이용 대상은 누구인가요?", "대상", "fact", gold_field="대상"),
        _q("m019", "국가장학금은 얼마를 지원하나요?", "금액", "money", gold_field="내용"),
        _q("m020", "실업급여를 외국인도 받을 수 있나요?", "조건", "fact", gold_service="실업급여", gold_field="대상"),
        _q("m021", "복지서비스 신청에 필요한 서류를 한 번에 정리해 주세요.", "헷갈림", "summary", gold_field="방법"),
        _q("m022", "산정특례 문의 연락처가 아니라 신청 기관을 알려주세요.", "헷갈림", "fact", gold_service="건강보험 산정특례", gold_field="방법"),
        _q("m023", "복지서비스 포인트를 카드로 충전하는 방법은 문서에 있나요?", "없는질문", "abstain", requires_abstention=True),
        _q("m024", "2026년 신설된 복지정책 신청 사이트를 알려주세요.", "없는질문", "abstain", requires_abstention=True),
        _q("m025", "지원 금액과 문의 전화번호를 동시에 알려주세요.", "복합", "multi_field", gold_field="내용"),
        _q("m026", "실업수당 신청법 알려줘", "paraphrase", "procedure", gold_service="실업급여", gold_field="방법"),
    ]


def _extract_answer_snippet(text: str, field: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if field == "문의":
        for ln in lines:
            if re.search(r"\d{2,4}-\d{3,4}-\d{4}", ln):
                return ln
    if field == "내용":
        for ln in lines:
            if re.search(r"\d|%|원", ln):
                return ln
    return lines[0]


def generate_auto_questions(
    sections: Sequence[Section],
    *,
    n_questions: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pairs: List[Tuple[str, str, int, str]] = []
    for sec in sections:
        if not sec.service or not sec.field:
            continue
        pairs.append((sec.service, sec.field, sec.page_start, sec.text))

    if not pairs:
        return []

    templates = {
        "대상": [
            "{s}의 지원 대상은 누구인가요?",
            "{s}는 어떤 자격 조건이 필요한가요?",
            "{s}를 받을 수 있는 사람 기준은?",
        ],
        "내용": [
            "{s}의 지원 내용은 무엇인가요?",
            "{s}는 무엇을 지원하나요?",
            "{s} 혜택을 요약해 주세요.",
            "{s} 본인부담률이나 금액 정보가 있나요?",
        ],
        "방법": [
            "{s} 신청 방법은 무엇인가요?",
            "{s} 신청 절차를 알려주세요.",
            "{s}는 어디에서 신청하나요?",
        ],
        "문의": [
            "{s} 문의처는 어디인가요?",
            "{s} 문의 전화번호 알려주세요.",
            "{s}는 어디에 연락하면 되나요?",
        ],
    }

    qset: List[Dict[str, Any]] = []
    for i in range(int(n_questions)):
        service, field, page, text = rng.choice(pairs)
        tpl = rng.choice(templates.get(field, ["{s} 관련 정보를 알려주세요."]))
        prompt = tpl.format(s=service)
        qset.append(
            _q(
                question_id=f"a{i+1:04d}",
                prompt_text=prompt,
                question_type=field,
                answer_type="fact",
                gold_service=service,
                gold_field=field,
                gold_answer=_extract_answer_snippet(text, field),
                gold_page=int(page),
                requires_abstention=False,
                set_name="auto",
            )
        )

    return qset


def generate_stress_questions(sections: Sequence[Section], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    services = sorted({sec.service for sec in sections if sec.service})
    out: List[Dict[str, Any]] = []

    negatives = [
        "복지서비스 포인트를 코인으로 환전하는 방법이 있나요?",
        "해외 거주자 전용 신규 복지급여 신청 링크는 무엇인가요?",
        "챗봇으로 자동 접수 가능한 서비스가 있나요?",
        "2027년 신설 청년주택포인트 제도는 어디서 신청하나요?",
        "복지서비스 NFT 바우처가 문서에 나오나요?",
    ]
    for i, q in enumerate(negatives, start=1):
        out.append(
            _q(
                question_id=f"s{i:03d}",
                prompt_text=q,
                question_type="없는질문",
                answer_type="abstain",
                requires_abstention=True,
                set_name="stress",
            )
        )

    confuse_templates = [
        "{s}와 비슷한 이름의 {t}는 같은 제도인가요?",
        "{s} 문의처와 {t} 문의처를 각각 알려주세요.",
        "{s} 신청 방법을 {t}와 비교해 주세요.",
    ]

    for j in range(10):
        if len(services) < 2:
            break
        s, t = rng.sample(services, 2)
        tpl = rng.choice(confuse_templates)
        out.append(
            _q(
                question_id=f"s{j+100:03d}",
                prompt_text=tpl.format(s=s, t=t),
                question_type="헷갈림",
                answer_type="comparison",
                gold_service=s,
                set_name="stress",
            )
        )

    exact_templates = [
        "{s}의 문의 전화번호를 정확히 알려주세요.",
        "{s}의 본인부담률(%)을 정확히 알려주세요.",
        "{s} 지원 금액을 숫자 포함해 알려주세요.",
    ]
    pick = services[: min(10, len(services))]
    for k, s in enumerate(pick, start=1):
        tpl = rng.choice(exact_templates)
        out.append(
            _q(
                question_id=f"s{200+k:03d}",
                prompt_text=tpl.format(s=s),
                question_type="정확추출",
                answer_type="numeric_or_phone",
                gold_service=s,
                set_name="stress",
            )
        )

    return out


def ensure_question_sets(
    *,
    out_dir: Path,
    sections: Sequence[Section],
    seed: int,
    auto_n: int = 120,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manual = default_manual_questions()
    auto = generate_auto_questions(sections, n_questions=auto_n, seed=seed)
    stress = generate_stress_questions(sections, seed=seed)

    manual_path = out_dir / "manual_questions.jsonl"
    auto_path = out_dir / "auto_questions.jsonl"
    stress_path = out_dir / "stress_questions.jsonl"

    write_jsonl(manual_path, manual)
    write_jsonl(auto_path, auto)
    write_jsonl(stress_path, stress)

    return {
        "manual": manual_path,
        "auto": auto_path,
        "stress": stress_path,
    }
