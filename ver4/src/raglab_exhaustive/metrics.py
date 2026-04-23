from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9%.-]+")
PHONE_RE = re.compile(r"\d{2,4}-\d{3,4}-\d{4}")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:원|만원|천원)")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if (pred or "").strip() == (gold or "").strip() else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p or not g:
        return 0.0
    pc = Counter(p)
    gc = Counter(g)
    common = sum((pc & gc).values())
    precision = safe_div(common, len(p))
    recall = safe_div(common, len(g))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def answer_similarity(pred: str, gold: str) -> float:
    return float(SequenceMatcher(None, pred or "", gold or "").ratio())


def groundedness_heuristic(answer: str, context: str) -> float:
    at = tokenize(answer)
    ct = set(tokenize(context))
    if not at:
        return 0.0
    supported = sum(1 for t in at if t in ct)
    return safe_div(supported, len(at))


def unsupported_claim_heuristic(answer: str, context: str) -> float:
    return 1.0 - groundedness_heuristic(answer, context)


def citation_overlap(citation_chunk_ids: Sequence[int], gold_chunk_ids: Sequence[int]) -> float:
    if not citation_chunk_ids or not gold_chunk_ids:
        return 0.0
    c = set(int(x) for x in citation_chunk_ids)
    g = set(int(x) for x in gold_chunk_ids)
    return safe_div(len(c & g), len(g))


def abstained(answer_text: str) -> bool:
    return "문서에서 확인되지 않습니다" in (answer_text or "")


def abstention_correct(answer_text: str, requires_abstention: bool) -> float:
    is_abs = abstained(answer_text)
    return 1.0 if bool(is_abs) == bool(requires_abstention) else 0.0


def regex_exactness(answer_text: str, gold_answer: str) -> Dict[str, float]:
    pred_phone = set(PHONE_RE.findall(answer_text or ""))
    gold_phone = set(PHONE_RE.findall(gold_answer or ""))

    pred_pct = set(PERCENT_RE.findall(answer_text or ""))
    gold_pct = set(PERCENT_RE.findall(gold_answer or ""))

    pred_money = set(MONEY_RE.findall(answer_text or ""))
    gold_money = set(MONEY_RE.findall(gold_answer or ""))

    return {
        "phone_match": 1.0 if gold_phone and pred_phone == gold_phone else 0.0,
        "percent_match": 1.0 if gold_pct and pred_pct == gold_pct else 0.0,
        "money_match": 1.0 if gold_money and pred_money == gold_money else 0.0,
    }


def dcg(rels: Sequence[float]) -> float:
    score = 0.0
    for i, rel in enumerate(rels, start=1):
        score += (2.0 ** float(rel) - 1.0) / math.log2(i + 1)
    return score


def retrieval_metrics_from_hits(hits_at_k: Sequence[int], first_hit_rank: Optional[int], k: int) -> Dict[str, float]:
    hit1 = 1.0 if len(hits_at_k) >= 1 and hits_at_k[0] else 0.0
    hit3 = 1.0 if any(hits_at_k[: min(3, len(hits_at_k))]) else 0.0
    hit5 = 1.0 if any(hits_at_k[: min(5, len(hits_at_k))]) else 0.0

    rr = 0.0 if not first_hit_rank else 1.0 / float(first_hit_rank)

    rels = [1.0 if x else 0.0 for x in hits_at_k[:k]]
    idcg = dcg(sorted(rels, reverse=True))
    ndcg = 0.0 if idcg == 0 else dcg(rels) / idcg

    recall = safe_div(sum(rels), max(1.0, float(sum(rels) if sum(rels) > 0 else 1.0)))
    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "mrr": rr,
        "ndcg@5": ndcg,
        "recall@5": recall,
    }


def evaluate_answer(
    *,
    question: Dict[str, Any],
    answer_text: str,
    context_text: str,
    citation_chunk_ids: Sequence[int],
    evidence_chunk_ids: Sequence[int],
) -> Dict[str, float]:
    gold_answer = str(question.get("gold_answer", "") or "")
    requires_abstention = bool(question.get("requires_abstention", False))

    em = exact_match(answer_text, gold_answer) if gold_answer else 0.0
    f1 = token_f1(answer_text, gold_answer) if gold_answer else 0.0
    sim = answer_similarity(answer_text, gold_answer) if gold_answer else 0.0

    c_overlap = citation_overlap(citation_chunk_ids, evidence_chunk_ids)
    grounded = groundedness_heuristic(answer_text, context_text)
    abs_corr = abstention_correct(answer_text, requires_abstention)

    regex_scores = regex_exactness(answer_text, gold_answer)

    return {
        "exact_match": em,
        "token_f1": f1,
        "answer_similarity": sim,
        "citation_overlap": c_overlap,
        "groundedness_heuristic": grounded,
        "abstention_correct": abs_corr,
        "unsupported_claim_heuristic": unsupported_claim_heuristic(answer_text, context_text),
        "answer_length": float(len(answer_text or "")),
        **regex_scores,
    }


def weighted_overall_score(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    total_w = 0.0
    score = 0.0
    for key, w in weights.items():
        if key not in row:
            continue
        total_w += float(w)
        score += float(w) * float(row.get(key, 0.0) or 0.0)
    return score / total_w if total_w > 0 else 0.0
