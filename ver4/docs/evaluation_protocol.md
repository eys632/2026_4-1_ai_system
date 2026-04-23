# Evaluation Protocol

## Retrieval Metrics
- Hit@1 / Hit@3 / Hit@5
- MRR
- nDCG@5
- Recall@5

정답 매칭 기준(우선순위):
1) gold evidence chunk id
2) gold service + gold field
3) gold page overlap

## Answer-level Auto Metrics
- exact_match
- token_f1
- answer_similarity
- citation_overlap
- groundedness_heuristic
- unsupported_claim_heuristic
- abstention_correct
- phone/percent/money regex match
- answer_length

## LLM-as-a-Judge
- intent_fulfillment
- correctness
- groundedness
- completeness
- readability
- abstention_safety
- overall_preference
- comment

판정은 JSON 강제 출력 + temperature 0.0으로 고정

## Pairwise Judge
시나리오:
- overall best vs overall worst
- retrieval best vs answer best

출력:
- preferred_run_id
- preferred_answer
- pairwise_reason
- pairwise_dimension_scores
