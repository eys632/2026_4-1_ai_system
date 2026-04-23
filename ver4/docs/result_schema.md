# Result Schema

## artifacts/builds/build_<hash>/
- build_meta.json
- pages.jsonl
- sections.jsonl
- chunks.jsonl
- embeddings.npy (dense/hybrid)
- index_dense.faiss (dense/hybrid)
- sparse.joblib (sparse/hybrid)

## artifacts/runs/run_<hash>/
- run_meta.json
- checkpoint.json
- answers.jsonl
- retrievals.jsonl
- auto_scores.jsonl
- judge_scores.jsonl
- pairwise_scores.jsonl
- errors.jsonl

## results/experiment_results.csv 주요 컬럼
- run_id
- dataset_name
- question_id
- question_type
- loader_name
- cleaning_name
- chunking_name
- representation_name
- retrieval_name
- post_retrieval_name
- generation_name
- hyperparams_json
- answer_text
- hit@1, hit@3, hit@5, mrr, ndcg@5, recall@5
- exact_match, token_f1, answer_similarity
- citation_overlap, groundedness_heuristic, abstention_correct
- llm_judge_intent, llm_judge_correctness, llm_judge_groundedness
- llm_judge_completeness, llm_judge_readability, llm_judge_abstention
- llm_judge_overall, llm_judge_comment
- overall_score, elapsed_sec, success
