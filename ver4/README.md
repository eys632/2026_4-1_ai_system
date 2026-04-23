# ver4: PDF-RAG Full Factorial Experiment Lab

이 프로젝트는 한국어 복지서비스 PDF를 대상으로, RAG 파이프라인을 단계별 축으로 분해해
전 조합(full factorial)을 실행하고, 모든 질문-답변-근거-평가 결과를 완전 복원 가능하게 저장하는 실험 프레임워크입니다.

핵심은 다음 한 문장입니다.

"어떤 조합이 어떤 질문에 왜 잘/못했는지, 나중에 100% 재구성할 수 있어야 한다."

## 이 프로젝트가 해결하는 문제

일반적인 RAG 실험은 아래 문제가 자주 발생합니다.

- 조합별 결과가 부분적으로만 저장되어 재현이 어려움
- retrieval 점수와 최종 답변 품질이 분리되어 해석이 어려움
- 실패(run abort) 이후 이어서 실행하기 어려움
- 발표용 best/worst 사례를 수동으로 골라야 함

ver4는 이를 해결하기 위해 설계되었습니다.

- 단계별 조합 자동 생성 및 전수 실행
- build 캐시 + run 체크포인트 + 실패 복구
- retrieval/자동지표/LLM judge/pairwise 비교 통합
- 질문 단위 best/worst 및 발표용 사례 자동 추출

## 데이터 특성 반영 포인트

대상 문서는 단순 산문이 아니라, 서비스/제도 안내 구조를 반복하는 복지 PDF입니다.

- 서비스명/제도명/문의처/전화번호/신청방법 등 정확 매칭 중요
- 대상/내용/방법/문의와 같은 필드 구조 중요
- 표/목록/레이아웃 노이즈(footer, page number) 영향 큼

그래서 ver4는 일반적 RAG 구성 외에 아래를 실험 축으로 명시 포함합니다.

- field-aware cleaning
- field-aware chunking
- metadata-aware post-retrieval filtering

## 실험 설계(요약)

RAG 단계를 아래와 같이 분리해 조합합니다.

1. Loader / Extraction
2. Cleaning / Normalization / Structuring
3. Chunking
4. Representation / Indexing
5. Retrieval
6. Post-retrieval / Re-ranking / Filtering
7. Generation
8. Evaluation

기본 매트릭스는 아래 조합을 사용합니다.

- loader(3) x cleaning(3) x chunking(3) x representation(3)
- retrieval(3) x post_retrieval(3) x generation(3)

총 stage 조합 수:

- 2187

질문셋 3종(manual/auto/stress)까지 포함한 총 실행 단위:

- 6561

## 로컬 LLM 원칙

기본 설정은 외부 유료 API가 아니라 로컬 모델(vLLM/transformers) 기준입니다.

- generation backend: local_vllm 기본
- judge backend: local_vllm 기본
- 필요 시 local_transformers 또는 local_rule(smoke)로 대체 가능

모델 설정은 configs/experiment_matrix.json에서 관리합니다.

## 실행 환경

중요: 새 가상환경을 만들지 않습니다.

- conda env: /home/a202192020/miniconda3/envs/ys_conda1_env
- 내부 실행 스크립트는 use_ys_conda1_env.sh를 통해 해당 환경을 강제 사용합니다.

## 빠른 시작

```bash
cd /data2/a202192020/4-1/ai_sys/2026_4-1_ai_system/ver4
source /home/a202192020/miniconda3/bin/activate ys_conda1_env
pip install -r requirements.txt
```

전체 파이프라인 순차 실행:

```bash
bash run_all.sh
```

단계별 실행:

```bash
bash run_builds.sh
bash run_experiments.sh
bash run_eval.sh
bash run_judge.sh
bash run_pairwise.sh
bash export_best_worst.sh
```

실패 run만 재개:

```bash
bash resume_failed.sh
```

## 장시간 실험 운영 팁

전 조합 실행은 오래 걸릴 수 있습니다. 아래 흐름을 권장합니다.

1. smoke 설정으로 end-to-end 연결 검증
2. build 선계산(run_builds)
3. 본 매트릭스(run_experiments) 장기 실행
4. 실패 발생 시 resume_failed 재개
5. 실험 완료 후 eval/judge/pairwise/export 실행

## 결과 해석 관점

이 프로젝트는 단일 점수보다 "질문별 차이"를 중요하게 봅니다.

- retrieval 중심: hit@k, mrr, ndcg@k
- answer 자동지표: exact_match, token_f1, groundedness, abstention
- judge 지표: intent/correctness/groundedness/completeness 등
- pairwise: 동일 질문에서 조합 간 우열 비교

발표용 산출물은 자동 생성됩니다.

- results/best_examples.jsonl
- results/worst_examples.jsonl
- results/pairwise_examples.jsonl
- results/presentation_cases.md

## 프로젝트 구조

```text
ver4/
	configs/
		experiment_matrix.json
		generation_prompts.yaml
		judge_prompts.yaml
		scoring_weights.yaml
	data/
		questions/
			manual_questions.jsonl
			auto_questions.jsonl
			stress_questions.jsonl
	artifacts/
		builds/build_<hash>/
		runs/run_<hash>/
	results/
		experiment_results.csv
		experiment_results.parquet
		leaderboards/
		best_examples.jsonl
		worst_examples.jsonl
		pairwise_examples.jsonl
		presentation_cases.md
	docs/
		experiment_design.md
		how_to_run.md
		result_schema.md
		evaluation_protocol.md
		final_report.md
	src/raglab_exhaustive/
```

## 주요 CLI

- python -m src.raglab_exhaustive.build_all
- python -m src.raglab_exhaustive.run_matrix
- python -m src.raglab_exhaustive.evaluate_answers
- python -m src.raglab_exhaustive.judge_answers
- python -m src.raglab_exhaustive.compare_pairwise
- python -m src.raglab_exhaustive.export_best_worst
- python -m src.raglab_exhaustive.export_human_eval
- python -m src.raglab_exhaustive.final_report
- python -m src.raglab_exhaustive.resume_failed

## 문서

상세 설계와 프로토콜은 docs 폴더를 참고하세요.

- docs/experiment_design.md
- docs/how_to_run.md
- docs/result_schema.md
- docs/evaluation_protocol.md
- docs/final_report.md
