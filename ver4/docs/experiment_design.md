# Experiment Design (ver4)

## 1) RAG 단계 정의

| 단계 | 역할 | 이 PDF에서 중요한 이유 |
|---|---|---|
| Loader / Extraction | PDF에서 텍스트를 페이지 단위 추출 | 표/목록/레이아웃 보존 여부가 retrieval 품질에 직접 영향 |
| Cleaning / Normalization / Structuring | footer/page-no 제거, 필드 라벨 유지, 반복 문구 정리 | 문의처/전화번호/지원금액 등 정확 추출에서 잡음 제거가 핵심 |
| Chunking | 검색 단위 문서 조각 생성 | 대상/내용/방법/문의 필드 보존 여부가 정밀 검색 성능을 좌우 |
| Representation / Indexing | sparse/dense/hybrid 인덱스 생성 | exact match 질의와 의미 질의를 동시에 커버하려면 다중 표현 필요 |
| Retrieval | top-k 근거 탐색 | 서비스명 exact 질의와 paraphrase 질의를 함께 처리해야 함 |
| Post-retrieval | 필드/서비스 필터링 및 재정렬 | "신청방법", "문의처" 질의에서 field-aware 후처리가 큰 이점 |
| Generation | 근거 기반 답변 생성/abstain | 문서 외 추론 억제와 인용 규칙 준수가 필수 |
| Evaluation | retrieval+answer+judge+pairwise 종합 | best/worst를 질문 단위로 복원하려면 다층 평가 필요 |

## 2) 단계별 후보

| 단계 | 후보 |
|---|---|
| Loader | pdftotext_layout, pymupdf, pypdf |
| Cleaning | minimal, regex_rules, field_layout_aware |
| Chunking | fixed_size, recursive_paragraph, field_aware |
| Representation | sparse_tfidf, dense_minilm, hybrid_dual |
| Retrieval | sparse, dense, hybrid |
| Post-retrieval | none, metadata_field_filter, heuristic_rerank |
| Generation | strict_grounded, structured_field_grounded, concise_grounded |

## 3) Full Factorial

- stage combinations = 3^7 = 2187
- dataset split(manual/auto/stress)까지 포함한 실행 = 2187 x 3 = 6561

## 4) 재현성/복구 설계

- build hash, run hash 저장
- artifacts/builds 캐시 재사용
- checkpoint 기반 question-level resume
- failed run 재시도 모드(resume_failed)
- judge only 재실행 가능(run_judge)
- pairwise only 재실행 가능(run_pairwise)
