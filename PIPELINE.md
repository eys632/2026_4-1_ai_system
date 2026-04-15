# 📊 JJ RAG 파이프라인 상세 설명

## 목차
1. [전체 아키텍처](#전체-아키텍처)
2. [오프라인 단계 (인덱싱)](#오프라인-단계-인덱싱)
3. [온라인 단계 (질의)](#온라인-단계-질의)
4. [사용된 모델](#사용된-모델)
5. [기술 스택](#기술-스택)
6. [청킹 전략 비교](#청킹-전략-비교)

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    JJ RAG 시스템                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [OFFLINE PHASE]              [ONLINE PHASE]              │
│  (사전 준비)                   (질의 시점)                 │
│                                                             │
│  1. 데이터 수집          →    사용자 질문 입력             │
│  2. 청킹 (2가지)                    ↓                      │
│  3. 임베딩              →    질문 임베딩화                │
│  4. 벡터 인덱싱                     ↓                      │
│                          벡터 검색 (FAISS)                 │
│                                    ↓                       │
│                         근거 문서/청크 추출                │
│                                    ↓                       │
│                         프롬프트 조합 (LLM)               │
│                                    ↓                       │
│                      답변 생성 (Qwen 1.5B)               │
│                                    ↓                       │
│                         최종 답변 + 시간 표시             │
└─────────────────────────────────────────────────────────────┘
```

---

## 오프라인 단계 (인덱싱)

### 1단계: 데이터 수집 (`collect_data.py`)

**목표**: 웹페이지 + PDF에서 원본 텍스트 추출

**입력 데이터**:
- 📄 웹페이지 (3개):
  - `https://ai.jj.ac.kr/ai/info/greeting.do` (인사말)
  - `https://ai.jj.ac.kr/ai/info/intro.do` (학과소개)
  - `https://ai.jj.ac.kr/ai/info/faculty.do` (교수진소개)
- 📑 PDF (1개):
  - `2025 나에게 힘이 되는 복지서비스.pdf` (312KB)

**처리 과정**:

```python
# 웹페이지 처리
def load_web_document(url: str) -> Document:
    1. requests로 HTML 다운로드
    2. BeautifulSoup으로 파싱
    3. script/style/nav/footer 제거
    4. 텍스트 추출 및 정규화
    → Document 객체 생성 (source, title, text, metadata)

# PDF 처리
def load_pdf_document(pdf_path: str) -> Document:
    1. PyPDF로 PDF 읽기
    2. 각 페이지별 텍스트 추출
    3. "[page N]" 마크와 함께 결합
    4. 정규화
    → Document 객체 생성
```

**출력**: `data/processed/documents.jsonl`
- 4개 문서 (웹 3 + PDF 1)
- 각 문서는 JSON 라인 형식 저장

---

### 2단계: 청킹 (2가지 전략 병렬 처리)

#### **전략 A: 기본 청킹 (BaselineChunker)**

```python
def chunk(doc: Document) -> list[Chunk]:
    1. 문서 텍스트를 단락으로 분할 (\n\n 기준)
    2. 단락들을 패킹해서 지정 크기로 묶음
       - chunk_chars = 900 (기본값)
       - overlap = 150 (앞뒤 겹침)
    3. 각 청크에 고유 ID 부여
    4. 메타데이터 첨부 (chunker: "baseline", chunk_index)
```

**특성**:
- ✓ 빠름 (LLM 미사용)
- ✓ 예측 가능
- ✗ 문맥 경계에서 짤림 가능성
- **결과**: 522개 청크 (4개 문서)

#### **전략 B: LLM 청킹 (LLMChunker)** ⭐

```python
def chunk(doc: Document) -> list[Chunk]:
    1. 청킹 LLM (Qwen 0.5B)에 문서 전달
    2. LLM이 다음 프롬프트로 의미 단위 청킹 수행:
       
       "너는 문서를 RAG용으로 '의미 단위'로 청킹하는 도구다.
        규칙:
        - 각 청크는 450~1200자 사이, 문맥 끊기지 말 것
        - 제목/소제목/목록 구조 보존
        - 출력은 JSON 배열만 (설명 금지)
        - 형식: [{"title": ..., "text": ...}, ...]
        
        [문서 제목]
        ...
        
        [문서 본문]
        ..."
    
    3. LLM 응답에서 JSON 배열 추출
    4. 실패 시 BaselineChunker로 폴백
    5. 각 청크에 ID 부여 (chunker: "llm")
```

**특성**:
- ✓ 의미 보존 (문맥 단절 회피)
- ✓ 리칭 확률 높음
- ✗ 느림 (LLM 호출)
- **결과**: 381개 청크 (4개 문서, 522개보다 26% 적음)

**청킹 비교**:

| 구분 | 기본 청킹 | LLM 청킹 |
|-----|---------|--------|
| 청크 개수 | 522개 | 381개 |
| 평균 크기 | 약 900자 | 약 1,200자 |
| 처리 시간 | ~1초 | ~102초 |
| 문맥 보존 | 낮음 | 높음 ⭐ |
| 비용 | 매우 낮음 | 중간 |

---

### 3단계: 임베딩 (`build_index.py`)

**목표**: 각 청크를 고차원 벡터로 변환

**모델**: `intfloat/multilingual-e5-small`
- 384차원 벡터
- 다국어 지원 (한국어 포함)
- 경량 모델 (~471MB)

**처리 방식** (E5 권장사항):
```python
def embed_passages(texts: list[str]) -> np.ndarray:
    # 문서는 "passage: " 프리픽스 추가
    texts = [f"passage: {t}" for t in texts]
    # L2 정규화
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings  # [N, 384] 형태

def embed_query(query: str) -> np.ndarray:
    # 쿼리는 "query: " 프리픽스 추가
    query_prefixed = f"query: {query}"
    embedding = model.encode([query_prefixed], normalize_embeddings=True)
    return embedding[0]  # [384] 형태
```

**출력**: 두 가지 인덱스
- `index/baseline/`: 기본 청킹 임베딩 (522개)
- `index/llm/`: LLM 청킹 임베딩 (381개)

---

### 4단계: 벡터 인덱싱 (FAISS)

**목표**: 빠른 벡터 검색을 위한 인덱싱

**알고리즘**: `IndexFlatIP` (Inner Product)
- 코사인 유사도 등과 동등 (정규화된 임베딩 기준)
- 정확 검색 (근사 검색 아님)
- 메모리 효율적

**저장 구조**:
```
index/
├── baseline/
│   ├── index.faiss          # FAISS 인덱스 바이너리
│   ├── chunks.jsonl         # 청크 메타데이터
│   ├── meta.json            # {"dim": 384}
│   └── stats.json           # 통계
├── llm/
│   ├── index.faiss
│   ├── chunks.jsonl
│   ├── meta.json
│   └── stats.json
```

---

## 온라인 단계 (질의)

### 전체 흐름 (Gradio 웹 UI)

```
사용자 입력 (질문)
    ↓
answer_both() 함수 호출
    ├─ 기본 청킹 RAG.answer()  ──→ RAGSystem ← LLM 청킹 RAG.answer()
    │   [병렬 처리]
    ├─────→ 질문 임베딩
    ├─────→ 벡터 검색 (top_k=6)
    ├─────→ 프롬프트 구성
    ├─────→ LLM 생성
    ├─────→ 시간 측정
    └─ 답변 반환
    ↓
웹 UI에 두 답변 동시 표시
```

---

### Step 1-1: 질문 임베딩화 (기본 청킹)

```python
def answer(question: str) -> RAGAnswer:
    # Step 1: 질문을 E5 모델로 임베딩
    q_emb = self.embedder.embed_query(question)
    # 입력: "질문이 어떻게 되나?"
    # 처리: "query: 질문이 어떻게 되나?" → E5 임베딩 → [384]
    # 출력: numpy 배열 (384차원)
```

**소요 시간**: ~100-200ms (E5 모델 로드 후)

---

### Step 1-2: 벡터 검색 (FAISS)

```python
def search(query_emb: np.ndarray, top_k: int) -> list[RetrievalResult]:
    # Step 2: FAISS 인덱스에서 유사 청크 검색
    scores, indices = self.index.search(query_emb[None, :], top_k=6)
    # 입력: 쿼리 임베딩 [1, 384]
    # 처리: 코사인 유사도 계산, 상위 6개 반환
    # 출력: RetrievalResult[] (chunk + score)
```

**설정**:
- `top_k = 6` (기본값, env에서 조정 가능)
- 유사도 점수 범위: 0~1 (정규화된 임베딩)

**소요 시간**: ~5-10ms

---

### Step 1-3: 근거 추출

```python
contexts = [(r.chunk.source, r.chunk.title, r.chunk.text) for r in results]
# 예시:
# [
#   ("https://ai.jj.ac.kr/ai/info/faculty.do", "학과안내-교수진소개", "교수님 정보..."),
#   ("2025.pdf", "복지서비스", "복지 관련 내용..."),
#   ...
# ]
```

---

### Step 2: 프롬프트 구성 (Prompt Templating)

```python
def build_rag_prompt(question: str, contexts: list[tuple[str, str, str]]) -> str:
    # 템플릿:
    return f"""너는 검색 기반 질의응답(RAG) 도우미다.
규칙:
- 아래 '근거'에 있는 내용만 사용해 답해라. 근거에 없으면 '근거에서 확인할 수 없습니다'라고 말해라.
- 답변은 한국어로, 핵심 위주로 간결하게 작성해라.
- 마지막 줄에 사용한 근거 번호를 (근거: 1,2) 형태로 적어라.

[질문]
{question}

[근거]
[근거 1]
- 출처: {contexts[0][0]}
- 제목: {contexts[0][1]}
- 내용:
{contexts[0][2]}

[근거 2]
...

[근거 6]
...

[답변]
"""
```

**프롬프트 구조**:
1. 시스템 지시문 (역할 + 규칙)
2. 사용자 질문
3. 검색된 근거 (최대 6개)
4. 답변 시작 마크

**소요 시간**: ~1-2ms

---

### Step 3: LLM 생성 (답변 생성)

```python
def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    # Step 3: Qwen 1.5B 모델로 답변 생성
    
    # 3-1. 토크나이징
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    
    # 3-2. 토큰 생성
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,           # 최대 생성 토큰
            do_sample=True,               # 확률 샘플링
            temperature=0.2,              # 낮음 = 일관성 높음
            pad_token_id=...,
            eos_token_id=...
        )
    
    # 3-3. 디코딩
    gen_ids = output_ids[0][input_ids.shape[-1]:]  # 프롬프트 부분 제외
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()
```

**파라미터**:
- `max_new_tokens=512`: 최대 생성 길이
- `temperature=0.2`: 낮을수록 결정적 (다양성 감소)
- `do_sample=True`: 확률 기반 샘플링

**소요 시간**: ~800-2000ms (Qwen 1.5B 모델)

---

### Step 4: 시간 측정 및 출력

```python
def answer_both(question: str) -> tuple[str, str]:
    # 기본 청킹
    start_baseline = time.time()
    a1 = rag_baseline.answer(question).answer
    elapsed_baseline = (time.time() - start_baseline) * 1000  # ms
    
    # LLM 청킹
    start_llm = time.time()
    a2 = rag_llm.answer(question).answer
    elapsed_llm = (time.time() - start_llm) * 1000  # ms
    
    # 결과 포맷팅
    a1_with_time = f"⏱️ 소요 시간: {elapsed_baseline:.0f}ms\n\n{a1}"
    a2_with_time = f"⏱️ 소요 시간: {elapsed_llm:.0f}ms\n\n{a2}"
    
    return a1_with_time, a2_with_time
```

**최종 출력 예시**:

```
[기본 청킹]
⏱️ 소요 시간: 1250ms

전주대학교 인공지능학과의 교육 목표는 다음과 같습니다...
(근거: 1,2)

[LLM 청킹]
⏱️ 소요 시간: 1180ms

전주대학교 인공지능학과의 교육 목표는 다음과 같습니다...
(근거: 1,3)
```

---

## 사용된 모델

### 1. 임베딩 모델: `intfloat/multilingual-e5-small`

**목적**: 텍스트를 고정 크기 벡터로 변환

**사양**:
- 모델 크기: 471MB
- 출력 차원: 384
- 언어: 다국어 (한국어 포함)
- 라이브러리: Sentence-Transformers

**특징**:
- E5 (Entailment, Embedding, Expansion) 아키텍처
- 검색 + 는 "passage"/"query" 프리픽스 활용
- 코사인 유사도 최적화

---

### 2. 청킹 LLM: `Qwen/Qwen2.5-0.5B-Instruct`

**목적**: 문서를 의미 단위로 동적 청킹

**사양**:
- 모델 크기: 988MB (양자화 전)
- 파라미터: 0.5B
- 컨텍스트: 131K
- 라이브러리: Transformers

**역할**:
- JSON 형식으로 청크 출력
- 문맥 보존 청킹
- 제목/소제목 구조 인식

---

### 3. 답변 LLM: `Qwen/Qwen2.5-1.5B-Instruct`

**목적**: RAG 근거를 기반으로 답변 생성

**사양**:
- 모델 크기: 3.09GB (원본)
- 파라미터: 1.5B
- 컨텍스트: 131K
- 라이브러리: Transformers

**특징**:
- Instruction Fine-Tuning
- 한국어 포함 다국어
- Chat 템플릿 지원

---

## 기술 스택

### 데이터 수집/처리
| 계층 | 기술 | 용도 |
|------|------|------|
| 웹 크롤링 | `requests`, `BeautifulSoup4` | HTML 다운로드 & 파싱 |
| PDF 처리 | `PyPDF` | PDF 텍스트 추출 |
| 텍스트 정규화 | 정규표현식 | 공백/개행 정리 |

### 모델 & 임베딩
| 계층 | 기술 | 용도 |
|-----|------|------|
| 임베딩 | `Sentence-Transformers` | 텍스트 벡터화 |
| LLM | `Transformers`, `Torch` | 모델 로드 & 추론 |
| 가속화 | `Accelerate` | GPU/multi-GPU 지원 |

### 벡터 검색
| 계층 | 기술 | 용도 |
|------|------|------|
| 인덱싱 | `FAISS` | 벡터 검색 (Inner Product) |
| 계산 | `NumPy` | 벡터 연산 |

### 데이터 저장
| 계층 | 기술 | 용도 |
|------|------|------|
| 고급 문서 | JSONL | 청크/문서 저장 |
| 벡터 인덱스 | FAISS Binary | 빠른 로드 |
| 설정 | JSON, `.env` | 환경 변수 관리 |

### 웹 UI
| 계층 | 기술 | 용도 |
|------|------|------|
| 프론트엔드 | `Gradio` | 사용자 인터페이스 |
| 백엔드 | Python | 로직 처리 |

---

## 청킹 전략 비교

### 기본 청킹의 문제점

```
원문: "AI 학과는 머신러닝을 중심으로 한다. 
     Deep Learning 연구가 활발하다. 
     최신 기술을 배운다."

고정 크기 청킹 (900자):
[청크 1] "AI 학과는 머신러닝을 중심으로 한다. Deep Learning 연구가"
[청크 2] "활발하다. 최신 기술을 배운다."  ← 문맥 끊김!
```

문제:
- ✗ 문장 중간에 끊김
- ✗ 주제 단위 미보존
- ✗ 리랭킹 시 컨텍스트 손실

---

### LLM 청킹의 장점

```
원문: (동일)

LLM 청킹 결과:
[청크 1] "AI 학과는 머신러닝을 중심으로 한다. Deep Learning 연구가 활발하다."
[청크 2] "최신 기술을 배운다."  ✓ 문맥 보존!
```

장점:
- ✓ 의미 경계 인식
- ✓ 제목 구조 보존
- ✓ 길이 동적 조정 (450-1200자)
- ✓ 검색 확률 증가

---

### 성능 실제 측정값

**4개 문서 기준**:

| 메트릭 | 기본 청킹 | LLM 청킹 |
|-------|---------|--------|
| 총 청크 수 | 522 | 381 |
| 청크 압축률 | base | -27% |
| 청킹 시간 | ~1초 | ~102초 |
| 평균 청크 크기 | 600자 | 820자 |
| 메모리 (임베딩) | ~51MB | ~37MB |
| 검색 속도 | ~8ms | ~8ms |

**검색 속도는 동일** (청킹 후 임베딩이므로 FAISS는 동등)

---

## 디버깅 & 모니터링

### 로그 확인

```bash
# 데이터 수집 통계
cat data/processed/summary.json

# 청킹 통계
cat index/baseline/stats.json
cat index/llm/stats.json

# 샘플 청크 보기
head -1 index/baseline/chunks.jsonl | python -m json.tool
```

### 환경 변수

```bash
# .env 파일 설정
JJ_RAG_DEVICE=auto              # auto|cpu|cuda
JJ_RAG_EMBED_MODEL=intfloat/... # 임베딩 모델
JJ_RAG_ANSWER_MODEL=Qwen/...    # 답변 LLM
JJ_RAG_CHUNK_MODEL=Qwen/...     # 청킹 LLM
JJ_RAG_TOP_K=6                  # 검색 개수
JJ_RAG_BASE_CHUNK_CHARS=900     # 기본 청크 크기
JJ_RAG_LLM_CHUNK_MIN=450        # LLM 청크 최소
JJ_RAG_LLM_CHUNK_MAX=1200       # LLM 청크 최대
```

---

## 성능 최적화 팁

### 1. 검색 개수 조정
```bash
JJ_RAG_TOP_K=3   # 빠르지만 근거 부족 가능
JJ_RAG_TOP_K=10  # 느리지만 정확도 높음
```

### 2. LLM 생성 파라미터
```python
# temperature 조정
0.0  → 결정적 (항상 같은 답변)
0.2  → 기본값 (일관성 높음)
0.7  → 발산적 (다양한 답변)

# max_new_tokens 조정
256  → 짧은 답변
512  → 기본값 (중간 길이)
1024 → 상세 답변
```

### 3. 배치 처리
```python
# 다량의 질문 처리 시
embeddings = embedder.embed_passages(texts, batch_size=64)  # 배치 증가
```

---

## 결론

### 핵심 특징

✅ **2단계 청킹 비교**: 기본(빠름) vs LLM(정확함)  
✅ **독립적 LLM**: 청킹 LLM ≠ 생성 LLM (간섭 회피)  
✅ **실시간 성능 측정**: 각 청킹의 소요 시간 표시  
✅ **효율적 스택**: FAISS 벡터 검색, Sentence-Transformers 임베딩  
✅ **한국어 최적화**: 다국어 모델, RAG 템플릿 한국어

### 다음 단계 (선택사항)

1. **리랭킹 추가**: LLM 기반 재순위화
2. **하이브리드 검색**: 벡터 + 키워드 조합
3. **메타데이터 필터링**: 소스별 필터링
4. **Fine-tuning**: 도메인 특화 모델
5. **평가 메트릭**: NDCG, BLEU, ROUGE

