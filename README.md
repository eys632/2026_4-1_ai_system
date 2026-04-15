# jj_rag (전주대 인공지능학과 + 복지서비스 PDF RAG)

이 프로젝트는 다음 2가지 데이터를 대상으로 RAG를 구성하고, **청킹 전략(기본 vs LLM-기반 청킹)**에 따른 성능 차이를 같은 질문에 대해 비교 출력합니다.

- 전주대학교 인공지능학과 웹사이트(학과안내 탭: 인사말/학과소개/교수진소개/강의실습실/행정안내 등)
- `2025 나에게  힘이 되는 복지서비스.pdf`

추가 문서:
- 파이프라인 상세: `PIPELINE.md`

## 빠른 실행(권장 흐름)

아래 명령은 이 프로젝트 경로에서 실행합니다.

```bash
cd /data2/a202192020/4-1/ai_sys/0415_ver2/jj_rag
```

또한 이 과제는 지정된 conda 환경을 사용합니다.

```bash
source /home/a202192020/miniconda3/bin/activate ys_conda1_env
```

1) 패키지 설치

```bash
pip install -r requirements.txt
pip install -e .
```

2) 데이터 수집/추출

```bash
python scripts/collect_data.py
```

참고:
- 웹 데이터는 현재 다음 3개 페이지를 수집합니다: 인사말/학과소개/교수진소개
- `강의실습실(lab.do)`과 `행정안내(admin.do)`는 현재 사이트에서 404로 확인되어 수집 스크립트가 자동으로 스킵합니다.
- 수집 대상 URL은 `scripts/collect_data.py`의 `web_urls`에서 수정할 수 있습니다.

3) 인덱스 구축(기본 청킹/LLM 청킹 각각)

```bash
python scripts/build_index.py --chunker baseline
python scripts/build_index.py --chunker llm
```

4) 웹 앱 실행(같은 질문에 대해 답변 2종 출력)

```bash
python app.py
```

브라우저에서 접속:
- http://localhost:7860

웹 UI 동작:
- 질문 1개 입력
- 동일 질문에 대해 (기본 청킹 인덱스) vs (LLM 청킹 인덱스) 답변을 동시에 출력
- 각 답변 상단에 소요 시간(ms)을 표시

---

## 종료/재실행 방법

### 종료(웹 앱)
- 터미널에서 웹 앱이 실행 중인 상태에서 `Ctrl + C`를 눌러 종료합니다.

### 종료 후 재실행
1) (필요 시) conda 환경 활성화
```bash
source /home/a202192020/miniconda3/bin/activate ys_conda1_env
```
2) 웹 앱 재실행
```bash
cd /data2/a202192020/4-1/ai_sys/0415_ver2/jj_rag
python app.py
```

### 포트가 이미 사용 중일 때(선택)
간혹 이전 프로세스가 남아 7860 포트를 점유할 수 있습니다.

```bash
fuser -k 7860/tcp || true
```

---

## 저장 데이터/산출물 위치

이 프로젝트는 실행 결과를 크게 `data/`와 `index/` 아래에 저장합니다.

### 1) 원천(raw) 데이터
- `data/raw/web/`
	- 웹에서 가져온 원본 HTML이 페이지별로 저장됩니다.

### 2) 전처리(processed) 데이터
- `data/processed/documents.jsonl`
	- 수집된 문서(웹 3 + PDF 1)의 텍스트가 JSONL로 저장됩니다.
- `data/processed/summary.json`
	- 수집 개수/경로 요약이 저장됩니다.

### 3) 벡터 인덱스(청킹 방식별)
- `index/baseline/`
	- `index.faiss`: FAISS 인덱스 파일
	- `chunks.jsonl`: 청크 본문+메타데이터
	- `meta.json`: 임베딩 차원 정보
	- `stats.json`: 인덱스 통계

- `index/llm/`
	- 위와 동일한 구조로 LLM 청킹 결과 인덱스를 저장합니다.

### 4) 입력 PDF 위치
- PDF는 프로젝트 상위 경로에 있어도 자동으로 포함됩니다:
	- `/data2/a202192020/4-1/ai_sys/0415_ver2/2025 나에게  힘이 되는 복지서비스.pdf`

---

## 구현 내용 전반 설명(파이프라인)

이 시스템은 다음 단계로 구성됩니다.

### A. Loader(수집/로더)
- 웹 로더: `requests`로 HTML 수집 후 `BeautifulSoup(lxml)`로 주요 텍스트만 추출
- PDF 로더: `pypdf`로 페이지별 텍스트를 추출해 하나의 문서로 결합

### B. Chunking(청킹)
동일한 문서셋에 대해 청킹 전략 2가지를 각각 적용하고 인덱스를 별도로 구축합니다.

1) 기본 청킹(baseline)
- 문단 단위로 먼저 나눈 뒤, 고정 길이(기본 900자)로 패킹
- 청크 간 오버랩(기본 150자)을 적용

2) LLM 청킹(llm)
- 별도의 청킹 LLM이 원문을 읽고 의미 단위로 JSON 배열 형태의 청크를 출력
- 문맥 단절을 줄이는 것이 목표
- 실패 시 안전장치로 기본 청킹 방식으로 폴백

### C. Embedding(임베딩)
- 임베딩 모델이 각 청크를 벡터로 변환
- E5 방식에 맞춰 문서는 `passage:` / 질문은 `query:` 접두어를 붙임

### D. Vectorstore/Retrieval(검색)
- FAISS `IndexFlatIP`(inner product)로 top-k(기본 6개) 청크를 검색
- 에너지 질문처럼 특정 도메인 키워드가 분명한 경우, post-retrieval에서 관련 키워드 중심으로 근거를 정리/다양화하는 휴리스틱을 적용

### E. Prompting(프롬프트 구성)
- 질문 + 근거(top-k 청크)를 하나의 RAG 프롬프트로 구성
- 규칙: 근거 밖의 내용은 추측하지 말고, 중복을 피하며, 마지막에 (근거: 1,2,...) 표기

### F. Generation(답변 생성)
- 답변 LLM이 위 프롬프트를 입력으로 받아 최종 답변 생성
- 웹 UI에서는 기본 청킹 인덱스/LLM 청킹 인덱스에 대해 동일 질문을 각각 실행해 2개의 답변을 나란히 보여줌
- **두 답변 모두 같은 답변 LLM(Qwen)을 사용**하며, 차이는 (검색에 사용된 인덱스/청크 단위)에서 발생합니다.

---

## 모델 기본값

## 모델 기본값

- 임베딩: `intfloat/multilingual-e5-small`
- 답변 LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- 청킹 LLM: `Qwen/Qwen2.5-0.5B-Instruct`

모델/장치 설정은 `.env` 또는 실행 인자로 바꿀 수 있습니다(코드 참고).

---

## LLM 실행 방식(로컬 실행)

이 프로젝트의 Qwen LLM은 **Hugging Face API(원격 호출)로 동작하지 않습니다.**
`transformers`의 `AutoModelForCausalLM.from_pretrained(...)`로 모델을 내려받아(캐시에 저장) **현재 머신에서 직접 로드해서 실행**합니다.

- GPU 사용 여부: 기본값은 `JJ_RAG_DEVICE=auto`이며, CUDA 사용 가능하면 GPU로 자동 매핑됩니다.
- 즉, 앱을 실행하면 모델이 (필요 시) 디스크 캐시에서 로드되어 GPU/CPU 메모리로 올라가고, 그 상태에서 답변을 생성합니다.

---

## LLM 파일(가중치) 저장 경로

모델 파일은 프로젝트 폴더 안에 들어있는 것이 아니라, **Hugging Face 캐시 디렉터리**에 저장됩니다.
기본적으로는 보통 아래 경로 중 하나를 사용합니다.

- 기본(많이 사용): `~/.cache/huggingface/hub`

다만 아래 환경변수에 따라 실제 저장 위치가 바뀔 수 있습니다.

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`

### 현재 환경에서 “실제 캐시 경로” 확인

아래 명령을 실행하면 이 환경에서 모델이 저장되는 Hugging Face 캐시 경로를 정확히 확인할 수 있습니다.

```bash
python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
```

또는 Transformers 캐시 경로를 보고 싶으면:

```bash
python -c "import transformers; from transformers.utils import hub; print(hub.TRANSFORMERS_CACHE)"
```

### 캐시에 내려받은 모델 확인(예시)

```bash
ls -1 ~/.cache/huggingface/hub | head
```

모델 디렉터리는 보통 `models--Qwen--Qwen2.5-1.5B-Instruct` 같은 형태로 생성됩니다.

---

## 환경 변수(.env)

`.env.example`을 참고해 `.env`를 만들면 기본 설정을 쉽게 바꿀 수 있습니다.

- `JJ_RAG_DEVICE`: `auto|cpu|cuda`
- `JJ_RAG_EMBED_MODEL`: 임베딩 모델 ID
- `JJ_RAG_ANSWER_MODEL`: 답변 모델 ID
- `JJ_RAG_CHUNK_MODEL`: 청킹 모델 ID
- `JJ_RAG_TOP_K`: 검색 top-k
- `JJ_RAG_BASE_CHUNK_CHARS`, `JJ_RAG_BASE_CHUNK_OVERLAP`: baseline 청킹 파라미터
- `JJ_RAG_LLM_CHUNK_MIN`, `JJ_RAG_LLM_CHUNK_MAX`: LLM 청킹 길이 제약
- `JJ_RAG_MAX_NEW_TOKENS`: 답변 생성 길이

---

## 처음부터 다시 만들기(선택)

데이터/인덱스를 모두 지우고 처음부터 다시 구축하려면:

```bash
rm -rf data/processed index/baseline index/llm
python scripts/collect_data.py
python scripts/build_index.py --chunker baseline
python scripts/build_index.py --chunker llm
python app.py
```
