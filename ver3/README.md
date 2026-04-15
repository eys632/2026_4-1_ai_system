# ver3: PDF 기반 RAG 데모(복지서비스)

이 폴더는 **단일 PDF(책자)에서 텍스트를 추출 → 정제/청킹 → 인덱싱(검색) → RAG 답변 생성(vLLM/Qwen) → Gradio 시현 UI**까지 한 번에 보여주는 버전입니다.

> 주의: PDF 원문은 레포에 포함하지 않았습니다(저작권/용량). 로컬에 PDF 파일을 준비해서 경로를 인자로 넘겨 실행하세요.

## 구성
- `src/raglab/rag_lab.py` : PDF 로딩/정제/청킹/임베딩/인덱싱 + 검색 CLI
- `src/raglab/web_demo.py` : Gradio 탭 UI(질문→답변, 검색디버그, PDF 특성, 평가, 발표 체크)
- `src/raglab/eval_retrieval.py` : retrieval hit@k, MRR 간단 평가 CLI
- `eval/questions.jsonl` : 예시 평가 질문(작게 시작)
- (런타임 생성) `artifacts/` : 인덱스/청크/메타데이터 산출물

## 사전 요구사항
- Python 패키지: `pypdf`, `sentence-transformers`, `faiss`, `scikit-learn`, `joblib`, `gradio`, (선택) `vllm`
- 시스템 바이너리(권장): `pdftotext` (Ubuntu 기준 `poppler-utils`에 포함)

## 실행 환경(권장)
- 이 프로젝트는 **conda 환경(예: `ys_conda1_env`)**에서 실행하는 걸 기준으로 작성했습니다.
- 서버에 `python` 명령이 없다면(= conda 미활성화 상태) 아래처럼 먼저 활성화하세요.

```bash
source /home/a202192020/miniconda3/bin/activate ys_conda1_env
```

## 빠른 시작
아래 명령은 **레포 루트에서** 실행한다고 가정합니다.

### 0) artifacts가 이미 있으면 빌드 생략 가능
`ver3/artifacts/`에 아래 파일이 있으면, 곧바로 웹 실행으로 넘어가도 됩니다.
- `faiss.index`, `tfidf.joblib`, `chunks.jsonl`, `build_meta.json`

### 1) 인덱스 빌드(필수: 최초 1회)
```bash
cd ver3

# (권장) pdftotext(-layout) 기반 로더 + 필드 기반 청킹(대상/내용/방법/문의)
python -m src.raglab.rag_lab build \
  --pdf "/path/to/2025 나에게  힘이 되는 복지서비스.pdf" \
  --loader pdftotext \
  --chunking fields \
  --device cuda
```

빌드가 끝나면 `ver3/artifacts/`에 아래 파일들이 생성됩니다.
- `pages.jsonl`, `chunks.jsonl`
- `embeddings.npy`, `faiss.index`, `tfidf.joblib`
- `build_meta.json`

### 2) (선택) PDF 프로브 파일 생성(웹 탭 표시용)
`web_demo.py`의 “PDF/데이터 특성” 탭은 `artifacts/pdf_probe.json`이 있으면 표시합니다.
```bash
cd ver3
mkdir -p artifacts
python -m src.raglab.rag_lab probe --pdf "/path/to/2025 나에게  힘이 되는 복지서비스.pdf" > artifacts/pdf_probe.json
```

### 3) 검색 CLI로 빠르게 확인(LLM 없이)
```bash
cd ver3
python -m src.raglab.rag_lab query -q "실업급여 신청 방법" --method hybrid -k 5
```

### 4) 웹 시현(Gradio)
```bash
cd ver3

# GPU 선택(예: 3번 GPU만 사용)
export CUDA_VISIBLE_DEVICES=3

# 권장 옵션:
# - --enforce-eager: vLLM의 초기 컴파일/그래프(warmup) 때문에 첫 응답이 매우 느려지는 현상 완화
# - --max-model-len 2048: 초기화 부담을 낮춰 데모 반응성을 우선
python -m src.raglab.web_demo \
  --artifacts artifacts \
  --embed-device cuda \
  --host 0.0.0.0 --port 7860 \
  --gpu-mem-util 0.6 \
  --enforce-eager \
  --max-model-len 2048
```

브라우저에서 `http://<서버IP>:7860` 접속.

서버 IP 확인(둘 중 아무거나):
```bash
hostname -I
ip addr | grep -E "inet "
```

## 평가(리트리벌)
인덱스를 빌드한 뒤:
```bash
cd ver3
python -m src.raglab.eval_retrieval --method hybrid --k 5
```

## 실험(설정 비교: loader~post-retrieval)
아래 스크립트는 **로더/청킹/리트리벌 방식/간단 post-retrieval(서비스/필드 라우팅)** 조합을 한 번에 비교하고,
`ver3/experiments/` 아래에 결과(`results_*.json`, `results_*.csv`, `top10_*.txt`)를 저장합니다.

```bash
cd ver3

# 최소: 수동 질문셋(eval/questions.jsonl)로만 비교
python -m src.raglab.experiment_ablation \
  --pdf "/path/to/2025 나에게  힘이 되는 복지서비스.pdf" \
  --k 5 \
  --embed-device cpu

# 권장: 템플릿 기반 질문을 추가로 생성해서(auto) 비교(예: 120문항)
python -m src.raglab.experiment_ablation \
  --pdf "/path/to/2025 나에게  힘이 되는 복지서비스.pdf" \
  --k 5 \
  --auto-questions 120 \
  --seed 7 \
  --embed-device cpu
```

## 실행 팁
- vLLM 메모리 이슈가 나면 `--gpu-mem-util`을 낮추세요(예: 0.60~0.75).
- 첫 답변이 오래 걸리면(수 분 이상) `--enforce-eager` + `--max-model-len 2048` 조합을 우선 권장합니다.
- `--trust-remote-code`는 기본적으로 꺼져 있습니다. 필요할 때만 켜세요.
