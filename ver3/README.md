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

## 빠른 시작
아래 명령은 **레포 루트에서** 실행한다고 가정합니다.

### 1) 인덱스 빌드(필수)
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

### 3) 검색 CLI로 빠르게 확인
```bash
cd ver3
python -m src.raglab.rag_lab query -q "실업급여 신청 방법" --method hybrid -k 5
```

### 4) 웹 시현(Gradio)
```bash
cd ver3

# GPU 선택(예: 3번 GPU만 사용)
export CUDA_VISIBLE_DEVICES=3

# 임베딩은 cuda 권장, LLM은 vLLM(Qwen)로 로컬 생성
python -m src.raglab.web_demo \
  --artifacts artifacts \
  --embed-device cuda \
  --host 0.0.0.0 --port 7860 \
  --gpu-mem-util 0.75
```
브라우저에서 `http://<서버IP>:7860` 접속.

## 평가(리트리벌)
인덱스를 빌드한 뒤:
```bash
cd ver3
python -m src.raglab.eval_retrieval --method hybrid --k 5
```

## 실행 팁
- vLLM 메모리 이슈가 나면 `--gpu-mem-util`을 낮추세요(예: 0.60~0.75).
- `--trust-remote-code`는 기본적으로 꺼져 있습니다. 필요할 때만 켜세요.
