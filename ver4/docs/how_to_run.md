# How To Run

## 0) 환경 준비
```bash
source /home/a202192020/miniconda3/bin/activate ys_conda1_env
cd /data2/a202192020/4-1/ai_sys/2026_4-1_ai_system/ver4
pip install -r requirements.txt
```

주의:
- 새로운 가상환경을 만들지 않습니다.
- 모든 스크립트는 내부적으로 use_ys_conda1_env.sh를 통해 ys_conda1_env를 강제 활성화합니다.

## 1) 빌드 캐시
```bash
bash run_builds.sh
```

## 2) Full Matrix 실행
```bash
bash run_experiments.sh
```

특정 dataset만 실행:
```bash
bash run_experiments.sh \
  "/data2/a202192020/4-1/ai_sys/2026_4-1_ai_system/2025 나에게  힘이 되는 복지서비스.pdf" \
  manual
```

## 3) 자동평가 집계
```bash
bash run_eval.sh
```

## 4) LLM Judge
```bash
bash run_judge.sh
```

## 5) Pairwise Judge
```bash
bash run_pairwise.sh
```

## 6) Best/Worst export + 리포트
```bash
bash export_best_worst.sh
```

## 7) 실패 run 복구
```bash
bash resume_failed.sh
```
