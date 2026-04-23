# Reproduction

1) 동일 PDF를 사용한다.
2) 동일 conda env(`ys_conda1_env`)를 활성화한다.
3) configs/experiment_matrix.json에서 model/backend/path를 확인한다.
4) run_builds.sh -> run_experiments.sh -> run_eval.sh -> run_judge.sh -> run_pairwise.sh -> export_best_worst.sh 순서로 실행한다.
5) 중단 시 resume_failed.sh로 실패/부분완료 run만 재개한다.
6) results/experiment_results.csv와 artifacts/runs/run_<hash>/를 기준으로 모든 질문별 결과를 복원한다.
