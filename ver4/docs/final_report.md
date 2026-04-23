# Final Report

## Top Configurations
- Retrieval best: {'run_id': 'run_04c4ccd9f72e', 'retrieval': 0.15384615384615385, 'answer': 0.0, 'judge': 1.0, 'abstention': 0.9230769230769231, 'latency': 5.1939501785315e-05, 'overall': 0.35193905075436355, 'efficiency': 6775.941983599624}
- Answer best: {'run_id': 'run_04c4ccd9f72e', 'retrieval': 0.15384615384615385, 'answer': 0.0, 'judge': 1.0, 'abstention': 0.9230769230769231, 'latency': 5.1939501785315e-05, 'overall': 0.35193905075436355, 'efficiency': 6775.941983599624}
- Judge best: {'run_id': 'run_04c4ccd9f72e', 'retrieval': 0.15384615384615385, 'answer': 0.0, 'judge': 1.0, 'abstention': 0.9230769230769231, 'latency': 5.1939501785315e-05, 'overall': 0.35193905075436355, 'efficiency': 6775.941983599624}
- Abstention best: {'run_id': 'run_04c4ccd9f72e', 'retrieval': 0.15384615384615385, 'answer': 0.0, 'judge': 1.0, 'abstention': 0.9230769230769231, 'latency': 5.1939501785315e-05, 'overall': 0.35193905075436355, 'efficiency': 6775.941983599624}
- Latency-efficient best: {'run_id': 'run_a2b2b385640f', 'retrieval': 0.11538461538461539, 'answer': 0.0, 'judge': 1.0, 'abstention': 0.9230769230769231, 'latency': 4.790522731267489e-05, 'overall': 0.34838337480712966, 'efficiency': 7272.345719878329}

## Ablation Summary
- loader_name: [{'loader_name': 'pdftotext_layout', 'overall_score': 0.34908621264912304}]
- cleaning_name: [{'cleaning_name': 'regex_rules', 'overall_score': 0.34908621264912304}]
- chunking_name: [{'chunking_name': 'field_aware', 'overall_score': 0.34908621264912304}]
- representation_name: [{'representation_name': 'hybrid_dual', 'overall_score': 0.34908621264912304}]
- retrieval_name: [{'retrieval_name': 'hybrid', 'overall_score': 0.35193905075436355}, {'retrieval_name': 'sparse', 'overall_score': 0.34838337480712966}, {'retrieval_name': 'dense', 'overall_score': 0.34693621238587596}]
- post_retrieval_name: [{'post_retrieval_name': 'none', 'overall_score': 0.34908621264912304}]
- generation_name: [{'generation_name': 'strict_grounded', 'overall_score': 0.34908621264912304}]

## Best/Worst Case Analysis
- BEST m001: best=run_3ed020bfbd31 worst=run_04c4ccd9f72e
- BEST m002: best=run_04c4ccd9f72e worst=run_a2b2b385640f
- BEST m003: best=run_a2b2b385640f worst=run_3ed020bfbd31
- BEST m004: best=run_04c4ccd9f72e worst=run_3ed020bfbd31
- BEST m005: best=run_04c4ccd9f72e worst=run_a2b2b385640f
- WORST m001: best=run_3ed020bfbd31 worst=run_04c4ccd9f72e
- WORST m002: best=run_04c4ccd9f72e worst=run_a2b2b385640f
- WORST m003: best=run_a2b2b385640f worst=run_3ed020bfbd31
- WORST m004: best=run_04c4ccd9f72e worst=run_3ed020bfbd31
- WORST m005: best=run_04c4ccd9f72e worst=run_a2b2b385640f

## Pairwise Analysis
- overall_best_vs_worst q=m001 preferred=run_04c4ccd9f72e reason=
- overall_best_vs_worst q=m002 preferred=run_04c4ccd9f72e reason=
- overall_best_vs_worst q=m003 preferred=run_04c4ccd9f72e reason=
- overall_best_vs_worst q=m004 preferred=run_04c4ccd9f72e reason=
- overall_best_vs_worst q=m005 preferred=run_04c4ccd9f72e reason=
