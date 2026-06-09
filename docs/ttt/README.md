# TTT / Progress Predictor Archive

이 디렉터리는 예전 연구 방향인 **TTT/VITA progress predictor 기반 실패 루프 탈출** 문서를
보존한다. 현재 메인 연구 라인은 latent steering이며, 새 실험 계획과 결과는
[`../steering/`](../steering/README.md)를 기준으로 한다.

TTT 문서는 삭제하지 않는다. progress predictor 구현, Eagle pre-LLM cache, GR00T+TTT 연결
시도에서 얻은 재현 절차와 실패 지점은 후속 baseline 또는 ablation으로 다시 필요할 수 있다.

## 현재 상태

- 상태: 보존 / 비메인 track.
- 목적: VLA backbone을 직접 바꾸지 않고 external progress predictor 또는 TTT token으로 실패
  루프를 탈출하는 초기 가설 기록.
- 현재 신규 작업 기준: latent steering 문서를 먼저 보고, TTT는 과거 설계나 비교 baseline이
  필요할 때만 참조한다.

## Reading Order

1. [TTT Pipeline](ttt_pipeline.md) — GR00T N1.6 + TTT × RoboCasa atomic pretrain end-to-end
   pipeline, Phase 0/1/2 실행 절차와 함정 기록.
2. [Progress Predictor](progress_predictor.md) — VITA 기반 ProgressPredictor 구조, dataset,
   학습 설정.
3. [2026-04-01 Status](2026_04_01_status.md) — 당시 Phase 1 학습 결과와 재학습 준비 상태.

## Legacy

- [_legacy/groot_loop_analysis_plan.md](_legacy/groot_loop_analysis_plan.md) — 초기 GR00T
  RoboCasa loop 분석 실행 계획. 현행 N1.6 RoboCasa 평가는
  [`../groot/n16_02_eval.md`](../groot/n16_02_eval.md)를 기준으로 한다.
