# Documentation Map

이 디렉터리는 실행 절차, 실험 결정, 논문 reference를 함께 보관한다. 현재 기준 문서는 아래 순서로 본다.

## Current Runbooks

- [Docker Guide](docker_guide.md): 컨테이너 빌드, 실행, GPU, VNC/X11, troubleshooting.
- [Adding Checkpoint](adding_checkpoint.md): 새 VLA 체크포인트를 profile/serve/eval 경로에 붙이는 체크리스트.
- [GR00T RoboCasa Docs](groot/README.md): N1.6/N1.5 학습·평가, SAFE feature export, detector 재현 문서의 통합 맵 (번호 prefix로 reading order 표현).

## GR00T N1.6 Reading Order

1. [Fine-Tuning](groot/n16_01_finetune.md)
2. [Evaluation](groot/n16_02_eval.md)
3. [SAFE Overview](groot/n16_03_safe_overview.md)
4. [SAFE Collection](groot/n16_04_safe_collection.md) — ah8/ah16 mode 포함
5. [Scenario Reproduction](groot/n16_05_safe_env_reproduction.md) — scenario seed + ep_meta
6. [Inference Datapoint Semantics](groot/n16_06_safe_inference_semantics.md) — 한 datapoint 의미
7. [SAFE Detector](groot/n16_07_safe_detector.md) — split + LSTM + CP 운영점
8. [SAFE Visualization](groot/n16_08_safe_visualization.md) — t-SNE/silhouette
9. [SAFE Parity](groot/n16_09_safe_parity.md) — ZMQ official baseline parity + HTTP `/act` parity
10. [SAFE Reproduction Report](groot/n16_10_safe_report.md) — 축소 재현 결과 보고서

관련 ADR: [0001 Dedicated SAFE GR00T N1.6 ZMQ server](adr/0001-dedicated-safe-groot-n16-zmq-server.md).

최종 detector artifact:

```text
outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector
```

## GR00T N1.5 Reading Order

1. [N1.5 Fine-Tuning](groot/n15_01_finetune.md)
2. [N1.5 Evaluation](groot/n15_02_eval.md)

## RoboCasa Reference

- [RoboCasa Task Name Mapping](robocasa_task_name_mapping.md): RoboCasa v0.2 (`robocasa_v02`) task와 robocasa365 v1.0 task 이름 대응.

## TTT Methodology

- [TTT Pipeline](ttt/ttt_pipeline.md): GR00T N1.6 + TTT RoboCasa pipeline 정리.

## Phase / Loop Analysis

- [Phase 1 Progress Predictor](ttt/progress_predictor.md): VITA 기반 progress predictor 구현 및 학습 정리.
- [Phase 1 Status](ttt/2026_04_01_status.md): Phase 1 진행 상태 snapshot.
- [GR00T Loop Analysis Plan](ttt/legacy_groot_loop_analysis_plan.md): 초기 loop analysis 계획. 문서 상단의 legacy note를 먼저 확인한다.

## Legacy / Older Setup Notes

- [GR00T Legacy Fine-Tuning Setup](groot/_legacy/robocasa_finetune_setup.md): 초기 GR00T x RoboCasa fine-tuning setup. 현행 N1.6 기준은 `groot/n16_01_finetune.md`다.

## Paper References

논문 PDF는 운영 runbook과 분리해서 [references/](references/) 아래에 둔다.

- `references/CoT-VLA.pdf`
- `references/Scaling World Model.pdf`
- `references/VITA.pdf`
- `references/robocasa365.pdf`

## Organization Rule

- GR00T 실행 절차는 `groot/` 아래에 둔다.
- GR00T N1.6 SAFE feature export, detector, visualization 문서는 `groot/n16_03_safe_overview.md`부터 `groot/n16_10_safe_report.md`까지 번호 prefix 문서로 둔다.
- TTT 연구 방법론, phase 진행 기록, loop analysis는 `ttt/` 아래에 둔다.
- 장기 결정은 `adr/`에 둔다.
- 논문 PDF와 외부 reference는 `references/` 아래에 둔다.
