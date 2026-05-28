# Documentation Map

이 디렉터리는 실행 절차, 실험 결정, 논문 reference를 함께 보관한다. 현재 기준 문서는 아래 순서로 본다.

## Current Runbooks

- [Docker Guide](docker_guide.md): 컨테이너 빌드, 실행, GPU, VNC/X11, troubleshooting.
- [Cache Paths](cache_paths.md): 체크포인트·데이터셋 cache 위치(`~/.cache/temporal_vla`, 컨테이너 `/cache`)와 코드에서 참조하는 단일 소스 규칙.
- [Adding Checkpoint](adding_checkpoint.md): 새 VLA 체크포인트를 profile/serve/eval 경로에 붙이는 체크리스트.
- [GR00T N1.6 RoboCasa Eval](groot/n16_robocasa_eval.md): GR00T N1.6 RoboCasa ZMQ evaluation 기준 문서.
- [GR00T N1.6 RoboCasa Fine-Tuning](groot/n16_robocasa_finetune.md): GR00T N1.6 RoboCasa fine-tuning 기준 문서.
- [GR00T N1.5 RoboCasa Eval](groot/n15_robocasa_eval.md): GR00T N1.5 전용 eval 문서.
- [GR00T N1.5 RoboCasa Fine-Tuning](groot/n15_robocasa_finetune.md): GR00T N1.5 전용 fine-tuning 문서.

## SAFE Wiring

- [SAFE x GR00T N1.6 RoboCasa Report](safe/groot_n16_robocasa_safe_report.md): SAFE 논문식 failure detection을 GR00T N1.6 RoboCasa에 축소 재현한 기술 보고서.
- [SAFE x GR00T N1.6 RoboCasa Wiring](safe/groot_n16_robocasa_wiring.md): GR00T N1.6 base checkpoint와 SAFE detector를 연결하는 현재 기준 문서. 최종 SAFE-LSTM 운영점은 `split_cp`, `alpha=0.2`, `neg_success`, threshold `0.5301596522331238`; functional CP band 결과도 함께 저장한다.
- [ADR 0001](adr/0001-dedicated-safe-groot-n16-zmq-server.md): SAFE GR00T N1.6에 dedicated ZMQ feature server를 두기로 한 결정.

최종 detector artifact:

```text
outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector
```

## RoboCasa Reference

- [RoboCasa Task Name Mapping](robocasa_task_name_mapping.md): RoboCasa v0.2 (`robocasa_v02`) task와 robocasa365 v1.0 task 이름 대응.

## TTT Methodology

- [TTT Pipeline](ttt/ttt_pipeline.md): GR00T N1.6 + TTT RoboCasa pipeline 정리.

## Phase / Loop Analysis

- [Phase 1 Progress Predictor](ttt/progress_predictor.md): VITA 기반 progress predictor 구현 및 학습 정리.
- [Phase 1 Status](ttt/2026_04_01_status.md): Phase 1 진행 상태 snapshot.
- [GR00T Loop Analysis Plan](ttt/legacy_groot_loop_analysis_plan.md): 초기 loop analysis 계획. 문서 상단의 legacy note를 먼저 확인한다.

## Legacy / Older Setup Notes

- [GR00T Legacy Fine-Tuning Setup](groot/legacy_robocasa_finetune_setup.md): 초기 GR00T x RoboCasa fine-tuning setup. 현행 N1.6 기준은 `groot/n16_robocasa_finetune.md`다.

## Paper References

논문 PDF는 운영 runbook과 분리해서 [references/](references/) 아래에 둔다.

- `references/CoT-VLA.pdf`
- `references/Scaling World Model.pdf`
- `references/VITA.pdf`
- `references/robocasa365.pdf`

## Organization Rule

- GR00T 실행 절차는 `groot/` 아래에 둔다.
- TTT 연구 방법론, phase 진행 기록, loop analysis는 `ttt/` 아래에 둔다.
- 장기 결정은 `adr/`에 둔다.
- SAFE 관련 wiring, split, visualization 문서는 `safe/` 아래에 둔다.
- 논문 PDF와 외부 reference는 `references/` 아래에 둔다.
