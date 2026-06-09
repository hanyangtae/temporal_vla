# Temporal VLA Documentation

이 디렉터리는 Temporal VLA의 실행 절차, 실험 결정, 결과 보고, 논문 reference를 함께
보관한다. 문서는 단순 파일 목록이 아니라 **무엇을 하는 프로젝트인지 → 어떤 절차를
따라야 하는지 → 결과와 남은 판단은 어디에 있는지**를 빠르게 찾게 하는 entrypoint다.

## Project Goal

현재 프로젝트 목표는 VLA latent에서 성공/실패 표현을 구분하고, 추론 시 활성화를 성공
부분공간으로 steering 해서 RoboCasa/Calvin/LIBERO 같은 벤치마크의 Success Rate를 올리는
것이다. VLA 백본 추가학습 없이 intervention 효과를 확인하는 것이 핵심이며, 이전 TTT/VITA
progress-predictor 방향은 `ttt/` 아래의 보존 문서로 남긴다.

인프라는 Docker container로 모델 서버와 벤치마크를 분리하고, 공통 FastAPI `/act`
계약으로 여러 policy를 같은 evaluator에 붙이는 구조다. GR00T RoboCasa처럼 upstream parity가
중요한 경로는 별도 adapter/runbook을 둔다.

## How To Use These Docs

1. 공통 serving/eval 작업은 `01` → `02` → `03` 순서로 읽는다.
2. GR00T RoboCasa 기준선, SAFE wiring, N1.5/LeRobot 실험은 `groot/README.md`에서
   N1.6/N1.5 trunk를 고른다.
3. 메인 연구 결과와 steering 방향은 `steering/README.md`에서 phase별 문서를 따라간다.
4. 환경 재현성, task 이름, seed 의미는 `benchmarks/`와 `groot/n16_05_*`를 같이 본다.
5. 장기 결정은 `adr/`, 외부 논문은 `references/`, 예전 방향은 `_legacy/` 또는 `ttt/`에 둔다.

번호 prefix(`01_`, `02_`, ...)는 각 scope 안에서 **읽기 순서**를 의미한다.

## Project-Wide Reading Order

처음 들어오면 아래 순서대로 본다. 각 문서는 자기 위·아래 문서를 cross-ref한다.

1. [01 Serving Interface](01_serving_interface.md) — 통일 HTTP API 단일 출처. endpoint(`/act`, `/act_with_features`, `/reset`, `/health`), sub-key 네임스페이스, 모델 × 벤치마크 호환 매트릭스, 운영 패턴 5종. 모델/벤치/체크포인트 작업의 모든 출발점.
2. [02 Docker Guide](02_docker_guide.md) — 컨테이너 구성·기동·VNC/X11·troubleshooting. 위 API 를 실제로 띄우는 방법.
3. [03 Adding Checkpoint](03_adding_checkpoint.md) — 새 VLA 체크포인트를 profile/serve/eval 경로에 붙이는 7단계 체크리스트.
4. [CONTRIBUTING](CONTRIBUTING.md) — git/PR 워크플로우.

부가 runbook: [Cache Paths](cache_paths.md) — 체크포인트·데이터셋의 repo 밖 cache 위치와 코드에서의 경로 참조 규칙(`path_setup.py` / `cache_env.sh`).

## Current Results And Status

- GR00T RoboCasa runtime/SR/SAFE 결과는 [`groot/README.md`](groot/README.md)의
  "최근 실행 결과"와 [`groot/n16_10_safe_report.md`](groot/n16_10_safe_report.md)를 본다.
- Latent steering 표현 분석과 intervention 후보는 [`steering/README.md`](steering/README.md)에서
  phase별 문서를 따라간다.
- 환경 결정성, task 이름 차이, seed replay 문제는 [`benchmarks/`](benchmarks/)와
  [`groot/n16_05_safe_env_reproduction.md`](groot/n16_05_safe_env_reproduction.md)를 같이 본다.
- 일회성 상태 점검 snapshot은 root runbook으로 두지 않고 `_legacy/` 아래에 보관한다.

## GR00T RoboCasa

GR00T 학습·평가·SAFE feature export 문서는 [`groot/`](groot/README.md) 아래에 별도 번호로 정리. README에 reading order 가 있다.

### N1.6
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
11. [HTTP Act Changes](groot/n16_11_http_act_changes.md) — HTTP `/act` + `/act_with_features` wiring 변경 일지 (GR00T 한정)

최종 detector artifact:

```text
outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector
```

### N1.5
1. [N1.5 Fine-Tuning](groot/n15_01_finetune.md)
2. [N1.5 Evaluation](groot/n15_02_eval.md)
3. [N1.5 LeRobot RoboCasa365 Pipeline](groot/n15_03_lerobot_robocasa365.md) — serve→HTTP→robocasa365→analysis UI 4-stage. overview/status(`n15_03`) + stage별 명세 `n15_04`(serve) `n15_05`(obs bridge)
4. [N1.5 Native ZMQ OpenFridge Smoke](groot/n15_07_native_zmq_openfridge.md) — LeRobot wrapper mismatch를 분리하기 위한 Isaac-GR00T N1.5 native baseline
5. [N1.5 LeRobot Internal Parity](groot/n15_08_lerobot_internal_parity.md) — SR 외 checkpoint-load 검증과 historical internal evidence

## Latent Steering (메인 연구 라인)

succ/fail latent 구분 → steer 로 SR↑ (COAST 계열). 메인 method 의 표현 분석·재현·진행 기록은
[`steering/`](steering/README.md) 아래에 번호 prefix 로 정리. README 에 reading order 가 있다.

1. [seen18 Latent Analysis](steering/01_seen18_latent_analysis.md) — succ/fail 분리 표현 분석 (길이 confound 통제 전제)
2. [seen18 Handoff](steering/02_seen18_handoff.md) — cross-task 실패 심화 + steering/detector 핸드오프
3. [COAST Report](steering/03_coast_report.md) — conceptor 구현 + GR00T N1.6 적용 진행 보고
4. [COAST Reproduction Map](steering/04_coast_reproduction_map.md) — 코드 ↔ COAST 논문 식/섹션 매핑
5. [SAFE lerobot Collection](steering/05_safe_lerobot_collection.md) — SAFE 수집 lerobot 멀티벤치 확장 (plan+status)

## TTT Methodology

- [TTT Pipeline](ttt/ttt_pipeline.md): GR00T N1.6 + TTT RoboCasa pipeline 정리.
- [Phase 1 Progress Predictor](ttt/progress_predictor.md): VITA 기반 progress predictor 구현 및 학습 정리.
- [Phase 1 Status](ttt/2026_04_01_status.md): Phase 1 진행 상태 snapshot.

## Benchmark References

- [RoboCasa Task Name Mapping](benchmarks/robocasa_task_name_mapping.md): RoboCasa v0.2 (`robocasa_v02`) task 와 robocasa365 v1.0 task 이름 대응.
- [RoboCasa Env Reproducibility](benchmarks/robocasa_env_reproducibility.md): RoboCasa env 결정성(seed/rollout)과 PC 간 재현 절차 (모델 무관).

## ADR (Architecture Decisions)

- [0001 Dedicated SAFE GR00T N1.6 ZMQ server](adr/0001-dedicated-safe-groot-n16-zmq-server.md)
- [0002 GR00T N1.6 SAFE feature dual transport](adr/0002-groot-n16-safe-feature-dual-transport.md)

## Paper References

논문 PDF는 운영 runbook과 분리해서 [`references/`](references/) 아래에 둔다. 일부는 추출 텍스트(`.txt`)도 함께 둔다.

- `references/COAST.pdf` (+ `COAST.txt`) — Contrastive Conceptor Activation Steering. latent steering 메인 reference.
- `references/SAFE.pdf` (+ `SAFE.txt`) — SAFE failure detection. latent 수집·detector reference.
- `references/NOT ALL FEATURES ARE CREATED EQUAL_ICLR2026.pdf` (+ `NOTALL.txt`) — VLA mechanistic study (ICLR 2026).
- `references/CoT-VLA.pdf`
- `references/Scaling World Model.pdf`
- `references/VITA.pdf`
- `references/robocasa365.pdf`

## Legacy

- [groot/_legacy/robocasa_finetune_setup.md](groot/_legacy/robocasa_finetune_setup.md) — 초기 GR00T x RoboCasa fine-tuning setup. 현행 N1.6 기준은 `groot/n16_01_finetune.md`.
- [ttt/_legacy/groot_loop_analysis_plan.md](ttt/_legacy/groot_loop_analysis_plan.md) — 초기 loop analysis 계획. 문서 상단의 legacy note 를 먼저 확인한다.
- [_legacy/status_report_20260526.md](_legacy/status_report_20260526.md) — 2026-05-26 read-only repo 상태 점검 snapshot. 현행 entrypoint가 아니다.

## Organization Rule

- **Project-wide runbook** 은 `docs/` 루트에 두고 번호 prefix(`01_`, `02_`, `03_`) 로 reading order 를 표현한다.
- **GR00T 실행 절차** 는 `groot/` 아래에 번호 prefix(`n16_01_` ~ `n16_NN_`, `n15_NN_`) 로 reading order 를 표현한다.
- **Latent steering (메인 method) 의 분석·재현·진행 기록** 은 `steering/` 아래에 번호 prefix(`01_` ~ `NN_`) 로 reading order 를 표현한다.
- **TTT 연구 방법론·phase 진행 기록** 은 `ttt/` 아래에 둔다 (무기한 연기됨).
- **벤치마크 reference** (task 이름 매핑 등) 는 `benchmarks/` 아래에 둔다.
- **장기 결정** 은 `adr/` 아래에 둔다.
- **논문 PDF / 외부 reference** 는 `references/` 아래에 둔다.
- **Legacy / 폐기된 문서** 는 해당 scope의 `_legacy/` 서브디렉터리에 둔다. Scope가 애매한 repo-wide snapshot은 `docs/_legacy/`에 둔다. 파일명 prefix `legacy_*` 는 사용하지 않는다.
