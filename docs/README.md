# Temporal VLA Documentation

이 디렉터리는 Temporal VLA의 실행 절차, 실험 결정, 결과 보고, 논문 reference를 함께
보관한다. 문서는 단순 파일 목록이 아니라 **무엇을 하는 프로젝트인지 → 어떤 절차를
따라야 하는지 → 결과와 남은 판단은 어디에 있는지**를 빠르게 찾게 하는 entrypoint다.

## Project Goal

현재 프로젝트 목표는 VLA latent에서 성공/실패 표현을 **pathway별(VL goal vs DiT motor)** 로
구분하고, 각 pathway의 **phase/time-dependent 신호**(VL 이른 신호 t≤8, DiT 늦은 신호 t≥12)에
맞춰 추론 시 활성화를 성공 부분공간으로 steering 해서 Success Rate를 올리는 것이다 (백본
재학습 없음). 핵심 난제는 **online phase/failure-type 식별** — 추론 중 어느 pathway·어느
phase에서 실패하는지 알아야 steering을 라우팅할 수 있다. 메인 라인 단일 출처는
[`steering/`](steering/README.md), 특히 `steering/14_pathway_phase_online_steering.md`. 이전
TTT/VITA progress-predictor 방향은 메인이 아니며 [`ttt/`](ttt/README.md) 보존 문서로 남기되,
phase 신호 공급용 보조 부품으로 복귀 검토 가능하다.

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
  "최근 실행 결과"와 [`groot/n16_07_safe_detector_report.md`](groot/n16_07_safe_detector_report.md)를 본다.
- Latent steering 표현 분석과 intervention 후보는 [`steering/README.md`](steering/README.md)에서
  phase별 문서를 따라간다.
- Activation steering 분야 전반(기초→LLM/VLM→산업→VLA) 학습·면접 대비 서베이는
  [`Activation_steering_basic/00_activation_steering_survey.md`](Activation_steering_basic/00_activation_steering_survey.md)
  (7-섹션 흐름 + 논문 정독 노트 52편 `Activation_steering_basic/notes/`, Notion 미러 "Activation steer 전반 공부").
  기초 논문 PDF는 repo에 안 담음 — `_handoff/research_raw/master_index.md`의 arXiv ID로 재다운로드.
- 환경 결정성, task 이름 차이, seed replay 문제는 [`benchmarks/`](benchmarks/)와
  [`groot/n16_05_safe_env_reproduction.md`](groot/n16_05_safe_env_reproduction.md)를 같이 본다.
- 일회성 상태 점검 snapshot은 root runbook으로 두지 않고 `_legacy/` 아래에 보관한다.

## GR00T RoboCasa

GR00T 학습·평가·SAFE feature export 문서는 [`groot/`](groot/README.md) 아래에 별도 번호로 정리. README에 reading order 가 있다.
N1.5와 N1.6의 DiT token layout은 대칭이 아니므로 feature shape 비교 전에
[`groot/README.md`](groot/README.md#n15-n16-feature-contract)의
quick reference를 먼저 확인한다. 핵심 차이는 N1.6 full residual `T=51 = state(1)+action(50)`,
N1.5 aligned residual `T=49 = state(1)+future_tokens(32)+action(16)`이다.

GR00T rollout·activation 데이터가 로컬/원격(승준 HDD) 어느 경로에 무엇이 있는지는
[GR00T Data Catalog](groot/00_groot_data_catalog.md)에 정리돼 있다. 원격 데이터로 분석·fit을
돌리기 전에 먼저 본다.

처음 구조를 파악할 때는 [GR00T Flow Map](groot/00_groot_flow_map.md)을 먼저 본다. 이 문서는
LeRobot/native, RoboCasa365, ZMQ/HTTP entry point가 어떤 파일과 함수를 지나 어떤 값을 전달하는지
초보자용 call chain으로 정리한다. 같은 흐름을 latent steering 수식과 함께 화면에서 훑을 때는
[GR00T Latent Steering Explorer](groot/00_groot_steering_explorer.html)를 연다.

### N1.6
1. [Fine-Tuning](groot/n16_01_finetune.md)
2. [Evaluation](groot/n16_02_eval.md)
3. [SAFE Overview](groot/n16_03_safe_overview.md)
4. [SAFE Collection](groot/n16_04_safe_collection.md) — ah8/ah16 mode 포함
5. [Scenario Reproduction](groot/n16_05_safe_env_reproduction.md) — scenario seed + ep_meta
6. [Inference Datapoint Semantics](groot/n16_06_safe_inference_semantics.md) — 한 datapoint 의미
7. [SAFE Detector + Report](groot/n16_07_safe_detector_report.md) — split+LSTM+CP 운영점 + t-SNE/silhouette + 축소 재현 결과 (구 n16_07/08/10 통합)
8. [SAFE Parity](groot/n16_09_safe_parity.md) — ZMQ official baseline parity + HTTP `/act` parity
9. [HTTP Act Changes](groot/n16_11_http_act_changes.md) — HTTP `/act` + `/act_with_features` wiring 변경 일지 (GR00T 한정)

최종 detector artifact:

```text
outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector
```

### N1.5
1. [N1.5 Fine-Tuning](groot/n15_01_finetune.md)
2. [N1.5 Evaluation](groot/n15_02_eval.md)
3. [N1.5 LeRobot RoboCasa365 Pipeline](groot/n15_03_lerobot_robocasa365.md) — serve adapter / obs bridge / native ZMQ baseline / internal parity 통합 (구 n15_04~07)

## Latent Steering (메인 연구 라인)

succ/fail latent 구분 → steer 로 SR↑ (COAST 계열). 메인 method 의 표현 분석·재현·진행 기록은
[`steering/`](steering/README.md) 아래에 번호 prefix 로 정리. README 에 reading order 가 있다.

0. [★ Pathway+Phase Online Steering (SSOT)](steering/14_pathway_phase_online_steering.md) — 현재 연구 방향·가설·open problem·ablation 단일 출처
1. [seen18 Latent Analysis](steering/01_seen18_latent_analysis.md) — succ/fail 분리 표현 분석 (길이 confound 통제 전제, cross-task 실패 구조)
2. [SAFE lerobot Collection](steering/05_safe_lerobot_collection.md) — SAFE 수집 lerobot 멀티벤치 확장 (plan+status)
3. [Steering Methods Survey](steering/07_steering_methods_survey.md) — conceptor steering(COAST 계열) + 이후 적용 후보
4. [Pathway Separation Analysis](steering/08_pathway_separation_analysis.md) — DiT 32-layer + VL/DiT 비교 + Phase 4 target (구 08+09 통합)
5. [Instruction Confound](steering/11_instruction_confound.md) — VL AUROC instruction 아티팩트 판정 + VL/DiT LDA 사분면
6. [Phase 4 N1.5 Instruction-Fixed Plan](steering/11_phase4_n15_instruction_fixed_plan.md) — N1.5 instruction-fixed runbook
7. [Lab-Meeting PPT Notes](steering/13_lab_meeting_ppt_notes.md) — 발표 덱 작업 노트 (구 13+17 통합)
8. [Research Structure](steering/15_research_structure.md) — RQ1~4 / 가설 C1~C4

한 화면 요약: [GR00T Latent Steering Explorer](groot/00_groot_steering_explorer.html) — GR00T runtime flow와
conceptor/hidden-state steering 수식, VL/DiT pathway 상태, code artifact map을 함께 보는 self-contained interactive HTML.

## TTT / Progress Predictor Archive

현재 메인 연구 라인은 아니다. 과거 실패 루프 탈출/ProgressPredictor 방향은
[`ttt/README.md`](ttt/README.md)에서 보존 문서로 관리한다.

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

## Related Work Notes

논문 원문과 별도로, 우리 방법과의 차이를 정리한 reviewer 대응용 노트는
[`related_work/`](related_work/) 아래에 둔다.

- [PPGuide](related_work/ppguide.md) — inference-time policy guidance와 우리 mode-conditional latent steering의 차이.
- [RoboMD](related_work/robomd.md) — 외부 VL embedding 기반 vulnerability diagnosis와 frozen VLA latent 개입의 차이.

## Legacy
- [_legacy/status_report_20260526.md](_legacy/status_report_20260526.md) — 2026-05-26 read-only repo 상태 점검 snapshot. 현행 entrypoint가 아니다.

## Organization Rule

- **Project-wide runbook** 은 `docs/` 루트에 두고 번호 prefix(`01_`, `02_`, `03_`) 로 reading order 를 표현한다.
- **GR00T 실행 절차** 는 `groot/` 아래에 번호 prefix(`n16_01_` ~ `n16_NN_`, `n15_NN_`) 로 reading order 를 표현한다.
- **Latent steering (메인 method) 의 분석·재현·진행 기록** 은 `steering/` 아래에 번호 prefix(`01_` ~ `NN_`) 로 reading order 를 표현한다.
- **TTT 연구 방법론·phase 진행 기록** 은 `ttt/` 아래 단일 README 로 둔다 (legacy/demoted; `src/ttt/` 코드는 보존).
- **벤치마크 reference** (task 이름 매핑 등) 는 `benchmarks/` 아래에 둔다.
- **장기 결정** 은 `adr/` 아래에 둔다.
- **논문 PDF / 외부 reference** 는 `references/` 아래에 둔다.
- **Related work 해석 노트** 는 `related_work/` 아래에 둔다.
- **Legacy / 폐기된 문서** 는 해당 scope의 `_legacy/` 서브디렉터리에 둔다. Scope가 애매한 repo-wide snapshot은 `docs/_legacy/`에 둔다. 파일명 prefix `legacy_*` 는 사용하지 않는다.
