# GR00T RoboCasa Documentation Map

이 디렉터리는 GR00T와 RoboCasa가 만나는 실행 문서의 기준 위치다. GR00T fine-tuning/eval, RoboCasa scenario 재현성, SAFE feature export와 detector 재현 문서를 한 트리에서 관리한다.

파일명에 두 자리 번호 prefix를 붙여 읽는 순서를 표시한다. N1.6 trunk와 N1.5 reference trunk는 독립이다.

## N1.6 Reading Order

1. [01 Fine-Tuning](n16_01_finetune.md) — Isaac-GR00T `n1.6-release` 기반 PandaOmron fine-tuning runbook
2. [02 Evaluation](n16_02_eval.md) — ZMQ eval workflow, SR 계산, troubleshooting
3. [03 SAFE Overview](n16_03_safe_overview.md) — SAFE wiring 결론, Checkpoint/Env, Run config, Pipeline order
4. [04 SAFE Collection](n16_04_safe_collection.md) — ZMQ feature server, RoboCasa365 collection, ah8/ah16 mode
5. [05 Scenario Reproduction](n16_05_safe_env_reproduction.md) — scenario seed + ep_meta manifest 재현 범위
6. [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md) — 한 datapoint의 시간/단위 의미
7. [07 SAFE Detector](n16_07_safe_detector.md) — Paper-faithful split + LSTM 학습 + CP 운영점
8. [08 SAFE Visualization](n16_08_safe_visualization.md) — t-SNE / overlay / silhouette 진단
9. [09 SAFE Parity](n16_09_safe_parity.md) — ZMQ official baseline parity + HTTP `/act` parity
10. [10 SAFE Report](n16_10_safe_report.md) — 축소 재현 end-to-end 결과 보고서

## N1.5 Reading Order

1. [N1.5 01 Fine-Tuning](n15_01_finetune.md)
2. [N1.5 02 Evaluation](n15_02_eval.md)

## 문서 경계

| Area | Canonical docs | Role |
|---|---|---|
| N1.6 train/eval | `n16_01_finetune.md`, `n16_02_eval.md` | GR00T N1.6 PandaOmron checkpoint 학습과 ZMQ 평가 |
| N1.6 SAFE pipeline | `n16_03_safe_overview.md`..`n16_10_safe_report.md` | feature export, split, detector, visualization, report |
| N1.6 SAFE reference | `n16_05_safe_env_reproduction.md`, `n16_06_safe_inference_semantics.md` | scenario reproduction + datapoint semantics |
| N1.5 | `n15_01_finetune.md`, `n15_02_eval.md` | Isaac-GR00T N1.5 GR1/PandaOmron reference workflow |
| Legacy | `_legacy/robocasa_finetune_setup.md` | 초기 setup note. 현행 runbook의 기준이 아니다 |

SAFE 관련 script는 계속 `scripts/safe/groot_n16/robocasa/` 아래에 둔다. 문서는 `docs/groot/`에서 번호 prefix로 reading order를 표현한다.

## 용어 기준

GR00T RoboCasa 문서에서 자주 쓰는 세부 용어는 아래 의미로 읽는다.

| 용어 | 의미 | 자세한 기준 |
|---|---|---|
| Upstream GR00T ZMQ evaluation | GR00T RoboCasa success-rate 기준 경로 | `n16_02_eval.md`, `n16_09_safe_parity.md` |
| Project FastAPI evaluation | 여러 VLA policy를 공통 HTTP `/act` API로 비교하는 경로 | `n16_09_safe_parity.md` |
| SAFE wiring | VLA rollout에서 action과 latent feature를 SAFE-readable artifact로 내보내는 연결 경로 | `n16_03_safe_overview.md` |
| SAFE feature vector | VLA latent feature를 token/horizon/diffusion 축에서 aggregation한 timestep-level detector input | `n16_06_safe_inference_semantics.md`, `n16_10_safe_report.md` |
| datapoint | `hidden_states[t]` 하나. GR00T inference 1회의 DiT action-token latent | `n16_06_safe_inference_semantics.md` |
| scenario / scene composition | layout/style, object cfg, texture, fixture reference, camera/config, robot base pose 수준의 task instance | `n16_05_safe_env_reproduction.md` |
| `scenario_seed` | RoboCasa env construction seed. 현재 collector의 `--seed` 값 | `n16_05_safe_env_reproduction.md` |
| `ep_meta` manifest | `(env_name, scenario_seed)`에 대응하는 RoboCasa scenario 기록 JSON | `n16_05_safe_env_reproduction.md` |
| `ah8` / `ah16` | SAFE feature export horizon과 RoboCasa execution step을 8 또는 16으로 맞춘 paired collection mode | `n16_04_safe_collection.md`, `n16_06_safe_inference_semantics.md` |
| SR | success rate. task 또는 run에서 success episode 비율 | `n16_02_eval.md`, `n16_10_safe_report.md` |
| T-det | detection time을 rollout length로 정규화한 값. 작을수록 early detection | `n16_10_safe_report.md` |
| CP operating point | detector score 위에 conformal threshold/band를 고정한 운영점 | `n16_07_safe_detector.md`, `n16_10_safe_report.md` |

현재 label scope는 rollout-level success/failure다. Inference-step-level failure onset/intervention label은 아직 별도 protocol로 정의해야 한다.
