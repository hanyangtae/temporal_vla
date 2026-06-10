# GR00T RoboCasa Documentation Map

이 디렉터리는 GR00T와 RoboCasa가 만나는 실행 문서의 기준 위치다. 목표는 GR00T N1.6/N1.5
체크포인트를 RoboCasa에서 재현 가능하게 평가하고, 같은 rollout에서 action과 latent feature를
SAFE-readable artifact로 내보내는 절차를 유지하는 것이다.

파일명에 두 자리 번호 prefix를 붙여 읽는 순서를 표시한다. N1.6 trunk와 N1.5 reference trunk는
독립이며, 새 실행 기록은 먼저 canonical runbook/result 문서에 흡수하고 필요할 때만 별도 문서로
분리한다.

구조상 가장 중요한 전제는 N1.5와 N1.6이 대칭이 아니라는 점이다. N1.6 reusable backend
library는 `src/policies/groot/`에 모여 있지만, `scripts/safe/groot_n15/robocasa/`는 그
N1.5판이 아니다. N1.5 tree는 eval client, split helper, checkpoint/runtime compatibility
helper만 맡고, model loader/schema/IO/serving/SAFE feature capture 역할은 갖지 않는다.
처음 읽을 때는 [00 Flow Map](00_groot_flow_map.md)의 "N1.5 And N1.6 Are Asymmetric"
섹션을 먼저 확인한다.

## 사용 절차

1. GR00T 관련 코드가 파일 사이에서 어떻게 이어지는지 처음 파악할 때는
   [00 Flow Map](00_groot_flow_map.md)으로 profile/server/client/adapter 흐름을 먼저 본다.
2. N1.6 기준선을 다룰 때는 `n16_01`에서 checkpoint/data 준비를 확인하고 `n16_02`로 ZMQ 평가를
   실행한다.
3. SAFE 수집이나 feature 의미를 다룰 때는 `n16_03` overview에서 현재 결론을 잡고 `n16_04`부터
   collection/replay/feature/detector/report 순서로 내려간다.
4. HTTP `/act` 또는 `/act_with_features`를 다룰 때는 `n16_09`의 parity 상태와 `n16_11`의 코드
   변경 범위를 같이 확인한다. HTTP 결과를 Upstream GR00T ZMQ evaluation 기준선으로 혼동하지 않는다.
5. N1.5와 LeRobot 실험은 `n15_03`을 overview로 보고, stage별 세부는 `n15_04`와 `n15_05`에서
   확인한다.

## Architecture Entry

- [00 Flow Map](00_groot_flow_map.md) — GR00T N1.6/N1.5, LeRobot/native,
  RoboCasa365, ZMQ/HTTP 경로를 entrypoint별 call chain과 전달 값 기준으로 설명하는 입문용 지도
- [00 Latent Steering Explorer](00_groot_steering_explorer.html) — 위 flow map과
  latent steering 핵심 수식(conceptor, hidden-state gate, VL/DiT pathway)을 한 화면에서
  추적하는 self-contained interactive HTML 지도

## N1.6 Reading Order

1. [01 Fine-Tuning](n16_01_finetune.md) — Isaac-GR00T `n1.6-release` 기반 PandaOmron fine-tuning runbook
2. [02 Evaluation](n16_02_eval.md) — ZMQ eval workflow, SR 계산, troubleshooting
3. [03 SAFE Overview](n16_03_safe_overview.md) — SAFE wiring 결론, Checkpoint/Env, Run config, Pipeline order
4. [04 SAFE Collection](n16_04_safe_collection.md) — ZMQ feature server, RoboCasa365 collection, ah8/ah16 mode
5. [05 Scenario Reproduction](n16_05_safe_env_reproduction.md) — scenario seed + ep_meta manifest 재현 범위
6. [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md) — 한 datapoint의 시간/단위 의미
7. [07 SAFE Detector](n16_07_safe_detector.md) — Paper-faithful split + LSTM 학습 + CP 운영점
8. [08 SAFE Visualization](n16_08_safe_visualization.md) — t-SNE / overlay / silhouette 진단
9. [09 SAFE Parity](n16_09_safe_parity.md) — ZMQ official baseline parity + HTTP `/act`/`/act_with_features` parity
10. [10 SAFE Report](n16_10_safe_report.md) — 축소 재현 end-to-end 결과 보고서
11. [11 HTTP Act Changes](n16_11_http_act_changes.md) — 이번 HTTP `/act` wiring 변경점과 검증 상태
12. [12 RoboCasa Refactor Report](n16_12_robocasa_refactor_report.md) — GR00T RoboCasa HTTP/ZMQ 리팩토링 구조와 책임 경계

## N1.5 Reading Order

1. [N1.5 01 Fine-Tuning](n15_01_finetune.md)
2. [N1.5 02 Evaluation](n15_02_eval.md)
3. [N1.5 03 LeRobot RoboCasa365 Pipeline — Overview & Status](n15_03_lerobot_robocasa365.md) — serve→HTTP→robocasa365→analysis UI 4-stage map, 검증 상태/향후 계획
4. [N1.5 04 Serve Adapter Spec](n15_04_lerobot_serve_adapter.md) — stage [1]: checkpoint 형식, profile 필드 명세, serve/smoke, 구현 구조
5. [N1.5 05 Closed-loop Obs Bridge Spec](n15_05_lerobot_obs_bridge.md) — stage [2][3]: 카메라/state 키 매핑 명세, gap 분석, 수정안
6. [N1.5 07 Native ZMQ OpenFridge Smoke](n15_07_native_zmq_openfridge.md) — LeRobot mismatch 분리용 Isaac-GR00T N1.5 ZMQ comparison note
7. [N1.5 08 LeRobot Internal Parity](n15_08_lerobot_internal_parity.md) — SR가 아닌 checkpoint-load 검증과 historical internal evidence

## 문서 경계

| Area | Canonical docs | Role |
|---|---|---|
| N1.6 train/eval | `n16_01_finetune.md`, `n16_02_eval.md` | GR00T N1.6 PandaOmron checkpoint 학습과 ZMQ 평가 |
| N1.6 SAFE pipeline overview | `n16_03_safe_overview.md` | 현재 결론, run config, pipeline order, 다음 단계 |
| N1.6 SAFE collection | `n16_04_safe_collection.md` | ZMQ feature server와 rollout 수집 runbook |
| N1.6 replay semantics | `n16_05_safe_env_reproduction.md` | `scenario_seed` / `ep_meta` 보장 범위와 PC 간 replay 절차 |
| N1.6 feature semantics | `n16_06_safe_inference_semantics.md` | datapoint, time axis, feature/action horizon 의미 |
| N1.6 detector/visualization/report | `n16_07_safe_detector.md`, `n16_08_safe_visualization.md`, `n16_10_safe_report.md` | detector 학습, 시각화 진단, 결과 보고 |
| N1.6 parity validation | `n16_09_safe_parity.md` | ZMQ/HTTP action parity, SAFE transport parity, runtime validation 수치, 남은 replay 한계의 단일 출처 |
| N1.6 HTTP implementation changelog | `n16_11_http_act_changes.md` | HTTP `/act` / `/act_with_features` 코드 계약과 변경점. 검증 수치는 `n16_09`로 링크 |
| N1.6 RoboCasa refactor architecture | `n16_12_robocasa_refactor_report.md` | GR00T RoboCasa 전용 processor, HTTP/ZMQ transport, shared contract 책임 경계 |
| N1.5 reference | `n15_01_finetune.md`, `n15_02_eval.md` | Isaac-GR00T N1.5 GR1/PandaOmron reference workflow (ZMQ) |
| N1.5 LeRobot pipeline | `n15_03`(overview/status), `n15_04`(serve spec), `n15_05`(obs bridge spec) | LeRobot serve→HTTP→robocasa365→analysis UI 4-stage. n15_03이 map, 04~05가 stage별 명세 |
| N1.5 native/internal comparison | `n15_07_native_zmq_openfridge.md`, `n15_08_lerobot_internal_parity.md` | native Isaac-GR00T N1.5 ZMQ smoke와 SR 외 내부값 parity 검증 |
| Legacy | `_legacy/robocasa_finetune_setup.md` | 초기 setup note. 현행 runbook의 기준이 아니다 |

SAFE 관련 script는 계속 `scripts/safe/groot_n16/robocasa/` 아래에 둔다. 문서는 `docs/groot/`에서 번호 prefix로 reading order를 표현한다.

## 용어 기준

Repo-wide 용어는 [`../../CONTEXT.md`](../../CONTEXT.md)를 기준으로 한다. GR00T RoboCasa 문서에서 자주 쓰는 세부 용어는 아래 의미로 읽는다.

| 용어 | 의미 | 자세한 기준 |
|---|---|---|
| Upstream GR00T ZMQ evaluation | GR00T RoboCasa success-rate 기준 경로 | `n16_02_eval.md`, `n16_09_safe_parity.md` |
| Project FastAPI evaluation | 여러 VLA policy를 공통 HTTP `/act` API로 비교하는 경로 | `n16_09_safe_parity.md` |
| SAFE wiring | VLA rollout에서 action과 latent feature를 SAFE-readable artifact로 내보내는 연결 경로 | `n16_03_safe_overview.md`, `../../CONTEXT.md` |
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

## 최근 실행 결과

### Runtime Recheck 2026-06-09

목적은 SR 재측정이 아니라 현재 checkout/container에서 각 serving path가 정상 action을
반환하는지 확인하는 것이다. Host sandbox에서 일부 HTTP port 접근이 끊기는 경우가 있어
HTTP health/smoke는 해당 model container 내부 `127.0.0.1` 기준으로 확인했다.
이 섹션은 실행 당시 로그 요약이다. `outputs/debug/recheck_20260609_*` one-off 영상
디렉터리는 현재 host에 남아 있지 않으므로, 재사용 가능한 artifact가 필요하면 같은 명령으로
다시 생성한다.

| Case | Current action check | Closed-loop note |
|---|---|---|
| LeRobot pi0.5 + LIBERO HTTP | `lerobot_pi05__libero` `/health` OK, warm smoke OK. `/act` returned finite `action.eef_pos` `[1,3]`, `action.eef_axisangle` `[1,3]`, `action.gripper` `[1,1]`. | First CUDA load OOMed while LeRobot GR00T N1.5 was still on GPU. After freeing VRAM it loaded. First `/act` spent more than 60s in Torch Inductor/autotune; warm call was ~149ms. LIBERO closed-loop rollout was not rerun in this pass. |
| Native GR00T N1.5 ZMQ + RoboCasa365 | Real OpenFridge reset obs through `groot_n15` ZMQ returned finite keys `action.end_effector_position`, `action.end_effector_rotation`, `action.gripper_close`, `action.base_motion`, `action.control_mode`, each `[1,16,D]`. | Current OpenFridge target 1ep smoke ended `success_rate=0.0`; this official client uses `get_task_horizon(args.task)` when `--max-episode-steps` is omitted, so the failure is not explained by a shortened 400-step cap. No seed is exposed by this official client, so this is not a deterministic repeat of the earlier successful scene. |
| Native GR00T N1.6 ZMQ + RoboCasa365 | `run_gr00t_server.py` with checkpoint `grootn16_robocasa365_multitask_learning/checkpoint-120000` and `NEW_EMBODIMENT` returned finite real-obs action keys `[1,16,D]`. | Current OpenFridge 1ep smoke reached the env/action loop but ended `success_rate=0.0` with `MAX_STEPS=400`. The N1.6 eval runbook/default script uses `MAX_STEPS=720`, so this is an action-loop smoke result, not a canonical SR failure. During this pass `scripts/eval/groot_robocasa_zmq_eval.py` needed a compatibility guard because the current upstream helper does not accept `eval_seed`. |
| Native GR00T N1.6 HTTP + RoboCasa365 | `groot__robocasa365_ckpt120000` `/health` OK and smoke OK. `/act` returned finite `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode`, each `[16,D]`. | Closed-loop HTTP SR was not established in this pass; endpoint/schema/action health is OK. Generic `scripts/eval/robocasa_eval.py` defaults to `--num-steps 400`, so N1.6 HTTP SR 확인은 `--num-steps 720`로 별도 rerun해야 한다. |
| LeRobot GR00T N1.5 HTTP + RoboCasa365 | `lerobot_groot_n15__robocasa365_ckpt120000` `/health` OK and smoke OK. `/act` returned finite five-key response with `[1,D]` per key. | OpenFridge target seed 0 rerun succeeded: `success_rate=1.0`, first success step `209`. The official LeRobot HTTP eval client also uses the RoboCasa task horizon when `--max-episode-steps` is omitted, so this path is not blocked by the default time setting. |

결론: action 반환 자체는 위 다섯 경로 모두 현재 재현된다. 다만 native N1.5/N1.6 ZMQ의
단일 OpenFridge rollout은 이번 pass에서 실패했으므로, "action 정상"과 "SR 정상"은
분리해서 보고해야 한다. 성능 재확인은 seed 고정 가능 client 또는 task별 반복 rollout으로
별도 수행한다.

### Runtime Recheck 720-step 3ep 2026-06-09

위 action smoke 이후, 같은 다섯 case를 3 episodes/trials로 다시 확인했다. RoboCasa
case는 `OpenFridge` 기준이며 max horizon을 `720`으로 명시했다. LIBERO case는 같은
`720` cap과 deadline으로 `libero_10` 첫 task 3 trials만 확인했다.

Artifact root:

```text
outputs/eval/recheck_720_3ep_20260609_202347
```

| Case | Eval target | Result | Notes |
|---|---|---:|---|
| LeRobot pi0.5 + LIBERO HTTP | `libero_10`, task 0, 3 trials, `max_steps=720` | 3/3 | terminated steps: 249, 249, 241. First run downloaded LIBERO assets into the container user cache. |
| Native GR00T N1.5 ZMQ + RoboCasa365 | `robocasa/OpenFridge`, `split=target`, 3 episodes, `--max-episode-steps 720` | 2/3 | Results: `[True, False, True]`. Videos saved under `native_groot_n15_zmq/videos`. |
| Native GR00T N1.6 ZMQ + RoboCasa365 | `robocasa_panda_omron/OpenFridge_PandaOmron_Env`, 3 episodes, `MAX_STEPS=720` | 0/3 | `per_episode.tsv` records all three failures. The server log shows this run loaded the RoboCasa365 checkpoint with `ROBOCASA_PANDA_OMRON`, while the checkpoint profile expects `NEW_EMBODIMENT`. |
| Native GR00T N1.6 HTTP + RoboCasa365 | `OpenFridge`, `--use-groot-env`, 3 rollouts, `--num-steps 720`, seed 0 | 2/3 | Success at steps 652 and 458; seed 2 reached 720 without success. Combined video: `native_groot_n16_http/videos/OpenFridge.mp4`. |
| LeRobot GR00T N1.5 HTTP + RoboCasa365 | `robocasa/OpenFridge`, `split=target`, 3 episodes, `--max-episode-steps 720`, seed 0 | 3/3 | Success steps: 204, 355, 253. Videos saved under `lerobot_groot_n15_http/videos`. |

Operational conclusion: the `720` horizon fixes the interpretation of the earlier short smoke, but it does
not make every path pass. The strongest new signal is the N1.6 split: HTTP succeeds 2/3 with
`new_embodiment`, while ZMQ failed 0/3 after starting the same checkpoint with `ROBOCASA_PANDA_OMRON`.
For RoboCasa365 checkpoint-120000, the ZMQ server should be launched with
`EMBODIMENT_TAG=NEW_EMBODIMENT`; rerun closed-loop ZMQ after that setting before making a final SR claim.

후속 확인 (`outputs/eval/n16_zmq_new_emb_3ep_20260609_212237`): 같은
RoboCasa365 checkpoint-120000을 ZMQ server에서 `EMBODIMENT_TAG=NEW_EMBODIMENT`로
띄우면 `CloseFridge_PandaOmron_Env` 3ep/720은 `[False, False, True]` (`1/3`),
`OpenFridge_PandaOmron_Env` 3ep/720도 `[False, False, True]` (`1/3`)로 끝났다.
따라서 이전 N1.6 ZMQ `0/3`은 checkpoint 자체의 action 반환 불능이라기보다
server embodiment mismatch 영향으로 보는 것이 맞다. 다만 이 결과는 3-episode smoke라
task-level SR 기준선은 target task set으로 더 크게 다시 산출해야 한다.
