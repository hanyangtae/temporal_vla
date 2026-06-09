# GR00T N1.5 RoboCasa — LeRobot RoboCasa365 Pipeline Overview & Status

`n15_02_eval.md`의 후속 wiring 문서다. 기존 Isaac-GR00T N1.5 base/ZMQ 평가 경로는
그대로 두고, RoboCasa365 checkpoint를 LeRobot framework(serve·dataset·시각화) 위에서
이 repo의 통일 HTTP API로 돌리는 end-to-end 파이프라인의 **map + 현재 상태**다.

상세 명세는 stage별 child 문서로 분리한다 (아래 "문서 구성"). 이 문서는 전체 그림과
검증 상태·향후 계획의 단일 출처만 유지한다.

## 목표 파이프라인

```text
 [1] lerobot serve              [3] robocasa365 컨테이너          [4] LeRobot analysis UI
     (GR00T N1.5 ckpt)            closed-loop eval                  rerun 뷰어
     scripts/serve/lerobot.py     robocasa_eval.py                  lerobot-dataset-viz
            │                            │   ▲                            ▲
            └──────[2] HTTP /act ───────▶│   │                            │
                   ◀──── action.* ───────┘   └──── rollout ──────────────-┘
                                              (LeRobotDataset v3.0 기록)
```

- **[1] serve** — profile 기반 `GrootPolicy` 로딩. 명세: [`n15_04`](n15_04_lerobot_serve_adapter.md).
- **[2] HTTP** — 통일 API ([`../01_serving_interface.md`](../01_serving_interface.md)).
  `VLAClient`/processor가 `/act` 호출, server가 sub-keyed `action.*` 반환.
- **[3] robocasa365 eval** — official `robocasa/<Task>` split closed-loop rollout.
  stage 1↔3 obs 계약은 serve unit test와 OpenFridge smoke로 확인했다. 명세·gap:
  [`n15_05`](n15_05_lerobot_obs_bridge.md), 내부 parity:
  [`n15_08`](n15_08_lerobot_internal_parity.md).
- **[4] analysis UI** — rollout을 `LeRobotDataset`(v3.0)으로 기록 후 rerun 시각화.
  현재는 writer가 없으므로 TODO로 남긴다. 필요한 최소 작업은 `run_vla_rollouts_groot`
  계열에 `--record-lerobot-dataset`을 추가하고 action/state/image/timestamp를
  LeRobotDataset feature schema로 저장하는 것이다.

## 문서 구성

| 문서 | 범위 | 성격 |
|---|---|---|
| **n15_03** (이 문서) | 파이프라인 map, 검증 상태, 향후 계획 | living status |
| [`n15_04`](n15_04_lerobot_serve_adapter.md) | stage [1] serve adapter: checkpoint 형식, profile 필드 명세, serve/smoke, 구현 구조 | stable spec |
| [`n15_05`](n15_05_lerobot_obs_bridge.md) | stage [2][3] obs 계약: 카메라/state 키 매핑 명세, gap 분석, 수정안 | spec + gap |
| [`n15_07`](n15_07_native_zmq_openfridge.md) | native Isaac-GR00T N1.5 ZMQ vs LeRobot HTTP behavior 비교 | evidence |
| [`n15_08`](n15_08_lerobot_internal_parity.md) | SR 외 checkpoint-load 검증과 historical internal evidence | evidence |

## 경계

- 수정하지 않는 것: `lerobot/` submodule, 기존 N1.5 ZMQ server/eval runbook.
- 새로 추가하는 것: `scripts/serve/lerobot.py`의 공통 HTTP serve 경로와
  `scripts/serve/lerobot_adapters/`의 policy adapter registry,
  `configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml`,
  repo-local runtime helper `scripts/safe/groot_n15/robocasa/utils/runtime.py`,
  (예정) LeRobotDataset recorder.
- 통신 방식: LeRobot native eval이 아니라 project HTTP framework다. RoboCasa 쪽은
  `VLAClient`/processor가 `/act`를 호출하고, server는 sub-keyed `action.*`를 반환한다.

## 검증 상태 (2026-06-09)

인프라가 어디까지 갖춰졌는지의 **단일 출처**. ✅=코드 확인, ⏳=미실행, ❌=경로 없음, ❓=미확인.

| stage | 항목 | 상태 | 근거 / 상세 |
|---|---|---|---|
| [1] | Profile 스키마 / action dim (12D) | ✅ | `tests/test_serve_lerobot.py::TestGrootAdapterSpecs`, `load_profile`. [n15_04](n15_04_lerobot_serve_adapter.md) |
| [1] | HF subfolder snapshot allow-pattern (inference 파일만) | ✅ | 위 테스트 + `groot_snapshot_allow_patterns` |
| [1] | `GrootPolicyAdapter` feature spec (3-view, state 20D, action 12D) | ✅ | `build_groot_feature_specs` 단위 테스트. Internal image key `00_side_0/01_side_1/02_wrist_0`로 LeRobot sorted pack order를 N1.5 RoboCasa data config video order에 맞춤. State는 official PandaOmron order + quaternion→rotation_6d |
| [2][3] | GR00T RoboCasa obs bridge key contract | ✅ | `TestGrootRobocasaObsBridge`: `wrist_0/side_0/side_1` direct + `wrist/left/right` alias, 20D state alias |
| [1] | Docker HTTP smoke (`/health`→`/reset`→`/act`) | ✅ | `docker compose run ... lerobot` server + `smoke_test_serve.py`; `/act 200 OK` |
| [1] | Checkpoint metadata order / stats flatten | ✅ | profile state/action order를 `experiment_cfg/metadata.json`에 맞추고 metadata stats를 flat LeRobot stats로 변환. [n15_04](n15_04_lerobot_serve_adapter.md) |
| [1][2] | deterministic `/act` probe | ✅ | `inference_seed`를 server가 적용/echo. 같은 payload repeat action stable. 과거 one-off probe artifact: `outputs/debug/lerobot_groot_n15_payload_probe_seed100010_infer777_final_profile_repeats3` |
| [1][2] | raw-language runtime patch | ✅ / ⚠️ | LeRobot `str([lang])` wrapping을 우회하면 action scale이 크게 줄어듦. baseline `semantic_current` L2 0.963 → raw-language probe L2 0.145. 단, final restarted profile에서는 L2 0.529라 추가 parity 확인 필요 |
| [2][3] | RoboCasa → :8400 closed-loop | ✅ / ⚠️ | 과거 `CloseBlenderLid` probes는 실패했지만, official `OpenFridge target seed=0` LeRobot HTTP smoke는 성공(`success_rate=1.0`, first success step 172). 산출물: `outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_seed0_success_probe`. 단일-task smoke이므로 전체 SR claim은 아님. [n15_07](n15_07_native_zmq_openfridge.md) |
| [2][3] | N1.5 data-config camera/crop/internal evidence | ✅ / ⚠️ | `RobocasaPandaOmron10TaskDataConfig` 기준 `side_0,side_1,wrist_0` + crop 0.95 + 224 resize를 profile에 반영. retained verifier는 checkpoint-load 764 tensors checked이고, input/action boundary는 historical evidence로만 남긴다. Activation bit parity는 미확립. [n15_08](n15_08_lerobot_internal_parity.md) |
| [3→4] | rollout → LeRobotDataset recorder | ❌ | writer 부재 (reader/adapter만 존재). stage 4 유일 신규 piece |
| [4] | `lerobot-dataset-viz` (rerun) 시각화 | ✅도구 / ⏳입력 | lerobot 제공, recorder 입력 dataset 의존 |

## 향후 계획

전체 파이프라인 동작까지 남은 인프라 작업. 위→아래 순서, **1번이 전체의 전제**다.

1. **obs bridge 수정** — ✅ unit/runtime probe fixed. lerobot serve의 `GrootPolicyAdapter`가
   RoboCasa 키를 직접 수용(generic remap 우회). 상세·수정안: [`n15_05`](n15_05_lerobot_obs_bridge.md).
2. **Docker HTTP smoke** — ✅ `/health`→`/reset`→`/act` 통과. RTX 4070(sm89)에서는
   120-only flash-attn wheel 때문에 Eagle attention eager fallback이 적용된다.
3. **post-fix closed-loop smoke** — ✅ OpenFridge target seed-0 success 확인. 전체 task SR이
   아니라 wiring smoke로만 취급한다.
4. **LeRobot N1.5 checkpoint-load verifier** — ✅ retained parity script는
   [`n15_08`](n15_08_lerobot_internal_parity.md)에 기록. Activation bit parity와 closed-loop
   per-step trace parity는 남은 진단 대상이지만, 현재 별도 scripts는 두지 않는다.
5. **N1.6 ZMQ baseline 대비 action/SR parity** — N1.5 내부 parity가 더 안정화된 뒤에 비교한다.
   N1.6 YAML/profile은 N1.5 profile source of truth가 아니다.
6. **[stage 4] LeRobotDataset recorder 구현** — `run_vla_rollouts_groot`에
   `--record-lerobot-dataset` 추가. (1~3 이후 착수 가능하나, policy behavior 검증과
   별도 작업이다.)
7. **[stage 4] `lerobot-dataset-viz`로 시각화 확인** — 기록한 dataset 검수.

stage 1~3으로 "serve → HTTP → robocasa365" 연결은 확인됐다. stage 4가 끝나야 "serve → HTTP → robocasa365 → analysis UI" 전체
파이프라인이 동작한다고 말할 수 있다.

## 정리 상태 (closed-loop와 독립)

- **helper 위치 정리**: runtime helper는
  `scripts/safe/groot_n15/robocasa/utils/runtime.py`만 source of truth로 둔다.
  `GrootPolicyAdapter`는 이 파일을 path-based loader로 가져온다.
- **one-off probe 정리**: payload variant exploration script는 retained diagnostic surface에서
  제거했다. 재현 가능한 내부 검증 축은 [`n15_08`](n15_08_lerobot_internal_parity.md)의 parity
  scripts로 둔다.
- **중복 smoke 정리**: 단순 checkpoint load smoke script는 제거했다. checkpoint load 검증은
  [`n15_08`](n15_08_lerobot_internal_parity.md)의 internal parity verifier가 담당한다.
