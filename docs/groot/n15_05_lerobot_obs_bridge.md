# GR00T N1.5 RoboCasa — Closed-loop Obs Bridge Spec (stage [2][3])

파이프라인 `stage [2] HTTP`·`stage [3] robocasa365 eval`의 obs 계약 명세와 수정 상태다.
전체 그림·상태는 [`n15_03`](n15_03_lerobot_robocasa365.md), serve 로딩은
[`n15_04`](n15_04_lerobot_serve_adapter.md).

## 요구 계약 (GrootPolicy가 받아야 하는 HTTP 입력)

`scripts/serve/lerobot_adapters/groot.py`의 `GrootPolicyAdapter`는 profile로부터 아래
feature spec을 만든다 (`build_groot_feature_specs`). 즉 serve가 GrootPolicy에 넘기는 키:

- images: `observation.images.00_side_0`, `01_side_1`, `02_wrist_0` (각 `[3,224,224]`)
- state: `observation.state` =
  `eef_pos_rel(3) ⊕ eef_quat_rel→rotation_6d(6) ⊕ gripper_qpos(2) ⊕
  base_position(3) ⊕ base_rotation→rotation_6d(6)` = **20D**, RoboCasa benchmark
  fork의 `PandaOmronDataConfig` 순서와 transform을 따른다.

`00_` prefix는 LeRobot 0.4.4 `GrootPackInputsStep`의 alphabetical image-key sort를
N1.5 data config video concat order(`side_0, side_1, wrist_0`)로 맞추기 위한 internal key다.
RoboCasa env → `/act` payload는 아래 semantic key/alias를 그대로 사용한다.

## 카메라 키 매핑 명세

| GrootPolicy 기대 | profile image | direct HTTP key | generic processor emit | groot-env processor emit |
|---|---|---|---|---|
| `00_side_0` (left view) | `side_0` | `observation.images.side_0` | `side_0` ✓ (3-cam only) | `left` ✓ (alias) |
| `01_side_1` (right view) | `side_1` | `observation.images.side_1` | `side_1` ✓ (3-cam only) | `right` ✓ (alias) |
| `02_wrist_0` (wrist) | `wrist_0` | `observation.images.wrist_0` | `wrist_0` ✓ (3-cam only) | `wrist` ✓ (alias) |

- generic `RoboCasaObsProcessor` (`src/processor/obs/robocasa.py`): **3-cam 모드
  (`static_cam2` 지정 시)에서만** `side_0/side_1/wrist_0` alias emit. 2-cam default는
  `static`/`wrist`만 emit → side 키 없음.
- groot-env `make_groot_robocasa_processors`: `schema.py::GROOT_ENV_VIDEO_TO_UNIFIED_CAM`
  이 `side_0→left, side_1→right, wrist_0→wrist`로 매핑 → HTTP 키가 `left/right/wrist`.

### 이전 serve remap 버그

`_build_remap_config` (`scripts/serve/lerobot.py`)는 들어오는 키를 unified
`static/wrist/wrist2`로 **가정**하고 GrootPolicy visual_keys(`side_0/side_1/wrist_0`)에
순서대로 매핑 → `{static→side_0, wrist→side_1, wrist2→wrist_0}`. 그 뒤
`_apply_input_remap`이 `batch["side_1"] = batch.pop("wrist")`를 실행 →

- **`wrist→side_1`이 올바른 `side_1`(right view)을 wrist 이미지로 덮어쓴다.** ✗
- `static`/`wrist2`는 GR00T 경로에 안 들어오므로 `side_0`/`wrist_0`는 remap으로 안 생긴다.

현재 `policy_type=groot`에서는 `GrootPolicyAdapter.build_remap_config()`가 아래 alias만
등록한다.

```text
left/static -> 00_side_0
right       -> 01_side_1
wrist       -> 02_wrist_0
```

`_apply_input_remap`은 destination key가 이미 있으면 alias 값으로 덮어쓰지 않는다. 따라서
generic 3-cam processor가 직접 내는 `wrist_0/side_0/side_1`와 groot-env processor가 내는
`wrist/left/right`를 모두 받을 수 있다. 내부 destination key는 N1.5 data config order를
보존한다.

## State 키 매핑 명세

| profile state key | dim | generic emit | groot-env emit | 현재 serve 결과 |
|---|---|---|---|---|
| `base_position` | 3 | `base_pos` ✓ (alias) | `base_position` ✓ | OK |
| `base_rotation` | 4→6 | `base_quat` ✓ (alias) | `base_rotation` ✓ | OK, native `quaternion`→`rotation_6d` |
| `eef_pos_rel` | 3 | `eef_pos` ✓ (alias) | `eef_pos_rel` ✓ | OK |
| `eef_quat_rel` | 4→6 | `eef_quat` ✓ (alias) | `eef_quat_rel` ✓ | OK, native `quaternion`→`rotation_6d` |
| `gripper_qpos` | 2 | `gripper_qpos` ✓ | `gripper_qpos` ✓ | OK |

generic processor는 `_rel`/`base_position`/`base_rotation` 키 자체가 없다. 현재
`policy_type=groot`에서는 `GrootPolicyAdapter.state_payload_keys()`가 아래 alias를 수용한다.

```text
eef_pos      -> eef_pos_rel
eef_quat     -> eef_quat_rel
base_pos     -> base_position
base_quat    -> base_rotation
gripper_qpos -> gripper_qpos
```

## 현재 결론

| 경로 | 카메라 | state |
|---|---|---|
| generic `make_robocasa_processors` | ✓ direct `side_0/side_1/wrist_0` 보존, alias clobber 방지 | ✓ alias로 20D 조립 |
| groot-env `make_groot_robocasa_processors` (`--use-groot-env`) | ✓ `left/right/wrist` alias 수용 | ✓ direct `_rel`/base keys 수용, rotation_6d 변환 |

Unit contract와 실제 `/act` payload probe는 통과했다. 2026-06-09에는 RoboCasa
benchmark fork의 official `robocasa/<Task>` split 경로도
`scripts/eval/lerobot_groot_n15_official_robocasa_eval.py`로 붙였다. 이후
OpenFridge target seed-0 LeRobot HTTP smoke는 성공했고, SR 외 내부값 검증은
[`n15_08_lerobot_internal_parity.md`](n15_08_lerobot_internal_parity.md)로 분리했다.
따라서 이 문서의 결론 범위는 camera/state bridge 계약까지이며, model/action parity
판단은 n15_08을 기준으로 한다.

## N1.6 YAML 비교 결론

N1.6 RoboCasa365 profile은 N1.5 LeRobot profile의 source of truth가 아니다. 로컬 확인 기준:

- N1.5 data config (`RobocasaPandaOmron10TaskDataConfig`):
  - video: `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`
  - state: `base_position`, `base_rotation`, `end_effector_position_relative`,
    `end_effector_rotation_relative`, `gripper_qpos`
  - action: `base_motion`, `control_mode`, `end_effector_position`,
    `end_effector_rotation`, `gripper_close`
- N1.6 processor metadata:
  - `robocasa_panda_omron`: `res256_image_side_0`, `res256_image_side_1`,
    `res256_image_wrist_0`, eef-first state/action
  - `new_embodiment`: `robot0_eye_in_hand`, `robot0_agentview_left`,
    `robot0_agentview_right`, eef-first state/action

즉 N1.6 쪽 camera alias와 N1.5 data config의 camera order는 결과적으로 같지만,
N1.6 action/state order를 N1.5 profile에 옮기면 N1.5 checkpoint statistics와 action
slicing이 깨진다. 이전 `side_0, side_1, wrist_0` + 224 crop probe/rollout은 raw-language
및 final profile 정렬 이전의 중간 실험이므로, 현재 profile은 N1.5 data config 기준으로 다시
검증한다.

## 이미 풀린 선례 (재사용 대상)

N1.6 GR00T HTTP serve (`scripts/serve/groot.py`, :8500)는 이 문제를 이미 해결했다. 코드 확인:

- `src/policies/groot/schema.py`:
  - `observation.images.side_0` (+ `static`/`left` alias) → `video.res256_image_side_0`
  - `observation.state.eef_pos_rel` → `state.end_effector_position_relative`
  - `observation.state.base_position` → `state.base_position`
- `service.build_groot_obs`가 이를 소비.

새 lerobot serve는 동일한 alias 원칙을 `GrootPolicyAdapter` 안에서만 적용한다. shared
`GROOT_ENV_VIDEO_TO_UNIFIED_CAM`은 건드리지 않는다.

## 적용한 수정안

`scripts/serve/lerobot_adapters/groot.py`의 `GrootPolicyAdapter`가 RoboCasa 키를 직접
수용한다. generic remap(`static/wrist/wrist2` 순서 매핑)은 `PiPolicyAdapter`에만 유지한다.
schema 공유 키(`GROOT_ENV_VIDEO_TO_UNIFIED_CAM`)는 변경하지 않았다.

## Stage [2][3] 검증

- unit: ✅ `tests/test_serve_lerobot.py::TestGrootRobocasaObsBridge`
  (`20D` state, PyTorch3D/native `wxyz` quaternion convention)
- official env helper: ✅ `tests/test_lerobot_groot_n15_official_eval.py`
- deterministic probe: ✅ historical one-off artifact
  `outputs/debug/lerobot_groot_n15_payload_probe_seed100010_infer777_final_profile_repeats3`.
  `semantic_current`와 `direct_wrist_left_right` action identical, repeat std 0.
- data-config profile deterministic probe: ✅ historical one-off artifact
  `outputs/debug/lerobot_groot_n15_payload_probe_seed100002_infer777_dataconfig_repeats3`.
  `semantic_current`와 `direct_left_right_wrist` action identical, repeat std 0. 이 profile에서는
  `RobocasaPandaOmron10TaskDataConfig`의 `left,right,wrist` order와 server-side crop 0.95 +
  224 resize가 적용된다.
- closed-loop smoke:
  - image-order post-fix 3 rollouts: SR 0/3,
    `outputs/eval/robocasa/lerobot_groot_n15/close_blender_lid_3rollouts_seed100000_imageorder`
  - raw-language 1 rollout: SR 0/1,
    `outputs/debug/lerobot_groot_n15_rawlang_1rollout_seed100010`
  - data-config profile 1 rollout on N1.6-ZMQ-success seed: SR 0/1,
    `outputs/debug/lerobot_groot_n15_dataconfig_1rollout_seed100002`
  - historical pre-repair official RoboCasa OpenFridge target, LeRobot HTTP: SR 0/1,
    `outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_wxyz`.
    같은 checkpoint의 native N1.5 ZMQ official env smoke는 SR 1/1이므로 이 결과는
    LeRobot wrapper parity issue로 추적했다. 현재 retained internal parity와 seed-0
    closed-loop 상태는 [`n15_08`](n15_08_lerobot_internal_parity.md)과
    [`n15_07`](n15_07_native_zmq_openfridge.md)을 기준으로 한다.

## 과거 Native ZMQ vs LeRobot HTTP 비교 결과

2026-06-09에 같은 official RoboCasa `OpenFridge`, `split=target`, `seed=0`
첫 observation을 native N1.5 ZMQ 서버(`:5558`)와 LeRobot HTTP 서버(`:8400`)에
동시에 넣어 action chunk를 비교했다. 이 one-off probe script는 cleanup에서 제거했고,
현재 retained parity script는 checkpoint-load verifier 하나만 둔다.

결과:

```text
flat_l2_all: 3.741768
flat_max_abs: 0.974393
first_step_l2: 0.657030
action.gripper_close first native: [0.0]
action.gripper_close first lerobot: [0.653539896]
action.control_mode first native: [0.0]
action.control_mode first lerobot: [0.015283823]
```

이 run에서 확인한 parity gap:

- Native config는 key-specific `StateActionTransform` mode를 사용한다. PandaOmron 기준
  `action.gripper_close`와 `action.control_mode`는 `binary`다. 반면 LeRobot
  `GrootActionUnpackUnnormalizeStep`는 12D action vector 전체에 flat min-max inverse를
  적용했다.
- Native image path는
  `VideoToTensor -> eval CenterCrop(0.95) -> Resize(224) -> VideoToNumpy -> GR00TTransform`.
  LeRobot HTTP server도 payload parse 시점에 같은 0.95 center crop + 224 resize를
  적용한다. 다만 LeRobot `GrootEagleCollateStep`는 Eagle processor에 명시적
  `images_kwargs`를 전달한다는 점에서 native collate와 달랐다.
- Native language text는 raw task text다. LeRobot upstream은 text를 `str([lang])` 형태로
  포맷한다. 이 repo는 `model_specific.raw_language: true`를 켜서
  `apply_chat_template` 전에 해당 표현을 unwrap한다.
- `new_embodiment` id, action horizon, denoising step은 맞았다:
  `new_embodiment=31`, `action_horizon=16`, `num_inference_timesteps=4`.

후속 수정:

- `model_specific.native_action_unapply: true`일 때 `GrootPolicyAdapter`가 LeRobot GR00T
  postprocessor를 감싼다. 이 wrapper는 continuous slice에는 LeRobot의 기존 flat min-max inverse를
  유지하고, `gripper_close` / `control_mode`만 raw model output 기준 native N1.5
  binary inverse(`x > 0.5`)로 덮어쓴다. 이 처리는 repo-local이며 `lerobot`
  submodule은 수정하지 않는다.
- stale LeRobot container를 재시작한 뒤의 과거 비교 결과:

  ```text
  flat_l2_all: 1.349810
  flat_max_abs: 0.458418
  first_step_l2: 0.174521
  action.gripper_close max_abs: 0.0
  action.control_mode max_abs: 0.0
  ```

- 당시 action fix 이후 남은 유력한 parity gap은 Eagle collation이었다. LeRobot은 여전히
  명시적 `images_kwargs`를 전달하지만 native N1.5 `GR00TTransform.collate`는 그렇지 않다.
  현재 retained verifier는 이 과거 action-level comparison을 더 이상 다루지 않는다.
- action fix 이후의 과거 closed-loop smoke는 `OpenFridge target seed=0`에서 여전히
  실패했다(`success_rate=0.0`, `episode_lengths=[200]`, video:
  `/temporal_vla/outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_after_native_action_unapply/videos/79e7da83-c4ad-43ad-a2ed-9f9c3e1e6624_success0.mp4`).
  Binary action fix는 확인된 parity repair 하나로만 보고, full policy equivalence로 보지
  않는다. 이후 OpenFridge target seed-0 LeRobot HTTP smoke는 성공했다. 자세한 내용은
  [`n15_07`](n15_07_native_zmq_openfridge.md)을 본다. 현재 retained SR-independent
  checkpoint-load parity는 [`n15_08`](n15_08_lerobot_internal_parity.md)에 기록한다.
