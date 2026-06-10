# GR00T N1.5 RoboCasa — LeRobot Serve Adapter Spec (stage [1])

파이프라인 stage [1]의 명세다. raw Isaac-GR00T N1.5 RoboCasa365 checkpoint를 LeRobot
`GrootPolicy`로 로딩해 통일 HTTP `/act`를 노출한다. 전체 그림·상태는
[`n15_03`](n15_03_lerobot_robocasa365.md).

## 체크포인트 형식

RoboCasa365 checkpoint는 LeRobot `Policy.save_pretrained()` 산출물이 아니다. HF repo
subfolder 안의 raw Isaac-GR00T N1.5 checkpoint다.

```text
robocasa/robocasa365_checkpoints
└── gr00t_n1-5/multitask_learning/checkpoint-120000/
    ├── config.json
    ├── experiment_cfg/
    ├── model-00001-of-00002.safetensors
    ├── model-00002-of-00002.safetensors
    ├── model.safetensors.index.json
    ├── optimizer.pt          ← 다운로드 안 함
    ├── rng_state.pth         ← 다운로드 안 함
    ├── scheduler.pt          ← 다운로드 안 함
    └── trainer_state.json    ← 다운로드 안 함
```

Adapter는 inference에 필요한 파일만 받는다 (`groot_snapshot_allow_patterns`):

```text
{subfolder}/config.json
{subfolder}/model.safetensors.index.json
{subfolder}/experiment_cfg/**
{subfolder}/model-*.safetensors
```

## 프로파일 필드 명세

`configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml`. 스키마 단일 출처는
`configs/checkpoints/README.md`. 이 checkpoint의 필드별 의미:

| 필드 | 값 | 의미 |
|---|---|---|
| `base_model` | `lerobot` | serve 스크립트 라우팅 (`scripts/serve/lerobot.py`) |
| `checkpoint_source.type` | `hf_repo` | HF subfolder snapshot 경로 (local dir 아님) |
| `checkpoint_source.id` | `robocasa/robocasa365_checkpoints` | HF repo id |
| `model_specific.policy_type` | `groot` | `GrootPolicyAdapter` 선택 (명시 registry) |
| `model_specific.checkpoint_subfolder` | `gr00t_n1-5/multitask_learning/checkpoint-120000` | snapshot 내 실제 ckpt 위치 |
| `model_specific.embodiment_tag` | `new_embodiment` | GrootConfig embodiment metadata 선택 |
| `model_specific.chunk_size` | `16` | GR00T action chunk 길이 |
| `model_specific.use_bf16` | `true` | bf16 추론 |
| `model_specific.raw_language` | `true` | LeRobot Eagle processor의 `str([lang])` wrapping을 우회하고 N1.5 원본 GR00T처럼 raw instruction을 tokenizer에 전달 |
| `model_specific.native_action_unapply` | `true` | LeRobot flat min-max action inverse 뒤에 N1.5 native `binary` action inverse를 gripper/control_mode slice에 적용 |
| `action_type` | `absolute` | checkpoint metadata의 action modalities가 absolute |
| `action_layout` | `eef_pos(3) eef_axisangle(3) gripper(1) base_motion(4) control_mode(1)` | RoboCasa benchmark fork `PandaOmronDataConfig` action order |
| `rotation_encoding` | `axisangle` | rotation sub-key가 `action.eef_axisangle` |
| `gripper_encoding` | `range [-1,1], binarize false, sign_flip false, threshold 0.0` | gripper 후처리 규약 |
| `normalization.scheme` | `min_max` | adapter가 checkpoint `experiment_cfg/metadata.json` statistics를 flat `observation.state`/`action` stats로 조립 |
| `observation_requirements.images` | `side_0, side_1, wrist_0` | N1.5 RoboCasa data config `video_concat_order` (`left, right, wrist`) |
| `observation_requirements.state` | `eef_pos_rel(3) eef_quat_rel→rotation_6d(6) gripper_qpos(2) base_position(3) base_rotation→rotation_6d(6)` | official N1.5 PandaOmron transform, 합 20D |
| `n_action_steps` | `16` | action queue 길이 (아래) |
| `image_preprocess` | `resolution 224, rotate_180 false, center_crop true, center_crop_scale 0.95` | N1.5 data config `VideoCrop(scale=0.95)` + `VideoResize(224)` |
| `emits_subkeys` | `action.eef_pos, .eef_axisangle, .gripper, .base_motion, .control_mode` | `/act` 응답 sub-key |

### `n_action_steps = 16`인 이유

LeRobot `GrootPolicy` 내부 action queue 길이와 GR00T N1.5 action horizon을 맞추기 위한
값. 8로 낮추면 `deque(maxlen=n_action_steps)`가 16-step chunk 앞부분을 버릴 수 있어서
HTTP profile은 16으로 둔다.

`/act`는 일반 LeRobot serving semantics를 유지해 queue에서 1-step action을 반환한다.
반대로 GR00T N1.5 `/act_with_features` collect path는 N1.6 SAFE collection과 같은
execution 단위를 맞추기 위해 hook 아래에서 `predict_action_chunk`를 직접 호출하고
`[H,D]` action sub-key를 반환한다.

### data config order / normalization

RoboCasa365 N1.5의 inference feature order는 이 repo의
`configs/policies/robocasa_n15_panda_omron_data_config.py`가 단일 출처다. 해당 config의
`ConcatTransform`은 video/state/action을 아래 순서로 묶는다.

```text
video  = robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand
state  = end_effector_position_relative,
         end_effector_rotation_relative -> rotation_6d,
         gripper_qpos,
         base_position,
         base_rotation -> rotation_6d
action = end_effector_position,
         end_effector_rotation,
         gripper_close,
         base_motion,
         control_mode
```

`experiment_cfg/metadata.json`의 statistics는 state/action normalization source로 쓴다.
LeRobot `GrootPolicy` processor는 flat `observation.state` / `action` stats를 받아
min-max normalize/unnormalize한다. `GrootPolicyAdapter`는 raw checkpoint의 nested metadata
stats를 profile order대로 flatten해서 넘긴다. 이걸 빼면 raw base pose/state가 normalized
training space와 맞지 않는다.

N1.6 RoboCasa365 profile(`groot__robocasa365_ckpt120000.yaml`)과 이 N1.5 profile은
호환 profile이 아니다. N1.6 user-facing profile은 7D relative EEF action 중심으로 쓰지만,
N1.5 raw checkpoint metadata contains all nested stats, but inference order must follow
the RoboCasa benchmark fork's `panda_omron` data config. Earlier base-first/raw-quaternion
experiments are wiring artifacts and should not be treated as policy quality results.

LeRobot 0.4.4 `GrootPackInputsStep`는 `observation.images.*` key를 내부에서
alphabetical sort한다. 따라서 adapter는 submodule을 수정하지 않고 internal visual key를
`00_side_0`, `01_side_1`, `02_wrist_0`로 생성한다. HTTP 입력 alias는 여전히
`left/right/wrist` 또는 `side_0/side_1/wrist_0`를 받지만, packer가 보는 순서는 N1.5 data
config video concat order와 맞는다.

## 구현 구조

`scripts/serve/lerobot.py`는 공통 FastAPI endpoint를 유지하고 policy별 차이는
`scripts/serve/lerobot_adapters/` package로 분리한다.

```text
HTTP payload parsing
  -> policy adapter registry  (make_policy_adapter)
     -> PiPolicyAdapter    : pi0/pi05/pi0_fast 등 기존 lerobot config 경로
     -> GrootPolicyAdapter : HF subfolder snapshot + GrootConfig feature shape 생성
  -> common /act sub-key emit  (_emit_subkeys, profile.action_layout 기반)
```

`GrootPolicyAdapter`는 process-local patch만 적용한다 (`patch_groot_runtime`,
`scripts/safe/groot_n15/robocasa/utils/runtime.py`) — `lerobot/` submodule 미수정:

- torch 2.10 Beta meta-device validation 회피 (`flow_matching.Beta(validate_args=False)`)
- transformers 5용 `GR00TN15.all_tied_weights_keys = {}` 보강
- Eagle processor `return_tensors="pt"` 기본값 보강
- Eagle processor language wrapping 우회 (`TEMPORAL_VLA_GROOT_RAW_LANGUAGE=1`):
  LeRobot 0.4.4 `GrootEagleEncodeStep`는 instruction을 `str([lang])`로 감싼 뒤 tokenizer에
  넣는다. 원본 Isaac-GR00T N1.5 transform은 raw `lang`을 직접 쓴다. Adapter profile의
  `model_specific.raw_language: true`는 이 차이를 process-local patch로 맞춘다.
- GR00T action postprocess 보강 (`model_specific.native_action_unapply: true`):
  LeRobot 0.4.4 `GrootActionUnpackUnnormalizeStep`는 12D action 전체를 flat min-max inverse로
  푼다. 원본 N1.5 PandaOmron config는 `gripper_close`와 `control_mode`만 `binary` inverse
  (`x > 0.5`)를 쓴다. Adapter는 LeRobot postprocessor를 감싼 뒤 해당 slice만 native 규칙으로
  덮어쓴다.

GR00T policy type은 generic LIBERO/pi-style camera remap을 우회하고 RoboCasa 3-cam/
relative-state alias를 직접 수용한다. Unit contract는 [`n15_05`](n15_05_lerobot_obs_bridge.md)에
정리한다. 이 문서의 주 범위는 stage [1] serve **로딩**이다.

## Serve 실행

```bash
docker compose run --rm --no-deps -T lerobot \
  python /temporal_vla/scripts/serve/lerobot.py \
    --profile /temporal_vla/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
    --host 0.0.0.0 \
    --port 8400
```

Smoke 확인 (`/health`→`/reset`→`/act` 왕복):

```bash
docker compose run --rm --no-deps -T lerobot \
  python /temporal_vla/scripts/utils/smoke_test_serve.py \
  --profile /temporal_vla/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
  --url http://127.0.0.1:8400 \
  --max-wait 1200
```

2026-06-09 result:

```text
server: Application startup complete; /health 200, /reset 200, /act 200
smoke: exit code 0
```

이 workstation의 GPU는 RTX 4070(sm89)이고, 현재 LeRobot image는 `FLASH_ATTN_CUDA_ARCHS=120`
으로 flash-attn을 빌드한다. `scripts/safe/groot_n15/robocasa/utils/runtime.py`는 이 mismatch를
감지하면 Eagle vision attention을 process-local eager attention으로 바꾼다. sm120 GPU에서는
flash-attn 경로를 유지한다.

## 캐시 경로

`docs/cache_paths.md`의 cache 규칙을 따른다.

- 컨테이너: `VLA_CACHE_ROOT=/cache`
- HF cache: `/cache/huggingface` because the `lerobot` compose service mounts
  repo-local `./data` to `/cache`.
- helper: `scripts/safe/groot_n15/robocasa/utils/runtime.py`

Runtime compatibility helper는 `scripts/safe/groot_n15/robocasa/utils/runtime.py` 하나만 source of
truth로 둔다. `GrootPolicyAdapter`는 path-based loader로 이 helper를 가져오고, parity/eval
diagnostics는 safe-tree utils path를 직접 추가해서 같은 helper를 가져온다.
Cleanup status는 [`n15_03`](n15_03_lerobot_robocasa365.md)에 기록한다.

## Stage [1] 검증

1. `tests/test_serve_lerobot.py` — adapter helper, profile-driven state/action shape,
   `/act` endpoint contract. ✅
2. `checkpoint_profile.load_profile(...)` — YAML schema 및 action dim. ✅
3. `TestGrootRobocasaObsBridge` — GR00T RoboCasa obs bridge unit contract. ✅
4. Docker HTTP smoke (위 명령) — ✅ `/act 200 OK`.
5. metadata order / stats flatten unit — ✅
6. deterministic `/act` probe with `inference_seed` — ✅ same payload/repeat action is stable.
7. closed-loop smoke — ✅ infra path runs, but policy behavior is still wrong; see
   [`n15_03`](n15_03_lerobot_robocasa365.md) and [`n15_05`](n15_05_lerobot_obs_bridge.md).
