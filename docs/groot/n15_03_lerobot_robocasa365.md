# GR00T N1.5 RoboCasa — LeRobot RoboCasa365 Pipeline

`n15_02_eval.md`의 후속 wiring 문서다. 기존 Isaac-GR00T N1.5 base/ZMQ 평가 경로는
그대로 두고, RoboCasa365 checkpoint를 LeRobot framework(serve·dataset·시각화) 위에서
이 repo의 통일 HTTP API로 돌리는 end-to-end 파이프라인을 한 문서로 통합한다.

원래 stage별 child 문서(serve adapter / obs bridge / native ZMQ smoke / internal parity)는
이 문서의 섹션으로 합쳤다. 이 문서가 LeRobot-N1.5 RoboCasa365 wiring의 단일 출처다.

> 관련 문서 (N1.5 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n15_01_finetune.md)
> - [02 Evaluation](n15_02_eval.md)
> - **03 LeRobot RoboCasa365 Pipeline (이 문서)**

이 문서의 섹션:

- [Pipeline overview + status](#pipeline-overview--status)
- [Serve adapter spec](#serve-adapter-spec-stage-1)
- [Obs bridge mapping](#obs-bridge-mapping-stage-23)
- [Native ZMQ baseline (OpenFridge smoke)](#native-zmq-baseline-openfridge-smoke)
- [Internal checkpoint-load parity](#internal-checkpoint-load-parity)

---

## Pipeline Overview + Status

### 목표 파이프라인

```text
 [1] lerobot serve              [3] robocasa365 컨테이너          [4] LeRobot analysis UI
     (GR00T N1.5 ckpt)            closed-loop eval                  rerun 뷰어
     scripts/serve/lerobot.py     robocasa_eval.py                  lerobot-dataset-viz
            │                            │   ▲                            ▲
            └──────[2] HTTP /act ───────▶│   │                            │
                   ◀──── action.* ───────┘   └──── rollout ──────────────-┘
                                              (LeRobotDataset v3.0 기록)
```

- **[1] serve** — profile 기반 `GrootPolicy` 로딩. 명세: [Serve adapter spec](#serve-adapter-spec-stage-1).
- **[2] HTTP** — 통일 API ([`../01_serving_interface.md`](../01_serving_interface.md)).
  `VLAClient`/processor가 `/act` 호출, server가 sub-keyed `action.*` 반환.
- **[3] robocasa365 eval** — official `robocasa/<Task>` split closed-loop rollout.
  stage 1↔3 obs 계약은 serve unit test와 OpenFridge smoke로 확인했다. 명세·gap:
  [Obs bridge mapping](#obs-bridge-mapping-stage-23), 내부 parity:
  [Internal checkpoint-load parity](#internal-checkpoint-load-parity).
- **[4] analysis UI** — rollout을 `LeRobotDataset`(v3.0)으로 기록 후 rerun 시각화.
  현재는 writer가 없으므로 TODO로 남긴다. 필요한 최소 작업은 `run_vla_rollouts_groot`
  계열에 `--record-lerobot-dataset`을 추가하고 action/state/image/timestamp를
  LeRobotDataset feature schema로 저장하는 것이다.

### 경계

- 수정하지 않는 것: `lerobot/` submodule, 기존 N1.5 ZMQ server/eval runbook.
- 새로 추가하는 것: `scripts/serve/lerobot.py`의 공통 HTTP serve 경로와
  `scripts/serve/lerobot_adapters/`의 policy adapter registry,
  `configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml`,
  repo-local runtime helper `scripts/safe/groot_n15/robocasa/utils/runtime.py`,
  (예정) LeRobotDataset recorder.
- 통신 방식: LeRobot native eval이 아니라 project HTTP framework다. RoboCasa 쪽은
  `VLAClient`/processor가 `/act`를 호출하고, server는 sub-keyed `action.*`를 반환한다.

### 검증 상태 (2026-06-09)

인프라가 어디까지 갖춰졌는지의 **단일 출처**. ✅=코드 확인, ⏳=미실행, ❌=경로 없음, ❓=미확인.

| stage | 항목 | 상태 | 근거 / 상세 |
|---|---|---|---|
| [1] | Profile 스키마 / action dim (12D) | ✅ | `tests/test_serve_lerobot.py::TestGrootAdapterSpecs`, `load_profile`. [Serve adapter spec](#serve-adapter-spec-stage-1) |
| [1] | HF subfolder snapshot allow-pattern (inference 파일만) | ✅ | 위 테스트 + `groot_snapshot_allow_patterns` |
| [1] | `GrootPolicyAdapter` feature spec (3-view, state 20D, action 12D) | ✅ | `build_groot_feature_specs` 단위 테스트. Internal image key `00_side_0/01_side_1/02_wrist_0`로 LeRobot sorted pack order를 N1.5 RoboCasa data config video order에 맞춤. State는 official PandaOmron order + quaternion→rotation_6d |
| [2][3] | GR00T RoboCasa obs bridge key contract | ✅ | `TestGrootRobocasaObsBridge`: `wrist_0/side_0/side_1` direct + `wrist/left/right` alias, 20D state alias |
| [1] | Docker HTTP smoke (`/health`→`/reset`→`/act`) | ✅ | `docker compose run ... lerobot` server + `smoke_test_serve.py`; `/act 200 OK` |
| [1] | Checkpoint metadata order / stats flatten | ✅ | profile state/action order를 `experiment_cfg/metadata.json`에 맞추고 metadata stats를 flat LeRobot stats로 변환. [Serve adapter spec](#serve-adapter-spec-stage-1) |
| [1][2] | deterministic `/act` probe | ✅ | `inference_seed`를 server가 적용/echo. 같은 payload repeat action stable. 과거 one-off probe artifact: `outputs/debug/lerobot_groot_n15_payload_probe_seed100010_infer777_final_profile_repeats3` |
| [1][2] | raw-language runtime patch | ✅ / ⚠️ | LeRobot `str([lang])` wrapping을 우회하면 action scale이 크게 줄어듦. baseline `semantic_current` L2 0.963 → raw-language probe L2 0.145. 단, final restarted profile에서는 L2 0.529라 추가 parity 확인 필요 |
| [2][3] | RoboCasa → :8400 closed-loop | ✅ / ⚠️ | 과거 `CloseBlenderLid` probes는 실패했지만, official `OpenFridge target seed=0` LeRobot HTTP smoke는 성공(`success_rate=1.0`, first success step 172). 산출물: `outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_seed0_success_probe`. 단일-task smoke이므로 전체 SR claim은 아님. [Native ZMQ baseline](#native-zmq-baseline-openfridge-smoke) |
| [2][3] | N1.5 data-config camera/crop/internal evidence | ✅ / ⚠️ | `RobocasaPandaOmron10TaskDataConfig` 기준 `side_0,side_1,wrist_0` + crop 0.95 + 224 resize를 profile에 반영. retained verifier는 checkpoint-load 764 tensors checked이고, input/action boundary는 historical evidence로만 남긴다. Activation bit parity는 미확립. [Internal checkpoint-load parity](#internal-checkpoint-load-parity) |
| [3→4] | rollout → LeRobotDataset recorder | ❌ | writer 부재 (reader/adapter만 존재). stage 4 유일 신규 piece |
| [4] | `lerobot-dataset-viz` (rerun) 시각화 | ✅도구 / ⏳입력 | lerobot 제공, recorder 입력 dataset 의존 |

### 향후 계획

전체 파이프라인 동작까지 남은 인프라 작업. 위→아래 순서, **1번이 전체의 전제**다.

1. **obs bridge 수정** — ✅ unit/runtime probe fixed. lerobot serve의 `GrootPolicyAdapter`가
   RoboCasa 키를 직접 수용(generic remap 우회). 상세·수정안:
   [Obs bridge mapping](#obs-bridge-mapping-stage-23).
2. **Docker HTTP smoke** — ✅ `/health`→`/reset`→`/act` 통과. RTX 4070(sm89)에서는
   120-only flash-attn wheel 때문에 Eagle attention eager fallback이 적용된다.
3. **post-fix closed-loop smoke** — ✅ OpenFridge target seed-0 success 확인. 전체 task SR이
   아니라 wiring smoke로만 취급한다.
4. **LeRobot N1.5 checkpoint-load verifier** — ✅ retained parity script는
   [Internal checkpoint-load parity](#internal-checkpoint-load-parity)에 기록. Activation bit
   parity와 closed-loop per-step trace parity는 남은 진단 대상이지만, 현재 별도 scripts는 두지 않는다.
5. **N1.6 ZMQ baseline 대비 action/SR parity** — N1.5 내부 parity가 더 안정화된 뒤에 비교한다.
   N1.6 YAML/profile은 N1.5 profile source of truth가 아니다.
6. **[stage 4] LeRobotDataset recorder 구현** — `run_vla_rollouts_groot`에
   `--record-lerobot-dataset` 추가. (1~3 이후 착수 가능하나, policy behavior 검증과
   별도 작업이다.)
7. **[stage 4] `lerobot-dataset-viz`로 시각화 확인** — 기록한 dataset 검수.

stage 1~3으로 "serve → HTTP → robocasa365" 연결은 확인됐다. stage 4가 끝나야 "serve → HTTP →
robocasa365 → analysis UI" 전체 파이프라인이 동작한다고 말할 수 있다.

### 정리 상태 (closed-loop와 독립)

- **helper 위치 정리**: runtime helper는
  `scripts/safe/groot_n15/robocasa/utils/runtime.py`만 source of truth로 둔다.
  `GrootPolicyAdapter`는 이 파일을 path-based loader로 가져온다.
- **one-off probe 정리**: payload variant exploration script는 retained diagnostic surface에서
  제거했다. 재현 가능한 내부 검증 축은 [Internal checkpoint-load parity](#internal-checkpoint-load-parity)의
  parity scripts로 둔다.
- **중복 smoke 정리**: 단순 checkpoint load smoke script는 제거했다. checkpoint load 검증은
  [Internal checkpoint-load parity](#internal-checkpoint-load-parity)의 internal parity verifier가 담당한다.

---

## Serve Adapter Spec (stage [1])

파이프라인 stage [1]의 명세다. raw Isaac-GR00T N1.5 RoboCasa365 checkpoint를 LeRobot
`GrootPolicy`로 로딩해 통일 HTTP `/act`를 노출한다.

### 체크포인트 형식

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

### 프로파일 필드 명세

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

#### `n_action_steps = 16`인 이유

LeRobot `GrootPolicy` 내부 action queue 길이와 GR00T N1.5 action horizon을 맞추기 위한
값. 8로 낮추면 `deque(maxlen=n_action_steps)`가 16-step chunk 앞부분을 버릴 수 있어서
HTTP profile은 16으로 둔다.

`/act`는 일반 LeRobot serving semantics를 유지해 queue에서 1-step action을 반환한다.
반대로 GR00T N1.5 `/act_with_features` collect path는 N1.6 SAFE collection과 같은
execution 단위를 맞추기 위해 hook 아래에서 `predict_action_chunk`를 직접 호출하고
`[H,D]` action sub-key를 반환한다.

#### data config order / normalization

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

### 구현 구조

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
relative-state alias를 직접 수용한다. Unit contract는
[Obs bridge mapping](#obs-bridge-mapping-stage-23)에 정리한다. 이 섹션의 주 범위는
stage [1] serve **로딩**이다.

### Serve 실행

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

### 캐시 경로

`docs/cache_paths.md`의 cache 규칙을 따른다.

- 컨테이너: `VLA_CACHE_ROOT=/cache`
- HF cache: `/cache/huggingface` because the `lerobot` compose service mounts
  repo-local `./data` to `/cache`.
- helper: `scripts/safe/groot_n15/robocasa/utils/runtime.py`

Runtime compatibility helper는 `scripts/safe/groot_n15/robocasa/utils/runtime.py` 하나만 source of
truth로 둔다. `GrootPolicyAdapter`는 path-based loader로 이 helper를 가져오고, parity/eval
diagnostics는 safe-tree utils path를 직접 추가해서 같은 helper를 가져온다.

### Stage [1] 검증

1. `tests/test_serve_lerobot.py` — adapter helper, profile-driven state/action shape,
   `/act` endpoint contract. ✅
2. `checkpoint_profile.load_profile(...)` — YAML schema 및 action dim. ✅
3. `TestGrootRobocasaObsBridge` — GR00T RoboCasa obs bridge unit contract. ✅
4. Docker HTTP smoke (위 명령) — ✅ `/act 200 OK`.
5. metadata order / stats flatten unit — ✅
6. deterministic `/act` probe with `inference_seed` — ✅ same payload/repeat action is stable.
7. closed-loop smoke — ✅ infra path runs; policy behavior 판단은 위
   [검증 상태](#검증-상태-2026-06-09)와 [Obs bridge mapping](#obs-bridge-mapping-stage-23)을 본다.

---

## Obs Bridge Mapping (stage [2][3])

파이프라인 `stage [2] HTTP`·`stage [3] robocasa365 eval`의 obs 계약 명세와 수정 상태다.

### 요구 계약 (GrootPolicy가 받아야 하는 HTTP 입력)

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

### 카메라 키 매핑 명세

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

#### 이전 serve remap 버그

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

### State 키 매핑 명세

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

### 현재 결론

| 경로 | 카메라 | state |
|---|---|---|
| generic `make_robocasa_processors` | ✓ direct `side_0/side_1/wrist_0` 보존, alias clobber 방지 | ✓ alias로 20D 조립 |
| groot-env `make_groot_robocasa_processors` (`--use-groot-env`) | ✓ `left/right/wrist` alias 수용 | ✓ direct `_rel`/base keys 수용, rotation_6d 변환 |

Unit contract와 실제 `/act` payload probe는 통과했다. 2026-06-09에는 RoboCasa
benchmark fork의 official `robocasa/<Task>` split 경로도
`scripts/safe/groot_n15/robocasa/eval/lerobot_http_eval.py`로 붙였다. 이후
OpenFridge target seed-0 LeRobot HTTP smoke는 성공했고, SR 외 내부값 검증은
[Internal checkpoint-load parity](#internal-checkpoint-load-parity)로 분리했다.
따라서 이 섹션의 결론 범위는 camera/state bridge 계약까지이며, model/action parity
판단은 internal parity 섹션을 기준으로 한다.

### N1.6 YAML 비교 결론

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

### 이미 풀린 선례 (재사용 대상)

N1.6 GR00T HTTP serve (`scripts/serve/groot.py`, :8500)는 이 문제를 이미 해결했다. 코드 확인:

- `src/policies/groot/core/schema.py`:
  - `observation.images.side_0` (+ `static`/`left` alias) → `video.res256_image_side_0`
  - `observation.state.eef_pos_rel` → `state.end_effector_position_relative`
  - `observation.state.base_position` → `state.base_position`
- `service.build_groot_obs`가 이를 소비.

새 lerobot serve는 동일한 alias 원칙을 `GrootPolicyAdapter` 안에서만 적용한다. shared
`GROOT_ENV_VIDEO_TO_UNIFIED_CAM`은 건드리지 않는다.

### 적용한 수정안

`scripts/serve/lerobot_adapters/groot.py`의 `GrootPolicyAdapter`가 RoboCasa 키를 직접
수용한다. generic remap(`static/wrist/wrist2` 순서 매핑)은 `PiPolicyAdapter`에만 유지한다.
schema 공유 키(`GROOT_ENV_VIDEO_TO_UNIFIED_CAM`)는 변경하지 않았다.

### Stage [2][3] 검증

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
    closed-loop 상태는 [Internal checkpoint-load parity](#internal-checkpoint-load-parity)와
    [Native ZMQ baseline](#native-zmq-baseline-openfridge-smoke)을 기준으로 한다.

### 과거 Native ZMQ vs LeRobot HTTP 비교 결과

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
  [Native ZMQ baseline](#native-zmq-baseline-openfridge-smoke)을 본다. 현재 retained
  SR-independent checkpoint-load parity는 [Internal checkpoint-load parity](#internal-checkpoint-load-parity)에 기록한다.

---

## Native ZMQ Baseline (OpenFridge Smoke)

LeRobot N1.5 behavior mismatch를 분리하기 위한 native Isaac-GR00T N1.5 ZMQ 비교 기록이다.
기준 runbook은 [`n15_02_eval.md`](n15_02_eval.md)이며, N1.6과 같은 `robocasa` simulator
client + policy server 구조로 실행한다.

### Runtime 메모

`groot_n15` compose service의 기본 cache mount는 `/cache`다. 현재 host
`/home/dongkyu/.cache/temporal_vla`가 `nobody:nogroup` 소유라 N1.5 server가
`/cache/datasets/huggingface` 아래 dynamic module cache를 만들 수 없다. 서버 실행
시 repo-local HF cache로 override한다.

```bash
docker compose run --rm --no-deps -T \
  -e HF_HOME=/temporal_vla/data/huggingface \
  -e HF_HUB_CACHE=/temporal_vla/data/huggingface/hub \
  -e HUGGINGFACE_HUB_CACHE=/temporal_vla/data/huggingface/hub \
  -e TRANSFORMERS_CACHE=/temporal_vla/data/huggingface/hub \
  -e HF_MODULES_CACHE=/temporal_vla/data/huggingface/modules \
  groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path /temporal_vla/data/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/14895998fe7c8f8f2441cc8957ec2c510302758b/gr00t_n1-5/multitask_learning/checkpoint-120000 \
  --embodiment-tag new_embodiment \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5558'
```

서버 확인:

```bash
ss -ltnp | rg ':5558'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

### OpenFridge Smoke: Legacy local env ID

```bash
docker exec temporal_vla-robocasa-run-3705634bbbf6 bash -lc '
export MUJOCO_GL=egl
PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_zmq_eval.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5558 \
  --env-name robocasa_panda_omron/OpenFridge_PandaOmron_Env \
  --n-episodes 1 \
  --n-envs 1 \
  --n-action-steps 16 \
  --max-episode-steps 400 \
  --steps-per-render 2 \
  --video-fps 10 \
  --seed 100002 \
  --video-dir /temporal_vla/outputs/debug/groot_n15_native_OpenFridge_seed100002/videos'
```

2026-06-09 결과:

```text
results:  ('robocasa_panda_omron/OpenFridge_PandaOmron_Env', [False], {})
success rate:  0.0
```

이 run은 raw `metadata.json` insertion order를 따르던 예전 repo-local data config를
사용했다. RoboCasa benchmark fork의 `PandaOmronDataConfig`는
`eef_pos/eef_rot/gripper/base_pos/base_rot` 정책 순서를 쓰고, state quaternion에는
6D rotation 변환을 적용한다. 따라서 이 예전 결과는 policy quality 결과가 아니라
wiring artifact로 본다.

산출물:

```text
outputs/debug/groot_n15_native_OpenFridge_seed100002/videos/3099486d-c4e7-4c98-8a05-554cb08b7ca2_s0.mp4
outputs/debug/groot_n15_native_OpenFridge_seed100002/contact_sheet.jpg
outputs/debug/groot_n15_native_OpenFridge_seed100002/contact_sheet_large.jpg
```

### OpenFridge Smoke: RoboCasa benchmark env

RoboCasa benchmark fork는 `panda_omron`을 `robocasa/<Task>`와 명시적인
`pretrain`/`target` split으로 평가한다. OpenFridge 기준으로는
`robocasa/OpenFridge`, `split=target`, 그리고 `get_task_horizon("OpenFridge")`에서
가져온 task horizon을 쓴다.

```bash
docker exec \
  -e MUJOCO_GL=egl \
  -e PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla:/temporal_vla/configs/policies \
  temporal_vla-robocasa-run-3705634bbbf6 \
  python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py \
    --host 127.0.0.1 \
    --port 5558 \
    --task OpenFridge \
    --split target \
    --n-episodes 1 \
    --n-action-steps 16 \
    --video-fps 10 \
    --steps-per-render 2 \
    --video-dir /temporal_vla/outputs/debug/groot_n15_officialrun_OpenFridge_target_verified/videos
```

2026-06-09 결과:

```text
Creating OpenFridge with split=target
EP 1 success: True; Cumulative success rate: 1.0
Collecting 1 episodes took 29.11 seconds
results:  ('robocasa/OpenFridge', [True])
success rate:  1.0
```

산출물:

```text
outputs/debug/groot_n15_officialrun_OpenFridge_target_verified/videos/616a8eaf-c1ee-40e0-a693-bf99753745d7_success1.mp4
outputs/debug/groot_n15_officialrun_OpenFridge_target_verified/contact_sheet_large.jpg
```

### OpenFridge Smoke: LeRobot HTTP official env

LeRobot HTTP 경로도 같은 official env id와 split convention을 사용한다. 다만 policy
호출은 이 repo의 통일 `/act` server를 거친다. Server-side GR00T adapter는 두 quaternion
state field를 같은 PyTorch3D/native `wxyz` convention으로 `rotation_6d`로 변환해
native state feature shape(`20D`)와 맞춘다.

```bash
docker exec \
  -e MUJOCO_GL=egl \
  -e NO_PROXY=127.0.0.1,localhost \
  -e no_proxy=127.0.0.1,localhost \
  -e PYTHONPATH=/temporal_vla:/temporal_vla/scripts/utils:/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite \
  temporal_vla-robocasa-run-3705634bbbf6 \
  python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/lerobot_http_eval.py \
    --vla-server http://127.0.0.1:8400 \
    --task OpenFridge \
    --split target \
    --n-episodes 1 \
    --seed 0 \
    --timeout 300 \
    --video-fps 10 \
    --steps-per-render 2 \
    --success-debug-every 80 \
    --video-dir /temporal_vla/outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_seed0_success_probe/videos
```

2026-06-09 seed-0 target 결과:

```text
Creating OpenFridge with split=target
[success-debug] episode=1 step=160 reward=0.0 success=False open=fridge_right_group_fridge_door_joint=0.633 info_keys=['success']
[success-debug] episode=1 step=172 reward=1.0 success=True open=fridge_right_group_fridge_door_joint=0.904 info_keys=['success']
EP 1 success: True; Cumulative success rate: 1.0
success_rate: 1.0
episode_lengths: [172]
first_success_steps: [172]
max_fixture_open: [0.9035059514573512]
```

산출물:

```text
outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_seed0_success_probe/videos/0eeb725c-c76b-46cc-9dc9-e12e5f2c9d18_success1.mp4
```

해석: official RoboCasa env가 LeRobot HTTP에 연결됐고, OpenFridge target 성공을 낼 수
있다. RoboCasa Gym wrapper는 `info["success"] = reward > 0`로 설정하며,
OpenFridge의 내부 checker는 `fixture.is_open(th=0.90)`이다. Video frame상으로는 문이
열려 보여도 normalized door joint가 `0.90`을 넘기 전까지는 success가 아니다. 시각적
성공과 RoboCasa success 판정이 어긋나면 `--success-debug-every`를 같이 사용한다.

관련 one-off probe: `OpenCabinet`(예전 naming 기준 `OpenSingleDoor`) target 1ep smoke는
success flag만 실패한 경우로 보이지 않았다. Debug output은 `reward=0.0`,
`success=False`, `max_fixture_open=0.006`에 머물렀고, 해당 run은 RoboCasa open-door
threshold를 물리적으로 넘지 못했다.
산출물: `outputs/debug/lerobot_groot_n15_officialrun_OpenCabinet_target_success_probe/videos/582833ff-b990-4911-ae3c-a787f4b4e064_success0.mp4`.

### 주의점

현재 `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py` helper는 `eval_seed`,
`filename_prefix`, `one_episode_per_env`를 받지 않는다. N1.5 client는 local helper를
호출하기 전에 지원하지 않는 kwargs를 걸러낸다. 이 checkout에서는 imported helper가
matching parameter를 노출할 때만 `--seed`가 보존된다.

`scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py`의 benchmark-style client는 benchmark
fork와 같은 env id/split convention을 사용한다. 다만 RoboCasa container에 새 dependency를
추가하지 않기 위해 repo-local N1.5 ZMQ msgpack client는 유지한다.

---

## Internal Checkpoint-load Parity

SR(success rate) 비교가 아니라, LeRobot HTTP 경로가 RoboCasa365 GR00T N1.5 checkpoint를
올바르게 로드하는지 확인하는 진단 섹션이다.

파일 수를 줄이기 위해 action/preprocess/ZMQ-vs-HTTP one-off probe scripts는 제거했다.
현재 유지하는 parity 표면은 checkpoint-load verifier 하나다.

### 유지하는 진단 스크립트

| Script | 역할 |
|---|---|
| `scripts/safe/groot_n15/robocasa/eval/internal_parity.py` | raw Isaac-GR00T checkpoint prefix/value parity verifier |

Runtime compatibility glue는 `scripts/safe/groot_n15/robocasa/utils/runtime.py` 한 곳에만 둔다.
Import shadowing을 피하기 위해 예전 eval-side helper duplicate는 제거했다.

### 현재 결론

현재 재현 가능한 retained verifier로 말할 수 있는 것은 아래까지다.

| Layer | 결론 | 근거 |
|---|---|---|
| checkpoint key/shape load | OK | 764 tensors checked, missing/shape/unexpected 0 |
| vision tower weights | exact | 448 tensors, `max_abs=0.0` |
| Eagle projector `backbone.eagle_model.mlp1` | exact | 2 tensors, `max_abs=0.0` |
| action head weights | dtype-materialized equivalent | 314 tensors, raw fp32-vs-bf16 `max_abs=0.007767`, checkpoint cast to model dtype gives mismatch 0 |

현재 주장할 수 있는 것은 "LeRobot과 native가 완전히 동일하다"가 아니다. 방어 가능한
결론은, 이전 loader gap에서 중요했던 raw checkpoint prefix들이 LeRobot-wrapped model 안에
들어갔고 shape 및 dtype-cast value가 맞는다는 것이다. Closed-loop SR과 activation-level
identity는 별도 질문으로 남긴다.

### 체크포인트 로드 검증

검증 명령:

```bash
python scripts/safe/groot_n15/robocasa/eval/internal_parity.py \
  --device cpu \
  --output outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json
```

2026-06-09에 사용한 container 명령:

```bash
docker exec temporal_vla-lerobot-run-0b327aff8915 bash -lc '
cd /temporal_vla &&
PYTHONPATH=/temporal_vla/scripts/safe/groot_n15/robocasa/eval:/temporal_vla/scripts/safe/groot_n15/robocasa/utils:/temporal_vla/scripts:/temporal_vla/scripts/serve:/temporal_vla/lerobot/src:/temporal_vla \
python scripts/safe/groot_n15/robocasa/eval/internal_parity.py \
  --device cpu \
  --output outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json'
```

산출물:

```text
outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json
```

요약:

```json
{
  "checked_count": 764,
  "missing_count": 0,
  "shape_mismatch_count": 0,
  "value_mismatch_count": 314,
  "dtype_cast_value_mismatch_count": 0,
  "unexpected_count": 0,
  "max_abs": 0.007767438888549805,
  "dtype_cast_max_abs": 0.0,
  "mismatch_count": 314
}
```

Prefix별 breakdown:

| Prefix | Checkpoint tensors | Checked | Missing | Shape mismatch | Raw value mismatch | Dtype-cast mismatch |
|---|---:|---:|---:|---:|---:|---:|
| `backbone.eagle_model.vision_model.` | 448 | 448 | 0 | 0 | 0 | 0 |
| `backbone.eagle_model.mlp1.` | 2 | 2 | 0 | 0 | 0 | 0 |
| `action_head.` | 314 | 314 | 0 | 0 | 314 | 0 |

Action head raw mismatch는 fp32 checkpoint tensor와 bf16 loaded model tensor를 직접 비교해서
생긴 차이다. Checkpoint tensor를 loaded model tensor dtype으로 cast하면 모든 action-head
tensor가 정확히 맞는다.

### 보존하지 않는 과거 근거

진단 과정에서 제거한 one-off probe들은 OpenFridge 첫 observation을 native/LeRobot boundary에서
비교했다. 최종 기록은 아래와 같다.

- 저장된 fixture key 기준 `GR00T prepare_input` boundary가 맞았다
  (`backbone_input__*`, `action_input__*` 모두 `max_abs=0.0`; bool mask mismatch 0).
- 선택한 parameter fingerprint가 정확히 맞았다.
- 첫 observation action chunk 차이는 bf16-scale increment 수준이었다
  (`flat_max_abs=0.00390625`; gripper/control mode exact).
- activation tensor는 bit-exact가 아니었으므로 full activation parity는 주장하지 않았다.

해당 scripts는 의도적으로 보존하지 않는다. 유지하는 regression surface는 위의
checkpoint-load verifier와 adapter state/action contract unit test다.

### `mlp1`만으로 단정하지 않는 이유

`backbone.eagle_model.mlp1`는 구체적으로 검증된 mismatch였다. Raw Isaac-GR00T checkpoint는
이 tensor를 `backbone.eagle_model.mlp1.*` 아래에 저장하지만, LeRobot wrapper는 해당 module을
`policy._groot_model.backbone.eagle_model.mlp1`로 노출한다. Repo-local adapter는 이제 이 두
tensor를 명시적으로 로드하고 검증한다.

하지만 이것이 `mlp1`이 유일한 문제였다는 뜻은 아니다. 현재 evidence는 아래까지다.

- `mlp1`은 이제 정확히 로드된다.
- 선택한 vision tower weight가 정확히 로드된다.
- action head key/shape는 로드됐고 dtype-cast 기준으로 동등하다.
- 과거 first-observation input/action evidence는 일관되게 보였다.
- 저장된 dump에서 activation tensor는 bit-exact가 아니었다.

따라서 올바른 결론은 이렇다. `mlp1`은 확인된 high-impact loader gap 중 하나였고, 이를
수리한 뒤 관측 가능한 action mismatch는 큰 semantic mismatch에서 작은 bf16-scale 차이로
줄었다. Runtime activation parity에는 아직 추가 차이가 남아 있을 수 있다.

### 검증 명령

재사용 checker unit test:

```bash
python -m pytest tests/test_lerobot_groot_n15_internal_parity.py -q
```

결과:

```text
5 passed
```

Target env 체크포인트 로드 verifier:

```bash
docker exec temporal_vla-lerobot-run-0b327aff8915 bash -lc '
cd /temporal_vla &&
PYTHONPATH=/temporal_vla/scripts/safe/groot_n15/robocasa/eval:/temporal_vla/scripts/safe/groot_n15/robocasa/utils:/temporal_vla/scripts:/temporal_vla/scripts/serve:/temporal_vla/lerobot/src:/temporal_vla \
python scripts/safe/groot_n15/robocasa/eval/internal_parity.py --device cpu \
  --output outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load_cpu.json'
```

결과:

```text
checked_count=764
mismatch_count=314
dtype_cast_value_mismatch_count=0
```
