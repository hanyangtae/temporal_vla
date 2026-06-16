# Serving Interface

이 프로젝트의 **통일 HTTP API** 단일 출처(single source of truth)다. 모든 모델 서버(`scripts/serve/*.py`)와 벤치마크 평가 스크립트(`scripts/eval/*.py`)는 여기 정의된 endpoint·payload 계약을 따른다. 새 모델/벤치/체크포인트를 붙일 때 다른 문서보다 먼저 이 문서를 본다.

연결 문서:
- 운영자가 컨테이너를 띄우는 방법: [02_docker_guide.md](02_docker_guide.md)
- 새 체크포인트를 붙일 때 체크리스트: [03_adding_checkpoint.md](03_adding_checkpoint.md)
- 체크포인트 프로파일 YAML 스키마: [`configs/checkpoints/README.md`](../configs/checkpoints/README.md)
- GR00T HTTP 변경 일지: [groot/n16_11_http_act_changes.md](groot/n16_11_http_act_changes.md)

## 운영 의도

**모델 × 벤치마크 조합을 자유롭게 갈아끼우는 것**이 목표다. 모델 서버와 벤치마크 스크립트는 서로의 native 포맷을 모르고, 통일 sub-key 계약을 통해서만 소통한다.

```
┌──────────────────────────────┐    HTTP 통일 API
│  벤치마크 컨테이너            │    POST /act, /act_with_features
│  (robocasa / calvin / libero) │    POST /reset, GET /health
│                              │─────────────────────────────┐
│  ObsProcessor → VLAClient    │                             │
│  ActionProcessor ← response  │                             │
└──────────────────────────────┘                             ▼
                                  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                                  │    xvla     │  │  dreamvla   │  │    upvla    │  │ openvla_oft │  │   lerobot   │  │    groot    │
                                  │    :8100    │  │    :8200    │  │    :8300    │  │    :8400    │  │    :8400    │  │    :8500    │
                                  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

- 모델 서버는 한 컨테이너에 하나의 process 로 떠 있고, 벤치마크는 `--vla-server http://localhost:<port>` 만 바꿔서 같은 모델을 여러 벤치로 평가하거나 한 벤치를 여러 모델로 비교한다.
- `openvla_oft` 와 `lerobot` 은 같은 `:8400` 을 쓰는 상호배타 컨테이너라 동시에 띄우지 않는다. GR00T N1.5(`groot_n15`)는 HTTP 가 아니라 ZMQ 경로라 위 다이어그램에서 생략했다.
- `n_action_steps` 와 `action_type` 은 `/health` 가 알리고, 벤치 측 `ActionProcessor` 가 그에 맞춰 chunk 를 소비한다.
- 일반 벤치마크 경로는 `src/processor/` 의 processor pipeline 이 env-native obs/action 을 통일 HTTP schema 로 바꾼다. GR00T `GrootRoboCasaEnv` native-key 경로(`robocasa_eval.py --use-groot-env`, SAFE wiring)는 `make_groot_robocasa_processors()` 를 쓰고, 실제 key mapping 은 `src.policies.groot.robocasa.io` adapter 를 단일 출처로 둔다.

## Endpoint 계약

모든 모델 서버는 `/act`, `/reset`, `/health` 를 같은 의미로 노출한다. `/act_with_features` 는 features 지원 모델만 노출한다.

| Endpoint | Method | 역할 |
|---|---|---|
| `/act` | POST | observation → sub-keyed action |
| `/act_with_features` | POST | `/act` 와 같은 응답 + `features.*` namespace (모델이 features 를 지원할 때만) |
| `/reset` | POST | episode 시작 시 server-side policy state 초기화 |
| `/health` | GET | server 상태 + sub-key 계약 메타 + features 메타 |

### `/health`

응답은 모든 모델 서버에 공통인 키를 포함하고, features 지원 모델은 그 뒤에 feature 메타 키(`supports_features` 이하)를 더 붙인다 (비지원 모델은 생략).

```json
{
  "status": "ok" | "not_loaded",
  "model": "<base_model_or_variant>",
  "profile": "<profile.name>",
  "action_type": "relative" | "absolute",
  "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
  "n_action_steps": 8,
  "supports_features": true,
  "feature_kind": "<model-specific identifier>",
  "feature_axes": ["denoising_step", "valid_action_step", "feature_dim"],
  "feature_slice": "valid" | "all",
  "feature_dtype": "float16" | "float32",
  "feature_action_horizon": null | 16
}
```

- `supports_features` 가 `false` 또는 부재면 `/act_with_features` 호출 결과는 `404`/`405` 또는 `features.*` 키 부재로 처리한다.
- `feature_kind` 는 모델별로 다르다 (예: GR00T N1.6 = `groot_n16_dit_valid_action_tokens_pre_velocity`). 클라이언트는 이 식별자를 통해 hidden state 의 의미를 알 수 있다.

### `/reset`

`POST /reset` 은 episode 경계에서 호출한다. 모델이 history queue, TTT inner state, 또는 stateful action chunk buffer 를 갖고 있으면 여기서 초기화한다. 없으면 no-op.

```json
{"status": "reset"}
```

### `/act`

#### 요청

벤치마크가 env observation 을 통일 sub-key 로 분리해 보낸다.

```json
{
  "observation.images.<view>": "<base64 PNG>",
  "observation.state.<key>": [float, ...],
  "task": "natural language instruction"
}
```

키 네임스페이스:

- `observation.images.<view>` — `view` 는 벤치/체크포인트가 정한 이름. 표준 alias: `static`, `left`, `right`, `wrist`, `side_0`, `side_1`, `wrist_0`. 모델 서버는 자신이 요구하는 `view` 만 사용하고 나머지는 무시.
- `observation.state.<key>` — 표준 sub-key: `eef_pos`, `eef_euler`, `eef_quat`, `eef_axisangle`, `eef_pos_rel`, `eef_quat_rel`, `gripper_qpos`, `gripper_qvel`, `gripper_opening`, `joint_pos`, `joint_vel`, `base_position`, `base_rotation`. 모델 서버가 필요한 키만 추출하고, 변환(quat→euler 등)이 필요하면 서버가 수행한다.
- `task` — string. 없으면 빈 문자열.

#### 응답 (sub-keyed)

```json
{
  "action.<subkey>": [[float, ...], ...],
  "latency_ms": 12.5
}
```

- 모델 서버는 자신의 native action 출력을 **표준 sub-key** 로 분리해 반환한다. 모든 sub-key 값은 2D list `[n_steps, dim]`.
- 표준 action sub-key:
  - `action.eef_pos` (3)
  - `action.eef_euler` (3) | `action.eef_axisangle` (3) | `action.eef_rot6d` (6) | `action.eef_quat` (4)
  - `action.gripper` (1)
  - `action.joint_pos` (7)
  - GR00T 전용: `action.base_motion` (4), `action.control_mode` (1)
- 응답이 `n_action_steps` 만큼의 chunk 를 반환할 수 있다 (모델별로 다름). 벤치 측 ActionProcessor 가 한 step 씩 소비한다.

#### 오류 응답

- 모델이 로드되지 않음 → `503 {"error": "model not loaded"}`
- payload 검증 실패 → `400 {"error": "<reason>"}`
- 알 수 없는 키는 무시되며 오류가 아니다 (다른 모델/벤치 조합 호환 보장).

### `/act_with_features`

`/act` 와 **같은 요청 payload** 를 받고, 같은 `action.*` sub-key 응답에 `features.*` namespace 를 추가한다. 모델이 SAFE-style hidden state 를 노출할 때만 의미가 있다.

#### 응답

```json
{
  "action.<subkey>": [[...]],
  "latency_ms": 12.5,
  "features.hidden_states": {
    "data": "<base64-encoded raw bytes>",
    "shape": [B, K, H, D],
    "dtype": "float16" | "float32"
  },
  "features.kind": "<model-specific identifier>",
  "features.axes": ["denoising_step", "valid_action_step", "feature_dim"],
  "features.slice": "valid" | "all",
  "features.exported_action_token_count": 16,
  "features.feature_action_horizon": 16,
  "features.valid_action_horizon": 16,
  "features.model_action_horizon": 50,
  "features.num_inference_timesteps": 4
}
```

- `features.hidden_states` 는 `data`(base64 raw bytes)+`shape`+`dtype` 로 구성된 JSON-safe feature blob. 클라이언트는 `scripts/utils/vla_client.py` `predict_with_features` 가 자동 복원하며, 공통 encode/decode 규약은 `src/utils/common/feature_blob.py` 에 둔다.
- `features.kind` 와 `features.axes` 는 모델마다 다르다 (예: GR00T N1.6 DiT pre-velocity action tokens). 클라이언트가 의미를 알아야 할 때 이 두 키를 사용.
- `features.exported_action_token_count`, `features.*_horizon`, `features.num_inference_timesteps` 는 `hidden_states` shape 해석에 필요한 메타. legacy alias normalization 은 `src/policies/safe_metadata.py` 에 둔다.
- 모델이 features 를 지원하지 않으면 endpoint 가 `404`/`405` 를 반환하거나 `features.*` 키 없이 `/act` 와 동일하게 응답한다 (구현 선택).
- 잘못된 slice/horizon 조합은 `400 {"error": "..."}`.

## VLA Client

`scripts/utils/vla_client.py` 의 `VLAClient` 가 위 계약의 reference client 다.

```python
from vla_client import VLAClient
import numpy as np

client = VLAClient("http://localhost:8500")
client.wait_until_ready(max_wait=600)        # /health 폴링
client.reset()                                # /reset

actions, latency_ms = client.predict(
    images={"left": img_hwc_u8, "right": ..., "wrist": ...},
    states={"observation.state.eef_pos_rel": np.array([0.0, 0.0, 0.0]), ...},
    instruction="pick up the red cup",
)
# actions: dict[str, np.ndarray]  e.g. {"action.eef_pos": [N, 3], ...}

actions, features, latency_ms = client.predict_with_features(
    images=..., states=..., instruction=...,
)
# features["hidden_states"]: np.ndarray (서버가 보낸 shape 그대로)
# features["kind"], features["axes"], features["slice"], features["*_horizon"]
```

`predict()` 는 sub-keyed dict 와 하위호환 flat array 둘 다 처리한다 (flat 은 신규 모델에서는 안 씀).

## Model × Benchmark 호환 매트릭스

행=모델 base, 열=벤치마크. 셀은 `configs/checkpoints/` 의 프로파일 stem (있는 것만). `n_action_steps` 와 `action_type` 도 같이 기재.

| Base model | Container | Port | Calvin | RoboCasa | LIBERO |
|---|---|---:|---|---|---|
| `xvla` | xvla | 8100 | `xvla__calvin_abc_d` (abs / 30) | — | — |
| `dreamvla` | dreamvla | 8200 | `dreamvla__calvin_dynamic_depth_semantic` (rel / 1) | — | — |
| `upvla` | upvla | 8300 | — | — | — |
| `lerobot` (pi0/pi05/groot) | lerobot | 8400 | `lerobot_pi05__calvin_sft` (rel / 50) | `lerobot_groot_n15__robocasa365_ckpt120000` (abs / 16) | — |
| `openvla_oft` | openvla_oft | 8400 | `openvla_oft__rlinf_calvin_sft` (rel / 8) | — | `openvla_oft__moojink_libero_spatial/object/goal/10` (rel / 8) |
| `groot` (N1.6) | groot | 8500 | — | `groot__robocasa_panda_omron`, `groot__robocasa365_ckpt120000`, `groot_ttt__robocasa_panda_omron` (rel / 16) | — |
| `groot_n15` | groot_n15 | — | — | ZMQ reference path only (`docs/groot/n15_02_eval.md`) | — |

각 셀의 sub-key emit 조합은 다음 절을 참조한다.

### 모델별 sub-key emit

| Base model | Emits sub-keys |
|---|---|
| `xvla` | `action.eef_pos`, `action.eef_rot6d`, `action.gripper` |
| `dreamvla` | `action.eef_pos`, `action.eef_euler`, `action.gripper` |
| `lerobot` (pi05) | `action.eef_pos`, `action.eef_euler`, `action.gripper` |
| `lerobot` (groot N1.5) | `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode` |
| `openvla_oft` | `action.eef_pos`, `action.eef_euler`, `action.gripper` |
| `groot` (N1.6) | `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode` |

### 벤치마크별 소비 가능 조합

| Bench | ActionProcessor | 필요 sub-key 조합 |
|---|---|---|
| Calvin | `src/processor/action/calvin.py` | `action.eef_pos` + (`action.eef_euler` \| `action.eef_rot6d` \| `action.eef_quat`) + `action.gripper` |
| RoboCasa | `src/processor/action/robocasa.py` | `action.eef_pos` + (`action.eef_axisangle` \| `action.eef_euler` \| `action.eef_rot6d` \| `action.eef_quat`) + `action.gripper` (선택: `action.base_motion`, `action.control_mode`) |
| LIBERO | OpenVLA-OFT 기준 | `action.eef_pos` + `action.eef_euler` + `action.gripper` (relative) |

새 모델/벤치 조합을 붙일 때는 **모델의 emit 조합이 벤치의 소비 조합을 부분집합으로 포함**해야 한다. 그렇지 않으면 프로파일의 `rotation_encoding` / `allow_conversions` 로 자동 변환을 선언한다.

GR00T `robocasa_eval.py --use-groot-env` 와 SAFE collection 은 위 generic RoboCasa `ActionProcessor` 대신 `src/processor/action/groot_robocasa.py` 를 사용한다. 이 processor 는 `src.policies.groot.robocasa.io` 를 감싸서 `GrootRoboCasaEnv` native action key 로 되돌린 뒤 `env.step()` 또는 SAFE rollout wrapper 에 전달한다.

## SAFE features

GR00T HTTP 서버는 `/act_with_features` 를 통해 DiT pre-velocity action token 을 export 한다. 같은 데이터를 ZMQ `get_action_with_features` 로도 export 한다 (upstream GR00T SAFE collector msgpack 호환). 둘 다 같은 `src/policies/groot/safe/features.py` 의 `capture_dit_features` 를 사용해 텐서 정의가 일치한다.

GR00T의 call-local RNG control은 transport별 위치만 다르다. HTTP `/act`/`/act_with_features`는 request payload의 `inference_seed`를 쓰고, ZMQ SAFE `get_action_with_features`는 request `options.inference_seed`를 쓴다. HTTP/ZMQ parity 검증에서는 같은 base seed에 policy-step index를 더한 schedule을 사용한다.

GR00T N1.6 특정 사항은 [`groot/n16_11_http_act_changes.md`](groot/n16_11_http_act_changes.md) 와 [`groot/n16_09_safe_parity.md`](groot/n16_09_safe_parity.md) 참조.

새 모델이 features 를 노출할 때 따를 일반 계약:

1. `/health` 에 `supports_features: true` 와 `feature_kind` / `feature_axes` 메타 노출.
2. `/act_with_features` 응답에 `features.hidden_states` feature blob (`data`+`shape`+`dtype`), `features.kind`, `features.axes`, `features.exported_action_token_count`, 그리고 모델이 export 하는 horizon 메타.
3. 가능하면 hidden state 정의를 `src/policies/<model>/safe/features.py` 같은 단일 모듈에 두고 HTTP/배치 추출 양쪽이 같은 함수를 호출하게 한다.

## 운영 패턴

### 1. 단일 모델 × 벤치마크 한 번 평가

```bash
# 1) 모델 서버 컨테이너 백그라운드
docker compose up -d xvla
docker compose exec xvla python /temporal_vla/scripts/serve/xvla.py \
    --profile /temporal_vla/configs/checkpoints/xvla__calvin_abc_d.yaml &

# 2) /health 가 준비될 때까지 대기
curl http://localhost:8100/health

# 3) 벤치 컨테이너에서 평가
docker compose exec calvin python /temporal_vla/scripts/eval/calvin.py \
    --vla-server http://localhost:8100 \
    --num-rollouts 50
```

### 2. 한 모델로 두 벤치마크 비교 (가능한 경우)

같은 모델 서버를 띄운 채 벤치만 바꾼다. `--vla-server` URL 은 그대로.

```bash
docker compose exec calvin    python .../calvin.py    --vla-server http://localhost:8500
docker compose exec robocasa  python .../robocasa_eval.py --vla-server http://localhost:8500
```

매트릭스의 같은 행에서 두 벤치 셀이 모두 채워져 있어야 가능.

### 3. 한 벤치마크에서 여러 모델 A/B

각 모델 서버를 동시에 띄우고 벤치만 `--vla-server` 를 바꿔서 반복.

```bash
# 두 서버를 다른 포트에 동시에 띄움 (다른 컨테이너이므로 GPU 메모리만 충분하면 OK)
docker compose exec xvla     python .../xvla.py     &     # :8100
docker compose exec dreamvla python .../dreamvla.py &     # :8200

for url in http://localhost:8100 http://localhost:8200; do
    docker compose exec calvin python .../calvin.py --vla-server "$url" --output-dir outputs/eval/...
done
```

### 4. SAFE feature 수집 (HTTP 경로)

```bash
docker compose exec groot python /temporal_vla/scripts/serve/groot.py \
    --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml \
    --feature-slice valid --feature-dtype float16
# 클라이언트는 vla_client.VLAClient.predict_with_features() 로 features.hidden_states 까지 수신
```

GR00T 의 기본 SAFE rollout 재현 경로는 upstream collector 호환을 위해 ZMQ 경로 (`scripts/safe/groot_n16/.../feature_server.py`) 를 유지한다. HTTP `/act_with_features` 는 Project FastAPI evaluation interface 로 SAFE feature 를 수집해야 할 때 쓰는 supported transport 이며, ZMQ `get_action_with_features` 와 같은 hidden state 정의를 export 한다.

### 5. 새 체크포인트/모델 붙이기

[`03_adding_checkpoint.md`](03_adding_checkpoint.md) 체크리스트 그대로. 핵심은:
1. 프로파일 YAML (`configs/checkpoints/<name>.yaml`) 작성 — sub-key 계약 선언.
2. `scripts/serve/<base_model>.py` 가 그 프로파일을 로드해 `/act` 응답을 그대로 만들 수 있게 분기.
3. 이 문서의 매트릭스에 행/셀 추가.
4. `bash scripts/serve/run_<model>_http_smoke.sh` 같은 gated smoke (있으면) 또는 `python scripts/utils/smoke_test_serve.py --profile ...` 로 `/health` → `/reset` → `/act` round trip 확인.

## 변경 정책

- 통일 API 계약 변경은 이 문서 + `scripts/utils/vla_client.py` + 모든 `scripts/serve/*.py` + 영향 받는 벤치마크 `ActionProcessor` 까지 한 번에 갱신한다.
- 모델별 endpoint 확장(예: `/act_with_features` 의 model-specific 필드)은 가능하면 `features.<model>_<field>` 처럼 모델별 namespace 를 만든다. `/health.feature_kind` 가 그 분기 식별자다.
- 새 sub-key 표준을 추가하면 이 문서의 "키 네임스페이스" 절과 호환 매트릭스를 같이 업데이트한다.
