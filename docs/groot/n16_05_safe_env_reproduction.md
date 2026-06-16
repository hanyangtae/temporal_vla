# GR00T N1.6 RoboCasa — Scenario Reproduction

이 문서는 `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py` 수집 경로에서 RoboCasa scenario seed와 `ep_meta` manifest가 어떤 재현 범위를 보장하는지 정리한다. 기준 환경은 `robocasa365` (`src/benchmarks/robocasa`) + `robosuite` (`src/benchmarks/robosuite`) + GR00T N1.6 SAFE feature collection이다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - **05 Scenario Reproduction (이 문서)**
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - [07 SAFE Detector](n16_07_safe_detector.md)
> - [08 SAFE Visualization](n16_08_safe_visualization.md)
> - [09 SAFE Parity](n16_09_safe_parity.md)
> - [10 SAFE Report](n16_10_safe_report.md)

## 결론

현재 수집 경로의 재현 단위는 **scene composition**이다.

- `--seed S`는 RoboCasa env construction까지 전달되어 `env.rng = np.random.default_rng(S)`를 만든다.
- `--ep-meta-dir`가 있으면 `(env_name, scenario_seed)`에 대응하는 `ep_meta` JSON manifest를 import/export한다.
- pkl payload에는 `seed`, `scenario_seed`, `ep_meta`와 replay/video config
  (`n_action_steps`, `max_episode_steps`, `video_fps`, `steps_per_render`,
  `inference_seed`)가 함께 저장된다.
- 같은 RoboCasa/robosuite commit, 같은 dependency, 같은 asset에서는 `ep_meta`가 없어도 같은 `(env_name, scenario_seed)`로 같은 scene composition을 다시 생성할 수 있다.
- 같은 manifest를 import하면 같은 layout/style, object identity, texture, fixture reference, robot base pose를 재사용한다.

현재 수집 경로는 **bit-identical initial sim state**를 보장하지 않는다.

- object exact `pos/quat`와 fixture exact placement는 `ep_meta`에 들어가지 않는다.
- replay에서도 placement sampler가 다시 호출되므로 object 좌표는 미세하게 달라질 수 있다.
- rendering pixel, model hidden state, policy trajectory bit equality는 GPU, renderer, model determinism에 따라 달라진다.

## 용어

| 용어 | 의미 |
|---|---|
| episode | reset부터 terminal/truncation까지 한 번의 rollout 실행 단위 |
| scenario_seed | RoboCasa env construction seed. `collect_rollout.py --seed` 값 |
| scenario / scene composition | layout/style, object cfg, texture, fixture reference, camera/config, robot base pose 수준의 task instance |
| ep_meta | RoboCasa `env.get_ep_meta()` 결과. 현재 pkl과 JSON manifest에 저장되는 scenario 기록 |
| manifest | `--ep-meta-dir` 아래 저장되는 seed-keyed `ep_meta` JSON |
| bit-identical sim state | object/fixture pose, qpos/qvel까지 완전히 같은 초기 simulator state |

## 현재 수집 계약

### 단일 collector

`collect_rollout.py`는 env를 직접 만든다.

```python
gym.make(env_name, enable_render=True, seed=scenario_seed)
```

이 seed는 `robocasa.utils.gym_utils.create_env_robosuite(..., seed=...)`를 거쳐 `robosuite.make(..., seed=...)`로 들어간다. robosuite base env는 construction 시점에:

```python
self.rng = np.random.default_rng(seed)
```

를 만든다. RoboCasa layout/object/placement/robot spawn sampling은 이 `env.rng`를 사용한다.

### task-set wrapper

`collect_task_set_via_docker_exec.sh`는 task마다 episode를 하나씩 collector에 넘긴다.

```bash
seed=$((SEED_START + local_episode_idx))
python collect_rollout.py \
    --n-episodes 1 \
    --seed "${seed}" \
    --ep-meta-dir "${OUT_ROOT_CONTAINER}/_ep_metas/${task}"
```

따라서 `EPISODES_PER_TASK=100`, `SEED_START=100000`이면 각 task는:

```text
ep0  -> scenario_seed 100000
ep1  -> scenario_seed 100001
...
ep99 -> scenario_seed 100099
```

로 수집된다. 같은 task 안의 100개 episode는 서로 다른 deterministic scenario다.

### `collect_rollout.py --n-episodes K` 직접 호출

collector를 직접 `--n-episodes 3 --seed 100000`으로 호출해도 wrapper와 같은 schedule을
쓴다.

```text
local_ep_idx 0 -> scenario_seed 100000
local_ep_idx 1 -> scenario_seed 100001
local_ep_idx 2 -> scenario_seed 100002
```

같은 scenario를 의도적으로 반복 관찰하려면 `--n-episodes 1 --seed S`를 별도 호출로
반복하고, 그 run은 variance probe라고 명시한다. 기본 collector contract는 episode마다
서로 다른 deterministic scenario를 만드는 것이다.

### Paired N1.5/N1.6 comparison config

N1.5 LeRobot HTTP feature collection과 N1.6 HTTP/ZMQ SAFE collection을 비교할 때는
manifest만 공유하는 것으로는 부족하다. 다음 config를 같은 값으로 고정한다.

| config | paired 비교 기준 |
|---|---|
| `env_name` | 같은 RoboCasa env id |
| `scenario_seed` | 같은 `--seed + local_ep_idx` schedule |
| `ep_meta` | 같은 `--ep-meta-dir` manifest import/export |
| `n_action_steps` | 같은 action chunk 실행 길이 |
| `max_episode_steps` | 같은 episode truncation horizon |
| `video_fps`, `steps_per_render` | 같은 upstream video sampling |
| `task_description` | 같은 instruction text |
| `inference_seed` | 같은 per-policy-call seed base, model 차이는 별도 해석 |

## `ep_meta` import/export 동작

`--ep-meta-dir`를 지정하고 `--seed S`가 있으면 collector는 각 episode의
`scenario_seed = S + local_ep_idx`마다 다음 순서로 동작한다.

### manifest가 없을 때

```text
scenario_seed로 env 생성
reset
env.get_ep_meta() 캡처
rollout 실행
pkl에 ep_meta 저장
<ep-meta-dir>/<env-name>--seed<scenario_seed>.json export
```

로그에는 `ep_meta=exported`가 찍힌다.

### manifest가 있을 때

```text
scenario_seed로 env 생성
<ep-meta-dir>/<env-name>--seed<scenario_seed>.json load
env.set_ep_meta(manifest["ep_meta"])
reset
rollout 실행
pkl에 manifest ep_meta 저장
```

로그에는 `ep_meta=imported`가 찍힌다. RoboCasa reset 중 `ep_meta` dict 내부 일부가 in-place로 바뀔 수 있으므로, collector는 `set_ep_meta()`에 deepcopy를 넘기고 pkl에는 manifest 원본을 JSON-safe 형태로 저장한다.

### `--seed`가 없을 때

manifest key가 없으므로 import/export를 하지 않는다. 그래도 pkl에는 해당 run에서 캡처한 `ep_meta`를 저장한다.

## 보장 범위

### 보장하는 것

같은 코드/asset/env에서 같은 `(env_name, scenario_seed)` 또는 같은 manifest를 쓰면 다음 수준이 재현된다.

| 항목 | 보장 방식 |
|---|---|
| layout/style | `ep_meta["layout_id"]`, `ep_meta["style_id"]` |
| object identity/config | `ep_meta["object_cfgs"]` |
| generative textures | `ep_meta["gen_textures"]` |
| fixture references | `ep_meta["fixture_refs"]` |
| robot base init | `ep_meta["init_robot_base_pos"]`, `ep_meta["init_robot_base_ori"]` |
| instruction text | task `get_ep_meta()`가 저장한 `ep_meta["lang"]` |
| camera setup | 현재 collector는 `randomize_cameras=False` 경로라 config가 안정적 |

이 범위가 본 repo의 SAFE rollout comparison에서 말하는 “같은 scenario”다.

### 보장하지 않는 것

| 항목 | 이유 |
|---|---|
| object exact `pos/quat` | `object_placements`가 `ep_meta`에 저장되지 않고 reset 때 sampler가 다시 호출됨 |
| fixture exact `pos/quat` | fixture placement도 sampler가 다시 호출됨 |
| initial `qpos/qvel` bit identity | placement, MuJoCo state propagation, renderer/backend 차이 |
| rendered RGB bit identity | OpenGL/EGL, GPU, driver 차이 |
| hidden_state bit identity | GR00T inference, GPU precision, action divergence 영향 |
| trajectory equality | 같은 scenario에서도 policy action과 sim state가 미세하게 갈라질 수 있음 |

정확한 object pose까지 비교해야 하면 첫 reset 직후 `object_placements` 또는 `sim.data.qpos/qvel`을 별도 artifact로 저장해서 비교한다.

### 2026-05-29 HTTP replay validation note

HTTP eval path도 `--ep-meta-dir` import/export와 `--inference-seed`를 지원한다. Runtime 검증에서는 import/export 경로가 동작하는 것이 확인됐지만, `ep_meta`가 full reset state replay artifact가 아니라는 결론도 같이 확인됐다. 상세 수치와 artifact는 [09 SAFE Parity](n16_09_safe_parity.md#runtime-validation-2026-05-29)에 둔다.

따라서 “같은 scenario”는 layout/style/object config/texture/fixture/instruction 수준의 의미다. Closed-loop action trace equality 또는 PC 간 bit-level replay가 필요하면 `ep_meta`에 더해 reset 직후 full MuJoCo state (`qpos/qvel` 등)를 별도 manifest로 저장하고 복원해야 한다.

## PC 간 replay 절차

### 수집 PC

전체 task 수집은 wrapper가 manifest를 자동 저장한다.

```bash
TASK_SET=target_atomic_seen18 \
EPISODES_PER_TASK=100 \
SEED_START=100000 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

산출물:

```text
<OUT_ROOT>/<Task>/task<task_id>--ep<episode_idx>--succ<0|1>.pkl
<OUT_ROOT>/<Task>/task<task_id>--ep<episode_idx>--succ<0|1>.mp4
<OUT_ROOT>/<Task>/task<task_id>--ep<episode_idx>--succ<0|1>.csv
<OUT_ROOT>/_ep_metas/<Task>/<env-name>--seed<scenario_seed>.json
```

### 다른 PC

같은 task scenario를 replay하려면 다음을 맞춘다.

- `src/benchmarks/robocasa` commit
- `src/benchmarks/robosuite` commit
- RoboCasa assets
- Python / MuJoCo / NumPy / numba 버전
- `env_name`
- `scenario_seed`
- `_ep_metas/<Task>/*.json` manifest

pkl 전체를 옮기면 SAFE feature analysis까지 바로 가능하다. scene composition replay만 필요하면 `_ep_metas` manifest만 옮겨도 된다.

### seed만 있는 경우

manifest 없이 `scenario_seed`만 있어도 현재 collector는 seed를 env construction에 넣는다. 따라서 같은 RoboCasa/robosuite commit, 같은 dependency, 같은 asset, 같은 `env_name`에서는 PC가 달라도 같은 scene composition을 다시 생성할 수 있다.

이 mode는 동일한 실험 환경을 복원할 수 있을 때 충분하다. 그러나 seed-only replay는 RoboCasa 내부 sampling 순서와 code path가 그대로 유지된다는 가정에 의존한다. RoboCasa/robosuite commit, object config 생성 순서, texture sampling 순서, wrapper reset 흐름이 바뀌면 같은 seed에서도 scene composition이나 세부 placement가 달라질 수 있다.

PC 간 전달, 장기 보존, 코드 변경 뒤 replay는 manifest를 기준으로 한다. `ep_meta` manifest는 실제 생성된 scenario 기록을 저장하고 replay 시 명시적으로 주입하므로 seed-only보다 code-path drift에 강하다.

## 검증 방법

### 정적 검증

collector와 task-set wrapper를 수정한 뒤 먼저 host에서 정적 검증을 실행한다.

```bash
python -m py_compile scripts/safe/groot_n16/robocasa/collect/collect_rollout.py
bash -n scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
git diff --check
```

RoboCasa container 안에서 collector CLI import와 option wiring도 확인한다.

```bash
docker exec \
  -e ROBOCASA_ENV_SOURCE=robocasa365 \
  -e PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
  robocasa bash -lc \
  "cd /temporal_vla && python scripts/safe/groot_n16/robocasa/collect/collect_rollout.py --help | grep -E 'seed|ep-meta'"
```

### manifest와 pkl ep_meta 비교

```bash
docker exec \
  -e PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
  robocasa bash -lc "cd /temporal_vla && python - <<'PY'
import json
import pickle
from pathlib import Path
from src.policies.groot.robocasa.scenario_replay import json_safe

root = Path('outputs/eval/robocasa/groot_n16/<run>/raw_rollouts')
task = 'CloseFridge'
manifest_path = next((root / '_ep_metas' / task).glob('*.json'))
manifest = json.load(open(manifest_path))
manifest_digest = json.dumps(json_safe(manifest['ep_meta']), sort_keys=True)

for path in sorted((root / task).glob('*.pkl')):
    payload = pickle.load(open(path, 'rb'))
    digest = json.dumps(json_safe(payload['ep_meta']), sort_keys=True)
    print(path.name, payload['scenario_seed'], digest == manifest_digest)
PY"
```

### object placement drift 확인

같은 manifest replay 뒤 object pose 차이를 보고 싶으면 first reset 직후 다음 정보를 별도 dump한다.

```python
placements = {}
for name, (pos, quat, _obj) in env.object_placements.items():
    placements[name] = {
        "pos": [float(x) for x in pos],
        "quat": [float(x) for x in quat],
    }
```

object class/cfg가 같고 pos 차이가 작은 수준이면 같은 scene composition의 다른 placement sample로 본다. object class나 fixture reference가 달라지면 manifest import/export 누락을 의심한다.

## Historical smoke run

2026-05-28에 `CloseFridge`로 실제 collection smoke를 수행했다. 이 기록은 당시
import/export wiring을 확인한 historical result다. 현재 collector는 같은 명령의
`--n-episodes 3 --seed 100000`을 아래 schedule로 해석한다.

```text
ep0 -> scenario_seed 100000 -> ...--seed100000.json
ep1 -> scenario_seed 100001 -> ...--seed100001.json
ep2 -> scenario_seed 100002 -> ...--seed100002.json
```

실행 명령:

```bash
docker exec \
  -e ROBOCASA_ENV_SOURCE=robocasa365 \
  -e MUJOCO_GL=egl \
  -e PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
  robocasa bash -lc \
  "python /temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5557 \
    --env-name robocasa_panda_omron/CloseFridge_PandaOmron_Env \
    --robocasa-env-source robocasa365 \
    --output-dir /temporal_vla/outputs/eval/robocasa/groot_n16/scenario_replay_smoke_3ep/raw_rollouts/CloseFridge \
    --task-id 1 \
    --episode-start-idx 0 \
    --n-episodes 3 \
    --seed 100000 \
    --ep-meta-dir /temporal_vla/outputs/eval/robocasa/groot_n16/scenario_replay_smoke_3ep/raw_rollouts/_ep_metas/CloseFridge"
```

```text
outputs/eval/robocasa/groot_n16/scenario_replay_smoke_3ep/raw_rollouts/CloseFridge
```

당시 결과:

| episode | result | manifest mode | steps |
|---|---:|---|---:|
| ep0 | succ1 | exported | 23 |
| ep1 | succ1 | imported | 30 |
| ep2 | succ1 | imported | 22 |

당시 세 pkl은 모두 `scenario_seed=100000`이었고, 세 pkl의 `ep_meta`는
`_ep_metas/CloseFridge/robocasa_panda_omron_CloseFridge_PandaOmron_Env--seed100000.json`
manifest와 일치했다. 이 결과는 현재 multi-episode seed schedule의 증거가 아니라,
collection path와 manifest import/export가 동작했다는 historical smoke로만 읽는다.
hidden_state bit identity 검증도 아니다.

## 기존 collection의 한계

이 변경 전 수집된 rollout pkl에는 `ep_meta` manifest가 없을 수 있다. 그런 collection은 hidden_states/actions/video/success label artifact로는 유효하지만, 같은 scenario를 manifest 기반으로 replay할 수 없다. 필요한 경우 현재 collector로 재수집한다.
