# RoboCasa 환경 결정성과 PC 간 재현 가이드

이 문서는 `src/benchmarks/robocasa` (RoboCasa365 v1.0.0, `hanyangtae/robocasa` fork) 기반 eval 시:

1. `seed` / `rollout` 인덱스가 환경에 미치는 영향
2. 어떤 파일의 어떤 변수가 무엇을 담당하는지
3. 변수 변경 시 환경이 어떻게 달라지는지
4. 한 PC에서 만든 환경을 다른 PC에서 똑같이 재현하는 절차

를 정리한다. 본 repo의 `scripts/eval/robocasa_eval.py` 와 `groot_*_zmq_eval.py` eval 경로, 그리고 SAFE 수집 경로 (`scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`, §11) 를 모두 다룬다.

> **참고:** §11 은 기존 `docs/groot_n16_robocasa_env_reproduction.md` (GR00T 수집 재현 가이드) 를 대체한다. 그 문서의 "같은 seed → 동일 scene" 주장은 코드·런타임으로 반증되어 본 문서로 통합·정정했다.

---

## 1. 대상 식별

| 항목 | 값 | 근거 |
|------|-----|------|
| 벤치마크 | RoboCasa365 | `src/benchmarks/robocasa/README.md` 1행, "365 tasks" |
| robocasa 버전 | 1.0.0 | `src/benchmarks/robocasa/setup.py:43` |
| robosuite 버전 | master fork (`src/benchmarks/robosuite`) | submodule |
| 필수 pin | `mujoco==3.3.1`, `numpy==2.2.5`, `numba==0.61.2`, python 3.11 | `setup.py:19-22` |

---

## 2. seed → 환경 결정성 메커니즘

### 2.1 핵심 객체: `env.rng`

- 정의: `src/benchmarks/robosuite/robosuite/environments/base.py:144-145`
  ```python
  self.seed = seed
  self.rng = np.random.default_rng(seed)
  ```
- **`__init__` 에서 단 한 번** 생성됨. `env.reset()` 으로 **재시드되지 않음**.
- robocasa Kitchen 환경은 `hard_reset=True` 로 robosuite 를 호출하므로 (`kitchen.py:521`), 매 `reset()` 마다 `_load_model()` → `_setup_model()` → `_reset_internal()` 가 호출되고 그 안에서 `self.rng` 가 **계속 advance** 함.

### 2.2 같은 `seed` 일 때

| 조건 | 결과 |
|------|------|
| `seed=42`, `num_rollouts=1` | 항상 같은 환경 (완전 결정적) |
| `seed=42`, `num_rollouts=N` | **결정적 시퀀스**의 N개 서로 다른 환경. 재실행 시 같은 N개가 같은 순서로 재생됨 |
| `seed=None` | `np.random.default_rng(None)` → 완전 랜덤 |

즉 **rollout i 와 rollout j 는 다른 환경**이지만, **(seed, rollout_idx) 쌍이 같으면 어떤 PC에서 돌려도 같은 환경**이 나온다 — 단, 2.4 의 모든 조건이 일치할 때.

### 2.3 어떤 파일의 어떤 변수가 무엇을 담당하나

#### 직접 환경 생성 인자 (`scripts/eval/robocasa_eval.py:63` `create_eval_env`, 또는 `robosuite.make`)

| 변수 | 위치 | 담당 | 변경 시 영향 |
|------|------|------|-------------|
| `seed` | `kitchen.py:391`, `robosuite/base.py:144` | `env.rng` 초기 상태 | 시드를 바꾸면 layout/object/placement/robot pose 시퀀스 전체가 다른 시퀀스로 바뀜 |
| `layout_and_style_ids` | `kitchen.py:392, 424-438` | 가능한 (layout, style) 후보 리스트 | 매 reset에 `self.rng.choice(layout_and_style_ids)` 로 1개 샘플 (`kitchen.py:595`). 리스트 길이가 1이면 scene 고정, 5이면 5종 중 1종 |
| `obj_instance_split` | `kitchen.py:401, 422` | object 인스턴스 풀 분할 | `"pretrain"` / `"target"` / `None` — 같은 카테고리(mug 등) 안에서 어느 인스턴스 집합에서 뽑을지 결정. 본 fork 는 `"pretrain"`/`"target"` 만 인식 (`gymnasium_basic.py:78-80`) |
| `generative_textures` | `kitchen.py:396, 450-451` | AI 생성 텍스처 사용 여부 | `"100p"` 이면 reset 마다 cab/counter/wall/floor 텍스처 4종 재샘플 (`kitchen.py:1321-1338`). `None` 이면 비활성 |
| `randomize_cameras` | `kitchen.py:404, 455` | wrist/agentview 카메라에 가우시안 노이즈 | `True` 면 매 reset 카메라 pose 가 흔들림 |
| `robot_spawn_deviation_pos_x` / `_pos_y` / `_rot` | `kitchen.py:405-407` | 로봇 base spawn 시 deviation | `env.rng.uniform` 으로 deviation 범위 내 sampling (`env_utils.py:1531, 1602`). 0 이면 anchor 위치 고정 |
| `obj_registries` | `kitchen.py:397` | object 소스 풀 | `("objaverse", "lightwheel")` 디폴트. `("aigen",)` 추가하면 AI 생성 오브젝트 포함 |
| `enable_fixtures`, `update_fxtr_cfg_dict`, `clutter_mode` | `kitchen.py:395, 408-409, 447-449` | scene fixture 구성, 잡동사니 양 | scene 자체의 fixture 집합 변경 |
| `use_distractors` | `kitchen.py:402, 453` | 방해 object 추가 | `True` 면 task 와 무관한 object 가 scene 에 등장 |
| `use_novel_instructions` | `kitchen.py:411, 497-501, 532-558` | task instruction 변형 CSV 사용 | `True` 면 `models/assets/novel_instructions/task_instruction_variants.csv` 의 Variant 1/2/3 중 `self.rng.choice` (4. 참조) |
| `camera_names/widths/heights/depths` | `kitchen.py:384-387` | obs 카메라 spec | 모델 입력 해상도 결정 |
| `robots`, `controller_configs` | `kitchen.py:364-366` | 로봇 종류 & 컨트롤러 | PandaOmron (=PandaMobile) 가 본 repo 기본 |

#### `self.rng` 가 매 reset 마다 소비하는 지점 (=rollout i ↔ j 사이에 달라지는 모든 것)

| 항목 | 위치 |
|------|------|
| (layout, style) 선택 | `kitchen.py:595` `self.rng.choice(self.layout_and_style_ids)` |
| fixture 인스턴스 선택 | `kitchen.py:1709, 1745` `self.rng.choice(matches/close_fixtures)` |
| object 인스턴스 / 크기 / 회전 | `kitchen.py:1620` `sample_kitchen_object(rng=self.rng, split=...)` |
| object placement 좌표 | `env_utils.py:916, 1005` `SequentialCompositeSampler(rng=env.rng)` |
| robot base spawn (yaw, xy) | `env_utils.py:1531, 1602` `env.rng.uniform(...)` |
| generative textures 4종 | `kitchen.py:1321-1338` `get_random_textures(self.rng)` (활성 시) |
| task 별 추가 (예: 싱크 핸들 상태) | 각 task class `_setup_scene`, 예: `wash_fruit_colander.py:25` `self.sink.set_handle_state(rng=self.rng)` |
| `use_novel_instructions=True` 일 때 instruction 변형 | `recycle_bottles_by_type.py:128` 등 `self.rng.choice(self.novel_instructions)` |

#### rollout 간 **바뀌지 않는** 것

- task class (`TurnOnMicrowave`, `PnPCounterToCab` 등 — eval loop 가 같은 env_name 유지)
- 로봇 종류, 컨트롤러
- camera 종류/해상도 (`randomize_cameras=False` 일 때 pose 도 고정)
- `obj_instance_split` 등 init 인자 전체

### 2.4 본 repo 의 두 eval 경로별 seed 처리

| 경로 | seed 전달 위치 | 결정성 |
|------|---------------|--------|
| `robocasa_eval.py` 기본 (`create_eval_env`) | `:84,92` `robosuite.make(..., seed=args.seed)` | seed 전달됨 → `env.rng` 결정적 |
| `robocasa_eval.py --use-groot-env` | `:376` `gym.make(env_id, enable_render=True)` — **seed 없음** | `env.rng = default_rng(None)` → 완전 랜덤. 결정성 필요하면 `GrootRoboCasaEnv.__init__` 에서 seed forwarding 패치 필요 |
| `groot_*_zmq_eval.py` (`run_rollout_gymnasium_policy`) | `WrapperConfigs(seed=...)` 인자가 있긴 하나 `gr00t/eval/rollout_policy.py:67-76` 의 dataclass 에는 `seed` 필드가 없음 → TypeError 또는 silently dropped | 현재 환경 결정적 보장 안 됨 |
| `gymnasium_basic.py:264` `reset(seed=...)` | `np.random.seed(seed)` — **글로벌 numpy RNG** 만 건드림 | `env.rng` (PCG64) 무관 → 환경 재현 효과 없음 |

> **결론:** 현재 코드에서 결정성을 안전하게 누리려면 `robocasa_eval.py` 의 legacy 경로 (`--use-groot-env` 미사용) + `--seed` 명시 + 환경마다 새 env 생성 (또는 같은 env 내 N reset).

---

## 3. rollout 수와 결정성

### 3.1 매 rollout 새 env 생성 vs. 한 env 에서 N reset

```python
# 패턴 A — 같은 env 에서 N reset (현재 robocasa_eval.py 의 방식)
env = create_eval_env(seed=42, ...)
for i in range(N):
    obs = env.reset()          # rng 가 advance, 매 reset 다른 scene
    run_one_rollout(env, ...)
env.close()
```

```python
# 패턴 B — 매 rollout 새 env (가장 단순한 재현 단위)
for i in range(N):
    env = create_eval_env(seed=42 + i, ...)  # 또는 같은 seed 라도 init 마다 reset 0번
    obs = env.reset()
    run_one_rollout(env, ...)
    env.close()
```

- 패턴 A: 1번의 `seed` 로 N rollouts. (seed, rollout_idx) 가 환경 식별자.
- 패턴 B: rollout 별로 독립 seed → 같은 i 에 같은 seed 만 박으면 항상 같은 환경. 멀티프로세스/병렬화에 유리.

### 3.2 N rollout 시 첫 번째 와 두 번째의 차이

`seed` 고정 + 패턴 A 일 때 rollout 1 ↔ rollout 2 사이에 달라지는 것은 **2.3 의 "self.rng 가 매 reset 마다 소비하는 지점" 전체** — scene, fixture, object 인스턴스, object 위치, robot spawn, 텍스처, 싱크 상태 등 거의 모두.

**바뀌지 않는 것:** task class 자체, robot 종류, controller, camera spec.

### 3.3 완전 결정적 1-rollout 셋업

```python
create_eval_env(
    env_name="TurnOnMicrowave",
    seed=42,
    layout_and_style_ids=((1, 1),),   # 단일 scene
    obj_instance_split="target",       # 또는 "pretrain"
    generative_textures=None,
    randomize_cameras=False,
)
```
+ `num_rollouts=1` 또는 매 rollout 마다 env 재생성.

---

## 4. Instruction text 의 출처

한 task class 안의 instruction 다양성은 두 경로로 만들어진다.

### 4.1 (A) novel_instructions 모드

- 활성화: `Kitchen(..., use_novel_instructions=True)` (`kitchen.py:497-501`)
- CSV: `src/benchmarks/robocasa/robocasa/models/assets/novel_instructions/task_instruction_variants.csv`
  - 컬럼: `Task, Original Instruction, Variant 1, Variant 2, Variant 3`
- 로드: `kitchen.py:532-558` `_load_novel_instructions()`. CSV 에 해당 task row 가 **없으면 init 에서 AssertionError**.
- 선택: 각 task class 의 `get_ep_meta()` 에서 `ep_meta["lang"] = self.rng.choice(self.novel_instructions)` (`recycle_bottles_by_type.py:128`, `wash_fruit_colander.py:28` 등).
- 최대 변형 수: 4 (Original + Variant 1/2/3). task 마다 다를 수 있음.

### 4.2 (B) object-templated 모드 (기본)

- 표현 템플릿은 1개. 샘플된 object/fixture 이름으로 fill.
- 예: `juice_fruit_reamer.py:34` `f"Juice the {obj_lang} by pressing it against the reamer."`
- → instruction text 가 바뀌려면 **샘플된 object 가 바뀌어야** 함 = `ep_meta["object_cfgs"]` 가 다름.

### 4.3 한 task class 안의 모든 instruction 을 enumerate 하려면

- 모드 A: CSV row 직접 파싱 → 모든 변형을 강제로 한 번씩 사용하려면 `self.rng.choice` 대신 인덱스 주입 monkeypatch.
- 모드 B: N rollouts 돌려서 `unique(env.get_ep_meta()["lang"])` 모으기 (object 가 다 다르게 뽑힐 때까지).

---

## 5. `_ep_meta` 기반 재현 메커니즘 (RoboCasa 공식 방법)

### 5.1 `ep_meta` 의 정체

`Kitchen.get_ep_meta()` (`kitchen.py:1167-1214`) 가 dump 하는 dict. 다음 키를 포함:

| 키 | 의미 |
|----|------|
| `layout_id`, `style_id` | scene 식별자 |
| `object_cfgs` | 샘플된 각 object 의 cfg (카테고리, mesh path/info, scale, placement spec 등) |
| `fixtures`, `fixture_refs` | fixture 이름 / 클래스 / 참조 |
| `gen_textures` | generative textures 현재 선택값 |
| `init_robot_base_pos`, `init_robot_base_ori` | 로봇 base 시작 pose |
| `cam_configs` | 카메라 설정 |
| `lang` | 본 episode 의 instruction text |

### 5.2 우선순위 규칙: `set_ep_meta` 로 주입하면 rng 보다 우선

- 주입: `env.set_ep_meta(meta)` (`robosuite/base.py:404-410`)
- 다음 `env.reset()` 의 `_load_model()` 안에서 ep_meta 값이 **시드 샘플링을 대체**:
  - `kitchen.py:591` `if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:` → ep_meta 값 사용
  - `kitchen.py:854` `if "object_cfgs" in self._ep_meta:` → object 인스턴스/배치 그대로 복원
  - `kitchen.py:1118` `if "init_robot_base_pos" in self._ep_meta:` → 로봇 위치 복원
  - `kitchen.py:608` `self._curr_gen_fixtures = self._ep_meta.get("gen_textures")` → 텍스처 복원
- 결과: `env.rng` 상태와 무관하게 같은 scene/object/placement/robot pose 가 나옴.

### 5.3 한계

- task class 가 `get_ep_meta()` 에서 **추가로** `self.rng` 를 호출하면 (예: novel_instructions 의 `self.rng.choice`) 그 부분은 ep_meta 만으로는 결정 안 됨. 필요시 ep_meta dump 시점의 seed 도 같이 저장.
- `_setup_scene` 단계의 task-specific randomization (예: 싱크 핸들 상태) 도 마찬가지.

---

## 6. 다른 PC 에서 환경 재현 — 두 가지 전략

### 6.1 전략 1: Seed 기반 (간단, 취약)

**공유 항목:**
- env 생성 인자 dict (env_name, seed, layout_and_style_ids, obj_instance_split, …)
- rollout 인덱스 (또는 rollout 횟수)
- 코드 버전: robocasa / robosuite git commit SHA, mujoco / numpy / numba 버전, python 버전
- 에셋: `python -m robocasa.scripts.download_kitchen_assets` 산출물 (~10GB), objaverse / lightwheel / aigen 풀

**장점:** 공유 데이터가 config 몇 줄.
**단점:** RNG 소비 순서가 1 곳이라도 바뀌면 (lib 마이너 업데이트, fork branch 변경 등) 전체 시퀀스가 어긋남.

### 6.2 전략 2: ep_meta replay (확실, 권장)

**공유 항목:**
- 각 rollout 의 `ep_meta` JSON (= `env.get_ep_meta()` 결과)
- env 생성 인자 (camera spec, robot 종류 등 — 이건 ep_meta 에 안 들어감)
- 에셋 동일성 (object_cfgs 안 mesh path 가 valid 해야 함)
- robocasa / robosuite git commit (xml 파서 호환성)
- mujoco 버전 (물리 시뮬레이션 호환성)

**장점:** RNG 소비 순서 무관, 라이브러리 마이너 업데이트에 강함.
**단점:** ep_meta JSON 파일 N개 공유 필요.

> **권장: 전략 2.** 전략 1은 디버깅/sanity 용으로만.

---

## 7. 전체 워크플로우 (전략 2, instruction 하나당 1 rollout)

### 7.1 PC1 — instruction 수집 + ep_meta dump

```python
# scripts/eval/dump_ep_metas.py  (예시)
import json
from pathlib import Path
from scripts.path_setup import configure_repo_paths
configure_repo_paths(include_script_utils=True, include_robocasa=True)
from scripts.eval.robocasa_eval import create_eval_env

TASK = "TurnOnMicrowave"
SEED = 42
N_PROBE = 200          # 충분히 많이 돌려서 unique instruction 다 모음
USE_NOVEL = True       # novel_instructions CSV 변형도 모으고 싶으면

env = create_eval_env(env_name=TASK, seed=SEED)
if USE_NOVEL:
    # Kitchen.__init__ 에 use_novel_instructions=True 를 forward 하도록
    # create_eval_env 를 직접 수정하거나 별도 호출 경로 추가
    env.use_novel_instructions = True
    env._load_novel_instructions()

seen = {}                # lang → meta
for i in range(N_PROBE):
    env.reset()
    meta = env.get_ep_meta()
    lang = meta.get("lang", "") or ""
    if lang and lang not in seen:
        # JSON 직렬화를 위해 numpy / Fixture 객체 제거 (kitchen.py:1172 copy_dict_for_json 이 이미 처리)
        seen[lang] = meta
env.close()

out = Path("ep_metas") / f"{TASK}.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(
        {"task": TASK, "seed": SEED, "use_novel_instructions": USE_NOVEL, "episodes": seen},
        f, indent=2, default=str,
    )
print(f"dumped {len(seen)} unique instructions → {out}")
```

같이 dump 해야 할 메타:
- env 생성 인자 (camera_names, camera_widths, camera_heights, robots, controllers, layout_and_style_ids, obj_instance_split, randomize_cameras, generative_textures)
- `robocasa` 와 `robosuite` 의 git commit SHA (`git -C src/benchmarks/robocasa rev-parse HEAD`)
- `pip show mujoco numpy numba` 결과

### 7.2 PC2 — replay

```python
# scripts/eval/replay_ep_metas.py  (예시)
import json
from scripts.path_setup import configure_repo_paths
configure_repo_paths(include_script_utils=True, include_robocasa=True)
from scripts.eval.robocasa_eval import create_eval_env, run_vla_rollouts

bundle = json.load(open("ep_metas/TurnOnMicrowave.json"))
TASK, episodes = bundle["task"], bundle["episodes"]

for lang, meta in episodes.items():
    env = create_eval_env(env_name=TASK, seed=bundle["seed"])  # seed 는 폴백용
    env.set_ep_meta(meta)
    obs = env.reset()                                          # meta 가 layout/object/robot pose 덮어씀
    assert env.get_ep_meta()["lang"] == lang, "instruction 재현 실패 — task class 가 lang 을 rng 로 재샘플"
    run_one_rollout(env, ...)
    env.close()
```

`env.set_ep_meta(meta)` 이후 reset 1번이면 scene/object/robot 이 PC1 과 동일. instruction 이 task class 의 `get_ep_meta()` 에서 `self.rng.choice` 로 매번 새로 뽑히는 경우(`use_novel_instructions=True` 인 일부 task)는 ep_meta 의 `lang` 값을 다시 덮어쓰거나, seed 까지 같이 박아두기.

### 7.3 PC1 = PC2 검증

같은 ep_meta replay 후 첫 step 의 obs (이미지 제외) 가 byte 단위로 같은지 확인:

```python
# PC1, PC2 양쪽에서
import numpy as np, hashlib
obs = env.reset()
# robot qpos, object pose 등 결정적 state 만 비교 (이미지는 GPU 드라이버 차이로 다를 수 있음)
state = np.concatenate([env.sim.data.qpos, env.sim.data.qvel])
print(hashlib.md5(state.tobytes()).hexdigest())
```

해시가 일치하면 물리 상태까지 완전 재현.

---

## 8. 주의 / 알려진 한계

1. **렌더링은 완전 결정적이지 않다.** Mujoco 물리는 결정적이지만 OpenGL/EGL 렌더링은 GPU 모델·드라이버·시스템에 따라 픽셀 단위로 다를 수 있다. object pose 까지는 동일하지만 RGB 가 정확히 같은 비트는 아닐 수 있음.
2. **에셋 동일성이 전제.** `ep_meta["object_cfgs"]` 안에 mesh 파일 경로가 들어가므로 같은 fork + 같은 assets 다운로드 결과여야 valid.
3. **본 repo `--use-groot-env` 경로는 seed forwarding 패치 없이는 결정성 깨짐.** `scripts/eval/robocasa_eval.py:376` `gym.make(env_id, enable_render=True)` — seed kwarg 추가 필요.
4. **`gymnasium_basic.py:264` `np.random.seed(seed)` 는 트랩.** 글로벌 numpy RNG 만 시드. `env.rng` (PCG64) 와 무관하므로 `env.reset(seed=...)` 로는 환경 재시드 안 됨.
5. **task class 내 `_setup_scene` 추가 randomization 은 ep_meta 가 덮지 않는다.** 완전 재현 필요하면 seed 도 같이 박을 것.
6. **lerobot dataset replay 경로** (`robocasa/scripts/dataset_scripts/dataset_states_to_obs.py`) 는 HDF5 의 `ep_meta` attr + `states` qpos sequence 를 같이 사용. 정책 평가가 아니라 demo replay 용.

---

## 9. 빠른 체크리스트

PC1 → PC2 재현 시 확인 항목:

- [ ] robocasa git commit SHA 일치
- [ ] robosuite git commit SHA 일치
- [ ] `mujoco`, `numpy`, `numba` 버전 일치 (setup.py 와 동일)
- [ ] python 3.11 동일
- [ ] `download_kitchen_assets` 산출물 동일 (해시 비교 권장)
- [ ] objaverse / lightwheel / aigen 풀 동일
- [ ] `novel_instructions/task_instruction_variants.csv` 동일 (repo 포함이라 SHA 만 같으면 OK)
- [ ] env 생성 인자 dict 동일
- [ ] (전략 1) seed + rollout_idx 시퀀스 동일
- [ ] (전략 2) ep_meta JSON 공유 + `set_ep_meta` 후 reset
- [ ] (검증) `env.sim.data.qpos` 해시 일치

---

## 10. 관련 파일 빠른 참조

| 역할 | 경로 |
|------|------|
| Eval 진입점 | `scripts/eval/robocasa_eval.py` |
| Env factory (eval용) | `scripts/eval/robocasa_eval.py:63` `create_eval_env` |
| Kitchen env (rng 소비 지점) | `src/benchmarks/robocasa/robocasa/environments/kitchen/kitchen.py` |
| Robosuite base (rng/seed/ep_meta 정의) | `src/benchmarks/robosuite/robosuite/environments/base.py` |
| Robot spawn rng | `src/benchmarks/robocasa/robocasa/utils/env_utils.py:1531, 1602` |
| Placement sampler | `src/benchmarks/robocasa/robocasa/utils/env_utils.py:916, 1005` |
| novel_instructions CSV | `src/benchmarks/robocasa/robocasa/models/assets/novel_instructions/task_instruction_variants.csv` |
| Gym wrapper (GR00T path) | `src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py`, `gymnasium_groot.py` |
| ZMQ eval (GR00T) | `scripts/eval/groot_*_zmq_eval.py`, `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py` |
| SAFE collection 진입점 | `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py` |
| GR00T env factory (seed 미전달) | `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py:79-92` `get_robocasa_env_fn` |

---

## 11. GR00T-N1.6 SAFE collection 경로 (`collect_rollout.py`) — seed 비재현 분석

> 이 절은 기존 `docs/groot_n16_robocasa_env_reproduction.md` (GR00T 수집 재현 가이드) 를 대체한다.
> 그 문서는 "같은 `--seed` 면 같은 layout / object / instruction / robot pose" 라고 했으나,
> 이는 **코드와 런타임 양쪽에서 반증된다**. 아래에 수집 경로의 실제 seed 흐름, 반증 근거,
> 수정 방법을 정리한다. 검증 환경: `robocasa` Docker 컨테이너 (python 3.11.15, mujoco 3.3.1,
> numpy 2.2.5), robocasa365 fork.

### 11.0 용어: episode ≠ scenario (초기상태)

문서 전반에서 두 개념을 구분한다.

- **episode**: 한 번의 rollout 실행 단위. `episode_idx` 로 번호가 매겨지고, 성공/실패 label, hidden_states, video 가 episode 단위로 남는다.
- **scenario (초기상태)**: episode 시작 시 샘플된 scene 구성 — layout/style, object 인스턴스·배치, robot base pose, texture 등. "초기상태의 다양성" 은 이 scenario 가 episode 마다 달라지는 것을 가리킨다.

수집 경로에서는 episode 마다 **새 random scenario** 가 뽑힌다 (11.1–11.3). 따라서 "100 episode = 100 scenario" 는 맞지만, 그 scenario 들은 seed 로 **식별·재현되지 않는 무작위** scenario 다. 이는 eval 경로 (Pattern A, §3 — 한 seed 로 env 1개를 만들고 N reset 하면 `env.rng` 가 advance 하며 **결정적** scenario 시퀀스를 만드는 방식) 와 구분해야 한다. 기존 문서가 episode·seed·scenario 를 1:1 로 묶어 "seed 가 scenario 를 고정한다" 고 본 것이 혼동의 핵심이었다.

### 11.1 수집 경로의 seed 흐름

`scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`:

- `episode_seed = args.seed + local_ep_idx` (`:538`) — episode 마다 seed 증가, episode 당 **새 env** 생성 (`:390` `gym.vector.SyncVectorEnv([...])`, reset 1회).
- env 생성 체인: `get_gym_env` → `get_robocasa_env_fn` → `gym.make(env_name, enable_render=True)` (`rollout_policy.py:90`, **seed 인자 없음**) → `create_env_robosuite(seed=None)` → `robosuite.make(seed=None)` → `base.py:145 self.rng = np.random.default_rng(None)` (OS 엔트로피, **랜덤**).
- `episode_seed` 는 `env.reset(seed=episode_seed)` 로만 전달 → `gymnasium_basic.py:264 np.random.seed(episode_seed)` (**글로벌 legacy RNG 만**).

→ 즉 eval 경로 (`robocasa_eval.py:92` `robosuite.make(seed=args.seed)`, §2.4) 와 달리, **수집 경로는 생성 시 seed 를 전혀 안 넘긴다**. seed 는 글로벌 `np.random` 만 건드린다.

### 11.2 왜 seed 가 scene 을 결정하지 못하나 (메커니즘)

robocasa Kitchen 의 scene 샘플링은 전부 `self.rng` (전용 `Generator`) 를 쓴다 — `kitchen.py` 에서 `self.rng` 10회, `np.random.*` **0회**. layout/style (`:595`), object (`:614, 1620`), texture (`:1321-1338`), fixture (`:1709, 1745`), robot spawn·placement (`env_utils.py:916, 1005, 1531, 1602`) 모두 `self.rng`/`env.rng`. 따라서:

- `np.random.seed(episode_seed)` 는 **글로벌 RNG** 를 시드하지만, scene 은 **`self.rng`** 가 결정 → 무관.
- `self.rng = default_rng(None)` 은 생성 시 OS 엔트로피로 잡히고 reset 이 재시드하지 않는다 (`base.py:277` reset 은 rng 불변). 또한 `default_rng(None)` 은 글로벌 `np.random` 상태와 독립.

런타임 확인: `np.random.seed(100000)` 후 `default_rng(None).integers(...)` = `985028737`, 동일 seed 반복 시 `3355865` (다름). 반면 `default_rng(100000)` 명시는 `159181479` 로 재현.

### 11.3 런타임 반증 (robocasa Kitchen, `TurnOnMicrowave`)

수집 현행 의미 (construction seed=None + reset 직전 `np.random.seed(S)`) 로 같은 `S` 를 2회:

| 시행 | seed | layout/style | robot_base |
|------|------|--------------|------------|
| A1 | 100000 | 7 / 10 | (0.93, −3.05, 0) |
| A2 | 100000 (반복) | 6 / 9 | (3.59, −1.69, 0) |

→ **같은 seed 인데 layout 7/10 ↔ 6/9 로 다름.** `env.rng` 초기상태 해시도 A1·A2 가 상이 (생성마다 랜덤).

### 11.4 기존 doc 의 주장 vs 사실

| `groot_n16_robocasa_env_reproduction.md` 주장 | 사실 |
|------|------|
| §1.2 "np.random.seed → robocasa 의 np.random.choice 가 결정적 (layout/object/instruction)" | robocasa 는 scene 에 `np.random.choice` 를 쓰지 않음 (`kitchen.py` 0회). `self.rng` 만 사용 → `np.random.seed` 와 무관 |
| §2 "NumPy global RNG 가 fix → layout/object/placement/robot/instruction 결정" | 전부 `self.rng` (별도 Generator) → 글로벌 fix 와 무관 |
| §4 "같은 seed → 같은 layout/object/robot pose, 다른 seed → 다른 scene" | 같은 seed 에서도 scene 이 달라짐 (11.3) |
| §6 "Layout-level 재현 = code SHA + seed" | seed 로는 layout 조차 재현 불가 |

오해의 출처: `gymnasium_basic.py:264` 에 `np.random.seed(seed)` 가 실제로 있어서, "글로벌만 시드하니 robocasa scene 도 글로벌 RNG 를 쓸 것" 으로 추정한 것으로 보인다. 그러나 robocasa/robosuite 는 dedicated `self.rng` 를 쓰므로 이 호출은 scene 에 대해 no-op 다.

### 11.5 수정 방법 (미래 수집을 재현 가능하게)

1. **construction seed forwarding (권장, 검증됨)** — seed 를 env **생성** 까지 전달하면 `env.rng = np.random.default_rng(S)` 가 되어 결정적이 된다. `RoboCasaEnv.__init__` 은 `**kwargs` 를 `create_env_robosuite(seed=...)` → `robosuite.make(seed=...)` 로 forward 하므로 (`gymnasium_basic.py:134-143`), **`gym.make(env_id, seed=S)` 한 줄로 충분** 하다. 수집 경로의 gap 은 `get_robocasa_env_fn` (`rollout_policy.py:90`) 이 `gym.make(env_name, enable_render=True)` 로 seed 를 빼고, `collect_rollout.py` 의 `_run_single_rollout` 가 seed 를 `env.reset(seed=)` 로만 넘긴다는 점뿐이다. eval 경로 (`robocasa_eval.py:92`) 는 이미 생성 시 seed 를 넘긴다.
   - **end-to-end 검증** (`scripts/eval/test_robocasa_env_reproducibility.py`, 수집과 동일한 GR00T gym env 경로): `gym.make("robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env", seed=100000)` 를 2회 → `qpos+qvel` md5 (`2d649621...`) 와 3개 카메라 이미지 md5 가 **모두 일치**, layout/style 4/4 동일. 즉 같은 seed → **bit-identical scenario**. (대조: 생성 시 seed 를 안 넘기면 11.3 처럼 같은 seed 라도 layout 7/10 ↔ 6/9 로 갈림.)
2. **ep_meta dump (가장 robust)** — episode 마다 `env.get_ep_meta()` 를 pkl 에 저장하고, replay 시 `set_ep_meta` + reset (§5 참조). RNG 소비 순서·라이브러리 마이너 버전 변화에 강함 (robocasa 공식 방법).
3. 둘 다: seed 로 빠른 sanity + ep_meta 로 확실한 replay.

### 11.6 기존 1800-rollout collection 재현 가능성 → **불가**

`target_atomic_seen18_ckpt120000_robocasa365_100ep` (2026-05-27 수집) 은 **생성 시 seed 가 전달되지 않아 재현 불가** 하다 (seed 가 scene 에 무력, 11.1–11.3; pkl 에 ep_meta/qpos 도 없음). 단, 수집된 raw rollout / latent feature 자체는 유효한 producer artifact 다. 재현 가능한 데이터가 필요하면 11.5 의 fix 를 적용해 재수집한다.
