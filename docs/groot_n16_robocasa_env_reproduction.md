# GR00T-N1.6 × RoboCasa: 환경 재현 가이드

다른 PC에서 동일한 RoboCasa rollout을 재현하기 위해 알아야 할 randomness source,
파일/변수의 역할, 그리고 검증 방법을 정리한다.

## 0. TL;DR

같은 PC에서든 다른 PC에서든 **bit-perfect 재현** (저장된 pkl 의 hidden_state까지
정확히 동일) 을 보장하려면:

1. **코드 버전 일치**: `src/policies/Isaac-GR00T`, `src/benchmarks/robocasa`,
   `src/benchmarks/robosuite` 모두 동일 commit SHA
2. **`--seed` 값 일치**: `collect_rollout.py --seed N` 에서 N 동일
3. **task_id × episode_start_idx 일치**: 같은 task 의 같은 episode 인덱스
4. **layout/style 설정 일치**: `gymnasium_basic.py:77`의
   `layout_and_style_ids` 가 코드에 하드코딩이라 commit이 같으면 자동 동일
5. **MuJoCo / robosuite / robocasa asset 버전 일치**:
   `pip install` 결과의 `__version__` 가 같아야 함 (특히 robocasa: 우리는
   `hanyangtae/robocasa` fork 사용)
6. **모델 추론 결과 동일**: GR00T checkpoint + GPU determinism은 별개 이슈
   (§6 참조). 환경만 재현하면 obs는 같지만, model이 다른 GPU/precision이면
   action이 미세 변동 가능 → action 변동이 다시 env 다음 obs를 미세하게
   바꿈 (compounding).

bit-perfect 가 아닌 **"기능적으로 같은 시뮬레이션"** (같은 layout, 같은 object,
같은 시작 robot pose) 만 원하면 1~5만 맞춰도 충분.

## 1. Randomness source 와 통제 변수

### 1.1 `collect_rollout.py` 에서 시작

`scripts/safe/groot_n16/robocasa/collect/collect_rollout.py:478-510`:

```python
parser.add_argument("--seed", type=int, default=None)  # 한 episode 의 seed 결정
...
for local_ep_idx in range(args.n_episodes):
    episode_idx = args.episode_start_idx + local_ep_idx
    episode_seed = None if args.seed is None else args.seed + local_ep_idx
    ...
    _run_single_rollout(env_name=..., seed=episode_seed)
```

- `--seed`: 정수. `None` 이면 episode마다 매번 다른 randomness.
- `episode_seed = args.seed + local_ep_idx`: `-n-episodes K` 로 batch 수집 시
  episode 0, 1, 2 가 각각 `seed, seed+1, seed+2` 사용.
- 우리 production batch script (`run_seen5_5task_*.sh`) 도 동일 규칙:
  ```bash
  SEED_START=241  # default
  seed=$((SEED_START + local_episode_idx))
  ```
  → 같은 `SEED_START` + 같은 `EPISODES_PER_TASK` 면 다른 PC에서도 같은
  seed 시퀀스 생성.

### 1.2 seed가 env.reset 으로 어떻게 흘러가는가

호출 체인:

```
collect_rollout.py:367
  └─ env.reset(seed=episode_seed)
       └─ MultiStepWrapper.reset                    # src/policies/Isaac-GR00T/gr00t/eval/sim/wrapper/multistep_wrapper.py:233
            └─ super().reset(seed=seed)             # → SafeVideoObservationFilter / VideoRecordingWrapper / GrootRoboCasaEnv
                 └─ GrootRoboCasaEnv.reset          # src/benchmarks/robocasa/.../gymnasium_groot.py:119
                      └─ RoboCasaEnv.reset          # src/benchmarks/robocasa/.../gymnasium_basic.py:263
                           ├─ np.random.seed(seed)  # ← Python NumPy global RNG
                           └─ self.env.reset()      # robosuite env. robosuite 도 자기 seed 가짐
```

**중요한 함의**:

- `np.random.seed(seed)` 는 NumPy의 **global RNG**을 reset. 그래서 robocasa
  내부에서 `np.random.choice(...)` 같은 호출이 결정적이 됨 (layout 선택, object
  pose sampling, instruction template 선택 등 모두 NumPy global 을 통함).
- robosuite 내부 RNG는 별도. `create_env_robosuite` 호출 시 `seed=seed` 가
  넘어가는데 (gymnasium_basic.py:41), 이건 `robosuite.make` 의 init seed임
  (env이 처음 만들어질 때 1회). 매 episode reset 마다는 NumPy global 만 다시 됨.
- 결과적으로 같은 `--seed`로 reset하면 **layout / object 위치 / instruction /
  robot 초기 pose 모두 동일**.

### 1.3 하드코딩된 통제 변수 (commit-locked)

`src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py:73-85`:

```python
if issubclass(env_class, Kitchen):
    env_kwargs.update({
        "layout_ids": None,
        "style_ids": None,
        "layout_and_style_ids": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
        "obj_instance_split": "target",
        "generative_textures": None,
        "randomize_cameras": False,
    })
```

| 변수 | 역할 | 변경 시 영향 |
|---|---|---|
| `layout_and_style_ids` | 5개 (layout, style) 쌍에서 NumPy random으로 1개 선택 | 후보 풀이 바뀌면 layout 시퀀스 바뀜 → **모든 데이터 무효** |
| `obj_instance_split` | object 풀을 "pretrain" / "target" 중 어느 절반에서 sampling | "target" 으로 split된 object asset 시퀀스 결정 |
| `randomize_cameras=False` | 카메라 위치 고정 | True 가 되면 매 reset 마다 카메라 변동 |
| `generative_textures=None` | procedural texture off | 켜면 매 reset 마다 텍스쳐 변동 |

다른 PC에서 위 값이 같은 commit 의 코드로 실행되면 자동 동일.
**hot-fix로 위 값을 바꾸면 commit에 반영해야 다른 PC에서 동일 결과**.

### 1.4 카메라 / 컨트롤러 설정

- camera_names / widths / heights: `make_key_converter(robots_name)` 에서 robot별
  기본값 가져옴. PandaOmron 의 경우 `gather_robot_observations` + `key_converter`
  가 일관되게 처리.
- controller_configs: `load_composite_controller_config(robot="PandaOmron")` 기본값
  사용. 코드 변경 없이는 동일.
- `MUJOCO_GL=egl`: rendering backend. **EGL 사용 못하는 host (예: headless 없이
  GUI만 있는 PC)** 에선 `MUJOCO_GL=osmesa` 등으로 바꿔야 함. 이건 rendering
  픽셀 비트만 미세 차이 (보통 의미 없음).

### 1.5 다른 PC에서 명시적으로 맞춰야 할 외부 요인

| 요인 | 통제 방법 |
|---|---|
| MuJoCo 버전 | `pip show mujoco` 비교; 우리는 robocasa fork가 의존하는 버전 그대로 |
| robosuite 버전 | submodule SHA 동일 (`git submodule status src/benchmarks/robosuite`) |
| robocasa 버전 | submodule SHA 동일 (우리 fork: `hanyangtae/robocasa`) |
| NumPy 버전 | 동일 major 권장 (1.x ↔ 2.x 사이는 RNG bit-level이 다를 수 있음) |
| Python 버전 | 우리 robocasa 컨테이너는 3.11 |
| OS / glibc | 통상 동일 Linux면 OK |

권장: **Docker 이미지로 환경 동결**. 우리는 `robocasa` 컨테이너로 띄움
(`run_seen5_5task_*.sh:67-70`에서 `docker exec robocasa ...`).

## 2. Episode 단위에서 무엇이 결정되는가

`--seed S` 로 reset하면 NumPy global RNG가 fix되고, 그 결과 결정되는 것:

1. **Layout/style**: 5개 후보 (`[[1,1], [2,2], [4,4], [6,9], [7,10]]`)에서 1개 선택
2. **Object instance**: target split 안에서 어떤 mug / 어떤 candle / etc.
3. **Object 초기 pose**: counter 위 어디에 놓일지 (placement initializer 가
   NumPy random 으로 sampling)
4. **Robot 초기 pose**: PandaOmron base 위치 + arm joint 초기값
5. **Instruction**: 같은 task여도 여러 instruction template 중 NumPy random 선택
   (예: PnPSinkToCounter 100ep에 instruction 29종 — §0번 manifest 참조)
6. **MuJoCo simulator 내부 stochastic 요소**: 거의 없음 (deterministic physics)

같은 seed → 위 1~5 모두 동일 → **동일 시작 obs**.

같은 시작 obs + 동일 policy → 동일 action chunk → 동일 next obs → 동일 trajectory.

## 3. 다른 PC에서 동일 rollout 만들기 — 체크리스트

### 단계 1: 코드 동기화

```bash
# repo root에서
git fetch --all
git checkout <commit-SHA>      # 데이터 수집했던 시점의 SHA
git submodule update --init --recursive

# submodule SHA 비교
git submodule status src/policies/Isaac-GR00T
git submodule status src/benchmarks/robocasa
git submodule status src/benchmarks/robosuite
```

`pkl` 파일에 저장되어 있지 않은 정보지만, 우리 `safe_train_logs/.../config.yaml`
의 SAFE 학습 시점에는 같은 코드 SHA 였음. 새 PC에서 commit SHA가 다르면
무효.

### 단계 2: Docker 컨테이너 동일하게 띄우기

```bash
# docker-compose.yml 에 robocasa 서비스 정의 있음. 같은 이미지로 띄움
docker compose up -d robocasa
```

`MUJOCO_GL=egl`, `PYTHONPATH` 등은 `run_seen5_5task_host.sh:67-70` 에 명시.

### 단계 3: GR00T policy server 동일하게 띄우기

```bash
# 같은 checkpoint
ls outputs/checkpoints/GR00T-N1.6-3B
# 같은 dtype / 같은 inference 옵션으로 ZMQ server 기동
python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
    --feature-slice valid --feature-dtype float16 ...
```

⚠️ **모델 추론에 GPU 결정성 이슈**: 다른 GPU 모델 (예: A100 ↔ RTX 4090) 사이에는
부동소수 연산 순서가 살짝 달라서 hidden_state bit-level 이 미세하게 다를 수
있음. action 자체는 거의 같지만 누적되면 trajectory 미세 차이 가능.
*같은 GPU 모델*에서는 보통 결정적.

### 단계 4: 동일 seed / episode_idx 로 수집

```bash
# 우리 production 설정과 동일
SEED_START=241 EPISODE_START_IDX=0 EPISODES_PER_TASK=100 \
RUN_ID=seen5_5task_repro \
bash scripts/safe/groot_n16/robocasa/collect/run_seen5_5task_host.sh
```

`SEED_START` 와 `EPISODE_START_IDX`만 같으면 task별로 동일 seed 시퀀스
(`241, 242, ..., 340`) 생성.

### 단계 5: 검증

원본 rollout과 새 rollout 비교:

```bash
python - <<'PY'
import pickle, numpy as np
a = pickle.load(open("outputs/.../rollouts_orig/CoffeeSetupMug/task0--ep0--succ1.pkl", "rb"))
b = pickle.load(open("outputs/.../rollouts_repro/CoffeeSetupMug/task0--ep0--succ1.pkl", "rb"))

# 메타 비교
print("seed/task_description match:", a["task_description"] == b["task_description"])
print("episode_success match:", a["episode_success"] == b["episode_success"])
print("T match:", len(a["hidden_states"]) == len(b["hidden_states"]))

# hidden_state 비교 (모델 추론까지 결정적인 경우만 의미 있음)
for t in range(min(len(a["hidden_states"]), len(b["hidden_states"]))):
    ha = a["hidden_states"][t].numpy()
    hb = b["hidden_states"][t].numpy()
    if not np.allclose(ha, hb, atol=1e-3):
        print(f"  diverge at t={t}, max diff = {np.max(np.abs(ha-hb)):.4f}")
        break
else:
    print("hidden_states bit-close")
PY
```

기대 결과:
- **task_description, episode_success, T**: 항상 동일 (코드 + seed 동일 시)
- **hidden_states**: 같은 GPU 모델에서는 동일. 다른 GPU에서는 미세 차이 가능

## 4. Seed를 바꾸면 무엇이 달라지는가

| 같은 seed | 다른 seed |
|---|---|
| layout / style 동일 | 다른 layout / style (5개 중) |
| 같은 object | 다른 object (target split 내 다른 instance) |
| 같은 object 초기 위치 | 다른 위치 |
| 같은 instruction text | 다른 instruction template |
| 같은 robot 초기 pose | 다른 robot pose |

따라서 "동일 시나리오 다른 policy 비교" 를 하고 싶으면 같은 seed로 다른 policy
서버 띄워 수집하면 됨. "policy 강건성 평가" 를 하고 싶으면 seed를 일부러 다르게
range로 쓸어서 수집.

우리 production 설정 (`SEED_START=241, EPISODES_PER_TASK=100`) 는 task당
seed `241~340` 의 100개 시드로 100 episode 수집 → policy 성능을 100개의
서로 다른 시나리오로 평가.

## 5. 흔한 함정

### 5.1 `--seed` 를 안 주면 (None)

`episode_seed = None` 이 `env.reset(seed=None)` 으로 들어가면
`np.random.seed(None)` → 시스템 entropy 로 seed → **매 실행마다 다른 결과**.
재현 불가. 명시적으로 정수 seed 주는 것이 필수.

### 5.2 `random.random()` 사용 코드

NumPy global RNG는 reset하지만 Python `random` 모듈의 global RNG는 별도.
robocasa/robosuite 코드 안에 `random.random()` 호출이 있으면 그건
non-deterministic. 우리가 사용하는 task들 (CoffeeSetupMug, OpenSingleDoor 등)
에서는 큰 영향 없는 것으로 보임. 의심 시 `random.seed(seed)` 도 같이 호출하는
래퍼 추가.

### 5.3 `terminate_on_success=True` 의 영향

`collect_rollout.py:497` 에서 `MultiStepConfig(terminate_on_success=True)` 설정.
즉 task success 신호 즉시 episode 종료. 다른 PC에서 성공 시점이 미세하게 달라지면
hidden_states 길이 T가 다를 수 있음. 같은 GPU/같은 코드면 보통 동일.

### 5.4 layout_and_style_ids 를 코드에서 수정한 경우

이건 그냥 `git diff` 하면 잡힘. 다만 production rollout과 다른 후보 풀로
새 PC에서 돌리면 *언뜻 보면* 비슷한데 실제 layout이 다른 함정 발생 가능.
배포 전 commit clean 필수.

### 5.5 EGL 없는 PC

`MUJOCO_GL=egl` 가 동작 안 함. `osmesa` 또는 `glfw` 로 바꿔야 하는데, 둘 다
rendering pixel이 미세 다름. obs (이미지) 가 다르면 model 입력이 달라져
trajectory가 분기. **headless GPU server 권장**.

## 6. 모델 + 환경 vs 환경만 — 단계별 재현

| 재현 수준 | 필요한 것 | 보장되는 것 |
|---|---|---|
| **Layout-level** | code SHA + seed | layout/object/instruction 동일, robot 초기 pose 동일 |
| **Trajectory-level** | + 동일 GR00T checkpoint + 같은 GPU 모델 | 매 inference action 거의 동일, T 동일, hidden_states bit-close |
| **Bit-perfect** | + 동일 CUDA/cuDNN, 동일 random seed in PyTorch | hidden_states 완전 동일 |

대부분의 경우 **Trajectory-level**로 충분 (silhouette, t-SNE 등 분석 결과는
동일하게 나옴). bit-perfect 가 필요한 건 매우 드뭄.

## 7. 참고 source

| 항목 | 경로 |
|---|---|
| `--seed` 인자 처리 | `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py:478,489,510` |
| env.reset → np.random.seed | `src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py:263-265` |
| layout_and_style_ids 하드코딩 | `src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py:73-85` |
| GR00T env factory (MUJOCO_GL=egl) | `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py:79-92` |
| robocasa env 등록 | `src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_groot.py:139-172` |
| Production batch (seed 시퀀스) | `scripts/safe/groot_n16/robocasa/collect/run_seen5_5task_host.sh:13,53` |
| control_freq=20 (kitchen) | `src/benchmarks/robocasa/robocasa/environments/kitchen/kitchen.py:381` |
