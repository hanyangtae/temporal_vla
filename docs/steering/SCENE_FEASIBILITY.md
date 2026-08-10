# Scene 실현가능성 필터 — fixture task seed는 물리적으로 불가능한 scene을 포함한다

**fixture를 조작하는 task에서 rollout을 수집하기 전에 seed를 걸러야 한다.** 특정 라운드용
통지가 아니라 상시 규약이다. 새 task로 확장할 때마다 §5의 이식 절차로 한 번은 확인해야 한다.

## 1. 왜 걸러야 하나

OpenStandMixerHead 스모크에서 seed 100010은 "정책이 못 여는" 게 아니라 **믹서 머리가 위쪽
선반에 물리적으로 막혀서** 열 공간 자체가 없는 scene이었다. 라벨러 phase로 보면
contact-head ↔ lift-open ↔ push-down이 진동해 "재시도 실패"처럼 보인다.

이런 판이 succ/fail 대조에 섞이면 두 가지로 해롭다.

1. **failure 클래스가 "실패 방향" 대신 "scene 기하"를 학습한다** — exp3까지 우리를 괴롭힌
   scene confound와 같은 계열.
2. **어떤 steering으로도 구제 불가능한 판**이라 rescue 실험의 분모를 조용히 갉아먹는다.

## 2. 걸러내는 방법 (정책 무관 · CPU only · seed당 수 초)

`scripts/safe/groot_n15/robocasa/analyze/mixer_scene_feasibility.py`

reset 직후 로봇은 건드리지 않고 **대상 관절만 0→1.0으로 스윕**하면서, 움직이는 body의 geom이
(그 fixture 자신·로봇 이외의) 외부 geom과 접촉하는 지점을 찾는다. 접촉 없이 도달 가능한
최대 정규화값 = `q_max_feasible`.

```
q_max_feasible < 성공임계  →  그 seed는 어떤 정책으로도 성공 불가 → 제외
```

실행 (robocasa 컨테이너):

```bash
docker exec -e MUJOCO_GL=egl \
  -e PYTHONPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla:/temporal_vla/scripts/safe/groot_n16/robocasa/collect" \
  robocasa python /temporal_vla/scripts/safe/groot_n15/robocasa/analyze/mixer_scene_feasibility.py \
  --seeds 100000-100099 --out /temporal_vla/outputs/.../scene_feasibility.json
```

**seed당 fresh 프로세스 필수.** 한 프로세스에서 `gym.make`를 연속 호출하면 두 번째부터 scene이
오염된다(실증: 구 `18_apple_success_rejudge` §2 → 현 [`RESULTS.md`](RESULTS.md)). 스크립트는 이미 그렇게 되어 있다.

## 3. 왜 이 필터는 편향을 만들지 않나

- 정책·체크포인트·steering arm·chunk 설정과 무관한 **seed만의 함수**다.
- 따라서 base/perm/gated 모든 arm에 같은 seed 집합이 적용된다 → arm 간 비교 불변.
- **제외는 fit·eval 양쪽 모두에서 동일하게** 해야 한다. 한쪽만 걸면 그게 새 confound다.
- 제외 seed 목록과 `q_max` 값을 manifest/NPZ meta에 기록해 사후 감사 가능하게 할 것.

## 4. 실측 (OpenStandMixerHead, seed 100000-100011)

| seed | q_max | 판정 |
|---|---|---|
| 100000~100009, 100011 | 1.000 | OK |
| **100010** | **0.575** | **BLOCKED** (`shelves_left_group_level0_shelf`) |

12개 중 1개(8.3%) 불가능. 100000-100099 전체 스캔은 미실시.

교차검증: seed 100010 rollout의 실제 최대 head 값 = 0.596, 기하 상한 0.575와 사실상 일치 →
정책은 천장까지 밀어붙였고 더 갈 데가 없었다. (상한을 살짝 넘는 건 접촉 후 밀어내는 탄성 변형.)

## 5. 다른 task로 확장할 때

스크립트는 현재 **StandMixer 전용**(`env.stand_mixer`, `_joint_names["head"]`)이다.
이식하려면 네 개만 바꾸면 된다: ① fixture 참조 ② 관절 이름 ③ 움직이는 body 이름 ④ 성공임계.

| task | 관절 | 임계 |
|---|---|---|
| CloseFridge | `fxtr._fridge_door_joint_names` (도어별, 다짝 전부) | ≤0.005 |
| OpenDrawer | `env.drawer` 관절 | ≥0.90 계열 |

**미확인 사항 두 개** — 새 task를 쓸 때마다 확인 대상이다:

- **drawer / fridge에도 같은 오염이 있는지 아직 확인 안 했다.**
- **exp2·exp3의 drawer / ppcc 결과에 이 오염이 얼마나 섞였는지 미확인.** 사후 스캔 가치 있음.

## 6. 한계

- **정적·관절만** 검사한다. 그리퍼가 fixture를 잡은 상태로 움직이다 그리퍼 자체가 선반에
  걸리는 경우는 못 잡는다(로봇 geom을 접촉 판정에서 제외 — reset 시 로봇이 근처면 오탐).
  필요하면 "그리퍼 포함" 변형을 플래그로 추가.

## 7. 관련

- `27_standmixer_phase_labeler.md` — mixer instruction/판정/라벨러/스모크
- chunk 길이 함정: CloseFridge는 실행 5/예측 16에서 SR 0/14, 실행 16에서 10/14.
  StandMixer는 실행 5에서도 2/6으로 살아남는다 — mixer를 고른 이유 중 하나.
