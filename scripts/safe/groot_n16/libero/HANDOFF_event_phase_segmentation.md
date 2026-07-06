# Handoff: BDDL phase labeler → event-anchored segmentation 전환

다른 세션에서 이 작업을 이어받기 위한 자족 문서. 코드 작성 전 단계이며, "왜 / 무엇으로 / 어떻게"
를 합의된 결정까지 포함해 정리한다.

---

## 0. 한 줄 요약

현재 LIBERO phase 라벨러는 **BDDL goal subgoal 만족 개수(running_max)** 로 phase 를 매겨서
너무 coarse 하다. 이를 **event-anchored segmentation**(grasp/release/in/close 등 sim 내부
이벤트를 경계로, 그 사이를 reach/transport phase 로 채우는 방식) 으로 바꾼다.

---

## 1. 현재 상태 (바꾸기 전)

- **대상 파일**: `scripts/safe/groot_n16/libero/bddl_phase_labeler.py`
  - 테스트: `tests/test_bddl_phase_labeler.py`
- **현재 로직**: `phase_t = running_max(#만족된 BDDL goal subgoal)`.
  - `goal_state` = BDDL `:goal` 의 `And` 항 = **터미널 subgoal 만** (예: SCENE6 = `[In(mug,microwave), Close(microwave)]`).
  - mujoco/torch 없이 도메인의 `_eval_predicate(state)` 만 호출 → mock domain 단위테스트 가능 (현재 장점).
- **소비처**: `scripts/eval/libero.py`
  - 수집 모드(`collector != None`)일 때만 `BddlPhaseLabeler(env)` 부착.
  - 매 env-step `labeler.step()` 호출, latent 발화 step 마다 `labeler.max_phase` 를 `feature_phases` 에 적재(hidden_states 와 1:1).
  - pkl payload 에 `phase_timeline`(per env-step) / `feature_phases`(per latent) / `max_phase` 동봉.
- **task 셋**: `libero_10` (10 tasks). BDDL: `src/benchmarks/LIBERO/libero/libero/bddl_files/libero_10/`.
- **연구 목적**: phase 별 succ/fail latent **분리 분석** + **phase-matched / phase-selective steering**
  의 입력. (메인 방향 = pathway-resolved + phase-matched activation steering; 단일출처
  `docs/steering/14_pathway_phase_online_steering.md`.)

---

## 2. 왜 바꾸나 (현재 coarse phase 의 한계)

- SCENE6 = phase 3개(배치전 / In / Close)뿐. **"배치전" 하나에 reach + grasp + transport + insert
  가 전부 뭉침** → grasp 실패 vs insert 실패(서로 다른 motor 실패) 구분 불가.
- pathway 분리 관점: VL(goal "what")은 goal-predicate 입자도로 충분하지만, **DiT(motor "how")
  steering 라우팅엔 모터 이벤트 입자도가 필요**.
- **교환가능(set-completion) task** 는 phase 번호가 특정 subtask 에 대응 안 함 → 이질적 pooling.
- **단일 subgoal task**(STUDY_SCENE1)는 phase-local outcome == global outcome → **길이 confound
  통제 실패** (실패=timeout 길이 아티팩트가 그대로 새어 들어옴).

---

## 3. 새 방식: event-anchored segmentation

### 3.1 핵심 아이디어
이벤트(이산 sim 상태 변화)가 **phase 경계**를 찍고, 그 **사이 구간(이동 중)** 은
reach/transport phase 로 자동으로 채워진다 → action timeline 전체가 빈틈없이 phase 로 분절.

- 이벤트 시점은 **episode 마다 실제 일어난 시점에 anchor** (고정 시간격자 아님).
  episode A 의 grasp=t7, B 의 grasp=t15 → reach 길이도 달라짐.
- cross-episode 정렬은 **timestep 이 아니라 phase-label(=event type) 로** → 빠른/느린 episode
  공정 비교 → **길이 confound 자동 제거** (현재 라벨러가 못 막던 부분).

### 3.2 이벤트 primitive 라이브러리 (재사용 검출기)
task 마다 라벨 코드를 새로 짜지 않는다. 아래 검출기 묶음을 한 번 구현하고, task 는 **순서 선언만** 한다.

| primitive | 내부 신호 정의 | 의미 |
|---|---|---|
| `grasped(obj)` | `env._check_grasp(gripper, obj_geoms)` **+ `Up(obj)`(들림 확인)** | 물체 집은 순간 |
| `released(obj, region)` | grasp 종료 AND `In/On(obj, region)` | 놓은 순간 |
| `in_region(obj, region)` | `In(obj, region)` (BDDL) | 안에 들어간 순간 |
| `on_region(obj, region)` | `On(obj, region)` (BDDL) | 위에 올라간 순간 |
| `contact(a, b)` | `check_contact(sim, a_geoms, b_geoms)` | 두 물체 닿은 순간 |
| `actuated(art, kind)` | `Open/Close/TurnOn(art)` predicate 전이 | 문 닫힘/스토브 켜짐 등 |

### 3.3 gap(이동 구간) 라벨링
gap 이름은 "reach" 고정이 아니라 **양옆 이벤트가 결정**한다:
- grasp **이전** gap = `reach-to-object` (빈손 접근)
- grasp ~ place **사이** gap = `transport` (물체 운반)
- release **이후** ~ actuate 사이 gap = `reach-to-door` 등 (빈손 접근)

---

## 4. 이미 합의된 구현 결정 (이 세션에서 확정)

1. **±n 창 방식 폐기 → segment-between-events 채택.** 경계=이벤트, phase=두 이벤트 사이 구간.
   - 이유: ±n 창은 (a) `n` 하이퍼파라미터가 물리적 근거 없고 (b) 인접 이벤트(insert↔release)에서
     **창이 겹쳐** step 소속 모호. segment 방식은 셋 다 해결(모든 step 유일 배정).
2. **이벤트 검출은 robosuite env 내부 신호 사용** — 순수 BDDL `_eval_predicate` 를 벗어남.
   - `env._check_grasp(gripper=env.robots[0].gripper, object_geoms=...)`,
     `check_contact(env.sim, geoms_1, geoms_2)`, + 기존 BDDL `In/On/Close/TurnOn`.
   - **gripper 는 BDDL `parsed_problem` object set 에 없음** → 라벨러가 **live robosuite env** 필요.
     현재의 mock 단위테스트 장점 일부 상실 → 검출 로직(이벤트→phase 변환)을 env 접근과 **분리**해
     테스트 가능하게 설계할 것.
3. **grasp 정의** = `_check_grasp`(양 finger geom 접촉) **+ `Up(obj)`(물체 따라 들림)**. raw 첫 접촉
   금지(스침 오탐). 접촉 flicker **debounce** 필요.
4. **close anchor 는 접촉이 아니라 `Close()` predicate=true** 기준(더 robust). 문 접촉은
   `reach-to-door` 끝 경계로만 사용.
5. **실패 robust 처리**: 이벤트를 rank 로 매기고 **running-max**(현재 라벨러 철학 유지).
   - 빠진 이벤트 = "해당 phase 미도달"(예: grasp 없음 → timeline 전체 `reach-to-object`,
     = "reach 에서 stall"). 반복 이벤트(집었다 떨어뜨림) = regression 무시.
   - 이게 phase-local outcome(advanced vs stalled) 및 실패 onset regime 분석과 직결.
6. **cross-episode 정렬 = phase-label(event type) 단위** (timestep 아님).

---

## 5. task 구조 3분류 + 각 task goal_state

**하나의 고정 phase 리스트로는 불가.** task 구조가 3종류로 갈리고, 특히 교환가능 task 가 핵심 난제.

| 구조 | task | event 순서 / 정렬 방식 |
|---|---|---|
| **순차 chain** (pick-place → actuate) | SCENE4, SCENE6 | 전역 phase 번호 = subtask. 아래 SCENE6 예시 그대로 |
| **교환가능 set-completion** (독립 2~3개, 임의 순서) | LIVING_ROOM 1/2/2/5/6, KITCHEN_SCENE8, KITCHEN_SCENE3 | ⚠️ **전역 순서 없음**. `(object, sub-event)` 쌍으로 라벨 + **event-type 으로** cross-episode 정렬 (전역 phase 번호 금지) |
| **단일** | STUDY_SCENE1 | `reach→grasp→transport→release` 한 줄 |

### 10개 task goal_state (BDDL `:goal`, 순서 그대로)

| task | goal subgoals | init 주의 |
|---|---|---|
| KITCHEN_SCENE3 | `Turnon(flat_stove_1)`, `On(moka_pot_1, stove_cook_region)` | — |
| KITCHEN_SCENE4 | `Close(white_cabinet_1_bottom_region)`, `In(akita_black_bowl_1, ...bottom_region)` | 서랍 **Open 시작** |
| KITCHEN_SCENE6 | `In(white_yellow_mug_1, microwave_heating_region)`, `Close(microwave_1)` | 전자레인지 **Open 시작** (문 열기 없음) |
| KITCHEN_SCENE8 | `On(moka_pot_1, cook)`, `On(moka_pot_2, cook)`, `Turnon(flat_stove_1)` | 3 subgoal 교환가능 |
| LIVING_ROOM_SCENE1 | `In(alphabet_soup_1, basket)`, `In(cream_cheese_1, basket)` | 교환가능 |
| LIVING_ROOM_SCENE2(a) | `In(alphabet_soup_1, basket)`, `In(tomato_sauce_1, basket)` | 교환가능 |
| LIVING_ROOM_SCENE2(b) | `In(cream_cheese_1, basket)`, `In(butter_1, basket)` | 교환가능 |
| LIVING_ROOM_SCENE5 | `On(porcelain_mug_1, plate_1)`, `On(white_yellow_mug_1, plate_2)` | 교환가능 |
| LIVING_ROOM_SCENE6 | `On(porcelain_mug_1, plate_1)`, `On(chocolate_pudding_1, plate_right_region)` | 교환가능 |
| STUDY_SCENE1 | `In(black_book_1, desk_caddy_1_back_contain_region)` | **단일 subgoal** (degenerate) |

### SCENE6 구체 분절 (레퍼런스)
```
reach-to-mug → grasp(mug) → transport → in_region(mug,microwave)
            → release(mug@microwave) → reach-to-door → close(Close(microwave))
```
SCENE4 는 object/region 만 바꾼 동형: `bowl` / `drawer_bottom`.

---

## 6. 검증·참조 경로 (새 세션이 바로 찾을 위치)

- BDDL predicate 정의: `src/benchmarks/LIBERO/libero/libero/envs/predicates/base_predicates.py`
  - 사용 가능: `In, On, Up(z≥1.0), Open, Close, TurnOn, TurnOff, InContact`. (reach/transport 같은
    **연속 모션 predicate 는 없음** — 그래서 gap 으로 채우는 것.)
- `_check_grasp`: `src/benchmarks/robosuite/robosuite/environments/manipulation/manipulation_env.py:331`
- `check_contact(sim, geoms_1, geoms_2)`: `src/benchmarks/robosuite/robosuite/utils/sim_utils.py:8`
- object check_contact/check_contain: `src/benchmarks/LIBERO/libero/libero/envs/object_states/base_object_states.py`
- env wrapper 체인에서 도메인 찾기: 현재 `bddl_phase_labeler.find_domain()` 참고(`parsed_problem`+`_eval_predicate` 보유 객체).
  gripper/geoms 는 그 도메인이 아니라 robosuite env(`env.sim`, `env.robots[0].gripper`)에서 접근.

---

## 7. 범위 / 비범위

- **범위(이번)**: event-anchored 라벨러 설계+구현, libero_10 전 task 분절, `scripts/eval/libero.py`
  소비처 배선 갱신, 단위테스트 갱신.
- **비범위(지금)**: reach/transport **연속 모션을 더 잘게** 쪼개기. 그건 sim 이산 신호로 불가 →
  VLM segmenter(INSIGHT 파일럿, `scripts/analysis/insight_seg`) 별도 실험.
- **규율(사다리 ablation)**: coarse(현재 goal-predicate) 에서 succ/fail 분리 신호가 보이는 phase 만
  finer 로 내려간다. 무신호 phase 를 미리 쪼개면 noise fit (길이 confound 아티팩트 재발 위험).
