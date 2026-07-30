# OpenStandMixerHead — instruction · 성공 판정 · phase 라벨러 · 스모크 결과

작성 2026-07-22. CloseFridge(docs/steering/26) 대안으로 검토한 exp4 후보 task.
COAST 논문 GR00T N1.5 RoboCasa 표에서 Δ 상위(+0.20 대조 / +0.33 positive-only)이면서
exp2·exp3 에서 아직 안 건드린 두 task 중 하나.

---

## 1. instruction — 1종뿐

`OpenStandMixerHead.get_ep_meta()` 가 `ep_meta["lang"] = "Open the stand mixer head."` 로
**하드코딩**(`kitchen_stand_mixer.py:20`). 물체 이름 변주도, fixture 종류 분기도 없다.
ep_meta 50 seed 실측: 전부 동일 문장, fixture 클래스도 `StandMixer` 하나.

CloseFridge 와 비교 — 이 task 를 고르는 이유:

| | CloseFridge | OpenStandMixerHead |
|---|---|---|
| instruction | 2종 (door / doors) | **1종** |
| fixture 모델 | 3종 | **1종** |
| 모델↔layout | 완전히 얽힘 (BottomFreezer=layout1 전부) | layout 1/2/4/6/7 에 10/11/8/11/10 |
| 성공 판정 | 모든 fridge 도어 ≤0.005 (다짝 + near-miss) | `head > 0.99` 단일 관절 |
| 판정 이슈 | 육안상 닫힘인데 실패(0.007) 존재 | 결과가 **이봉분포**(0.00 근처 vs 1.00) |
| 6월 baseline SR | 0.67 | 0.73 |

instruction confound([[instruction-confound]])·fixture 이질성이 구조적으로 없어
confound-audit 의 instruction balance 게이트가 자동 통과된다.

## 2. 성공 판정

`_check_success` = `stand_mixer.get_state(self)["head"] > 0.99` (정규화 head 관절, 0=닫힘).
`CloseStandMixerHead` 는 `< 0.01`. reset 시 head ≈ -0.0005 (닫힘).
`env.objects` 는 **비어 있다**(distractor 없음) → wrong-grasp 이 구조적으로 발생 불가.

라벨러는 `get_state`(내부 캐시, `update_state` 호출 순서에 의존) 대신
`fxtr.get_joint_state(env, [_joint_names["head"]])` 로 관절을 직접 읽는다.

## 3. StandMixerPhaseLabeler

`FridgePhaseLabeler` 를 상속하고 ① fixture 참조(`env.stand_mixer`) ② 관절(head 1개)
③ 대상 geom ④ 라벨 이름만 교체. 임계값은 env 판정과 동일(OPEN_TH 0.99 / CLOSED_TH 0.01).

geom 은 **head body 소속 geom 전체**를 쓴다 — fridge 처럼 이름 붙은 손잡이 geom 이 없고
head body 하위가 전부 익명(g0..gN)이기 때문. 표면거리(박스/원통/구 해석적)는 fridge 에서
만든 것을 그대로 재사용.

phase: `reach-to-head` / `contact-head` / `lift-open` / `push-down` / `disengage` / `open-done`.

유닛테스트: `tests/test_fridge_phase_labeler.py` (mixer 2개 추가, 전체 20개 통과 —
drawer/fridge 회귀 포함). **실 env 에서 수정 0회**(fridge 는 다짝 대상선택·표면거리 2회 수정).

## 4. 스모크 결과 (18판, seed 100000+ep)

`outputs/eval/robocasa/groot_n15/mixer_smoke/`. N1.5 ckpt120000, HTTP serve, 캡처 OFF.

### 4.1 실행 chunk 길이별 SR

| arm | 실행/예측 | SR | 같은 seed(ep0–5) 대조 |
|---|---|---|---|
| nas16 | 16 / 16 | **9/12 = 0.75** | 5/6 |
| nas5 | **5** / 16 (수집 표준) | **2/6 = 0.33** | 2/6 |

16-실행 0.75 는 6월 `coast_faithful` 0.73·논문 base 0.60 과 정합.
**중요: CloseFridge 와 달리 5-실행에서도 성공이 나온다**(0/14 → 2/6). 즉 우리 수집 표준을
바꾸지 않고도 succ/fail 대조 fit 재료를 얻을 수 있다. 오히려 0.33 이 0.75 보다 클래스
균형이 낫다.

### 4.2 결과가 이봉분포 — 판정 애매성 없음

`allBest` = head 관절이 목표 방향으로 간 최대치(성공 = >0.99):

- 성공 12판: 전부 **1.000–1.005**
- 실패 6판: 0.001 / 0.004 / 0.048 / 0.076 / 0.561 / 0.596 / 0.775

CloseFridge 의 "0.007 vs 임계 0.005" 같은 경계 판정이 없다. 0.56–0.78 구간은 애매한 게
아니라 **부분 개방 후 놓침**이라는 실패 유형으로 읽힌다.

### 4.3 성공은 단조, 실패는 유형이 갈린다

성공 전형 (12판 중 9판이 이 형태, 6–10 inference):
```
reach-to-head → contact-head → lift-open → open-done
```

실패 유형:

| 유형 | 예시 | phase 패턴 |
|---|---|---|
| 접근·재접근 루프 | nas16_a ep0 (best 0.001) | contact 후 `disengage↔reach` 6회 왕복, 끝내 못 듦 |
| 들었다 놨다 진동 | nas16_b ep10 (0.596) | `contact↔lift-open↔push-down` 반복 |
| 부분 개방 후 놓침 | nas16_b ep9 (0.775) | lift-open ×59 → disengage 후 복귀 실패 |
| 완전 실패 | nas5 ep1 (0.004) | contact 만 길게, lift 거의 없음 |

첫 두 유형은 프로젝트 원래 동기였던 **"같은 실패 궤적 반복"** 의 교과서적 사례다.

### 4.4 ⚠ 길이 confound가 극단적

성공 = 조기 종료(nas16 6–10 inference, nas5 20–22), 실패 = timeout(nas16 45, nas5 144).
**성공/실패 길이 비가 5~7배** — [[seen18-rollout-length-confound]] 보다 심하다.
time-pooled feature 분리는 전부 아티팩트가 되므로 truncation 표준
([[truncation-length-standard]]) 적용이 필수. 성공 rollout 이 짧아 **fit 재료가 얇다**는
문제도 있다(nas16 성공 1판 ≈ 7 record). 5-실행이 record 수 면에서 유리하다
(성공 1판 ≈ 20 record, 성공률 0.33) — 수집 표준 유지의 또 다른 근거.

### 4.5 대표 영상

`outputs/eval/robocasa/groot_n15/mixer_smoke/annot/` (배너: instruction / phase / env-step + 색띠)

| 유형 | 파일 |
|---|---|
| 성공 (교과서형, 5-실행) | `nas5--task0--ep5--succ1--annot.mp4` |
| 성공 (16-실행) | `nas16_a--task0--ep1--succ1--annot.mp4` |
| 실패 — 접근·재접근 루프 | `nas16_a--task0--ep0--succ0--annot.mp4` |
| 실패 — 들었다 놨다 진동 | `nas16_b--task0--ep10--succ0--annot.mp4` |
| 실패 — 부분 개방 후 놓침 | `nas16_b--task0--ep9--succ0--annot.mp4` |
| 같은 scene 대조 (seed 100001) | `nas16_a--ep1--succ1` vs `nas5--ep1--succ0` |

## 4.6 scene 실현가능성 필터 — 불가능 scene 은 fit 전에 제외

ep10(seed 100010) 실패 영상을 눈으로 보니 정책 실패가 아니라 **머리가 위쪽 선반에 막혀서**
못 여는 scene 이었다. 이런 판은 latent 실패가 아니라 scene 기하이므로 failure 클래스에
섞이면 안 된다(어떤 steering 으로도 구제 불가 → rescue 분모도 갉아먹는다).

판별: `scripts/safe/groot_n15/robocasa/analyze/mixer_scene_feasibility.py`.
reset 직후 로봇을 건드리지 않고 head 관절만 0→1.0 스윕하며, head body geom 이
(믹서 자신·로봇 이외의) 외부 geom 과 접촉하는 지점을 찾는다. 접촉 없이 도달 가능한 최대
정규화값이 `q_max_feasible` 이고, 성공임계 0.99 미만이면 **어떤 정책으로도 성공 불가**.

| seed | q_max_feasible | 판정 | 막는 geom |
|---|---|---|---|
| 100000–100009, 100011 | 1.000 | OK | — |
| **100010** | **0.575** | **BLOCKED** | `shelves_left_group_level0_shelf` |

12개 중 1개(8.3%) 불가능. 교차검증: seed 100010 rollout 의 실제 최대 head 값 0.596 ≈
기하 상한 0.575 → 정책은 천장까지 밀어붙였다(초과분은 접촉 후 밀림에 의한 탄성 변형).

편향 없음: 정책·체크포인트·arm·chunk 와 무관한 **seed 만의 함수**라 모든 arm 에 같은
seed 집합이 적용된다. 단 **fit·eval 양쪽에 동일 적용**해야 하고(한쪽만 걸면 새 confound),
제외 seed·q_max 는 manifest 에 기록한다.

한계: 정적·관절만 검사 → 그리퍼가 fixture 를 잡은 채 선반에 걸리는 경우는 못 잡는다
(로봇 geom 은 오탐 방지로 접촉 판정에서 제외). 스크립트는 현재 StandMixer 전용이며
fridge/drawer 이식은 fixture 참조·관절·body·임계 4개만 바꾸면 된다 — **drawer/fridge 에
같은 오염이 있는지는 미확인**(exp2·exp3 결과 사후 스캔 가치 있음).

다른 세션용 요약: [`SCENE_FEASIBILITY.md`](SCENE_FEASIBILITY.md).

## 5. 한계

- proximity 라벨은 시뮬레이터 상태를 읽는 **oracle** — 온라인 phase 식별 문제와 별개.
- 접촉은 그리퍼 site 기준 → 팔뚝으로 미는 경우 `contact-head` 없이 `lift-open` 만 찍힌다.
- n=18 스모크. SR 수치는 방향 판단용이고 판정용 표본이 아니다.
