# CloseFridge — instruction 인벤토리 · 성공 판정 · phase 라벨러

작성 2026-07-22. 대상 task = **CloseFridge** (seen18 task set, `robocasa365`).
exp4 대상 task 4종 중 하나 ([[exp4-final-plan]]) 이라 drawer 와 같은 급의 phase GT 가 필요.

---

## 1. task 안에 존재하는 instruction

### 1.1 코드에서 유도되는 경우의 수 = 2가지

`ManipulateDoor.get_ep_meta()`
(`src/benchmarks/robocasa/robocasa/environments/kitchen/atomic/kitchen_doors.py:169`):

```
lang = f"{behavior.capitalize()} the {fxtr.nat_lang} {door_name}."
door_name = "doors" if isinstance(fxtr, (HingeCabinet, FridgeFrenchDoor)) else "door"
```

- `CloseFridge` 는 `fixture_id=FixtureType.FRIDGE` 로 고정 → `fxtr` 는 항상 `Fridge` 서브클래스.
  `Fridge.nat_lang` 은 하드코딩된 `"fridge"` (`models/fixtures/fridge.py:432`) → 물체·상태에 따른
  변주가 **없다**. HingeCabinet 은 FRIDGE 타입으로 잡히지 않으므로 분기에서 탈락.
- 따라서 가능한 문장은 정확히 두 개:

| instruction | 조건 |
|---|---|
| `Close the fridge door.` | fxtr 가 FridgeFrenchDoor 가 **아님** |
| `Close the fridge doors.` | fxtr 가 `FridgeFrenchDoor` |

PnP task(사과/빵 등)처럼 **물체 이름에 따른 instruction 변주가 없다** — 이 task 의 instruction
변주는 사실상 *냉장고 모델* 변주의 대리변수다.

### 1.2 실측 (ep_meta 56 seed)

기존 4개 수집 run 의 `ep_meta/CloseFridge/*.json` 118개(고유 seed 56개, seed 100000~100069)를
집계한 결과 — 위 2종 외 문장은 0건:

| instruction | fridge fixture | seed 수 | layout |
|---|---|---|---|
| `Close the fridge door.` | `FridgeSideBySide` (Refrigerator031) | 29 | 2 (14), 7 (15) |
| `Close the fridge door.` | `FridgeBottomFreezer` (Refrigerator060) | 19 | 1 (19) |
| `Close the fridge doors.` | `FridgeFrenchDoor` (Refrigerator064) | 8 | 4 (4), 6 (4) |

⚠️ **fridge 모델과 layout 이 완전히 얽혀 있다**(BottomFreezer=layout1 전부, FrenchDoor=layout4/6 전부).
아래 SR 차이를 "모델 때문"이라고 단정할 수 없다 — layout/style/로봇 배치와 분리 불가.

### 1.3 instruction·모델별 baseline SR (참고값, n 작음)

`coast_faithful_7task_30ep`(2026-06, N1.5 baseline, seed 100000+ep) 파일명 succ 플래그와
ep_meta 를 조인:

| instruction | fridge 모델 | succ / n |
|---|---|---|
| `Close the fridge door.` | FridgeSideBySide | **14 / 15 = 0.93** |
| `Close the fridge doors.` | FridgeFrenchDoor | 5 / 8 = 0.62 |
| `Close the fridge door.` | FridgeBottomFreezer | **1 / 7 = 0.14** |

전체 20/30 = 0.67. 같은 instruction(`door.`) 안에서 SideBySide 0.93 vs BottomFreezer 0.14 로
갈리므로, **instruction 만으로 층화하면 안 되고 fixture 모델(=layout)로 층화해야 한다.**
drawer 의 left/right 비대칭과 같은 성격의 within-task 이질성.
(n=7~15 의 관찰값이고 위 layout confound 가 있어 확정 아님 — 층화 필요성의 근거로만 사용.)

---

## 2. 성공 판정 기준

`ManipulateDoor._check_success()` → `fxtr.is_closed(env)` →
`Fridge.is_closed(compartment="fridge", reg_type="door", th=0.005)` →
`Fixture.is_closed(joint_names=_fridge_door_joint_names, th=0.005)`:

- 대상 관절 = **fridge 칸 도어 관절만**. freezer 도어·서랍 관절은 판정에 안 들어간다.
- 정규화 열림도 `norm_qpos ∈ [0,1]` (0=닫힘)가 **모든** 대상 관절에서 `≤ 0.005` 여야 성공.
- FridgeFrenchDoor 는 fridge 도어가 2개(`fridge_left_door_joint`, `fridge_right_door_joint`)
  → **두 짝 다** 닫아야 성공. 한 짝만 닫으면 실패.
- reset 시 `_setup_scene` 이 `open_door(min=0.90, max=1.0)` 으로 열어 둔다(실측 0.93~0.96).

임계 0.005 는 매우 빡빡하다(전체 행정의 0.5%). apple 판정(docs/steering/18)처럼 "거의 닫았는데
실패" 가 존재할 수 있어 `door_worst_timeline` 을 저장해 near-miss 를 사후 집계할 수 있게 했다.

---

## 3. FridgePhaseLabeler

`scripts/safe/groot_n16/robocasa/collect/robocasa_event_labeler.py`.
`make_robocasa_event_labeler` 가 `OpenFridge|CloseFridge`(단 `FridgeDrawer` 제외)를 이 라벨러로
라우팅한다. drawer 라벨러와 동일한 공개 인터페이스라 collector·EnvStepGT 는 무수정.

### 3.1 drawer 와 달라지는 점 (실측 기반)

1. **도어가 여러 개일 수 있다.** French door 는 2짝이고 성공은 두 짝 모두 요구.
2. **fixture API 가 없다.** `Fridge` 에는 `get_door_state` 도 `handle_name` 도 없다(둘 다 `None`).
   관절은 `fxtr._fridge_door_joint_names`, 손잡이/패널은 geom 이름 규약으로 직접 해석:
   `<door>_handle_main | _handle_1 | _handle_2 | _main` (3개 모델 공통, 실측 확인).
3. **파지가 아니라 밀기.** grasp 대신 근접(`contact-door`)을 쓴다.
4. **거리를 geom 중심이 아니라 표면으로 재야 한다.** drawer 손잡이는 작아서 중심거리로 충분했지만
   도어 패널 geom 은 큰 박스다(실측 half-extent 0.224×0.005×0.861 = 0.45m×1.7m 판). 중심거리를
   쓰면 판 가장자리를 밀고 있어도 0.8m 넘게 나온다 — 초안 실측에서 그리퍼가 문에 붙어 있는
   프레임이 전부 `reach-to-door` 로 찍혔다(`sidebyside ep2`: 721 step 전부 reach). 박스/원통/구는
   해석적 표면거리, 그 외(mesh)는 중심거리 fallback. 이 수정 후 임계값은 0.06 m.

### 3.2 phase 정의

| phase | 조건 |
|---|---|
| `close-done` | 모든 fridge 도어 ≤ 0.005 (= env 성공 판정과 동일) |
| `wrong-grasp` | 도어와 무관한 물체(door_obj/distractor) 파지, HOLD debounce |
| `push-close` | 대상 도어가 목표 방향으로 움직이는 중 (Δq < −1e-3, HOLD 연속) |
| `swing-open` | 대상 도어가 반대로 열리는 중 |
| `contact-door` | 그리퍼–대상 도어 **표면** 최근접 거리 < 0.06 m |
| `disengage` | 한 번 접촉한 뒤 도어에서 멀어지는 중 (Δd > 3e-3, HOLD 연속) |
| `reach-to-door` | 그 외 (최초 접근 + disengage 후 재접근) |

phase 는 현재 상태의 순수 함수(비단조) — drawer 와 동일 규약.

### 3.3 대상 도어 선택 (다짝 처리)

**대상 = 아직 목표에 도달하지 않은 도어 중 그리퍼에 가장 가까운 것.**

초안은 "가장 열린 짝"을 대상으로 잡았는데, 실측 ep0(FrenchDoor)에서 로봇이 *덜 열린* 짝을
먼저 닫는 동안 대상 도어의 q 가 안 변해 `push-close` 가 721 step 중 5 step 밖에 안 잡혔다.
거리 기준으로 바꾸면 "지금 로봇이 다루고 있는 도어"를 따라간다. 대상 전환 시 Δq·Δd 연속성이
끊기므로 3 cm 히스테리시스(`TARGET_MARGIN`)를 둔다.

`_done` 은 대상과 무관하게 **항상 전체 도어**로 판정 — env 판정과 어긋나지 않게.

### 3.4 저장 필드

record 단위 라벨(`feature_phases`)과 env-step 단위 GT(`env_step_phases`, `env_step_phase.py` 규약)를
둘 다 저장. fixture task 진단용으로 추가:

- `env_step_door_state` — 대상 도어 정규화 열림도 궤적
- 라벨러 내부 `door_worst_timeline` — 전체 도어 중 목표에서 가장 먼 값(= 성공을 지배하는 값)

### 3.5 검증

- 유닛테스트 `tests/test_fridge_phase_labeler.py` 9종(스텁 env, robocasa 불필요):
  dispatch, FridgeDrawer 오라우팅 방지, phase 진행, swing-open 구분, disengage→reach 복귀,
  다짝 대상 선택, 덜 열린 짝 진행 검출, 성공 임계 = env 판정 동치, wrong-grasp rising edge,
  인터페이스 계약.
- 실 env 게이트: seed 100000(FrenchDoor) 1판 수집 → 라벨/영상 프레임 대조 (§4).

---

## 4. 샘플 영상 (instruction × 성공/실패)

수집: `outputs/eval/robocasa/groot_n15/fridge_smoke/` (캡처 OFF — json 사이드카 + mp4),
주석 영상: `annotate_phase_video.py` (배너에 instruction / phase / env-step + phase 색띠).

seed 100000+ep, 14 ep × 2 arm(실행 chunk 길이). N1.5 ckpt120000, HTTP serve, EVAL_SEED 표준.

### 4.1 실행 chunk 길이가 이 task 의 SR 을 지배한다

| arm | 실행/예측 | SideBySide | BottomFreezer | FrenchDoor | 전체 |
|---|---|---|---|---|---|
| `fridge_smoke` | **5** / 16 (우리 수집 표준) | 0/4 | 0/4 | 0/6 | **0/14 = 0.00** |
| `fridge_smoke_nas16` | **16** / 16 | 4/4 | 2/4 | 4/6 | **10/14 = 0.71** |

같은 seed·같은 체크포인트인데 0.00 vs 0.71. 16-실행 값은 COAST 논문의 GR00T N1.5 CloseFridge
base 0.67, 우리 6월 `coast_faithful` 0.67 과 일치한다. **CloseFridge 를 5-실행 표준으로 수집하면
성공 롤아웃이 안 나와 대조 fit 자체가 불가능**하다 — exp4 에서 이 task 를 쓰려면 실행 길이를
16 으로 두거나 task 를 바꿔야 한다.

16-실행에서는 6월 baseline 의 모델별 비대칭도 재현된다(SideBySide 4/4, BottomFreezer 2/4).

### 4.2 near-miss — 육안상 닫혔는데 실패

`allMin` = 전체 도어 중 목표에서 가장 먼 값의 궤적 최소치(성공 = ≤0.005):

| arm | ep | 판정 | allMin | 해석 |
|---|---|---|---|---|
| nas16 | french 6 | 실패 | **0.007** | 임계 0.005 를 0.002 차이로 놓침 |
| nas5 | french 1 | 실패 | 0.029 (추적도어 최소 0.005046) | 육안상 닫힘 |
| nas5 | french 12 / 25 | 실패 | 0.929 / 0.971 | 한 짝만 닫음 (진짜 실패) |
| nas16 | bottomfreezer 4 / 10 | 실패 | 0.617 / 0.835 | 문에 손도 못 댐 (진짜 실패) |

즉 CloseFridge 실패는 **① 한 짝만 닫음(French), ② 접근 자체 실패(BottomFreezer),
③ 임계 near-miss** 세 종류로 갈린다. ③ 은 apple(문서 18)과 같은 판정 이슈 —
steering Δ 를 볼 때 재채점 대상이 될 수 있다.

### 4.3 phase 점유

| arm | reach-to-door | contact-door | push-close | disengage | swing-open | close-done |
|---|---|---|---|---|---|---|
| nas5 (전 실패) | 86.1% | 3.7% | 4.7% | 4.6% | 0.9% | 0% |
| nas16 | 48.3% | 18.8% | 17.9% | 11.5% | ~1% | ~2% |

성공이 나는 arm 에서는 phase 가 고르게 분포한다(reach 48 / contact 19 / push 18 / disengage 12).
`close-done` 은 프레임이 극소(성공 즉시 종료) → drawer 의 `open-done` 과 같이 fit 에서 제외 권장.

### 4.4 대표 영상 (주석: instruction / phase / env-step + 색띠)

`outputs/eval/robocasa/groot_n15/fridge_smoke_nas16/annot/`

| instruction | 결과 | 파일 |
|---|---|---|
| `Close the fridge doors.` | 성공 | `french_doors--task0--ep0--succ1--annot.mp4` |
| `Close the fridge doors.` | 실패 (한 짝만) | `french_doors--task0--ep25--succ0--annot.mp4` |
| `Close the fridge door.` | 성공 | `sidebyside--task0--ep2--succ1--annot.mp4` |
| `Close the fridge door.` | 실패 (접근 실패) | `bottomfreezer--task0--ep4--succ0--annot.mp4` |

같은 instruction·같은 fixture 안에서 성공/실패 대조가 필요하면 BottomFreezer
`ep19--succ1` vs `ep4--succ0`, FrenchDoor `ep0--succ1` vs `ep6--succ0`(near-miss) 를 쓴다.
5-실행 arm 영상은 `../fridge_smoke/annot/` (전부 실패).

### 4.5 알려진 한계

- 접촉은 **그리퍼 site 기준**이다. 로봇 팔뚝/베이스로 문을 밀면 `contact-door` 없이
  `push-close` 만 찍힌다(관절이 움직이므로 push 는 잡힘). BottomFreezer 실패판에서
  `contact-door` 없이 `swing-open` 만 나오는 구간이 이 경우다.
- proximity 라벨은 시뮬레이터 상태를 읽는 **oracle** — 온라인 phase 식별 문제와 별개
  (conceptor-pipeline skill Stage 2 규약).
