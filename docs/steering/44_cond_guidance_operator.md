# 44 — 상태-조건부 대조 guidance 연산자 (condg)

2026-08-18. 정본 스펙: Notion 3bc63918d42a80a1b256c87aa6b18f73 하단 "★ 차기 연산자 확정
스펙" + "⚠ 정정(길이 confound 감사)" (개정 조항까지가 스펙). 판별 원본:
`scripts/analysis/grid_phase/cond_margin.py` (설계 세션 이관본, 이 파일이 정확식의 단일 출처).

## 1. 연산자 정의

phase별·scene-중심화 후, 성공/실패 각각 상태→활성화 릿지 회귀:

- 상태 φ(s) = [eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)] 9 + 인접 record 차분 속도 9
  = 18차원 (첫 record 속도=0). scene별 z-score (평균 mp, 표준편차 sp+1e-8; train record 풀).
- 활성화 h = DiT L12·마지막 denoise(step 3)·49토큰 mean, scene별 중심화(train 성공-해당phase
  record 평균 mh; 성공 없으면 scene 전체).
- 릿지(무절편): W = (PᵀP+λI)⁻¹PᵀX, λ = 1e-3·n, n = 그 클래스 train record 행수. 클래스별
  W_s, W_f.
- margin m = ‖h̃−φ̃Wₛ‖² − ‖h̃−φ̃W_f‖² (h̃,φ̃=전처리 후). m 클수록 실패 쪽.
- 개입 (SAFE 발화 latch 후, record마다): m > τ 일 때만
  d̂ = normalize(φ̃W_f − φ̃Wₛ), h̃′ = h̃ − β·⟨h̃−φ̃Wₛ, d̂⟩·d̂ (게이트 g=1[m>τ] 하드). β 0.1–0.3.

## 2. 등록 게이트 (개정 조항 — cell별)

- held-out(episode 단위 6:4, rng(0), scene 층화)에서 고정B margin AUROC가 **길이단독 AUROC
  를 strict 초과**할 때만 그 (task×phase) cell 등록, 미달 = identity.
  B = max(3, train 성공 dwell 25퍼센타일). (이 규칙이 Dish contact 부호역전 0.28 과
  Oven reach margin=길이=1.0 판정유보를 자동 배제하는지 fit 로그로 확인할 것.)
- τ = held-out 성공 episode 고정B margin 분포의 90퍼센타일.
- 배포 연산자 = train-split W + held-out τ (재적합 없음 — 캘리브레이션 일관성).
- 최소 표본: 클래스당 ~15 train episode 미달 task는 미등록.

## 3. 대조군

- **condg_pl (위약)**: episode 라벨 scene-층화 순열 후 동일 절차. 단 등록 게이트는
  **우회**(처치가 등록된 cell에서 강제 등록, τ는 자기 캘리브레이션) — 게이트를 통과 못하면
  위약이 identity가 되어 타이밍-섭동 대조 기능을 잃기 때문. AUROC는 기록.
- **condg_hs (ablation)**: ĥ_s 단독 — d̂ 없이 h̃′ = h̃ − β·(h̃−φ̃Wₛ)·? 형태가 아니라, 스펙
  문언대로 "성공-모방 단독": h̃′ = (1−β)h̃ + β·φ̃Wₛ. margin 게이트는 동일 적용.
- **무게이트 ablation**: g≡1 (발화 후 상시).

## 4. 저장 (docs/04 규약)

NPZ per (slug, variant): phase별 W_s/W_f/τ/B/sign, scene별 mh/mp/sp + global fallback
(unseen scene 용), gate 표(전 phase AUROC: margin 고정B·길이단독·record), meta(입력 pkl
sig 집합, λ식, split seed, 스펙판 44). 경로: 승준
`outputs/steer/online_pipe/<slug>/condg_s5m5/` → 로컬 회수 동일 상대경로.

## 5. serve 배선 (op=condg)

- `steering_hooks.py` `CondGuidanceSteering`: NPZ 로드, per-request 상태 주입
  `set_state(phi9)` (속도는 hook 내부 버퍼 차분, /reset에서 초기화), scene 주입(중심화
  파라미터 선택, 미지 scene은 global fallback), phase는 기존 /steering_phase 확장
  `{"phase":…, "scene": int}`.
- 적용 시점: **마지막 denoise call 한정** (fit 표적과 일치). 토큰: all(기본) + future 변형
  (setM eef 진동 전례 → fut arm 병행).
- 스위칭: 기존 online-gated 러너(EP_MODE=replay)·SAFE latch 그대로.

## 7. 결과 (2026-08-18 — drawer-L·Dish 완주, drawer-R은 수집 양보로 유예)

replay 40셀, β0.3, arm = online(condg)/online_pl(위약)/online_hs(성공-모방). 등록 phase:
drawer-L grasp-handle, Dish contact-rack. **scene 축은 전부 fit-노출** (판정 (c) 체제) —
in-fit-noise(n0,1) vs held-out-noise(n5,6)로 분리 보고.

| task | 분할 | base SR | 3 arm SR | 구제 (발화·실개입) | 파손 |
|---|---|---|---|---|---|
| drawer-L | in-fit n0,1 | 0.450 | **0.400 (3 arm 동일)** | 0/11 (11·8) | 1/9 — s8n0, 3 arm 공동 |
| drawer-L | held-out n5,6 | 0.700 | 0.700 (동일) | 0/6 (6·4) | 0/14 |
| Dish | in-fit n0,1 | 0.350 | 0.350 (동일) | 0/13 (12·2) | 0/7 |
| Dish | held-out n5,6 | 0.450 | 0.450 (동일) | 0/11 (11·3) | 0/9 |

- **판정: condg도 null** — 처치·위약·성공-모방이 셀 단위까지 완전 동일. 실개입 분모
  (drawer 12판·Dish 5판 + 성공셀 개입 4판)에서 뒤집힘 0, 유일한 변화(s8n0 파손)는 세
  arm 공동 = 개입 존재의 타이밍 섭동. read(margin AUROC 0.78–0.86)≠write 재확인.
- β0.3에서는 위약조차 경계셀을 거의 못 뒤집음(β1.0 라운드 대비) — 개입 자체가 sub-섭동.
- **Dish의 실개입 병목**: 발화 23/24인데 등록 phase(contact-rack) 재방문이 5판뿐 —
  단일-phase 등록 게이트의 커버리지 대가.
- hs는 러너 게이팅 버그(수정 커밋 33cb2b9 이후 fix)로 1차분 폐기·재실행분만 사용.
- **α 부수 산출**: 기록 failure_scores 사후 sweep(재계산=실측 35/35 일치, drawer-L) —
  α0.2는 FPR 동일(2/20)·발화 record 38→33 (공짜 개선), α0.3은 record 1까지 당겨지나
  FPR 15/20 붕괴. **SAFE 신호는 컨택-후에만 분리** (43 절제 detector가 preW 0.17→1.00
  로 이를 해소하는 별도 축 — 후속 arm 후보).
- **drawer-R 완주 (08-19 resume)**: base 40/40 수집 완전 재현(srv50 결정성 검증).
  개입 3 arm — in-fit: SR 0.650 동결(구제 0/7, 실개입 6), held-out: 구제 0/6(실개입 6),
  파손 1~2건이 arm별 상이 셀(s8n6=pl·hs 공동, s5n0=hs 단독) = 내용-무관 섭동 노이즈.
  detector 특이: unseen noise 성공 셀 오발화 9/14 (FPR 0.64 — reach 등록 phase 상시
  재방문과 겹쳐 성공 셀 개입 6판까지 발생, 그래도 파손은 1~2). **3-task 총계: 구제
  0 / 실패 47셀 (실개입 ~35판)** — condg null 최종 확정.

### 7.1 절제 detector × condg (조기-발화 분리 검증, drawer-L)

연산자·β·셀 동일, detector만 43 phase-gt 절제판으로 교체 (og_condg_trunc):

| | full detector | 절제 detector |
|---|---|---|
| 발화 시점 | record 38 일률 | **22–32 (중앙 32), 이벤트-결속** |
| 실패 셀 실개입 | 12/17 | **16/17** |
| 구제 | 0 | **0** |
| 파손 | s8n0 (처치=위약) | s8n0 (동일) |
| 오발화 | 2/20 | 6/20 |

**타이밍 가설 기각** — 43 예측대로 발화가 컨택 전후 창까지 당겨지고 개입 커버리지가
거의 전 판으로 늘어도 처치=위약 셀 단위 동일·구제 0. 병목은 발화 시점이 아니라
**개입 내용/강도**. (다음: β 인상 + denoise step-0 개입 — layer 재선정 sweep 진행 중.)

## 6. eval 계획

replay 40셀(수집 머신 매칭): 등록 통과 cell 보유 task 우선 —
OpenDrawer_right(reach·grasp), OpenDrawer_left(grasp), DishwasherRack_out(reach).
arm = online(condg) / online_pl / online_hs / (여유 시 fut·무게이트). 판정 = 셀-paired
구제/파손, **모든 succ/fail 수치에 고정B + 길이단독 baseline 병기** (사용자 규칙).
