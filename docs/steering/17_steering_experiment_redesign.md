# Steering 실험 전면 재설계 v2 (2026-07-09, A100-추적 세션 작성)

## ⚠️ α 배선 감사 결과 (2026-07-09 실물 확인 — HANDOFF §1.1 "α=0.3 균일" 기술은 부정확)
- fit 단계 α 선택(overlap 밴드)은 **매 fit 실행되고 있었음**: 전 fit 1160건 분포 =
  0.1(49%) / 0.3(28%) / 1(15%) / 3(6%) / 10(3%). NPZ에는 {선택 α, 안전 default 0.3} 두 키 저장.
- serve는 STEER_ALPHA 미지정 시 NPZ **첫 키**를 로드하는데, 저장이 python set 순회라 순서가
  hash 우연에 좌우: 실측 {0.1,0.3}→첫 키 0.1 (**선택값 적용됨**), {0.3,1}/{0.3,3}→첫 키 0.3
  (**선택값 무시, 0.3으로 억눌림**). 즉 실제 적용 α는 {0.1, 0.3} 혼합, 선택이 ≥1인 23%는 조용히 0.3.
- 재설계 필수 배선: ① 선택 α를 fit meta(JSON)에서 읽어 serve에 **명시 전달**(STEER_ALPHA),
  ② NPZ 저장을 set이 아닌 명시 순서로, ③ HANDOFF §1.1 기술 정정.

## fit/layer/β 확정 (2026-07-09 사용자 결정)
- **fit = {15, 30}만** (60 제거; COAST 표준=15). 샘플링 = **첫 N판 창 + 클래스 부족 시 창 밖 backfill**:
  창 안 실패<3이면 창 이후 에피소드에서 가장 이른 실패판을 가져와 창 안 마지막 성공판과 교체
  (총량 N 유지; 성공<3도 대칭). fit 스크립트 `--min-per-class=3` 기존 존재 — 게이트 호환.
  최종 사용 episode 목록은 manifest/NPZ meta에 기록.
- **layer 후보 = {quota top-1 single, 현행 multi 4-8-12}** (top-3 multi는 P4 옵션):
  bread84 held-out 증거 = 임의 고정 single은 파국(L4 31/60)~무효(L812 45/60), multi 4-8-12는
  +13%p(gatedps15 55/60 vs base 47/60) → multi를 후보에서 제거하지 않고 quota top-1과 함께
  선택 rollout이 결정. 단 우리가 시험한 single은 선택된 layer가 아니었음(L10 등 미시험).
- **β = {0.1, 0.3}은 P2 선택 rollout에서만 경쟁** — held-out 과학표 행 수는 불변(승자 1 config만 올림).
- **tie-break 규칙**: 선택 rollout(30판, SE≈9%p)에서 후보 SR이 SE 이내 동률이면 보수 쪽 채택
  = 작은 β, 적은 layer 수. (오선택의 하방 위험이 비대칭: 공격 config는 파국, 보수 config는 base 근처.)
- 과학표 갱신: {perm,gated}×{ps,x,gx}×{15,30}=12 + base + null + positive-only =
  **15 arm/cell × 8 cell = 120 arm** (held-out ep60–119).

> 목적: instruction 2 × scene 4, fit ep0–59 / test ep60–119 구조에서 conceptor steering의
> 공정한 종결 실험. 실행은 main 세션이 담당. 배경: 현 scene-seed 매트릭스에서 apple(고SR
> cell) 전 arm 붕괴 — 원인 = fit 실패 표본 2~6판(대조 불성립) + α=0.3 고정(연산자가 거의
> 0 행렬 → 3-layer 전방위 ~30% 감쇠) + per-cell hyperparameter 선택 부재 (COAST는 태스크마다
> ℓ/α/β/전략을 fitting rollout SR로 선택; GR00T RoboCasa 선택값 = 단일 L10, α=0.1, β 0.1–0.3).

## 핵심 설계 원칙: 과학 축 vs nuisance 축 분리

- full factorial 불가: 2(perm/gated)×L×A×B×3(fit)×3(scope)는 L=4,A=3,B=2만 돼도
  433 arm/cell → 8 cell ≈ 20.8만 ep ≈ 66일(6-lane). 금지.
- **과학 축 (held-out 전수 보고)**: scope {ps, x(동일 instr cross-scene), gx}, fit {15,30,60},
  전략 {perm, gated, (+positive-only 1개)}.
- **nuisance 축 (cell별 선택 후 고정)**: layer ℓ, aperture α, strength β — COAST A.10.2 이식.

## Phase 순서

### P0 — 수집 + scene 선정 게이트 (이번 라운드 최대 교훈 반영)
- instruction 2 × 후보 scene 6~8개 × ep0–59 base 수집 (수집 자체가 fit 재료).
- **게이트: fit 60판에서 succ ≥ 15 AND fail ≥ 15** (base SR ~0.25–0.75). 미달 scene은
  manifest에서 다음 seed로 교체. 게이트 통과 4 scene/instruction 확정.
- 비용: 후보 8×2=16 scene × 60판 ≈ 960판 ≈ 7~10h (6-lane). 탈락분은 매몰비용으로 수용.

### P1 — 기하 스크리닝 (rollout 0판, 순수 CPU)
- refit: `fit_phase_conceptor_n15.py` — 7 capture layer {0,2,4,8,10,12,15} × α {0.1,0.3,1,3,10}
  (스크립트에 DEFAULT_ALPHAS·select_alpha 밴드 로직 이미 내장; 현 final NPZ는 L4/8/12·α0.3만
  있어 재fit 필요. cell당 몇 분).
- layer 후보 = quota tr(C)/d top-1 (rung1; top-3 multilayer는 rung2 옵션 — 원 multilayer 근거는
  in-sample 아티팩트로 판명된 상태이므로 기본값 금지, β 감쇠 보정(예: 3-layer면 β 하향) 동반).
- α 후보 = overlap(C_s,C_f) 밴드 [0.85,0.95] 최소 α (COAST Fig.7: overlap은 α에 단조).

### P2 — 선택 rollout (select-half에서만, 보고 금지)
- **07-10 정정 (Gate1)**: fit(ep0–59 층화 랜덤)과 P2(ep0–29)가 episode 를 공유하면 in-sample
  rescue 아티팩트가 선택을 오염 → scene 당 60판을 층화 고정-seed **30/30 disjoint split** 으로
  나눠 fit 은 fit-half 에서만 표본, P2 는 **select-half 30판**에서 평가 (교차 backfill 금지).
- 후보 2 config × β {0.1,0.3} → 선택 규칙 = **하방-위험 우선**: base 보다 2 episode 이상 낮은
  후보 탈락 → 잔여 중 SE 이내 동률이면 보수(작은 β·적은 layer).
- cell당 4 arm-equiv ≈ 반나절/lane. 선택 수치는 절대 보고하지 않음 (in-sample).

### P3 — held-out 본실험 (ep60–119, n=60, 보고 대상)
cell당 arm 목록 (선택된 (ℓ,α,β) 고정) — **07-10 정정: 상단 확정(fit{15,30})과 일치하도록
fit60 제거** (구판 21 arm/168 은 폐기, Gate1 원장 docs/collab/2026-07-10-steering-redesign-gate1.md):
1. base (1)
2. **null 대조** (scene×instruction 층 내 episode 라벨 permutation fit conceptor, 동일 β) (1)
   — 방향성 vs 일반 섭동 분리, 필수. record 단위 셔플 금지(길이 편향 재유입).
3. positive-only (C_steer=C_succ; 게이트 탈락 고SR scene이 있다면 그쪽의 유일 합법 전략) (1)
4. {perm, gated} × {per_scene, cross_scene, grand} × {fit15, 30} = 12
→ **15 arm/cell × 8 cell = 120 arm ≈ 1.8일 (실측 65 arm/일, 5-lane)**

### P4 (옵션) — 대표 cell 1~2개 국소 확장
- β ±1단계 미니 sweep, top-3 multilayer, denoise per-step(COAST global vs per-step; 현재 K축
  mean-pool을 푸는 것), (A)안 채택 시 x-full/gx-full arm. 전이 실험은 계획 제외.

## scope 축 확정 정의 (2026-07-09 사용자 확정: 전이 실험 없음)
- **ps (per-scene)**: 해당 scene의 fit 데이터만으로 fit, 그 scene에만 적용.
- **x (instruction-pooled)**: instruction 내 4 scene fit 데이터를 합쳐 fit한 **공유 conceptor 1개**를
  그 4 scene 각각의 held-out에 평가. (scene A fit → scene B 적용 같은 전이 아님 — 전이는 계획 제외.)
- **gx (grand-pooled)**: 두 instruction 8 scene 전체 pool로 fit한 conceptor 1개를 전 scene에 평가.
- **fit 크기 = 총량 고정 (A, 2026-07-09 사용자 확정)**: fitN = 어느 tier든 **총 N판**.
  ps fit15 = 자기 scene 60판 중 15판, x fit15 = instruction 240판 pool에서 15판(4 scene 골고루),
  gx fit15 = 480판 pool에서 15판(8 scene 골고루).
- **샘플링 제약 (사용자 확정)**: 뽑힌 fitN 안에 **성공 ≥ 3 그리고 실패 ≥ 3 반드시 포함**
  (구현은 MIN_CLASS 파라미터로; fit30/60은 비례 상향 권장 — 예: fit30 ≥6, fit60 ≥12 — 사용자 확인 필요).
  cross/grand tier는 추가로 scene 하한(예: x-tier scene당 ≥3, gx-tier scene당 ≥1)과 병행.
  ⚠️ 클래스 하한과 scene 하한을 동시에 per-scene per-class로 걸면 fit15에서 불충족(4 scene×2 class×2 > 15)
  — class 하한은 pool 전역, scene 하한은 scene 전역으로만.
- **재현성**: 첫 N판 고정 방식(현행, apple fit15 실패 0~2판 사고의 원인) 폐지 → 제약 층화 랜덤 추출
  + 고정 seed + **뽑힌 episode 목록을 NPZ meta/manifest에 기록** (검증 가능하게).
- (검토 노트) 하한 대신 성패 비율 고정(예: fit15=succ10+fail5)이 cell 간 fit 구성 confound를 더 강하게
  차단하는 대안 — 하한 방식은 base SR이 fit 구성에 반영되는 설계(사용자 선택), 비율 고정은 반영 차단.
  기본 = 사용자 확정(하한 3), 비율 고정은 P4 감도 체크 후보.
- 절충 arm: fit60 지점에서만 x-full(240판)·gx-full(480판) 2개 추가 검토 ("데이터 다 쓰면?" 질문 전용).
- **선택 granularity**: hyperparameter 선택은 conceptor 공유 단위와 일치 — ps=scene별,
  x=instruction별, gx=전역 1개.

## 사용자 원안 대비 빠졌던 세트
1. null 대조 arm (필수)
2. positive-only 전략
3. scene 선정 게이트 (succ/fail 밴드) — arm 아님, 설계의 일부
4. (옵션) denoise per-step 축
5. ~~cross-instruction 전이~~ — 계획 제외 확정 (위 scope 정의 참조)

## 배선 체크리스트 (main 세션)
- serve: STEER_ALPHA 명시 필수 (미지정 시 NPZ 첫 키 로드 함정), STEER_LAYERS=선택값
- 신규 파일 원칙 (실행 중 스크립트 무수정), pq 큐 arm type으로 통합, MACHINE.txt 출처 기록
- 선택/보고 seed 분리 검증은 artifacts 기준 (fit ep0–59 ∩ eval ep60–119 = ∅)
- 고아 상태 ab_sweep_chain은 이 설계로 흡수 (bread84 파일럿으로 P1→P3 절차 먼저 1 cell 검증)
- 통계: n=60 SE≈6.5%p; per-seed flip 분해는 머신 내 한정; cell별 machine 각주
- CPU cap (OMP≤16), GPU 양보(0–3 동료), 장시간 run setsid, 검증 전 삭제 금지

## 기대치
- 가장 유력한 결과: "per-cell 선택 후에도 scene-일관 개선 없음" → raw conceptor 종결 + SAE 진입
  근거 확정. 선택값의 cell 간 수렴 여부(예: 전부 L10 근처)는 부산물 정보.
- bread84류(균형 잡힌 succ/fail) cell에서만 국소 +가 남는지가 관전 포인트.
