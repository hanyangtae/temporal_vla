# Gate 1 — conceptor steering 재설계 라운드 실행 계획 반론 (2026-07-10)

- **게이트**: Gate 1 (계획 토론)
- **thread_id**: `019f4b31-1516-7e63-b7b6-828f36ae6207` (2 라운드, codex-cli 0.144.1)
- **입력**: docs/HANDOFF_current.md §6 실행 플랜 + docs/steering/17(재설계 v2)·18(재채점) +
  배선 gap 탐사 결과(파일:라인)
- **결정자**: 사용자 (2026-07-10, AskUserQuestion 3회)

## Round 1 요지 (Codex 반론)

핵심: "현재안 그대로면 null 이 나와도 raw conceptor 종결 근거가 약하다."

1. **corrected-일치 제외 = post-treatment selection bias** [高] — arm 출력이 제외 여부를
   결정 → arm 간 estimand 불일치, null 대비 Δ 해석 불능. 대안: corrected 0.10 단일 primary,
   0.07 은 제외 없는 secondary 열.
2. **셔플 1회는 null 분포가 아님** [高] — episode 단위·층화 permutation, seed ≥3 권고.
   record 단위 셔플은 길이 편향 재유입 (fit 이 record 를 쌓으므로).
3. **bread 천장 cell(base .78/.83)** [高] — n=60 에서 개선 검출력 낮음, "유의 개선 없음"을
   무효 증거로 쓰면 안 됨. harm 전용 지정 + arm 축소 권고.
4. **P2 선택 winner's curse** [高] + **fit(ep0-59 층화)↔P2(ep0-29) episode 중첩** [高] —
   in-sample rescue 아티팩트 재발 경로. 하방-위험 선택 규칙 + 교집합 0 강제 권고.
5. **P0 scene 선발을 SR 순위로 하면 selection bias** [高] — 게이트 통과 중 seed 순서로.
6. **총량 고정 pool 의 희석** [高] — x fit15 = scene 당 3.75판; "pooling 우열" 주장 금지.
7. **배선**: episode 단위 min-class 회귀 테스트, 빈-fit rc≠0, serve preflight(key·α·β·layer·
   NPZ hash ↔ manifest), multi/gated per-NPZ selected_alpha 이질성, closest-α 경고 분리.
8. **종결 규칙 사전 동결** [高] — primary contrast·δ(등가 margin)·다중성 규칙 없으면
   null 종결이 검정력 부족과 구분 불가.

## Round 2 요지 (메커니즘 수렴)

- corrected 단일 primary 일관 적용 동의 + 보완: step 별 두 임계 conjunction 기록,
  live/replay 2×2 일치표 100% 게이트, discordant_rate 는 진단 전용(필터 재사용 금지).
- fit/P2 분리: 시간순 분할 대신 **corrected 라벨 층화 고정-seed 30/30 disjoint split**
  (fit=전반 half, P2=후반 half), 양쪽 min-class ≥6 미충족 시 scene 탈락(교차 backfill 금지).
- null: 3-seed 사전 고정·전부 유지, **primary config 에만 matched**, ‖M−I‖F 는 공변량
  (norm 사후 재생성 금지).
- 천장 cell: primary arm 1 + 기계적 최대공격 arm 1 로 동결, positive-only 는 safety
  fallback 으로만(전략 일반 비교 제외).
- 추정기(phase 별 episode-equal-weight) 채택 시 α grid 전면 재수행 필요(과거 α 와 비교 금지).
- primary contrast = **gated-ps-fit15 vs base** (직전 라운드 국소 양성 + COAST 표준 —
  가장 유리한 조건 정면 재검증), replication = 각 instruction ≥3/4 scene ΔSR>0 ∧ pooled
  Δ>0, δ = +7.5%p (중간SR cell 단측 95% CI 상한).
- P0 seed-순서 선발 동의.

## 사용자 결정 (최종)

| # | 항목 | 결정 |
|---|---|---|
| D1 | 채점 | **eval = corrected 0.10 단일 채점(제외 없음) + 0.07 병기, fit 만 discordant(0.07~0.10) 판 제외.** 원 의도가 이것이었음 — HANDOFF §5 의 "전 단계 표본 제외" 기술이 의도를 잘못 압축한 것. **bread 는 재채점하지 않음** (술어 이상 없다고 판단, 원판정 그대로) — 문서 18 §5·HANDOFF §6 R③ 의 bread 프로파일 항목은 계획에서 제외 |
| D2 | P3 구성 | **원안 120 arm 유지** (위약 1개/cell, 천장 cell 축소 없음) — "올라가든 내려가든 정보" (해악 패턴 데이터 온전히 확보 우선) |
| D3 | fit 추정기 | **COAST식 record-pool 유지** — 길이 confound 는 해석 각주(confound-audit)로 처리, episode-equal-weight 는 P4 sensitivity 후보 |
| D4 | 종결 규칙 | **사전 동결 생략** (재시도는 사용자 재량) — primary contrast·δ 는 집계 보고서의 비구속 권고 기준으로만 기재 |

Codex 합의 채택분 (구현 세부, 기존 결정과 충돌 없음): 30/30 split(비대칭 ≤2 기록)·위약
episode 층화 permutation·P0 seed-순서 선발·P2 하방-위험 선택·live/replay 일치 게이트·
serve preflight·NPZ 키 순서 고정·episode 단위 min-class·빈-fit rc≠0·pool 총량-고정
estimand 명시.

## 추가 결정 (2026-07-10 저녁, P0 실측 후)

- **apple 게이트 완화** (사용자 확정): 후보 12 scene 수집 결과 SR 분포 = 1.0×5, 0.95×3,
  0.85×1, 0.0×1 — 원안 밴드(succ≥15∧fail≥15) 통과 0. apple scene 은 all-or-nothing
  이부모델로 실증됨(밴드 scene 부재, 재사용 ppcs_apple 0.74 가 유일 예외).
  → 게이트 = **succ≥6 ∧ fail≥6** (corrected), fit30 은 fail≥12∧succ≥12 일 때만
  (미달 scene 은 fit15 전용), 천장/바닥 각주 필수. 선발은 여전히 seed 순서.
  구현: pq2/p0_gate.py. 통과 scene 3개 확보까지 배치 스캔 계속.

## COAST 대조 감사 (2026-07-11, P3 중간 — perm 해악 패턴 계기)

perm 파이프라인은 COAST 와 수식·주입점(α 선택 포함) 일치 확인. 갈린 곳 3:
1. **Stage1 layer 선택 기준 편차 (확정)**: COAST 는 quota 를 **α=10** 에서 계산 → 우리 데이터로
   재계산 시 전 cell **L15>L12>L10** (COAST 공표 L10/L11 과 정합). 우리 P2 후보는 선택 α(≈1)
   기준 quota 라 L0/L2/L4 로 몰림 — COAST 가 골랐을 층이 후보에 없었음.
   → 진단 arm 4개 추가 (ho_diag_perm_per_scene_fit30_L15, β0.1, bread 4 cell).
2. **평가 무대 차이 (설계)**: COAST task = 장면 랜덤 분포(매 rollout 랜덤), 우리 cell = 장면
   1개 고정(diffusion noise 만 변동). COAST 세팅에 가장 가까운 6월 native replan5 재현은
   n=30 에서 ΔSR +0.014 (비재현).
3. denoise K 축: COAST 는 step 별 벡터 개별 stack(GR00T 4 step 전부 hook 발화, A.7.2),
   우리는 mean-pool. 토큰 풀링도 상이(COAST GR00T=49토큰 전체, 우리=action 16개 —
   fit·적용 모두). rollout 단위 pool 은 양쪽 다 안 함.
기록: perm 이 base 를 깨끗이 넘은 기록은 전 라운드 통틀어 0 (최선 동률) —
단 pq2 P3 에서 s300033 perm_ps_fit15 +10판·s300028 cross 계열 +2~+5 첫 양성 관측.

### denoise 진단 결과 (2026-07-13) — 전면 비교 라운드 미발동
직접 전후쌍: s300033 pool +10 vs K-stack +1 (**pool 우세**), s400020 −2 vs 0 (동등).
L15: 4 cell 전부 base 근처~미달 (s400020 K-stack +6 이 유일 약양성, SE 이내; s300033
K-stack×L15 = −12 최악). → COAST 충실 세팅(K-stack·L15)이 현행 세팅보다 열등 —
"우리 부진이 denoise/층 편차 탓" 가설 기각, s300033 +10 은 pool 처리 고유 신호.
COAST +0.16 미재현의 잔여 후보 = 49토큰 풀링(fit·적용)·평가 무대(장면 랜덤)·논문 재현성.
사용자 분기 규칙의 "갈리면 전면 비교"는 방향 반대(현행 우세)로 동기 소멸 — 미발동.

### denoise 전후 비교 설계 (2026-07-12 사용자 확정)
"기준을 바꿨더니 결과가 바뀐다" 시나리오 자체가 발견이므로: ① fit `--denoise stack`
(COAST Global 충실, a8ba29c) 으로 저SR 2 cell(s300033·s400020) refit → **직접 전후쌍**
(P3 와 동일 config, denoise 처리만 상이) + L15×K-stack(COAST 최대 충실 조합) 진단 arm
② 결과가 갈리면 denoise 세팅 전후 전면 비교 라운드(P2′→P3′ steering arm 재실행, 수집물
재사용 — "처음부터"가 아니라 nuisance 축 ablation 확장)로 진행.

## 구현 상태 (이 세션에서 배선 완료)

- fit v2 + `pq2/make_fit_manifests.py` (a6311c5) — 승준 원격 스모크·회귀(rc=3) 통과
- robocasa fork 술어 0.10 + 이중 채점 (서브모듈 459c70f, daf532a) — 컨테이너 스모크 통과
- steering loader meta-α 폴백 + preflight 로그 (5615da9) — 3-모드 테스트 통과
- 실행 계획 전문: `~/.claude/plans/sleepy-giggling-newell.md`

## 최종 결과 (2026-07-13, P3 완주 — 라운드 종결)

115 arm × 60 ep held-out 완주 (경고 0, fit/eval 분리 아티팩트 검증: fit 29건 ep≤59 ∧
eval 117건 ep60-119). 집계: `outputs/eval/robocasa/groot_n15/steer_eval_pq2/aggregate_v2/`,
Notion "pq2 재설계 라운드 최종 보고" (parent 38e63918…03a3).

- **Primary contrast(gated-ps-fit15 vs base): pooled Δ = +6판/8 cell — δ(+4.5판/cell) 명확 미달.**
- **위약이 결정타**: 최대 양성 s300033(perm ps15 +10, gated ps15 +8)은 같은 cell 위약 +8과
  동급 — 방향성 신호 아님. 나머지 7 cell 위약 base±3.
- cross_scene: s300028(+2~+5, SE 이내) 외 전 cell 0~음수. positive-only 최악(−16/−13).
- 잔여 관찰: s400020 L15 진단 계열 +6~+9 (단일 scene, SE 언저리) — 확대 해석 불가.
- confound-audit 8게이트 전부 통과/해당없음, 주장 강도 = intervention effect.
- **판정: 구라운드 "raw 대조 conceptor 종결"이 공정 조건 + 위약 대조에서 확증됨.**
  스코프 각주: COAST식 record-pool fit·고정 장면 무대에 한정 (D3·설계 결정). COAST 재현
  잔여 후보 = 49토큰 풀링·장면 랜덤 무대·논문 재현성.
