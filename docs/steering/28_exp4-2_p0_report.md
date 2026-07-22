# exp4-2 P0 결과 — 유도 실패 파일럿 · 비퇴화 진단 · bridge 게이트 (2026-07-22)

계획: [`24b_exp4-2_perturb_conceptor_plan.md`](24b_exp4-2_perturb_conceptor_plan.md) (+공유 24).
섭동 메뉴는 07-22 사용자 개정판(**C1 카메라 / G1 그리퍼 초기위치 / P1 displace / P2 force** —
WAM_Steer 정렬, P3/P4/이미지노이즈 제외). 실행 = worktree `exp/exp4-2-induced-failures`,
데이터 = `outputs/eval/robocasa/groot_n15/exp42_induced/` (전부 로컬, ppcc_bread scene 100084).
**주장 등급: 전부 diagnostic evidence** (SR 개입 없음 — intervention 효과는 P1 이후).

## 1. Smoke S1–S5 (하드 게이트) — 전부 PASS

| smoke | 결과 | 비고 (실측으로 확정된 사양) |
|---|---|---|
| S1 (Track P) | PASS 19 / SOFT 1 / FAIL 0 | sham 4모드 csv bitwise ≡ baseline·double-run 결정론·실효(≠) 전부 통과. SOFT = sham_p1f 실측: **δ=0+`sim.forward()`만으로 bitwise 깨짐**(warmstart) → P1-sham=skip판 동결, C1도 forward 제거 |
| S2 (DiT 주입) | PASS 3/3 | self-donor W=3 **bitwise 일치** (bf16→fp16 무손실), fired 창 정확 |
| S3 (VL 주입) | PASS 4/4 | self-VL은 fp16 양자화로 bitwise 원리상 불가(원값 fp32) — 첫-편차 기준(9.77e-04 ≤ 5e-3) 도입해 배선 항등 판정. T_vl 상이 donor 무에러+실효 |
| S4 (fit 절단) | PASS 2/2 | `--record-start-manifest` 산술 무결 + 누락 fail-loud (passB 실데이터) |
| S5 (exp4-1 계약) | PASS 3/3 | `--steering-phase-npz-base` 로드 + `/steering_phase` gated + 1ep 완주 |

부수 검증: serve 인스턴스 재시작 교차에서도 bitwise 재현(머신 결정론 재확인), 캡처 ON↔OFF 판정 일치.

## 2. P0 유도 실패율 (게이트 40–70%, baseline 실패율 0.28 = 5/18)

| 모드 | config | n | 실패율 | 게이트 | 채택 |
|---|---|---|---|---|---|
| C1 카메라 | scale 2.0 (δpos σ=0.20m·δrot 16°·δfov 10°) | 12 | **0.50** | PASS | **✓ c1_s200** |
| C1 | scale 1.0 (WAM 기본) | 11 | 0.36 | 미달 | ✗ |
| G1 그리퍼 초기 | σ=15cm (closed-loop 추종) | 12 | **0.50** | PASS | **✓ g1_x015** |
| G1 | σ=5/10cm | 4/4 | 0.25 | 미달 | ✗ |
| P1 displace | δ=3cm | 12 | **0.67** | PASS | **✓ p1_d003** |
| P1 | δ=8/15cm | 4/4 | 0.75 | 초과 | ✗ |
| P2 force | 40N×2rec | 12 | **0.58** | PASS | **✓ p2_f040d2** |
| P2 | 5–15N | 4–12 | 0.00–0.25 | 미달 | ✗ |
| Track I B2 (donor 주입) | w3/w6 | 4/4 | 0.25 | 미달 | ✗ (창 확장 후보) |
| Track I B4 (noise) | w3 (scale 0.5/1.0) | 4/4 | 0.50 | PASS | ✓ (탐색용 pooled) |
| Track I B4 | w6 | 4×3 | 0.75–1.00 | 초과 | ✗ (과강) |

채택 config 4종 × 12ep 캡처 재실행 완료(결정론 재수집, action_token_mean+vlln_mean).
fit 표본 = 55ep (perturbed-fail 30 / perturbed-succ 25; Track I 미발화 1판 anti-circularity 폐기).

## 3. H1 — conceptor 비퇴화 진단 (`diag_conceptor_nondegen.py`, α=table14 선택값)

exp3 자연실패 fit 기준값: R-가중 이득 **0.006–0.007 (≈영행렬)**, C_succ 포화.

| fit (global) | eff.rank | gain(fit R) | gain(held R) | C_succ quota(held) | 포화 |
|---|---|---|---|---|---|
| c1_s200 L8 / L12 | 147 / 101 | 0.013 / **0.059** | 0.077 / 0.068 | 0.47 / 0.44 | 없음 |
| g1_x015 L8 / L12 | 128 / 147 | 0.012 / 0.013 | 0.073 / 0.057 | 0.45 / 0.54 | 없음 |
| p1_d003 L8 / L12 | 114 / 128 | 0.011 / 0.014 | 0.060 / 0.044 | 0.43 / 0.51 | 없음 |
| p2_f040d2 L8 / L12 | 66 / 67 | 0.005 / 0.010 | 0.080 / 0.043 | 0.47 / 0.50 | 없음 |
| b4_w3 L8 / L12 | 58 / 56 | 0.014 / 0.030 | 0.058 / 0.033 | 0.42 / 0.48 | 없음 |

- **판정: H1 방향 지지** — held-R 이득 0.033–0.080 = exp3 기준의 **5–11배**, effective rank 56–147,
  포화 없음, quota-floor(0.01) 전 config 생존, α 선택도 0.1–0.5 정상 대역. VL fit 도 overlap
  0.58–0.79 로 자연실패(밴드 상단 유착)보다 크게 벌어짐.
- ⚠️ **perm null 해석 주의**: 관측 이득이 순열 null 의 **하한 꼬리**(p_upper 0.79–1.0 ⇒ 하한
  유의) — "진짜 라벨의 fail 분포가 succ 부분공간을 순열보다 강하게 덮는" 실구조이나, fail
  record 수 우세(timeout dwell)와 얽혀 있어 **라벨 정보성의 방향 해석은 보류**. 비퇴화 자체
  (이득≫exp3·rank·비포화)는 이 주의와 무관하게 성립.
- diag 는 P0 소표본이라 held=fit(순열만 보정) — 진짜 held-out 분리는 P1 계약대로.

## 4. bridge 게이트 — 유도축 ↔ 자연축 정렬 (`bridge_axis_check.py`)

자연측 = patchceil passB 자연실패 7 / 성공 9 (scene s300033·s400020 — **cross-scene**).
참조선(자연 vs 자연, cross-scene): cos 0.43–0.59, cross-AUROC(ep) 0.81–1.0.

| 층화 | cos (L8 / L12) | perm p(양측) | cross-AUROC ep (i→n) |
|---|---|---|---|
| **pooled global** | **0.340 / 0.339** | **0.003 / 0.003** | **0.98 / 0.97** |
| P1 displace | 0.296 / 0.360 | 0.010 / 0.010 | 1.00 / 0.97 |
| G1 gripper | 0.311 / 0.313 | 0.010 / 0.020 | 0.89 / 0.86 |
| Track I b4_w3 | 0.290 / 0.327 | 0.079 / 0.020 | 0.95 / 0.91 |
| P2 force | 0.198 / 0.179 | 0.119 / 0.129 | 0.92 / 0.95 |
| C1 카메라 | 0.183 / 0.155 | 0.168 / 0.188 | 0.83 / 0.78 |
| phase: reach | 0.398 / 0.482 | 0.010 / 0.003 | 0.88 / 0.89 |
| phase: place | 0.385 / 0.445 | 0.013 / 0.003 | 0.91 / 0.91 |
| phase: grasp | 0.052 / 0.000 | 0.74 / 1.00 | 0.56 / 0.52 |

- **게이트 판정: 정렬 신호 있음 → P1 중단 조건 비발동.** pooled cos 0.34 (참조선의 60–80%),
  cross-AUROC 는 참조선과 대등.
- **모드 서열 P1>G1>b4>P2>C1**: C1(지각 OOD)은 비유의 — 24b 원문의 "시각·센서 비채택" 사유가
  데이터로 재현됨. 모든 모드의 fail 이 동일한 timeout 구조라 이 서열은 길이만으로 설명 불가
  (내부 통제).
- **grasp bin 무정렬**: 유도 실패는 reach/place 축만 자연실패와 공유 — phase-matched 관점의
  핵심 단서 (grasp 실패축은 현 메뉴로 재현 안 됨).

## 5. Confound audit (skill 규격)

| # | 게이트 | 판정 | 근거 |
|---|---|---|---|
| 1 | 길이 | **부분 — 판정 보류 항목 있음** | fail=timeout(144rec) 우세. bridge 는 episode 동등가중+phase-bin 이나 **dwell-matched 아님** → cos·AUROC 크기는 보류. 단 모드 서열·grasp 무정렬은 동일 길이 구조 내 대비라 길이만으로 설명 불가 |
| 2 | task 정체성 | N/A | 단일 task(ppcc_bread) within-task |
| 3 | instruction 균형 | N/A | 단일 instruction cell |
| 4 | in-sample rescue | N/A (SR eval 없음) | diag held=fit 명시(§3); P1 에서 3분할 계약(fit/locked) 가동, split_contract.json 누적 중 |
| 5 | rollout pooling | 통과 | feature 는 per-record 유지, phase-bin 보존. episode-mean 은 점수 집계에만 사용 |
| 6 | phase/dwell | 부분 | phase-bin 조건부 제공(reach/place/grasp), dwell-matched 는 P1 |
| 7 | 관찰≠인과 | 통과 (등급 명시) | 전 결과 diagnostic evidence 라벨. ΔSR 주장 없음 |
| 8 | scene-국소 | 주의 명기 | 유도측 단일 scene(100084); 자연측 cross-scene 전이가 검증축이나 일반화 주장 금지 |

## 6. P1 설계 함의 + 사용자 결정 요청

1. **P1 진행 여부** — bridge 게이트 통과. 진행 시 P1 본수집(채택 4 config × 40+ep, fail ≥20ep/변형).
2. **C1 처리**: 정렬 비유의(지각 OOD 재현). (a) P1 에서 제외 (b) 대조군으로 유지(정렬 실패의
   음성 대조 — 보고 가치) 중 선택.
3. **Track I**: B2 창 확장(w9/w12)으로 실패율 재캘리브레이션 여부; B4 는 w3 채택.
4. **잔여 항목**: B1(VL donor — ep_meta lang 편집 replay 경로 확인됨)·B3(OpenDrawer donor 수집)
   — 배선은 S2/S3 로 기검증, 수집만 남음. P1 과 병행 가능.
5. **개선 항목(P1 전)**: sham 출력 디렉토리 분리(스템 충돌), fit record 서브샘플링(dwell 통제,
   Gate1 공통 통제의 fit 적용), dwell-matched bridge 재계산, 자연측 VL 축 재수집(nopatch 6–8판).

## 부록 — 재현 경로

- runbook: `scripts/safe/groot_n15/robocasa/steer/induced/README.md`
- 산출물: `p0_failure_rates.tsv`, `fits/<cfg>/global/{dit_L8,dit_L12,vl}/conceptors.npz`,
  `diag/<cfg>_L{8,12}.json`, `bridge_p0.json`, `manifests/cal/`(+split_contract.json),
  `bridge_sanity/nat_vs_nat.json`(참조선)
- exp4-1 전달 계약(공유 24 §4): fit 산출 `global/dit_L{8,12}` → `<base_B>/steer/dit_L{L}` 복사
  (S5 검증 절차) — **전달은 bridge 통과 config 확정(P1) 후 권장**
