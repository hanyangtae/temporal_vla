# pq3 fit30 라운드 최종 결과 (2026-07-21 완주)

fit15 라운드(docs/steering/21) 후속: fit 재료 2배(30 rollouts/cell) + eval 2배(seen30+unseen30
=60판/cell), 6 arm × 5 cell = **1,800판** (기대개수 전판 대조 통과). 판정
`aggregate_f30/` (decision_sha ab0e3abd, `--allow-resume-tags`: 재개 실행 run_tag 상이 허용
— spec_sha·seed 대조는 전 판 강제, 위반 0).

## 사전등록 판정 (6-Holm, 단측 exact paired McNemar) — 전부 기각 실패

| 가설 | n | W/L | Δ판 | p | p_adj | CI95 상한 |
|---|---|---|---|---|---|---|
| H1 drawer | 120 | 12/5 | +7 | .072 | .43 | +0.117 |
| H2 drawer | 120 | 10/5 | +5 | .151 | .75 | +0.092 |
| H3 drawer | 120 | 5/7 | −2 | .81 | 1.0 | +0.033 |
| H1 ppcc | 180 | 10/14 | −4 | .85 | 1.0 | +0.022 |
| H2 ppcc | 180 | 14/21 | −7 | .91 | 1.0 | +0.017 |
| H3 ppcc | 180 | 12/15 | −3 | .78 | 1.0 | +0.028 |

- **null 관문 양쪽 위반**: drawer null−base=+6, ppcc −5 (margin 4) → H1 은 형식상
  해석불가. 즉 위약 요동이 ±5~6판 — H1:drawer +7 도 위약 스케일 내.
- H1 CI 상한(+0.117/+0.022) < +0.16 → **COAST 비재현, fit30 에서도 재확정**.
- final_status = all_null_nonreplication, host-block 위반 0.

## 탐색 (within=per-instruction) — fit15 패턴 유지

| arm | drawer W/L | ppcc W/L |
|---|---|---|
| perm_within | **16/5 (+11)** | 13/19 (−6) |
| gated_within | **18/9 (+9)** | 13/18 (−5) |

- drawer within 이득이 2배 표본에서 유지·강화(fit15: 7/2·7/1). 단 **위약도 drawer +6**
  — 위약 위 순증 ~+3~+5, 미보정·탐색 지위. PPCC 는 재차 음수(비재현).
- cross gated 의 fit15 dleft seen-rescue(+4/−0)는 fit30 에서 소멸·역전(+2/−3) —
  in-sample rescue 관찰이 소표본 요동이었음을 확정.
- beer: 위약 포함 전 개입 순손실(높은 base 에서 개입=해악), pizza: 천장(SR≈.97) 무신호.

## Confound audit (docs/steering/21 과 동일 게이트 — 차이만)

전 게이트 pass/N-A 동일. 추가 사항: ① 운영상 재개 이력으로 arm 당 복수 run_tag 발생 —
`--allow-resume-tags` 로 허용하되 spec_sha·(env,noise) seed 전판 대조 유지 ② host 이동
다수(48↔50↔로컬) — 판정 arm 은 cell-블록 유지(위반 0), within 탐색 arm 일부는 host 혼합
(dleft within=50, dleft base=로컬 등) 각주 ③ pizza null 은 48→50 이동(A100 간 canary 완전
일치 근거). **Claim-strength: intervention effect.**

## 결론

fit 30 rollouts·eval 60판/cell 로 검출력을 2배로 올려도 **raw 대조 conceptor 는 위약과
구분되지 않는다** (pq2→fit15→fit30 3연속 확증). 유일하게 살아남은 관찰 = **drawer 한정
within-instruction 이득(+9~+11, 위약 +6 위)** — 방향이 두 라운드 연속 유지됐으므로
후속을 원하면 confirmatory fresh-seed 라운드(사전등록) 대상이나, PPCC 비재현이 병존하므로
task-일반 method 근거로는 불충분.
