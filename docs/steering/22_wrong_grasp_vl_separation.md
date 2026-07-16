# wrong-grasp VL activation 분리 분석 (sub 분석, 2026-07-16, v2 메커니즘 반영 재판정)

질문: wrong-grasp(distractor 파지) 에피소드의 VL activation이 나머지(succ+other-fail)와
분포가 다른가 — online failure-type 식별(docs/steering/14) 근거.

스크립트: `scripts/safe/groot_n15/robocasa/analyze/wrong_grasp_vl_separation.py`
데이터: 원격 `phase_event_6p/.../ppcc_bread` (succ 48 / other-fail 5 / wg 7, 단일 cell).
결과: `outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/wrong_grasp_vl_separation/ppcc_bread/`
wg는 이 cell에만 존재 — apple(80 fail)·potato(54 fail)·patchceil bread(77 fail) 전부 wg 0.

## 발생 메커니즘 (event 시퀀스 + 영상 실증)

wg 7개 중 6개가 동일 패턴 (라벨 육안검증 3/3, drop 전후 프레임 wg/succ 각 1건 확인):

```
reach → grasp(bread 정상 파지) → place(운반) → drop(rec 19~38)
  → bread 낙하 지점이 갈림을 결정:
    · counter에 남음(시야 안) → 재탐색 8~11 records → bread 재파지 → 성공 (succ-drop 6)
    · 바닥으로 소실(시야 밖) → 재탐색 19~44 records → distractor(배) 파지 = wrong-grasp
```

즉 wg는 **초기 goal 오독이 아니라 manipulation slip에서 파생된 2차 goal 재결합**.
예외 1건(ep39): drop 없이 insert 후 재탐색 중 wg — event-matched 분석에서 제외.

## 결과 (재판정)

| 검정 | AUROC | p | 해석 |
|---|---|---|---|
| W_pre (초기 reach 4 records) VL, wg7 vs rest53 | 0.437 | 0.72 | 무신호 — **메커니즘상 당연** (초기엔 아무 문제 없음; 가설 반증 아님) |
| W_early (절대 초기 5/10) VL | 0.42/0.51 | n.s. | 초기조건형 아님 |
| **W_postdrop (event-matched): wg6 vs succ-drop4, budget 8** | **1.000** | 0.071 (exact 210) | **drop 후 재탐색 구간에서 완벽 분리** — 단 n=10이라 완벽 분리로도 α=0.05 불가(최소 달성 p 0.07), descriptive-strong |
| W_postdrop drop_all (ofail-drop 포함, budget 1로 붕괴) | 0.643 | n.s. | ofail 재탐색이 1~5 records라 설계 잡음 |
| W_at / t_rel −1~−5 | 1.000 | — | onset 근접 사후 판독 (절대시간 confound 플래그) |

**dwell 경고**: W_postdrop dwell(재탐색 길이) AUROC도 1.000 — wg는 재탐색이 길다(19~44 vs
8~11). equal-budget(첫 8 records)으로 count 경로는 차단했지만, 표본상 "내용 분리"와
"탐색 길이"의 기여를 분리 확증 불가. **VL 없이 재탐색 dwell만으로도 구분됨**을 병기.

## 판정

1. **"wrong grasp = VL 단 오류" 원인론은 이 cell에서 지지되지 않음.** 갈림은 drop 물리
   (bread 가시성)가 결정하고, VL 분리는 그 상태의 시각 판독으로 설명 가능 — VL이 잘못
   읽은 게 아니라 물리적 소실을 정확히 반영.
2. **VL 분포 차이는 실재하며 위치는 post-drop 재탐색 구간부터** (event-matched AUROC 1.0,
   onset 20~44 records 전에 이미 분리) — "예측 신호"가 아니라 **위험 상태(target-lost) 검출**.
3. 실용 함의: online wrong-grasp 라우팅은 [drop 이벤트(proprio/gripper) + post-drop VL
   target-lost 신호 + 재탐색 dwell]의 조합으로 onset 전 검출 여지. steering 개입 지점 =
   재탐색 구간. succ-drop도 재탐색을 거치므로 FP 설계 필요 (dwell ≥ ~13 records가 경계).

## confound-audit

| 게이트 | 판정 | 근거 |
|---|---|---|
| 1 길이 | pass(주의) | equal-budget pool; 단 W_postdrop dwell AUROC 1.0 — 내용/길이 기여 분리 확증 불가 병기 |
| 2 task | N-A(통제) | 단일 cell |
| 3 instruction | N-A(통제) | 고정 |
| 4 in-sample | pass | LOO + episode-perm(exact) |
| 5 pooling | pass | per-record, 윈도 내 budget pool만 |
| 6 phase/dwell | pass | event-state 매칭 (drop 직후 재탐색끼리 비교) |
| 7 관측≠인과 | pass | 가시성-판독 해석 명시, 오독 인과 주장 안 함 |
| 8 scene-국소 | flagged | 1 cell 존재증명; wg 자체가 bread cell 국소 현상 |

Claim strength: **diagnostic evidence** (W_postdrop 1.0은 표본 한계로 통계 확증 불가 병기).

## 후속

- pq3 라운드 종료 후 pq3 셀 census → wg 있으면 동일 CLI 무변경 복제 (W_postdrop 포함).
- 확증에 필요한 것: drop-경험 에피소드 표본 확대 (succ-drop n≥10이면 exact p<0.05 가능).
- target-lost 검출기(post-drop VL) 설계는 별도 실험으로 — bread 가시성 GT(sim state)와
  VL 신호의 대응 확인이 선행.
