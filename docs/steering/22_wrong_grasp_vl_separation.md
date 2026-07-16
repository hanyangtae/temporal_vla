# wrong-grasp VL activation 분리 분석 (sub 분석, 2026-07-16)

질문: wrong-grasp(distractor 파지) 에피소드의 VL activation이 나머지(succ+other-fail)와
분포가 다른가 — 특히 **발생 전(pre-onset)** 에 이미 다른가 (online failure-type 식별 근거).

스크립트: `scripts/safe/groot_n15/robocasa/analyze/wrong_grasp_vl_separation.py`
데이터: 원격 `phase_event_6p/.../ppcc_bread` (succ 48 / other-fail 5 / wg 7, 단일 cell).
detector 라벨 육안 검증 3/3 (squash·칼꽂이 파지 확인). potato(fail 54)·apple(fail 80)은 wg 0.
결과: `outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/wrong_grasp_vl_separation/ppcc_bread/`

## 결론 (diagnostic evidence, cell-내 존재 증명)

1. **Primary NO-SIGNAL**: VL × W_pre(첫 4 reach records, 탈락 0) × wg vs rest-all —
   LOO AUROC 0.437, perm p=0.72 (null95 0.836). **reach 초기의 VL은 wg를 구분 못함.**
2. **W_early(절대 초기 5/10 records)도 무신호** (VL 0.42/0.51) — 초기조건형 아님.
3. **onset 근처에서는 완벽 분리**: positive control W_at AUROC 1.000;
   onset-정렬 t_rel −1~−5 single-record AUROC 1.0 (진단 전용).
   단 t_rel 비교는 절대시간 confound 있음 (wg onset t≈41–83 vs succ grasp t≈4–20).
4. budget sweep에서 b=13(비교군 8개로 축소, 45 탈락)일 때만 0.875 (p=0.106, 비유의) —
   onset에 가까운 늦은 reach 내용이 갈라진다는 t_rel 서사와 정합, selection-bias 영역이라 참고만.
5. wg vs other-fail (7v5, exact perm, descriptive 전용): 전 윈도 비유의.

**해석**: 이 cell에서 VL의 wg 신호는 "이미 궤적이 갈라진 뒤의 시각 결과 판독"이며
**조기(예측적) goal-오독 신호의 증거는 없음**. 단 n_wg=7 검정력 한계로 AUROC ≳0.84만
검출 가능 — "no signal ≠ no effect". 주장 상한: VL은 wg를 **onset 부근에서** 선형 판독
가능(사후 모니터로는 유효), "goal 오독이 VL에서 발생" 인과 주장 불가.

## confound-audit

| 게이트 | 판정 | 근거 |
|---|---|---|
| 1 길이 | pass | equal-budget pool(b=4, 탈락 0); dwell AUROC 0.949는 context로만 병기. t_rel 진단은 절대시간 confound 플래그 |
| 2 task | N-A(통제) | 단일 cell within-task |
| 3 instruction | N-A(통제) | 고정 instruction |
| 4 in-sample | pass | 분석 전용, LOO out-of-sample + episode-permutation |
| 5 pooling | pass | per-record 유지, phase-내 budget pool만 |
| 6 phase/dwell | pass | reach phase 내 비교 + dwell-matched subset(0.20, 비유의) |
| 7 관측≠인과 | pass(라벨) | diagnostic evidence, claim ceiling 명문화 |
| 8 scene-국소 | flagged | 1 cell — 일반화 불가, pq3 종료 후 복제 예정 |

## 후속

- pq3 라운드 종료 후 pq3 셀(bread/beer/pizza_cutter) census → wg 있으면 동일 CLI 무변경 복제.
- online failure-type 식별 관점: wg는 **onset 이후** VL로 안정 검출 가능(사후) —
  phase-matched steering의 트리거로는 "예방"이 아니라 "발생 후 rescue" 라우팅에 해당.
