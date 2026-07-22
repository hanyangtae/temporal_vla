# wrong-grasp VL activation 분석 결과 정리 (2026-07-16)

> 한 줄 요약: **다른 물체를 잡는 wrong-grasp의 경우, (drop 이후 재탐색 구간부터) VL
> activation이 확연히 다르다 (AUROC 1.0). 다만 표본이 6 vs 4라 우연일 확률이 7% 있다.
> 우연이 아니라면 wrong-grasp으로 가는 상태를 발생 20~44 records 전에 경보할 수 있다.**

## 1. 배경·가설

- 가설: "wrong grasp(목표물 대신 다른 물체를 잡는 실패)은 VL 단의 오류일 것 →
  wrong-grasp 에피소드는 나머지(succ + 다른 실패)와 VL activation 분포가 다를 것이다."
- 동기: online failure-type 식별 (docs/steering/14의 중심 미해결 문제) — 유형을
  온라인에 읽을 수 있어야 steering을 라우팅할 수 있다.

## 2. 데이터

- `phase_event_6p/.../ppcc_bread` 단일 cell (bread, 고정 seed·instruction), 60 에피소드
  = 성공 48 / 일반실패 5 / **wrong-grasp 실패 7**.
- 시간 단위: **1 record = 1 inference step** (get_action 1회 = 5 env-step 실행,
  chunk 16 예측/5 실행 표준). feature·phase 라벨 모두 per-record.
- wrong-grasp 라벨 = 7-phase event labeler의 phase (¬target-grasped ∧ distractor-grasped).
  영상 육안검증 3/3 일치 (배·칼꽂이 파지 확인).
- wrong-grasp은 이 cell 국소 현상: apple 실패 80개, potato 실패 54개, patchceil bread
  실패 77개에서 전부 0건.

## 3. 발생 메커니즘 (분석 전에 실증한 사실 — 실험설계의 근거)

per-record phase 시퀀스 + drop/grasp 이벤트 + 영상으로 확인. wg 7개 중 6개가 동일 패턴:

```
reach → grasp(bread 정상 파지) → place(운반) → drop(rec 19~38)
   ├─ bread가 counter에 남음(시야 안) → 재탐색 8~11 records → 재파지 → 성공   [succ-drop]
   └─ bread가 바닥으로 소실(시야 밖) → 재탐색 19~44 records → 배 파지 = wrong-grasp
```

- 즉 wrong-grasp은 **처음부터 목표를 오독한 게 아니라**, 정상 파지 후 운반 중 미끄러뜨린
  뒤 목표물이 시야에서 사라지면서 생기는 **2차 사건**. 갈림은 drop 물리(낙하 지점)가 결정.
- 예외 1건(ep39): drop 없이 insert 후 재탐색 중 wg — event 비교에서 제외.

## 4. 실험설계

핵심 아이디어: "언제의 VL을 비교하느냐"가 결론을 좌우하므로 **윈도를 3단계**로 나눔.

| 윈도 | 정의 | 무엇을 검정하나 |
|---|---|---|
| W_pre / W_early | 에피소드 초기 reach (첫 4~10 records) | "처음부터 VL이 오독하나?" |
| **W_postdrop (핵심)** | **drop 직후 ~ 다음 파지 전 재탐색 reach** | "같은 사건(drop)을 겪은 뒤, wrong-grasp으로 갈 에피소드의 VL이 다른가?" — **event-state를 맞춘 공정 비교** |
| W_at / t_rel | wrong-grasp 중 / 직전 | 사후 판독 (양성 대조) |

비교군·통제 장치:

- **W_postdrop 비교군 = drop을 똑같이 겪은 에피소드만** (succ-drop). drop을 안 겪은
  에피소드는 비교할 상태 자체가 없으므로 정의상 제외. wg 6 vs succ-drop 4
  (succ-drop 6개 중 2개는 drop 직후 재탐색 구간이 없어 탈락).
- **길이 통제 (budget)**: wg는 재탐색이 훨씬 길다(19~44 vs 8~11 records) — 그대로 평균하면
  "탐색 길이"를 분류하는 꼴. 그래서 전 에피소드에서 **재탐색 첫 8 records만**(포함
  에피소드의 최소 길이 = budget 8) 잘라 평균 → 에피소드당 1벡터. 길이 경로 차단.
- **통계**: 2048차원 vs 표본 10개라 in-sample 판별은 무의미(아무 라벨이나 완벽 분리) →
  leave-one-out으로 out-of-sample 점수 생성, 유의성은 **exact permutation**(가능한 라벨
  배치 210가지 전수)으로 판정.
- 그 외: DiT 7개 layer 동일 분석(VL 특이성 대조), dwell(재탐색 길이) AUROC 병기,
  budget sweep, 라벨-fit 없는 Mahalanobis 교차확인.

## 5. 결과

| 비교 | VL AUROC | p | 해석 |
|---|---|---|---|
| 초기 reach (W_pre, wg7 vs rest53) | 0.437 | 0.72 | 무신호 — **메커니즘상 당연** (초기엔 아무 문제 없음) |
| 에피소드 절대 초기 (W_early 5/10) | 0.42/0.51 | n.s. | 초기조건형 아님 |
| **drop 후 재탐색 (W_postdrop, wg6 vs succ-drop4, 첫 8 records)** | **1.000** | **0.071** | **완벽 분리 — 다만 6 vs 4에선 라벨을 우연히 섞어도 이런 극단 분리가 나올 확률 7%라 통계 확증은 불가** (그림 `postdrop_separation.png`: LOO score strip 겹침 0 + 비지도 PCA에서도 PC1 축 분리) |
| wrong-grasp 중 (W_at) / 직전 (t_rel −1~−5) | 1.000 | — | 사후 판독 (당연; 파이프라인 유효성 확인용) |

주의 깊게 볼 것:

- **dwell도 AUROC 1.0**: 재탐색 "길이"만으로도 두 군이 갈린다 (wg가 오래 헤맴).
  budget으로 길이 유입은 차단했지만, 이 표본으로는 "VL 내용의 분리"와 "탐색이 길어지는
  상황의 분리"를 완전히 떼어 확증할 수 없음.
- **분리의 정체 (가장 그럴듯한 해석)**: drop 후 bread가 시야에 있느냐/없느냐를 VL이
  정확히 읽는 것 — 즉 "VL이 잘못 읽어서 wrong grasp"이 아니라 **"물리적 target 소실을
  VL이 반영"**. 원래 가설(VL 오독 원인론)은 이 cell에서 기각.

## 6. 그래서 경보(online 검출)가 되나?

**우연이 아니라면 — 된다, 그것도 wrong-grasp 발생 전에.** 시간축으로 보면:

```
drop 발생 ──[재탐색 8 records(≈40 env-step): VL로 판독 가능(AUROC 1.0)]──
        ──[재탐색 13+ records: dwell만으로도 경보 가능]──
        ──[재탐색 19~44 records 후]── wrong-grasp 발생
```

- **VL 경보의 부가가치 = 시점**: 재탐색 시작 후 8 records만에 갈라진다 → dwell 경보
  (13 records는 기다려야 wg 영역)보다 이르고, wg 발생까지 11~36 records의 개입 여유.
- 현실적 경보 설계: **drop 이벤트(proprio/gripper) 감지 → post-drop VL target-lost 판독
  → (보조) 재탐색 dwell** 3단 조합. drop을 겪고도 성공하는 경우(succ-drop)가 있으므로
  drop만으로 경보하면 FP.
- steering 관점: 개입 지점은 재탐색 구간. 단 bread가 물리적으로 소실된 상태라면 어떤
  개입도 과제를 복구할 수 없음 — 이 경우 경보의 용도는 "구제"가 아니라 **조기 중단/재시도
  트리거** 쪽이 자연스러움.

## 7. 한계·후속

- **표본**: 6 vs 4 — 완벽 분리로도 p=0.071이 최소치. 확증하려면 drop-경험 에피소드 확대
  필요 (succ-drop n≥10이면 exact p<0.05 도달 가능).
- **단일 cell** (1 task×1 object×1 seed×1 instruction) — 일반화 불가, cell-내 존재 증명.
  exp3(구 pq3) 라운드 종료 후 exp3 셀(bread/beer/pizza_cutter) census → wg 있으면 동일 스크립트
  무변경 복제.
- "VL이 bread 가시성을 읽는다"는 해석의 직접 확인: sim state의 bread 위치(GT)와 post-drop
  VL 신호의 대응 검증이 후속 실험.

## 부록: confound-audit

| 게이트 | 판정 | 근거 |
|---|---|---|
| 1 길이 | pass(주의) | budget 8 균일 pool; 단 dwell AUROC 1.0 — 내용/길이 기여 분리 확증 불가 병기 |
| 2 task | N-A(통제) | 단일 cell |
| 3 instruction | N-A(통제) | 고정 |
| 4 in-sample | pass | LOO out-of-sample + exact permutation |
| 5 pooling | pass | per-record 유지, 윈도 내 budget pool만 |
| 6 phase/dwell | pass | event-state 매칭 (drop 직후 재탐색끼리) |
| 7 관측≠인과 | pass | 가시성-판독 해석 명시; "VL 오독 원인" 주장 안 함 |
| 8 scene-국소 | flagged | 1 cell; wrong-grasp 자체가 bread cell 국소 현상 |

Claim strength: **diagnostic evidence**.

산출물: 스크립트 `scripts/safe/groot_n15/robocasa/analyze/wrong_grasp_vl_separation.py`,
결과 `outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/wrong_grasp_vl_separation/ppcc_bread/`
(JSON + 그림 5종: **postdrop_separation.png(보고용 핵심 — LOO strip + 비지도 PCA)**,
layer_profile_wpre / trel_curve / budget_sensitivity / dwell_distributions).
JSON의 `postdrop.drop_succ.per_episode`에 에피소드별 LOO 점수·PCA 좌표·재탐색 dwell 수록.
