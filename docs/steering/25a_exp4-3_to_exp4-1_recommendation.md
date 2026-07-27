# exp4-3 → exp4-1 권고: 선형 latent 연산자 중단, SAE scene-feature 분리로 전환

작성 2026-07-26, exp4-3(분리도 지도) 세션. 수신 = exp4-1 세션. 사용자 결정 반영:
**다음 본류 = SAE scene-feature 분리** (Cosmos value best-of-N 은 나중 항목, §5 한 줄 기록만).

## 0. 결론 (사용자 확정)

1. **통합(평균+분산) 연산자 및 신규 선형 latent arm 추가 반대.** 이유 §2-3.
2. **다음 본류 = SAE scene-feature 분리** — "분리신호에서 scene 성분을 명시적으로 빼면
   outcome 신호가 남는가"를 표현 수준에서 직접 규명. 설계 가이드 §4.
3. 조건부 저비용 arm 하나만 예외적으로 유지 검토: same-scene × setM (§3, exp4-2 인프라).

## 1. 현 상황 (exp4-1 자체 보고)

N1.5에서 setM(평균)·conceptor(분산)·gated/perm 조합, 단일 layer 선택, future-token만 steering,
token 종류별 연산자 혼합 동시 steering까지 — **전부 noise_resample 미만의 구제율.**

## 2. 선형 연산자 가족은 공정한 시험을 소진했다 (null 이력 매트릭스)

| 축 | 시도 | 결과 |
|---|---|---|
| 연산자 | conceptor(분산) / setM(평균) | 둘 다 null |
| fit 범위 | cross-scene 자연실패(exp3·exp4-1) / **same-scene**(exp2, scene-fixed 115arm) / perturbed(exp4-2 smoke) | exp2·exp3 위약 동급, exp4-1 noise 미만 |
| 변형 | layer sweep·phase-gated·6/7-phase·token 종류별·multi-layer | 전부 null (multi-layer +0.20 은 in-sample 아티팩트) |

특히 **exp2 가 same-scene 대조 fit 로도 위약 동급**이었으므로 "scene confound 제거가 답"이라는
가설은 이미 기각돼 있다. 통합연산자는 같은 가족의 파라미터 증가 변형이며, 아래 §3-(i) 때문에
분산 성분은 null 축을 fit 에 추가하는 것과 같다.

## 3. exp4-3 실측이 주는 설명 (왜 null 인가)

3-모델(N1.5 / N1.6 / Cosmos-Policy-RoboCasa-Predict2-2B) × 동일 셀(drawer·bread) atlas 실측:

- **(i) 분산축은 모델 불변으로 비어 있다**: conceptor 형 var_z(held-out R-가중 이득 vs 순열 null)가
  3모델 전부 퇴화(|z|<2, 후반층 음수). COAST quota 도 3모델 전부 중간층 peak 없음(단조 감소).
  → 통합연산자의 분산 성분·conceptor 계열은 fit 할 실체가 없다.
- **(ii) 평균분리는 실재·강하나(z 5~15) 위치가 모델·task 의존**: bread 에서 N1.5 중간 L8-12 →
  N1.6 L31 → Cosmos L24 로 이동. phase 별 판별 phase 도 task 마다 다름(Cosmos: drawer=grasp-handle
  AUROC 0.90, bread=reach-to-object 조기층 kl_z 17.5). 읽기 신호는 크지만 세 라운드의 write null 이
  이 신호의 **비인과성(결과-상관 구조: scene·진행도 지배)** 을 방증.
- **(iii) noise_resample 이 informed steering 을 이긴다** = 구제 가능한 실패 대부분이 확률적-경계
  (아무 교란이나 재추첨이면 살아남). informed 가 이기려면 결정론적 실패를 살려야 하는데, onset-regime
  분석상 그 상당수는 초기조건형(지각·기하 기원) — DiT motor latent 를 밀어 구제할 대상이 아님.

**조건부 예외 (본류 아님)**: same-scene × setM 은 유일한 미검증 조합(exp2 는 setM 이전이라
conceptor 만, exp4-1 setM 은 cross-scene fit). exp4-2 perturbed 쌍 인프라에 mean-diff fit 하나
얹으면 되는 저비용 arm. 단 exp2 null + exp4-1 setM null 위라 기대치는 낮게, 사다리 게이트로만.

## 4. SAE 라인 구체 설계 가이드 (권고 본류)

**질문 재정식화**: succ/fail 분리신호에서 scene(암기) 성분을 명시적으로 제거하면 outcome 신호가
남는가. 남으면 그 잔여 방향이 steering 후보. 안 남으면 — 그 자체가 "latent steering 서사 종결"의
판정 근거가 되는 양방향 가치 실험.

**사다리 (각 게이트 통과 시에만 다음 단계)**:
- **G1** SAE 학습 + scene feature 식별. 검증 = scene 라벨(layout/style/scenario_seed) 예측력으로
  "이 feature 들이 실제 scene 을 인코딩"을 실측. (분리도만 보고 넘어가지 말 것.)
- **G2** scene-잔차화 후 succ/fail **read** 분리 잔존 검증 — 길이/phase 통제(dwell cap) 유지,
  episode-라벨 순열 null, held-out. 잔존 z 유의해야 통과.
- **G3** 잔여 방향 **write** (oracle rescue 규약 그대로: 위약·noise_resample arm 동시, fit-seed
  분리 held-out, EVAL_SEED=100000).

**데이터 자산 — 추가 수집 없이 시작 가능**:
| 데이터 | 위치 | 계약 | 비고 |
|---|---|---|---|
| N1.5 fit30 5셀 | 승준 HDD | [L,K,T,D=1536] | 기존 atlas/probe 와 동일 풀 |
| N1.6 90ep (3셀) | 동규 `~/pkt_ws/datasets/exp4_3_n16/` | **[32,51,1536] T보존** | per-token SAE 가능 (토큰 pool 금지 지시로 보존해 둠) |
| Cosmos 190ep (2셀) | 동규 `~/pkt_ws/datasets/exp4_3_cosmos/` | [28,2048] action-frame | 3모델 교차검증용 |

layer 선택은 atlas mean_z peak 기준(모델별: N1.5 L8-12 / N1.6 L12·L31 / Cosmos L24).
atlas JSON·figs = exp4-3 브랜치 `outputs/eval/robocasa/{groot_n15,groot_n16,cosmos}/exp4_3/`.

**선행 노트**: `docs/references/reading_notes/` SAE 3논문(Dr.VLA / Event-Grounded /
Observing&Controlling) — 셋 다 outcome-vs-scene 분리를 직접 풀지 않음(니치 확인), DiT 는 단일
feature steer 붕괴 소견 → 다차원 잔차화 접근 정당. conceptor 종결 문서의 "다음 = SAE scene 분리"
로드맵과 정합. **실패 대량수집**(종결 문서의 다른 반쪽)은 SAE 와 병행 검토 — scene 당 실패 표본
수가 G2 검정력을 결정한다.

## 5. 나중 항목 (기록만)

- Cosmos value best-of-N: Cosmos 는 action 과 함께 value 추정을 생성 — best-of-N 재추첨 +
  자기-value 선택은 "noise_resample + informed selection" 이라 기준선 대비 판정이 깨끗한
  개입-가족 전환 카드. **지금 하지 않음** (사용자: 나중).
- Cosmos steering 자체도 같은 이유로 보류 — 읽기 근거는 3모델 중 최강(mean_z L24=14.77)이나
  연산자 문제가 먼저.
