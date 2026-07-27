# 32. G2 결과 — scene 잔차화 후 succ/fail read (drawer_right scene-matched, L12)

작성 2026-07-27, exp5 세션. 데이터 = exp5-3 수집 drawer_right scene-matched 160판
(scenario_seed 20 × inference_seed 8, 승준). 코드 = `scripts/scene_sae/g2_residual_read.py`
(exp5-3 within-scene LOSO 프로토콜 문자 그대로 이식, codex 리뷰 5건 반영).
판정 프레임 = 대시보드 제안(창평균 0.847을 임계값으로 쓰지 않고, (a) 재현 앵커 →
(b) 잔차화 전 → (c) 잔차화 후 곡선 변화로 판정).

**⚠ 31 문서와의 관계**: 셀이 다르다(31=drawer_left fit30, 32=drawer_right scene-matched).
직접 비교가 아니라 재현·확장 관계.

## 0. 세 줄 결론

1. **배관 앵커 정합**: 우리 재계산 raw|all 창평균 = **0.847**, t=0 = **0.729** — exp5-3
   수치와 일치. 이후의 모든 차이는 배관이 아니라 처리의 차이다.
2. **outcome 신호는 scene 성분이 아니다**: train-scene 라벨 부분공간(rank 12~19)을 제거하면
   본 scene 식별은 궤멸(0.81→0.16)하는데 succ/fail read 는 **유지+상승**(0.847→0.88~0.90,
   null_z ≤5.3). within-scene 설계·t=0 증거와 합쳐 **"scene 과 무관한 outcome 신호 실재"**.
3. **scene 암기는 scene-특이적**: 어떤 제거(선형 rank19·SAE selective 35)도 **unseen scene 의
   식별 정보를 못 지운다**(fold별 probe 0.94~0.99 유지). 공유 저차원 "scene-coding" 축이
   없다는 뜻 — "SAE 로 scene feature 를 골라 제거"라는 전제의 실질 제약.

## 1. 설정

- 입력: `build_sae_inputs.py --scan-dir --split-by scene` 산출 (행 897k, per-token, K평균,
  fingerprint 대조). SAE = R_L12_m6144_k64_aux0_split_scene (scene축 학습, dead 0.166).
- read 프로토콜(전 arm 공통, analyze_sm2 이식): 창 [0,38) record, 혼재 scene만,
  within-scene 방향 + held-out scene 중심화 LOSO, episode 단위 AUROC, scene 내 episode 순열
  null (200/100회).
- record 집계: future 세그먼트 평균(1차, G1 근거) / 49토큰 전체 평균(=exp5-3 동치, 앵커) /
  action 세그먼트 (민감도).
- arm: raw / sae_full(재구성만) / sae_topN(selective 제거 후 재구성; selective 35개 =
  layout-라벨 quick probe 산정) / linear_between·linear_logreg (scenario_seed 라벨 부분공간
  r∈{1,2,4,8,12,16,19}, LOSO fold 내 재추정 — rank 상한 19 = 20클래스 이론 한계).

## 2. 결과 (창평균 AUROC | null_z | scene probe: 본scene참고 / fold별 unseen)

| arm (all-agg) | AUROC | null_z | scn(본) | scn(unseen) |
|---|---|---|---|---|
| raw | 0.847 | 4.2~4.4 | 0.812 | 0.992 |
| sae_full | 0.845 | 4.4 | 0.646 | 0.965 |
| sae_top10~all | 0.844~0.845 | 4.2~4.7 | 0.666~0.757 | 0.957~0.963 |
| linear_logreg r1~19 | 0.846~0.853 | 4.1~5.1 | 0.683~0.728 | 0.990~0.994 |
| **linear_between r12** | **0.884** | **4.97** | **0.162** | 0.968 |
| **linear_between r16** | **0.885** | **5.04** | **0.162** | 0.939 |
| **linear_between r19** | **0.879** | **5.05** | **0.162** | 0.995 |

future-agg 도 동일 패턴 (raw 0.800 → between r16 0.900/null_z 4.9; 본scene 0.767→0.339).
t=0 값은 전 arm 0.66~0.73 (raw|all 0.729 = exp5-3 일치). 공식 verdict 는 전 arm
`undecidable_removal_failed` — fold별(unseen) 기준을 만족하는 arm 이 없기 때문. §3-3 참조.

## 3. 판독

1. **(b)→(c) 판정: outcome 잔존 + 개선.** between r12~19 는 본 scene 식별을 20클래스
   우연(0.05) 근처(0.16)까지 지우는데 read 는 0.88~0.90 으로 **올라간다**. scene 구조가
   succ/fail 방향 추정의 잡음이었다는 뜻. 잔여 방향 = G3(write) 후보.
2. **SAE arm 은 동기였던 작업에서 선형에 진다**: selective 35 제거는 본 scene 식별도
   못 지움(0.65→0.67~0.76). 단서: selective 산정이 layout 라벨(5클래스)·quick probe 기반 —
   scenario_seed 기반 selectivity 로 재산정할 여지는 있으나, ③ 의 전이 문제는 그대로 남는다.
3. **unseen-scene 제거는 구조적으로 실패**: fold Q 는 train scene 라벨로 추정되므로 새 scene 의
   고유 방향을 원리적으로 못 담는다. 실측 0.94~0.99 유지가 그 증거. 해석: **scene 암기가
   scene 간 공유 부분공간이 아니라 scene 별 고유 방향에 저장**된다. 이는 (i) "scene feature
   제거 후 일반화" 접근(선형이든 SAE 든)의 근본 제약이자, (ii) within-scene 중심화(제거가
   아니라 scene 별 offset 상쇄)가 올바른 통제였다는 사후 정당화다.
4. **verdict 라벨 재해석**: `undecidable_removal_failed` 는 "unseen scene 식별까지 지워야
   성공"이라는 사전 기준의 산물. 위 ③ 이 구조적 불가능임이 밝혀졌으므로, G2 의 과학적 질문
   ("scene 과 무관한 outcome 신호가 있는가")에는 **본scene 제거 + read 잔존 + t=0 증거**로
   답한다: **있다**. 단 이 재해석은 사후이므로 명시해 둔다.
5. t=0 (같은 scene 이면 관측 동일, denoise noise 만 다름) = 0.729, within-scene 계산임을
   코드로 확인(승준 analyze_sm2.py loso/perm_null — 방향·중심화·순열 전부 scene 내부).
   noise 추첨이 최종 성패를 예측한다는 뜻이며, 이 신호가 "실패 원인 방향"인지 "운 좋은
   노이즈 방향"인지는 G3 개입만이 가른다.

## 4. G1-재판정 (drawer_right, scenario_seed 20클래스) — 진행 중

episode축 SAE(L12/L10/L8) × {scenario_seed, layout_id} probe 순열 50회가 실행 중
(순열당 ~90s). 완료 시 이 절 갱신. 예비 신호: scene축 SAE 의 layout probe 가
CV(본 scene) 0.99 vs held-out scene 0.63 — scene 일반화 격차 큼(§3-③ 과 정합).

## 5. 다음 단계 (사용자 결정 대기)

- **G3 후보 확정**: between r12~19 잔차화 + within-scene mean-diff 방향으로 write
  (oracle rescue 규약: noise_resample 동시, fit-seed 분리, EVAL_SEED=100000).
  실무상 within-scene 설계 자체가 scene 통제이므로 "잔차화 없이 within-scene 방향"과
  "잔차화 후 방향"을 나란히 fit 해 비교하는 게 정보량 최대.
- mixer(160판, NPZ 승준 도착)·beer(160판)로 §2 표 재현 → 3-cell 일반화.
- SAE 노선 판단 자료: scenario_seed 기반 selectivity 재산정 1회 시도 후에도 §3-② 가
  유지되면, scene 분리 목적의 SAE 는 종결하고 SAE 는 feature 해석 용도로 강등.

## 재현

- 산출물: `outputs/eval/robocasa/groot_n15/scene_sae/scene_matched_drawer_right/`
  `g2_L12_scene.json`(r≤8·SAE arm, n_perm 200) / `g2_L12_scene_r19.json`(r 12~19, n_perm 100)
- 명령: 위 디렉토리 파일들의 `source` 필드에 전체 argv 기록. seed 0, GPU3.
