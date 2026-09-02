# 상우(task_classification) cluster 할당 함수 이관 — 연산자 설계 세션 정면 비교용

출처: `sangwoo_desktop`(166.104.67.158, ssh config 별명) docker `task_classification`,
run `runs/ae-log_likelihood-union-K48-s0`. 회수일 2026-08-31.

## 목적

연산자 설계 세션 제안: 같은 로그 위에서 [cluster-소속 채점 vs 조건부 가우시안(AE-16
클래스별 log-우도비) 채점] 정면 비교. 이 디렉토리가 cluster 쪽 할당 함수 전체다.

## 할당 파이프라인 (스텝 독립 = 구조적 causal)

```
GR00T hidden state (layer 12, denoise 3 캐시)  [F=1536]
  → PCA 64 + whitening (pca.npz: mu[F], V[64,F], sqrt_lam[64]; train에서만 적합)
  → AE 인코더 (model.pt state_dict; mlp hidden 256, latent 16)
  → kmeans 최근접 중심 (model.pt kmeans_centers [48,16], `active` 마스크 적용 후 L2 argmin)
```

- 코드 사본: `run_ae-ll-union-K48-s0/code/` (pca 적용 = `pca_phase_data.py`,
  latent·assign = `inference_phase_analysis.py`의 `latents`/`assign_states`,
  AE 정의 = `autoencoder_phase_models.py`)
- `model.pt` 키: `state_dict, config, summary, kmeans_centers, gmm_means, active`
- 학습 데이터: phase_cls_pq3 / cell_union.json split (pq3 5-cell, 150ep), cluster는 train만.
- 원 레포 함정: `failure_detection/assign.py` 말미에 한글 자모 'ㄴ' 혼입으로 import 깨짐
  (2026-08-31 기준) — 로직 인라인으로 우회할 것.

## 파일

| 파일 | 내용 |
|---|---|
| `cluster_phase_outcome_unionK48s0.csv` | cluster×(GT phase 다수결·purity)×(succ/fail 쏠림) join 표 — 본 세션 산출 |
| `run_ae-ll-union-K48-s0/model.pt` | AE state_dict + kmeans_centers(48×16) + active |
| `run_ae-ll-union-K48-s0/pca.npz` | PCA64+whiten 파라미터 (cell_union train 적합) |
| `run_ae-ll-union-K48-s0/.hydra/config.yaml` | 전체 run 설정 |
| `run_ae-ll-union-K48-s0/derived_meta.json` | derived 캐시 메타 (L12-D3-pca64w-cell_union) |
| `run_ae-ll-union-K48-s0/code/` | 재구현용 코드 사본 |

## 판정 결과 요지 (K=48, union 150ep = succ 101 / fail 49)

- 같은 GT phase 안 3분류 실재 — grasp: succ={19,10,26} / 공용={14,16} / fail-전용={15,21,37}
  (fail-전용은 t<45 길이통제 대조 생존 = 진짜 실패 서명).
- reach-handle(drawer): succ-쏠림 cluster 부재, 성공 경로는 purity 1.0 공용 corridor(13,46,5,39).
- 제외: 1/22/27(깨진 관측 3ep의 데이터 결함 서명), 0/6/35/43(늦은 시점 전용 = 시간 아티팩트).
- fail-전용 cluster는 내부 성공 스텝 0 → within-cluster 대조 fit 구조적 불가.

토론 정본(양 세션 합의): fail-only 진입=발화 게이트, cluster taxonomy=재샘플 후보 채점
참조(write 타깃 아님). 관련 판정: 활성화 write 계열 null(44§7)·재샘플 verifier null(39).

## 영상 판독 판정 (2026-09-01, 사용자 눈 판정 + cell·시간·grasped 교차검증)

판독대 아티팩트(b7529f51…, K48 좌표) 기준. 교차검증 수치 = cell 구성·t/T 중앙값·grasped 비율.

| cluster | 사용자 판독 | 교차검증 | 판정 |
|---|---|---|---|
| c21·c37 | "놓친 직후, 서로 비슷" | **cross-cell**(beer/bread/pizza)·t/T 0.72/0.46·grasped 0.06/0.09 (기저 0.28) | 관계적 이벤트 상태(빈손 grasp 국면 장기체류) — scene 정체성 아님 |
| c15 | "다양해 보임" | 사실상 **bread 전용**(165/167스텝; drawer·beer는 1스텝씩) | bread-특이 실패 모드, cross-scene 주장 불가(초기 보고 정정) |
| c19 | "잡기 전, 비슷" | 3 cell 고른 분포·t/T 0.29·grasped 0.01 | 성공 접근 상태 앵커로 정합 |
| c12/c31 | 손잡이 접근/컨택 후, 단일 cell | drawer_left/right 전용 | scene-bound 물리 상태 |
| c13/c46 | drawer_right 초기 | drawer_right 전용·t/T 0.05/0.19 (drawer_left corridor는 c5/c39/c23) | corridor도 scene별 분리·초기 국면 |

종합: **write(cluster→cluster 이동) 대상 생존자는 사실상 없음** — c21/c37→c19는 "놓침"을
활성화로 되감는 것(물리거울 논거 그대로), reach-handle 계열은 scene-bound. 반면 **게이트
가치는 상승**: c21/c37은 scene을 가로지르는 "놓친 직후/빈손 배회" 검출기로, per-step
게이트→재샘플·재시도 개입과 정확히 맞물린다. wrong-grasp GT는 pq3에 0스텝이라 이 검출기의
GT 정합은 미확인(눈 판독+grasped flag 정합까지).

## 물리 잔차화 probe (`resid_cluster_probe.py`, `resid_probe_{raw,resid}_s0.csv`)

proprio 16d(gripper_qpos·base pos/rot·eef pos/quat rel) train-fit ridge 잔차화 후에도
phase 구조(purity 0.79→0.72~0.76, NMI 0.52→0.48~0.49)와 3분류(grasp succ/공용/fail 공존,
fail-전용 9~12 중 t<45 생존 7~8) 모두 생존. 선형 phys→L12pooled R²(held-out)=0.187.

**R² 수치 병기(연산자 설계 세션과 상호 확인, 모순 아님):** 이쪽 0.187은 [pooled 전
구간 × proprio 16d × L12 all-token mean] 조건, 연산자 설계 쪽 0.34~0.47은 [phase별·
고정창 × 속도 포함 φ18] 조건 — 통제(구간·창·속도 항) 차이로 설명된다. 한계: 선형
잔차화뿐이라 비선형 물리 부호화는 남고, proprio엔 물체·fixture 상태가 없어 잔차에
물체-상대 물리 정보가 잔존 가능(condg의 통제보다 약한 판).

## 4-arm 정면 비교 판정 (2026-08-31, 연산자 설계 세션 실행 — 6 task 15 phase-cell,
완전 held-out·고정B·5-seed 중앙값. 원자료 승준 ~/tmp_segb + Notion 3c363918… 하단)

| arm | pooled | scene-내 |
|---|---|---|
| 연속 LLR | **0.82** | **0.66** |
| proprio-잔차 LLR | 0.75 | 0.63 |
| cluster 소속(train fail log-odds) | 0.66 | 0.50 |
| hybrid(z-합) | 0.75 | 0.56 |
| 길이단독 | 0.80 | — |

쌍대결 LLR>cluster 11/15(양쪽), LLR>hybrid 10/15 — hybrid는 위상 기여가 아니라 희석.
**세션 간 사전 합의 기준(cluster 단독 패배+hybrid 무이득→강등)에 따른 판정: cluster
taxonomy는 채점기 후보에서 강등 — 시각화·해석·발화 게이트 규칙용.** "채점기=연속 LLR
단독, v3 재검증"은 **연산자 설계 세션의 자기 계획 선언**이다(그 세션의 재샘플+내부
read 선별 노선, docs/steering/39 서두) — **사용자가 채점기 방식 채택을 승인한 기록은
없음**; 방법 채택 여부는 별도 사용자 결정 사항. 발화 게이트 후보로
fail-only cluster 진입 규칙은 유지(margin 임계 vs cluster 진입 비교는 v3 라운드).
예외 각주: transport 계열 scene-내 3 cell(marsh 0.96·jug 0.83·bread 0.82)은 cluster가
우위 — phase-조건부 상보 가능성 미폐쇄. proprio-잔차 LLR 대체 유지(0.75/0.63)로
"채점 신호 ≠ 자세 선형 재서술" 방어 성립(단 소표본 2 cell 붕괴 — 방어 논증용, 운용 비추천).
