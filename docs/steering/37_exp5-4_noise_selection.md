# exp5-4 — "노이즈를 1-step만 돌려보고 성공할 draw 고르기" 판정

> **종결 범위 (2026-07-30 명확화)**: 종결된 것은 **"활성화를 보고 좋은 draw 를 고르는
> verifier"** 하나다. 13/13 이 seed 주효과로 설명되고(column 순열 p=.60), 활성 없이 action
> 통계(`chunk_tv`)만으로 동률이라 **활성 고유 정보가 없다**는 뜻이다.
>
> **살아있는 명제**: "어떤 seed 가 이 모델·task·scene 에서 좋은가/나쁜가"는 오히려 seed
> 주효과가 실재한다는 뜻이라 확인된 쪽이다. 실험 설계는 §6-1 `good-seed 전이 검증`에
> 이미 있다(활성 불요, CPU only) — **미착수**. caveat = winner's curse, n=10 검정력.

2026-07-28, exp5-4 세션 (worktree exp5-4-noise-select). 계획·사전등록 = `exp5-4_verifier_selection_plan.txt` §9 (Gate1 개정), Codex 원장 = `docs/collab/2026-07-28-exp5-4-gate1.md`.

## 한 줄 결론

**사전 등록 반증 게이트가 3 cell 전부에서 발동 — Phase A(새 seed 160판 인과 수집) 중단.**
exp5-1의 헤드라인(drawer 0.344→0.650, 혼재 13/13)은 **seed(=노이즈 draw) 주효과의 암기**로 설명되며, scene·seed를 모두 막은 prospective 이득(+0.106)은 자명 baseline(chunk_tv)과 동률·비유의다. 학습 축(t=0 활성 mean-diff)의 **고유 기여 근거 없음**.

## 1. 헤드라인이 무너진 지점 (기존 데이터 재분석, 승준 CPU)

기존 데이터는 20 scene이 **같은 8개 노이즈 draw(seed 0~7e6)를 공유**한다. 이 구조에서:

| 검정 (drawer L0, in-fold K=8) | 값 |
|---|---|
| 관측: base 0.344 → top-1 0.650 (Δ̂ +0.306, 혼재 13/13) | 재확인 |
| scene별 독립 라벨셔플 재fit null (2000회) | p = 0.001 |
| **8 seed column 공통 순열 (8!=40320 전수)** | **p = 0.602** |
| `seed_only` 음성대조 (활성 무시, seed 번호만으로 선택) | **Δ̂ +0.306 — 학습축과 완전 동률** |
| 활성 노름(a0_full_norm 큰쪽)·chunk_speed 등 자명 선택자 | 역시 +0.306 동률 |

즉 두 null이 갈리는 방향이 정확히 "seed 주효과": 선택자가 학습한 것은 "이 scene의 이 활성이 나쁘다"가 아니라 "**이 노이즈 draw는 (같은 머신에서) 전역적으로 나쁘다**"였다. 같은 inference_seed = 수치적으로 같은 노이즈 draw(머신-로컬)라는 §0 경고가 그대로 실현된 것.

## 2. 사전 등록 반증 게이트 (prospective, double scene/seed-out K=4)

옛 8 seed를 4+4로 분할, 평가 scene의 옛 판 전부 + test seed를 fit에서 제외:

| cell | 학습축 Δ̂ | fold별 | 최강 배포가능 baseline | 판정 |
|---|---|---|---|---|
| drawer_right | +0.106 | +0.200(p .005/.159†), +0.013(ns) | chunk_tv 큰쪽 **+0.106 (동률)** | 중단 |
| ppcc_beer | +0.044 | ns | act_norm_L0 +0.119 (**우세**) | 중단 |
| mixer | +0.037 | −0.100, +0.175 | chunk_speed +0.138 (**우세**) | 중단 |

† p_라벨셔플 / p_seedperm. 게이트 규칙(§9, 사전 등록): "Δ̂ ≤ 0 또는 최강 simple baseline 이하 → Phase A 중단" — 3 cell 전부 해당.

보조: 검정력 시뮬(관측 m_i 기반 exact)에서도 S=20(160판) 설계는 q=0.7 선택자 기준 power 0.13~0.39 — 설령 돌렸어도 결론이 안 나왔을 크기(S=40=320판 필요).

## 3. 그래도 확립된 것 (인프라·사실)

- **probe-only 모드** (`http_feature_collect.py --probe-seeds/--probe-out` + `probe_collect.sh` + `make_selection_manifest.py`): srv50 smoke **전 항목 PASS** —
  - probe ↔ 본 rollout record0 활성·action chunk **bit-identical** (cross-serve 포함, collector end-to-end)
  - capture-ON vs `--no-features`(skip_features) action bit 동일 (실경로 검증)
  - 같은 (scene, seed) rollout 2회 성패 재현
  - 비용 실측: **후보당 forward ~3.5초** (rollout 335초의 ~1%) — "1-step 선별"의 배포 비용 주장 실측치
- **선행 반증 게이트 + 봉인 파이프라인** (Gate1/Gate2 반영): 선택 manifest sha 봉인 없이는 rollout 불가(exit 4), 수집 실패 전파(exit 5), resume seeds 대조. seed manifest(scene별 고유 160 seed, sha c0e706a1) 동결 커밋.
- 흥미 부산물: **첫 action chunk의 total variation(chunk_tv)** 만으로도 prospective +0.106 — 활성 없이 action만 보는 초경량 선택자가 학습축과 동률. (비유의라 주장은 불가, 후속 후보)
- 문헌 (`docs/references/reading_notes/exp5_4_selection_lit.md`): 니치 3요소가 개별로는 전부 선점 — 특히 arXiv:2605.28527(동결 VLA succ/fail probe를 test-time selector로, LIBERO +17.6pp)·2603.15757(초기 노이즈가 SR 좌우, 단 rollout 필요). 조합 기여만 주장 가능했을 것.

## 4. 데이터 정오

- `sm_npz/exp41_mixer` == `sm_npz_mixer/exp53_mixer_sm` (활성까지 bit 동일) — 같은 데이터의 중복 사본. mixer는 1 cell로 셀 것.
- 분석 산출물: `outputs/analysis/exp5_4/` (gate/placebo/power/baselines JSON + `direction_L0_loso.npz` sha 2744…, `direction_L12_loso.npz` sha 214a…). smoke 산출물: 승준 `~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/exp5_4_smoke/` (MACHINE.txt 포함). srv50은 정리·원상 복원 완료.

## 5. 후속 ablation (판정 후 사용자 Q&A로 추가 실행 — 전부 기존 데이터·CPU)

**5.1 선택자가 실제로 한 일** (`selection_placebo.json` chosen_seed_idx): 학습축은 **20/20 scene에서
전부 seed 0을 선택**. 13/13 적중 = seed 0이 성공하는 scene 목록과 일치 (나머지 7 = 전패 scene).

**5.2 seed-column 제거** (`selection_drop_seed.py`): column SR은 완만히 분포
(.650/.550/.450/.350/.250/.200/.150/.150 — seed 0만 성공하는 것 아님). 그러나
- col0 제거(7 draw): 선택이 **col7(잔여 최고)로 20/20 재붕괴**, Δ̂ +0.250, column 순열 p=0.605
- col0+7 제거(6 draw): Δ̂ −0.008, p=0.99 — 신호 소멸
→ 어느 단계에서도 scene별 판별 없음. 축의 정보 = "draw 신원 확인 + 전역 승률 조회" 한 줄.
메커니즘: t=0은 scene 내 관측이 동일해 활성 차이가 100% noise 지문이고, scene 내 순위 비교가
관측 성분을 소거 → LOSO mean-diff는 공유 8개 지문에 전역 실패율을 매긴 조회 테이블로 수렴.

**5.3 record 1·2 재평가** (`selection_by_record.py`, r=inference 회차=5 env-step 간격):
- in-fold: r=0/1/2 전부 col0 20/20·column 순열 p 0.55~0.69 — 실행 후에도 신원 인식 그대로.
- prospective pooled: r=0 +0.106 → r=1 +0.181 → r=2(L0) +0.231로 단조 증가 힌트. 단 fold 이질
  (fold1 +0.10~0.15 vs fold2 +0.26~0.31), r=2에서 L12 +0.056으로 비재현, 유의 셀은 12셀 중
  1개(r2·L0·fold2 p_colperm .021) → 중단 판정 불변. 또 r≥1 선택은 후보당 부분 실행이 필요해
  "실행 전 게이팅"의 비용 이점(3.5초/후보)이 사라짐.

**5.4 "t=0 신호 0.712"와의 정합**: seed-out read AUROC 0.712는 부정되지 않음. AUROC 0.71을
8-후보 top-1로 환금하면 기대 이득이 +0.1 안팎인데 prospective가 정확히 그 크기(+0.106,
top-1 0.45)로 나옴 — 즉 신호는 실재하되 ①약하고 ②그 정보가 활성 고유가 아니라 action 통계
(chunk_tv)에도 동일하게 존재. exp5-3 "read 전용"에서 exp5-4 "read를 selection으로 환금해도
자명 방법 이상을 못 범"까지 좁혀짐.

## 6. 후속 옵션 (미착수)

1. **good-seed 전이 검증** (공짜, 승준 CPU): scene 10개로 best seed를 고르고 나머지 10 scene에서
   그 seed SR 측정 — "활성 verifier"는 죽었어도 "좋은 draw의 cross-scene 전이"(Golden Ticket류,
   머신-로컬) 명제는 별도로 살아있음. winner's curse·n=10 검정력 한계 명시 필요.
2. **chunk_tv 초경량 선택자 확인 실험**: 활성 불요. 단 검정력상 S=40(320판) 필요.
3. **시간축 추적**: r↑ 단조 증가 힌트(5.3)를 fold·layer 재현 조건으로 재검 — 상호작용이 자란
   지점에서의 선택은 이론적으론 유망하나 부분 실행 비용 문제를 함께 풀어야 함.
4. **2605.28527 본문 정독**: 그들의 probe-selector 성과도 scene 통제 부재라 우리와 같은
   신원-암기 confound 의심 — 재검 관점 자체가 기여가 될 수 있음.
5. exp5 본류(SAE scene 분리) 복귀 — 이 라운드로 "selection 우회로"는 닫힘.
6. Phase A 재개 시 인프라는 완비 (probe→봉인→rollout, srv50 세팅 문서화). 단 사전 등록상
   탐색 재개정 후에만.

## 7. 세션 요약 (2026-07-28, exp5-4 세션)

계획(Gate1 Codex 반론 반영·§9 사전등록) → 병렬 실행(승준 CPU 반증게이트+위약+baseline /
probe 인프라 구현+Gate2 / srv50 smoke / 문헌조사) → **게이트 발동으로 Phase A rollout 0판에
종결** → 후속 ablation 3종(§5)으로 귀인 확정. GPU는 smoke에만 사용(srv50 GPU3, 종료 후
원상복원). 커밋 이력: probe 모드(a0bdbd1) → seed manifest 동결(7dba20d) → Gate2 수정(eead170)
→ 분석 스크립트·방향 NPZ(승준 에이전트) → 종결 문서(cec6401) → ablation(3a9647a·65a7282·5d7b471).
