# 33. exp5-4 — "노이즈를 1-step만 돌려보고 성공할 draw 고르기" 판정

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

## 5. 후속 옵션 (미착수)

1. **chunk_tv 선택자 확인 실험**: 활성 불요·초경량. 단 위 검정력 계산상 S=40(320판) 필요 — 효과 크기 대비 비용 판단은 사용자 몫.
2. exp5 본류(SAE scene 분리)로 복귀 — 이 라운드로 "selection 우회로"는 닫혔고, read 신호의 용처는 다시 열린 문제.
3. Phase A를 그래도 돌리려면 인프라는 완비 상태 (probe→봉인→rollout, srv50 세팅 문서화됨). 단 사전 등록상 이는 탐색 재개정 후에만.
