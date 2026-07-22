# exp4 참고: WA-LQR 논문의 perturbation·steering·코드 맵

- 논문: arXiv:2607.14943 *Steering Robustness into World Action Models via MI and Optimal Control*
  (Georgia Tech Trustworthy Robotics Lab, RSS 2026 워크샵). 심층 노트:
  [`docs/references/reading_notes/steering_robustness_wam_lqr.md`](../references/reading_notes/steering_robustness_wam_lqr.md)
- 코드: https://github.com/trustworthyrobotics/steering_robustness_WAMs
  (아래 경로는 이 repo root 기준. 로컬 클론은 임시 디렉토리라 필요 시 재클론)
- 요지: 교란(환경/관측 수준)으로 SR을 떨어뜨려 놓고, clean−perturbed 대조 방향으로 DiT residual을
  되돌려 **교란 하 SR을 회복** (+11~40pp). nominal SR 개선 아님. 위약·유의성 검정 없음.

## 1. Perturbation 3종 (전부 입력/환경 수준, activation 교란 아님)

| 교란 | 주입 방식 | 파라미터 | 대조쌍 품질 | 수집 코드 (cosmos 기준) |
|---|---|---|---|---|
| Gaussian 노이즈 | `prepare_observation` 후 primary+wrist uint8 이미지에 i.i.d. N(0,σ) + clip[0,255]. clean rollout 관측에 **사후 주입** | σ=90 (extreme). DiT4DiT는 fit σ=75 / eval σ=22 (fit을 더 세게, 근거 없음) | **완벽 matched** (같은 state, 노이즈 유무만) | `cosmos_steering/notebooks/lqr/inputs/collect_policy_inputs_noise_extreme.py` (주입 246-265) |
| 카메라 pose/FOV | MuJoCo `sim.model.cam_pos/cam_quat/cam_fovy` 직접 변조 + `sim.forward()`. clean/perturbed env 2개에 **같은 sim_state replay** | δpos σ=0.10m, δrot σ=8°, δfov σ=5°, workspace 이탈 시 rejection 리샘플 | matched (state replay, proprio 일치 assert) | `.../collect_policy_inputs_camera_view_perturbation.py` (변조 169-178, 샘플러 262-293) |
| gripper 초기위치 | reset 후 OSC 액션을 여러 스텝 보내 EE를 δxyz만큼 **실제 이동** 후 policy 시작 | δxyz 등방 σ=10cm (preset `xyz_random_xlarge_3`) | ⚠️ **outcome 혼입** — 교란 하 성공→pos/실패→neg 버킷 + proprio NN 재짝짓기 | `.../collect_policy_inputs_gripper_xyz_perturbation.py` (shift 136-237, 버킷 507-536) + `svd/pair_inputs_by_similarity.py` |

- 활성 "샘플" = rollout 중 action-chunk 추론마다 저장한 관측을 **denoising 1-step만 open-loop
  재통과**시킨 DiT block output (rollout 전체 활성 아님):
  `cosmos_steering/notebooks/actadd/collect_acts_passive.py:131-143`.
- eval은 **같은 교란을 켠 채** steered vs unsteered (같은 seed·init), 태스크당 20~30판.

## 2. Steering 방법 2종

**ActAdd (open-loop)** — 우리 mean-diff와 동계열
- 방향: `pos.mean(0) − neg.mean(0)`, 층별 1벡터, 정규화 없음
  (`cosmos_steering/notebooks/actadd/make_contrastive_vec.py:61`).
- 적용: 전 블록 output에 `output + alpha·v`, **전 denoising step**
  (`notebooks/actadd/run_actadd_cosmos_policy.py:316-328`). γ(alpha) 민감도 큼(저자 인정).

**WA-LQR (closed-loop, 이 논문 신규)**
- 부분공간: (layer-partition × denoising-t)별 randomized SVD k=64 + 사영 대조평균 c_means.
  방향 v = c_means/‖c_means‖, setpoint 크기 μ = ‖c_means‖
  (`notebooks/lqr/svd/run_partition_svd_pairs_no_action.py`).
- 동역학: 층→층 Jacobian을 autograd(jvp/vjp)로 계산해 사영 = A. "층 진행 = LQR의 시간축"
  (`notebooks/lqr/jacobians/compute_jacobians_full.py:422-435`).
- 피드백: 매 층에서 z = Vᵀx 사영 → 오차 α = λμ − v·z → u = V_out·K·(α·v)를 block output에 가산.
  **α self-gating**: 사영이 setpoint에 이르면 u→0 자동
  (`notebooks/lqr/run_lqr_decay_cosmos_policy_cam_perturb.py:366-399`).
- **chunk decay**: R(c) = R_init·exp(c/τ), τ=3 — 첫 chunk에 가장 세고 지수 감쇠
  (rollout 시간축 조건부의 원시형, phase-matched 아님).
- 개입 지점: **전 28블록 residual stream** (마지막 layer 아님, layer 선택 ablation 없음).
  Cosmos는 **action 토큰 슬롯 제외**, world(미래관측/value) 슬롯만 steer — 이유 미기술.
- 발화: LQR은 선택된 denoising step에서만 (ActAdd는 전 step).

## 3. 코드에서 확인할 것 (WAM → 우리 VLA 대응)

| 저들 (WAM) | 우리 (GR00T VLA) 대응 | 볼 파일 |
|---|---|---|
| Cosmos 통합 DiT 28블록 (video+action 공동 denoise), world 슬롯 steer | GR00T action_head DiT block — world 슬롯이 없으니 **action 경로를 직접 steer해야 함** (저들의 "action 제외" 설계는 이식 불가) | `run_partition_svd_pairs_no_action.py:1-34` (슬롯 구조) |
| DiT4DiT: 별도 action DiT를 steer | 우리와 가장 유사한 구조 — hook 지점·B 행렬 포함 LQR 참고 | `DiT4DiT_steering/lqr/run_lqr_dit4dit_noised.py` (hook :277) |
| denoising 1-step 재통과로 활성 추출 | 우리 http_feature_collect는 실제 추론 활성 — 수집 비용/재현성 트레이드오프 비교용 | `collect_acts_passive.py:120-143` |
| matched pair 수집 (state replay) | 우리 EVAL_SEED 결정적 reset과 결합하면 동일 구성 가능 | `collect_policy_inputs_camera_view_perturbation.py:370-431` |
| chunk decay·α gating | 우리 gated steering/phase-gating serve에 이식 후보 부품 | `notebooks/lqr/lqr_injector.py` 대응 로직: LingBot `scripts/lqr/lqr_injector.py:91-94,166-227` |
| SVM hinge loss = steer 가능성 사전 예측자 | eval 0판 사전 게이트로 유용 — exp4-2에서 fit 전에 분리도 먼저 측정 | 논문 Fig.7 (r≈−0.7), 코드는 분석 노트북 쪽 |

## 4. exp4-2에 주는 것 / 조심할 것

- **가져올 것**: ① 교란 대조는 신호가 실재함을 3자 검증(우리 자연 succ/fail null과 대비 → 병목=대조축 강도)
  ② 교란 이식 난이도 순서: 픽셀 노이즈(env 무접촉) → 카메라(MuJoCo 변조+visibility 재구현) → gripper shift(eval 루프 수정)
  ③ 대조쌍 품질 순서: 노이즈(완벽) > 카메라(replay) > gripper(outcome 혼입 — 따라하지 말 것)
  ④ fit 전 SVM 분리도 사전 게이트.
- **우리 차별화 유지**: 자연 outcome 대조·rollout-phase 조건부·위약 대조·nominal SR 타깃 — 전부 저들에 없음.
- **주의**: 저들의 이득은 교란 하 측정. 우리 nominal eval로의 전이는 자동이 아님 → 24b의 다리 실험
  (자연 실패 활성을 교란 축에 사영) 먼저.
