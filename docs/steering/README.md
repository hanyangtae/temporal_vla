# Latent Steering — 문서 지도

메인 연구 라인의 문서 위치만 안내한다. **연구 방향·가설은 [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md),
실측 결과는 [`RESULTS.md`](RESULTS.md)** 가 단일 출처다. 프로젝트 전체 소개는 루트 [`README.md`](../../README.md).

## 상시 문서 (라운드 무관)

| 문서 | 내용 |
|---|---|
| [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md) | 방향·RQ1~4·가설 C1~C4·open problem·검증 설계 |
| [`RESULTS.md`](RESULTS.md) + [`results.tsv`](results.tsv) | exp2~exp5 SR 개입 실험 결과 원장 (arm 37행) |
| [`PITFALLS.md`](PITFALLS.md) | 배선·실행 함정 (α 배선, fit↔eval 분리, 위약, 코드 앵커) |
| [`../04_data_storage_convention.md`](../04_data_storage_convention.md) | activation·연산자 저장 규약 (sig·인덱스·전송·삭제) |
| [`SCENE_FEASIBILITY.md`](SCENE_FEASIBILITY.md) | fixture task의 기하 불가 seed 필터 |

## 표현 분석 (read)

| 문서 | 내용 |
|---|---|
| [`01_seen18_latent_analysis.md`](01_seen18_latent_analysis.md) | succ/fail 분리 — **길이 confound 통제가 모든 해석의 전제** |
| [`08_pathway_separation_analysis.md`](08_pathway_separation_analysis.md) | DiT 32-layer + VL/DiT 비교 ⚠ 분리 시점 주장은 반증됨 |
| [`22_wrong_grasp_vl_separation.md`](22_wrong_grasp_vl_separation.md) | wrong-grasp 시 VL activation 분리 |
| [`31_sae_g1_results.md`](31_sae_g1_results.md) · [`32_g2_scene_residual_results.md`](32_g2_scene_residual_results.md) | SAE scene 성분 분리 (G1 PASS → G2) |
| [`40_action_phase_readout_review.md`](40_action_phase_readout_review.md) | 동료 phase readout 라인 검증 — margin 기준·경계 비정렬은 해상도 차이·재현성 사다리 |

## 방법론

| 문서 | 내용 |
|---|---|
| [`07_steering_methods_survey.md`](07_steering_methods_survey.md) | steering 연산자 후보 + conceptor vs 평균차이 수식 대비(부록) |
| [`24d_exp4-3_variance_aware_direction_input.md`](24d_exp4-3_variance_aware_direction_input.md) | 평균+분산 연산자(whitened mean-diff) 설계 근거 |

## 라운드 문서 (살아있는 것만 — 종결 라운드는 RESULTS.md 흡수 후 archive, 목록 = `../review/LEDGER.tsv`)

| 문서 | 상태 |
|---|---|
| [`35_exp5-2_results.md`](35_exp5-2_results.md) | 섭동-유도 실패 회복 (RESULTS와 별도 유지 — 무대가 다름) |
| [`39_resample_verifier_round.md`](39_resample_verifier_round.md) | 재샘플+verifier 계보·rsN 연산자 배경 |
| [`41_grid_phase_separation.md`](41_grid_phase_separation.md) | grid phase 분리 — intrinsic k8 vs GT, OvenRack reach 암기 검증 |
| [`45_hotfix_scenario_spec.md`](45_hotfix_scenario_spec.md) | **hotfix 시나리오 스펙**(시나리오 구체화 세션). 요지는 RESEARCH_DIRECTION §0.5 |
| [`47_perstep_gating_pipeline.md`](47_perstep_gating_pipeline.md) | **per-step 게이팅 파이프 설계 정본** (latch 폐기, pre-hook detector 규칙) |

## 세션 핸드오프 → [`../handoff/`](../handoff/)

현행 = [`handoff_20260902_v4r_round.md`](../handoff/handoff_20260902_v4r_round.md) (중추 세션). Codex 원장은 [`../collab/`](../collab/) — 별개.

## 수집·운영

| 문서 | 내용 |
|---|---|
| [`05_safe_lerobot_collection.md`](05_safe_lerobot_collection.md) | SAFE 수집 lerobot 멀티벤치 확장 |
| [`../04_data_storage_convention.md`](../04_data_storage_convention.md) | 수집·저장 규약(지터 축 §3.1.1). 수집 라운드 기록(v2·v3 요청서)은 archive/handoff |
| `.claude/skills/robocasa-steer-eval/SKILL.md` | eval 러너 사용법 정본 |

> 번호 prefix는 생성 순서일 뿐 읽기 순서가 아니다. 전역 재명명은 레포 검토 완료 후 일괄 예정
> ([`../review/`](../review/)).
