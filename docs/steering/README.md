# Latent Steering — 문서 지도

**방향 = [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md), 결과 = [`RESULTS.md`](RESULTS.md)** 가 단일 출처.
종결 라운드·구 분석 문서는 archive(git 이력 + [`../review/LEDGER.tsv`](../review/LEDGER.tsv)) — 여기 없는 번호 문서는 거기서 찾는다.

## 상시 (라운드 무관)

| 문서 | 내용 |
|---|---|
| [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md) | 방향·시나리오(§0.5)·RQ·가설 C1~C4·미해결·검증 설계 |
| [`RESULTS.md`](RESULTS.md) + [`results.tsv`](results.tsv) | SR 개입 실험 결과 원장 (exp2 ~ v4r) |
| [`PITFALLS.md`](PITFALLS.md) | 배선·실행·운영 함정 |
| [`SCENE_FEASIBILITY.md`](SCENE_FEASIBILITY.md) | fixture task 기하 불가 seed 필터 |
| [`../05_gpu_server_rules.md`](../05_gpu_server_rules.md) · [`../04_data_storage_convention.md`](../04_data_storage_convention.md) | GPU 서버·세션 예약 규약 / 데이터 저장 규약 |

## 현행 설계

| 문서 | 내용 |
|---|---|
| [`45_hotfix_scenario_spec.md`](45_hotfix_scenario_spec.md) | hotfix 시나리오 스펙 (시나리오 구체화 세션). 요지 = DIRECTION §0.5 |
| [`47_perstep_gating_pipeline.md`](47_perstep_gating_pipeline.md) | per-step 게이팅 파이프 설계 (latch 폐기, pre-hook detector) |
| [`../collab_within_claude/handoff_20260902_v4r_round.md`](../collab_within_claude/handoff_20260902_v4r_round.md) | 중추 세션 현행 핸드오프 (라운드 서사·좌표·잔여) |

## 표현 분석 (DIRECTION이 근거로 인용하는 것만)

| 문서 | 내용 |
|---|---|
| [`01_seen18_latent_analysis.md`](01_seen18_latent_analysis.md) | succ/fail 분리 — 길이 confound 통제가 전제 (C1) |
| [`08_pathway_separation_analysis.md`](08_pathway_separation_analysis.md) | VL/DiT pathway 비교 ⚠ 분리 시점 주장 반증 (§4) |
| [`22_wrong_grasp_vl_separation.md`](22_wrong_grasp_vl_separation.md) | wrong-grasp 시 VL 분리 (§4 VL case) |

> 2026-09-02 archive: 05(수집 확장)·07(연산자 서베이)·24d(분산 연산자)·31·32(SAE)·35(exp5-2)·39(rsN 배경)·40·41(phase 분리). 요지는 RESULTS §1·§3, DIRECTION §5로 흡수.
