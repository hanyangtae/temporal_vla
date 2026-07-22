# Handoff: GR00T N1.5 + RoboCasa — event-phase 수집 → Rung 2 분석

다음 세션이 이 작업을 이어받기 위한 자족 문서. 플랜 단일출처: `~/.claude/plans/vla-reactive-acorn.md`
(마지막 섹션 "★ 전환: GR00T N1.5 + RoboCasa"). 연구 단일출처: `docs/steering/14_pathway_phase_online_steering.md`.

---

## 0. 한 줄 / 지금 상태

VLA latent에서 **action-phase별 succ/fail 분리 + phase-selective steering** 연구. pi05+libero는
SR~98%(실패 없음)라 **GR00T N1.5 + RoboCasa atomic PnP**(실패=timeout 자연발생)로 전환.
**event-anchored phase 라벨러 + N1.5 수집 파이프라인 완성·검증, 4 cell × 15 rollout 수집이
2026-06-30 ~08:35 발사됨**(3-GPU 병렬, detached). 완료 후 **Rung 2 분석**(phase별 분리도)이 다음 단계.

---

## 1. 빌드 완료물 (검증됨)

- **event 세그멘테이션 순수 코어**: `scripts/safe/groot_n16/libero/event_phase_labeler.py`
  (`PhaseSegmenter`/`EventPhaseLabeler`/`TaskEvent` + gap 라벨 reach-to-object/transport/insert-settle/reach-to-door/terminal). 테스트 `tests/test_event_phase_labeler.py`(8 pass).
- **RoboCasa 라벨러**: `scripts/safe/groot_n16/robocasa/collect/robocasa_event_labeler.py`
  — `make_robocasa_event_labeler(env, task_name)`. OU.* 술어로 grasp→place→release 검출.
  cabinet=`OU.obj_inside_of(env,"obj",env.cab)`, stove=`OU.check_obj_in_receptacle(env,"obj","container",th=0.07)`,
  grasp=`OU.check_obj_grasped`+debounce, release=placed+`OU.gripper_obj_far`. 테스트 `tests/test_robocasa_event_labeler.py`(5 pass).
- **N1.5 수집 배선**: `scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py`
  (N1.5 = LeRobot-HTTP 경로. N1.6의 collect_env.py/ZMQ 아님). 루프에서 `labeler.step()`을 get_action당 1회
  (env.step 전) → feature 1개당 phase 1개. 종료 후 터미널 step 1회 + `assert len(feature_phases)==len(records)`.
  pkl에 `phase_scheme/feature_phases/phase_timeline/event_steps/event_order` 동봉.
- **런처**: `collect_phase_event_4cell.sh`(worker별 round-robin), `watch_phase_event_cleanup.sh`(완료 감시→serve kill→succ_fail_split.tsv).

**검증(smoke)**: cabinet 실패=transport 정체(grasp만), stove/cabinet 성공=reach→transport→insert-settle→terminal
(grasp/place/release 다 발화), feature↔hidden 1:1 정렬. **seed-overlap 버그 수정으로 succ/fail 혼합 확인**(아래).

---

## 2. 수집 설정 (실행 중)

- **모델**: GR00T N1.5 (`configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml`). 3 serve = GPU 5/6/7, 포트 8400/8401/8402, `--collect --capture-vl --groot-dit-capture-layers 0,2,4,8,10,12,15`.
- **4 cell** (1 고정 scenario_seed/cell, 15 rollout, inference_seed=0,1000,…,14000):
  - cabinet: `ppcc_bread`/seed 100084, `ppcc_potato`/seed 200010
  - stove: `ppcs_apple`/seed 100050, `ppcs_onion`/seed 100000
  - (seed = `outputs/eval/robocasa/groot_n15/coast4_reused_remote/manifests/selected_instruction_seeds.tsv`의 cell별 첫 seed)
- **predict-16/execute-5**(n_action_steps=5), max_episode_steps=720, steps_per_render=2.
- **★ 핵심 버그 수정**: inference_seed가 호출당 `inf+len(records)`로 적용 → 연속 base(0,1,2)는 거의 동일 rollout. **base를 idx*1000 간격**으로 띄워 15개를 진짜 다르게(succ/fail 혼합) 만듦. **새 cell 추가 시 반드시 이 간격 유지.**

**pkl 스키마** (per rollout): `hidden_states`(DiT block-residual `[L=7, K=4, D=1536]` fp16 per record), `vl_hidden_states`(`[2048]`), `feature_phases`(list[str], record와 1:1), `phase_timeline`(per env-step at get_action resolution), `event_steps`(grasp/place/release first-fire step), `event_order`, `episode_success`, `cell_id`, `scenario_seed`, `inference_seed`, `ep_meta`. feature_kind=`groot_n15_dit_block_residual_action_tokens_denoise`.

---

## 3. 모니터링 / 완료 확인 (run dir = `outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/`)

```bash
D=outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell
find $D/raw_rollouts -name 'task*--ep*--succ*.pkl' | wc -l        # 진행 (target 60)
for d in $D/raw_rollouts/*/*/; do echo "$(basename $d): succ=$(ls $d*succ1.pkl 2>/dev/null|wc -l) fail=$(ls $d*succ0.pkl 2>/dev/null|wc -l)"; done
tail -f $D/logs/worker{0,1,2}.log                                 # worker 진척
ls $D/logs/ALL_DONE_cleaned $D/succ_fail_split.tsv 2>/dev/null    # 완료 + split
nvidia-smi                                                        # 완료 후 serve가 GPU 반환했는지
```
- 멈춤/조기중단 시 serve kill: `docker exec lerobot pkill -f 'serve/lerobot.py.*--collect'`.
- 임시 `smoke/`·`verify/` 하위는 삭제 가능.

---

## 4. ⚠️ 완료 후 먼저 확인할 것 (분석 전)

1. **cell별 succ/fail split** (`succ_fail_split.tsv`): **각 cell이 succ·fail 양쪽을 충분히(예: ≥5/class) 가지나.**
   한 cell이 all-fail 또는 all-succ면 그 cell은 분리 불가 → **scenario_seed를 manifest의 다른 seed로 교체해 재수집**
   (그 cell만). (08:35 시점 bread는 2succ/1fail로 혼합 — 양호.)
2. phase별 표본수: 각 phase(reach/transport/insert-settle)에 latent가 충분한가. insert-settle은 짧아(place~release 근접) 표본 적을 수 있음 → 필요시 transport에 병합.
3. **길이 confound 주의**: 실패는 720-step timeout, 성공은 조기종료 → 길이가 라벨 결정 위험([[seen18-rollout-length-confound]]).
   분석은 **phase별·동일 스텝예산 pool + length-only baseline + permutation null** 필수. **단, 같은 scene(고정 seed)·같은 instruction이라 layout/object/instruction confound는 이미 통제됨**(이게 fixed-seed 수집의 장점).

---

## 5. 다음 단계 — Rung 2 분석 (이 수집의 목적)

목표: **(Q1)** phase(event)별 succ/fail가 latent에서 분리되나·어느 layer/pathway에서, **(Q2)** operator가 phase-selective한가(영구 steer 가능성).
- 재사용: `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py` (절대-t 윈도 → **event-phase pooling**으로 치환; `mannwhitneyu`→numpy rank-AUROC, 원격 scipy 없음). cross-task/selectivity 스크립트도 동 디렉토리.
- grid: **{DiT layer 0,2,4,8,10,12,15} × {VL} × {reach/transport/insert-settle phase}** × succ/fail AUROC + length-only + null.
- **2b cross-phase 분리도** (phase끼리 구분되나=selectivity 전제) + **2c operator-selectivity**(phase-p operator를 전 phase에 적용해 응답 프로파일).
- 데이터 large면 원격 CPU 분석([[remote-compute-workflow]]); pkl은 torch라 `~/anaconda3/bin/python`.
- 사다리 GO 게이트: ≥1 phase 분리(cv-AUROC≥0.65 & length-only 대비 +0.07 & null 초과) + phase 의존성. 신호 보이면 → Rung 3(steering, permanent vs oracle-gated) 다음 세션.

---

## 6. 핵심 경로 한눈에

- 라벨러: `scripts/safe/groot_n16/{libero/event_phase_labeler.py, robocasa/collect/robocasa_event_labeler.py}`
- 수집: `scripts/safe/groot_n15/robocasa/collect/{http_feature_collect.py, collect_phase_event_4cell.sh, watch_phase_event_cleanup.sh}`
- 데이터: `outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/raw_rollouts/<cell>/...pkl` + `succ_fail_split.tsv`
- seed 매핑: `outputs/eval/robocasa/groot_n15/coast4_reused_remote/manifests/selected_instruction_seeds.tsv`
- 분석: `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py` (+ 동 dir)
- 테스트: `tests/test_{event_phase,robocasa_event,bddl_phase}_labeler.py` (20 pass)
- 플랜: `~/.claude/plans/vla-reactive-acorn.md`. 데이터 대량 시 archive: [[remote-data-archive]](kimseungjun HDD).
