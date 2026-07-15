# pq3 실행 런북 (COAST 축 정렬 검증 — 계획서 dynamic-riding-aurora v9)

설계 단일 출처: `~/.claude/plans/dynamic-riding-aurora.md` (v9) + `docs/steering/19_pq3_execution_handoff.md`.
Gate 1/2 원장: `docs/collab/2026-07-13-pq3-gate1.md`, `docs/collab/2026-07-15-pq3-gate2.md`.

**컨테이너 규칙**: 검증·실행 모두 기존 컨테이너에 `docker exec` — 컨테이너 생성/재시작 금지
(수집기·VNC 세션 사고, 2026-07-16 공지).

## 0. manifest (수집 전 — seed·noise 시리즈 단일 출처)

```bash
PQ3=scripts/safe/groot_n15/robocasa/steer/pq3
SEEDS=outputs/eval/robocasa/groot_n15/coast4_reused_remote/manifests/selected_instruction_seeds.tsv
MANI=outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests
python3 $PQ3/make_pq3_manifests.py plan --seeds-tsv $SEEDS --cell-id pq3_drawer_left  --tsv-cell-index 8 --out-dir $MANI
python3 $PQ3/make_pq3_manifests.py plan --seeds-tsv $SEEDS --cell-id pq3_drawer_right --tsv-cell-index 7 --out-dir $MANI
python3 $PQ3/make_pq3_manifests.py plan --seeds-tsv $SEEDS --cell-id pq3_ppcc_bread   --tsv-cell-index 5 --out-dir $MANI
# ppcc 신규 2종: bash $PQ3/pq3_c0_scan.sh 후 산출 tsv 로 동일하게 plan
```

## C. fit15 수집 (로컬 GPU 0/1/2 · full-token · 즉시 승준 직송)

```bash
CELL=pq3_drawer_left
row=$(source $PQ3/pq3_lib.sh && pq3_row_of $CELL) && IFS='|' read -r c task envn cidx _ instr <<<"$row"
CELL_ID=$CELL TASK=$task ENVN=$envn CELL_INDEX=$cidx INSTR="$instr" \
  MANIFEST=$MANI/$CELL/collect_plan.tsv SHIP=1 GPUS_L="0 0" PORTS_L="8410 8411" \
  bash $PQ3/pq3_collect_cell.sh          # 기본 S0..S14; backfill 은 EPLIST="15 16 ..." 로
python3 $PQ3/p0_gate_pq3.py --collected-dir outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts/$task/$CELL \
  --manifest-dir $MANI/$CELL             # rc=2 → BACKFILL_EPLIST 출력, rc=3 → cell 탈락
python3 $PQ3/make_pq3_manifests.py freeze --seeds-tsv $SEEDS --cell-id $CELL --tsv-cell-index 8 \
  --collected-dir <위 collected-dir> --out-dir $MANI \
  --pkl-prefix '~/workspace/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts/'$task/$CELL
python3 $PQ3/make_pq3_manifests.py pool --out-dir $MANI --cells pq3_drawer_left,pq3_drawer_right \
  --task-manifest $MANI/task_OpenDrawer_fit.tsv
```

수집 검증 표준(유실 사건 2026-07-16): SHIPPED.tsv(size+sha) 대조, `find -type f` 실물 카운트,
du 용량, 평균 크기 상식(fit pkl 수십 MB — ~1MB 면 중단). 심링크 서브셋 금지(манifest 직접 참조).

## D. fit (승준 anaconda python, 스레드 cap) — Stage1 → 사용자 게이트 → 본 fit

```bash
FIT=scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py
# Stage1 (regime별 layer 선택 — 결과는 사용자 보고 게이트)
python $FIT --cell x/pq3_OpenDrawer --manifest $MANI/task_OpenDrawer_fit.tsv \
  --groups global,<phase...> --denoise per_step --stage1-quota-sweep --stage1-alpha 10 \
  --require-capture-token-mode all_token_full --alphas table14 --quota-floor 0.01 \
  --eval-reserved $MANI/pq3_drawer_left/eval_reserved.json --out-dir <fit_out>
# 본 fit (선택 layer 만): perm 용 global + gated 용 phase-bin, per-step NPZ
python $FIT ... --denoise per_step --layers <L> --alphas table14 --quota-floor 0.01 \
  --require-capture-token-mode all_token_full --eval-reserved <...> --out-dir <fit_out>
# gated 성립 게이트 (통과해야 build_pq3_queue 가 gated arm 생성)
python $PQ3/pq3_gated_gate.py --tasks OpenDrawer=$MANI/task_OpenDrawer_fit.tsv,... \
  --phases "OpenDrawer=reach-to-handle+grasp-handle+pull,..." --layer <L_gated> --out <gate.json>
# β sweep (fit seed 재사용 — 참조=수집 base 라벨)
CELL_ID=... PERM_NPZ=... PERM_LAYERS=<L> GATED_NPZ=... GATED_LAYERS=<L> GPUS_L=... PORTS_L=... \
  bash $PQ3/beta_sweep.sh
python3 $PQ3/beta_decide.py --sweep-root .../steer_eval_pq3/sweep --manifest-dir $MANI \
  --task OpenDrawer --cells pq3_drawer_left,pq3_drawer_right --out <beta.json>
```

주의: fit CLI 기본값은 pq2 legacy 보존 — pq3 는 반드시 `--alphas table14 --quota-floor 0.01
--require-capture-token-mode all_token_full --denoise per_step --eval-reserved <...>` 명시.

## E. eval 600판 (분산·캡처 OFF)

```bash
python3 $PQ3/build_pq3_queue.py --arm-config <gate_d_arm_config.json> \
  --qroot outputs/eval/robocasa/groot_n15/steer_eval_pq3/work_queue \
  --gated-gate-report <gate.json>
# lane (로컬): GPU=0 PORTS="8410 8411" bash $PQ3/pq3_lane_local.sh
# lane (w2/.50): GPU=2 PORTS="8410 8411" CLS=w2 bash $PQ3/pq3_lane_w2.sh
# lane (w48):    GPU=2 PORTS="8410 8411" CLS=w48 MACHINE_TAG=worker1-48 HF_HOME_OVERRIDE=... bash $PQ3/pq3_lane_w2.sh
```

arm-config 의 npz_shas 는 필수 (Gate D 동결 sha12). host 는 cell-블록 배정.

## F. 판정

```bash
python3 $PQ3/aggregate_pq3.py --eval-root .../steer_eval_pq3/e1 --manifest-dir $MANI \
  --arm-config <arm_config.json> --out <aggregate_out> [--gated-na-tasks drawer]
```

판정 규칙 단일 출처 = `pq3_decision.py` (Gate D 시점 동결 — eval 후 수정 금지, sha 가
summary 에 기록됨). 보고 전 confound-audit 스킬 필수.
