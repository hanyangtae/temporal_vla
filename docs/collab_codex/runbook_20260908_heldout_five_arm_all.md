# 대상 jitter 성공·실패 전판 — heldout detector 5-arm 평가

## 계약

사용자 2026-09-08 지시: 준비된 heldout detector로 대상 jitter를 재평가하고 성공판 파괴율까지 측정.

- 기준 branch: `eval_whole_pipe_1`, kanu 코드 worktree `.claude/worktrees/eval-whole-pipe`.
- manifest: `configs/experiments/v6_heldout_all_20260908/episodes.tsv`.
- 기존 공통 49셀, 셀마다 noise 0~9: **490판 = 실패 225 + 성공 265**.
- kanu 110판(61/49), worker1 120판(53/67), worker2 260판(111/149). 괄호는 실패/성공.
- 5 arm: plain β0.8, jfair β0.9, reseed, reseed→plain β0.8, reseed→jfair β0.9.
- 별도 paired `base` 490판으로 replay 정합성 확인. 총 **2,940 episode-arm**.
- β는 기존 225 실패판 평가에서 고정: jfair 0.9/1.0 동률 중 작은 0.9, plain은 기존 후보 0.8만 평가됨. 결합 arm도 같은 β 사용.
- detector `detector_v6_ho`, `train_pool=other`, α0.1. 대상 j 성공·실패 모두 detector 학습 제외.
- phase=GT, NPZ=`instr_setm_v6_gt{,_plain}`의 L12, alpha0 명시, token all. 기존 denoise hook 범위 유지.
- `reseed_setm`: detector가 발화한 record에서 seed1+900000으로 새 DiT pass를 실행하고, **그 pass에서 생긴 activation에** setM hook 적용. GT phase는 같은 현재 환경 상태를 사용한다. 발화하지 않으면 원래 action. phase 미등록이면 항상 순수 reseed로 남기고 fallback 기록.
- 기존 plain/jfair도 기존 실험과 같은 fallback=reseed. 따라서 순수 연산자 적용 수와 fallback 수는 구분해서 해석한다.
- detector 입력은 첫 pass, 상태는 실제 실행한 두 번째 pass의 hidden으로 restore→step. 추가 후보 탐색 없음.
- raw activation은 저장하지 않음. sidecar JSON, per_episode.tsv, 영상만 생성.

## 좌우 키

replay는 구 plan `77e745c37b0f`, 기존 collection_plan.json과 원래 index 좌표 유지.
`grid_instruction`은 구 키, `artifact_slug`/`artifact_instruction`은 새 키다.

| replay | detector·operator |
|---|---|
| OvenRack/out-left | OvenRack/out-right |
| DishwasherRack/out-right | DishwasherRack/out-left |

단순히 instruction 문자열만 바꾸면 반대편의 성공/실패판을 뽑는다. 이번 manifest는
`old_instruction + scene + jitter + machine`으로 원래 index 행을 재선택했고,
각 셀 실패 noise 목록을 이전 평가표와 대조했다. `artifacts.json`에 입력 447파일 SHA256을 고정했다.

## 실행

서버의 repo 경로는 `~/pkt_ws/temporal_vla`, 기존 사용자 변경은 보존한다.
코드는 git으로만 동기화하고 detector·NPZ는 기존 배포본의 SHA256을 대조한다.
kanu worktree의 비어 있는 submodule 자리에는 `lerobot/src`를 메인 트리의 같은 경로로
상대 symlink 연결해야 한다. 컨테이너 mount는 메인 트리 `/temporal_vla`다.

아래 GPU는 예시다. **직전 nvidia-smi 프로세스 소유자·로컬 lease 확인 필수**.
kanu ≤3장, GPU당 2 serve; srv48/50 각 1장, GPU당 6 serve.

```bash
# kanu: 메인 트리에서 실행. wrapper를 detach하므로 lease PID가 전체 run을 감싼다.
mkdir -p outputs/eval/robocasa/groot_n15/og_v6_ho5_all_20260908
setsid nohup bash scripts/utils/with_gpu_lease.sh kanu '5 6 7' codex-ho5 'heldout all five-arm eval' -- \
  python .claude/worktrees/eval-whole-pipe/scripts/steer/online_gated/run_v6_heldout_all.py \
  --machine kanu --gpus 5,6,7 --lease-held --port-base 9200 \
  --manifest .claude/worktrees/eval-whole-pipe/configs/experiments/v6_heldout_all_20260908/episodes.tsv \
  --out outputs/eval/robocasa/groot_n15/og_v6_ho5_all_20260908 \
  > outputs/eval/robocasa/groot_n15/og_v6_ho5_all_20260908/launch.log 2>&1 < /dev/null &

# srv48: 로컬 lease wrapper가 SSH를 유지한다. 원격 python은 setsid로 실행.
mkdir -p outputs/tmp/ho5_launch
setsid nohup bash scripts/utils/with_gpu_lease.sh srv48 '0' codex-ho5 'heldout all five-arm eval' -- \
  ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 AISem_48_junhyeong \
  'cd ~/pkt_ws/temporal_vla && exec setsid python3 scripts/steer/online_gated/run_v6_heldout_all.py --machine worker1 --gpus 0 --lease-held --port-base 9200 --manifest configs/experiments/v6_heldout_all_20260908/episodes.tsv --out outputs/eval/robocasa/groot_n15/og_v6_ho5_all_20260908' \
  > outputs/tmp/ho5_launch/srv48.log 2>&1 < /dev/null &
# srv50: srv48→srv50, AISem_48→AISem_50, worker1→worker2, GPU는 당시 빈 1장으로 변경.
```

`--dry-run`은 각 arm의 실제 runner CLI를 생성한다. 배관 smoke는 별도 output에
`--cell OpenDrawer_left:2:1 --noises 0 --maxep 30`으로 실행했다. timeout을 짧게 한
smoke 결과는 구제율/파괴율에 포함하지 않는다.

완료 상태를 가진 output은 재사용 거부. 부분 작업도 자동 덮어쓰지 않고 실패한다.
runner 오류나 base/collection 불일치 발견 시 새 작업 발사를 중단하고 이미 실행 중인
슬롯만 마친 뒤 `INCOMPLETE.json`으로 종료한다. 복구 시 부분 폴더와 원인을 먼저 점검한다.

## 완료·집계

- 각 root: `contract.json`, 고정 `manifest.tsv`, `logs/`, arm별 per_episode와 sidecar,
  `episodes.tsv`, `summary.tsv`, `DONE.json` 또는 `INCOMPLETE.json`.
- DONE = 모든 arm의 정확한 noise·seed·label 행 확인, sidecar 존재/중복 확인,
  reseed seed2 offset 확인, runner rc0. `[done]` 로그만으로 완료 판정하지 않는다.
- **구제율** = 같은 판의 새 base 실패 중 arm 성공 / 새 base 실패 수.
- **파괴율** = 같은 판의 새 base 성공 중 arm 실패 / 새 base 성공 수.
- 전체 SR 및 paired ΔSR도 기록. 수집 label과 새 base가 다르면 별도 기록하고 해석 보류.
- 매 cell-job 종료 후 집계 갱신. 원격 결과 회수 뒤 `summarize_v6_heldout_all.py --results-root ...`로
  arm별·instruction별·scene별 수치를 모은다. 서로 다른 arm의 결과를 합쳐 판 수를 부풀리지 않는다.
- 종료 시 runner가 serve 정리, wrapper가 lease 해제. SSH 중단 시 원격 프로세스가 살아 있을 수
  있으므로 재발사 전 실제 프로세스·GPU 확인이 필요하다.

## 사전 confound audit (결과 판정 아님)

| gate | 상태 | 근거·해석 범위 |
|---|---|---|
| Length | N-A | 시계열 분류 AUROC 주장 없음. episode 단위 paired outcome 집계 |
| Task identity | 통제 계획 | 같은 instruction·scene·j·noise 쌍 비교, instruction 별도 집계 |
| Instruction balance | 통제 계획 | 고정 공통 49셀, arm마다 같은 판, instruction별 분모 공개 |
| In-sample rescue | 제한 있음 | detector는 타 j 40판만 사용. **연산자는 대상 j 실패판 fit 포함**; 대상 성공판은 fit 제외. β도 기존 실패 eval로 선정했으므로 전체 파이프 독립 holdout 아님 |
| Rollout pooling | N-A | detector per-record 입력 유지, episode-mean feature 분석 없음 |
| Phase/dwell | 제한 있음 | GT phase와 per-step gating 유지; dwell-matched geometry 주장 없음 |
| Observation/causation | 측정 계획 | 실제 intervention 결과만 비교; smoke는 배관 검증으로 한정 |
| Scene-local/general | 제한 있음 | 기존 성립 49셀의 scene-local 실험, 미등록 셀·새 scene 일반화 주장 금지 |

결과 claim은 제한된 **intervention effect** 범위로만 작성한다. detector까지 heldout인 조건과
연산자까지 heldout인 조건을 혼동하지 않는다.

## 구현 검증

- lerobot 컨테이너 `test_serve_lerobot.py`: 78개 통과(새 perstep 테스트 6개 포함).
- stdlib orchestrator helper 테스트: 6개 통과(누락/중복 noise, seed, paired 구제·파괴 분모).
- 별도 통합 집계 fixture: `collection_success`와 arm `success`를 반대로 둔 데이터에서
  rescue 1/1, destruction 1/1 확인(원래 label을 arm 결과로 읽는 오류 방지).
- serve·collector CLI smoke, shell syntax, git diff --check 통과.
- 런처는 실행 전 해당 머신이 사용할 모든 detector·NPZ·metadata SHA256을 대조한다.
