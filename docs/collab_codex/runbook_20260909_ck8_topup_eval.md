# 신규 PPCC scene k8 heldout 평가

사용자 승인: kanu의 bread s4, marshmallow s3/s4에서 target을 선정하여 기존 5 arm 평가.
수집 세션과 상호 보고하지 않으며 수집 파일·프로세스는 수정하지 않는다.

| scene | layout/style | env seed | target | target S/F | other40 S/F | GPU / port 시작 |
|---|---|---|---|---|---|---|
| bread s4 | 2/2 | 101893 | j3 | 8/2 | 18/22 | 5 / 9600 |
| marshmallow s3 | 1/1 | 100498 | j4 | 6/4 | 25/15 | 6 / 9640 |
| marshmallow s4 | 6/6 | 100165 | j2 | 5/5 | 21/19 | 7 / 9680 |

- 원본 plan `4fa6496cd684`, kanu, checkpoint `lerobot_groot_n15__robocasa365_ckpt120000`.
- source_index는 scene별 50판. target S/F 및 other40 모두 양 클래스인 후보 중 성공 수 최대; 동률이면 j 오름차순.
- 원격 인덱스 스냅샷 `outputs/analysis/topup_eval_index_20260909/rollouts.tsv`와 이관 대기 로컬 meta를 대조했다. `has_pkl`은 해당 스냅샷 시점 값이며 학습 가능 판정은 raw SHA256 검증으로 한다.
- source index의 pkl SHA256을 원격 실제 파일과 검증하고 meta의 plan/machine/checkpoint/env/layout/style/scene/j/noise/seed/label을 대조한다. 이관이 덜 된 원본은 기다리며 제외하거나 대체하지 않는다.
- detector: target j 전체 제외, 나머지40. cluster별 dwell cap, 기존 empirical LOO 설정, 이번 세 scene은 성공 수 충분하여 `min_calib_succ=9`.
- operator: 같은50에서 target 성공만 제외, target 실패 포함. cluster별 cap 및 episode 동일 기여, plain beta .8 / jfair beta .9.
- raw에서 L12/denoise3/all49 mean을 읽고 기존 segA의 float16 roundtrip을 재현해 production k8 assigner에 입력. GT phase fit/평가 및 base replay 없음.
- setM v2 common shift, L12 hook, denoise all calls. 누락 cluster는 reseed fallback; reseed→operator는 이미 reseed한 pass 유지.
- 각 scene target10 × reseed/plain/jfair/reseed→plain/reseed→jfair =50회, 총150회. 수집 성공판 파괴와 실패판 구제는 수집 라벨 기준이며 fresh base 인과 효과로 단정하지 않는다.

## 준비 및 실행

원격 CPU의 격리 repo `/home/kimseungjun/workspace/temporal_vla_ck8_dwell`에서 `remote_compute.sh`로만 실행한다. 원본은 아카이브에 남긴다.

```bash
REMOTE_REPO=/home/kimseungjun/workspace/temporal_vla_ck8_dwell \
  bash scripts/utils/remote_compute.sh run-bg ck8_topup_20260909 \
  bash scripts/analysis/grid_phase/build_ck8_topup_20260909.sh
```

새 build wrapper는 중복 실행을 거부한다. 기존 prepared/released를 지우고 재실행하지 말고 실패 지점을 확인한다.

scene별 dispatcher는 `configs/experiments/v6_ck8_topup_20260909/<cell>`을 사용한다. GPU는 발사 직전 free/lease 확인 후 잡는다. detector만 준비되면 reseed부터; 둘 다 준비되어 있으면 단일 all queue로 GPU당2모델을 채운다.

```bash
python3 .claude/worktrees/eval-whole-pipe/scripts/steer/online_gated/dispatch_v6_ck8_dwell.py \
  --main-root "$PWD" \
  --analysis-repo /home/kimseungjun/workspace/temporal_vla_ck8_dwell \
  --config-dir configs/experiments/v6_ck8_topup_20260909/PPCC_bread_s4 \
  --build-dir outputs/analysis/grid_phase/v6_ck8_topup_build_20260909/PPCC_bread_s4 \
  --state-dir outputs/analysis/v6_ck8_topup_dispatch_20260909/PPCC_bread_s4 \
  --out-root outputs/eval/robocasa/groot_n15/og_v6_ck8_topup_20260909/PPCC_bread_s4 \
  --resume-existing
```

실제 장시간 launch는 세션에서 `start_new_session=True`, stdin DEVNULL, 파일 로그로 분리하였다. 나머지 두 scene도 cell 이름을 바꿔 독립 dispatcher로 실행한다. live pid는 각 state 디렉토리 `dispatcher.pid`가 정본이다.

## 검증 및 관찰

- Docker 기존 관련 테스트12개 통과; 새 plan의 세 replay manifest 모두 정확히10셀, seed/jitter 교차검증 통과. 이는 수집 좌표표 검증이며 base replay 실행이 아니다.
- Python syntax, bash syntax, 최종 staged diff check 통과.
- build 로그: `/tmp/remote_compute_logs/ck8_topup_20260909.log` (remote helper tail).
- build release: `outputs/analysis/grid_phase/v6_ck8_topup_build_20260909/<cell>/released/ready.json`.
- local state: `outputs/analysis/v6_ck8_topup_dispatch_20260909/<cell>/`.
- eval: `outputs/eval/robocasa/groot_n15/og_v6_ck8_topup_20260909/<cell>/{all 또는 reseed,operators}`. stage 선택에 따라 경로가 달라진다.
- 최종 완료는 DONE, arm별10행/scene, errors 없음, 원본좌표 및 계약 일치와 serve/lease 회수로 판정한다.
