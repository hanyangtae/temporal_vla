# k8 추가 11scene 평가

사용자 09-09 지시로 instruction당 3scene 완비 제한을 풀고 지정 후보 및 추가 Drawer-right s1을 실행한다. 정본은 `configs/experiments/v6_ck8_extra_20260909/{targets,episodes}.tsv`. Drawer-right s0/s1 모두 j4 지정. 나머지는 제시한 후보 중 target 성공이 많은 j, 동률은 작은 j를 택했다.

| 원래 instruction | scene→j | arm당 판수 |
|---|---|---:|
| OpenDrawer/right | s0→j4, s1→j4 | 20 |
| DishwasherRack/out-left | s1→j2 | 10 |
| DishwasherRack/out-right | s1→j3, s2→j3 | 20 |
| OvenRack/out-left | s0→j0, s1→j0 | 20 |
| PPCC/bread | s0→j0 | 10 |
| PPCC/jug | s0→j2, s1→j1 | 20 |
| PPCC/marshmallow | s0→j0 | 10 |

110판/arm ×5arm=550판. 기존300판과 출력경로 분리. plain .8/jfair .9, reseed 및 reseed→plain/jfair, L12 common shift v2/global denoise/k8 유지. GT 및 baseline replay 없음. 원 수집 성공판 포함해 파괴율 집계; 신규 baseline 대조 아님.

Detector target전체 제외40판, 연산자 동일50판에서 target성공만 제외. jug s0와 dishwasher-left s1 sparse success. detector 성공calibration pool 최소3판에 맞춰 이번 build는 `--min-calib-succ 3`(기본9 유지). 기존 empirical LOO band 계산이며 90% coverage 보장 아님. target를 calibration에 추가하지 않는다. operator 누락cluster는 기존pure reseed fallback, 추가reseed 없음.

Oven/Dishwasher 원래instruction과 artifact slug 좌우 반전 mapping을 명시했다. Detector에는 매cell artifact stem만 담은singleton TSV를 전달해 반대쪽alias 매칭을 막는다. shard50좌표/label을 원래index와 대조한 뒤 학습한다.

원격 별도worktree에서 CPU8threads로 실행(raw전송 없음):

```bash
python scripts/analysis/grid_phase/build_v6_ck8_dwell.py \
 --store ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6 \
 --targets configs/experiments/v6_ck8_extra_20260909/targets.tsv \
 --episodes configs/experiments/v6_ck8_extra_20260909/episodes.tsv \
 --min-calib-succ 3 --out outputs/analysis/grid_phase/v6_ck8_extra_build_20260909
```

로컬 detached dispatcher:

```bash
python scripts/steer/online_gated/dispatch_v6_ck8_dwell.py \
 --main-root /home/dongkyu/pkt_ws/temporal_vla \
 --analysis-repo /home/kimseungjun/workspace/temporal_vla_ck8_dwell \
 --config-dir configs/experiments/v6_ck8_extra_20260909 \
 --build-dir outputs/analysis/grid_phase/v6_ck8_extra_build_20260909 \
 --state-dir outputs/analysis/v6_ck8_extra_dispatch_20260909 \
 --out-root outputs/eval/robocasa/groot_n15/og_v6_ck8_extra_20260909
```

머신별 detector 준비시 reseed 먼저, 완료후4arm. kanuGPU5/6/7, worker1/srv48GPU2, worker2/srv50GPU2. 발사시 빈GPU 및 lease 확인. machine은 원수집과 일치한다. 09-08 runbook의 길이통제/연산자 규약을 계승하며 이 문서의 scope/calibration 설정이 우선한다.
