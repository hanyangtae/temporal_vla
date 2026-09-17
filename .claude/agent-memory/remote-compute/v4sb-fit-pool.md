---
name: v4sb-fit-pool
description: v4sb(대상 scene 성공 제외) fit-pool 재학습 라운드 — 산출물 경로·필터 스크립트 위치·seed0로 7/7 cp_bands 확보(2026-09-01)
metadata:
  type: project
---

**v4sb = "대상 scene 의 성공 에피소드 record 를 fit pool 에서 뺀 v4"**. 타 4 scene 전 record
+ 대상 scene 의 실패 record 만으로 AE·KMeans(k8)·detector·CP 밴드를 전부 재학습한다.
대상 scene: PPCC_bread s1 / OpenDrawer_left s4 / PPCC_marshmallow s3 / PPCC_candle s3 /
PPCC_jug s4 / DishwasherRack_out s4 / OvenRack_out s4 (7 task, coffee·apple·drawer-R 제외).

산출물 (2026-09-01 완주, 전부 `_v4sb` 접미 — v4 원본 불변):
- 원격 shard 사본 `~/datasets/.../grid_phase_v4/segA_v4sb`(25G) · 재작성본 `segA_v4sb_ck8`(25G)
- 원격 체인·필터 스크립트: `~/workspace/temporal_vla/outputs/tmp/v4sb/{run_v4sb.sh,
  filter_v4sb.py,v4sb_stats.py}` — **git 에 없는 원격 전용 파일**(v4_align 체인과 같은 관행).
- 회수본: 로컬 `outputs/tmp/v4sb/` + `outputs/analysis/grid_phase/detector_v4sb/cluster-k8/`

실측 교훈:
- shard npz 멤버는 stored(무압축) → 행 단위 스트리밍 필터가 가능하고 7 shard 9분이면 끝난다.
  단 X.npy 가 4GB 를 넘어 `zipfile.open(w)` 에 **force_zip64=True 필수**(없으면 close 에서
  "File size too large").
- detector 분할은 [[v4-sega-scene-split]] 대로 3/1/1 seed 0 을 쓰면 v4sb 에서도 **7/7 task 가
  seed 0 한 번에 cp_bands 확보**(재추첨 불필요). 대상 scene 이 train 에 떨어진 task(dishwasher,
  drawer-L, oven, jug)와 test 에 떨어진 task(bread, candle, marshmallow)가 섞이므로,
  결과 해석 시 `outputs/tmp/v4sb/detector_split_v4sb.tsv` 의 target_split 열을 반드시 볼 것.

**Why:** 대상 scene 성공이 fit 에 들어가면 그 scene 에서의 read/steer 평가가 in-sample 이 된다.
**How to apply:** 같은 규약으로 다른 task/scene 을 돌릴 땐 `filter_v4sb.py --targets TASK:SCENE`
만 바꿔 체인을 재실행하면 된다. [[remote-node-hardware]]
