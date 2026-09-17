---
name: v4r-recollect-round
description: v4r 라운드(대상 scene을 신규 재수집 실패판으로 교체) 완주 — 원격 체인 outputs/tmp/v4r, seed0 7/7 cp_bands, drawer-L은 대상 scene 소멸
metadata:
  type: project
---

**v4r = v4sb 의 후속: 대상 scene 의 구 record 를 전량 빼고 신규 재수집본의 '실패' 판만
삽입**한다. fit pool = 타 4 scene 구 데이터 전부 + 대상 scene 신규 실패 (성공 제외는
[[v4sb-fit-pool]] 규약 그대로). 2026-09-01 완주.

- 신규 수집: `~/datasets/.../grid_phase_v4/v4r_collect/{kanu,worker1,worker2}/ps_base/
  <case_slug>/.../s{si}{kk:02d}/n{j}/ps_base/rollout.pkl` (160판 61G) + 라벨
  `grid_phase_v4/v4r_labels.tsv`. 경로의 `s400` 류는 **scene*100 + 지터 k** (kbase=99).
- 신규 pkl 의 `feature_phases` 는 수집 당시 serve cluster 코드('c5' 등)라 원본 shard 의
  GT codebook 과 어휘가 다르다 → segA 삽입 시 **`gt_phases` 를 써야** codebook 이 맞는다.
  또 `vl_hidden_states` 가 없어 신규 record 는 has_vl=False (다운스트림 미사용이라 무해).
- 원격 체인·스크립트: `~/workspace/temporal_vla/outputs/tmp/v4r/{run_v4r.sh, build_v4r.py,
  v4r_stats.py}` (git 밖 원격 전용). shard: `grid_phase_v4/segA_v4r`(24G) ·
  `segA_v4r_ck8`(24G). detector `outputs/analysis/grid_phase/detector_v4r/cluster-k8`.
- 결과: 신규 실패 60판/8640 record 삽입(oven 17·jug 22·candle 8·marsh 6·bread 5·dish 2·
  drawer-L 0). detector 3/1/1 seed 0 에서 **7/7 cp_bands**(α 0.05/0.1/0.2), 재추첨 불필요.
- **OpenDrawer_left 는 신규 10판이 전부 성공** → 대상 scene s4 가 shard 에서 통째로 사라져
  scene 4개짜리가 됐다. 3/1/1 이 train 2 scene 으로 줄고 target_split 이 '?' 다 (구제 대상
  없음). 이 task 만 해석이 다르다는 점을 결과 표에서 반드시 밝힐 것.

**Why:** 구 수집 라벨 신뢰도 문제로 대상 scene 만 재수집했고, 그 데이터로 AE·cluster·
detector 를 전부 다시 학습해야 연산자 설계가 같은 라벨 위에 선다.
**How to apply:** cluster 라벨 소비처는 `outputs/tmp/v4r/labels_tsv/cluster_labels_<task>_k8.tsv`
(ep_id·rec_idx·cluster·scene·noise·jitter·succ). [[remote-shard-stream-crc]]
