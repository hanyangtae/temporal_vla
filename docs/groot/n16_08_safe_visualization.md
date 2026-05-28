# SAFE x GR00T N1.6 RoboCasa — Visualization

Per-timestep detector input feature를 대상으로 t-SNE/silhouette/overlay 진단을 만든다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - [07 SAFE Detector](n16_07_safe_detector.md)
> - **08 SAFE Visualization (이 문서)**
> - [09 SAFE Parity](n16_09_safe_parity.md)
> - [10 SAFE Report](n16_10_safe_report.md)

## SAFE Feature Visualization

SAFE 논문 Figure 1류의 latent-space 진단은 SAFE loader가 만든 per-timestep detector input feature를 대상으로 한다. 초기 t-SNE artifact는 `mean/mean` aggregation으로 만들었고, 최종 detector의 aggregation은 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`이다. 현재 visualization/silhouette script의 기본값은 최종 detector aggregation이며, 초기 artifact를 재생성할 때만 `--horizon-idx-rel mean --diff-idx-rel mean`을 명시한다.

최종 aggregation 기준 detector input:

```text
[T, 4, 16, 1024] -> horizon mean, diff concat(first,last) -> [T, 2048]
```

Visualization 산출물은 GR00T N1.6 eval output tree 아래에 둔다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep
```

각 visualization directory는 `feats_projected_skip1.pkl`, `feats_vis_skip1-succ.png`, `feats_vis_skip1-taskid.png`, `manifest.json`을 가진다. task structure와 success/failure signal을 한 그림에서 보기 위해 후처리 overlay도 저장한다.

- `feats_vis_skip1-taskid_failred.png`: 기존 task-id t-SNE 좌표를 그대로 쓰고, success datapoint(=inference)은 task id 색, failure rollout의 datapoint(=inference)은 단색 빨강으로 칠한다.
- `feats_vis_skip1-taskid_failure_overlay.png`: task id 색상 위에 실패 rollout의 datapoint을 검은 테두리로 겹친다.
- `feats_vis_skip1-task_success_facets.png`: task별 subplot 안에서 success rollout의 datapoint(=inference)은 파란색, failure rollout의 datapoint(=inference)은 episode 내 상대 시간에 따라 붉게 표시한다.

`manifest.json`에는 source split/task, projector, aggregation, rollout count, feature count, 생성된 output 파일명을 기록한다.

현재 생성된 t-SNE artifacts:

| scope | rollout | timestep feature | path |
|---|---:|---:|---|
| all splits | 600 | 18,428 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/all/tsne_mean_mean` |
| `val_unseen` | 200 | 5,660 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean` |
| `val_unseen/OpenDrawer` | 100 | 2,041 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen_OpenDrawer/tsne_mean_mean` |
| `val_unseen/PnPCounterToCab` | 100 | 3,619 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen_PnPCounterToCab/tsne_mean_mean` |
| all splits, SAFE-style | 600 | 18,428 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/safe_style_visualize_features/all_hmean_dconcat_2-tsne` |
| `val_unseen`, SAFE-style | 200 | 5,660 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/safe_style_visualize_features/val_unseen_hmean_dconcat_2-tsne` |

Silhouette 산출물:

| aggregation | path | conclusion |
|---|---|---|
| `mean/mean` | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/silhouette_mean_mean` | success/failure silhouette near zero |
| `concat-2/0.0` | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/silhouette_hconcat2_d0p0` | 초기 detector-metric 후보에서도 static failure zone은 약함 |

재생성 runner:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py
```

Overlay runner:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py
```

예시:

```bash
/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py \
  --scope val_unseen \
  --task PnPCounterToCab \
  --projector tsne

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py \
  outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean
```

SAFE-style feature plot 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/plot_safe_style_feature_space.py \
  --scope val_unseen \
  --projector tsne
```

Silhouette 진단:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py
```

Rollout-mean separability 진단:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/diagnose_rollout_mean_feature_separability.py
```

초기 관찰:

- `val_unseen` 전체로 보면 task structure와 success/failure signal이 함께 섞인다.
- `OpenDrawer` 단독 t-SNE는 success/failure separation이 약하다.
- `PnPCounterToCab` 단독 t-SNE는 실패 rollout 후반부로 보이는 red/orange region이 더 뚜렷하다.
- overlay 기준으로도 `PnPCounterToCab`은 late-failure datapoint(=inference)이 특정 영역에 비교적 많이 몰리지만, `OpenDrawer`는 success/failure가 더 강하게 섞인다.
- 최종 aggregation의 original 2048-D silhouette에서도 `val_unseen` success/failure Mahalanobis score는 `-0.0027`이고, task+failure도 음수다. static failure zone 근거는 약하고, detector score trajectory 중심으로 해석한다.
