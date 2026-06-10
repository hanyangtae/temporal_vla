# SAFE x GR00T N1.6 RoboCasa 데이터 번들 안내

이 문서는 아래 압축 파일과 함께 전달하기 위한 사용 설명서입니다.

```text
safe_groot_n16_data.tar.gz
```

압축 파일에는 SAFE x GR00T N1.6 RoboCasa 실험에 사용한 rollout 데이터, split, SAFE detector 학습 결과, 시각화 결과, 관련 스크립트와 문서가 들어 있습니다.

## 압축 해제 방법

`temporal_vla` repo root에서 압축을 해제합니다.

```bash
cd /path/to/temporal_vla
tar -xzf /path/to/safe_groot_n16_data.tar.gz
```

압축 파일은 repo 기준 상대경로를 유지합니다. 따라서 압축을 풀면 아래 위치에 복원됩니다.

```text
outputs/eval/robocasa/groot_n16/...
scripts/safe/groot_n16/...
docs/...
```

## 포함된 항목

```text
outputs/eval/robocasa/groot_n16/rollouts_n16_seen5_20ep_upstream_video_20260519
outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep
outputs/eval/robocasa/groot_n16/safe_train_logs
outputs/eval/robocasa/groot_n16/safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep
scripts/safe/groot_n16
docs/safe_groot_n16_robocasa_wiring.md
docs/adr/0001-dedicated-safe-groot-n16-zmq-server.md
```

원본 기준 대략적인 크기는 다음과 같습니다.

```text
rollouts      2.7G
split         7.3M
train logs    21M
visualization 11M
scripts       212K
```

`safe_split...` 디렉토리는 원래 일부 파일이 원본 rollout을 가리키는 symlink였습니다. 이 번들은 `tar -h`로 만들도록 의도했기 때문에, 압축 해제 시 symlink가 아니라 실제 파일로 풀릴 수 있습니다. 재현에는 이쪽이 더 안전합니다.

## 데이터 요약

Split root:

```text
outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep
```

전체 규모:

```text
총 600 rollouts
총 6 tasks
task당 100 rollouts
```

Seen tasks:

```text
CoffeeSetupMug
OpenSingleDoor
PnPSinkToCounter
PnPCounterToStove
```

Unseen tasks:

```text
PnPCounterToCab
OpenDrawer
```

Episode 단위 success rate:

```text
train      141 / 300 = 0.470
val_seen    58 / 100 = 0.580
val_unseen 114 / 200 = 0.570

PnPCounterToCab 34 / 100 = 0.340
OpenDrawer      80 / 100 = 0.800
```

전체 split 요약은 아래 파일에서 확인할 수 있습니다.

```text
outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep/summary.tsv
```

## Feature Schema

각 rollout pkl은 environment step마다 하나의 feature tensor를 저장합니다.

Raw feature shape:

```text
[T_env, K, H, D]
```

이 번들의 실제 shape:

```text
[T_env, 4, 16, 1024]
```

각 축의 의미:

```text
T_env = rollout timestep 수
K     = GR00T N1.6 denoising step 수
H     = RoboCasa에서 실제로 사용하는 valid action horizon
D     = DiT output / action decoder input feature dimension
```

Feature source는 GR00T N1.6 action head의 DiT output action-token latent입니다. 위치는 action decoder에 들어가기 직전입니다. SAFE 논문에서 pi0 계열에 대해 사용하는 `pre_velocity` feature 위치에 대응시키는 설계입니다.

Detector 학습과 시각화에는 다음 aggregation을 사용했습니다.

```text
diff_idx_rel=mean
horizon_idx_rel=mean
```

따라서 각 timestep의 detector 입력은 다음과 같습니다.

```text
z_t = K축과 H축 평균, shape [1024]
```

## SAFE Repository 의존성

이 번들에는 SAFE repo 전체가 포함되어 있지 않습니다.

아래 스크립트는 SAFE repo 없이도 동작합니다.

```text
scripts/safe/groot_n16/robocasa/serve/feature_server.py
scripts/safe/groot_n16/robocasa/collect/collect_rollout.py
scripts/safe/groot_n16/robocasa/data/prepare_paper_split.py
scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py
scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py
```

아래 스크립트는 SAFE checkout이 필요합니다.

```text
scripts/safe/groot_n16/robocasa/train/run_lstm_mean_mean_seed0.sh
scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py
```

기본적으로 다음 위치를 기대합니다.

```text
/home/dongkyu/pdk_ws/SAFE
```

다른 위치에 있다면 다음처럼 지정할 수 있습니다.

```bash
SAFE_REPO=/path/to/SAFE
```

SAFE-LSTM을 다시 학습하거나 SAFE 원본 `visualize_features.py` 기반 시각화를 다시 실행하려면 SAFE repo가 필요합니다. 또한 이 실험에서 사용한 `groot_n16` dataset config가 SAFE repo 안에 있어야 합니다.

## GR00T Checkpoint 의존성

이 압축 파일에는 GR00T N1.6 checkpoint가 포함되어 있지 않습니다.

원래 사용한 로컬 checkpoint 경로는 다음과 같습니다.

```text
<cache>/checkpoints/nvidia/GR00T-N1.6-3B
```

Checkpoint는 모델을 다시 서빙하거나 새 rollout을 수집할 때만 필요합니다. 이미 저장된 rollout pkl, detector checkpoint, plot, silhouette 결과를 확인하는 데에는 필요하지 않습니다.

## 기존 결과 요약

SAFE-LSTM은 `diff_idx_rel=mean`, `horizon_idx_rel=mean` feature로 학습했습니다.

대략적인 평가 결과:

```text
val_unseen early max-so-far ROC-AUC: 0.701 +/- 0.047
val_unseen end max-so-far ROC-AUC:   0.849 +/- 0.121
functional CP alpha=0.2 bal-acc:     0.586 +/- 0.037
split CP alpha=0.2 bal-acc:          0.634 +/- 0.027

best diagnostic seed2 alpha=0.6:
  bal-acc 0.691
  T-det   0.453
  TPR     0.628
  TNR     0.754
```

해석:

```text
late/end failure monitoring에는 어느 정도 사용할 수 있습니다.
강한 proactive early failure detection claim은 아직 성립하지 않습니다.
```

## 시각화와 Silhouette

시각화 결과 root:

```text
outputs/eval/robocasa/groot_n16/safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep
```

주요 plot:

```text
all/tsne_mean_mean/feats_vis_skip1-taskid_failred.png
val_unseen/tsne_mean_mean/feats_vis_skip1-taskid_failred.png
val_unseen_OpenDrawer/tsne_mean_mean/feats_vis_skip1-taskid.png
val_unseen_PnPCounterToCab/tsne_mean_mean/feats_vis_skip1-taskid.png
```

Silhouette 결과:

```text
silhouette_mean_mean/silhouette_scores.tsv
silhouette_mean_mean/silhouette_scores.json
```

핵심 수치:

```text
original 1024-d, all
success/failure      0.008 euclidean / 0.016 cosine
task                -0.030 euclidean / -0.026 cosine
task+failure        -0.052 euclidean / -0.077 cosine

original 1024-d, val_unseen
success/failure      0.000 euclidean / 0.011 cosine
task                 0.017 euclidean / 0.039 cosine
task+failure        -0.014 euclidean / -0.012 cosine

t-SNE 2D, val_unseen
success/failure      0.006
task                 0.077
task+failure        -0.026
```

해석:

```text
현재 latent space에는 silhouette 기준으로 뚜렷한 success/failure cluster가 보이지 않습니다.
task cluster도 강하지 않습니다.
따라서 detector는 정적인 cluster 분리만으로 해석하기보다, trajectory score 변화와 CP threshold crossing을 중심으로 해석해야 합니다.
```

## 유용한 명령

Split 요약 확인:

```bash
sed -n '1,80p' outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep/summary.tsv
```

Silhouette 재계산:

```bash
python scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py
```

`val_unseen` task/failure overlay 재생성:

```bash
python scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py \
  outputs/eval/robocasa/groot_n16/safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean
```

SAFE repo가 있는 경우 SAFE-LSTM 학습 entrypoint:

```bash
SAFE_REPO=/path/to/SAFE \
bash scripts/safe/groot_n16/robocasa/train/run_lstm_mean_mean_seed0.sh
```

## 추가 문서

아래 문서를 함께 읽으면 wiring과 설계 결정을 더 자세히 볼 수 있습니다.

```text
docs/safe_groot_n16_robocasa_wiring.md
docs/adr/0001-dedicated-safe-groot-n16-zmq-server.md
```
