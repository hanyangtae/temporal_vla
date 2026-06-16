# GR00T N1.6 RoboCasa — SAFE Detector

Paper-faithful split을 만들고 SAFE-LSTM detector를 학습한다. Aggregation ablation, hparam sweep, CP 운영점 고정까지 한 흐름이다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - **07 SAFE Detector (이 문서)**
> - [08 SAFE Visualization](n16_08_safe_visualization.md)
> - [09 SAFE Parity](n16_09_safe_parity.md)
> - [10 SAFE Report](n16_10_safe_report.md)

## Paper-Faithful SAFE Split

SAFE 논문/레포 방식에 맞춰 task-level split과 seen-task episode split을 사용한다.

- raw rollout cap: `max_rollouts_per_task: 100`
- task-level split: `unseen_task_ratio: 0.25`
- seen-task episode split: `seen_train_ratio: 0.75`
- detector train: seen task의 train rollout
- threshold / conformal calibration: `val_seen`
- final evaluation: `val_unseen`

따라서 6개 task에서는 `round(0.25 * 6) = 2`개 task가 unseen이 되고, 나머지 4개 task가 seen이 된다. 각 task를 100 rollout으로 맞추면 전체 600 rollout이며, split은 대략 다음 크기가 된다. SAFE 레포 DROID 설정의 `60/task`보다 큰 cap이지만, task별 SR이 낮아 성공 rollout이 부족할 수 있으므로 N1.6 RoboCasa에서는 `100/task`를 사용한다.

이번 small reproduction에서는 taxonomy constraint를 둔다. unseen task는 Open 계열 1개와 PnP 계열 1개로 고정한다. 실제 unseen task는 `OpenDrawer`와 `PnPCounterToCab`이다. 이 선택은 `OpenSingleDoor`를 seen 쪽에 남겨 robocasa365의 `OpenCabinet` 대응 경로를 계속 점검할 수 있게 하고, `val_unseen` 성공/실패 비율도 `114/86`으로 균형을 유지한다.

| split | source | count |
|---|---|---:|
| `train` | 4 seen tasks × 75 rollout | 300 |
| `val_seen` | 4 seen tasks × 25 rollout | 100 |
| `val_unseen` | 2 unseen tasks × 100 rollout | 200 |

`val_seen`은 validation과 conformal calibration 역할을 함께 한다. 별도 CP-only split이나 seen-task test split을 만들면 논문식 재현에서 벗어난다.

Split 생성 전 count 확인:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py \
  --source-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/raw_rollouts \
  --split-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split \
  --dry-run
```

Split 생성은 새 run에서 한 번 수행한다. 현재 run의 source of truth는 아래 `split` directory다. 기존 `split` directory가 있으면 script가 중단되므로, 재생성은 새 `ROBOCASA_SAFE_RUN_ID`에서 수행한다.

Split 최초 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py \
  --source-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/raw_rollouts \
  --split-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split
```

생성된 split:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split
```

Split summary:

| split | total | success | failure | SR |
|---|---:|---:|---:|---:|
| `train` | 300 | 141 | 159 | 47.0% |
| `val_seen` | 100 | 58 | 42 | 58.0% |
| `val_unseen` | 200 | 114 | 86 | 57.0% |

`manifest.tsv`와 `summary.tsv`를 함께 저장해 이후 학습 seed와 split seed를 분리한다.

## SAFE LSTM Final Detector

SAFE repo에는 GR00T N1.6용 dataset loader/config를 추가했다.

- `/home/dongkyu/pdk_ws/SAFE/failure_prob/data/groot_n16.py`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/dataset/groot_n16.yaml`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/__init__.py`

Loader contract:

- split directory는 `train`, `val_seen`, `val_unseen`을 물리적으로 유지한다.
- per-step hidden feature `[4, 16, 1024]`를 읽는다.
- detector input은 train/eval config의 aggregation에 따라 만든다.
- 최종 SAFE-LSTM은 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`를 사용하므로 detector input은 `[T, 2048]`이다.
- `val_seen`은 validation과 conformal calibration에 쓰고, `val_unseen`은 held-out unseen-task 평가에 쓴다.

관련 script:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.sh
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/safe_feature_vectors.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py
```

`run_config.py` / `run_config.sh`가 run id, output root, 최종 aggregation, hparam sweep directory의 단일 출처다. `safe_feature_vectors.py`가 `[K,H,D]` Flow-matching SAFE feature를 timestep-level SAFE feature vector로 aggregation하는 공용 Module이다.

Aggregation ablation 실행:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

WANDB_MODE=online \
bash scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh
```

Aggregation ablation 요약:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py
```

최종 aggregation 기준 hparam sweep 실행:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

WANDB_MODE=online \
bash scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh
```

Hparam sweep 요약:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py
```

Final detector 고정 및 CP artifact 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py
```

최종 선택:

- model: SAFE `lstm`
- epochs: `1000`
- batch size: `64`
- lr: `3e-4`
- lambda_reg: `1`
- aggregation: `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`
- selected checkpoint seed: `2`
- W&B project: `vla-safe`
- timing plots: disabled, because current data has episode-level success/failure only and inference-step-level failure-onset label이 없다.

최종 checkpoint:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/model_final.ckpt
```

최종 산출물:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector
```

이 directory에는 `model_final.ckpt`, `config.yaml`, `manifest.json`, `final_operating_point.json`, `fixed_threshold_eval.csv`, `split_cp_eval.csv`, `functional_cp_eval.csv`, `functional_cp_bands.npz`, `per_rollout_scores.csv`, `README.md`가 있다.

초기 aggregation ablation 결과:

| rank | horizon | diff | dim | val_seen bal-acc | val_seen T-det | val_seen ROC-AUC | val_unseen bal-acc | val_unseen T-det | val_unseen ROC-AUC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `concat-2` | `0.0` | 2048 | `0.932 ± 0.011` | `0.574 ± 0.026` | `0.922 ± 0.034` | `0.785 ± 0.021` | `0.694 ± 0.015` | `0.749 ± 0.053` |
| 10 | `mean` | `mean` | 1024 | `0.854 ± 0.039` | `0.653 ± 0.040` | `0.854 ± 0.042` | `0.754 ± 0.025` | `0.702 ± 0.015` | `0.779 ± 0.015` |

이후 SAFE-style feature visualization과 timestep-level separability 진단에서 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`를 최종 후보로 고정하고 hparam sweep을 다시 수행했다.

Hyperparameter sweep 결과:

| metric | mean ± std |
|---|---:|
| best hparam | `lr=3e-4`, `lambda_reg=1` |
| `val_seen` bal-acc | `0.985 ± 0.012` |
| `val_seen` T-det | `0.539 ± 0.130` |
| `val_seen` ROC-AUC | `0.995 ± 0.006` |
| `val_unseen` bal-acc | `0.981 ± 0.028` |
| `val_unseen` T-det | `0.642 ± 0.052` |
| `val_unseen` ROC-AUC | `0.994 ± 0.008` |

Final pinned detector 결과:

| item | value |
|---|---:|
| selected checkpoint | `seed2` |
| fixed threshold baseline | `0.5487` |
| fixed threshold `val_unseen` bal-acc | `1.0000` |
| fixed threshold `val_unseen` TPR/TNR | `1.0000 / 1.0000` |
| fixed threshold `val_unseen` mean T-det | `0.8194` |

최종 운영점은 split conformal prediction으로 고정한다. Fixed threshold는 baseline으로 함께 기록한다.

| item | value |
|---|---:|
| method | `split_cp` |
| alpha | `0.2` |
| eval time | `by final end` |
| calibration label | `neg_success` |
| threshold | `0.5301596522331238` |
| `val_unseen` bal-acc | `0.9518` |
| `val_unseen` TPR/TNR | `1.0000 / 0.9035` |
| `val_unseen` acc/F1 | `0.9450 / 0.9399` |
| `val_unseen` mean T-det | `0.4114` |

해석:

- wiring은 닫혔다. GR00T N1.6 rollout feature가 SAFE loader를 통과하고, LSTM 학습/validation/CP table 생성/checkpoint 저장까지 완료됐다.
- 논문식 feature aggregation ablation과 LSTM hyperparameter sweep을 수행했고, 최종 detector/checkpoint/threshold를 별도 산출물로 고정했다.
- `val_unseen`에서도 failure monitoring 성능은 강하다. 최종 CP 운영점의 mean T-det는 `0.4114`로 이전 운영점보다 앞당겨졌다. 현재 label scope는 rollout-level success/failure이며, proactive intervention 평가는 inference-step-level onset/intervention label을 추가한 뒤 다룬다.
- CP alpha sweep은 최종 선택된 aggregation/hparam/seed2 checkpoint의 score 위에서 수행했다.
- Functional CP band도 SAFE repo 구현 그대로 계산했다. `alpha=0.2`, `by final end`, success-calibrated functional CP는 `val_unseen` bal-acc `0.9605`, TPR/TNR `1.0000 / 0.9211`, mean T-det `0.4251`이다. Best by-final-end functional point는 `alpha=0.05`에서 bal-acc `1.0000`, mean T-det `0.6982`다.
- static latent-space failure zone 근거는 약하다. detector 성능은 정적 cluster 분리보다 LSTM score trajectory와 threshold crossing으로 해석한다.

SAFE 논문 Figure 8류의 CP 시각화는 다음 위치에 생성한다. 이 그림은 CP operating point curve다.

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/conformal_figure/by_final_end
```

기본 산출물은 `cp_balacc_tdet.{png,pdf}`와 `cp_alpha_{fpr,fnr,tpr,tnr,bal_acc}.{png,pdf}`다. 입력은 `final_detector/split_cp_eval.csv`와 `final_detector/functional_cp_eval.csv`이며, 로컬 CSV를 source of truth로 사용한다.

CP curve 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py \
  --eval-time "by final end"
```
