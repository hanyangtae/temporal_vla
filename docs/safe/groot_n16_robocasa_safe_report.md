# SAFE x GR00T N1.6 RoboCasa Reproduction Report

## 범위

이 문서는 SAFE 논문식 failure detection을 GR00T N1.6 RoboCasa에 맞춰 작은 규모로 재현한 결과를 정리한다. 목표는 GR00T N1.6 rollout에서 VLA latent feature를 추출하고, SAFE detector 학습, conformal calibration, unseen-task evaluation, latent-space visualization까지 이어지는 end-to-end path를 닫는 것이다.

현재 결론은 명확하다. SAFE wiring은 닫혔고, trajectory 후반의 success/failure separability가 보인다. 최종 산출물은 rollout-level failure monitoring baseline, conformal operating point, latent-space diagnostic artifact다.

평가 범위는 held-out unseen-task evaluation이다. Model weight update, hparam selection, CP threshold calibration은 `train`과 `val_seen`에서 처리했다. 최종 aggregation 후보를 고정하는 과정에서는 `val_unseen` 시각화와 separability 진단을 참고했으므로, 다음 검증 축은 새 rollout seed로 수집한 test set에서 final detector와 CP threshold를 그대로 재평가하는 것이다.

## SAFE 이론

SAFE의 핵심 가정은 VLA policy의 내부 latent feature trajectory가 최종 rollout 성공/실패에 대한 정보를 담고 있다는 것이다. VLA를 action 출력 black box 관점에서 확장해, 매 policy step마다 생성되는 latent feature를 failure detector의 입력으로 사용한다.

rollout을 \(\tau\), timestep을 \(t\), VLA latent feature를 \(h_t\)라고 하자. SAFE detector는 현재까지의 prefix feature를 보고 failure score를 낸다.

```text
s_t = f_\theta(h_{1:t})
```

여기서 \(s_t\)는 timestep \(t\)까지 관찰한 정보로부터 계산한 failure score다. LSTM detector는 prefix sequence \(h_{1:t}\)를 직접 모델링하고, MLP류 detector는 timestep feature를 독립적으로 볼 수 있다. 본 재현에서는 SAFE LSTM을 우선 baseline으로 사용했다.

SAFE의 threshold는 calibration split의 score 분포에서 추정하고, conformal prediction 방식으로 threshold 또는 score band를 만든다. False-positive control 관점에서는 success/negative rollout의 score로 one-sided threshold를 잡는 것이 자연스럽다. SAFE repo의 split CP 평가는 binary conformal table을 label-conditional하게 남기므로, 결과를 읽을 때 `calib on=pos`와 `calib on=neg` 행을 구분한다. Rollout-level score는 다음처럼 정의할 수 있다.

```text
S(\tau) = \max_{t \le T_\tau} s_t
```

Success/negative calibration score 집합은 다음과 같이 쓸 수 있다.

```text
\mathcal{S}_{cal}^{succ}
  = \{ S(\tau_i) : y_i = 0,\ \tau_i \in \mathcal{D}_{cal} \}
```

One-sided success-calibrated threshold는 error tolerance \(\alpha\)에 대해 success rollout score의 상위 quantile로 잡는다.

```text
q_{1-\alpha}
  = \operatorname{Quantile}_{1-\alpha}(\mathcal{S}_{cal}^{succ})
```

이 threshold를 넘는 첫 timestep을 detection time으로 둔다.

```text
\hat{t}_{det}(\tau)
  = \min \{ t : s_t > q_{1-\alpha} \}
```

Functional CP는 scalar rollout score 대신 timestep별 score curve를 calibration한다. Success rollout들의 timestep별 score를 \(u_i(t)\)라고 하면, timestep별 band는 다음처럼 쓸 수 있다.

```text
q_{1-\alpha}(t)
  = \operatorname{Quantile}_{1-\alpha}
    \{ u_i(t) \}_{i \in \mathcal{D}_{cal}^{succ}}
```

Functional CP의 intervention rule은 다음과 같다.

```text
\hat{t}_{det}(\tau)
  = \min \{ t : s_\tau(t) > q_{1-\alpha}(t) \}
```

Detection time은 rollout length로 정규화해 비교한다.

```text
T_{det} = \frac{\hat{t}_{det}}{T_\tau}
```

작을수록 early detection이다. CP는 detector score 위에서 threshold를 정하는 방법이므로, early score separability가 CP 기반 early detection의 상한을 결정한다.

## GR00T N1.6 Feature 설계

SAFE repo의 pi0 diffusion loader는 policy feature를 다음 구조로 다룬다.

```text
(n_diff_steps, n_pred_horizon, dim_feats)
```

Loader는 먼저 action horizon axis를 `horizon_idx_rel`로 줄이고, 다음으로 diffusion axis를 `diff_idx_rel`로 줄인다. pi0 diffusion 계열에서는 diffusion step과 prediction horizon 축을 보존한 뒤 detector train/eval 단계에서 aggregation한다.

pi0-FAST는 다르다. token prediction 계열이므로 feature가 다음 구조로 저장되고, `token_idx_rel`로 token axis를 줄인다.

```text
(n_tokens, dim_feats)
```

GR00T N1.6은 action generation이 flow-matching 계열이므로, SAFE 이식에서는 pi0-FAST token aggregation보다 pi0 diffusion loader의 축 구조가 더 가깝다. 이에 따라 GR00T N1.6 SAFE feature는 DiT output의 action-token 영역에서 추출하고, 수집 단계에서 diffusion step과 prediction horizon 축을 보존한다.

GR00T N1.6 checkpoint의 model-level action horizon은 50이다. 따라서 DiT output의 마지막 50 token은 action-token 영역으로 본다.

```python
all_action_tokens = model_output[:, -50:, :]
```

선택한 RoboCasa checkpoint profile의 modality config에서 실제 decoded/executed action horizon은 16이다. 기본 SAFE feature는 action-token block의 leading 16 positions를 사용한다.

```python
safe_tokens = model_output[:, -50:, :][:, :16, :]
```

여기서 \(H=16\)은 action-token block 내부의 leading 16 positions를 뜻한다. 처리 순서는 전체 sequence에서 마지막 50 token action block을 잡고, 그 block 내부에서 RoboCasa가 실제 decode하는 앞 16 positions를 선택하는 방식이다.

결과적으로 한 policy step의 raw feature shape은 다음과 같다.

```text
[K, H, D] = [4, 16, 1024]
```

- \(K\): flow-matching denoising step axis
- \(H\): valid RoboCasa action horizon
- \(D\): GR00T hidden dimension

Rollout 전체로는 다음 shape이 저장된다.

```text
[T, K, H, D] = [T, 4, 16, 1024]
```

SAFE loader는 이 feature를 읽은 뒤 detector train/eval config에서 aggregation한다. 최종 SAFE-LSTM은 validation 성능 기준으로 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`를 선택했다.

```text
[T, 4, 16, 1024] -> [T, 2048]
```

이 설계는 SAFE repo의 pi0 diffusion loader와 같은 원칙을 따른다. feature 축은 수집 시점에 보존하고, aggregation choice는 detector training/evaluation config에서 선택한다.

## 실험 설정

Base checkpoint는 GR00T N1.6 RoboCasa PandaOmron checkpoint다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B
```

GR00T N1.6 success-rate 기준선은 upstream GR00T ZMQ evaluation path로 둔다. HTTP `/act` path는 observation/action parity 검증 후 SR 지표에 편입한다.

SAFE rollout collection은 dedicated ZMQ feature server로 수행했다. Feature endpoint는 normal action path를 유지하면서 action과 feature를 함께 저장한다. Official direct policy action과 SAFE feature path action의 동등성은 action key별 비교에서 `max_abs=0.0`으로 확인했다.

Task 후보는 RoboCasa v0.2 (`robocasa_v02`) eval task와 robocasa365 v1.0 atomic data가 의미적으로 대응되는 task 중에서 고른다. 최종 6-task set은 다음이다.

| task id | RoboCasa v0.2 task | robocasa365 v1.0 task | official SR |
|---:|---|---|---:|
| 0 | `CoffeeSetupMug` | `CoffeeSetupMug` | 31.0% |
| 1 | `OpenSingleDoor` | `OpenCabinet` | 81.5% |
| 2 | `PnPCounterToCab` | `PickPlaceCounterToCabinet` | 47.5% |
| 3 | `PnPSinkToCounter` | `PickPlaceSinkToCounter` | 50.0% |
| 4 | `PnPCounterToStove` | `PickPlaceCounterToStove` | 63.2% |
| 5 | `OpenDrawer` | `OpenDrawer` | 81.1% |

SAFE 논문/레포 방식에 맞춰 task-level seen/unseen split과 seen-task episode split을 사용했다.

- rollout cap: 100 per task
- task-level unseen ratio config: 0.25. With 6 tasks this rounds to 2 unseen tasks, so the realized held-out task fraction is 2/6.
- seen-task train ratio: 0.75
- detector train: seen-task train rollout
- validation / conformal calibration: `val_seen`
- final evaluation: `val_unseen`

Unseen task는 taxonomy constraint를 두어 Open 계열 1개와 PnP 계열 1개로 고정했다. 실제 unseen task는 `OpenDrawer`와 `PnPCounterToCab`이다.

| split | source | count |
|---|---|---:|
| `train` | 4 seen tasks x 75 rollout | 300 |
| `val_seen` | 4 seen tasks x 25 rollout | 100 |
| `val_unseen` | 2 unseen tasks x 100 rollout | 200 |

Split 요약:

| split | total | success | failure | SR |
|---|---:|---:|---:|---:|
| `train` | 300 | 141 | 159 | 47.0% |
| `val_seen` | 100 | 58 | 42 | 58.0% |
| `val_unseen` | 200 | 114 | 86 | 57.0% |

## RoboCasa365 18-Task Collection 결과

기존 SAFE-LSTM detector 재현은 위의 6-task split을 사용한다. 별도로, RoboCasa365 `target_atomic_seen18` 전체 18개 task에 대해 GR00T N1.6 checkpoint-120000 SAFE feature collection을 완료했다. 이 collection은 latent-space visualization과 후속 SAFE split 구성의 raw rollout source로 사용한다.

```text
run id: target_atomic_seen18_ckpt120000_robocasa365_100ep
root: /home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/raw_rollouts
checkpoint profile: configs/checkpoints/groot__robocasa365_ckpt120000.yaml
env source: robocasa365
task set: target_atomic_seen18
episodes: 18 tasks x 100 = 1800
seed start: 100000
seed range per task: 100000..100099
seed formula: seed = 100000 + episode_idx
artifacts: 1800 pkl, 1800 mp4, 1800 csv
feature kind: groot_n16_dit_valid_action_tokens_pre_velocity
feature shape per policy step: [4, 16, 1024]
verifier: status=ok
completed at: 2026-05-27 04:02:09 KST
```

동일 `episode_idx`는 모든 task에서 같은 seed를 사용한다. 예를 들어 각 task의 `episode_idx=0`은 seed `100000`, `episode_idx=99`는 seed `100099`다. Seed 기록은 `raw_rollouts/collection_summary.tsv`에 남아 있다.

Total SR은 `967/1800 = 53.7%`다.

| task id | RoboCasa365 task | success | failure | SR |
|---:|---|---:|---:|---:|
| 0 | `CloseBlenderLid` | 15 | 85 | 15.0% |
| 1 | `CloseFridge` | 82 | 18 | 82.0% |
| 2 | `CloseToasterOvenDoor` | 48 | 52 | 48.0% |
| 3 | `CoffeeSetupMug` | 29 | 71 | 29.0% |
| 4 | `NavigateKitchen` | 42 | 58 | 42.0% |
| 5 | `OpenCabinet` | 56 | 44 | 56.0% |
| 6 | `OpenDrawer` | 37 | 63 | 37.0% |
| 7 | `OpenStandMixerHead` | 82 | 18 | 82.0% |
| 8 | `PickPlaceCounterToCabinet` | 66 | 34 | 66.0% |
| 9 | `PickPlaceCounterToStove` | 61 | 39 | 61.0% |
| 10 | `PickPlaceDrawerToCounter` | 41 | 59 | 41.0% |
| 11 | `PickPlaceSinkToCounter` | 76 | 24 | 76.0% |
| 12 | `PickPlaceToasterToCounter` | 81 | 19 | 81.0% |
| 13 | `SlideDishwasherRack` | 60 | 40 | 60.0% |
| 14 | `TurnOffStove` | 25 | 75 | 25.0% |
| 15 | `TurnOnElectricKettle` | 81 | 19 | 81.0% |
| 16 | `TurnOnMicrowave` | 48 | 52 | 48.0% |
| 17 | `TurnOnSinkFaucet` | 37 | 63 | 37.0% |
| all | `target_atomic_seen18` | 967 | 833 | 53.7% |

이 결과는 rollout-level success/failure label과 GR00T N1.6 latent feature가 함께 저장된 producer artifact다. Detector 성능 claim은 아직 이 18-task collection에 대해 새 split, train, calibration, evaluation을 수행한 뒤에만 업데이트한다.

초기 latent-space diagnostic artifact도 생성했다. Aggregation은 최종 SAFE-LSTM 입력과 같은 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`이며, per-step feature는 `[T, 2048]`이다.

| projection | source points | output |
|---|---:|---|
| PCA | 54,582 timestep | `outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/visualizations/feature_space/safe_style_visualize_features/all_hmean_dconcat_2-pca-target_atomic_seen18_100ep_full` |
| t-SNE | 20,000 sampled timestep | `outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/visualizations/feature_space/safe_style_visualize_features/all_hmean_dconcat_2-tsne-target_atomic_seen18_100ep_sample20k-sample20000` |

원본 `[T, 2048]` feature 기준 군집도 진단도 생성했다. Silhouette은 5,000 timestep sample, kNN purity는 20,000 timestep sample로 계산했다.

| diagnostic | label | value | baseline / note |
|---|---|---:|---|
| silhouette, euclidean | success/failure | 0.0128 | near-zero separation |
| silhouette, euclidean | task id | -0.1173 | weak global task cluster by silhouette |
| silhouette, mahalanobis shrinkage | success/failure | 0.0041 | near-zero separation |
| silhouette, mahalanobis shrinkage | task id | -0.0920 | weak global task cluster by silhouette |
| kNN purity, k=10 | task id | 0.2238 | random same-label baseline 0.0587 |
| kNN purity, k=10 | success/failure | 0.6132 | random same-label baseline 0.5693 |

따라서 18-task feature에는 local task-neighborhood signal이 있지만, static success/failure cluster separation은 약하다. Success/failure 분석은 정적 2D embedding보다 rollout trajectory, SAFE-LSTM score, CP threshold crossing 기준으로 해석한다.

## Detector 학습

SAFE repo에 GR00T N1.6 dataset loader/config를 추가했다.

```text
/home/dongkyu/pdk_ws/SAFE/failure_prob/data/groot_n16.py
/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/dataset/groot_n16.yaml
```

Loader contract는 다음과 같다.

- split directory는 `train`, `val_seen`, `val_unseen`을 물리적으로 유지한다.
- per-step hidden feature `[4, 16, 1024]`를 읽는다.
- detector input은 aggregation 후 `[T, D']`다. 최종 설정에서는 `[T, 2048]`이다.
- `val_seen`은 validation과 conformal calibration에 쓰고, `val_unseen`은 held-out unseen-task 평가에 쓴다.

최종 학습 설정:

| item | value |
|---|---|
| model | SAFE `lstm` |
| seeds | `0`, `1`, `2` |
| epochs | `1000` |
| batch size | `64` |
| learning rate | `3e-4` |
| regularization | `lambda_reg=1` |
| aggregation | `horizon_idx_rel=mean`, `diff_idx_rel=concat-2` |
| selected checkpoint | `seed2` |
| final CP alpha | `0.2` |

현재 데이터에는 episode-level success/failure label만 있고 inference-step-level failure-onset label이 없으므로 timing plot은 비활성화했다.

## 결과

Aggregation ablation은 `horizon_idx_rel`과 `diff_idx_rel`을 `{0.0, 1.0, mean, concat-2}`에서 비교했다. 아래 표는 초기 detector-metric 기준 결과다. 이후 SAFE-style feature visualization과 timestep-level separability 진단을 반영해 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`를 최종 후보로 고정하고 hparam sweep을 다시 수행했다.

| metric | mean ± std |
|---|---:|
| initial metric-best aggregation | `horizon_idx_rel=concat-2`, `diff_idx_rel=0.0` |
| feature dim | `2048` |
| `val_seen` bal-acc | `0.932 ± 0.011` |
| `val_seen` T-det | `0.574 ± 0.026` |
| `val_seen` ROC-AUC | `0.922 ± 0.034` |
| `val_unseen` bal-acc | `0.785 ± 0.021` |
| `val_unseen` T-det | `0.694 ± 0.015` |
| `val_unseen` ROC-AUC | `0.749 ± 0.053` |

그 다음 aggregation을 고정하고 `lr`와 `lambda_reg` sweep을 수행했다.

| metric | mean ± std |
|---|---:|
| best hparam | `lr=3e-4`, `lambda_reg=1` |
| `val_seen` bal-acc | `0.985 ± 0.012` |
| `val_seen` T-det | `0.539 ± 0.130` |
| `val_seen` ROC-AUC | `0.995 ± 0.006` |
| `val_unseen` bal-acc | `0.981 ± 0.028` |
| `val_unseen` T-det | `0.642 ± 0.052` |
| `val_unseen` ROC-AUC | `0.994 ± 0.008` |

최종 detector는 hparam sweep의 `seed2` checkpoint로 고정했다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/model_final.ckpt
```

Fixed threshold baseline은 `val_seen`에서 max-so-far balanced accuracy를 최대화하는 threshold를 고르고 이를 `val_unseen`에 그대로 적용한 결과다.

| metric | value |
|---|---:|
| threshold | `0.5487` |
| `val_seen` bal-acc | `1.0000` |
| `val_seen` TPR/TNR | `1.0000 / 1.0000` |
| `val_seen` mean T-det | `0.7212` |
| `val_unseen` bal-acc | `1.0000` |
| `val_unseen` TPR/TNR | `1.0000 / 1.0000` |
| `val_unseen` acc/F1 | `1.0000 / 1.0000` |
| `val_unseen` mean T-det | `0.8194` |

최종 운영점은 split conformal prediction으로 고정했다. Alpha sweep은 최종 선택된 aggregation/hparam/seed2 checkpoint의 score 위에서 수행했다.

| metric | value |
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

가장 실용적인 운영점은 `split_cp`, `alpha=0.2`, `calib_label=neg_success`, `by final end`다. 이 설정은 failure recall을 유지하면서 성공 rollout에 대한 false alarm을 calibration한다. mean T-det는 `0.4114`로 이전 운영점보다 앞당겨졌고, 현재 산출물은 rollout-level failure monitoring 운영점으로 기록한다.

Functional CP band도 SAFE repo 구현 그대로 계산했다. `val_seen`의 successful score curve를 calibration curve로 쓰고, `FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)`로 timestep-wise upper band를 만든다.

| metric | value |
|---|---:|
| method | `functional_cp` |
| alpha | `0.2` |
| eval time | `by final end` |
| calibration label | `neg_success` |
| `val_unseen` bal-acc | `0.9605` |
| `val_unseen` TPR/TNR | `1.0000 / 0.9211` |
| `val_unseen` acc/F1 | `0.9550 / 0.9503` |
| `val_unseen` mean T-det | `0.4251` |

Functional CP의 best by-final-end point는 `alpha=0.05`에서 bal-acc `1.0000`, T-det `0.6982`다. Functional CP도 정상적으로 동작한다. `alpha=0.2` 기준으로는 split CP 운영점보다 false alarm은 조금 줄고 detection은 약간 늦다. 현재 final operating point는 더 이른 detection을 우선해 split CP `alpha=0.2`로 둔다.

SAFE 논문 Figure 8류의 CP 시각화도 로컬 CSV에서 생성했다. 이 그림은 CP threshold/band 변화에 따른 balanced accuracy, T-det, FPR/FNR/TPR/TNR curve를 보여준다.

| artifact | path |
|---|---|
| CP plot script | `scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py` |
| source tables | `final_detector/split_cp_eval.csv`, `final_detector/functional_cp_eval.csv` |
| functional bands | `final_detector/functional_cp_bands.npz` |
| output directory | `outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/conformal_figure/by_final_end` |
| main plot | `cp_balacc_tdet.png` / `cp_balacc_tdet.pdf` |
| alpha sweep plots | `cp_alpha_{fpr,fnr,tpr,tnr,bal_acc}.png` |

Leakage sanity check는 다음과 같이 수행했다.

| check | result |
|---|---:|
| split symlink target overlap | `0` |
| task/episode id cross-split overlap | `0` |
| completed hparam checkpoints | `36 / 36` |

Leakage sanity check에서 split 중복은 0이다. Fixed threshold baseline의 `val_unseen` perfect score는 margin이 좁다. Final detector의 rollout-level max score 기준으로 `val_unseen` failure minimum은 약 `0.5490`, success maximum은 약 `0.5441`, fixed threshold는 `0.5487`이다. 실제 운영 성능은 calibration을 포함한 CP operating point를 기준으로 본다.

추가 확인이 필요한 항목은 다음과 같다.

- 같은 6 task에서 새 rollout seed test set을 추가 수집한다.
- 현재 final detector, aggregation, hparam, CP threshold를 고정한 채 재평가한다.
- random-label sanity를 수행해 label shuffle 시 성능이 chance level로 떨어지는지 확인한다.
- task-only 또는 length-only baseline을 확인해 episode length/task identity confound를 배제한다.

## 시각화

SAFE feature 시각화는 detector input feature를 사용한다. GR00T N1.6에서는 다음 feature를 뜻한다.

```text
[T, 4, 16, 1024] -> aggregation -> [T, D']
```

초기 t-SNE는 `mean/mean` aggregation으로 만들었다. 최종 detector aggregation은 `mean/concat-2`이며, 이 경우 `[T, 2048]` feature가 된다. 현재 visualization/silhouette script의 기본값은 최종 detector aggregation이며, 초기 artifact를 재생성할 때만 `--horizon-idx-rel mean --diff-idx-rel mean`을 명시한다.

t-SNE artifact는 전체 split, `val_unseen`, 그리고 `val_unseen`의 task별 subset에 대해 생성했다.

| scope | rollout | timestep feature |
|---|---:|---:|
| all splits | 600 | 18,428 |
| `val_unseen` | 200 | 5,660 |
| `val_unseen/OpenDrawer` | 100 | 2,041 |
| `val_unseen/PnPCounterToCab` | 100 | 3,619 |

Feature space에는 task 방향성이 일부 있고, 전역적인 success/failure 분리는 약하다. 이 결과는 detector input feature의 diagnostic artifact로 사용한다.

## 해석

이번 구현은 SAFE x GR00T N1.6 wiring 및 detector baseline을 end-to-end로 닫았다.

- GR00T N1.6 rollout feature export가 동작한다.
- SAFE-readable pkl schema가 생성된다.
- SAFE loader가 `[T, 4, 16, 1024]` feature를 읽는다.
- LSTM detector aggregation/hparam sweep, validation, conformal calibration, unseen-task evaluation이 완료된다.
- final checkpoint와 final operating point가 artifact로 고정됐다.
- t-SNE/overlay visualization artifact가 detector input feature에서 생성된다.

현재 detector는 이 작은 재현 범위에서 seen/unseen task failure monitoring에 사용할 수 있다. Selection bias는 새 rollout seed 평가로 점검한다. 아래 label schema는 rollout-level success/failure다.

```text
y_\tau \in \{0, 1\}
```

Frame-level onset label을 추가하면 다음 target을 정의할 수 있다.

```text
z_t \in \{0, 1\}
```

Failure onset time annotation을 추가하면 다음 변수를 둘 수 있다.

```text
t_{onset}
```

현재 SAFE LSTM 학습 target은 다음 mapping이다.

```text
f_\theta(h_{1:t}) \to y_\tau
```

Onset-supervised extension은 다음 target을 사용할 수 있다.

```text
g_\phi(h_{1:t}) \to z_t
```

또는 다음 target을 사용할 수 있다.

```text
g_\phi(h_{1:t}) \to \mathbf{1}[t \ge t_{onset}]
```

따라서 failed rollout의 중후반부에서 failure score가 상승하는 현상은 prefix가 점점 failed rollout class score를 높이는 것으로 읽는다. Proactive intervention 평가는 onset/intervention label을 추가한 뒤 별도 protocol로 다룬다.

## 한계와 다음 단계

다음 개선 축은 early separability다. CP는 threshold를 calibration하고, rollout 초반의 detector score separability가 early signal 품질을 결정한다.

다음 단계:

1. HTTP-vs-ZMQ observation/action parity를 검증한 뒤 HTTP `/act` SR을 통합 지표로 편입한다.
2. Proactive intervention이 목표라면 inference-step-level onset/intervention label을 수집하거나 정의한다.
3. `--feature-slice all`로 model-level `H=50` feature를 수집해 current `H=16` valid-horizon export와 비교할 수 있다.
4. Taskwise score-trajectory plot을 추가해 CP threshold crossing이 어느 phase에서 발생하는지 더 명확히 본다.

## 관련 파일

- [SAFE wiring runbook](groot_n16_robocasa_wiring.md)
- [Dedicated ZMQ feature server ADR](../adr/0001-dedicated-safe-groot-n16-zmq-server.md)
- `scripts/safe/groot_n16/robocasa/run_config.py`
- `scripts/safe/groot_n16/robocasa/run_config.sh`
- `scripts/safe/groot_n16/robocasa/safe_feature_vectors.py`
- `scripts/safe/groot_n16/robocasa/serve/feature_server.py`
- `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`
- `scripts/safe/groot_n16/robocasa/collect/collect_task_set_in_container.sh`
- `scripts/safe/groot_n16/robocasa/collect/collect_task_set_official_uv_host.sh`
- `scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh`
- `scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py`
- `scripts/safe/groot_n16/robocasa/train/train_lstm_mean_mean.sh`
- `scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh`
- `scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh`
- `scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py`
- `scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py`
- `scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py`
- `scripts/safe/groot_n16/robocasa/analyze/diagnose_rollout_mean_feature_separability.py`
- `scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py`
- `scripts/safe/groot_n16/robocasa/vis/plot_safe_style_feature_space.py`
- `scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py`
- `scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py`
- `scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py`
- `outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/README.md`
- `outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/final_operating_point.json`
- `outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/conformal_figure/by_final_end/README.md`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/data/groot_n16.py`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/dataset/groot_n16.yaml`
