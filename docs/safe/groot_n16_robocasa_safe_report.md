# SAFE x GR00T N1.6 RoboCasa Reproduction Report

## 범위

이 문서는 SAFE 논문식 failure detection을 GR00T N1.6 RoboCasa에 맞춰 작은 규모로 재현한 결과를 정리한다. 목표는 SAFE 전체 benchmark를 동일 규모로 복제하는 것이 아니라, GR00T N1.6 rollout에서 VLA latent feature를 추출하고, SAFE detector 학습, conformal calibration, unseen-task evaluation, latent-space visualization까지 이어지는 end-to-end path를 닫는 것이다.

현재 결론은 명확하다. SAFE wiring은 닫혔고, trajectory 후반의 success/failure separability는 보인다. 하지만 현재 detector는 supervised failure-onset detector가 아니며, 강한 proactive intervention claim을 하기는 아직 어렵다.

## SAFE 이론

SAFE의 핵심 가정은 VLA policy의 내부 latent feature trajectory가 최종 rollout 성공/실패에 대한 정보를 담고 있다는 것이다. VLA를 단순히 action을 출력하는 black box로 보지 않고, 매 policy step마다 생성되는 latent feature를 failure detector의 입력으로 사용한다.

rollout을 \(\tau\), timestep을 \(t\), VLA latent feature를 \(h_t\)라고 하자. SAFE detector는 현재까지의 prefix feature를 보고 failure score를 낸다.

```text
s_t = f_\theta(h_{1:t})
```

여기서 \(s_t\)는 timestep \(t\)까지 관찰한 정보로부터 계산한 failure score다. LSTM detector는 prefix sequence \(h_{1:t}\)를 직접 모델링하고, MLP류 detector는 timestep feature를 독립적으로 볼 수 있다. 본 재현에서는 SAFE LSTM을 우선 baseline으로 사용했다.

SAFE의 threshold는 임의로 정하지 않는다. Calibration split의 score 분포를 추정하고, conformal prediction 방식으로 threshold 또는 score band를 만든다. False-positive control 관점에서는 success/negative rollout의 score로 one-sided threshold를 잡는 것이 자연스럽다. 다만 SAFE repo의 split CP 평가는 binary conformal table을 label-conditional하게 남기므로, 결과를 읽을 때 `calib on=pos`와 `calib on=neg` 행을 구분해야 한다. Rollout-level score는 다음처럼 정의할 수 있다.

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

Functional CP는 scalar rollout score가 아니라 timestep별 score curve를 calibration한다. Success rollout들의 timestep별 score를 \(u_i(t)\)라고 하면, timestep별 band는 다음처럼 쓸 수 있다.

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

작을수록 early detection이다. 다만 CP는 detector score 위에서 threshold를 정하는 방법이다. Early timestep에서 score 자체가 success/failure를 잘 분리하지 못하면, CP를 적용해도 early intervention 성능은 좋아지지 않는다.

## GR00T N1.6 Feature 설계

SAFE repo의 pi0 diffusion loader는 policy feature를 다음 구조로 다룬다.

```text
(n_diff_steps, n_pred_horizon, dim_feats)
```

Loader는 먼저 action horizon axis를 `horizon_idx_rel`로 줄이고, 다음으로 diffusion axis를 `diff_idx_rel`로 줄인다. 즉 pi0 diffusion 계열에서는 feature를 단일 vector로 미리 저장하지 않고, diffusion step과 prediction horizon 축을 보존한 뒤 detector train/eval 단계에서 aggregation한다.

pi0-FAST는 다르다. token prediction 계열이므로 feature가 다음 구조로 저장되고, `token_idx_rel`로 token axis를 줄인다.

```text
(n_tokens, dim_feats)
```

GR00T N1.6은 action generation이 flow-matching 계열이므로, SAFE 이식에서는 pi0-FAST token aggregation보다 pi0 diffusion loader의 축 구조가 더 가깝다. 이에 따라 GR00T N1.6 SAFE feature는 DiT output의 action-token 영역에서 추출하고, 수집 단계에서는 pooling하지 않는다.

GR00T N1.6 checkpoint의 model-level action horizon은 50이다. 따라서 DiT output의 마지막 50 token은 action-token 영역으로 본다.

```python
all_action_tokens = model_output[:, -50:, :]
```

하지만 RoboCasa PandaOmron modality config에서 실제 decoded/executed action horizon은 16이다. 뒤 34 token은 RoboCasa output으로 직접 decode되지 않으므로 기본 SAFE feature에서는 제외한다.

```python
safe_tokens = model_output[:, -50:, :][:, :16, :]
```

여기서 \(H=16\)은 전체 sequence의 마지막 16 token이라는 뜻이 아니다. 먼저 GR00T action-token block인 마지막 50 token을 잡고, 그 block 내부에서 RoboCasa가 실제 decode하는 leading 16 positions를 사용한다는 뜻이다.

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

SAFE loader는 이 feature를 읽은 뒤 detector train/eval config에서 aggregation한다. 최종 SAFE-LSTM은 validation 성능 기준으로 `horizon_idx_rel=concat-2`, `diff_idx_rel=0.0`을 선택했다.

```text
[T, 4, 16, 1024] -> [T, 2048]
```

이 설계는 SAFE repo의 pi0 diffusion loader와 같은 원칙을 따른다. feature 축은 수집 시점에 보존하고, aggregation choice는 detector training/evaluation config에서 선택한다.

## 실험 설정

Base checkpoint는 GR00T N1.6 RoboCasa PandaOmron checkpoint다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B
```

GR00T N1.6 success-rate 기준선은 project HTTP path가 아니라 upstream GR00T ZMQ evaluation path로 둔다. HTTP `/act` path는 유지하지만, observation/action parity 검증 전까지 SR 지표로 해석하지 않는다.

SAFE rollout collection은 dedicated ZMQ feature server로 수행했다. Feature endpoint는 normal action path를 유지하면서 action과 feature를 함께 저장한다. Official direct policy action과 SAFE feature path action의 동등성은 action key별 비교에서 `max_abs=0.0`으로 확인했다.

Task 후보는 GR00T fork RoboCasa v0.2 eval task와 local robocasa365 v1.0 atomic data가 의미적으로 대응되는 task 중에서 고른다. 최종 6-task set은 다음이다.

| task id | GR00T fork v0.2 task | local robocasa365 v1.0 task | official SR |
|---:|---|---|---:|
| 0 | `CoffeeSetupMug` | `CoffeeSetupMug` | 31.0% |
| 1 | `OpenSingleDoor` | `OpenCabinet` | 81.5% |
| 2 | `PnPCounterToCab` | `PickPlaceCounterToCabinet` | 47.5% |
| 3 | `PnPSinkToCounter` | `PickPlaceSinkToCounter` | 50.0% |
| 4 | `PnPCounterToStove` | `PickPlaceCounterToStove` | 63.2% |
| 5 | `OpenDrawer` | `OpenDrawer` | 81.1% |

SAFE 논문/레포 방식에 맞춰 별도 `train / CP / test` 3-way seen split을 만들지 않았다. 대신 task-level seen/unseen split과 seen-task episode split을 사용했다.

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
| regularization | `lambda_reg=1e-1` |
| aggregation | `horizon_idx_rel=concat-2`, `diff_idx_rel=0.0` |
| selected checkpoint | `seed2` |
| final CP alpha | `0.2` |

현재 데이터에는 episode-level success/failure label만 있고 frame-level failure-onset label이 없으므로 timing plot은 비활성화했다.

## 결과

Aggregation ablation은 `horizon_idx_rel`과 `diff_idx_rel`을 `{0.0, 1.0, mean, concat-2}`에서 비교했다. 선택 규칙은 `val_seen` max-so-far threshold curve에서 balanced accuracy 평균이 가장 높은 설정을 고르는 것이다.

| metric | mean ± std |
|---|---:|
| best aggregation | `horizon_idx_rel=concat-2`, `diff_idx_rel=0.0` |
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
| best hparam | `lr=3e-4`, `lambda_reg=1e-1` |
| `val_seen` bal-acc | `0.950 ± 0.023` |
| `val_seen` T-det | `0.586 ± 0.027` |
| `val_seen` ROC-AUC | `0.958 ± 0.029` |
| `val_unseen` bal-acc | `0.844 ± 0.086` |
| `val_unseen` T-det | `0.710 ± 0.108` |
| `val_unseen` ROC-AUC | `0.833 ± 0.113` |

최종 detector는 hparam sweep의 `seed2` checkpoint로 고정했다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_lstm_final_detector_seen4_unseen2_openDrawer_pnpCab_100ep/model_final.ckpt
```

Fixed threshold baseline은 `val_seen`에서 max-so-far balanced accuracy를 최대화하는 threshold를 고르고 이를 `val_unseen`에 그대로 적용한 결과다.

| metric | value |
|---|---:|
| threshold | `0.9284` |
| `val_seen` bal-acc | `0.9795` |
| `val_seen` TPR/TNR | `0.9762 / 0.9828` |
| `val_seen` mean T-det | `0.5571` |
| `val_unseen` bal-acc | `0.9345` |
| `val_unseen` TPR/TNR | `0.8953 / 0.9737` |
| `val_unseen` acc/F1 | `0.9400 / 0.9277` |
| `val_unseen` mean T-det | `0.6884` |

최종 운영점은 split conformal prediction으로 고정했다. Alpha sweep은 최종 선택된 aggregation/hparam/seed2 checkpoint의 score 위에서 수행했다.

| metric | value |
|---|---:|
| method | `split_cp` |
| alpha | `0.2` |
| eval time | `by final end` |
| calibration label | `neg_success` |
| threshold | `0.8474180102348328` |
| `val_unseen` bal-acc | `0.9533` |
| `val_unseen` TPR/TNR | `0.9767 / 0.9298` |
| `val_unseen` acc/F1 | `0.9500 / 0.9438` |
| `val_unseen` mean T-det | `0.5395` |

가장 실용적인 운영점은 `split_cp`, `alpha=0.2`, `calib_label=neg_success`, `by final end`다. 이 설정은 false alarm을 어느 정도 억제하면서 failure recall을 높인다. 다만 mean T-det가 `0.5395`이므로 trajectory 초반 개입 detector라기보다는 중반 이후 failure monitoring detector에 가깝다.

## 시각화

SAFE feature 시각화는 detector score나 CP threshold가 아니라 detector input feature를 사용한다. GR00T N1.6에서는 다음 feature를 뜻한다.

```text
[T, 4, 16, 1024] -> aggregation -> [T, D']
```

초기 t-SNE는 `mean/mean` aggregation으로 만들었다. 최종 detector aggregation은 `concat-2/0.0`이며, 이 경우 `[T, 2048]` feature가 된다.

t-SNE artifact는 전체 split, `val_unseen`, 그리고 `val_unseen`의 task별 subset에 대해 생성했다.

| scope | rollout | timestep feature |
|---|---:|---:|
| all splits | 600 | 18,428 |
| `val_unseen` | 200 | 5,660 |
| `val_unseen/OpenDrawer` | 100 | 2,041 |
| `val_unseen/PnPCounterToCab` | 100 | 3,619 |

시각화 결과는 조심스럽게 해석해야 한다. Feature space에는 task 방향성이 일부 있지만, 전역적인 success/failure 분리는 모든 task에서 강하게 나타나지 않는다. 최종 aggregation인 `concat-2/0.0`에서도 original 2048-D success/failure silhouette은 거의 0에 가깝다. 따라서 이 결과는 diagnostic evidence이지, proactive intervention capability의 독립적인 증거는 아니다.

## 해석

이번 구현은 SAFE x GR00T N1.6 wiring 및 detector baseline으로는 성립한다.

- GR00T N1.6 rollout feature export가 동작한다.
- SAFE-readable pkl schema가 생성된다.
- SAFE loader가 `[T, 4, 16, 1024]` feature를 읽는다.
- LSTM detector aggregation/hparam sweep, validation, conformal calibration, unseen-task evaluation이 완료된다.
- final checkpoint와 final operating point가 artifact로 고정됐다.
- t-SNE/overlay visualization artifact가 detector input feature에서 생성된다.

현재 detector는 이 작은 재현 범위에서 seen/unseen task failure monitoring에는 사용할 수 있다. 그러나 supervised failure-onset detector는 아니다.

현재 label은 rollout-level이다.

```text
y_\tau \in \{0, 1\}
```

현재 데이터에는 frame-level onset label이 없다.

```text
z_t \in \{0, 1\}
```

주석 처리된 failure onset time도 없다.

```text
t_{onset}
```

따라서 현재 SAFE LSTM이 학습하는 것은 다음 mapping이다.

```text
f_\theta(h_{1:t}) \to y_\tau
```

직접 학습하는 형태는 다음이 아니다.

```text
g_\phi(h_{1:t}) \to z_t
```

또는 다음 형태도 아니다.

```text
g_\phi(h_{1:t}) \to \mathbf{1}[t \ge t_{onset}]
```

따라서 failed rollout의 중후반부에서 failure score가 상승한다면, 올바른 해석은 prefix가 점점 failed rollout과 닮아간다는 것이다. 이것은 모델이 정확한 failure onset을 supervised하게 식별하도록 학습되었다는 증거가 아니다.

## 한계와 다음 단계

가장 큰 한계는 early separability다. CP는 threshold를 calibration할 수 있지만, rollout 초반의 detector score가 success/failure를 분리하지 못하면 early signal을 새로 만들어내지는 못한다.

다음 단계:

1. HTTP-vs-ZMQ observation/action parity가 검증될 때까지 HTTP `/act` SR은 별도 지표로 둔다.
2. Proactive intervention이 목표라면 frame-level onset/intervention label을 수집하거나 정의한다.
3. `--feature-slice all`로 model-level `H=50` feature를 수집해 current `H=16` valid-horizon export와 비교할 수 있다.
4. Taskwise score-trajectory plot을 추가해 CP threshold crossing이 어느 phase에서 발생하는지 더 명확히 본다.

## 관련 파일

- [SAFE wiring runbook](groot_n16_robocasa_wiring.md)
- [Dedicated ZMQ feature server ADR](../adr/0001-dedicated-safe-groot-n16-zmq-server.md)
- `scripts/safe/groot_n16/robocasa/serve/feature_server.py`
- `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`
- `scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py`
- `scripts/safe/groot_n16/robocasa/train/train_lstm_mean_mean.sh`
- `scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh`
- `scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh`
- `scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py`
- `scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py`
- `outputs/eval/robocasa/groot_n16/safe_lstm_final_detector_seen4_unseen2_openDrawer_pnpCab_100ep/README.md`
- `outputs/eval/robocasa/groot_n16/safe_lstm_final_detector_seen4_unseen2_openDrawer_pnpCab_100ep/final_operating_point.json`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/data/groot_n16.py`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/dataset/groot_n16.yaml`
