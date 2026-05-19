# SAFE x GR00T N1.5 RoboCasa 정리

작성일: 2026-05-19

## 1. SAFE가 풀려는 문제

SAFE는 이미 학습된 VLA policy의 내부 feature를 이용해, 실행 중 실패 가능성을 감지하는 runtime failure detector다. 목표는 policy 자체를 다시 학습하지 않고도 rollout 도중 failure score를 계산해 stop, retry, human takeover 같은 개입을 가능하게 하는 것이다.

SAFE가 피하려는 방식은 task별 detector다. generalist VLA는 여러 instruction과 장면을 처리하므로, task마다 별도 detector를 학습하면 확장성이 낮다. SAFE는 여러 seen task의 success/failure rollout으로 하나의 detector를 학습하고, 그 detector가 새로운 rollout에서도 failure-like state를 감지할 수 있는지를 본다.

핵심 가정은 VLA hidden feature 안에 task 진행 상태와 실패 정보가 이미 일부 들어 있다는 것이다. 논문은 successful rollout과 failed rollout의 VLA feature가 t-SNE 상에서 분리되고, failed rollout은 task가 달라도 공통적인 failure zone으로 이동하는 경향을 보인다고 해석한다.

중요한 제한도 있다. SAFE 기본 설정은 frame-level failure onset label을 요구하지 않는다. 학습 label은 rollout-level success/failure다. 따라서 SAFE detector는 "정확히 몇 번째 frame부터 실패가 시작됐는가"를 supervised 방식으로 배우는 모델이 아니다.

## 2. Detector와 Score의 의미

매 timestep의 VLA hidden feature를 `h_t`라고 두면, SAFE detector는 지금까지의 feature sequence를 입력으로 받아 failure score `s_t`를 낸다.

```text
h_1, ..., h_t -> f_theta -> s_t
```

LSTM detector의 구조는 다음과 같다.

```text
features: (B, T, D)
-> LSTM
-> Linear(hidden_dim, 1)
-> sigmoid
-> scores: (B, T)
```

SAFE 원본 코드에서 `cumsum=false`일 때 loss는 rollout-level label을 모든 timestep에 broadcast한다.

```text
target = 1 - success_label
loss = BCE(score_t, target) for all valid t
```

즉 failed rollout 안의 초반 frame도 학습상 failure label을 받는다. 그래서 score trajectory가 중간부터 올라가는 현상은 detector가 representation 상에서 failure-like region으로 이동한다고 해석할 수는 있지만, ground-truth failure onset을 맞힌다고 해석하면 안 된다.

## 3. Early Metric과 CP의 의미

SAFE 코드에서 early metric은 실제 실패 발생 시점 기준이 아니다.

```text
early score = max(score[:task_min_step])
end score   = max(score[:rollout_length])
```

`task_min_step`은 같은 task rollout들 중 최소 길이에서 유도되는 cutoff다. RoboCasa가 실패를 인식한 시점이나 사람이 정의한 intervention point가 아니다.

논문식 thresholding은 conformal prediction(CP)을 사용한다. successful calibration score curve들로 time-varying one-sided band를 만들고, test rollout의 score가 그 band를 벗어나면 failure alert를 낸다. 이때 `alpha`는 false positive와 detection time 사이의 trade-off를 조절한다.

논문에 충실한 평가를 하려면 역할이 다른 세 데이터가 필요하다.

```text
train/val_seen: detector 학습 및 내부 validation
CP: threshold 또는 functional CP band 보정
test: 최종 성능 평가
```

단, SAFE `train.py`는 별도 `val` 폴더를 입력으로 받지 않는다. `dataset.data_path`로 들어온 seen rollout들을 내부에서 `seen_train_ratio`에 따라 `train`과 `val_seen`으로 나눈다.

## 4. 현재 실험 범위

이번 실험은 SAFE 논문의 full unseen-task protocol이 아니다. 목표는 GR00T N1.5 tuned policy에 대해, 5개 seen RoboCasa task에서 failure monitoring이 가능한지 작게 재현하는 것이다.

Policy checkpoint 경로:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/train/260513094637-subset100
```

대상 task:

```text
0 CloseFridge
1 CloseToasterOvenDoor
2 PickPlaceSinkToCounter
3 OpenCabinet
4 SlideDishwasherRack
```

Feature server 설정:

```text
groot_n15 container
scripts/safe/groot_n15/robocasa/serve/feature_server.py
port 5556
endpoint get_action_with_features
feature_pool masked_mean
```

Detector 설정:

```text
model = LSTM
epochs = 1000
batch_size = 64
hidden_dim = 256
lr = 3e-4
lambda_reg = 1.0
cumsum = false
dropout = 0
n_layers = 1
unseen_task_ratio = 0.0
```

## 5. 현재 표준 데이터 구조

활성 출력 루트:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n15
```

현재 source:

```text
rollouts_seen5_100ep_per_task_subset100
```

이 source의 목표 크기는 task당 100 episode, 총 500 episode다. episode 0-59는 기존 60 episode/task source를 가리키는 symlink이고, episode 60-99는 새로 수집 중이다.

수집 완료 후 만들 split:

```text
split_seen5_trainval75_cp15_test10_subset100/
  trainval/   # 75/task, SAFE train.py에 입력
  cp/         # 15/task, conformal calibration
  test/       # 10/task, 최종 외부 평가
```

학습 입력은 다음과 같이 둔다.

```text
DATA_ROOT=.../split_seen5_trainval75_cp15_test10_subset100/trainval
seen_train_ratio=0.8
```

`trainval`에 75 episode/task가 들어 있으므로, SAFE 내부 split은 대략 다음처럼 된다.

```text
train    60/task
val_seen 15/task
```

별도의 물리적 `val` 폴더는 만들지 않는다. 현재 SAFE loader/split logic에서는 `val` 폴더를 만들어도 training script가 자동으로 읽지 않는다.

## 6. 사용 스크립트

수집:

```text
scripts/safe/groot_n15/robocasa/collect/run_seen5_5task_host.sh
scripts/safe/groot_n15/robocasa/collect/run_seen5_5task_container.sh
```

장시간 수집에는 container-side script를 쓴다. 이 스크립트는 `robocasa` 컨테이너 내부에서 돌고, 이미 수집된 episode는 skip하며, `rollouts_seen5_100ep_per_task_subset100`에 결과를 쓴다.

Split 생성:

```text
scripts/safe/groot_n15/robocasa/data/prepare_seen5_split.py
```

100 episode/task source에서 `trainval/cp/test` symlink split을 만든다.

학습:

```text
scripts/safe/groot_n15/robocasa/train/train_lstm_detector.sh
```

학습 wrapper는 seed별 output 폴더를 분리한다.

```text
safe_lstm_detector_seen5_trainval75_cp15_test10_subset100_seed0
safe_lstm_detector_seen5_trainval75_cp15_test10_subset100_seed1
safe_lstm_detector_seen5_trainval75_cp15_test10_subset100_seed2
```

이렇게 해야 SAFE 기본 동작 때문에 여러 seed가 같은 `model_final.ckpt`를 덮어쓰는 문제를 피할 수 있다.

외부 평가:

```text
scripts/safe/groot_n15/robocasa/eval/eval_checkpoint.py
scripts/safe/groot_n15/robocasa/eval/analyze_failure_timing.py
scripts/safe/groot_n15/robocasa/eval/visualize_tsne.py
```

## 7. 연결 검증 결과

현재 목표를 성능 검증이 아니라 연결 검증으로 두면, 확인해야 할 경로는 다음이다.

```text
GR00T N1.5 tuned server
-> get_action_with_features
-> RoboCasa collector
-> SAFE-format pkl
-> SAFE groot_n15 loader
```

이 경로는 통과했다.

### 7.1 Server

`groot_n15` 컨테이너 안에서 tuned checkpoint를 바라보는 SAFE feature server가 떠 있다.

```text
script = /temporal_vla/scripts/safe/groot_n15/robocasa/serve/feature_server.py
model  = /temporal_vla/outputs/train/260513094637-subset100
port   = 5556
data_config = robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig
embodiment_tag = new_embodiment
```

server는 일반 action endpoint와 별도로 `get_action_with_features` endpoint를 제공한다. 이 endpoint는 policy action과 함께 action head 이전 backbone feature를 반환한다.

반환 feature의 종류:

```text
feature_kind = groot_n15_backbone_features_pre_action_head
```

### 7.2 Collector

RoboCasa rollout collector는 `get_action_with_features` endpoint를 호출한다.

```text
script = scripts/safe/groot_n15/robocasa/collect/collect_rollout.py
endpoint = get_action_with_features
feature_pool = masked_mean
```

collector는 매 policy step마다 다음 정보를 저장한다.

```text
hidden_state
attention_mask
action
action_vector
groot_action_vector
```

episode가 끝나면 SAFE loader가 읽을 수 있는 pkl triplet을 만든다.

```text
task<ID>--ep<IDX>--succ<0|1>.pkl
task<ID>--ep<IDX>--succ<0|1>.csv
task<ID>--ep<IDX>--succ<0|1>.mp4
```

### 7.3 실제 pkl schema 확인

새로 수집된 rollout과 기존 symlink rollout 모두 같은 schema를 갖는다.

예시:

```text
path = rollouts_seen5_100ep_per_task_subset100/CloseFridge/task0--ep60--succ1.pkl
task_id = 0
episode_idx = 60
episode_success = 1
feature_kind = groot_n15_backbone_features_pre_action_head
hidden_states = list length 16, each tensor shape (2048,)
action_vectors = (16, 12)
attention_masks = list length 16
```

기존 60 episode/task source에서 연결된 symlink rollout도 같은 형식이다.

```text
path = rollouts_seen5_100ep_per_task_subset100/CloseFridge/task0--ep000--succ0.pkl
task_id = 0
episode_idx = 0
episode_success = 0
feature_kind = groot_n15_backbone_features_pre_action_head
hidden_states = list length 45, each tensor shape (2048,)
action_vectors = (45, 12)
attention_masks = list length 45
```

따라서 old source와 newly collected source가 SAFE 관점에서 같은 schema로 연결된다.

### 7.4 SAFE loader smoke 확인

SAFE repo의 `failure_prob.data.groot_n15.load_rollouts()`로 pkl 하나를 직접 로드했다.

확인 결과:

```text
loaded = 1
task_id = 0
episode_idx = 60
episode_success = 1
hidden_states = (16, 2048), torch.float32
action_vectors = (16, 12)
task_min_step = 16
cfg.dataset.dim_features = 2048
cfg.dataset.dim_action = 12
```

즉 SAFE loader는 GR00T collector가 만든 pkl을 정상적인 `Rollout` 객체로 변환한다.

### 7.5 연결 검증 기준

이번 단계의 통과 기준은 성능이 아니라 연결이다.

통과 기준:

```text
1. tuned GR00T N1.5 server가 feature endpoint를 제공한다.
2. RoboCasa collector가 그 endpoint를 호출한다.
3. collector가 SAFE loader가 읽을 수 있는 pkl schema를 저장한다.
4. SAFE groot_n15 loader가 hidden/action/label/task 정보를 손실 없이 읽는다.
5. train script가 이후 생성될 trainval split을 바라본다.
```

현재 1-4는 실제로 확인했다. 5는 script default 기준으로 준비되어 있다.

```text
DATA_ROOT = .../split_seen5_trainval75_cp15_test10_subset100/trainval
dataset = groot_n15
feature_pool = masked_mean
seen_train_ratio = 0.8
```

따라서 연결 관점의 결론은 다음이다.

```text
GR00T N1.5 RoboCasa rollout feature가 SAFE detector 학습 입력까지 도달하는 경로는 정상이다.
```

아직 검증하지 않은 것은 성능이다. 100 episode/task 수집이 완료된 뒤 split 생성, detector 학습, CP/test 평가를 해야 성능을 말할 수 있다.

## 8. 60 Episode/Task Baseline 결과

100 episode/task로 확장하기 전 baseline은 다음 구조를 사용했다.

```text
split_seen5_train36_cp12_test12_subset100
safe_lstm_detector_seen5_train36_cp12_test12_subset100
```

당시 SAFE script는 180개 train rollout을 다시 `seen_train_ratio=0.99`로 내부 split했다. 그 결과 validation은 전체 5 rollout에 불과했으므로 신뢰하기 어렵다.

외부 평가는 CP 60 rollout, test 60 rollout으로 수행했다.

임계값 없이 본 분리도:

```text
max_to_task_min ROC-AUC = 0.6328
max_to_task_min AP      = 0.6505
max_to_end ROC-AUC      = 0.8906
max_to_end AP           = 0.9231
```

`max_to_end` 기준 best fixed threshold:

```text
balanced accuracy = 0.8304
TPR               = 0.8750
TNR               = 0.7857
avg_det_time      = 0.5146
```

Split CP, `alpha=0.2`, end score:

```text
balanced accuracy = 0.7991
TPR               = 0.8125
TNR               = 0.7857
avg_det_time      = 0.5840
```

Functional CP, `alpha=0.2`, earliest stop:

```text
balanced accuracy = 0.4888
TPR               = 0.15625
TNR               = 0.8214
avg_det_time      = 0.8861
```

해석:

- rollout 중후반 또는 끝 기준 failure separability는 좋다.
- early separability는 약하다.
- 현재 detector는 proactive early intervention보다는 seen-task late failure monitoring 쪽으로 해석하는 것이 더 타당하다.

## 9. t-SNE 해석 기준

t-SNE 시각화의 한 점은 하나의 frame이다. 다만 label은 여전히 episode-level label이다. 각 점에는 최소한 다음 정보가 있어야 한다.

```text
task_id
episode_idx
frame_idx
success/failure episode label
SAFE score
```

유용한 질문은 failed episode가 초반에는 success-like region에 있다가 후반에 failure-like region으로 이동하는지다. 일부 trajectory에서는 그런 패턴이 보였지만, 이것이 supervised onset detection을 의미하지는 않는다.

정확한 해석:

```text
일부 failed trajectory는 score 또는 hidden-state 상에서
success-like region에서 failure-like region으로 이동하는 패턴을 보인다.
```

과한 해석:

```text
모델이 실제 frame-level failure onset을 학습했다.
```

## 10. 현재 다음 작업

먼저 expanded source 수집을 끝낸다.

```text
rollouts_seen5_100ep_per_task_subset100
```

진행 확인:

```bash
find outputs/eval/robocasa/groot_n15/rollouts_seen5_100ep_per_task_subset100 \
  -maxdepth 2 \( -type f -o -type l \) -name '*.pkl' | wc -l

tail -f outputs/eval/robocasa/groot_n15/logs/collect_seen5_ep60_99_container_20260519.log
```

count가 500에 도달하면 split과 학습을 진행한다.

```bash
python scripts/safe/groot_n15/robocasa/data/prepare_seen5_split.py
conda run -n vla-safe bash scripts/safe/groot_n15/robocasa/train/train_lstm_detector.sh
```

그 다음 seed별 checkpoint를 같은 `cp`와 `test` split에서 평가한다. seed 선택은 내부 `val_seen` 결과와 외부 CP/test metric을 함께 보고 결정한다.

## 11. 실무적 결론

연결에 대해서는 다음과 같이 결론낼 수 있다.

```text
GR00T N1.5 tuned server에서 나온 backbone feature는
RoboCasa collector를 거쳐 SAFE loader까지 정상적으로 연결되어 있다.
```

성능에 대해서 현재 근거로 말할 수 있는 것은 다음이다.

```text
5개 seen RoboCasa task에서 GR00T hidden state는 failure 정보를 일부 담고 있다.
특히 failed rollout의 중후반부에서는 failure separability가 꽤 강하다.
```

아직 말하기 어려운 것은 다음이다.

```text
detector가 proactive intervention에 충분할 만큼 failure를 일찍 예측한다.
```

이 강한 주장을 하려면 100 episode/task 확장 실험에서 early separability와 CP-calibrated early detection이 test split에서 개선되어야 한다.
