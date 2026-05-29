# Phase 1: Progress Predictor 구현 및 학습 정리

## 목표

VLA 모델이 실패 루프에 빠졌는지 감지하기 위한 **task 진행률 예측기(ProgressPredictor)** 를 학습.  
외부 TTT(Test-Time Training) 모듈을 통해 VLA 백본 수정 없이, 관측 시퀀스에서 `v(s_t) ≈ t/T` 를 예측.

참고 논문: **VITA (Ziakas & Russo, ICLR 2026)**

---

## 아키텍처

### 모듈 구조 (`src/ttt/`)

```
ProgressPredictor
├── TTTModule           ← SSL self-update backbone
│   ├── P_K, P_V, P_Q  ← outer-loop meta-learned projections
│   ├── TTTInnerModel   ← f_adapt (MLP), inner state = temporal memory
│   └── eta = 0.1      ← fixed inner-loop LR (VITA 기준)
└── ProgressHead        ← 2-layer MLP + Sigmoid → [0, 1]
```

| 파라미터 | 값 | 비고 |
|---|---|---|
| `input_dim` | 1024 | CLIP z_t 차원 |
| `proj_dim` | 64 | TTT projection 차원 |
| `inner_model_type` | mlp | linear 대비 표현력 |
| `eta_base` | 0.1 | VITA 논문 기준 고정값 |
| `learnable_eta` | False | VITA는 η 고정, θ_0만 메타학습 |
| `head_hidden_dim` | 128 | ProgressHead MLP hidden |
| **총 파라미터** | **~210K** | |

### 핵심 설계 결정

**MAML 방식 학습 (functional inner update)**  
기존 `param.data = param - η·grad` 방식은 computational graph를 끊어 outer gradient가 흐르지 않음.  
`ttt_step_functional()`을 구현하여 `create_graph=True`로 new params dict를 반환, graph를 유지함.

```
inner loop: θ_t = θ_{t-1} - η · ∇_θ ℓ_self(z_t; θ_{t-1})
outer loop: min_{θ_0, P_K, P_V, P_Q} L_pred (differentiates through inner loop)
```

**Validation에서 inner update 처리**  
`torch.no_grad()` 내에서 `ttt_step_functional()`의 `torch.autograd.grad()`가 실패.  
→ `meta_forward(create_graph=False)`로 inner update는 수행하되, outer graph만 차단.

**learnable η 제거**  
초기에는 `η(z) = η_base × σ(θ_lr · z)` 형태로 구현했으나,  
배치 입력 시 eta shape `[B, 1]`이 gradient shape `[proj_dim, proj_dim]`과 충돌.  
VITA 논문 확인 결과 η = 0.1 고정이 원래 설계 → `learnable_eta=False`로 변경.

---

## 데이터셋

### BridgeData V2 (LeRobot v0.4.4 포맷)

| | 값 |
|---|---|
| HuggingFace repo | `FedorX8/bridge_v2_lerobot` |
| 총 에피소드 | 50,418 |
| 총 프레임 | 1,801,162 |
| 이미지 키 | `observation.images.primary` (3×480×640, float32) |
| task 키 | `item["task"]` (자연어 string) |

### CLIP 임베딩 (`src/datasets/phase1_dataset.py`)

```
z_t = L2-normalize([φ_v(o_t) ; φ_g(g)]) ∈ R^1024
  φ_v: OpenCLIP ViT-B/32 visual encoder (frozen)
  φ_g: OpenCLIP ViT-B/32 text encoder (frozen)
```

- 초기화 시 전체 에피소드 임베딩 precompute → `<cache>/datasets/bridge_v2_lerobot_clip_embeddings.pt` 캐싱 (775MB)
- 이후 실행부터는 캐시에서 즉시 로드, CLIP 모델 불필요

### Dissimilarity-based Window Sampling (VITA Eq. 5)

에피소드당 sliding window 후보 중 가장 다양한 k개 선택:

```
s(w) = Σ_{v ∈ W_selected} || mean(w) - mean(v) ||²
```

greedy하게 max dissimilarity window를 순차 선택. 실제 window mean embedding 기반.

| 파라미터 | 값 |
|---|---|
| `window_size` | 8 |
| `max_windows_per_episode` (k) | 8 |
| stride | 4 (window_size // 2) |

### Train/Val Split

- **VITA 기준**: train 2,986 에피소드, val 287 에피소드
- **주의**: 순서 기반 split은 BridgeData V2의 날짜/태스크 정렬로 인해 distribution shift 발생 → **랜덤 셔플 후 split** 적용 (`split_seed=42`)

```python
all_indices = list(range(2986 + 287))
random.shuffle(all_indices)  # seed=42
train_indices = all_indices[:2986]
val_indices   = all_indices[2986:]
```

---

## 학습

### 설정 (`scripts/train/phase1_predictor.sh`)

| 하이퍼파라미터 | 값 | 비고 |
|---|---|---|
| `total_steps` | 100,000 | VITA 기준 |
| `batch_size` | 32 | |
| `lr` | 1e-4 | Adam |
| `lambda_self` | 0.5 | SSL loss 가중치 |
| `val_interval` | 1,000 steps | |
| `save_interval` | 10,000 steps | |

### Objective

```
L = L_pred + λ_self · L_ssl

L_pred = MSE(v(s_t), t/T)
L_ssl  = SSL self-supervised loss (TTT inner update용)
```

> mono loss (`λ_mono`)는 초기에 추가했으나 현재 비활성화 상태.

### 학습 환경

- 컨테이너: `lerobot` (Docker)
- PYTHONPATH: `/temporal_vla`
- 임베딩 캐시 경로: `<cache>/datasets/bridge_v2_lerobot_clip_embeddings.pt`

### 관찰된 이슈

| 이슈 | 원인 | 해결 |
|---|---|---|
| `ttt_step_functional` shape mismatch | learnable η가 배치 차원 [B,1] 반환 | `eta.mean()`으로 scalar화 → 이후 `learnable_eta=False`로 근본 해결 |
| validation에서 `no grad_fn` 에러 | `@torch.no_grad()`가 inner autograd 차단 | `meta_forward(create_graph=False)` 사용 |
| val loss 상승 | 순서 기반 split → distribution shift | 랜덤 셔플 split으로 변경 |
| train/pred_loss 불안정 | MAML 2차 gradient의 높은 분산 | 학습 추세는 하향이므로 허용. 향후 cosine LR scheduler 적용 가능 |

---

## 평가 결과

평가 스크립트: `scripts/eval/phase1_predictor.py`  
각 split에서 100 에피소드 랜덤 샘플, 에피소드 전체를 inference 모드로 순차 처리.

### step 50k vs final (100k)

| Split | ckpt | MSE | MAE | Pearson r | Mono rate |
|---|---|---|---|---|---|
| train | step 50k | 0.0428 | 0.1603 | 0.858 | 0.600 |
| train | final | 0.0302 | 0.1308 | **0.880** | 0.601 |
| val | step 50k | 0.0617 | 0.1969 | **0.713** | 0.584 |
| val | final | 0.0675 | 0.1997 | 0.686 | 0.585 |
| unseen | step 50k | 0.0685 | 0.2073 | **0.654** | 0.569 |
| unseen | final | 0.0755 | 0.2105 | 0.595 | 0.559 |

**결론: step 50k 체크포인트가 val/unseen 일반화 성능이 더 좋음 (Early stopping 기준).**  
→ Phase 2에 사용할 체크포인트: `<cache>/checkpoints/phase1/step_0050000.pt`

### 해석

- Pearson r train(0.86) → val(0.71) → unseen(0.65): 오버피팅이 존재하나 unseen에서도 랜덤(0.0) 대비 유의미한 상관관계
- Mono rate ~0.58: 진행률이 약 58% 확률로 단조증가. 랜덤(0.5) 대비 유의미
- 초반부(0~0.3 normalized time) 예측 불안정 → TTT inner update가 누적될수록 안정
- 시각화: `eval_results/phase1_step50k/progress_curves_{train,val,unseen}.png`

---

## 파일 구조

```
src/ttt/
├── ttt_module.py       ← TTTModule (SSL self-update, functional inner update)
├── progress_head.py    ← ProgressHead (2-layer MLP + Sigmoid)
├── predictor.py        ← ProgressPredictor, TTTVLAAdapter
├── losses.py           ← phase1_loss, progress_prediction_loss
└── __init__.py

src/datasets/
└── phase1_dataset.py   ← Phase1EpisodicDataset (CLIP embedding + dissimilarity sampling)

scripts/
├── train_phase1_predictor.py   ← 학습 스크립트 (step 기반, WandB)
├── train_phase1_predictor.sh   ← 실행 스크립트 (VITA 하이퍼파라미터)
└── eval_phase1_predictor.py    ← 평가 스크립트 (MSE/MAE/Pearson r/Mono rate)

<cache>/checkpoints/phase1/
├── step_0010000.pt ~ step_0100000.pt
└── phase1_final.pt     ← 100k step 최종

eval_results/
├── phase1/             ← final 체크포인트 평가 결과
└── phase1_step50k/     ← step 50k 평가 결과 (권장)

data/
└── bridge_v2_lerobot_clip_embeddings.pt   ← CLIP 임베딩 캐시 (775MB)
```

---

## 다음 단계 (Phase 2)

- `step_0050000.pt` 로드 → `TTTVLAAdapter`로 VLAProjector 추가
- TTTModule + ProgressHead freeze, VLAProjector만 학습
- Δv = v(s_{t+1}) - v(s_t) 기반 loss: `-Δv · log P(a_expert | s_t)`
- 주입 모드: `hidden_add` / `logit_shift` / `film` / `token`

### 성능 향상 후보

- [ ] Cosine LR scheduler 추가 (train loss 불안정 완화)
- [ ] window_size 증가 (16 → 더 긴 context)
- [ ] Monotonicity loss 재활성화 및 가중치 튜닝
- [ ] 더 많은 에피소드로 학습 (현재 3,273개 중 2,986 사용)
- [ ] inner model을 linear → mlp로 변경 (이미 적용됨)
- [ ] CLIP 대신 더 강력한 visual backbone 실험
