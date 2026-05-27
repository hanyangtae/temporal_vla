# `src/conceptor` — COAST Conceptor Algebra

COAST(**Co**ntrastive Conceptor **A**ctivation **St**eering)의 conceptor 계산,
Boolean algebra, multiplicative steering 을 NumPy 로 구현한 모듈이다.
성공/실패 활성화 분포로부터 **contrastive conceptor** 를 fitting 하고, 이를
hidden state 에 곱해주는 multiplicative gate 로 변환한다.

본 프로젝트의 목표(실패 루프 탈출)에 맞춰, GR00T N1.6 의 hidden state 에서
"성공 방향은 보존하고 실패 방향만 억제하는" steering 행렬을 만드는 데 쓴다.

---

## 1. 수식 정리

활성화 행렬 $X \in \mathbb{R}^{N \times d}$ ($N$ = 샘플 수, $d$ = hidden dim) 가 주어졌을 때:

**1. Mean-center**

$$\tilde{X} = X - \frac{1}{N}\sum_{i=1}^{N} X_i$$

**2. Correlation matrix**

$$R = \frac{1}{N}\, \tilde{X}^\top \tilde{X} \in \mathbb{R}^{d \times d}$$

**3. Conceptor** (aperture $\alpha$)

$$C = R\,\bigl(R + \alpha^{-2} I\bigr)^{-1}$$

**4. Eigenvalue 관계** — $R$ 의 고유값을 $\sigma_i$ 라 하면 $C$ 의 고유값은

$$\lambda_i = \frac{\sigma_i}{\sigma_i + \alpha^{-2}} \in [0, 1)$$

$\sigma_i \ge 0$, $\alpha^{-2} > 0$ 이므로 $C$ 는 항상 PSD 이고 고유값이 $[0,1)$ 에 있다.

**5. Boolean algebra** (Jaeger 2014)

$$\lnot C = I - C$$

$$A \wedge B = \bigl(A^{+} + B^{+} - I\bigr)^{+} \quad (\,^{+}\text{ = Moore-Penrose pseudoinverse})$$

$$A \vee B = (R_A + R_B)\,\bigl(R_A + R_B + \alpha^{-2} I\bigr)^{-1}$$

OR 는 두 분포의 covariance 를 합쳐서 다시 conceptor 로 fitting 하는 것과 동치다.

**6. Contrastive conceptor** (COAST Eq. 4)

$$C_{\text{steer}} = C_{\text{success}} \wedge \lnot C_{\text{failure}} = \bigl(C_s^{+} + (I - C_f)^{+} - I\bigr)^{+}$$

성공 부분공간 중 실패 부분공간과 겹치지 않는 방향만 남긴다.

**Steering** (Sec. 3.2) — multiplicative gate $\beta \in [0,1]$:

$$M = (1-\beta) I + \beta\, C_{\text{steer}}, \qquad h' = h\,M^\top$$

$\beta=0$ 이면 identity(steering 없음), $\beta=1$ 이면 full conceptor.

---

## 2. API 사용 예시 (toy)

```python
import numpy as np
from src.conceptor import (
    compute_conceptor, contrastive_conceptor,
    build_steering_matrix, apply_steering,
    conceptor_quota, conceptor_overlap, eigenvalue_spectrum,
)

rng = np.random.default_rng(0)
d = 1024
X_success = rng.standard_normal((40, d))   # 성공 episode 활성화 (N_s, d)
X_failure = rng.standard_normal((30, d))   # 실패 episode 활성화 (N_f, d)

# 1) contrastive conceptor fitting
C_s, C_f, C_steer = contrastive_conceptor(X_success, X_failure, alpha=1.0)

# 2) 진단
print("quota(success)", conceptor_quota(C_s))
print("overlap(s, f) ", conceptor_overlap(C_s, C_f))   # 성공/실패 부분공간 겹침
print("top eigvals   ", eigenvalue_spectrum(C_steer)[:5])

# 3) steering 행렬 생성 후 hidden state 에 적용
M = build_steering_matrix(C_steer, beta=0.3)
h = rng.standard_normal((2, 16, d))        # (B, S, d)
h_steered = apply_steering(h, M)           # 같은 shape
```

---

## 3. GR00T N1.6 통합 예시

`scripts/safe/groot_n16/robocasa/safe_feature_vectors.py` 의 pooling 규약과
동일하게, 각 timestep 의 hidden state `[K, H, D]` (diff steps × horizon tokens × dim)
를 diff·horizon 축에 대해 평균내어 `z_mean ∈ ℝ^D` 를 얻는다.

```python
import numpy as np
from src.conceptor import contrastive_conceptor, build_steering_matrix, as_float32

def episode_z_means(record):
    """per-pkl record["hidden_states"]: list[T] of [K, H, D] -> (T, D) z_mean."""
    # diff(axis=0)·horizon(axis=1) 축 평균 -> [D]
    return np.stack([np.asarray(h, np.float32).mean(axis=(0, 1)) for h in record["hidden_states"]])

# 성공/실패 episode 들에서 z_mean 누적 -> (N_s, D), (N_f, D)
X_success = np.concatenate([episode_z_means(r) for r in success_records], axis=0)
X_failure = np.concatenate([episode_z_means(r) for r in failure_records], axis=0)

C_s, C_f, C_steer = contrastive_conceptor(X_success, X_failure, alpha=1.0)
M = build_steering_matrix(C_steer, beta=0.3)

np.save("steering_matrix.npy", as_float32(M))   # float32 로 저장/전송
```

추론 시 GR00T backbone 의 해당 레이어 hidden state `h` 에 `apply_steering(h, M)`
를 hook 으로 끼워 넣어, 실패 방향을 억제한 활성화를 다음 레이어로 흘려보낸다.

---

## 4. COAST paper section 매핑

| 함수 | 수식 | COAST 위치 |
|---|---|---|
| `compute_correlation` | $R = \tilde{X}^\top\tilde{X}/N$ | Sec. 3.1, App. A.9.1 |
| `compute_conceptor` | $C = R(R+\alpha^{-2}I)^{-1}$ | Sec. 3.1, App. A.9.1 |
| `not_conceptor` | $\lnot C = I - C$ | App. A.9.2 (Jaeger 2014) |
| `and_conceptor` | $(A^{+}+B^{+}-I)^{+}$ | App. A.9.2 |
| `or_conceptor` | $(R_A+R_B)(R_A+R_B+\alpha^{-2}I)^{-1}$ | App. A.9.2 |
| `contrastive_conceptor` | $C_s \wedge \lnot C_f$ | Eq. 4 |
| `build_steering_matrix` / `apply_steering` | $M=(1-\beta)I+\beta C$, $h'=hM^\top$ | Sec. 3.2 |
| `conceptor_quota` | $q(C)=\tfrac{1}{d}\operatorname{tr} C$ | App. A.10.2 Stage 1 (layer 선택) |
| `conceptor_overlap` | $\tfrac{\operatorname{tr}(C_aC_b)}{\sqrt{\operatorname{tr}C_a^2\operatorname{tr}C_b^2}}$ | App. A.10.2 Stage 2 (aperture 선택), Sec. 4.4 |
| `eigenvalue_spectrum` | $\operatorname{sort}_\downarrow \lambda_i(C)$ | Fig. 3A |
| `failure_containment` | $\operatorname{tr}(C_{\text{src}}C_{\text{tgt}})/\operatorname{tr}C_{\text{src}}^2$ | Sec. 4.4 (cross-task transfer) |

---

## 5. Numerical precision 주의사항 (App. A.9.4)

- **float64 계산**: 모든 correlation / matrix inverse 는 float64 로 수행한다.
  입력이 float32 (예: GR00T hidden state) 여도 내부에서 float64 로 승격한다.
- **float32 저장**: 결과 행렬을 저장/전송할 때는 `as_float32(C)` 로 다운캐스트한다.
  계산은 float64, 저장은 float32 로 분리한다.
- **`pinv` 사용 이유**: COAST 실제 세팅은 $N \ll d$ (예: $N=15$, $d=1024$) 라
  $R$ 이 rank-deficient → singular 하다. `AND` 의 역행렬은 항상
  `np.linalg.pinv` (Moore-Penrose) 로 계산해 singular case 에서도 안정적이다.
- **conceptor 본체 inverse**: $C = R(R+\alpha^{-2}I)^{-1}$ 의 $R+\alpha^{-2}I$ 는
  항상 positive-definite 라 정칙이다. $R$ 과 $R+\alpha^{-2}I$ 가 commute 하는 성질을
  이용해 `np.linalg.solve` 로 대칭 결과를 직접 구한다.
- 부동소수점 오차로 인한 비대칭을 막기 위해 모든 출력은 $\tfrac{1}{2}(C+C^\top)$ 로
  대칭화한다.

---

## 6. Hyperparameter 가이드

### Aperture $\alpha$ (regularization)

- $\lambda_i = \sigma_i/(\sigma_i + \alpha^{-2})$ 이므로 **$\alpha$ 가 클수록** 정규화가
  약해져 더 많은 분산 방향을 보존한다 → quota $q(C)$ 증가.
- COAST 의 quota / overlap heuristic (App. A.10.2):
  - **Stage 1 (layer 선택)**: 여러 레이어에서 conceptor 를 fitting 한 뒤
    `conceptor_quota` 가 적당히 높으면서(분포를 충분히 포착) 포화되지 않은
    레이어를 고른다. quota 가 1 에 너무 가까우면 거의 identity → steering 효과 없음.
  - **Stage 2 (aperture 선택)**: $\alpha$ 를 sweep 하며 성공/실패 conceptor 의
    `conceptor_overlap(C_s, C_f)` 가 **최소**가 되는 지점(두 부분공간이 가장 잘
    분리되는 aperture)을 고른다. 보통 $\alpha \in [0.1, 10]$ 로그스케일 sweep.

### Steering 강도 $\beta$

- $\beta=0$ → 원본 유지, $\beta=1$ → conceptor 로 완전 projection.
- 작은 값($0.1\!\sim\!0.5$)부터 키워가며 success rate 가 오르는 지점을 찾는다.
  너무 크면 hidden state 분포가 망가져 backbone 출력이 붕괴할 수 있다.
- 실패 루프 탈출 용도: 실패가 감지된 step 에서만 $\beta>0$ 로 켜는 conditional
  steering 도 가능하다.

### Cross-task 전이 (Sec. 4.4)

- `failure_containment(C_src, C_tgt)` 가 1 에 가까우면 source task 에서 만든
  실패 conceptor 가 target task 실패도 포함 → steering 행렬 재사용 가능.
