# 02 — INSIGHT의 "VLA가 progress를 예측한다" 메커니즘과 GR00T 재현성 분석

> 분석/설명 문서 (학습 실험 아님). INSIGHT(arXiv:2606.24884, *InSight: Self-Guided
> Skill Acquisition via Steerable VLAs*)의 progress channel을 코드 수준에서 정확히
> 기술하고, 우리 GR00T latent-steering 스택에서 재현 가능한지 판정한다. 메인 method
> ([`../steering/RESEARCH_DIRECTION.md`](../steering/RESEARCH_DIRECTION.md))가
> 요구하는 **online phase/progress 신호** 후보로서의 적합성에 초점.

---

## 1. 무엇 — INSIGHT의 progress channel

### 1.1 정의: per-primitive 정규화 timestep

INSIGHT은 action space에 **학습된 progress channel ∈ [0, 1)** 을 한 차원 추가한다. 라벨은
**각 primitive segment 내부의 정규화 timestep**이다 — task 전체가 아니라 primitive 단위.

논문 Appendix A (paper p.13, `steerable_vlas.txt` 줄 833–836):

> "...outputs end-effector deltas, an absolute gripper command, and a learned progress
> channel ∈[0, 1) supervised with the normalized timestep within each primitive segment."

라벨 생성 코드 — `arange(n)/max(n-1,1)` 를 8번째 action 차원으로 append:

`training/preprocess/filter_normalize_twist.py:190-202`

```python
n = len(df)
progress = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
old_actions = np.stack(df["actions"].values).astype(np.float32)  # (n, 7)
...
progress_col = progress.reshape(-1, 1)
new_actions = np.concatenate([old_actions, progress_col], axis=1)  # (n, 8)
df["actions"] = list(new_actions)
```

`pour` 전처리도 동일 (`filter_normalize_pour.py:158-161`). 여기서 결정적인 것은 `n =
len(df)` 가 **(필터·정규화된) 단일 primitive 에피소드의 길이**라는 점이다 — 각 segment가
독립 training 에피소드이므로 (Appendix A: "each segmented primitive as a separate
training episode"), `arange(n)/(n-1)` 은 곧 primitive 내부 진행률이다.

### 1.2 per-primitive 리셋 (단조 아님)

따라서 14-primitive long-horizon rollout에서 progress 값은 **각 primitive 시작마다 0으로
리셋**되며, 시퀀스 전체에 걸쳐 단조증가하지 않는다. Figure 5 caption (paper p.7, 줄 430–433):

> "The step/progress value shown in each panel is the learned per-primitive progress
> channel, which resets at the start of each primitive rather than increasing
> monotonically across the full 14-primitive sequence."

이 "리셋" 성질이 1.4의 종료/전이 신호로 직결된다 — task-level progress라면 1.0 근처 한 번만
의미를 갖지만, primitive-level이면 매 primitive 종료마다 1.0 근처를 친다.

### 1.3 어디에 구현 — action head 8번째 dim (π0.5)

- 백본: **π0.5** [Black et al. 2025] (LoRA fine-tune, Gemma-2B + Gemma-300M action
  expert, Appendix A 줄 828–831). INSIGHT은 "VLA-agnostic"이라 명시하지만 본 실험은 π0.5.
- 출력 8차원 = `[pose(6), gripper(1), progress(1)]`. policy output transform이 명시적으로
  앞 8 dim을 slice한다 — `src/openpi/policies/xarm_policy.py:73-78`:

```python
class XarmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return 8 dims: [pose(6), gripper(1), progress(1)]. Older clients
        # that slice action[:6] still work — they get the pose unchanged.
        return {"actions": np.asarray(data["actions"][:, :8])}
```

핵심: progress는 **별도 head/모델이 아니라 action expert가 매 step 함께 내놓는 그냥 또 하나의
action 차원**이다. 추가 forward·추가 네트워크가 없다 (2장의 설계 의도와 직결).

### 1.4 종료 조건 3종

primitive는 다음 셋 중 하나가 fire하면 종료된다 (Appendix A 줄 833–836):

> "A primitive terminates when the progress channel exceeds a threshold (typically
> 0.95), when end-effector motion falls below an auto-advance threshold, or (for
> out-of-distribution 'move to' primitives) when a VLM completion check fires."

1. **progress > 0.95** (학습된 신호) — `real/xarm_flywheel/args.py:136`
   `progress_threshold: float = 0.95`, 단일 프레임 스파이크를 막으려고 연속 3 step 요구
   (`progress_consecutive_required: int = 3`, args.py:138). 종료 판정 코드는
   `runner.py:_check_progress_done` (719–743): `progress = float(action[7]); ...
   progress >= args.progress_threshold`.
2. **EE-motion auto-advance** — end-effector 이동량이 임계 미만이면 자동 전이
   (`runner.py:_check_auto_advance`, 765–).
3. **VLM completion check** — OOD "move to" primitive 폴백. drawer-closing 실험에서 OOD
   초기상태(이미 열린 서랍)일 때 학습된 progress가 신뢰 불가라 VLM이 종료를 trigger
   (paper §4.2 줄 390–397).

즉 progress channel은 **in-distribution에서 무료·빠른 1차 종료 신호**이고, OOD에서는 비싼
VLM check가 폴백한다.

---

## 2. 왜 action head에 굽나 — 설계 의도

INSIGHT이 별도 progress 모델 대신 **action head에 progress를 같이 굽는** 이유:

- **primitive 종료/전이 신호 → composition 런타임 자동화.** INSIGHT의 본질은 primitive를
  순차 chaining해 long-horizon task를 구성하는 것(Figure 5, 14-primitive twist-then-pour).
  각 primitive가 "내가 끝났다"를 스스로 알려야 다음 primitive로 closed-loop 전이가 가능하다.
  progress > 0.95가 그 전이 트리거다.
- **별도 모델/추가 forward 없음.** progress가 action expert의 한 출력 차원이라, 정책의
  primitive-conditioning(언어 라벨 prompt)과 visual grounding을 **그대로 재사용**한다. 매
  step action을 뽑을 때 progress가 공짜로 따라 나온다 — 추론 비용 증가 0.
- **action과 시간 정렬 (closed-loop).** progress는 action과 같은 tick에 같은 head에서
  나오므로, "이 action을 낸 시점의 진행률"이 정의상 정합한다. 외부 predictor였다면 관측→예측
  지연·표현 불일치를 따로 관리해야 한다.
- **supervision이 무료.** 라벨이 `t/(T-1)` (1.1) — 데모를 primitive로 자른 순간 길이 `T`가
  결정되므로 추가 annotation 없이 regression target이 즉시 생긴다.
- **in-distribution 빠른 무료 신호 / OOD VLM 폴백.** 1.4의 3종 종료가 비용-신뢰도 사다리를
  이룬다: 학습 progress(싸고 ID에서 정확) → EE-motion(기하적) → VLM(비싸지만 OOD robust).

정리하면 progress channel은 INSIGHT에서 **"steerability를 closed-loop composition으로
바꾸는 접착제"**다 — 표현 분석용 신호가 아니라 런타임 제어 신호.

---

## 3. GR00T 재현성 분석

우리 목표는 INSIGHT의 composition 자동화가 아니라, 메인 method가 요구하는 **online
phase/progress 신호**의 공급원으로 progress 예측을 쓸 수 있는지다. 세 측면으로 나눈다.

### (a) 쉬운 부분 — 라벨·head는 자명, 우리 git history에 이미 존재

라벨 `y_t = t/T` 와 progress head(작은 MLP + sigmoid)는 구현이 trivial하다. 실제로 우리는
이미 만들었다가 제거했다 — `src/ttt/progress_head.py` (3c8b73c에서 삭제, git history에 보존).
복원 시 기준선이 되는 실제 코드:

```python
# src/ttt/progress_head.py (git 3c8b73c^)
class ProgressHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),     # → [0, 1]
        )
```

label과 입력 정의는 `src/ttt/predictor.py` (3c8b73c^)와
[`../ttt/README.md`](../ttt/README.md)에 기록되어 있다:

> "label `y_t = t/T`. ... 입력 = Eagle pre-LLM 임베딩(dim 2048 = DiT KV dim 고정)."

`predictor.py`의 `ProgressPredictor.meta_forward` 주석:
`pred_loss = F.mse_loss(result["progress_seq"][pred_mask], targets[pred_mask])` — 즉
progress regression 학습 루프까지 이미 구현돼 있었다. **head·label·학습 루프는 재현
난점이 아니다.**

(주: 우리 옛 구현은 INSIGHT의 단순 MLP head에 더해 TTT inner-loop self-update를 얹은 변형
이다. INSIGHT 재현 관점에서는 TTT는 불필요하고 `ProgressHead` 한 조각이면 충분하다.)

### (b) 차이/난점 — INSIGHT은 action-head baked-in 재학습, 우리는 백본 재학습 없음

| | INSIGHT | 우리 GR00T |
|---|---|---|
| progress 위치 | π0.5 action expert의 8번째 **출력 dim** (baked-in) | action_head는 **DiT flow-matching** 구조 — velocity field를 예측, "추가 출력 dim"이 같은 의미가 아님 |
| 학습 | LoRA로 **action head 재학습** (progress dim 포함) | 메인 원칙 = **백본 재학습 없음** (latent steering) |

INSIGHT 방식(action head에 dim 추가)을 그대로 따르려면 GR00T action_head를 재학습해야
하는데, 이는 우리 프로젝트 원칙과 충돌한다. flow-matching head는 임의 timestep의 noisy
action을 denoise하는 구조라, "scalar progress를 한 출력 채널로 붙이는" π0.5식 회귀 head와
배선 의미가 다르다 (단순 차원 추가로 끝나지 않음).

→ **결론: GR00T에는 backbone을 건드리지 않는 외부(external) progress predictor가 적합.**
이는 우리 옛 `ProgressPredictor`(frozen VLA feature → 작은 head)와 정확히 같은 형태다.
입력 후보는 doc 14의 pathway tap 지점 — VL `action_head.vlln` (D=2048) 또는 DiT
`transformer_blocks[i]` 중간 활성화. (옛 구현은 Eagle pre-LLM 2048-dim을 썼다.)

### (b') 실측 — GR00T robocasa 학습데이터엔 이미 task-level progress 컬럼이 있다

조사 중 발견: GR00T 학습에 쓰인 robocasa v1.0 pretrain LeRobot 데이터셋
(`~/.cache/temporal_vla/datasets/robocasa/v1.0/pretrain/atomic/<Task>/.../lerobot/`)의
parquet 에 **`progress` feature [1] 이 이미 들어있다**. 값은 episode 전체에 걸쳐 **단조 0→1**
(head=[0,0,0.01...], tail=[...0.99,1,1]) — 즉 **task-level `t/T`** 이고, INSIGHT 의
**per-primitive 리셋과는 다른** 버전이다.

함의 두 가지:
1. GR00T 계열은 이미 이 task-level progress 를 supervised 신호로 가질 수 있다(데이터에 존재). 따라서
   "progress 를 회귀하는 것" 자체는 새롭지 않다 — 새로움은 **per-primitive 또는 phase-conditional**
   progress 다.
2. 이 기본 progress 는 정확히 아래 (c)의 **길이 confound 형태**(task-level 단조 t/T)라, 그대로 쓰면
   "진행률"이 아니라 "얼마나 오래 돌았나(=timeout 여부)"를 학습하기 쉽다. INSIGHT 가 굳이
   **per-primitive 로 리셋**한 것이 이 confound 를 피하는 핵심 설계임을 역으로 확인해 준다.

### (c) ★ 핵심 caveat — 길이 confound

가장 중요한 함정. **우리 rollout은 task-level이고 실패는 항상 timeout**이다 (메모리
`seen18-rollout-length-confound`): seen18에서 실패=항상 45-step timeout / 성공=조기종료라
**길이가 사실상 라벨을 결정**(time-pooled feature의 AUROC 0.998은 아티팩트). 이 환경에서
naive하게 `y = t/T_task` (task 전체 정규화 timestep)로 progress predictor를 학습하면:

- 성공 rollout은 T가 짧고 실패는 T가 길다 → `t/T` 가 **success/failure·길이와 교란**된다.
- predictor가 "진행률"이 아니라 **"내가 timeout rollout 안에 있나"**(=길이 confound)를
  학습한다. 표현 분리 메모(`seen18-rollout-length-confound`,
  `seen18-safe-detector-verified`)가 경고하는 바로 그 아티팩트.

반면 **INSIGHT progress는 per-PRIMITIVE 정규화**라 의미가 근본적으로 다르다. primitive는
종료 조건이 명확한 짧은 segment이고, segment 내 `t/(T_prim-1)`은 timeout과 무관한
"이 동작이 얼마나 진행됐나"를 잰다. task-level `t/T_task` 와 같은 양이 아니다.

→ **의미 있는 재현의 전제조건: primitive(=subtask) 경계 segmentation이 선행돼야 한다.**
GR00T rollout을 primitive로 자르는 파일럿 없이 task-level progress를 학습하는 것은 길이
confound를 다시 회귀시키는 것에 불과하다. INSIGHT은 이 segmentation을 VLM(Gemini 3
Flash) + gripper transition + EE dominant-motion으로 오프라인 자동화한다 (paper §3.1.2,
Appendix B.1) — 우리도 같은 종류의 경계 신호(예: subtask phase 접근/파지/이송/배치)가
필요하다.

### (d) 메인 method 연결 — online phase 신호로서

doc 14 "phase를 online에 어떻게 아나" 절은 세 후보를 든다:

- 절대 t-bin: 싸지만 거칠다(길이 달라 t 의미 다름).
- **progress-normalized(0~1): 의미는 맞지만 online 계산 불가 (총길이 T 모름).**
- subtask phase: 최선이나 검출기 필요.

학습된 progress predictor의 핵심 가치는 바로 이 **"offline only" 문제의 우회**다. `t/T`는
오프라인에서 T를 알아야 계산되지만, predictor는 **현재 관측 한 장에서 progress를 직접
회귀**하므로(INSIGHT처럼 매 step `action[7]` 산출) T를 몰라도 online으로 진행률을 읽는다.
INSIGHT의 progress > 0.95 종료가 정확히 이 online 사용 사례다 — 미래를 모른 채 "지금 이
primitive가 끝났나"를 매 tick 판정.

단, 메인 method에 투입하려면 두 가지를 **별도로** 검증해야 한다:

1. **신뢰성**: predictor가 (c)의 길이 confound를 학습하지 않았는지 — primitive-normalized
   라벨 + 길이 통제(`truncation-length-standard` 류) 하에서만 진짜 진행률을 잰다고 주장 가능.
   OOD에서 신뢰 붕괴(INSIGHT drawer §4.2: OOD에서 progress 신뢰 불가 → VLM 폴백)도 우리
   환경에서 재현될 수 있다.
2. **인과**: online progress 신호로 phase를 라우팅한 steering이 실제 ΔSR을 내는지는
   상관(검출)과 별개다. doc 14 사다리(pathway-split → +phase-bin)에서 phase-bin을 progress로
   격상할 때 ΔSR로 직접 인과 재측정해야 한다 (`eval-seed-standard`: EVAL_SEED=100000).

---

## 4. 결론 & 권고

| 컴포넌트 | INSIGHT | 우리 재현 가능성 | 비고 |
|---|---|---|---|
| 라벨 `y = t/T` | per-primitive `arange(n)/(n-1)` | 매우 쉬움 | 단, **per-primitive** 여야 의미 있음 |
| progress head | action expert 8번째 출력 dim (π0.5) | 쉬움 (외부 head) | git `src/ttt/progress_head.py` 복원 |
| head 위치 | action head baked-in, 재학습 | **부적합** (백본 재학습 금지·DiT flow-matching) | → 외부 predictor로 우회 |
| primitive segmentation | VLM+gripper+EE motion (오프라인) | **선행 필요** (미구현) | 없으면 길이 confound 재학습 |
| online 사용 | progress>0.95 종료, EE/VLM 폴백 | 가능 (관측→progress 직접 회귀) | T 모름 문제 우회의 핵심 |
| 길이 confound 통제 | 무관 (primitive가 짧고 종료 명확) | **필수** | task-level `t/T`는 confound |

**판정 요약 (5줄):**

1. INSIGHT progress channel = action head의 8번째 출력 dim, 라벨은 **per-primitive
   정규화 timestep** `t/(T_prim-1)` (task-level 아님, 매 primitive 리셋·비단조).
2. 개념·코드(라벨·MLP head·학습 루프)는 **재현 매우 쉽다** — 우리 git history의
   `src/ttt/progress_head.py`/`predictor.py`에 이미 구현돼 있었다(복원만 하면 됨).
3. 단, INSIGHT은 action head를 재학습하는데 우리는 백본 재학습 금지 + GR00T는 DiT
   flow-matching head → **외부(frozen-feature) progress predictor가 적합한 형태**.
4. **핵심 함정 = 길이 confound**: 우리 rollout은 task-level·실패=항상 timeout이라 naive
   `t/T_task` 학습은 진행률이 아니라 timeout 여부를 학습한다 → **primitive 경계
   segmentation 파일럿이 선행돼야** INSIGHT progress와 같은 의미가 된다.
5. 메인 method(doc 14) 관점에서 학습된 predictor는 "progress는 의미 맞지만 online 계산
   불가(T 모름)" 문제를 **관측→직접 회귀로 우회**하는 유망한 online phase 신호원 — 단
   신뢰성(길이 통제)·인과(ΔSR)는 별도 검증 필요. **실제 학습·평가는 후속 작업.**
