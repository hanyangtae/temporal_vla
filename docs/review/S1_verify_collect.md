# S1 검증 — 수집 파이프라인을 무엇으로 어떻게 확인하는가

`http_feature_collect.py`는 1261줄이다. **위에서 아래로 읽지 말고, 질문별로 확인 지점을 찍는다.**
그리고 대부분은 코드를 읽는 것보다 **산출물을 검사하는 쪽이 강한 증거**다 — 코드는 "그렇게
하려는 의도"만 보여주고, 산출물은 "실제로 그렇게 됐는지"를 보여준다.

도구: `scripts/review/inspect_rollout.py` (pkl 무결성·정합·bias 자동 점검)

```bash
P=/home/dongkyu/miniconda3/envs/lerobot_safe/bin/python   # torch 필요
$P scripts/review/inspect_rollout.py <cell_dir>            # 개별 + 요약
$P scripts/review/inspect_rollout.py <cell_dir> --summary  # 요약만
```

## 0. 파일 구조 지도 (읽어야 할 곳만)

| 줄 | 무엇 | 어느 질문 |
|---|---|---|
| 97–322 | `class N15LerobotHttpFeatureClient` | ① activation 캡처 |
| 235–322 | ㄴ `get_action()` — 추론 요청 → record 생성 | ① ③ |
| 434–449 | `_resolve_task_dir` / `_resolve_cell_dir` / `_normalize_instruction` | ② 분류 저장 |
| 452–489 | `make_env()` — `gym.make(seed=scenario_seed)` | ② scene 고정 |
| 492–675 | `parse_args()` (184줄) | 전체 — 어떤 스위치가 있는지 |
| 325–344 | `_post_steering_phase()` | ③ steering |
| 384–431 | `_action_tremor_summary()` | ③ 개입 부작용 진단 |
| 885–1257 | `run()` — 메인 루프 | ③ ④ |

나머지(`_run_probe` 152줄 등)는 exp5-4 전용 경로라 지금 안 봐도 된다.

## ① 지정한 activation이 잘 캡처되는가

**코드에서 볼 곳**: `get_action()` 292–322행. `features["hidden_states"]`를 받아 record에 담는다.
`vl_hidden_states`·`cross_attn`은 있을 때만 추가된다.

**중요 — 캡처를 결정하는 건 이 파일이 아니다.** 여기는 요청만 하고, 무엇을 캡처할지는
**serve 쪽 플래그**가 정한다(`--capture-vl`, `--capture-cross-attn`, capture layers).
그래서 "지정한 게 캡처됐나"는 **산출물의 메타로 확인**해야 한다.

**산출물에서 확인** — pkl이 자기서술적이다:

```
feature_kind         groot_n15_dit_block_residual_action_tokens_denoise
capture_layers       [0, 2, 4, 8, 10, 12, 15]
layer_count          7
hidden_states        per-record (7, 4, 1536)   ← (layer, denoise K, D)
vl_feature_kind      groot_n15_vlln_seq_meanpool
vl_hidden_states     per-record (2048,)
```

`hidden_states` 첫 축(7) == `len(capture_layers)`(7) 인지가 핵심 게이트다. 도구가 자동 검사한다.

**실측 결과** (`phase_event_6p/.../ppcs_apple_s100215`, 60 rollout): 60/60 문제 없음.
shape record 간 일관, NaN·Inf 0, 분산 0 record 0.

⚠ **주의 — fp16 저장**. activation은 `float16`으로 저장된다. 측정해보면:

| 항목 | 값 |
|---|---|
| absmax | 957.5 (fp16 한계 65504 — 오버플로 없음) |
| 상대 양자화 오차 | median 7.0e-4, p99 9.7e-4 |
| 크기 의존성 | 없음 (부동소수점이라 상대오차 균일) |

즉 **편향이 아니라 0.07% noise floor**다. 다만 값의 0.68%가 |x|>256이라 **fp16으로 제곱하면
오버플로**한다 → 통계 계산 전 캐스팅은 필수다. 실 파이프라인은 지킨다(§4 감사 결과).

## ② instruction + scene에 맞게 분류 저장되는가

**코드에서 볼 곳**: `_resolve_cell_dir()` 440–445행이 `<root>/<task>/<cell_id>` 경로를 만든다.
`make_env()` 473행 `gym.make(env_name, seed=scenario_seed)`가 scene을 고정한다.

**산출물에서 확인** — 같은 사실이 **네 곳에 중복 기록**되므로 교차 검증이 된다:

| 사실 | 경로 | 파일명 | pkl |
|---|---|---|---|
| task | `PickPlaceCounterToStove/` | `task1--` | `robocasa_task` |
| cell(scene) | `ppcs_apple_s100215/` | — | `cell_id`, `scenario_seed`, `seed` |
| episode | — | `ep25--` | `episode_idx` |
| 성공 | — | `succ1` | `episode_success` |
| instruction | — | — | `canonical_instruction`, `task_description` |

도구가 이 5쌍을 전부 대조한다. 특히:

- `cell_id`의 `_s100215` 접미사 == `scenario_seed` (scene 고정이 실제로 됐는지)
- `canonical_instruction` == `task_description` (**모델에 보낸 문장과 라벨이 같은지**)
  — `--instruction-override`를 쓰면 일부러 다르게 할 수 있으므로, 이 불일치는 arm에 따라
  정상일 수도 있다. 그때는 어떤 arm인지 확인할 것.

**실측**: 60/60 전 항목 일치. `inference_seed / episode_idx = 1000` 일관 → **결정적 유도**.

## ③ steering이 적절히 적용되는가

**이 파일은 steering을 하지 않는다.** 통지·요청만 한다. 실제 개입은 serve 쪽
(`scripts/serve/lerobot.py` + `steering_hooks.py`)에서 일어나고, 그건 **S5의 검증 대상**이다.

이 파일에서 확인할 것은 **"개입이 의도한 시점에 켜졌는가"** 세 갈래다:

| 방식 | 스위치 | 코드 |
|---|---|---|
| phase-gated | `--gated-steering` | `_post_steering_phase()` 325–344 |
| record latch | `--steer-from-record N` | `run()` 내 `progress_before < steer_from_record` |
| noise 재샘플 (대조) | `--reseed-from-record N` | `get_action()` 249–252 |

**설계상 좋은 점 두 개**를 확인했다:

- `_post_steering_phase()`는 실패 시 **즉시 예외**를 던진다 — "무음 미조향"(steering이 조용히
  안 걸린 채 실험이 완주되는 것)을 막는다. `PITFALLS.md` §1의 α 사고와 같은 계열의 방어다.
- `--steer-from-record`와 `--gated-steering`은 **상호배타로 강제**된다(둘 다 주면 ValueError).

**산출물에서 확인**: steering arm의 pkl에는 개입 관련 필드가 추가로 있어야 한다. 위 실측 대상은
fit 수집(개입 없음)이라 그 필드가 없다. **steered arm 하나를 골라 같은 도구로 돌려 비교**하는
것이 다음 확인 절차다.

`_action_tremor_summary()`(384–431)는 개입이 액션 궤적에 준 떨림(jerk)을 record 해상도로 낸다.
exp5-3에서 "action 토큰 개입이 파괴의 주범(jerk 0.94× vs full 붕괴)"을 잡은 지표가 이것이다.

## ④ 손상이 없는가

도구가 자동으로 보는 것:

| 검사 | 무엇이 나오면 문제 |
|---|---|
| 언피클 | 예외 → 파일 손상 |
| record 축 길이 | `hidden_states` / `actions` / `states` / `feature_phases` / `action_vectors` 가 서로 다름 |
| `phase_timeline` | record 수도 record+1도 아님 |
| shape 일관성 | record 간 shape 변동 |
| NaN / Inf | 0이 아님 |
| 분산 0 | 상수 텐서 = 캡처 실패 |
| 라벨 일치 | 파일명 ↔ pkl (성공·episode·cell·task·seed) |

**실측 60 rollout: 전부 통과.**

보존 검증(아카이브·삭제 전)은 별도다 → [`../steering/DATA_HANDLING.md`](../steering/DATA_HANDLING.md) §1.
**평균 파일 크기 상식 체크**가 거기 핵심이고, exp2 fit 유실이 그걸 안 해서 났다.

## ⑤ 파이프라인이 만드는 bias

### 5.1 dtype — 감사 완료, 문제 없음

activation이 fp16인데 conceptor는 `R = E[hhᵀ]`(제곱)을 쓴다. fp16으로 계산하면 오버플로한다.
**전 계산 경로가 캐스팅 후 계산하는지 확인했다**:

| 경로 | dtype |
|---|---|
`src/conceptor/core.py` | **float64** 전체. 문서에 근거 명시(COAST App A.9.4) |
`fit_phase_conceptor_n15.py` | 로드 float32 → 누적 float64 |
`fit_mean_diff.py` (setM) | `astype(np.float64)` 후 einsum |
`fit_within_scene_setM.py` · `fit_setm_induced.py` | float32/float64 캐스팅 후 |
`build_sae_inputs.py` | float32 (layer 인덱싱을 캐스팅보다 먼저 — 메모리) |
`pathway_separation.py` · `vis/core/features.py` | float32 |

이미 유사 사고를 잡은 이력도 있다 — `fit_mean_diff.py`에 *"float32 평균 누적 때문에 성립하지
않는다 — 07-23 beer에서 실측"*이라며 float32 누적 vs float64 episode-합 불일치를 발견하고
**float64 경로를 배포본으로** 삼았다고 적혀 있다.

> 교훈: 이 절을 쓰기 전에 제 점검 도구가 fp16 std를 계산해 `inf`를 뱉었다. 그걸 데이터 이상으로
> 오해할 수 있었다. **도구의 버그와 데이터의 문제를 먼저 갈라야 한다.**

### 5.2 ★ succ/fail 극단 쏠림 — 실측에서 발견된 실제 문제

`phase_event_6p` fit 트리 전 cell을 파일명만으로 집계한 결과:

| cell | fail | succ | SR | 대조 fit |
|---|---|---|---|---|
| ppcs_apple_s100106 | 0 | 60 | 1.00 | ✗ |
| ppcs_apple_s100172 | 0 | 60 | 1.00 | ✗ |
| ppcs_apple_s100184 | 0 | 60 | 1.00 | ✗ |
| **ppcs_apple_s100202** | **60** | **0** | **0.00** | ✗ |
| ppcs_apple_s100215 | 0 | 60 | 1.00 | ✗ |
| ppcs_apple_s100243 | 3 | 57 | 0.95 | OK (겨우) |
| ppcs_apple_s100395 | 11 | 49 | 0.82 | OK |
| ppcs_apple_s100422 | 6 | 54 | 0.90 | OK |
| ppcs_apple_s100425 | 0 | 60 | 1.00 | ✗ |

**9 cell 중 6개가 대조 fit 불가.** 5개는 실패 0판, 1개는 성공 0판.

`PITFALLS.md` §4에 "고SR scene은 fit 창에 실패 2~6판뿐"이라고 적었는데 **실물은 0판**이다.
문서가 문제를 과소기술하고 있었다. 그리고 이건 코드 버그가 아니라 **scene 선정의 결과**다 —
seed를 고정하면 그 scene의 난이도가 고정되고, apple은 대부분 SR 1.0 아니면 0.0으로 갈린다.

**함의**: 이 트리로 fit한 conceptor는 6/9 cell에서 빈 fit이거나 한쪽 클래스만 본 것이다.
`PITFALLS.md` §4의 "빈 fit도 `[done]` 통과" 게이트 구멍과 합치면, **조용히 무의미한
conceptor가 만들어졌을 수 있다.** exp2 결과 재해석 시 확인 대상.

### 5.3 그 밖에 도구가 보는 bias 신호

| 신호 | 왜 |
|---|---|
succ/fail **record 수 분포 겹침** | 겹치지 않으면 길이만으로 라벨이 결정된다(AUROC 1.0 아티팩트) |
instruction별 SR 편차 | 크면 VL 분리가 instruction 아티팩트일 수 있다 |
phase 도달률 | 후반 phase 도달률이 낮으면 phase별 분석에 선택 효과가 낀다 (실패는 후반 phase에 못 감) |

**실측 — `ppcs_apple_s100395`(fail 11 / succ 49), 60 rollout 무결성 전부 통과:**

```
길이 confound — record 수 분포
   succ=1  n= 49  mean= 49.24  min= 41  max= 78
   succ=0  n= 11  mean=144.00  min=144  max=144
   ⚠ 범위가 전혀 겹치지 않음 — 길이만으로 라벨이 완전 결정된다(AUROC 1.0)
```

**실패 11판이 전부 정확히 144 record**다. `n_action_steps=5` × 144 = 720 = `max_episode_steps` —
즉 **실패는 예외 없이 timeout**이고 성공은 41~78에서 조기 종료한다. 두 분포가 **한 점도 겹치지
않으므로 record 수만으로 라벨이 100% 맞는다.** 문서에 적혀 있던 "time-pooled 분리는 길이
아티팩트(AUROC 0.998)"가 여기서 수치로 확정된다 — 이 cell에서는 0.998이 아니라 **1.0**이다.

**함의**: 이 데이터로 succ/fail을 나누는 어떤 분석도 길이를 통제하지 않으면 무의미하다.
통제 방법은 고정-t 절단(`truncation` 창)이고, 겹치는 구간이 아예 없으므로 **성공 길이 범위
[41, 78] 안에서만** 비교해야 한다.

```
phase 도달률 (60 rollout 중)
     60  grasp / reach-to-object
     53  transport
     51  place
     45  insert-settle
```

후반 phase로 갈수록 도달률이 떨어진다(60 → 45). **phase별 분석은 후반 phase에서 표본이
성공 쪽으로 치우친다**(실패는 후반 phase에 도달하지 못함) → phase별 succ/fail 비교에는
선택 효과가 낀다. 표본수를 반드시 병기할 것.

## 다음 확인 절차 (제안)

1. ~~길이 confound 실측~~ — 완료(§5.3). 결과: 범위 겹침 0, AUROC 1.0
2. steered arm pkl 하나 골라 실행 → **③의 개입 필드 유무 대조**
3. exp2가 실제로 어느 cell로 fit했는지 확인 → **6개 불성립 cell이 쓰였는지**
4. `archive_on_done.sh`가 `DATA_HANDLING` §2 규약(`-L`·상대 심링크)을 지키는지 (32줄, 즉시 가능)
