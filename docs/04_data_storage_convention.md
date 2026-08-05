# Activation·연산자 저장 규약

이 문서는 rollout activation 과 steering 연산자를 **어떤 단위로 식별하고 어디에 저장하는지**의
단일 출처다. 수집기·fit 스크립트·분석 스크립트는 모두 이 규약을 따른다.

작성 2026-07-31. 근거 실측은 승준 아카이브(`/home/kimseungjun/datasets/temporal_vla_outputs`,
133,011 파일 / 819G) 전수 스캔 결과.

---

## 0. 왜 필요한가 (해결하려는 실제 고장)

기존 저장 방식에서 아래가 **실측으로** 확인됐다. 규약은 이 셋을 각각 구조적으로 막는다.

| 고장 | 실측 | 원인 |
|---|---|---|
| 같은 경로에 다른 내용 | 776 건 | 같은 arm 을 여러 머신에서 실행 → 경로가 내용을 식별하지 못함 |
| 연산자 출처 끊김 | `fit_inputs.json` 1,943 참조 중 **672 건(35%)** 경로 무효 | 절대경로 기록. 그중 도커 내부 경로(`/temporal_vla/...`), 옛 머신명(`/home/dongkyu/pdk_ws/...`) |
| 중복 판정 비용 | md5 매니페스트 20만 행 대조 필요 | 내용 동일성을 저장 구조가 보장하지 않음 |

핵심 원인은 하나다. **경로가 식별자 역할을 하는데, 경로는 내용을 결정하지 않는다.**

### 2026-08 정리로 확인된 추가 사실

| 사실 | 실측 | 규약이 막는 방식 |
|---|---|---|
| 같은 내용이 여러 경로에 중복 저장 | **19,257 종 / 21.03G** (436G 의 4.8%) | 내용 주소 저장이면 같은 sig 는 한 자리에만 존재 (§2) |
| 캡처 밀도가 기록되지 않아 pkl 을 직접 열어야 판정 가능 | pkl 132 개 수동 분류 | 캡처 밀도 5 열을 인덱스에 (§4) |
| 라운드 개명으로 같은 데이터가 두 이름에 | `pq2/exp2`, `pq3/exp3`, `deprecated/` 중복 | `legacy_path_map` 으로 옛 이름 해석 (§4) |

정리 결과 재고는 846G → 436G 가 되었고, 남은 activation 은 `[7,4,49,1536]` 9 run 686 판
289.6G(전부 N1.5), 연산자는 505 개다. 삭제 원장은 승준 `~/DELETED_20260803.tsv`.

---

## 1. 용어와 단위

### rollout — 에피소드 1회 실행

두 종류가 있고 **구성 파일이 다르다**. 아카이브 실측 42,945 에피소드 기준:

| 종류 | 개수 | 구성 | 용도 |
|---|---|---|---|
| **수집 rollout** (`activations/`) | 11,477 (27%) | `rollout.pkl` + `traj.csv` + `video.mp4` + `meta.json` | conceptor/setM fit 의 입력 |
| **평가 rollout** (`evals/`) | 31,468 (73%) | `traj.csv` + `video.mp4` + `meta.json` (pkl 없음) | SR·ΔSR 판정 근거 |

평가 rollout 에 pkl 이 없는 것은 결손이 아니라 의도다(exp3 부터 eval 캡처 off).
**두 종류를 한 디렉토리에 섞지 않는다** — 용량 특성과 사용처가 완전히 다르다.

한 rollout 을 결정하는 축:

```
model(백본·체크포인트) × task × cell × env_seed × inference_seed × machine × 실행시각
```

pkl 내부는 다시 `layer × rollout-step × denoising-step × token 위치` 로 나뉜다.
**이 내부 축은 파일을 쪼개지 않고 pkl 안에서 관리한다** (현행 유지).

### operator — 연산자 (conceptor / setM / SAE / direction)

여러 rollout 을 **어떻게 묶어서** **어떤 연산으로** 만들었는지로 결정된다.

```
operator = f( 입력 rollout 집합, 연산 파라미터 )
```

종류가 넷이고 **식별 파일이 다르다**. 인덱서는 셋 다 인식해야 한다 —
`conceptors.npz` 만 보면 exp5 산출물을 통째로 놓친다(실제로 놓쳤다).

| 종류 | 식별 파일 | 메타 |
|---|---|---|
| conceptor / setM | `conceptors.npz` | `config.json` (구 `metadata.json`) |
| SAE | `model.pt` | `config.json` + `metrics.json` |
| G3 direction | `fit_meta.json` | 자체 포함 |

### ★ `config.json` 필수 — 이것이 연산자 저장의 표준이다

**연산자 디렉토리에는 `config.json` 이 반드시 있어야 하고, 그 안에 입력 rollout 이
기록돼야 한다.** exp5 SAE 가 이 형태를 이미 갖추고 있어 이를 표준으로 삼는다:

```json
{
  "cell": "scene_matched_mixer", "layer": 0,
  "m": 6144, "k": 64, "seed": 0, "aux_k": 0,
  "split_col": "split_scene", "split_axis_scene_heldout": true,
  "train_episode_fingerprint": "c86834a1b63e",   ← 입력 집합의 지문
  "n_train_episodes": 112,
  "train_episodes": [0, 1, 2, ...]               ← 입력 목록
}
```

필수 키: 연산 파라미터 전부 + **입력 rollout 의 `sig` 목록**(신규는 episode index 가
아니라 sig 로 쓴다) + 그 집합의 지문. conceptor 계열의 `fit_inputs.json` 은 같은 역할을
하므로 `config.json` 에 흡수한다.

**출처가 기록되지 않은 연산자는 보관하지 않는다.** 재현할 수 없는 연산자는 자산이 아니라
부채다 — activation 이 살아 있으면 언제든 다시 fit 하면 된다. 2026-08 정리에서 이 기준으로
504 개(10.44G)를 삭제하고 100 개를 남겼다.

---

## 2. 식별자

### sig — rollout 지문

```
수집 rollout : sha256(rollout.pkl 전체 바이트)[:16]
평가 rollout : sha256(traj.csv 전체 바이트)[:16]
```

기존 `scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py` 의
`_content_sig()` 와 **동일한 정의**다. 이미 기록된 1,943 건의 sig 를 그대로 쓸 수 있으므로
변환 로직이 필요 없다.

부분 해시(앞부분만 읽기)는 금지. 수십 MB pkl 의 중간 변조를 못 잡는다.

### opsig — 연산자 지문

```
opsig = sha256( 연산자 실체 파일 전체 )[:16]
        실체 = conceptors.npz | model.pt | (direction 은 fit_meta.json 이 실체)
```

**파라미터 해시가 아니라 내용 해시다.** 처음에는 `sha256(입력 sig 목록 + 파라미터 JSON)`
으로 정의했으나 2026-08 실측에서 **604 개 중 393 개가 90 개 지문으로 충돌**했다. 원인 둘:

1. **파라미터 목록이 연산자를 결정하지 못함** — `setM_future_only_placebo_loo` 의 LOO
   제외 판(`ep15` vs `ep6`)이 파라미터에 없어 서로 다른 연산자가 같은 지문을 얻었다.
   축을 하나 추가해도 다음 축에서 같은 일이 반복된다.
2. **메타 파일 자체가 없는 경우** — 파라미터가 전부 `None` 이 되어 13 개가 한 지문으로
   뭉쳤다. 파라미터 확장으로는 원리적으로 해결 불가.

내용 해시는 rollout 의 `sig` 와 규칙이 같고("실체의 내용이 곧 지문"), 메타가 없어도 항상
작동한다. **"같은 입력·같은 설정 → 중복 fit" 판정은 그대로 유지된다** — fit 이 결정적이면
같은 입력·설정은 같은 바이트를 내므로 opsig 도 같다.

"왜 다른지"(어느 판을 뺐는지 등)는 지문이 아니라 `operators.tsv` 의 열이 답한다:
`loo_holdout`(경로의 `ep\d+` 또는 config 의 held-out), `params_json`, `legacy_dir`.

### 길이를 16자(64bit)로 정한 근거

| 접두 | 이론 충돌확률(46,238개) | 실측(133,011 파일) |
|---|---|---|
| 8자 (32bit) | 1/486 | 5 건 |
| 12자 (48bit) | 1/1,240만 | 0 건 |
| **16자 (64bit)** | **1/172억** | **0 건** |
| 32자 (128bit) | 1/3×10²⁹ | 0 건 |

100만 개로 늘어도 1/3,689만이다. 32자가 수학적으로 낫지만 실무 차이가 없고,
16자는 기존 기록과 호환된다.

### 충돌 방어 — 비트 수가 아니라 쓰기 검사로 막는다

**저장 시 필수 절차**:

1. 대상 `sig` 디렉토리가 없으면 → 그대로 생성
2. 이미 있으면 → **기존 파일의 전체 sha256 을 다시 계산해 대조**
   - 같으면 → 중복. 새로 쓰지 않고 인덱스에 행만 추가
   - 다르면 → **에러로 중단.** 절대 덮어쓰지 않는다

이 검사가 있으면 충돌이 나더라도 조용히 손실되는 일이 없다.

---

## 3. 디렉토리 레이아웃

`<store_root>` = **`/home/kimseungjun/datasets/temporal_vla_store`** (승준 HDD `/dev/sda2`).
2026-08-04 이관 완료. 구 `temporal_vla_outputs/` 는 제거됐고 이 트리가 유일본이다.

```
<store_root>/
  groot/
    n15/
      activations/<sig>/          rollout.pkl  traj.csv  video.mp4  meta.json
      evals/<sig>/                traj.csv  video.mp4  meta.json
      steerer/<opsig>/            conceptors.npz|model.pt  config.json  …
      sae_inputs/<cell>/inputs/   X_L*.npz  stats_L*.npz   (SAE 학습 입력 행렬)
      runs/<run_id>/              집계 tsv · 플롯 png · 실행 로그 · manifest
      index/                      rollouts.tsv  operators.tsv
                                  operator_inputs.tsv  legacy_path_map.tsv
    n16/runs/                     (activation 은 전량 폐기 — §0)
  pi05/ , cosmos/v1/ , xvla/v1/   (runs 만)
  checkpoints/<name>/             파인튜닝 산출물 (safetensors · optimizer.pt 등)
  _hostcopies/runs/<host>/        머신 재실행 대조분 + REPLICATES.tsv
```

**실측 재고 (2026-08-04 이관 직후)**

| 경로 | 개수 | 용량 |
|---|---|---|
| `groot/n15/activations` | 686 | 291G |
| `groot/n15/evals` | 30,665 | 23G |
| `groot/n15/steerer` | 103 | 18G |
| `groot/n15/sae_inputs` | 5 cell | 35G |
| `groot/n15/runs` + `n16/runs` 외 | — | 17G |
| `checkpoints` | 1 | 23G |
| **합계** | 127,229 파일 | **419G** |

`sae_inputs/` 는 rollout 도 연산자도 아닌 **파생 활성 행렬**이라 자리를 따로 준다.
층별로 미리 뽑아둔 `[N, D]` 행렬이며 SAE 학습의 직접 입력이다.

`views/` (인덱스에서 생성하는 symlink 트리)는 필요해지면 만든다 — 아직 없다.

`model/version` 을 최상위 물리 분리로 두는 이유: 백본이 다르면 특징 공간이 달라 **섞일 일이
없고**, "n15 전체 폐기" 같은 조작이 한 번에 된다.

`runs/` 는 에피소드도 연산자도 아닌 산출물(플롯·집계표·로그·manifest)의 자리다.
지문 대상이 아니며 현행 디렉토리 구조를 그대로 유지한다.

`run_id` 는 **이관 전 경로에서 rollout 디렉토리 윗부분을 그대로 딴 문자열**이다.
`raw_rollouts/` 앞까지를 `/` → `__` 로 치환한다. 예:

```
옛 경로 : groot_n15/steer_eval_pq3/e1/pq3_ppcc_bread/ho_base/raw_rollouts/.../task5--ep10--succ0.pkl
run_id  : steer_eval_pq3__e1__pq3_ppcc_bread__ho_base
```

`rollouts.tsv` 의 `source_run` 이 이 값이며, 같은 내용(`sig`)이 서로 다른 run 에서 관찰되면
`sig` 는 하나지만 `source_run` 이 다른 행이 여러 개 생긴다.

`views/` 는 필요할 때 인덱스에서 생성하는 symlink 트리다. 용량을 쓰지 않고, 보고 싶은 축이
바뀌면 **뷰만 다시 만든다** — 실체 800G 를 다시 옮기지 않는다.

```
views/by-cell/<model>/<task>/<cell>/env<seed>/ep<N>  ->  ../../../activations/<sig>
```

---

## 4. 인덱스 (3표)

역추적이 "연산자 ↔ 입력 rollout" **다대다** 관계이므로 잇는 표가 반드시 따로 필요하다.

### `rollouts.tsv`

복합키 **(sig, source_run)**. 내용이 같아도 출처가 다르면 별도 행이다
(머신이 달라도 mp4 가 바이트 동일한 사례가 실재하므로, 저장은 합치되 출처는 남긴다).

| 열 | 설명 |
|---|---|
| `sig` | rollout 지문 |
| `kind` | `activation` \| `eval` |
| `source_run` | 원 수집/평가 run 식별자 (`legacy_path_map` 의 run 부분) |
| `model` | 백본 계열. 사이드카 `model_family` (예 `lerobot_groot_n15`) |
| `ckpt` | 체크포인트 식별자. `configs/checkpoints/*.yaml` 프로파일명, 없으면 빈 칸 |
| `task`, `task_id` | 예 `PickPlaceCounterToCabinet`, `5` |
| `cell` | 예 `pq3_ppcc_bread` |
| `instruction` | 사이드카 `task_description` |
| `env_seed` | 사이드카 `scenario_seed` |
| `inference_seed` | 사이드카 `inference_seed` |
| `machine` | serve `/health` 의 `serve_machine` (예 `kanu:gpu3`). 구 수집분은 `MACHINE.txt` |
| `machine_source` | `MACHINE.txt` \| `runs/MACHINE.txt` \| `unrecorded` — 값의 출처 |
| `episode_idx` | |
| `success` | 0/1 |
| `steps`, `n_inferences`, `n_action_steps`, `chunk_len` | |
| `has_pkl` | 0/1 |
| `meta_source` | 이 행의 값이 어디서 왔는지 (`sidecar` \| `summary_tsv` \| `per_episode_tsv` \| `filename`) |

**캡처 밀도 열** — 아래 5 개는 `kind=activation` 행에 필수다. 없으면 "L12 activation 줘" 같은
질의에 4MB/판 짜리와 565MB/판 짜리가 섞여 나온다(실측 판당 크기 편차 140 배).

| 열 | 설명 | 예 |
|---|---|---|
| `capture_token_mode` | 캡처 시 토큰 모드 | `all_token_full` \| `action_token_mean` \| `valid` \| `full` \| (없음) |
| `feature_kind` | 캡처 계약 문자열 | `groot_n15_dit_block_residual_full_tokens_denoise` |
| `feature_axes` | 축 이름 목록 | `layer,denoise_step,model_token,feature_dim` |
| `record_shape` | record 1 개의 실제 shape | `[7,4,49,1536]` |
| `capture_layers` | 캡처한 층 인덱스 | `[0,2,4,8,10,12,15]` |

**★ 판정은 `feature_axes`/`record_shape` 로 한다. `ndim` 만으로 하면 틀린다.**
`ndim=3` 에는 서로 다른 두 가지가 섞여 있다:

| shape | 축 | 토큰 |
|---|---|---|
| `[7,4,49,1536]` | `layer·denoise_step·model_token·feature_dim` | 49 전부 보존 |
| `[7,49,1536]` | `layer·model_token·feature_dim` | 49 전부 보존 (denoise 축 없음) |
| `[7,51,1536]` | `layer·token_pos·feature_dim` | 51 전부 보존 (denoise 평균) |
| `[7,4,1536]` | `layer·denoise_step·feature_dim` | **토큰 소실**(평균됨), 가운데 4 는 denoise |
| `[32,1536]` | `layer·feature_dim` | 전부 평균 |

가운데 숫자가 49/51 이면 토큰이고 4 면 denoise 다. 2026-08 정리 때 이 구분을 놓쳐
`kmean_perT`(184G)를 토큰평균으로 오분류할 뻔했다.

N1.5 는 49 토큰(= state 1 + future 32 + action 16, `_token_segments()` 가 경계를 나눔),
N1.6 은 51 로 DiT 시퀀스 구성이 다르다. **두 백본의 토큰 자리는 대응하지 않는다** —
`model/version` 물리 분리(§3)의 근거이기도 하다.

### `operators.tsv`

| 열 | 설명 |
|---|---|
| `opsig` | 연산자 지문 |
| `operator` | 예 `setM_gated_future_only_placebo` |
| `cell`, `phase`, `layer` | |
| `alpha`, `perm_id`, `length_control` | 연산 파라미터 |
| `n_input_succ`, `n_input_fail` | 입력 판수 |
| `provenance` | `full` (입력 sig 전부 기록) \| `unknown` (복원 불가) |
| `created_at`, `created_by` | |

### `operator_inputs.tsv`

| 열 | 설명 |
|---|---|
| `opsig` | |
| `sig` | 입력 rollout |
| `label` | 0=fail / 1=succ |
| `fit_start_record`, `fit_records` | 길이통제 창 |

### `legacy_path_map.tsv` — 영구 보존

| 열 | 설명 |
|---|---|
| `legacy_path` | 이관 전 경로 (호스트 접두 포함) |
| `sig` 또는 `opsig` | |
| `host` | 그 경로가 유효했던 호스트 |

기존 exp2~exp5 문서·노트북·스크립트가 옛 경로를 참조하므로 **이 표 없이는 과거 문서가
전부 미아가 된다.** 옛 문서를 최신 경로로 갱신할 때의 기준표이기도 하다.

---

## 5. 메타데이터 수집 규칙

현재 에피소드 정보는 다섯 군데에 흩어져 있다. 인덱스 구축 시 **아래 우선순위로** 병합한다.

1. `task*--ep*--succ*.json` 사이드카 — 가장 풍부(`inference_seed` 등). 단 5,751 개뿐
2. `collection_summary.tsv` — `task_id · task · episode_idx · seed · exit_code · pkl`
3. `per_episode.tsv` — `episode_idx · success · language`
4. `MACHINE.txt` — 머신·시각·`eps_added`
5. 파일명 — `task{T}--ep{N}--succ{0,1}`

**결측은 추측으로 채우지 않는다.** 빈 칸으로 두고 `meta_source` 에 출처를 남긴다.
빈 칸 자체가 "이 판을 분석에 쓸지" 판단하는 근거다.

---

## 6. 신규 생성 시 지켜야 할 것

수집기·fit 스크립트가 새로 산출물을 만들 때:

1. **절대경로를 기록하지 않는다.** 다른 파일을 가리킬 때는 `sig`/`opsig` 를 쓴다.
   (도커 안에서 돌면 `/temporal_vla/...` 가 기록되어 호스트에서 영원히 못 찾는다 — 실제 발생)
2. **저장 전 §2 쓰기 검사를 수행한다.**
3. rollout 을 만들면 `meta.json` 을 반드시 함께 쓰고, 최소한 아래를 담는다:
   `model, ckpt, task, task_id, cell, instruction, env_seed, inference_seed, machine,
   episode_idx, success, steps, n_action_steps, chunk_len`
   — 현재 사이드카에 **없는 것은 `machine` 과 `ckpt` 뿐**이므로 이 둘을 추가하면 된다.
   `kind=activation` 이면 캡처 밀도 5 열(`capture_token_mode`·`feature_kind`·`feature_axes`·
   `record_shape`·`capture_layers`)도 함께 쓴다. 이 값들은 현재 pkl 안에만 있어
   **사이드카에 없다** — 수집 rollout 은 `raw_rollouts/` 에 json 자체가 없는 경우가 많다.
   그래서 2026-08 정리 때 pkl 132 개를 직접 열어 분류해야 했다.

   **`machine`·`ckpt` 는 serve `/health` 가 정본이다** (2026-08-05 배선). serve 가 도는
   머신이 수집기와 다를 수 있으므로 클라이언트가 자기 호스트명을 쓰면 틀린다.

   - serve(`scripts/serve/lerobot.py`) 가 `serve_machine`(`<host>:gpu<N>`) 과
     `serve_ckpt`(프로파일명) 를 `/health` 에 노출
   - 수집기(`http_feature_collect.py`)의 `_get_serve_identity()` 가 이를 받아
     `machine`·`ckpt` 로 pkl·사이드카에 기록

   왜 필요한가: `machine` 은 **실험 요인**이다 — 같은 seed·조건이라도 머신이 다르면
   개별 판정이 12.7% 뒤집히고 arm SR 이 ±7~9pp 흔들린다(449 판 대조,
   `_hostcopies/REPLICATES.tsv`). `ckpt` 는 `model_family`(계열명, 686 판 전부 동일)로는
   베이스와 파인튜닝을 구분할 수 없어 필요하다 — 아카이브에 이미
   `checkpoints/groot_n1_5/260513094637-subset100` 이 있다.

   **구 수집분 소급 복원은 부분적으로만 가능했다** (2026-08-05):
   eval 413 판 복원, activation 686 판 중 **526 판은 복원 불가**. `scene_matched_exp41`·
   `phase_event_pq3`·`exp41_mixer` 등이 `MACHINE.txt` 를 남기지 않았고 pkl·사이드카에도
   흔적이 없다. **추측으로 채우지 않고** `machine_source="unrecorded"` 로 명시했다 —
   여러 머신에 걸친 수집일 수 있어 하나로 뭉뚱그리면 층화가 틀린 근거를 갖는다.
   `ckpt` 는 복원 경로 자체가 없다.
4. 연산자를 만들면 `inputs.json` 에 **입력 sig 목록**을 쓴다(경로 아님).
5. 인덱스 3표에 행을 추가한다.

---

## 7. 이관 절차 (기존 819G)

**복사하지 않는다.** 2026-08 정리 후 대상은 436G, HDD 여유는 781G 라 복사도 가능하지만,
아카이브 전체가 같은 파일시스템(`/dev/sda2`) 위에 있으므로 **hardlink 로 새 트리를 만들면
추가 용량이 0 이고 I/O 도 없다.** 436G 를 읽고 쓸 이유가 없다.

```
1. 인덱스 생성        읽기만. 파일 무변경
2. hardlink 트리 구성  용량 0. 원본 그대로 살아있음
3. sig 재검증         새 트리에서 다시 해시 → 인덱스와 대조
4. legacy_path_map 고정
5. 옛 디렉토리 제거    ← 별도 승인. 링크만 끊기고 데이터는 새 트리가 유지
```

3 단계까지는 **원본과 새 트리가 동시에 존재**하며 용량은 그대로다. 문제가 보이면 새 트리만
지우면 원상복구다.

### 실행 기록 (2026-08-04, 완료)

| 단계 | 결과 |
|---|---|
| 인덱스 v2 | 40 분. 실패 0. activation 캡처밀도·신원 **100%** |
| hardlink 이관 | 링크 60,809 + 잔여 66,481. **중복 sig 1,568 자동 병합** |
| 검증 | 링크수 1인 pkl **0** · 인덱스↔디스크 686/686·103/103 · sig 재계산 표본 5/5 |
| 옛 트리 제거 | HDD 여유 798G **불변** — 이름 하나를 뗀 것이므로 정상 |

**hardlink 제거 전 필수 관문:** `find <옛트리> -type f -links 1` 이 **0** 이어야 한다.
링크수 1 = 새 트리에 안 딸려온 파일 = 지우면 유실. 실제로 이 관문이 두 번 작동했다 —
1 차에 87.85G(SAE 입력 34.8G·체크포인트 22.1G·run 산출물 31G), 2 차에 3 파일을 잡았다.
이관기가 `task*--ep*--succ*` 패턴만 rollout 으로 봐서 나머지를 통째로 빠뜨렸기 때문이다.

### ★ 이관이 실증한 것 — 입력 기록은 연산자 디렉토리 안에 있어야 한다

conceptor 96 개가 이관 직후 `provenance=unknown` 으로 떨어졌다. `fit_inputs.json` 이
**arm 루트**에 있어 연산자 디렉토리(`<arm>/<phase>/<layer>/`)를 옮길 때 딸려오지 않았다.
§1 이 경고한 상황이 그대로 재현된 것이다. 이관 전 인덱스에서 sig 를 복원해 각 연산자
디렉토리에 `config.json` 을 새로 썼고, 현재 **103/103 이 입력 sig 를 보유**한다.

같은 사고를 막기 위해 `src/utils/operator_config.py` 의
:func:`write_operator_config` 가 입력 sig 없이 저장하면 ``ValueError`` 를 낸다.

### 기존 연산자 출처 복원

`fit_inputs.json` 77 개 1,943 참조가 **100% sig 를 보유**한다(경로가 깨진 672 건 포함).
경로를 버리고 sig 만 옮기면 **깨진 672 건이 그대로 살아난다.**

sig 가 없는 더 옛 라운드 연산자는 입력을 복원할 수 없다. `operator_inputs` 에 행을 만들지 않고
`operators.provenance = unknown` 으로 표시한다. **있는 척하지 않는다.**

### 함께 이관할 것

- `_hostcopies/` — 머신 간 "같은 경로 다른 내용" 1,039 건 격리분
- `_hostcopies/REPLICATES.tsv` — 449 판 정본↔재실행 대조표
  (집계 SR 0.5947 vs 0.5880, 개별 판정 12.7% 뒤집힘)

---

## 8. 금지 사항

- 산출물 안에 **절대경로 기록 금지** (§6-1)
- 기존 파일 **덮어쓰기 금지** — 충돌 시 에러 중단 (§2)
- 수집 rollout 과 평가 rollout **혼재 금지** (§1)
- 정본 arm 디렉토리에 머신 재실행분 **병합 금지** — 일부 분석 코드가 `rglob` 을 쓰므로
  이중 계수된다 (`_hostcopies/README.txt` 참조)
- pkl 내부 축(layer/step/token)을 **파일로 쪼개지 않는다** — 파일 수 폭증

---

## 9. 관련 문서

- [`docs/01_serving_interface.md`](01_serving_interface.md) — 통일 API 규격
- [`docs/cache_paths.md`](cache_paths.md) — 체크포인트·데이터셋 cache 경로
- `_hostcopies/README.txt` (승준 아카이브) — 머신 재실행분 취급 규칙
