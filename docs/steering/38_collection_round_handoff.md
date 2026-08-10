# 재수집 라운드 핸드오프 — instruction × scene × noise 그리드

**이 문서는 데이터 수집 세션의 진입점이다.** 2026-08-05 작성.

저장 규약은 [`docs/04_data_storage_convention.md`](../04_data_storage_convention.md) 이 단일
출처다. **수집 스크립트를 만지기 전에 반드시 읽는다** — 좌표 기반 레이아웃(§3.1), `<machine>`
층의 근거(§3.2), base·arm 짝(§3.3)이 이번 라운드의 전제다.

---

## 0. 왜 다시 모으는가

기존 activation 686 판은 캡처 밀도(`[7,4,49,1536]`)는 기준을 만족했지만 **`machine` 이
23% 만 기록**돼 있었다. 2026-08-05 실측에서 머신이 다르면 hidden state 원소의 93% 가 갈리고
개별 판정이 12.7% 뒤집히는 것이 확인됐다(§3.2). 머신을 모르면 이 효과를 층화할 수 없다.

소급 복원을 시도했으나 526 판은 `MACHINE.txt` 자체가 없어 실패했다. 그래서 전량 폐기하고
좌표·머신이 처음부터 박히는 구조로 다시 모은다. 2026-08-05 기준 store 는 61G(체크포인트·
runs·index 만)이고 HDD 여유는 **1.7T** 다.

---

## 1. 확정된 것

### instruction 8 종 (2026-08-05 사용자 확정)

| # | instruction | env | 필터 |
|---|---|---|---|
| 1 | `Pick the bread from the counter and place it in the cabinet.` | `PickPlaceCounterToCabinet_PandaOmron_Env` | 물체=bread |
| 2 | `Pick the apple from the counter and place it in the cabinet.` | 〃 | 물체=apple |
| 3 | `Pick the candle from the counter and place it in the cabinet.` | 〃 | 물체=candle |
| 4 | `Open the left drawer.` | `OpenDrawer_PandaOmron_Env` | side=left |
| 5 | `Open the right drawer.` | 〃 | side=right |
| 6 | `Pick the mug from the counter and place it under the coffee machine dispenser.` | `CoffeeSetupMug_PandaOmron_Env` | 없음 |
| 7 | `Fully slide the top dishwasher rack out.` | `SlideDishwasherRack_PandaOmron_Env` | direction=out |
| 8 | `Fully slide the oven rack out.` | `SlideOvenRack_PandaOmron_Env` | out + **층 구문 없음**(단단 오븐) |

전부 env 이름 앞에 `robocasa_panda_omron/` 를 붙인다.

### seed → instruction 스캔 결과 (seed 100000–100299, 300 판, 에러 0)

산출물: `outputs/analysis/seed_scan/<Task>.tsv` (seed \t instruction).
도구: `scan_seed_instructions.py` (브랜치 `feat/seed-scan-probe-tools`, 정책·GPU 불필요).

| instruction | 300 seed 중 | 비율 | scene 10 개에 필요한 seed 범위 |
|---|---|---|---|
| PPCC bread | 4 | 1.3% | ~750 |
| PPCC apple | 5 | 1.7% | ~600 |
| PPCC candle | 6 | 2.0% | ~500 |
| OpenDrawer left | 138 | 46% | 300 으로 충분 |
| OpenDrawer right | 162 | 54% | 〃 |
| CoffeeSetupMug | 300 | 100% | 〃 |
| DishwasherRack out | 154 | 51% | 〃 |
| OvenRack out (단단) | 53 | 18% | 300 으로 충분 |

PPCC 3 종은 물체가 96 가지로 흩어져 있어 **seed 범위를 넓혀야 한다**. 위 스캔을 그대로
`--seed-start 100300 --seed-end 101000` 으로 이어 돌리면 된다.

⚠ **OvenRack 주의**: "층 구문 없음"은 kitchen layout 이 단단 오븐일 때만 나온다(300 중 104 =
35%). 이 instruction 만 **layout 이 특정 종류로 편향**되므로 scene 다양성이 다른 7 종과 다르다.
분석에서 이 점을 각주로 달아야 한다.

---

## 2. 미정 — 수집 시작 전에 정해야 할 것

### 2.0 ★★ 라벨러 커버리지 블로커 (2026-08-06 발견)

**8 종 중 3 종은 phase 라벨러가 지원하지 않는다** — `src/collect/robocasa/event_labeler.py`
의 `make_robocasa_event_labeler` 는 (StandMixer / Fridge / Drawer / `TASK_EVENTS` 등록 PnP)
만 처리하고, 그 외는 `lookup_task_events` 가 **KeyError** 를 던진다. n15 허브는 라벨러를
옵션 없이 무조건 생성하므로(`http_feature_collect.py` 라벨러 생성부) 해당 instruction 은
**수집 첫 에피소드에서 즉사**한다.

| instruction | 라벨러 |
|---|---|
| PPCC bread/apple/candle | ✓ (`PickPlaceCounterToCabinet` 등록) |
| OpenDrawer left/right | ✓ (Drawer 분기) |
| CoffeeSetupMug | ✗ 미등록 |
| SlideDishwasherRack out | ✗ 미등록 |
| SlideOvenRack out | ✗ 미등록 |

선택지: (a) 3 종 라벨러 구현(이벤트 정의 + `TASK_EVENTS`/분기 등록 + 테스트),
(b) 라벨러 opt-out 배선 후 3 종은 phase 라벨 없이 수집(사후 라벨링·`env_step_gt_retro`
계열로 소급 가능하나 라벨러 구현이 선행돼야 하는 건 동일), (c) task 교체.

### 2.1 ★ SR 파일럿 (최우선)

**`m`(noise 수)은 SR 에서 역산해야 한다.** scene 을 고정하면 succ/fail 을 가르는 건 policy
샘플링뿐이고, conceptor·setM fit 은 한 scene 안에 두 클래스가 다 있어야 성립한다.

- SR 0.5 → m=30 이면 소수 클래스 15 판 (충분)
- SR 0.7/0.3 → 9 판 (가능)
- SR 0.85/0.15 → 4 판 (**fit 위태**)

exp2 에서 "고SR scene 은 fit 창에 실패 2~6 판" 때문에 대조가 불성립한 전례가 있다.

**할 일**: 8 instruction × 20 판으로 SR 을 먼저 재고, 극단적인 것은 `m` 을 늘리거나 task 를
교체한다. GPU 필요, 반나절 규모.

### 2.2 그리드 크기

사용자 선호는 **scene 5–10**. 저장 예산(1.7T, 7층 432MB/판):

| 그리드 (8 instr) | 판수 | 7층 | 12층 | 16층 |
|---|---|---|---|---|
| n=10 × m=20 | 1,600 | 691G | 1.19T | 1.58T |
| **n=10 × m=30** | **2,400** | **1.04T** | 1.78T ✗ | ✗ |
| n=10 × m=50 | 4,000 | 1.69T (여유 0) | ✗ | ✗ |

**제안: n=10, m=30, 7 층** — 1.04T, 여유 700G. 근거는 (a) 사용자 선호 범위 상단, (b) n=10 이면
leave-one-scene-out 10-fold 가 되고, (c) 과거 분석이 층을 다 쓴 적이 없는 반면(주로 L0·L2·L8·
L10·L12·L15) 실패판 부족으로 fit 이 죽은 사례는 여러 번이다.

⚠ **판당 432MB 는 기존 686 판의 평균**이다. task 별 편차가 크다(drawer 565MB, mixer 267MB) —
`plan.estimate_bytes(records_per_rollout=...)` 에 실제 길이를 넣어 재확인할 것.

### 2.3 층 집합

`capture_layers` 는 **사후 변경이 불가능**하다(층 추가 = 전량 재수집). `m` 추가는 부분
재수집으로 되지만 층은 안 된다. 기존 표준은 `0,2,4,8,10,12,15` (7 층).

### 2.4 scene seed 선정

스캔 TSV 에서 instruction 이 맞는 seed 를 뽑되, **기하 실현가능성 필터를 함께 적용**한다
([`SCENE_FEASIBILITY.md`](SCENE_FEASIBILITY.md)) — mixer 100010 처럼 정책과 무관하게 성공
불가능한 fixture 배치가 존재한다. fit 과 eval 양쪽에서 동일하게 제외해야 한다.

---

## 3. 실행 환경

### 3.1 머신·GPU 실측 (2026-08-05)

| 머신 | 접속 | GPU | 여유 | serve/GPU |
|---|---|---|---|---|
| **kanu** (로컬) | — | A4000 16GB × 8 | GPU0·4 비어있음 | **1** (serve 13.3GB) |
| **srv50** | `junhyeong@166.104.35.50` | A100 80GB × 4 | GPU0·1·2 비어있음 | **6** |
| **srv48** | `junhyeong@166.104.35.48` | A100 80GB × 4 | 전부 사용중(35~74GB) | 여유 확인 필요 |
| **pdk_external** | `-p 11115 dongkyu@166.104.44.23` | RTX 4070 Ti SUPER 16GB × 1 | GPU0 비어있음 | **1** |

규칙 (CLAUDE.md 평가 표준 + `robocasa-steer-eval` 스킬):

- **GPU 양보 default**: 비어있는 것 중 **최대 3 GPU** 까지 사용.
- **한 GPU 에 여러 serve 시도**: GR00T 는 A100 80GB 에 6 개, 16GB 카드엔 1 개.
- **kanu 는 GPU 4/5/6 만** — GPU 0–3 은 동료 예약. (실측상 4 만 비어있음)
- 발사 직전 `nvidia-smi` 로 소유자 재확인 — 점유가 수시로 바뀐다.

### 3.2 ★ 머신 배정 규칙 (신규 — docs/04 §3.2)

**한 좌표의 base 와 arm 은 반드시 같은 머신에서 돈다.** 머신이 다르면 hidden state 93% 가
갈리고 paired 판정에 머신 효과가 섞인다(12.7% ≈ exp3 위약 요동과 동급).

**GPU 는 좌표가 아니다.** 같은 머신이면 GPU 가 달라도 bitwise 동일하므로 **한 칸을 여러 GPU 에
자유롭게 병렬 배정**할 수 있다. 병렬화는 GPU 축에서 하고, 머신 축에서 하지 않는다.

권장 배정: instruction 단위로 머신을 통째로 할당한다(예 srv50 = PPCC 3 종, kanu = drawer 2 종).
그러면 한 instruction 안의 모든 좌표가 자동으로 같은 머신이 된다.

### 3.3 실행 패턴

serve (호스트 conda 또는 docker `lerobot`):

```bash
setsid nohup "$PY" scripts/serve/lerobot.py \
  --profile configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
  --host '*' --port "$p" --device cuda \
  --groot-dit-token-pool all_token_full --groot-dit-capture-layers 0,2,4,8,10,12,15
```

수집 (docker `robocasa` 컨테이너):

```bash
docker exec -e PYTHONPATH="$PYP" robocasa \
  python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
  --seed "$S" --inference-seed "$inf" --n-action-steps 5 --max-episode-steps 720 ...
```

참고 구현: `scripts/safe/groot_n15/robocasa/steer/exp5_3/*.sh` (포트 8600–8602 패턴).

⚠ **장시간 run 은 `setsid nohup` 으로 띄운다** — agent 백그라운드 job 은 harness 가 중간에
kill 해 trap cleanup 이 빈 결과 + 가짜 `[done]` 을 남긴다. 완료는 **결과 행수**로 판정한다.

---

## 4. 코드 상태

### 배선 완료 (2026-08-06 갱신)

| 항목 | 위치 |
|---|---|
| `machine`·`ckpt` 를 serve → 수집기로 | `src/utils/common/serving.py` 의 `serve_provenance()`; HTTP(`serve/lerobot.py`)·ZMQ(`groot_n16/.../feature_server.py`) 양쪽 |
| 연산자 `config.json` 강제 | `src/utils/operator_config.py` — 입력 sig 없이 저장하면 `ValueError` |
| fit 스크립트 배선 | `fit_phase_conceptor_n15.py`, `exp4_1/fit_mean_diff.py` |
| **좌표 계획·경로·armsig 단일 출처** | `src/collect/plan.py` (구 `src/utils/collection_plan.py`) — `CollectionPlan`·`GridCell.rel_path`·`resolve_grid`·`grid_dir_for`·`arm_signature` |
| **수집기 좌표 배선** | n15 `http_feature_collect.py`·n16 `collect_rollout.py` 둘 다 `--grid-root/--plan-json/--scene-idx/--noise-idx` 수용 |
| **`grid_dir` 필수화** | `src/collect/artifacts.py` `write_safe_triplet` — 좌표 없이 부르면 RuntimeError(§8). §2 쓰기 검사(동일 skip / 상이 에러) 포함 |
| **수집 공용 부품 `src/collect/` 승격** | artifacts·schema·policy_clients·라벨러(robocasa/libero)·step_phase — sys.path 해킹 제거, 테스트 63 passed |
| **eval 좌표화 (2026-08-06)** | n15 허브 `--no-features` + 좌표 인자 → `write_eval_artifacts` 가 arm 디렉토리에 `meta.json·traj.csv·video.mp4·config.json`. armsig 는 serve `/health` steering 지문 + 클라이언트 latch/gating 인자에서 자동 계산(`_resolve_arm`). base(무개입) eval 은 수집 rollout 이 그 자체(동일 머신·seed → bitwise 동일)라 금지 |

### 손봐야 하는 것

1. ~~serve_provenance() GPU 포함~~ — **완료 확인(08-10)**: hostname 만 반환, GPU 는
   serve_gpu 감사 필드로 분리. 규약 주석까지 반영돼 있음.
2. **수집 러너 `.sh` 가 좌표 인자를 안 넘긴다** — 지금 구 러너를 그대로 돌리면
   `grid_dir 없이 수집할 수 없다` 로 **즉시 실패**한다(의도된 동작). 이번 라운드 러너는
   `collection_plan.json` 기반으로 새로 쓴다.
3. **인덱서** — `arm_bindings.tsv` 신설(docs/04 §3.3)과 좌표 기반 `rollouts.tsv` 복합키가
   미반영. 구 인덱서(`~/index_build2` 계열)는 sig 평면 전제라 그대로 못 쓴다.
3-1. ~~capture-ON arm 수집의 `config.json`~~ — **완료(2026-08-06)**: `write_safe_triplet`
   에 `arm_config` 배선, 허브가 steered 수집이면 자동으로 넘긴다. base 는 안 쓴다.
4. **seed 스캔 도구 브랜치 미병합** — `scan_seed_instructions.py` 는
   `feat/seed-scan-probe-tools` 에만 있다(산출물 TSV 는 로컬 `outputs/analysis/seed_scan/` 에
   실존). PPCC 확장 스캔(100300–101000)을 돌리려면 이 브랜치를 먼저 병합/체리픽할 것.

---

## 5. 함정 (과거 사고 기록)

- **`--profile` 은 단일 체크포인트**: `lerobot_groot_n15__robocasa365_ckpt120000`
  (HF `robocasa/robocasa365_checkpoints`, checkpoint-120000). 다른 걸 쓰면 `ckpt` 열이 달라진다.
- **eval 캡처는 끈다** — exp3 이후 표준. 수집(activation)과 평가(SR)는 저장 위치가 다르다.
- **fit ↔ eval seed 분리 필수**: fit 에 쓴 에피소드로 eval 하면 in-sample rescue 아티팩트가
  나온다(multi-layer +0.20 → held-out −0.067 사건).
- **결과 판정 전 `confound-audit` 스킬**을 돈다 — 길이·scene·instruction confound 상시 점검.
- **정리 후 GPU 반납**: `pgrep -f serve/lerobot.py` → kill, `nvidia-smi` 로 메모리 회수 확인.

---

## 6. 착수 순서 (2026-08-06 갱신)

1. ~~`collection_plan.py` 현황 확인~~ → 완료: `src/collect/plan.py` 로 확정 (§4 배선 완료 표)
2. **라벨러 커버리지 해소** (§2.0) — 3 종 구현 / opt-out / task 교체 결정
3. `serve_provenance()` 에서 GPU 제거
4. 수집 러너 `.sh` 좌표 배선 (§4 손봐야 하는 것 2)
5. **SR 파일럿** (8 instr × 20 판) → `m` 확정
6. PPCC 3 종 seed 범위 확장 스캔 (100300–101000; 스캔 도구 브랜치 병합 선행)
7. scene seed 선정 + 기하 실현가능성 필터
8. `collection_plan.json` 작성 → `plan_id` 확정
9. 머신별 instruction 배정 → 수집 실행
10. 인덱서 갱신 (좌표 기반 + `arm_bindings.tsv`)
