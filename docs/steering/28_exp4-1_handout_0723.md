# exp4-1 현황 핸드아웃 (2026-07-23 12:30 실측)

이어받는 세션은 이 문서 + `docs/steering/24a`(계획) + `24`(공유)만 읽으면 자기완결이다.
아래 수치·경로는 전부 이 시각 실측이며, 추정은 "추정"으로 표기했다.

---

## 0. 한 줄 요약

주석·연산자 fit은 **전부 완료**. 07-23 오후에 발견된 **setM 토큰 공간 배선 버그**로 그 전에
돈 rollout은 전량 폐기했고, 연산자를 **세그먼트형(v2)** 으로 재설계해 재fit 중이다.
**남은 일 = 재fit 마무리 → rollout 재실행 → 감사 → 집계**. 사용자 대기 항목은 없다.

---

## 1. git 상태 (실측)

| 항목 | 값 |
|---|---|
| 본 트리 브랜치 | `exp/exp4-1-oracle-rescue` (정상) |
| 로컬 HEAD | `3f56e39` (LOO phase-registry) |
| origin/exp4-1 | `3f56e39` — **동기 완료** (12:28 push) |
| worktree | exp4-2(`exp/exp4-2-induced-failures`), exp4-3(`exp/exp4-3-atlas`), patching-ceiling |

- 한때 본 트리가 exp4-3 브랜치로 전환돼 있었으나 exp4-3 세션이 자기 worktree로 빠지면서 복귀됨.
- origin/exp4-1 에는 **동료의 Event-SAE 커밋 `967f993`** 이 포함돼 있다(내 커밋들이 그 위에 얹힘).
  잘못 들어온 커밋이지만 남의 작업이라 건드리지 않았다. 제거 여부는 사용자 판단.
- 원격 repo 브랜치: 승준=`exp4-1`(967f993 시점), srv48/srv50=`exp4-1`(7a415dc 시점)
  → **rollout 재실행 전 세 호스트 모두 `git fetch && git reset --hard origin/exp/exp4-1-oracle-rescue` 필요**.

---

## 2. ★ 07-23 배선 버그와 교정 (이번 라운드의 핵심)

### 2.1 무엇이 잘못됐나
- fit(`fit_mean_diff.py`)은 **49토큰 평균 공간**에서 방향 r̂ 과 스칼라 setpoint s 를 산출.
- serve(`SetpointSteering`)는 기본 `token_select=last_horizon` 으로 **action 16토큰에 per-token** 적용.
- 러너가 `--steering-token-select` 를 넘기지 않아 exp3 의 `all` 정렬이 누락된 회귀.
- 결과: pooled s(−132)를 action 사영(−256)에 강제 → β=1 이 **+4.1σ**(beer) 이동.
  `token_select=all` 로만 바꿔도 drawer_right state 가 **−6.7σ** 라 근본 해결이 아니었다.

### 2.2 진단 (rollout 0판, CPU) — `diag_token_space_*.json`
세그먼트 = state[0:1] / future[1:33] / action[33:49] (T=49).

| cell | state AUROC(z) | future AUROC(z) | action AUROC(z) | 세그먼트 방향 cos |
|---|---|---|---|---|
| drawer_left L2 | .691 (2.80) | **.715 (4.81)** | .626 (4.32) | st·fu −0.12, fu·ac +0.07 (≈직교) |
| drawer_right L4 | .628 (2.11) | **.731 (3.60)** | .624 (2.95) | 전부 ≈0 (직교) |
| beer L10 | .770 (2.97) | **.813 (3.28)** | .538 (**0.69**) | st·fu +0.69, ac +0.25~0.29 |
| bread L10 | .716 (3.96) | **.754 (4.04)** | .625 (3.33) | st·fu +0.73, ac +0.17~0.21 |

- **future 가 4/4 cell 최강** (WA-LQR 이 world 슬롯만 steer 한 설계와 정합).
- action 무신호는 **beer 한정** — "action=음성 대조군" 전제는 성립하지 않음.
- drawer 는 세그먼트 방향이 서로 **직교** → pooled 방향 하나로는 어느 세그먼트도 못 겨냥.

### 2.3 s_t(토큰 위치별 setpoint) 정당성 — `diag_st_sig_*.json`
초기 판단은 record 단위 SE(잘못)로 했고, 대시보드 지적대로 **episode-cluster bootstrap(B=1000)** 재검정:

| cell | 성공 ep | future 유의 편차 | action 유의 편차 | record-SE 과소배수 |
|---|---|---|---|---|
| drawer_left | 17 | 27/32 (84%) | 5/16 (31%) | 2.2× |
| drawer_right | 16 | 27/32 (84%) | 1/16 (6%) | 3.3× |
| beer | 19 | 27/32 (84%) | 13/16 (81%) | 1.6× |
| bread | 23 | 28/32 (88%) | 13/16 (81%) | 1.0× |

→ **s_t 유지 확정** (future 는 4/4 cell 에서 84~88% 토큰이 세그먼트 평균과 유의하게 다름).

### 2.4 교정된 연산자 (v2)
- **방향 = 세그먼트별 r̂_seg [S=3, D=1536]**, **setpoint = 토큰 위치별 s_t [T=49]**.
- 적용: `h'_t = h_t − β[(h_t·r̂_seg(t)) − s_t]·r̂_seg(t)`, `seg_mask=0` 인 세그먼트는 무개입.
- NPZ v2 키: `alpha0_v_seg` · `alpha0_s_tok` · `alpha0_seg_bounds` · `alpha0_seg_mask`.
- serve: `SetpointSteering.set_segment/_apply_segment`. **T 불일치 시 RuntimeError**,
  `setpoint_seg` 는 `--steering-token-select all` 강제(미지정 시 기동 abort).
- 배포 게이트: 이동/갭 중앙 ≤ 3.0. 실측 **1.00~1.45** 통과 (구 배선 4.1σ → 정상화).
- 구 pooled 산출물은 `_v1_pooled_*` 로 보관(배포 금지).
- 검증: hook 수식 numpy 대조 2.4e-7, future 사영이 s_t 도달, state/action bitwise 불변.

---

### 2.5 ★ 위약 선택 공간 교정 (07-23 오후, 배선 버그와 같은 층위)

배포 연산자는 v2 세그먼트형인데 **위약 순열 선택 기준만 구 pooled(49토큰 평균 방향·스칼라 s)로
남아 있었다.** pooled |cos| 가 통과해도 실제 개입 축에서는 준직교가 깨진다:

| cell | 교정 전 cos_seg [state, future, action] | pooled cos(구 기준) | 교정 후 |
|---|---|---|---|
| drawer_left | **−0.55, −0.46, +0.44** | −0.03 통과 | +0.03, −0.07, +0.07 |
| mixer | **−0.43**, −0.13, **+0.35** | −0.01 통과 | −0.24, −0.02, −0.24 |
| bread | +0.15, +0.10, **+0.41** | +0.18 통과 | −0.22, −0.19, −0.13 |
| beer | −0.22, −0.19, +0.16 | −0.11 통과 | +0.06, +0.04, −0.10 |
| drawer_right | −0.03, +0.03, +0.18 | +0.08 통과 | +0.10, +0.06, −0.29 |

gated 위약은 더 심했다 (bread 3 phase 전부 위반: 0.40 / 0.34 / 0.41).
LOO 위약도 전체fit 순열을 물려받아 재fit 하는 구조라 부분표본에서 회전
(**bread 0.72** · drawer_right 0.47 · drawer_left 0.36 · beer 0.30).

**dose 문제**도 겹쳤다. 순열 방향은 실제 클래스 갭이 없어 이동량이 구조적으로 작다 —
bread 는 200 순열 중 밴드(±25%) 통과가 **0개**여서 위약이 처치의 65% 세기로만 개입했다
(위약에 유리한 비대칭).

**교정 규약 (현행)**
1. 후보마다 세그먼트 연산자를 만들어 `max_s |cos(r̂_seg^pl, r̂_seg^treat)| ≤ 0.3`.
   후보 없으면 순열 풀 200 → 1000 확장, 그래도 없으면 최소-cos 폴백(metadata 기록).
2. dose 는 밴드 필터 대신 **세그먼트별 스케일** `scale_s = dose_s^treat / dose_s^pl` 을
   `alpha0_seg_mask` 에 실어 처치와 **정확히** 매칭. hook 이 mask 를 float 승수로 쓰므로
   serve 변경 불필요. 스케일은 [0.5, 2.0] 클립(클립 시 기록 — bread future 2.01→2.00).
3. permanent · gated(phase별) · LOO(대상별) **전부 독립 선택**. pairing 은 episode 단위라 유지.
4. 선택 경로(episode 토큰합 float64)와 재fit 경로(float32 누적)의 일치 검사:
   Δv ≤ 2e-3 · Δs/|s|max ≤ 2e-3 (실측 Δv~1e-4, min cos=1.000000). 라벨 1개만 뒤집어도
   Δv=0.32 로 검출됨을 자체검증.

검수 도구: `steer/exp4_1/inspect_npz.py` (처치·위약·LOO·gated 일괄, 위반 시 exit 1).

**처치 연산자는 이 교정과 무관** — 재fit 전후 배열 sha256 **비트 동일**(5 cell × 2 arm 확인)
이라 교정 전에 돌린 처치 rollout 은 그대로 유효하다.

---

### 2.6 conceptor 길이통제 정합 (07-23 저녁, 사용자 지적)

"conceptor gated 가 phase 별로 길이제어하는 게 setM 이랑 다른가?" → **달랐다. 아예 없었다.**

| | 길이통제 |
|---|---|
| `setM_gated` | phase 별 성공 dwell 의 `ceil(μ+1σ)` (`phase_dwell_caps`) |
| `setM_permanent` | 성공 episode record 수의 `ceil(μ+1σ)` |
| `conceptor_*` (구) | **없음** — `phase_records()` 가 "raw, truncate 전", phase 내 전 record pool |

실패는 phase 안에 오래 머무는 경향(timeout 계열)이라 무제한 pooling 은 실패 공분산을 긴 dwell 이
지배한다 — succ/fail 대비에 길이가 섞인다.

**교정**: `fit_phase_conceptor_n15.py --length-control` (`compute_length_caps` → `_roll_records`
가 cap 만큼만 반환). setM 과 같은 규약이고 global(=permanent)도 함께 통제. `fit_inputs.json` 에
`length_control`·`length_caps` 기록. 미지정 시 구 동작 유지.

실측 cap (setM 과 동일값):
| cell | global | phase 별 |
|---|---|---|
| bread | 50 | — |
| beer | 77 | reach 10 · grasp 22 · transport 4 · place 24 · insert-settle 29 |
| drawer_left | 51 | reach-to-handle 27 · grasp-handle 16 · pull 13 · push-back 3 |
| drawer_right | 76 | (동일 절차) |
| mixer | 62 | reach-to-head 14 · contact-head 20 · push-down 71 · lift-open 16 · disengage 16 |

교정 전 연산자로 시작하려던 beer eval `conceptor_permanent` run 은 중단·삭제(0판)했고,
5 cell 전부 재fit 후 재배포했다.

---

## 3. arm 정의 (9종, 실행 순서 = 사용자 확정)

| # | arm | 개입 | fit-풀 |
|---|---|---|---|
| 1 | `A0` | 없음 | ✗ (fit30 수집 자체가 무개입) |
| 2 | `noise_resample` | t0부터 denoise seed +500000 (방향 없음) | ✓ |
| 3 | `setM_permanent` | t0부터 끝까지 세그먼트 연산자 | ✓ (LOO) |
| 4 | `setM_permanent_placebo` | 준직교 순열 위약 | ✓ (LOO) |
| 5 | `setM_gated` | t0 이후 **현재 phase** 연산자 | ✗ (LOO 없음) |
| 6 | `setM_gated_placebo` | phase별 독립 위약 | ✗ |
| 7 | `setM_future_only` | 3과 동일 연산자, mask=[0,1,0] | ✓ (LOO, 비용 0) |
| 8 | `setM_future_only_placebo` | 4와 동일, mask 적용 | ✓ (LOO) |
| 9 | `conceptor_permanent` / `conceptor_gated` | exp3 legacy 참조선 (β0.1, per_step, all) | ✗ (LOO 없음) |

- **fit-풀에서 conceptor·gated 금지** = in-sample 평가 차단(러너가 exit 2로 거부).
  전례: `multilayer-conceptor-steering-works`(+0.20 → held-out −0.067 null).
- 위약 규약: |cos(위약, 처치)| ≤ 0.3 준직교 + dose-match ±25%. gated 는 **phase별 독립 선택**
  (동결 순열 재사용은 phase 부분공간에서 준직교가 깨짐 — 실측 cos +0.62/−0.71).

---

## 4. 데이터 자산 (실측 경로: `outputs/eval/robocasa/groot_n15/exp4_1/`)

### 4.1 주석 (전부 완료, 동결)
| 파일 | 판수 | 상태 |
|---|---|---|
| `annotation_t0.tsv` | 149 (eval 107 + fit 42*) | 149/149 주석 |
| `annotation_t0_mixer.tsv` | 16 (eval 7 + fit 9) | 16/16 주석 |
| `t0_manifest_v4.tsv` | legacy 동결본 (sha 58737fb5) | **이걸 쓸 것** |
| `t0_manifest_mixer_v1.tsv` | mixer 동결본 (sha 0b9fcbdb) | **이걸 쓸 것** |

\* beer fit 은 오염 3판(ep4/13/14) 제외 후 8판 → legacy fit 총 42, 전체 구제 대상 **165판**
(eval 114 + fit 51). v1~v3 은 중간 스냅샷(불변 검증 통과분), v4 가 최신.

### 4.2 제외 기록
- `corrupted_obs_excluded.json` — beer fit ep4/13/14 (렌더 손상 + DiT norm 외곽치 z −2.9/−2.6/−2.2)
- `mixer_nondeterministic_excluded.json` — mixer 5판 (재현 시 fail→succ 뒤집힘, 21→16)
- `mixer_feasibility.json` (95/100 feasible), `drawer_feasibility.json` (94/94 통과)

### 4.3 연산자 NPZ — `npz/<cell>/`
5 cell: `pq3_ppcc_bread`, `pq3_ppcc_beer`, `pq3_drawer_left`, `pq3_drawer_right`, `exp41_mixer`

배포 layer (setM = 분리도 기준, conceptor = 분산 기준):
| cell | setM layer | conceptor layer |
|---|---|---|
| bread | L10 | L15 |
| beer | L10 (오염 제외 후 L12→L10) | L4 |
| drawer_left | L2 | L15 |
| drawer_right | L4 | L4 |
| mixer | L15 | L0 |

conceptor 는 **전 cell·전 layer 퇴화**(|z|<2, mixer 최대 0.04) — legacy 참조선 역할.

setM_gated 등록 phase (quota: record ≥50·episode ≥3/클래스, 미달은 **미등록=무개입**):
- bread: reach(z −0.24)·grasp(0.96)·**place(2.17)** / transport·insert-settle SKIP
- beer: reach·grasp·insert-settle·**place(2.22)** (dwell-cap 적용 후) / transport SKIP
- drawer_left: **reach-to-handle(3.27)**·**grasp-handle(3.55)** / pull·disengage·push-back SKIP
- drawer_right: **reach-to-handle(3.56)**·**grasp-handle(4.38)** / 나머지 SKIP
- mixer: reach-to-head(0.87)·contact-head·lift-open / push-down·disengage SKIP

### 4.4 rollout 현황 (07-23 저녁 기준)
**폐기분에서 복원**: `noise_resample` 은 steering 을 전혀 쓰지 않는 arm(사이드카
`steering_npz=null`·`serve_steering=null`, 개입은 `--reseed-from-record` 뿐)이라 토큰 배선
버그와 무관한데 폐기 배치에 함께 옮겨져 있었다. audit 전수 통과(beer eval 16/16, mixer 7/7)
확인 후 원위치 복원 — 사유는 `eval/RESTORED_noise_resample_0723.md`.

| cell·풀 | A0 | noise_resample | setM_permanent | setM_future_only |
|---|---|---|---|---|
| drawer_left fit(13) | 13 | 13 | 13 ✔ | 진행 |
| drawer_right fit(14) | 14 | 14 | 14 ✔ | 진행 |
| bread fit(7) | 7 | 7 | 7 ✔ | 7 ✔ |
| beer fit(8) | — (fit30 수집 자체가 무개입) | 대기 | 대기 | 대기 |
| beer eval(16) | — (exp3 ho_base 실패분) | 16 | 16 ✔ | 16 ✔ |
| mixer eval(7)·fit(9) | — | 7(eval) | 대기 | 대기 |

**폐기 유지** — `eval/_invalid_tokenwiring_0723/`: beer eval 6arm×16, mixer 6arm×7
(conceptor 는 `token_select` 정렬도 어긋나 함께 폐기), legacy fit-풀의 구 setM arm.

---

## 5. 실행 인프라

### 5.1 머신 귀속 (결정론은 머신-로컬 — 반드시 지킬 것)
| 대상 | 머신 |
|---|---|
| eval-풀 bread·drawer 좌우 | **srv48** (junhyeong@166.104.35.48) |
| eval-풀 beer | **srv50** (junhyeong@166.104.35.50) |
| eval-풀 mixer | **srv50** (신규 cell, srv50 귀속) |
| fit-풀 전체(legacy) | **kanu** (로컬 docker) |
| fit-풀 mixer | **srv50** |
| fit·진단 CPU | 승준 (kimseungjun@166.104.146.37:11112) |

### 5.2 GPU 규칙 (위반 시 질책 이력 있음)
- **타인 프로세스가 하나라도 있는 GPU 사용 금지.** 완전히 빈 GPU(≈0MiB·프로세스 0)만.
- kanu: 전 세션 합산 **GPU 3개 · GPU당 serve 2개**. srv48/srv50: 서버당 **GPU 1개 · serve 6개**.
- 발사 전 `nvidia-smi --query-compute-apps` 로 **소유자까지** 확인.
- 12:30 실측 여유: kanu GPU 2·3·4·6·7 / srv50 GPU3 / **srv48 없음**.

### 5.3 실행 커맨드
```bash
# eval-풀 (srv48/srv50, host serve 6)
CELL_ID=<cell> ARM=<arm> POOL=eval \
T0_MANIFEST=outputs/eval/robocasa/groot_n15/exp4_1/t0_manifest_v4.tsv \
NPZ_ROOT=outputs/eval/robocasa/groot_n15/exp4_1/npz \
OUT_ROOT=outputs/eval/robocasa/groot_n15/exp4_1/eval \
SERVE_MODE=host GPUS_L="<G> <G> <G> <G> <G> <G>" PORTS_L="8620 8621 8622 8623 8624 8625" \
bash scripts/safe/groot_n15/robocasa/steer/exp4_1/run_oracle_rescue.sh

# fit-풀 (kanu, docker serve 2/GPU) — POOL=fit, 나머지 동일. LOO 는 자동으로
# <arm>_loo/ep{E} phase registry 를 쓰므로 serve 재기동 없음.
```
- 체인 스크립트: `/tmp/exp41_cell_eval_chain.sh <CELL> <GPU>` (9 arm 순차, 각 서버에 배포됨)
- mixer 용: `/tmp/exp41_mixer_chain.sh exp41_mixer <GPU>` (mixer manifest 사용)

### 5.4 감사·집계
```bash
python scripts/safe/groot_n15/robocasa/steer/exp4_1/audit_flags.py \
  --t0-manifest <manifest> --arm-root <eval/<cell>/<arm>> --arm <arm>   # 0 = 통과

python scripts/safe/groot_n15/robocasa/steer/exp4_1/aggregate_rescue.py \
  --pool eval --t0-manifest <v4> --t0-manifest <mixer_v1> \
  --arm <name>:<root> ... --setm-npz-root <npz> --out <json>
```
- **`--pool eval`(primary, 완전 held-out) / `--pool fit`(secondary, LOO) 분리 — 합산 금지.**
- primary contrast: `setM_permanent vs placebo`, `setM_gated vs placebo` (각각 4-task Holm).
- 보고 전 `confound-audit` skill 경유.

---

## 6. 지금 실행 중 / 다음 할 일

**실행 중 (07-23 13:20 UTC)**
- 승준: 위약 교정 반영 재fit 3회차 (`/tmp/remote_compute_logs/refit_r3.log`) —
  permanent+LOO 5 cell → gated 5 cell 순차. LOO 위약이 대상마다 독립 선택되면서
  cell 당 5~8분.
- srv50 GPU3(serve 6): beer eval — 처치 2종 완주(16+16) → `setM_permanent_placebo` 진행 중
  → `setM_future_only_placebo`. conceptor NPZ(beer·mixer) 스테이징 완료.
- kanu GPU 2·3·4(각 serve 2): fit-풀 — drawer 좌/우 `setM_future_only` 진행,
  bread 완주 후 beer fit-풀로 이동.
- srv48: 4 GPU 전부 타인 점유 → bread·drawer **eval**-풀(91판) 대기.

**다음 순서**
1. refit 완료 확인 → `rsync` 로 `npz/` 회수 → 검수 (세그먼트 갭·이동/갭 ≤3·위약 cos ≤0.3·LOO 판수)
2. 세 호스트 git 동기 (`reset --hard origin/exp/exp4-1-oracle-rescue`) + npz·manifest 스테이징
3. rollout 재실행
   - srv50 GPU3: beer(9 arm×16) → mixer(9 arm×7) → mixer fit-풀(5 arm×9)
   - srv48: GPU 나면 drawer_left→drawer_right→bread (9 arm × 각 38/29/24)
   - kanu 빈 GPU: legacy fit-풀 (5 arm × 42판, A0·noise_resample 은 이미 있음 → 3 arm 만)
4. arm별 `audit_flags.py` 전수 통과 확인 (LOO 는 `steer_phase_name==ep{E}` 검사 포함)
5. `--pool eval` / `--pool fit` 집계 → confound-audit → 결과 문서 (번호는 작성 시점 dev 확인)

**주의점**
- 재실행 전 폐기 디렉토리와 섞이지 않게 `eval/<cell>/<arm>/` 이 비어 있는지 확인(resume 이
  기존 산출을 건너뛴다).
- `--steering-token-select all` 이 러너에 이미 박혀 있으나, 수동 실행 시 빠뜨리면 serve 가
  abort 한다(정상 동작).
- 결과 해석: 배선 버그 이전 beer 완주분에서 **noise_resample 7/16 · setM_permanent 6/16 ·
  위약 5/16** 이었다 — 무방향 개입만으로도 상당한 구제가 나오므로 위약 대조가 판정의 핵심.
