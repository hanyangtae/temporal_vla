# 30. SAE scene-feature 분리 — G1 이식·구현 핸드아웃

작성 2026-07-27 (exp4-1 세션 → 새 SAE 세션 인계). **선행 문서 — 새 세션이 읽을 순서**:
1. **이 문서(30)** 먼저 — 실행판. 나머지는 근거·참조.
2. **왜 SAE 인가** = `docs/steering/25a_exp4-3_to_exp4-1_recommendation.md` (exp4-3 권고, 사용자 확정,
   85줄). ★2026-07-27 exp4-3 worktree 에서 이 브랜치로 복사함 — 새 세션이 바로 읽을 수 있다.
3. **무엇을 이식하나** = `docs/steering/29_sae_port_review.md` (동료 레포 전수 검토, 파일:라인 근거, 228줄).
4. **SAE 설계 지침** = `docs/references/reading_notes/SAE_synthesis_and_design.md` (우리가 쓴 종합).
   3논문 개별 노트 = 같은 폴더 `dr_vla_sae.md`·`event_grounded_sae.md`·`observing_controlling.md`
   (셋 다 outcome-vs-scene 분리를 직접 안 풂 = 니치 확인, DiT 단일-feature steer 붕괴 → 다차원 잔차화 정당).
5. **layer peak 실물** = `docs/steering/sae_g1_refs/n15_atlas_all.tsv` (exp4-3 atlas 복사본).
6. **코드 참조** = `scripts/safe/groot_n15/robocasa/steer/exp4_1/fit_mean_diff.py`
   (`load_cell_rolls` = pkl 로더, truncation/순열 관례) — 전부 tracked, import 가능.

이 문서는 **실행판**이다: 새 세션이 이 문서만 읽고 G1 을 끝까지 갈 수 있게 쓴다.

---

## 0. 한 줄 목표

동료(robots-oh) top-k SAE 코어를 lift 해서, **N1.5 DiT activation(원본 1536-d, per-token)** 에
overcomplete SAE 를 학습하고, **G1 게이트 = "SAE feature 가 scene 을 실제로 인코딩하는가"를
scene 라벨(layout/style/scenario_seed) probe 로 실측**한다. G1 통과 시에만 G2(잔차화 read)로.

**용어**:
- **G = Gate(게이트)**. 25a 사다리의 관문 번호: G1=SAE+scene feature 식별 실측 / G2=scene 잔차화
  후 succ/fail **read** 잔존 / G3=잔여 방향 **write**(steering). 앞 게이트 통과 시에만 다음 착수.
- **overcomplete SAE** = 사전 크기 m > 입력차원 D. superposition(D보다 많은 개념이 겹쳐 저장)을
  희소 활성(top-k)으로 풀어 개념별 feature 를 분리하는 전제 — "scene feature 만 골라 제거"가
  가능하려면 필요. 동료 세팅(PCA64→m=128)은 원본 1536 기준 사실상 undercomplete → 재설계 사유.

## 1. 배경 (3 문단 요약 — 상세는 25a)

선형 latent 연산자 가족(conceptor=분산, setM=평균)은 exp2/exp3/exp4-1 세 라운드에 걸쳐
cross-scene·same-scene·perturbed fit, layer/phase/token/multi 변형 전부 **위약 동급 또는
noise_resample 미만**으로 종결됐다. exp4-3 3-모델 atlas 실측이 이유를 설명한다: 분산축은 모델
불변으로 퇴화(|z|<2), 평균분리는 강하지만(z 5~15) **분리 신호가 최대가 되는 layer 깊이(atlas
mean_z peak)가 모델마다 이동**(같은 bread 인데 N1.5 L8-12 → N1.6 L31 → Cosmos L24)하고 **판별이
가장 강한 phase 는 task 마다 다름**(Cosmos: drawer=grasp-handle, bread=reach-to-object 조기층)
+ write 전부 null = **scene·진행도가 지배하는 결과-상관 구조**(비인과).

따라서 질문을 재정식화한다: **succ/fail 분리신호에서 scene(암기) 성분을 명시적으로 제거하면
outcome 신호가 남는가.** 남으면 그 잔여 방향이 steering 후보(G3), 안 남으면 그 자체가 "latent
steering 서사 종결"의 판정 근거 — 양방향 가치 실험.

사다리: **G1**(SAE + scene feature 식별 실측) → **G2**(scene-잔차화 후 succ/fail read 잔존:
길이통제 dwell cap·episode-라벨 순열 null·held-out) → **G3**(잔여 방향 write, oracle rescue 규약:
위약·noise_resample 동시, fit-seed 분리 held-out, EVAL_SEED=100000). **각 게이트 통과 시에만 다음.**

## 2. 소스 레포 (이식 원천)

`task_classification/` — repo 루트에 클론돼 있음 (dev 핀 커밋 `88543a2`,
https://github.com/robots-oh/task_classification). dev 브랜치에선 서브모듈.

### 2.1 그대로 lift 할 코어 (hydra/wandb 비의존 — torch/numpy/sklearn 만)

| 소스 | 내용 | 비고 |
|---|---|---|
| `phase/models/autoencoder.py` | `EncoderTopK`(:95)·`DecoderLinearDict`(단위노름 사전)·`BaseAE` | top-k SAE, L1 없음(구조적 희소) |
| `phase/models/factory.py` | `build_model` | config dict → 모델 |
| `phase/train/_loop.py` | fit/epoch 루프 | 미니배치 지원, val-loss early stop |
| `phase/clustering/posthoc.py` | `fit_clusters`/`assign`·`drop_inactive` | dead-feature 사후 처리 |
| `phase/metrics/` | U-coefficient + **clock 시간분위 기준선** | 길이통제 기준선 내장 — 우리 원칙 정합 |
| `phase/data/pca.py` | train-only whitening | 누수 안전 (G1 에선 미사용, 코드만 확보) |

동료 하이퍼(참고·그대로 쓰지 말 것): input=PCA64, m=128, k=16, Adam wd0, mse.

### 2.2 이식 블로커 4 (29 문서 §갭 — 반드시 우회)

1. ★동료 빌더 `make_phase_dataset.py:102-109` 가 **토큰 축을 평균**으로 제거 — 우리는 per-token
   보존 필수(memory `feedback-no-rollout-pooling`: phase 는 timestep 구분이 load-bearing).
   → **빌더는 재사용 금지, 신규 작성.**
2. 동료는 1536→**PCA64 압축 후** SAE — overcomplete(m=128)가 PCA64 대비만 2×.
   → 우리는 **원본 1536-d 입력 + m=4~8×D**(6144~12288) 재설계. PCA 없이 시작(화이트닝은
   feature-wise standardize 정도만, train-split 통계로).
3. 평가 라벨이 phase(oracle)지 scene 아님 — scene probe·잔차화·순열 null·write 코드 전무 → 신규.
4. 동료 `cell` = task×object 이지 scenario_seed 아님 → **scene 라벨(layout_id/style_id/
   scenario_seed) 을 우리 rollout 메타에서 부착**하는 라벨러 신규.
   (주의: cell 과 scenario_seed 는 같은 게 아니라 **계층**이다 — cell 은 instruction 으로 고정되는
   굵은 단위(PPCC×bread, 대상 object 범주는 seed 가 바뀌어도 불변), scenario_seed 는 그 cell 안에서
   episode 마다 바뀌는 주방 인스턴스(layout·style·fixture 배치·distractor). cell 1 : seed N.
   우리가 분리할 "scene 암기"는 seed/layout/style 수준이라 cell 라벨로는 해상도 부족.)

### 2.3 우리 레포 기존 코드와의 관계

- `scripts/event_sae/` = **무관**(영상 MP4 프레임용 Event-SAE 어댑터, 비전 측). 건드리지 말 것.
- 신규 배치: 라이브러리 = **`src/sae/`**, 실행 스크립트 = **`scripts/scene_sae/`**.
- serve write(G3)는 `scripts/serve/steering_hooks.py` — G1/G2 에선 무접촉.

## 3. 데이터 자산 (2026-07-27 실물 검증 상태)

| 데이터 | 위치 | 계약 | 검증 상태 |
|---|---|---|---|
| **N1.5 fit30 5셀** (G1 주력) | 승준 `~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30/` — 셀 디렉토리(pq3_ppcc_bread·beer·drawer_left·drawer_right·pizza_cutter) + manifest tsv(`task_PPCC_fit.tsv`·`task_PPCC_fit_beerclean.tsv`·`task_OpenDrawer_fit.tsv`) | rollout pkl, record 별 hidden `[L, K, T=49, D=1536]` | ✅ 07-27 ls 확인 |
| N1.6 90ep 3셀 | 25a 는 `~/pkt_ws/datasets/exp4_3_n16/` 라 하나 **로컬 실물 없음** | [32,51,1536] T보존 | ⚠ 미확인 — exp4-3 세션에 실경로 문의 필요 (worktree 엔 atlas 요약 58M 뿐) |
| Cosmos 190ep 2셀 | 25a 는 `~/pkt_ws/datasets/exp4_3_cosmos/` — **로컬 실물 없음** | [28,2048] | ⚠ 동상 |
| atlas(layer peak) | exp4-3 worktree `.claude/worktrees/exp4-3-atlas/outputs/eval/robocasa/{groot_n15,groot_n16,cosmos}/exp4_3/atlas/` | mean_z 층별 표 | ✅ 존재 (`atlas_all.tsv`) |

**G1 은 N1.5 fit30 만으로 시작한다** (N1.6/Cosmos 는 G1 통과 후 교차검증 카드 — 경로 확인은
그때 exp4-3 세션/사용자에게).

- **layer 선택 — 단일 확정 금지, sweep 대상**: 평균분리 자체는 강하나 **어느 depth 가 scene 을
  인코딩하는지는 G1 에서 밝힐 대상**이지 미리 못 박지 말 것. atlas 실물
  (`docs/steering/sae_g1_refs/n15_atlas_all.tsv`, __global__ phase)에서 N1.5 분리는 **여러 layer 에
  걸쳐 유의**: drawer_left L10(mean_z 5.60)·L12(5.54)·L8(5.41)·L2(5.39)·L0(4.52) 전부 z>3,
  bread L10(5.08)·L12(3.07). peak 는 L10 이나 **L10 하나로 고정하지 말 것.**
  → 방침: **후보 대역(L0·L2·L8·L10·L12)에서 layer 별로 SAE 를 돌려 비교**, 그리고 **multi-layer
  SAE(여러 layer concat 또는 layer 별 SAE 앙상블)도 열어둔다** — scene feature 가 한 depth 에만
  있으리란 보장 없음. capture_layers 와 교집합만 사용(pkl 에서 확인, 하드코딩 금지). "어느 layer(들)이
  scene 을 인코딩하나"는 G1 의 부산 발견물로 기록.
- **pkl 로딩 참조 구현**: `scripts/safe/groot_n15/robocasa/steer/exp4_1/fit_mean_diff.py` 의
  `load_cell_rolls`(manifest tsv → rollout dict: `tok [n,L,T,D]`·`success`·`phases`·`length`·
  `scenario_seed`·`inference_seed`) — 그대로 import 해서 쓰면 됨.
- **scene 라벨 소스**: rollout pkl 메타(또는 사이드카)의 `ep_meta.layout_id`·`ep_meta.style_id`·
  `scenario_seed`. fit30 은 scene 당 1 rollout 이 아님 — **scenario_seed 별 rollout 수를 먼저
  집계**해서 probe 가 학습 가능한 클래스 수/표본 수인지 확인(§6 함정 1).

## 4. 구현 계획 (단계별 체크리스트)

### Phase A — 코어 lift (`src/sae/`)
- [ ] A1. `src/sae/models.py` ← `phase/models/autoencoder.py` (EncoderTopK/DecoderLinearDict/BaseAE)
      + `factory.py`. hydra 의존 제거, 출처 주석 유지(memory `feedback-preserve-provenance-comments`).
- [ ] A2. `src/sae/train.py` ← `phase/train/_loop.py`. 미니배치 필수(1536-d × 수십만 record 는
      full-batch 불가).
- [ ] A3. `src/sae/cluster.py`·`src/sae/metrics.py` ← posthoc/metrics (G1 엔 부수, G2 에 필요).
- [ ] A4. 단위 테스트 `tests/test_sae_core.py`: (i) top-k 후 활성 feature 수 == k,
      (ii) decoder 열 norm == 1, (iii) 재구성 loss 가 학습으로 감소, (iv) 합성 데이터
      (알려진 sparse 사전) 복원 sanity.

### Phase B — 데이터 빌더 (`scripts/scene_sae/build_sae_inputs.py`)
- [ ] B1. `load_cell_rolls` 재사용, **1셀(권장 pq3_drawer_left — 판수 최다)** × **후보 layer 여러 개**
      (L0·L2·L8·L10·L12 ∩ capture_layers) 슬라이스 → layer 별 `X_L [N_records × T, D=1536]`
      (**토큰 평균 금지**, 토큰=행으로 편다. K(denoise) 축은 마지막 K 사용 — fit_mean_diff 관례와
      동일한지 pkl 에서 확인). layer 는 단일 확정 아님(§3 참조) — 빌더가 layer 인자를 받아 여러 개 산출.
- [ ] B2. 행 단위 메타 병행 저장: `episode_idx, record_idx, token_idx, token_seg(state/future/action),
      phase, success, scenario_seed, layout_id, style_id` — G2 잔차화·길이통제가 전부 이 메타에 의존.
- [ ] B3. split: **episode 단위** train/val/test (record/token 단위 split 금지 — 누수).
- [ ] B4. 표준화: train-split feature-wise mean/std 만 (PCA 안 씀).
- [ ] 실행 위치: **승준**(pkl 이 거기 있음, CPU·torch = `~/anaconda3/bin/python`,
      `OMP/OPENBLAS_NUM_THREADS≤16`). 산출 NPZ 만 로컬 회수(remote_compute.sh 규약).

### Phase C — SAE 학습 (`scripts/scene_sae/train_scene_sae.py`)
- [ ] C1. 규모: D=1536, m=4×D=6144 부터(8×D 는 G1 통과 후), k 격자 {16, 32, 64} 소규모 sweep,
      선택 기준 = val 재구성 + dead-feature 비율(<50%).
- [ ] C2. 학습 자원 판단: X 가 (판수 ~30ep × ~140 record × 49 token ≈ 2×10⁵ 행) × 1536-d ≈
      1.2GB(float32) — **승준 CPU 로 가능 범위**(top-k SAE 는 가벼움). 느리면 NPZ 회수 후 로컬
      GPU 1장(빈 GPU 확인·cap 준수) — 어느 쪽이든 **fit-seed 고정·재현 가능**하게.
- [ ] C3. 체크포인트 = state_dict + config json (`outputs/eval/robocasa/groot_n15/scene_sae/<cell>/`).

### Phase D — G1 게이트: scene probe (`scripts/scene_sae/probe_scene.py`)
- [ ] D1. SAE feature 활성(z, top-k 후) → scene 라벨 probe: 라벨 3종 각각
      (scenario_seed[다중클래스]·layout_id·style_id) 로지스틱/선형 probe, **episode-held-out**.
- [ ] D2. 대조 기준선 2개: (i) 원본 activation probe(SAE 가 정보를 잃지 않았는지 상한),
      (ii) **episode-라벨 순열 null**(우연 수준). scene feature "식별" = 개별 feature 단위
      selectivity(feature별 scene 상호정보 or 단일-feature probe) 상위 목록 산출.
- [ ] D3. **G1 판정 기준(사전 등록)**: held-out scene probe 가 순열 null 대비 유의(z>3)하고
      원본 probe 의 ≥80% 를 SAE feature 로 회복 + scene-selective feature 집합이 비자명
      (전체 feature 의 <30%). 통과 → G2 설계 착수, 실패 → m/k 재조정 1회 후 그래도 실패면
      SAE 접근 자체를 사용자에게 보고.
- [ ] D4. 결과 문서 `docs/steering/31_sae_g1_results.md` (한글, 표 포함).

### Phase E — (G1 통과 후, 이 핸드아웃 범위 밖 — 예고만)
G2 = scene-selective feature 성분 제거(잔차화) 후 succ/fail read: **길이통제(성공 dwell cap)
유지·순열 null·held-out**. 참조 구현 = `fit_mean_diff.py` 의 truncation/순열 관례.

## 4.5 브랜치 규정 (exp5 = 이 SAE 세션 — 반드시 준수, 틀리기 쉬움)

**호칭**: 이 SAE 세션 = **exp5**. exp4 라인(exp4-1/4-2/4-3)의 후속이 아니라 **새 실험 라인**.

**base = dev.**
- exp4-1(oracle rescue·tremor·conceptor future_only·SAE 이식 문서)은 **dev 로 머지**된다
  (2026-07-27 결정, 진행 상태는 §끝 각주 확인). SAE 문서·atlas·fit 코드가 전부 dev 에 있어야
  exp5 가 읽는다.
- **exp5 작업 브랜치 = `feat/scene-sae`, dev 에서 분기**:
  ```
  git checkout dev && git pull origin dev
  git checkout -b feat/scene-sae
  ```

**하지 말 것 (실수 방지)**:
- ❌ **worktree 만들지 말 것.** exp4-1 은 메인 체크아웃에서 끝나 dev 로 갔다. exp5 도 **메인
  체크아웃**에서 feat/scene-sae 로 작업. (`.claude/worktrees/` 는 exp4-2·exp4-3 전용 — 그쪽은
  각자 세션이 독립 진행 중이고 dev 머지도 그쪽 몫. **건드리지 말 것.**)
- ❌ **`exp/exp4-1-oracle-rescue` 브랜치에 커밋 금지** — 종료된 브랜치. 세션 시작 시
  `git branch --show-current` 로 확인, exp4-1 위에 있으면 위 명령으로 dev→feat/scene-sae 재분기.
- ❌ exp4-2/exp4-3 worktree 의 코드·산출물 수정 금지 (참조는 읽기만).

**문서·PR 규약**:
- 새 문서 번호는 **31 부터** (30=핸드아웃, 29=이식검토, 25a=권고, 24* 계열=exp4). ⚠ 번호 충돌 주의:
  exp4-3 worktree 가 별도로 `29_related_works_map.md` 를 쓰고 있다(우리 29 와 번호 겹침) — dev 머지 시
  재확인. G1 결과는 `31_sae_g1_results.md`.
- PR 은 **dev 로**, 로컬 `gh pr create` (이 PC gh 설치·로그인됨). 머지 전에 열 것.
- 커밋 한글, `feat:`/`script:`/`docs:` 접두사.

## 5. 자원·운영 규칙 (요약 — memory 가 단일 출처)

- fit/분석 = **승준**(`kimseungjun@166.104.146.37:11112`, repo `~/workspace/temporal_vla`,
  python = `~/anaconda3/bin/python`, scipy 없음). 코드는 **git sync**(scp 금지),
  헬퍼 `scripts/utils/remote_compute.sh`. 결과 소용량만 회수.
- 로컬 GPU 쓸 땐: **완전히 빈 GPU 만**(nvidia-smi compute-apps 소유자 확인), exp4-1+exp4-2
  합산 ≤3 GPU. exp4-2 가 로컬 GPU 6·7 사용 이력(포트 8484/8485).
- 로컬 CPU 캡: OMP/OPENBLAS ≤16 스레드.
- 커밋: 한글, `feat:`/`script:`/`docs:` 접두사, 브랜치 `feat/scene-sae` (dev 분기).
- 산출물 대용량이면 승준 HDD 아카이브 규약([[remote-data-archive]] — pkl·csv·mp4 전부, HDD만).

## 6. 함정 목록 (이번 라운드들에서 실제로 맞은 것)

1. **scene 라벨 표본 불균형**: fit30 은 scene(scenario_seed)당 rollout 수가 균일하지 않다.
   probe 전에 라벨 분포 집계 → 클래스당 최소 표본 미달이면 layout/style 처럼 굵은 라벨로.
2. **episode-단위 split 필수**: record/token split 은 같은 episode 의 이웃 record 가 train/test
   양쪽에 들어가 probe 정확도가 허위 상승(자기상관 누수).
3. **길이 confound**: scene probe 자체도 실패 episode(길게 체류)의 record 가 많아 라벨-길이
   상관이 생길 수 있음 — probe 표본을 episode 당 균등 subsample 하거나 record 가중 보정.
4. **in-sample 아티팩트**: exp4-1 의 multi-layer +0.20 사건 — 모든 판정은 held-out. fit 에 쓴
   episode 는 G3 eval 에서 제외(fit-seed 분리).
5. **베이스 python 에 torch 없음**(승준) — `~/anaconda3/bin/python` 명시. scipy 금지(없음).
6. **fake-done**: 원격/백그라운드 완료 판정은 로그 문자열이 아니라 **산출물 개수 대조**
   (이번 세션 NVML 사고: "ALL DONE" 로그 밑에 82판 미실행이 숨어 있었다).
7. **pkl 계약 확인 우선**: `tok` 축이 [n_records, L, T, D] 인지, K(denoise) 축이 접혀 있는지
   pkl 하나를 실제로 열어 shape 를 찍고 시작하라(fit_mean_diff 의 li 인덱싱 참조).
8. 동료 코드 lift 시 **출처 주석 보존**, 우리가 지운 기능(PCA·토큰평균)은 "왜 안 쓰는지" 주석.

## 7. 완료 정의 (이 핸드아웃의 끝)

- [ ] `src/sae/` 코어 + 테스트 통과
- [ ] drawer_left × peak-layer SAE 학습 완료(재현 가능: seed·config 기록)
- [ ] G1 probe 결과 표 + 판정(기준 §4-D3) → `docs/steering/31_sae_g1_results.md`
- [ ] 사용자에게 G1 판정 보고 → G2 go/no-go 결정 받기

## 8. 새 세션 시작 프롬프트 (복붙용)

```
docs/steering/30_sae_g1_port_handout.md 읽고 SAE G1 이식·구현 시작해.
브랜치 feat/scene-sae 로 dev 에서 분기. Phase A(코어 lift)부터 체크리스트 순서대로.
fit 데이터는 승준 N1.5 fit30 (핸드아웃 §3), 실행 규칙은 §5, 함정 §6 준수.
G1 판정(§4-D3)까지 가고 결과는 31 문서로. 막히면 중단하고 보고.
```
