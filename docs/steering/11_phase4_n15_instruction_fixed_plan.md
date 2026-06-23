# Phase 4: GR00T N1.5 instruction-fixed COAST/VL 검증 계획

작성일: 2026-06-10

## 문서 운영 방식

이 문서는 Phase 4 N1.5 instruction-fixed 작업의 living plan이다. 구현을 진행하면서 새 사실,
결정, 실패 원인, 검증 결과를 이 문서에 바로 반영한다.

- 작업 시작 전 이 문서의 "현재 작업 board"와 "필요한 코드 변경 범위"를 먼저 본다.
- 구현이 끝난 항목은 체크박스를 갱신하고, 사용한 검증 명령을 "검증 기록"에 남긴다.
- 실험 단위, artifact layout, seed 기준, feature shape가 바뀌면 코드보다 먼저 이 문서를 갱신한다.
- 임시 판단은 본문에 남기지 않고, 확정된 contract만 남긴다.
- 이 문서와 실제 코드가 어긋나면 코드 변경 또는 문서 수정을 같은 작업 단위에서 끝낸다.

## 현재 작업 board

- [x] Instruction-fixed 실험 단위 정의
- [x] 대상 15 instruction cell 초안 확정
- [x] N1.5 DiT/VL feature 수집 필요성 확인
- [x] `task/instruction_cell` nested artifact layout 채택
- [x] Instruction cell config 단일 출처 추가
- [x] Canonical instruction seed/ep_meta selector 추가
- [x] Cell별 seed shard launcher 추가
- [x] N1.5 HTTP collector가 nested layout과 cell metadata를 저장하도록 확장
- [x] Verifier가 nested layout과 cell metadata를 검증하도록 확장
- [x] Cell별 success/failure balance summary script 추가
- [x] 1-cell x 3-episode VL+DiT smoke 수집
- [x] 15-cell x 1-episode seed probe로 canonical instruction coverage 확인
- [x] 15-cell x 50-episode 본수집 완료 (`completed=750 expected=750`, full verifier `status=ok`)
- [x] Cell별 success/failure balance summary 생성 (`analysis/instruction_success_rates.tsv`)
- [x] N1.5 schema용 VL/DiT separation loader 추가
- [x] N1.6-aligned N1.5 DiT block residual capture option 추가
- [x] N1.6-aligned N1.5 DiT block residual 1-cell smoke로 runtime shape 확정
- [x] N1.5 instruction-cell conceptor fit wrapper 추가
- [x] `open_drawer_right` DiT/VL conceptor smoke fit 완료
- [x] 10/10 balance 통과 7개 cell DiT/VL conceptor full fit 완료
- [x] N1.5 steering eval matrix planner 추가
- [x] N1.5 steering eval row runner 추가
- [x] Held-out eval seed manifest용 collection-seed exclusion gate 추가
- [x] 실행 중 run의 pkl 기반 `ep_meta` archive backfill helper 추가
- [ ] Cell별 conceptor fit
- [x] N1.5 HTTP steering serve hook 연결
- [ ] N1.6-aligned N1.5 block residual 15 cell x 50 수집 완료
- [x] N1.6-aligned N1.5 block residual layer-preserved cache loader 준비
- [x] N1.6-aligned `dit_layer<i>`/`vl` conceptor set fit wrapper 추가
- [x] N1.5 steering eval wrapper 및 runtime smoke 연결
- [ ] N1.5 steering eval full SR matrix 실행
- [ ] SR eval 결과표와 Phase 4 분석 문서 작성

## 한 줄 결론

N1.5 COAST/VL 검증의 실험 단위는 task가 아니라 **instruction cell**이어야 한다. VL pathway는
language/goal semantics를 직접 보므로, 같은 task 안에서도 instruction이 달라지면 latent와 SR이
함께 달라진다. 따라서 Phase 4 데이터 수집, conceptor fit, SR eval은 모두 instruction을 고정한
cell 단위로 진행한다.

## 이어받는 기준 문서

- COAST 수식/conceptor fit/SR eval 재현 기준 문서는 정리됨(재현 실패—원인 미상); conceptor 수학 provenance는 `src/conceptor/README.md`.
- `docs/steering/08_phase3_dit32_separation.md`: DiT pre-failure 분리력 분석 방식
- `docs/steering/09_phase3_vl_dit_comparison.md`: VL(goal) vs DiT(motor) pathway 분리력 결론
- `docs/steering/10_session_handoff.md`: Phase 3에서 Phase 4로 넘어가는 최신 runbook
- `docs/benchmarks/robocasa_env_reproducibility.md`: RoboCasa construction seed와 ep_meta replay 경계
- `docs/groot/n16_05_safe_env_reproduction.md`: SAFE 수집 경로의 scenario_seed/ep_meta contract
- `scripts/safe/groot_n15/robocasa/README.md`: N1.5 RoboCasa feature collection contract
- `CLAUDE.md`: seed, per-episode logging, 원격 compute 노드 사용 경계

## 왜 instruction 고정인가

기존 N1.6 moderate10 분석은 task 단위로 묶었다. 이 방식은 DiT motor pathway 분석에는 쓸 수
있었지만, VL pathway 분석에는 confound가 생긴다.

- VL은 goal/language pathway다. task가 같아도 "left drawer"와 "right drawer", "door"와
  "doors", "onion"과 "apple"은 서로 다른 goal embedding을 만든다.
- 실제 SR도 instruction variant별로 크게 갈린다. SlideDishwasherRack의 "slide in"과
  "slide out", OpenCabinet의 "door"와 "doors"가 대표 예다.
- task 단위로 pooling하면 성공/실패 latent가 instruction 차이를 같이 먹는다. 이때 conceptor가
  failure subspace를 잡는 것인지, 쉬운 instruction과 어려운 instruction을 구분하는 것인지
  해석이 불가능해진다.
- steering의 최종 metric은 ΔSR이다. 조건별 ΔSR을 비교하려면 baseline, fit data, steered eval이
  같은 instruction 분포를 가져야 한다.

따라서 Phase 4의 기본 contract는 다음이다.

1. instruction cell마다 독립적으로 50 episode를 수집한다.
2. success/failure label은 cell 내부에서만 비교한다.
3. VL/DiT separation, conceptor fit, steering eval은 cell_id를 기본 group key로 쓴다.
4. task 평균은 cell 결과를 먼저 만든 뒤 보조 집계로만 낸다.

## 대상 instruction set

총 10 task group, 15 instruction cell이다. 기본 목표는 50 episode per cell, 총 750 episode다.
SlideDishwasherRack은 이번 N1.5 instruction-fixed 본수집에서 제외한다.

| cell_id | task | canonical instruction | 선정 이유 |
|---|---|---|---|
| ppcs_onion | PickPlaceCounterToStove | Pick the onion from the plate and place it in the pan. | seed-selected object instruction |
| ppcs_apple | PickPlaceCounterToStove | Pick the apple from the plate and place it in the pan. | seed-selected object instruction |
| ppdc_tongs | PickPlaceDrawerToCounter | Pick the tongs from the drawer and place it on the counter. | seed-selected object instruction |
| ppdc_wooden_spoon | PickPlaceDrawerToCounter | Pick the wooden spoon from the drawer and place it on the counter. | seed-selected object instruction |
| ppcc_potato | PickPlaceCounterToCabinet | Pick the potato from the counter and place it in the cabinet. | seed-selected object instruction |
| ppcc_bread | PickPlaceCounterToCabinet | Pick the bread from the counter and place it in the cabinet. | seed-selected object instruction |
| open_cabinet_door | OpenCabinet | Open the cabinet door. | "doors" variant 제외, 단일문으로 통일 |
| open_drawer_right | OpenDrawer | Open the right drawer. | left/right를 별도 cell로 유지 |
| open_drawer_left | OpenDrawer | Open the left drawer. | left/right를 별도 cell로 유지 |
| close_toaster_oven_door | CloseToasterOvenDoor | Close the toaster oven door. | 단일 canonical instruction |
| turn_on_microwave | TurnOnMicrowave | Press the start button on the microwave. | 코드상 canonical instruction |
| turn_on_sink_faucet | TurnOnSinkFaucet | Turn on the sink faucet. | 코드상 canonical instruction |
| navigate_fridge | NavigateKitchen | Navigate to the fridge. | seed-selected navigation target |
| navigate_coffee_machine | NavigateKitchen | Navigate to the coffee machine. | seed-selected navigation target |
| coffee_setup_mug | CoffeeSetupMug | Pick the mug from the counter and place it under the coffee machine dispenser. | mug/goal 고정 |

주의할 점:

- `--task-description`은 현재 N1.5 feature collector에서 artifact metadata로만 쓰인다. 모델에
  들어가는 prompt는 env observation의 `annotation.human.*task_description`에서 추출된다.
- 따라서 instruction 고정은 단순히 `--task-description`을 넘기는 방식으로는 안 된다.
- 우선순위는 **collection env path의 scenario_seed로 실제 env instruction을 고정**하는 것이다.
  prompt override는 별도 인과 실험으로만 다룬다.
- PnP object instruction과 `NavigateKitchen` target은 모두 같은 방식으로 처리한다. collection env
  path에서 `scenario_seed`를 스캔하고, 실제 reset instruction이 canonical instruction과 정확히
  일치하는 seed만 채택한다.
- RoboCasa upstream source나 env constructor 인자를 실험별로 바꾸지 않는다. sparse object cell은
  느려도 seed scan + checkpoint/resume으로 해결한다.

## 데이터 수집 contract

Run root:

```text
outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/
```

Seed-scan run root:

```text
outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_natural/
```

Sharded seed-scan run root:

```text
outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_sharded_natural/
```

권장 구조:

```text
raw_rollouts/
  PickPlaceCounterToStove/
    ppcs_onion/
      task00--ep0000--succ1.pkl
      task00--ep0000--succ1.csv
      task00--ep0000--succ1.mp4
    ppcs_apple/
      task01--ep0000--succ0.pkl
      task01--ep0000--succ0.csv
      task01--ep0000--succ0.mp4
  OpenDrawer/
    open_drawer_right/
    open_drawer_left/
  ...
ep_meta/
  PickPlaceCounterToStove/
    ppcs_onion/
    ppcs_apple/
  OpenDrawer/
    open_drawer_right/
    open_drawer_left/
manifests/
  selected_instruction_seeds.tsv
  shards/
    <cell_id>.tsv
analysis/
  seed_scan_status.tsv
  seed_sample_summary.tsv
  instruction_success_rates.tsv
  pathway_separation/
conceptor/
steer_eval/
```

seed 기준:

- scenario seed는 `100000`부터 시작하는 표준 eval seed band를 사용한다.
- 각 cell은 seed scanner로 **collection env path**를 reset하고 `ep_meta["lang"]` 또는 reset
  observation의 instruction이 canonical instruction과 정확히 일치하는 seed만 채택한다.
- 채택 seed는 `selected_instruction_seeds.tsv`에 `cell_id`, `task`, `env_name`, `scenario_seed`,
  `instruction`, `canonical_instruction`, `ep_meta_path`로 기록한다.
- collection은 채택 seed마다 `--n-episodes 1 --seed <scenario_seed>`로 실행한다.
- deterministic policy 비교가 필요하므로 `--inference-seed <scenario_seed>`를 같이 둔다.
- held-out SR eval seed manifest는 `select_instruction_seeds.py --exclude-selected-seeds <collection_manifest>`로
  만든다. selector는 같은 `cell_id/scenario_seed`가 collection manifest에 있으면 instruction match여도
  선택하지 않고 다음 matching seed를 찾으며, resume 중 기존 output row가 exclude manifest와 겹치면
  즉시 실패한다.
- collection에서 `--replay-ep-meta`를 켜는 경우, 해당 seed의 run-root ep_meta JSON은 필수다.
  replay manifest가 없으면 natural reset으로 조용히 fallback하지 말고 즉시 실패한다.
- seed scanner는 `selected_instruction_seeds.tsv`와 별도로 `*_scan_progress.tsv`를 checkpoint로
  저장한다. sparse instruction cell에서 중단되면 `--resume`으로 마지막 non-match scan 이후 seed부터
  이어간다.
- seed scanner는 `*_scan_samples.tsv`도 저장한다. 이 파일은 scanned seed마다 실제 instruction과
  canonical instruction, match 여부를 남겨서 sparse object sampling과 canonical string 오류를
  구분하는 audit log로 쓴다. `--resume`은 selected manifest, progress checkpoint, sample log의
  최대 seed를 모두 보고 재시작 위치를 정한다.
- seed preselection은 cell별 shard로 병렬 실행할 수 있다. 각 shard는
  `manifests/shards/<cell_id>.tsv`를 쓰고, collection 직전
  `merge_instruction_seed_shards.py --shard-dir <.../manifests/shards> --require-complete`로 단일
  `selected_instruction_seeds.tsv`를 만든다. `--shard-dir`는 seed shard만 자동 발견하고
  `_scan_progress.tsv`, `_scan_samples.tsv` audit 파일은 merge 입력에서 제외한다.
- `launch_instruction_seed_shards.py`는 active tmux session과 이미 target count를 채운 shard를
  skip하고, `--max-new-sessions`로 동시에 새로 띄울 shard 수를 제한한다.
- launcher가 `tmux list-sessions` 권한 문제를 만나면 active session을 못 본 채 진행하지 않고
  실패한다. 이 경우 host tmux 접근 권한이 있는 환경에서 dry-run을 다시 확인한다.
- `--require-complete`가 실패하면 collection을 시작하지 않는다. 이 gate는 모든 cell이 target count를
  채웠고 instruction mismatch/old env forcing schema가 없다는 것을 확인하기 위한 것이다.
- `scenario_seed`와 collection env path는 env layout, object/target, instruction sampling을 고정하는
  1차 authority다.
- `selected_instruction_seeds.tsv`의 `ep_meta_path`는 본수집 run root의
  `ep_meta/<task>/<cell_id>/...json`을 가리킨다. 초기 seed-scan root 경로는 보존되어 있지 않아
  현재 manifest에서는 쓰지 않는다.
- selected manifest를 외부 merge/backfill로 갱신한 뒤에는
  `materialize_selected_ep_meta.py --selected-seeds <manifest> --cell-id ...`로 missing ep_meta를 먼저
  채우고, instruction mismatch가 하나라도 나오면 해당 cell의 selected seed를 현재 runtime 기준으로
  다시 뽑는다.
- `inference_seed`는 `/act` 또는 `/act_with_features` 요청마다 policy-side RNG를 고정한다. 이것은
  instruction을 고르는 seed가 아니라, 같은 observation 조건에서 stochastic action sampling을
  재현 가능하게 만드는 보조 seed다.
- 이번 instruction-fixed 수집에서 authority는 collection env path의 `scenario_seed`다. 본수집
  rollout pkl에는 reset 직후 캡처한 `ep_meta`를 항상 저장하고, wrapper는 `--ep-meta-dir`를
  항상 넘겨 run-root JSON archive도 export한다.
- 초기 `n15_instruction_fixed50_collect_720` 세션은 wrapper 수정 전에 시작된 프로세스라 captured
  command에 `--ep-meta-dir`가 없었다. 그 구간은 pkl 내부 `ep_meta`를 authority로 두고,
  run-root JSON archive를 pkl backfill로 맞췄다. wrapper 수정 이후 새 collect process는
  `--ep-meta-dir`를 직접 넘긴다.
- 따라서 본수집의 ep_meta completeness gate는 raw rollout pkl의 `ep_meta` 필드와
  selected manifest의 `ep_meta_path` JSON 존재/내용 대조다.
- 2026-06-11 repair note: active run은 12개 cell 600/750까지 pkl 내부 `ep_meta`가 정상 저장됐지만,
  미수집 3개 cell(`navigate_fridge`, `navigate_coffee_machine`, `coffee_setup_mug`)의 run-root
  ep_meta JSON이 없었다. 이 상태에서 `--replay-ep-meta`를 켜면 collector가 자연 reset으로 fallback할
  수 있었고, 실제로 `navigate_fridge` seed 100023 command는 `Navigate to the toaster.` mismatch로
  중단됐다. 이후 missing replay manifest는 hard fail로 바꿨고, stale selected seed는 repair shard로
  재선별한다.
- 이유: OpenDrawer는 `ep_meta["lang"]`은 저장하지만 env가 읽는 `drawer_side` sampled state를
  replay 가능한 key로 저장하지 않는다. 이 상태에서 ep_meta replay를 켜면 layout/fixture RNG
  소비가 달라져 같은 seed가 left/right를 바꿀 수 있다.
- ep_meta replay는 paired-scene 재현 같은 별도 목적에서만 `--replay-ep-meta`로 명시적으로 켠다.

cell metadata 기준:

- `cell_id`는 분석과 사람이 읽는 경로의 기본 key다.
- `cell_index`는 기존 filename과 verifier 호환을 위해 unique integer로 둔다.
- pkl payload에는 `task_id=<cell_index>`, `cell_id`, `robocasa_task`, `canonical_instruction`을 모두 저장한다.
- `robocasa_task`는 RoboCasa env/task 이름이고, `cell_id`는 instruction-fixed 실험 단위다.
- `task_description`은 실제 env에서 나온 canonical instruction과 같아야 한다.

feature 기준:

- N1.5 HTTP server는 `scripts/serve/lerobot.py --collect --capture-vl`로 띄운다.
- DiT feature: `groot_n15_dit_action_tokens_pre_decode`, expected shape `4,16,1024`.
- VL feature: `groot_n15_vlln_seq_meanpool`, expected dim `2048`.
- 현재 N1.5 본수집의 DiT feature는 `action_head.model` 최종 출력의 action token
  pre-decode representation이다. pkl의 `hidden_states[step]`는
  `[K_denoise=4, H_action=16, D_final=1024]`다.
- N1.6 Phase 3 pathway run의 `hidden_states[step]=[7,51,1536]`는 DiT
  transformer block residual stream을 여러 layer에서 캡처한 별도 scheme이다. 이번 N1.5
  final-DiT 본수집은 이 block-residual `[L,T,D_block]` feature를 저장하지 않는다.
- 따라서 현재 수집으로 바로 fit 가능한 steering matrix는 `dit_final` D=1024와 `vl` D=2048이다.
  N1.5 layer-specific DiT block steering(`--steering-layer i`)에 맞추려면 새로 추가한
  `--groot-dit-capture-layers` mode로 block-feature 수집을 별도 run으로 만들어야 한다.
- N1.6-aligned N1.5 DiT block residual capture는 다음 contract다.
  - alignment 기준은 feature pathway, axis, residual dimension, layer sampling ratio, VL feature
    dim이다. N1.6의 token count 51을 N1.5에 padding/truncation으로 맞추지 않는다. N1.5
    runtime token count는 smoke에서 확정한 49를 verifier contract로 고정한다.
  - N1.6 reference layer sampling: `0,2,4,8,16,24,31` over a 32-layer DiT.
  - N1.5 runtime layer sampling: `0,2,4,8,10,12,15` over a 16-layer DiT. N1.6 layer index
    `16/24/31`은 N1.5에서 out-of-range라 그대로 쓸 수 없다.
  - server option: `--groot-dit-capture-layers 0,2,4,8,10,12,15`
  - DiT feature kind: `groot_n15_dit_block_residual_tokens`
  - DiT feature axes: `["layer", "model_token", "feature_dim"]`
  - pkl metadata: `capture_layers`, `layer_count`, `token_count`, `model_action_horizon`,
    `num_inference_timesteps`
  - feature shape target: `[L=7, T_model_token=49, D_block=1536]`; N1.6 reference는
    `[7,51,1536]`이고 N1.5 smoke runtime은 `[7,49,1536]`이다.
- aligned mode에서는 `exported_action_token_count` / `feature_action_horizon`가 action-token chunk
  불변식이 아니므로 `None`일 수 있다. writer/verifier gate는 `feature_axes[0] == "layer"`인
  block residual feature를 action-token horizon check에서 제외한다.
- verifier는 N1.6 collection verifier를 재사용하되 N1.5 expectation override와
  `--require-vl-hidden-states`를 반드시 켠다.
- aligned smoke verifier는 feature kind/shape뿐 아니라
  `--expected-feature-axes layer,model_token,feature_dim`,
  `--expected-capture-layers 0,2,4,8,10,12,15`,
  `--expected-layer-count 7`, `--expected-token-count 49`까지 검증한다.
  block residual mode에서는 `--expected-feature-action-horizon`를 넘기지 않는다.
- aligned cache는 `instruction_pathway_features.py --preserve-dit-layers`로 만든다.
  산출 NPZ는 다음 keys를 가진다:
  - `dit`: token mean 후 layer mean까지 적용한 `[N_step,1536]` 진단용 pooled feature.
  - `dit_layers`: token mean 후 layer를 보존한 `[N_step,7,1536]` feature.
  - `dit_capture_layers`: actual N1.5 layer ids `[0,2,4,8,10,12,15]`.
  - `dit_layer0`, `dit_layer2`, `dit_layer4`, `dit_layer8`, `dit_layer10`, `dit_layer12`,
    `dit_layer15`: 각 layer별 `[N_step,1536]` feature. layer-specific conceptor fit과
    `--steering-layer` eval은 이 keys를 쓴다.
  - `vl`: `[N_step,2048]` goal pathway feature.
- aligned conceptor fit은 `fit_aligned_instruction_conceptors.py`를 단일 entrypoint로 쓴다. 이
  wrapper는 cache의 `dit_layer<i>` keys를 숫자순으로 발견하고 `vl`을 추가한 뒤, 기존
  `fit_instruction_conceptors.py` 로직을 pathway별로 호출한다. dry-run은
  `aligned_eligibility_summary.tsv`, 실제 fit은 `aligned_fit_summary.tsv`를 쓴다.
- active aligned collection이 끝나기 전에는 같은 port/GPU server를 steering eval server로 바꾸지
  않는다. cache 생성, aligned conceptor fit, SR eval은
  `target_instruction_fixed15_block_residual_50ep` verifier가 full `status=ok`를 낸 뒤 시작한다.
- COAST/N1.6-aligned collection에서는 `max_episode_steps=720`을 고정한다. N1.5 wrapper를
  `--max-episode-steps` 없이 실행하면 RoboCasa task registry horizon을 써서
  `PickPlaceCounterToStove=400`, `PickPlaceDrawerToCounter=500`처럼 task별 cap이 섞인다. 이 상태는
  N1.6 alignment 기준에서 superseded run으로 취급한다.

성공/실패 balance:

- 1차 수집은 50 episode per cell로 고정한다.
- conceptor fit 최소 조건은 cell 내부 `success >= 10` and `failure >= 10`이다.
- 50 episode 후 이 조건을 못 맞춘 cell은 +25 episode 단위로 top-up한다.
- top-up 여부는 task가 아니라 cell별로 판단한다.

## 실행 순서

### Phase A: preflight

목표: instruction-fixed 수집이 실제로 가능한지 먼저 증명한다.

1. N1.5 checkpoint와 LeRobot HTTP `/act_with_features` 서버를 확인한다.
2. `--capture-vl` health metadata에서 DiT/VL feature kind와 shape를 확인한다.
3. seed scanner로 각 cell당 3개 seed만 먼저 찾는다.
4. 1 cell을 골라 3 episode smoke collect를 수행한다.
5. verifier로 DiT/VL pkl, csv, mp4 triplet을 검증한다.

통과 기준:

- pkl마다 `hidden_states`와 `vl_hidden_states`가 모두 있다.
- 저장된 `task_description`이 canonical instruction과 정확히 일치한다.
- `scenario_seed`, `ep_meta`, `inference_seed`, video metadata가 pkl에 기록된다.

### Phase B: 본수집

목표: 15 instruction cell x 50 episode의 N1.5 VL+DiT feature 데이터를 만든다.

1. `selected_instruction_seeds.tsv`를 cell별 50개 seed로 채운다.
2. cell별로 HTTP feature collect를 실행한다.
3. 중간 실패는 cell 단위로 재시도한다. 성공한 cell의 raw_rollouts는 건드리지 않는다.
4. 전체 수집 후 verifier를 cell 전체에 대해 실행한다.
5. `instruction_success_rates.tsv`를 만들어 cell별 `succ/total`, SR, 평균 step, instruction을 남긴다.

수집은 로컬 전용이다. RoboCasa Docker와 simulator가 필요하므로 원격 compute 노드에서 돌리지 않는다.

### Phase B-1: N1.6-aligned block residual smoke

목표: final-DiT 본수집을 완료한 뒤, 같은 instruction seed manifest를 재사용해 N1.6-style
DiT block residual feature가 실제 N1.5 model에서 어떤 shape로 저장되는지 확정한다.

전제:

- `target_instruction_fixed15_pathway_50ep` final-DiT collection과 verifier를 먼저 끝낸다.
- active `lerobot_n15_instruction_feature_server`를 aligned mode로 바꾸기 전 기존 collection process가
  완전히 종료됐는지 확인한다.
- smoke는 1 cell x 1 episode만 먼저 실행한다. wrapper의 `--max-episodes-per-cell 1`로 selected
  manifest 첫 seed만 사용한다.

server는 LeRobot compose service에서 띄운다. host Python은 cache/env 차이 때문에 재현 기준으로 두지
않는다.

```bash
docker compose run --rm --no-deps -T lerobot \
  python /temporal_vla/scripts/serve/lerobot.py \
  --profile /temporal_vla/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
  --device cuda \
  --host 127.0.0.1 \
  --port 8400 \
  --collect \
  --capture-vl \
  --groot-dit-capture-layers 0,2,4,8,10,12,15
```

smoke collect는 RoboCasa env가 준비된 기존 RoboCasa container에서 실행한다. 새
`docker compose run robocasa`는 VNC prompt에 걸릴 수 있으므로 canonical path로 쓰지 않는다.

```bash
docker exec -e MUJOCO_GL=egl temporal_vla-robocasa-run-3705634bbbf6 \
  python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py \
  --selected-seeds /temporal_vla/outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv \
  --output-dir /temporal_vla/outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_smoke/raw_rollouts \
  --ep-meta-dir /temporal_vla/outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_smoke/ep_meta \
  --cell-id open_drawer_right \
  --max-episodes-per-cell 1 \
  --n-action-steps 16 \
  --max-episode-steps 720 \
  --video-fps 20 \
  --steps-per-render 2 \
  --wait-ready
```

smoke 결과:

- pkl:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_smoke/raw_rollouts/OpenDrawer/open_drawer_right/task7--ep0--succ0.pkl`
- `task_description == canonical_instruction == "Open the right drawer."`
- `scenario_seed == inference_seed == 100000`
- `feature_kind == groot_n15_dit_block_residual_tokens`
- `hidden_states[0].shape == (7,49,1536)`, dtype `torch.float16`
- `vl_hidden_states[0].shape == (2048,)`, dtype `torch.float16`
- `capture_layers == [0,2,4,8,10,12,15]`
- `token_count == 49`, `layer_count == 7`, `model_action_horizon == 16`
- `max_episode_steps == 720`, `n_action_steps == 16`, `video_fps == 20`,
  `steps_per_render == 2`

single-cell smoke verifier는 `--allow-partial`을 켠다. verifier의 `--tasks-override OpenDrawer`는
OpenDrawer의 right/left 두 cell을 모두 기대하므로, right 1개만 모은 smoke는 `partial-ok`가 정상이다.

```bash
python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_smoke/raw_rollouts \
  --layout task_instruction \
  --cell-config configs/robocasa/n15_instruction_fixed_cells.tsv \
  --tasks-override OpenDrawer \
  --episodes-per-task 1 \
  --allow-partial \
  --expected-model-family lerobot_groot_n15 \
  --expected-policy-transport http \
  --expected-task-suite-name lerobot_groot_n15_robocasa \
  --expected-feature-kind groot_n15_dit_block_residual_tokens \
  --expected-feature-axes layer,model_token,feature_dim \
  --expected-hidden-shape 7,49,1536 \
  --expected-model-horizon 16 \
  --expected-valid-horizon none \
  --expected-n-action-steps 16 \
  --expected-capture-layers 0,2,4,8,10,12,15 \
  --expected-layer-count 7 \
  --expected-token-count 49 \
  --expected-max-episode-steps 720 \
  --expected-video-fps 20 \
  --expected-steps-per-render 2 \
  --require-vl-hidden-states \
  --expected-vl-hidden-shape 2048 \
  --expected-vl-feature-kind groot_n15_vlln_seq_meanpool \
  --expected-vl-feature-dim 2048
```

### Phase C: representation 분석

목표: N1.5에서도 VL(goal)과 DiT(motor)의 분리 타이밍이 N1.6과 같은지 확인한다.

분석 단위:

- group key: `cell_id`
- time cutoff: t = 4, 8, 12, 16, 20
- VL: `vl_hidden_states` step mean
- DiT: 현재 본수집에서는 `hidden_states`의 denoising/action-token pool, 즉 D=1024
  `dit_final` representation을 사용한다. D=1536 block residual layer 분석은 추가 수집 없이는
  수행하지 않는다. aligned 재수집 run에서는 `hidden_states`의 layer/model-token pool을 사용한다.
- metric: PCA to LDA direction, 5-fold CV AUROC, permutation null

해석 기준:

- VL이 t<=8에서 DiT보다 높으면 goal pathway early signal 재현이다.
- DiT가 t>=12에서 높으면 motor pathway late signal 재현이다.
- 둘 다 null과 구분되지 않는 cell은 steering 기대치를 낮게 둔다.

원격 compute 노드는 여기부터 쓴다. `scripts/utils/remote_compute.sh sync-code`로 코드 동기화 후,
원격의 `raw_rollouts`에서 analysis JSON/plot/TSV만 생성하고 `pull-results`로 회수한다.

### Phase D: conceptor fit

목표: cell별 success/failure contrastive conceptor를 만들고, pathway별 steering 후보를 고른다.

fit 원칙:

- success/failure는 같은 cell 안에서만 나눈다.
- length confound를 막기 위해 고정 t 또는 success mean W truncation만 쓴다.
- headline method는 COAST conceptor다.
- VL은 primary pathway, DiT는 PnP/open-drawer/open-cabinet 계열의 secondary pathway다.
- beta 후보는 `0.1`, `0.3`으로 시작한다.

fit 산출물:

```text
conceptor/
  vl/truncated_w10/<cell_id>/conceptors.npz
  vl/truncated_w10/<cell_id>/metadata.json
  vl/truncated_w10/eligibility_summary.tsv
  vl/truncated_w10/fit_summary.tsv
  dit/truncated_w10/<cell_id>/conceptors.npz
  dit/truncated_w10/<cell_id>/metadata.json
  dit/truncated_w10/eligibility_summary.tsv
  dit/truncated_w10/fit_summary.tsv
  fit_summary.tsv
```

N1.5 wrapper:

```bash
python scripts/safe/groot_n15/robocasa/steer/fit_instruction_conceptors.py \
  --cache outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/analysis/pathway_separation/final_dit_vl_step_features_mean_mean.npz \
  --pathway vl \
  --agg-mode truncated \
  --max-len 10 \
  --dry-run
```

wrapper contract:

- input은 `instruction_pathway_features.py`가 만든 N1.5 cache다.
- group key는 `task_id`가 아니라 `cell_id`다.
- success/failure balance gate는 episode 수 기준 `--min-episodes-per-class 10`이다.
- dry-run은 `eligibility_summary.tsv`를 쓰고, 실제 fit은 `fit_summary.tsv`와 cell별
  `conceptors.npz`/`metadata.json`을 쓴다.
- conceptor 계산 자체는 N1.6의 `fit_conceptor_steering.py`의 `fit_group`/`save_group`를
  재사용한다.

현재 conceptor fit 상태:

- `truncated_w10`, default alpha grid 기준 10/10 balance를 통과한 7개 cell은 DiT/VL 모두 full
  fit 완료했다.
- fitted cells:
  `close_toaster_oven_door`, `open_cabinet_door`, `open_drawer_left`,
  `open_drawer_right`, `ppcc_bread`, `ppdc_tongs`, `turn_on_sink_faucet`.
- DiT outputs: `conceptor/dit/truncated_w10/<cell_id>/conceptors.npz`,
  all matrices shape `(1024,1024)`.
- VL outputs: `conceptor/vl/truncated_w10/<cell_id>/conceptors.npz`,
  all matrices shape `(2048,2048)`.
- skipped cells:
  `coffee_setup_mug`, `navigate_coffee_machine`, `navigate_fridge`, `ppcc_potato`,
  `ppcs_apple`, `ppcs_onion`, `ppdc_wooden_spoon`, `turn_on_microwave`.
- skipped 이유는 모두 `--min-episodes-per-class 10` 미달이다. 이 8개는 top-up 후 fit하거나
  fit-exclude로 명시해야 한다.

선택 기준:

- VL-dominant cell: VL conceptor를 primary eval 후보로 둔다.
- DiT-dominant cell: DiT conceptor를 primary eval 후보로 둔다.
- 둘 다 약한 cell: always-on steering eval에서 제외하거나 exploratory로만 둔다.

### Phase E: SR eval

목표: N1.5에서 instruction-fixed steering이 실제 Success Rate를 올리는지 검증한다.

기본 matrix:

- baseline: no steering
- VL always-on: beta `0.1`, `0.3`
- DiT always-on: beta `0.1`, `0.3`
- type-matched: cell별 best pathway만 적용

episode 수:

- 1차: 20 episode per condition per cell
- 유망한 조건: 50 episode per condition으로 확장

평가 기준:

- 같은 cell 안에서 baseline과 steered를 비교한다.
- seed band는 collection과 같은 `100000` 계열을 사용하되, eval leakage를 피하려면 held-out seed
  manifest를 별도로 둔다.
- held-out manifest는 collection manifest를 `--exclude-selected-seeds`로 넘겨 같은
  `cell_id/scenario_seed`가 재사용되지 않음을 selector 단계에서 보장한다.
- steering eval planner도 `--forbid-selected-overlap <collection_manifest>`를 받아 eval manifest와
  collection/conceptor manifest가 같은 `cell_id/scenario_seed`를 공유하면 matrix를 쓰기 전에 실패한다.
- 결과표는 cell별 `baseline_sr`, `steered_sr`, `delta_sr`, Wilson CI, `n_success/n_total`을
  기본 컬럼으로 둔다.
- task 평균은 cell별 결과를 만든 뒤 보조로만 낸다.

현재 eval planning 상태:

- planner:
  `scripts/safe/groot_n15/robocasa/steer/plan_instruction_steering_eval.py`
- runner:
  `scripts/safe/groot_n15/robocasa/steer/run_instruction_steering_eval.py`
- matrix:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/steer_eval/steer_eval_instruction_fixed15_alpha_all_b01_b03/eval_matrix.tsv`
- rows: 97 = 7 baseline + 50 DiT steered + 40 VL steered.
- default condition expansion: fitted 7 cell, selected alpha 전체, beta `{0.1,0.3}`.
- `server_command`는 LeRobot HTTP server를 `--collect --capture-vl`로 띄우고 steering 조건에서는
  `--steering-npz`, `--steering-pathway`, `--steering-beta`, `--steering-alpha`, `--steering-key C_steer`
  를 명시한다.
- `collect_command`는 RoboCasa container에서
  `collect_instruction_fixed_http_features.py --max-episodes-per-cell 20`을 실행한다.
- command 안의 runtime path는 모두 container path(`/temporal_vla/...`)로 변환한다. host absolute path를
  container에 넘기지 않는다.
- 현재 matrix는 기존 selected seed manifest를 사용한다. 최종 SR claim 전에는 held-out eval seed
  manifest로 교체해야 leakage risk를 줄일 수 있다.
- held-out manifest 생성 예시:

```bash
python scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py \
  --cells-config configs/robocasa/n15_instruction_fixed_cells.tsv \
  --output-tsv outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_50ep/manifests/heldout_eval_instruction_seeds.tsv \
  --ep-meta-dir outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_50ep/heldout_eval_ep_meta \
  --target-per-cell 20 \
  --seed-start 400000 \
  --max-seeds-per-cell 20000 \
  --exclude-selected-seeds outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv \
  --resume
```

- aligned steering planner는 이 manifest를 `--selected-seeds <heldout_eval_instruction_seeds.tsv>`로
  받아야 하며, `--forbid-selected-overlap <collection_selected_instruction_seeds.tsv>`를 같이 넘긴다.
  eval collect도 natural reset `ep_meta`를 authority로 두므로 `--replay-ep-meta`를 켜지 않는다.
- runner는 matrix row를 순차 실행한다. 각 row마다 LeRobot HTTP server를 `subprocess.Popen`으로
  띄우고 `/health`를 기다린 뒤 RoboCasa collect command를 실행하고, pkl success count를
  `results.tsv`에 기록한다.
- server process가 health 통과 전에 종료되면 즉시 실패한다. Docker API permission 같은 서버 시작
  실패가 10분 timeout으로 숨지 않게 하기 위한 gate다.
- planner는 각 row의 `docker compose run` server command에 deterministic `--name`을 넣는다.
  runner는 종료 시 일반 process termination 뒤에도 해당 named container를 `docker rm -f`로
  정리한다. 이는 row 사이 port reuse를 보장하기 위한 cleanup contract다.
- runtime smoke:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/steer_eval/steer_eval_smoke_open_drawer_right_first_p8402_named/`
  에서 `open_drawer_right` baseline 1ep + DiT `alpha=0.1,beta=0.1` 1ep를 실행했다.
  두 row 모두 pkl/csv/mp4 triplet과 `ep_meta` JSON을 생성했고, 8402 port와 LeRobot 임시 container
  cleanup이 확인됐다.
- 같은 seed `100000` 기준 baseline과 DiT steered의 action trajectory는 다르다:
  45 step 비교에서 action L2 `4.36035`, max abs `1.03992`, step0 hidden L2 `8.05469`.

## 필요한 코드 변경 범위

문서 기준으로 다음 변경이 필요하다. 구현은 항목별로 작게 진행하고, 각 항목이 끝나면 이 표의
상태와 아래 "검증 기록"을 갱신한다.

| 상태 | 목적 | 파일 | 검증 |
|---|---|---|---|
| [x] | instruction cell 목록 단일 출처 | `configs/robocasa/n15_instruction_fixed_cells.tsv` | config row 수 15개, `cell_index` 중복 없음 |
| [x] | seed-only instruction preselection contract | `configs/robocasa/n15_instruction_fixed_cells.tsv`, `select_instruction_seeds.py` | config/selected manifest에 env forcing field 없음, RoboCasa submodule clean |
| [x] | canonical instruction과 일치하는 seed 선택 | `scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py` | collector env path 기준으로 canonical instruction만 manifest에 기록 |
| [x] | sparse instruction scan checkpoint/resume | `scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py` | `*_scan_progress.tsv`가 non-match scan 위치를 저장하고 `--resume`이 이어감 |
| [x] | cell별 seed shard launcher | `scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py` | active/complete shard skip, `--max-new-sessions`, seed-only command 테스트 |
| [x] | cell별 seed shard merge gate | `scripts/safe/groot_n15/robocasa/collect/merge_instruction_seed_shards.py` | shard manifest merge, `--shard-dir` discovery, old env forcing schema reject, `--require-complete` gate |
| [x] | nested layout과 cell metadata 저장 | `scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py` | collector unit test에서 `raw_rollouts/<task>/<cell_id>`와 pkl metadata 확인 |
| [x] | extra metadata 저장 contract | `scripts/safe/groot_n16/robocasa/collect/collect_artifacts.py` | triplet test에서 `cell_id`, `robocasa_task`, `canonical_instruction` 확인 |
| [x] | nested layout verifier | `scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py` | verifier test에서 `raw_rollouts/*/*/*.pkl` 통과 |
| [x] | selected seed manifest 기반 collect wrapper | `scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py` | dry-run이 실행 명령 15 cell을 정확히 출력, ep_meta replay 기본 off |
| [x] | 실행 중 run의 pkl 기반 `ep_meta` archive 보정 | `scripts/safe/groot_n15/robocasa/collect/backfill_instruction_ep_meta.py` | pkl `ep_meta`에서 run-root JSON 재생성, selected manifest `ep_meta_path` 대조 |
| [x] | selected seed scan audit | `scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_scan.py` | cell별 selected/progress/sample/ep_meta integrity TSV 생성 |
| [x] | sampled instruction 분포 audit | `scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_samples.py` | cell별 scanned/matches/top sampled instruction TSV 생성 |
| [x] | cell별 수집 요약 | `scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_cells.py` | fixture에서 `succ/total`, SR, instruction mismatch 수 출력 |
| [x] | N1.5 VL/DiT separation loader | `scripts/safe/groot_n15/robocasa/analyze/instruction_pathway_features.py` | fixture에서 DiT `[4,16,1024]`, VL `[2048]` 로드 |
| [x] | N1.6-aligned N1.5 DiT block residual capture | `scripts/serve/safe_hooks.py`, `scripts/serve/lerobot.py`, `scripts/utils/vla_client.py`, `collect_artifacts.py` | block residual hook, `/health`, VLAClient, pkl writer focused tests 통과 |
| [x] | aligned block residual verifier metadata gate | `scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py` | `feature_axes`, `capture_layers`, `layer_count`, `token_count` expectation 테스트 통과 |
| [x] | aligned smoke용 per-cell command limiter | `scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py` | `--max-episodes-per-cell 1` dry-run이 cell당 1 command만 생성 |
| [x] | aligned block residual 1-cell smoke | `target_instruction_fixed15_block_residual_smoke` | `open_drawer_right` 1ep, DiT `[7,49,1536]`, VL `[2048]`, verifier `partial-ok` |
| [ ] | aligned block residual 15 cell x 50 full collection | `target_instruction_fixed15_block_residual_50ep` | restarted after wrong-layer purge: latest verifier `completed=3 expected=750`, DiT `[7,49,1536]`, VL `[2048]`, layers `[0,2,4,8,10,12,15]` |
| [x] | aligned block residual layer-preserved cache loader | `scripts/safe/groot_n15/robocasa/analyze/instruction_pathway_features.py` | smoke cache keys `dit_layers`, selected `dit_layer*`, `vl`; tests 통과 |
| [x] | aligned `dit_layer<i>`/`vl` conceptor set fit wrapper | `scripts/safe/groot_n15/robocasa/steer/fit_aligned_instruction_conceptors.py` | `dit_layer*` numeric discovery, combined aligned summary, focused tests 통과 |
| [x] | N1.5 cache 기반 cell_id conceptor fit wrapper | `scripts/safe/groot_n15/robocasa/steer/fit_instruction_conceptors.py` | fixture test, actual cache dry-run, 7 eligible cell DiT/VL full fit |
| [x] | VL/DiT conceptor fit 입력 연결 | `scripts/safe/groot_n15/robocasa/steer/fit_instruction_conceptors.py` | `open_drawer_right` DiT/VL smoke에서 cell_id group npz 생성 |
| [x] | 7 eligible cell conceptor fit | `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/conceptor/{dit,vl}/truncated_w10` | DiT 7 rows `(1024,1024)`, VL 7 rows `(2048,2048)` |
| [ ] | 8 imbalance cell top-up 또는 fit-exclude 결정 | `analysis/topup_recommendations.tsv`, `conceptor/*/eligibility_summary.tsv` | 10/10 미달 cell 처리 policy 확정 |
| [x] | N1.5 HTTP steering serve hook 연결 | `scripts/serve/lerobot.py` | fake steering hook registration test 통과 |
| [x] | N1.5 `/health` feature metadata 노출 | `scripts/serve/lerobot.py` | `/health`가 DiT/VL feature metadata 노출 |
| [x] | N1.5 steering eval matrix planner | `scripts/safe/groot_n15/robocasa/steer/plan_instruction_steering_eval.py` | 97-row alpha/beta matrix 생성, container path 검증 |
| [x] | N1.5 steering eval row runner | `scripts/safe/groot_n15/robocasa/steer/run_instruction_steering_eval.py` | row filtering, dry-run result TSV, server process fast-fail, named container cleanup |
| [x] | N1.5 steering eval runtime smoke | `steer_eval_smoke_open_drawer_right_first_p8402_named` | baseline/DiT 2 rows, pkl triplet, verifier `partial-ok`, port/container cleanup 확인 |
| [x] | held-out eval seed exclusion gate | `scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py` | `--exclude-selected-seeds`로 collection seed overlap skip/reject 테스트 통과 |
| [x] | held-out eval planner overlap gate | `scripts/safe/groot_n15/robocasa/steer/plan_instruction_steering_eval.py` | `--forbid-selected-overlap`로 eval/collection seed overlap reject 테스트 통과 |
| [ ] | N1.5 steering eval held-out manifest | seed selector/materializer | collection seed와 분리된 eval seed manifest 생성 |
| [x] | aligned `dit_layer<i>` eval planner mapping | `scripts/safe/groot_n15/robocasa/steer/plan_instruction_steering_eval.py` | `dit_layer0` summary가 `--steering-pathway dit --steering-layer 0` command 생성 |

가장 먼저 구현할 것은 seed selector다. 이게 없으면 instruction-fixed collection이 아니라
task-random collection이 된다.

## 최종 수집 상태

Run root:

```text
outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/
```

최종 상태:

- raw rollout pkl: 750개, 15 instruction cell x 50 episode.
- full verifier: `completed=750 expected=750`, `status=ok`.
- feature scheme: DiT final `groot_n15_dit_action_tokens_pre_decode` with
  `hidden_states[step].shape == (4,16,1024)`, VL `groot_n15_vlln_seq_meanpool` with
  `vl_hidden_states[step].shape == (2048,)`.
- rollout metadata: `scenario_seed == inference_seed`, `max_episode_steps=720`,
  `n_action_steps=16`, `video_fps=20`, `steps_per_render=2`.
- instruction replay: `instruction_mismatch=0` and `has_vl_hidden_states=50` for every cell in
  `analysis/instruction_success_rates.tsv`.
- selected manifest repair backup:
  `manifests/selected_instruction_seeds.before_repair.tsv`.
- final selected manifest:
  `manifests/selected_instruction_seeds.tsv`.
- final SR table:
  `analysis/instruction_success_rates.tsv`.
- top-up decision table:
  `analysis/topup_recommendations.tsv`.

N1.6-aligned block residual smoke root:

```text
outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_smoke/
```

Aligned restart first-pkl smoke 상태:

- collected cell: `ppcs_onion`, first 3 episodes after restart.
- pkl shape: DiT `hidden_states[step].shape == (7,49,1536)`, VL
  `vl_hidden_states[step].shape == (2048,)`.
- metadata: `capture_layers == [0,2,4,8,10,12,15]`, `layer_count == 7`,
  `token_count == 49`, `model_action_horizon == 16`, `max_episode_steps == 720`.
- verifier: restarted full-root 기준 `completed=3 expected=750`, `status=partial-ok`.
- 이 smoke는 wrong-layer run 삭제 후 새 layer set shape/metadata contract 확정용이다.

N1.6-aligned block residual full collection root:

```text
outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_50ep/
```

Aligned full collection 상태:

- status: running.
- tmux:
  - `lerobot_n15_block_residual_server`
  - `n15_block_residual50_collect`
- server health: `feature_kind=groot_n15_dit_block_residual_tokens`,
  `feature_axes=["layer","model_token","feature_dim"]`,
  `groot_dit_capture_layers=[0,2,4,8,10,12,15]`, VL dim `2048`.
- first pkl:
  `raw_rollouts/PickPlaceCounterToStove/ppcs_onion/task0--ep0--succ1.pkl`.
- first pkl shape: DiT `hidden_states[0].shape == (7,49,1536)`, VL
  `vl_hidden_states[0].shape == (2048,)`.
- first pkl metadata: `scenario_seed == inference_seed == 100000`,
  `max_episode_steps=720`, `n_action_steps=16`, `ep_meta` present.
- initial verifier: `completed=2 expected=750`, `status=partial-ok`.

최종 cell별 SR:

| cell_id | task | success/total | SR |
|---|---|---:|---:|
| ppcs_onion | PickPlaceCounterToStove | 41/50 | 82.0% |
| ppcs_apple | PickPlaceCounterToStove | 44/50 | 88.0% |
| ppdc_tongs | PickPlaceDrawerToCounter | 15/50 | 30.0% |
| ppdc_wooden_spoon | PickPlaceDrawerToCounter | 8/50 | 16.0% |
| ppcc_potato | PickPlaceCounterToCabinet | 43/50 | 86.0% |
| ppcc_bread | PickPlaceCounterToCabinet | 39/50 | 78.0% |
| open_cabinet_door | OpenCabinet | 20/50 | 40.0% |
| open_drawer_right | OpenDrawer | 15/50 | 30.0% |
| open_drawer_left | OpenDrawer | 23/50 | 46.0% |
| close_toaster_oven_door | CloseToasterOvenDoor | 24/50 | 48.0% |
| turn_on_microwave | TurnOnMicrowave | 5/50 | 10.0% |
| turn_on_sink_faucet | TurnOnSinkFaucet | 19/50 | 38.0% |
| navigate_fridge | NavigateKitchen | 3/50 | 6.0% |
| navigate_coffee_machine | NavigateKitchen | 1/50 | 2.0% |
| coffee_setup_mug | CoffeeSetupMug | 3/50 | 6.0% |

Conceptor fit 기준으로 보면 `success >= 10 and failure >= 10`을 만족하지 못하는 cell은 8개다.
성공 부족 cell은 `ppdc_wooden_spoon`, `turn_on_microwave`, `navigate_fridge`,
`navigate_coffee_machine`, `coffee_setup_mug`이고, 실패 부족 cell은 `ppcs_onion`,
`ppcs_apple`, `ppcc_potato`다. 이 cell들은 +25 episode 단위 top-up 또는 fit 제외 판단이 필요하다.

| cell_id | issue | current | observed-rate estimate | next chunk |
|---|---|---:|---:|---:|
| ppcs_onion | failure<10 | 41S / 9F | +6 ep | +25 |
| ppcs_apple | failure<10 | 44S / 6F | +34 ep | +50 |
| ppcc_potato | failure<10 | 43S / 7F | +22 ep | +25 |
| ppdc_wooden_spoon | success<10 | 8S / 42F | +13 ep | +25 |
| turn_on_microwave | success<10 | 5S / 45F | +50 ep | +50 |
| navigate_fridge | success<10 | 3S / 47F | +117 ep | +100 or fit-exclude |
| navigate_coffee_machine | success<10 | 1S / 49F | +450 ep | fit-exclude likely |
| coffee_setup_mug | success<10 | 3S / 47F | +117 ep | +100 or fit-exclude |

다음 실행 단계:

1. running 중인 `target_instruction_fixed15_block_residual_50ep`를 완료하고 verifier를 통과시킨다.
2. 완료 후 aligned DiT `[7,49,1536]` + VL `[2048]` cache를 만들고, N1.6 steering 문서와 같은
   pathway 단위로 separation/conceptor fit을 다시 계산한다.

```bash
python scripts/safe/groot_n15/robocasa/analyze/instruction_pathway_features.py \
  outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_50ep/raw_rollouts \
  --output-npz outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_50ep/analysis/pathway_separation/aligned_dit_layers_vl_step_features_mean_mean.npz \
  --require-vl \
  --preserve-dit-layers
```

Aligned pathway set dry-run fit:

```bash
python scripts/safe/groot_n15/robocasa/steer/fit_aligned_instruction_conceptors.py \
  --cache outputs/eval/robocasa/groot_n15/target_instruction_fixed15_block_residual_50ep/analysis/pathway_separation/aligned_dit_layers_vl_step_features_mean_mean.npz \
  --agg-mode truncated \
  --max-len 10 \
  --dry-run
```

이 명령은 cache 안의 `dit_layer0/2/4/8/10/12/15`와 `vl`을 자동 발견해
`conceptor/aligned_eligibility_summary.tsv`를 만든다. 특정 layer만 확인해야 하면 같은 wrapper에
`--pathway dit_layer0`처럼 pathway filter를 넘긴다.

Layer-specific eval planner는 conceptor fit summary의 pathway key가 `dit_layer<i>`이면 server command에서
`--steering-pathway dit --steering-layer <i>`로 변환한다. `pathway` column과 condition name은
`dit_layer<i>`를 유지해 어떤 feature key에서 fit한 matrix인지 추적 가능하게 둔다.

3. final-DiT/VL dataset 기준 8개 10/10 미달 cell의 top-up 또는 fit-exclude 여부를 정한다. 현재 `truncated_w10`
   eligibility 기준 fit 가능한 cell은 7개다:
   `close_toaster_oven_door`, `open_cabinet_door`, `open_drawer_left`,
   `open_drawer_right`, `ppcc_bread`, `ppdc_tongs`, `turn_on_sink_faucet`.
4. final-DiT conceptor full fit은 위 7개 eligible cell에 대해 DiT/VL 모두 완료했고, steering
   runtime smoke도 통과했다. 하지만 N1.6 기준으로 맞춰 가려면 aligned full run 결과를 primary로 둔다.
5. steering eval은 aligned conceptor fit 후 held-out eval seed manifest를 만들어 진행한다.

## 이전 seed scan 및 수집 상세 로그

아래 항목은 완료 전 진행 로그다. 현재 authority는 위 "최종 수집 상태"와 full verifier 결과다.

- `n15_instruction_seedprobe_v4`와 `n15_instruction_seedselect_50`은 env constructor/object forcing
  shortcut을 포함한 superseded run이다. 두 run의 artifact는 최종 authority로 쓰지 않는다.
- `n15_instruction_seedselect_50` tmux session은 중단했다.
- superseded partial manifest:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv`
- RoboCasa submodule local patch는 제거했고, `src/benchmarks/robocasa`는 clean 상태를 유지한다.
- 현재 확정 contract는 natural seed scan이다. PnP object cell도 env constructor forcing 없이
  collection env path에서 canonical instruction과 match되는 `scenario_seed`를 찾는다.
- 기존 serial natural seed precollection root:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_natural/`
  이 run은 너무 느려 중단했다. partial artifact는 참고만 하고 최종 selected manifest로 쓰지 않는다.
- 새 sharded natural seed precollection root:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_sharded_natural/`
- 당시 실행 중이던 shard tmux sessions:
  `n15_seedscan_navigate_fridge_r2`,
  `n15_seedscan_ppcc_bread_r4`,
  `n15_seedscan_ppcc_potato_r3`,
  `n15_seedscan_ppcc_potato_r4`,
  `n15_seedscan_ppcc_potato_r5`,
  `n15_seedscan_ppcc_potato_r7`.
- sharded partial merge:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_sharded_natural/manifests/selected_instruction_seeds.partial.tsv`
  최신 audit 기준 742 selected seed가 들어 있는 partial artifact이며 collection 입력으로 쓰지 않는다.
- sharded seed scan audit:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_sharded_natural/analysis/seed_scan_status.tsv`
  를 최종 collection 전 gate로 쓴다. 최신 갱신 기준 14 cell이 50/50 complete이고,
  `navigate_fridge`만 42/50으로 남아 있다. `--require-complete`는 아직 expected failure이며
  collection은 시작하지 않는다.
- sharded sample distribution audit:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_sharded_natural/analysis/seed_sample_summary.tsv`
  를 sparse instruction cell 진단에 쓴다. 현재 `ppcc_potato`, `ppcc_bread`는 scanned sample에서
  match rate가 약 1.10%, 2.34%라 계속 scan 중이고, `navigate_fridge`,
  `navigate_coffee_machine`은 각각 canonical match 42개, 35개가 확인됐다. `coffee_setup_mug`은
  canonical instruction match rate 1.0이다.
- 0-match cell을 source 기준으로 점검한 결과, 지금은 canonical string 오류보다 sparse sampling으로
  해석한다. `PickPlaceCounterToCabinet`은 target object가 `obj_groups="all"` 및 `graspable=True`
  에서 sampled되고, `get_ep_meta()`는 `Pick the {obj_lang} from the counter and place it in the
  cabinet.` 형태를 만든다. object registry에는 `potato`, `bread`, `bread_flat`이 존재하고,
  `get_obj_lang()`은 `bread flat`을 `bread`로 치환한다. 반면 `PickPlaceCounterToStove`는
  `obj_groups="food"` 및 `cookable=True`라 onion/apple scan보다 `ppcc_*` scan이 훨씬 희소하다.
  정적 metadata 파싱 기준 `all` graspable 후보는 112개이고, food+graspable+cookable 후보는
  28개다.
- `NavigateKitchen`은 target 후보에 `CoffeeMachine`과 fridge 계열 fixture를 포함하고,
  instruction은 `Navigate to the {target_fixture.nat_lang}.`로 생성된다. 따라서 `navigate_fridge`와
  `navigate_coffee_machine`도 source상 불가능하다고 판단하지 않는다. 이후 own shard에서도 두 cell
  모두 canonical match 1개가 확인됐다.
- 결론: `--require-complete` merge gate 전까지 0-match cell을 교체하지 않는다. 계속 sharded
  seed scan을 진행하고, target count를 못 채우는 cell이 있으면 sample summary와 source evidence를
  함께 보고한 뒤 instruction 교체 여부를 별도 결정한다.
- `ppcc_potato`/`ppcc_bread` 병목을 줄이기 위해 같은 cell의 disjoint seed range shard를 추가했다.
  `ppcc_potato_r1.tsv`, `ppcc_bread_r1.tsv`는 seed `200000`부터 scan하며, 기존 base shard는
  `100000` band를 계속 맡는다. merge는 row의 `(cell_id, scenario_seed)`로 중복을 막고 파일 이름은
  보지 않으므로 range shard를 같은 `--shard-dir` 입력으로 합칠 수 있다.
- `ppcs_onion`/`ppcs_apple`/`navigate_fridge`/`navigate_coffee_machine`도 낮은 match rate 때문에
  launcher의 `--shard-suffix _r1 --seed-start 200000` 옵션으로 추가 range shard를 시작했다.
- `ppcs_onion`/`ppcs_apple`/`ppcc_potato`/`ppcc_bread`는 object sampling 병목을 줄이기 위해
  seed `300000`부터 시작하는 `_r2` range shard도 추가했다.
- `ppdc_tongs`/`ppdc_wooden_spoon`은 seed `200000`부터 시작하는 `_r1` range shard를 추가했다.
- 일부 incomplete shard가 complete 전에 종료되어, 같은 suffix와 `--resume`으로 복구했다.
  `open_cabinet_door`, `turn_on_sink_faucet`, `navigate_fridge`, `navigate_coffee_machine`은 seed
  `300000`부터 시작하는 `_r2` range shard를 추가했고, `ppcs_onion_r2`도 같은 range에서 재실행했다.
- `ppcc_bread_r2`/`ppcc_potato_r2`/`ppcs_onion_r2`도 조기 종료 후 `--resume`으로 재실행했다.
  `ppcc_bread`는 match rate가 1% 미만이라 seed `400000`부터 시작하는 `_r3` range shard도
  추가했다. 추가 shard는 CPU load가 18대로 내려왔을 때 시작했고, 시작 후 load가 21대로 올라
  더 이상 확장하지 않는다.
- 이후 load가 다시 16대로 내려와 `ppcc_bread` 전용 seed `500000` 시작 `_r4` range shard를
  한 개만 추가했다. 실행 확인 기준 selector pid는 14149이고, log는
  `[cell] ppcc_bread: target=50 existing=0 start_seed=500000`로 시작했다.
- `ppcc_potato`도 base/r1이 종료되고 `_r2`만 active라 seed `400000` 시작 `_r3` range shard를
  한 개 추가했다. 실행 확인 기준 selector pid는 14226이고, log는
  `[cell] ppcc_potato: target=50 existing=0 start_seed=400000`로 시작했다.
- 이후 live merge는 426 selected seed까지 올랐다. load average가 20.99/19.39/18.75 수준이라
  추가 range shard는 더 띄우지 않고 현재 active shard 진행을 유지한다.
- `ppcc_potato_r3`는 12 sample 뒤 종료되어 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] ppcc_potato: target=50 existing=0 start_seed=400011`로 이어졌다.
- load가 다시 17대로 내려왔을 때 near-complete cell인 `open_drawer_right`와
  `turn_on_sink_faucet` base shard를 resume했다. 실행 확인 기준 selector pid는 각각 14381,
  14380이고, live merge에서 `open_drawer_right`는 47/50, `turn_on_sink_faucet`은 45/50까지 올랐다.
  이후 load average가 25.32/20.34/19.12로 올라 추가 shard는 보류했다.
- 이후 `open_drawer_right`와 `turn_on_sink_faucet`은 모두 50/50 complete가 됐다. live merge는
  451 selected seed까지 올랐고, `--require-complete`는 아직 expected incomplete다:
  `ppcs_onion` 9/50, `ppcs_apple` 10/50, `ppdc_tongs` 33/50, `ppdc_wooden_spoon` 24/50,
  `ppcc_potato` 4/50, `ppcc_bread` 4/50, `open_cabinet_door` 39/50, `open_drawer_left` 40/50,
  `navigate_fridge` 19/50, `navigate_coffee_machine` 19/50.
- load가 15대로 내려온 뒤 inactive 상태였던 `open_cabinet_door_r2`를 같은 suffix로 append-log
  resume했다. 실행 확인 기준 tmux session은 `n15_seedscan_open_cabinet_r2`이고, log는 기존
  `[scan] ... found=9/50 seed=300024` 뒤에 새 `[cell] ... existing=9 start_seed=300028`가 이어진다.
- post-resume audit에서 live merge는 456 selected seed까지 올랐다. `open_cabinet_door_r2`가 새
  match를 잡아 `open_cabinet_door`는 40/50이 됐고, `--require-complete`는 아직 expected incomplete다:
  `ppcs_onion` 9/50, `ppcs_apple` 11/50, `ppdc_tongs` 34/50, `ppdc_wooden_spoon` 25/50,
  `ppcc_potato` 4/50, `ppcc_bread` 4/50, `open_cabinet_door` 40/50, `open_drawer_left` 42/50,
  `navigate_fridge` 20/50, `navigate_coffee_machine` 19/50.
- `ppcs_onion`은 base shard만 active라 10/50에서 느리게 진행 중이었고, inactive였던
  `ppcs_onion_r1`은 seed `200000` band에서 이미 3개 match를 가지고 있었다. load가 15대인 상태에서
  같은 suffix로 append-log resume했고, log는 기존 `found=3/50 seed=200025` 뒤에 새
  `[cell] ... existing=3 start_seed=200030`로 이어진다.
- latest sequential audit에서 live merge는 470 selected seed까지 올랐다. `open_cabinet_door`는
  45/50까지 진행됐고, `--require-complete`는 아직 expected incomplete다:
  `ppcs_onion` 10/50, `ppcs_apple` 11/50, `ppdc_tongs` 36/50, `ppdc_wooden_spoon` 25/50,
  `ppcc_potato` 4/50, `ppcc_bread` 4/50, `open_cabinet_door` 45/50, `open_drawer_left` 43/50,
  `navigate_fridge` 22/50, `navigate_coffee_machine` 20/50.
- `ppcc_potato`와 `ppcc_bread`가 계속 4/50에 머물러 있어, load가 18 미만인 시점에 range shard를
  각각 한 개씩만 추가했다. `ppcc_potato_r4`는 seed `500000`, `ppcc_bread_r5`는 seed `600000`에서
  시작한다. 두 shard 모두 새 suffix라 기존 manifest와 충돌하지 않는다.
- latest sequential audit에서 live merge는 480 selected seed까지 올랐다. `ppcc_potato_r4`가 즉시
  1개 match를 잡아 `ppcc_potato`는 5/50이 됐고, `open_cabinet_door`는 47/50,
  `open_drawer_left`는 45/50까지 진행됐다. `--require-complete`는 아직 expected incomplete다:
  `ppcs_onion` 11/50, `ppcs_apple` 12/50, `ppdc_tongs` 36/50, `ppdc_wooden_spoon` 25/50,
  `ppcc_potato` 5/50, `ppcc_bread` 4/50, `open_cabinet_door` 47/50, `open_drawer_left` 45/50,
  `navigate_fridge` 25/50, `navigate_coffee_machine` 21/50.
- latest live merge는 482 selected seed까지 올랐다. `ppcs_onion_r1`과 `ppcc_bread_r2`는 active
  tmux 목록에서 빠졌고, 남은 active shard 진행을 유지한다. `--require-complete`는 아직 expected
  incomplete다: `ppcs_onion` 11/50, `ppcs_apple` 12/50, `ppdc_tongs` 36/50,
  `ppdc_wooden_spoon` 25/50, `ppcc_potato` 5/50, `ppcc_bread` 4/50,
  `open_cabinet_door` 47/50, `open_drawer_left` 46/50, `navigate_fridge` 25/50,
  `navigate_coffee_machine` 22/50.
- latest sequential audit에서 live merge는 488 selected seed까지 올랐다. `open_cabinet_door`는
  48/50까지 진행됐고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 12/50,
  `ppcs_apple` 12/50, `ppdc_tongs` 37/50, `ppdc_wooden_spoon` 25/50, `ppcc_potato` 5/50,
  `ppcc_bread` 4/50, `open_cabinet_door` 48/50, `open_drawer_left` 46/50,
  `navigate_fridge` 26/50, `navigate_coffee_machine` 23/50.
- latest sequential audit에서 live merge는 518 selected seed까지 올랐다. `open_cabinet_door`와
  `open_drawer_left`가 모두 50/50 complete가 되어 complete cell은 7개가 됐다. 완료된
  `open_cabinet_door_r2`는 더 이상 collection gate에 필요 없어서 tmux session과 container
  orphan selector PID를 정리했다. `--require-complete`는 아직 expected incomplete다:
  `ppcs_onion` 13/50, `ppcs_apple` 14/50, `ppdc_tongs` 42/50, `ppdc_wooden_spoon` 30/50,
  `ppcc_potato` 6/50, `ppcc_bread` 8/50, `navigate_fridge` 31/50,
  `navigate_coffee_machine` 24/50.
- `ppcs_onion`과 `ppcs_apple`이 12/50에서 느리게 진행 중이라, inactive 상태였던
  `ppcs_onion_r2`와 `ppcs_apple_r2`를 같은 suffix로 append-log resume했다. 실행 확인 기준
  selector pid는 각각 14830, 15204이고, load average는 16.95/17.16/16.93이다.
- `ppdc_wooden_spoon_r1`은 29/50 전에 session이 빠져 같은 suffix로 append-log resume했다.
  새 실행은 `[cell] ppdc_wooden_spoon: target=50 existing=13 start_seed=200095`로 이어졌고,
  최신 selector pid는 15364다.
- `ppcc_potato`가 5/50에서 가장 느린 병목이라, inactive 상태였던 `ppcc_potato_r1`을 같은 suffix로
  append-log resume했다. 새 실행은 `[cell] ppcc_potato: target=50 existing=2 start_seed=200024`로
  이어졌고, selector pid는 15049다. 추가 후 load average는 16.83/16.81/16.75다.
- `navigate_fridge_r2`도 30/50 전에 session이 빠져 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] navigate_fridge: target=50 existing=9 start_seed=300114`로 이어졌고, selector pid는 15127다.
- `ppcc_potato_r4`도 match 0/50 상태로 session이 빠져 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] ppcc_potato: target=50 existing=0 start_seed=500038`로 이어졌고, selector pid는 15281다.
- `ppcc_bread_r4`도 match 0/50 상태로 session이 빠져 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] ppcc_bread: target=50 existing=0 start_seed=500067`로 이어졌고, selector pid는 15441다.
- latest sequential audit에서 live merge는 527 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 14/50,
  `ppcs_apple` 15/50, `ppdc_tongs` 43/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50,
  `ppcc_bread` 10/50, `navigate_fridge` 31/50, `navigate_coffee_machine` 26/50.
  collection은 시작하지 않는다.
- `ppcc_bread_r5`는 완료 전 session이 빠져 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] ppcc_bread: target=50 existing=4 start_seed=600063`로 이어졌고, selector pid는 15518다.
- `navigate_coffee_machine_r1`도 완료 전 session이 빠져 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] navigate_coffee_machine: target=50 existing=6 start_seed=200047`로 이어졌고, selector pid는
  15589였다. `navigate_coffee_machine`은 resume 후 24/50에서 26/50으로 늘었고, 이후 tmux/pgrep 기준
  다시 inactive 상태다.
- `ppcc_potato_r2`는 한 차례 tmux와 container pgrep 기준 inactive였고 orphan selector가 없었다. 당시
  load average가 18대라 즉시 추가 복구하지 않고, active shard 진행을 우선 봤다.
- latest sequential audit에서 live merge는 536 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 15/50,
  `ppcs_apple` 16/50, `ppdc_tongs` 43/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50,
  `ppcc_bread` 11/50, `navigate_fridge` 33/50, `navigate_coffee_machine` 30/50.
  collection은 시작하지 않는다.
- load average가 16대로 내려와 `ppcc_potato_r2`를 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] ppcc_potato: target=50 existing=2 start_seed=300114`로 이어졌고, selector pid는 15684다.
- latest sequential audit에서 live merge는 539 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 17/50,
  `ppcs_apple` 16/50, `ppdc_tongs` 44/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50,
  `ppcc_bread` 11/50, `navigate_fridge` 33/50, `navigate_coffee_machine` 30/50.
  collection은 시작하지 않는다.
- `ppcs_onion_r1`은 완료 전 session이 빠져 같은 suffix로 append-log resume했다. 새 실행은
  `[cell] ppcs_onion: target=50 existing=4 start_seed=200047`로 이어졌고, selector pid는 15767다.
  resume 후 `ppcs_onion`은 17/50까지 늘었다.
- `ppcs_apple` base shard도 완료 전 session이 빠져 append-log resume했다. 새 실행은
  `[cell] ppcs_apple: target=50 existing=5 start_seed=100161`로 이어졌고, selector pid는 15838다.
  load average가 21대까지 올라 추가 shard 확장은 보류한다.
- post-check에서 `ppcs_apple_r1`과 `navigate_fridge_r2`는 tmux와 container pgrep 기준 inactive가 됐다.
  현재 load average가 18~19대라 즉시 복구하지 않고 active shard 진행을 우선 본다.
- latest sequential audit에서 live merge는 544 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 18/50,
  `ppcs_apple` 18/50, `ppdc_tongs` 46/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50,
  `ppcc_bread` 11/50, `navigate_fridge` 33/50, `navigate_coffee_machine` 30/50.
  collection은 시작하지 않는다.
- load average가 16대로 내려와 inactive였던 `navigate_fridge_r2`를 같은 suffix로 append-log resume했다.
  새 실행은 `[cell] navigate_fridge: target=50 existing=10 start_seed=300148`로 이어졌고, selector
  pid는 15927다.
- latest sequential audit에서 live merge는 552 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 18/50,
  `ppcs_apple` 18/50, `ppdc_tongs` 47/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 7/50,
  `ppcc_bread` 12/50, `navigate_fridge` 38/50, `navigate_coffee_machine` 30/50.
  collection은 시작하지 않는다.
- load average가 16~17대로 내려온 동안 inactive였던 `ppdc_wooden_spoon_r1`,
  `navigate_coffee_machine_r1`, `ppcs_apple_r1`을 같은 suffix로 append-log resume했다. 확인된 selector
  pid는 각각 16010, 16081, 16152다. 이후 load average가 18.75까지 올라 추가 shard 확장은 보류한다.
- `ppcs_onion` base shard는 다시 inactive가 됐고, 현재 `ppcs_onion`은 `_r1`/`_r2` active shard 진행을
  우선 본다.
- latest sequential audit에서 live merge는 556 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 19/50,
  `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 33/50, `ppcc_potato` 7/50,
  `ppcc_bread` 12/50, `navigate_fridge` 38/50, `navigate_coffee_machine` 31/50.
  collection은 시작하지 않는다.
- `ppcc_potato`가 497 scan에 7 match로 가장 느린 병목이라, load average 17대에서 seed `600000`
  시작 `_r5` range shard를 한 개 추가했다. 실행 확인 기준 selector pid는 16235이고, log는
  `[cell] ppcc_potato: target=50 existing=0 start_seed=600000`로 시작했다. 추가 확장은 현재 보류한다.
- latest sequential audit에서 live merge는 559 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 19/50,
  `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 33/50, `ppcc_potato` 7/50,
  `ppcc_bread` 12/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 33/50.
  load average가 19대로 올라 추가 shard 확장은 보류하고 collection은 시작하지 않는다.
- post-check에서 `navigate_fridge_r2`는 tmux와 container pgrep 기준 inactive가 됐다. `navigate_fridge`는
  39/50까지 확보됐지만 load average가 18대라 즉시 복구하지 않고 active shard 진행을 우선 본다.
- latest sequential audit에서 live merge는 562 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 20/50,
  `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 33/50, `ppcc_potato` 8/50,
  `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 33/50.
  collection은 시작하지 않는다.
- `ppcs_apple`이 511 scan에 18 match로 정체라 seed `400000` 시작 `_r3` range shard를 한 개 추가했다.
  실행 확인 기준 selector pid는 16318이고, log는
  `[cell] ppcs_apple: target=50 existing=0 start_seed=400000`로 시작했다. 추가 확장은 현재 보류한다.
- latest sequential audit에서 live merge는 564 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 20/50,
  `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 34/50, `ppcc_potato` 8/50,
  `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 34/50.
  collection은 시작하지 않는다.
- `ppcc_potato_r1`은 한 차례 inactive가 됐지만 load average가 다시 17대로 내려와 같은 suffix로
  append-log resume했다. 실행 확인 기준 selector pid는 16401이고, log는
  `[cell] ppcc_potato: target=50 existing=4 start_seed=200077`로 이어졌다. 추가 확장은 현재 보류한다.
- post-check에서 `ppcc_bread_r4`는 tmux와 container pgrep 기준 inactive가 됐다. `ppcc_bread`는 13/50까지
  확보됐지만 load average가 19대로 올라 즉시 복구하지 않고 active shard 진행을 우선 본다.
- latest sequential audit에서 live merge는 566 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 20/50,
  `ppcs_apple` 19/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 34/50, `ppcc_potato` 8/50,
  `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50.
  load average가 20대로 올라 추가 shard 확장은 보류하고 collection은 시작하지 않는다.
- latest sequential audit에서 live merge는 569 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 20/50,
  `ppcs_apple` 19/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50,
  `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50.
  collection은 시작하지 않는다.
- load average가 16대로 내려와 inactive였던 `ppcc_bread_r3`를 같은 suffix로 append-log resume했다.
  실행 확인 기준 selector pid는 16496이고, log는
  `[cell] ppcc_bread: target=50 existing=3 start_seed=400125`로 이어졌다. 추가 확장은 현재 보류한다.
- latest sequential audit에서 live merge는 570 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 20/50,
  `ppcs_apple` 19/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50,
  `ppcc_bread` 14/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50.
  collection은 시작하지 않는다.
- post-check에서 `ppcc_potato_r1`은 tmux와 container pgrep 기준 inactive가 됐다. `ppcc_potato`는
  8/50으로 가장 큰 병목이지만 load average가 18~19대라 즉시 복구하지 않고 active shard 진행을 우선 본다.
- load average가 16대로 내려와 inactive였던 `ppcc_potato_r1`을 같은 suffix로 append-log resume했다.
  실행 확인 기준 selector pid는 16597이고, log는
  `[cell] ppcc_potato: target=50 existing=4 start_seed=200107`로 이어졌다.
- latest sequential audit에서 live merge는 574 selected seed까지 올랐다. complete cell은 7개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 22/50,
  `ppcs_apple` 20/50, `ppdc_tongs` 49/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50,
  `ppcc_bread` 14/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50.
  collection은 시작하지 않는다.
- load average가 16대로 유지되어 inactive였던 `ppcc_bread_r4`를 같은 suffix로 append-log resume했다.
  실행 확인 기준 selector pid는 16686이고, log는
  `[cell] ppcc_bread: target=50 existing=3 start_seed=500136`로 이어졌다.
  post-resume merge는 아직 574 selected seed로 동일하며 collection은 시작하지 않는다.
- latest sequential audit에서 live merge는 576 selected seed까지 올랐다. complete cell은 8개로
  늘었고, `ppdc_tongs`가 50/50 complete가 되어 `--require-complete` 미완료 목록에서 빠졌다.
  아직 expected incomplete다: `ppcs_onion` 22/50, `ppcs_apple` 20/50,
  `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50,
  `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. collection은 시작하지 않는다.
- 완료된 `ppdc_tongs_r1`은 더 이상 collection gate에 필요 없어서 tmux session과 container orphan
  selector PID 12936을 정리했다. post-check에서 `ppdc_tongs_r1.tsv` selector가 남아 있지 않음을 확인했다.
- `ppcc_potato_r1`은 짧게 재실행된 뒤 다시 inactive가 됐고, sample log는 seed 200117까지 늘었지만
  추가 match는 없었다. 반복 재개 대신 병목을 줄이기 위해 disjoint seed `700000` 시작 `_r6` range shard를
  한 개 추가했다. 실행 확인 기준 selector pid는 16793이고, log는
  `[cell] ppcc_potato: target=50 existing=0 start_seed=700000`로 시작했다.
- latest sequential audit에서 live merge는 583 selected seed까지 올랐다. complete cell은 8개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 23/50,
  `ppcs_apple` 24/50, `ppdc_wooden_spoon` 39/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50,
  `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. collection은 시작하지 않는다.
- latest sequential audit에서 live merge는 587 selected seed까지 올랐다. complete cell은 8개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 24/50,
  `ppcs_apple` 25/50, `ppdc_wooden_spoon` 40/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50,
  `navigate_fridge` 40/50, `navigate_coffee_machine` 35/50. collection은 시작하지 않는다.
- `ppcc_potato`는 8/711 sample로 가장 느린 병목이라 disjoint seed `800000` 시작 `_r7` range shard를
  추가했다. 실행 확인 기준 selector pid는 16959이고, log는
  `[cell] ppcc_potato: target=50 existing=0 start_seed=800000`으로 시작했다.
- post-audit에서 live merge는 587 selected seed로 유지됐다. `ppcc_potato` sample은 8/730으로
  더 늘었지만 추가 match는 없고, `ppcc_bread_r5`와 `navigate_fridge_r2`는 tmux/pgrep 기준 inactive다.
  load average가 18대로 올라 즉시 재실행은 보류했다.
- 이후 load average가 다시 15대로 내려와 high-yield `navigate_fridge_r2`를 같은 suffix로 append-log
  resume했다. 실행 확인 기준 selector pid는 17042이고, log는
  `[cell] navigate_fridge: target=50 existing=17 start_seed=300193`으로 이어졌다. post-launch
  merge는 587 selected seed로 유지됐다.
- latest sequential audit에서 live merge는 589 selected seed까지 올랐다. complete cell은 8개로
  유지되고, `--require-complete`는 아직 expected incomplete다: `ppcs_onion` 24/50,
  `ppcs_apple` 25/50, `ppdc_wooden_spoon` 40/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50,
  `navigate_fridge` 42/50, `navigate_coffee_machine` 35/50. collection은 시작하지 않는다.
- `ppcs_onion`은 24/50인데 active shard가 `_r1` 하나뿐이라, load가 15-16대일 때 `_r2`를
  append-log resume했다. 실행 확인 기준 selector pid는 17125이고, log는
  `[cell] ppcs_onion: target=50 existing=1 start_seed=300127`으로 이어졌다. post-launch merge는
  589 selected seed로 유지됐다.
- 2026-06-11 오전 sequential audit에서 live merge는 742 selected seed까지 올랐다. complete cell은
  14개이고 `--require-complete`는 아직 expected incomplete다: `navigate_fridge` 42/50.
  나머지 14개 cell은 mismatch 0, duplicate 0, ep_meta missing 0으로 50/50 complete다.
  collection은 시작하지 않는다.
- 같은 audit 중 `summarize_instruction_seed_scan.py`를 `--path-base` run root로 실행하면
  `ep_meta_path`가 이중으로 붙어 0 complete로 보인다. selected manifest의 `ep_meta_path`는 repo 기준
  상대경로이므로 기본 cwd 기준으로 실행해야 한다.
- `tests/test_groot_n15_http_feature_collect.py` focused test가 host env에서 `gymnasium` top-level import로
  7개 실패했다. root cause는 env 생성에서만 필요한 GR00T wrapper config와
  `wrap_groot_robocasa_eval_env`를 `http_feature_collect.py` top-level에서 import한 것이다. 해당 import를
  `make_env()` 내부로 늦춘 뒤 focused regression은 40 passed로 회복됐다.
- `navigate_fridge`가 유일한 gate blocker이고 active shard가 빠져 있어, load average 5대에서
  `navigate_fridge_r2`를 같은 suffix로 append-log resume했다. 실행 확인 기준 tmux session은
  `n15_seedscan_navigate_fridge_r2`이고, log는 기존 `[match] ... found=2/33 seed=300199` 뒤에
  새 `[cell] navigate_fridge: target=50 existing=19 start_seed=300207`로 이어진다.
- 이후 final shard merge gate를 통과했다. 최종 manifest는
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv`
  이며 header 포함 751줄, 15 instruction cell x 50 seed다. `--require-complete` merge가 성공했고,
  seed scan status는 15/15 cell complete로 갱신됐다.
- 완료된 seed scanner tmux session과 orphan selector process는 정리했다. 현재 seed를 찾거나 모으는
  phase는 끝났고, final selected seed manifest는 유지 중이다.
- 본수집은
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep/raw_rollouts/<task>/<cell_id>/`
  layout으로 pkl/csv/mp4 triplet을 쓴다. 서버는
  `lerobot_n15_instruction_feature_server` tmux session에서 `--collect --capture-vl`로 실행 중이며,
  `/health` 기준 DiT feature는 `groot_n15_dit_action_tokens_pre_decode` shape `4,16,1024`,
  VL feature는 `groot_n15_vlln_seq_meanpool` dim `2048`이다.
- 2026-06-11에 시작한 첫 본수집은 `--max-episode-steps`를 명시하지 않아 task registry horizon으로
  실행됐다. 이 run은 `ppcs_onion`/`ppcs_apple` 100 episode와 `ppdc_tongs` 일부 episode를 만들었지만,
  `max_episode_steps`가 400/500으로 섞여 COAST/N1.6-aligned 기준과 다르다. 해당 collection tmux
  session은 중단했고, 쓸모 없는 `raw_rollouts`, partial SR summary, collection log는 삭제했다.
  seed manifest와 seed-scan audit은 보존했다.
- 720 cap으로 본수집을 재시작했다. 실행 session은 `n15_instruction_fixed50_collect_720`이고,
  dry-run 기준 750개 command 모두 `--max-episode-steps 720`을 포함한다. 첫 산출물
  `raw_rollouts/PickPlaceCounterToStove/ppcs_onion/task0--ep0--succ1.{pkl,csv,mp4}`는 pkl payload에서
  `scenario_seed=100000`, `inference_seed=100000`, `max_episode_steps=720`, `n_action_steps=16`을
  확인했다. `ppcs_onion`은 50/50 base 수집을 완료했고, success 41 / failure 9다. 별도 파일명 감사에서
  episode index 0..49가 모두 존재하고 missing은 없다. 이후 `ppcs_apple`이 시작됐고 verifier 시점의
  전체 raw_rollouts는 51개 artifact 상태이며 `--allow-partial --expected-max-episode-steps 720` 기준
  `status=partial-ok`다. 이후 `ppcs_apple`도 50/50 base 수집을 완료했고, success 44 / failure 6이다.
  별도 파일명 감사에서 `ppcs_apple` episode index 0..49가 모두 존재하고 missing은 없다. 다음 cell
  `ppdc_tongs`도 50/50 base 수집을 완료했고, success 15 / failure 35다. 별도 파일명 감사에서
  `ppdc_tongs` episode index 0..49가 모두 존재하고 missing은 없다. 완료 verifier 시점의 전체
  raw_rollouts는 `ppdc_wooden_spoon` 2개 포함 152개 artifact 상태이며, `--allow-partial`과
  `--expected-max-episode-steps 720` 기준 `status=partial-ok`다. selected seed manifest 대조 결과
  `missing_manifest=0`, `seed_mismatch=0`, `cell_mismatch=0`, `task_mismatch=0`이고 같은
  task/seed duplicate와 같은 task_id/episode/seed duplicate도 0이다. `ppcs_onion`과 `ppcs_apple`은
  conceptor fit 최소 조건 failure >= 10에 각각 1개, 4개 모자라므로 base 15x50 완료 후 top-up
  후보로 둔다.
- 이어서 `ppdc_wooden_spoon`이 25/50 checkpoint에 도달했다. checkpoint verifier 시점의 전체
  raw_rollouts는 175개 artifact 상태이며, `ppdc_wooden_spoon`은 success 5 / failure 20이고
  `status=partial-ok`다. selected seed manifest 대조 결과 `missing_manifest=0`, `seed_mismatch=0`,
  `cell_mismatch=0`, `task_mismatch=0`이고 같은 task/seed duplicate와 같은 task_id/episode/seed
  duplicate도 0이다.
- 이후 `ppdc_wooden_spoon`도 50/50 base 수집을 완료했고, success 8 / failure 42다. 별도 파일명
  감사에서 episode index 0..49가 모두 존재하고 missing은 없다. 완료 verifier 시점의 전체 raw_rollouts는
  `ppcc_potato` 1개 포함 201개 artifact 상태이며 `status=partial-ok`다. selected seed manifest 대조
  결과 `missing_manifest=0`, `seed_mismatch=0`, `cell_mismatch=0`, `task_mismatch=0`이고 같은
  task/seed duplicate와 같은 task_id/episode/seed duplicate도 0이다. `ppdc_wooden_spoon`은 success가
  10개 미만이라 base 15x50 완료 후 top-up 후보로 둔다.
- 다음 cell `ppcc_potato`는 26/50 checkpoint에서 success 23 / failure 3이다. checkpoint verifier
  시점의 전체 raw_rollouts는 226개 artifact 상태이며 `status=partial-ok`다. selected seed manifest
  대조 결과 `missing_manifest=0`, `seed_mismatch=0`, `cell_mismatch=0`, `task_mismatch=0`이고 같은
  task/seed duplicate와 같은 task_id/episode/seed duplicate도 0이다. 현재는 failure가 10개 미만이라
  base 50 완료 후 top-up 여부를 다시 판단한다.
- 이후 `ppcc_potato`도 50/50 base 수집을 완료했고, success 43 / failure 7이다. 별도 파일명 감사에서
  episode index 0..49가 모두 존재하고 missing은 없다. 완료 verifier 시점의 전체 raw_rollouts는
  `ppcc_bread` 2개 포함 252개 artifact 상태이며 `status=partial-ok`다. selected seed manifest 대조
  결과 `missing_manifest=0`, `seed_mismatch=0`, `cell_mismatch=0`, `task_mismatch=0`이고 같은
  task/seed duplicate와 같은 task_id/episode/seed duplicate도 0이다. `ppcc_potato`은 failure가
  10개 미만이라 base 15x50 완료 후 top-up 후보로 둔다.
- 다음 cell `ppcc_bread`는 28/50 checkpoint에서 success 20이고 verifier 기준 `status=partial-ok`다.
  checkpoint 시점의 전체 raw_rollouts는 278개 artifact 상태이며, selected seed manifest 대조 결과
  `missing_manifest=0`, `seed_mismatch=0`, `cell_mismatch=0`, `task_mismatch=0`이고 같은 task/seed
  duplicate와 같은 task_id/episode/seed duplicate도 0이다. 현재는 failure가 10개 미만이라 base 50
  완료 후 top-up 여부를 다시 판단한다.
- `launch_instruction_seed_shards.py`는 range shard를 재현 가능하게 띄우기 위해 `--shard-suffix`와
  `--seed-start`를 지원한다. completed 판단도 `<cell_id>*.tsv` seed shard를 합산한다.
- shard 재실행 시 이전 실패 원인을 보존하기 위해 launcher는 shard log를 overwrite하지 않고 append한다.
  같은 suffix를 `--resume`으로 다시 띄워도 기존 traceback/sample progress를 유지한다.
- `ppcs_onion` base shard는 append-log contract로 재실행했다. `ppcs_onion.log`에는 이전
  `[scan] ... found=5/50 seed=100174` 뒤에 새 `[cell] ... existing=5 start_seed=100180`가 이어진다.
- range shard 추가 이후 `summarize_instruction_seed_scan.py`는 같은 cell의 여러
  `_scan_progress.tsv`를 합산하고, `last_scanned_seed`는 progress/sample/selected seed의 최댓값으로
  계산한다.
- serial seed scan audit:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan50_natural/analysis/seed_scan_status.tsv`
  (`ppcs_onion` selected 3/50, mismatch 0, duplicate 0, ep_meta missing 0).
- superseded object-forcing selector process가 남아 있던 것을 확인하고 중단했다. 현재 authority가 있는
  seed selector process는 `target_instruction_fixed15_pathway_seedscan50_sharded_natural`의 cell별
  shard session이다.
- natural seed scan의 known smoke expectation: `ppcs_apple`은 seed `100050`에서 첫 match가 확인된 적이
  있어, PnP object cell은 sparse하지만 scan/resume으로 처리 가능하다.
- natural smoke artifact:
  `outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_seedscan_natural_smoke/`
  (`ppcs_apple` seed `100050`, manifest schema에 env forcing field 없음).

## 검증 checklist

수집 전:

- `scripts/safe/groot_n15/robocasa/README.md`의 N1.5 feature shape contract와 일치한다.
- LeRobot server `/health`가 `supports_features=true`와 VL metadata를 노출한다.
- seed selector output에서 모든 row의 instruction이 canonical instruction과 정확히 일치한다.

수집 후:

- verifier가 모든 cell raw_rollouts에 대해 통과한다.
- `selected_instruction_seeds.tsv`와 실제 pkl의 `scenario_seed`가 일치한다.
- pkl의 `task_description`이 cell table과 일치한다.
- cell별 `succ/total`과 SR이 생성된다.
- success/failure 최소 개수 미달 cell이 명시된다.

분석 후:

- AUROC 표는 cell 단위로 먼저 저장된다.
- VL/DiT 비교는 t=4,8,12,16,20을 모두 포함한다.
- length-only baseline 또는 fixed-t 길이통제 설명이 결과에 붙는다.

eval 후:

- baseline과 steered는 같은 cell, 같은 seed policy로 비교된다.
- 평균 ΔSR보다 cell별 ΔSR이 먼저 보고된다.
- 유망 조건은 20 episode 결과만으로 claim하지 않고 50 episode 확장 대상으로 표시한다.

## 검증 기록

| 날짜 | 변경/실험 | 명령 | 결과 |
|---|---|---|---|
| 2026-06-10 | 계획 문서 초안 작성 | 임시 marker 검색 | 임시 marker 없음 |
| 2026-06-10 | 계획 문서 whitespace 확인 | `git diff --no-index --check /dev/null docs/steering/11_phase4_n15_instruction_fixed_plan.md` | whitespace warning 없음 |
| 2026-06-10 | instruction cell config 추가 | `python -m pytest tests/test_groot_n15_instruction_fixed_cells.py -q` | 1 passed |
| 2026-06-10 | seed/ep_meta selector 추가 | `python -m pytest tests/test_groot_n15_instruction_seed_selector.py -q` | 2 passed |
| 2026-06-10 | selector static/CLI 확인 | `python -m py_compile ...`, `python scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py --help`, `git diff --check -- ...` | 통과 |
| 2026-06-10 | nested collector/cell metadata 추가 | `python -m pytest tests/test_groot_n15_http_feature_collect.py -q` | 7 passed |
| 2026-06-10 | collect artifact 호환 확인 | `python -m pytest tests/test_safe_groot_collect.py::test_safe_triplet_writer_records_model_family_and_transport_metadata -q`, `python -m pytest tests/test_safe_groot_collect.py::test_collect_main_advances_scenario_seed_per_episode -q` | 각 1 passed |
| 2026-06-10 | collector static/CLI 확인 | `python -m py_compile ...`, `python scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py --help`, `git diff --check -- ...` | 통과 |
| 2026-06-10 | nested verifier/cell metadata 추가 | `python -m pytest tests/test_safe_groot_verify_rollout_collection.py -q` | 5 passed |
| 2026-06-10 | verifier static/CLI 확인 | `python -m py_compile ...`, `python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py --help`, `git diff --check -- ...` | 통과 |
| 2026-06-10 | collect wrapper 추가 | `python -m pytest tests/test_groot_n15_instruction_collect_wrapper.py -q` | 2 passed |
| 2026-06-10 | cell summary 추가 | `python -m pytest tests/test_groot_n15_instruction_cell_summary.py -q` | 2 passed |
| 2026-06-10 | wrapper/summary 연결 검증 | `python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_instruction_cell_summary.py tests/test_groot_n15_http_feature_collect.py tests/test_safe_groot_verify_rollout_collection.py -q` | 19 passed |
| 2026-06-10 | N1.5 pathway feature loader 추가 | `python -m pytest tests/test_groot_n15_instruction_pathway_features.py -q` | 2 passed |
| 2026-06-10 | smoke 전 코드 경로 연결 검증 | `python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_instruction_cell_summary.py tests/test_groot_n15_instruction_pathway_features.py tests/test_groot_n15_http_feature_collect.py tests/test_safe_groot_verify_rollout_collection.py -q` | 21 passed |
| 2026-06-10 | N1.5 steering serve hook 연결 검증 | `timeout 60 python -m pytest tests/test_serve_lerobot.py::TestSteeringRegistration -q`, `python -m py_compile scripts/serve/lerobot.py tests/test_serve_lerobot.py` | 3 passed, py_compile 통과 |
| 2026-06-10 | N1.5 `/health` feature metadata 추가 | `timeout 60 python -m pytest tests/test_serve_lerobot.py::TestHealthEndpoint tests/test_serve_lerobot.py::TestSteeringRegistration -q` | 9 passed |
| 2026-06-10 | open_drawer_right seed scan smoke | `docker exec ... select_instruction_seeds.py --cell-id open_drawer_right --target-per-cell 3` | seeds 100000, 100003, 100005 |
| 2026-06-10 | ep_meta replay 경계 진단 | collector env reset vs ep_meta replay 비교 | seed 100000은 collector env에서 right지만 ep_meta replay 후 left로 바뀜. wrapper 기본 replay off로 수정 |
| 2026-06-10 | selector를 collection env path 기준으로 정렬 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py -q`, `docker exec ... select_instruction_seeds.py --cell-id open_drawer_right --target-per-cell 3` | 5 passed, seeds 100000/100003/100005 재확인 |
| 2026-06-10 | open_drawer_right 3-episode VL+DiT smoke 수집 | `docker exec ... collect_instruction_fixed_http_features.py --selected-seeds ... --wait-ready` | 3 triplets 생성, success 2/3 |
| 2026-06-10 | smoke verifier | `python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py ... --layout task_instruction --require-vl-hidden-states ...` | completed=3 expected=3, status=ok |
| 2026-06-10 | smoke summary/pathway loader | `summarize_instruction_cells.py ...`, `instruction_pathway_features.py ... --require-vl` | SR 2/3, instruction_mismatch=0, DiT `(59,1024)`, VL `(59,2048)` |
| 2026-06-10 | smoke 이후 focused regression | `timeout 90 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_instruction_cell_summary.py tests/test_groot_n15_instruction_pathway_features.py tests/test_groot_n15_http_feature_collect.py tests/test_safe_groot_verify_rollout_collection.py tests/test_serve_lerobot.py::TestHealthEndpoint tests/test_serve_lerobot.py::TestSteeringRegistration -q` | 31 passed |
| 2026-06-10 | selector scan checkpoint/resume 추가 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py tests/test_groot_n15_instruction_seed_selector.py` | 7 passed, py_compile 통과 |
| 2026-06-10 | selector scan sample log 추가 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py tests/test_groot_n15_instruction_seed_selector.py` | 8 passed, py_compile 통과 |
| 2026-06-10 | selector sample-log 기반 resume 보강 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py tests/test_groot_n15_instruction_seed_selector.py`, `git diff --check -- ...` | 9 passed, py_compile/diff check 통과 |
| 2026-06-10 | 15-cell seed probe 시작 | `tmux new-session -d -s n15_instruction_seedprobe_v2 "... select_instruction_seeds.py ... --target-per-cell 1 --resume"` | 실행 중. `ppcs_onion` seed 100000, `ppcs_apple` seed 100050, `ppdc_tongs` seed 100006, `ppdc_wooden_spoon` seed 100002 확보. `ppcc_potato` scan 중 |
| 2026-06-10 | [superseded] PnP env kwargs/object 고정 추가 | `timeout 90 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py -q`, `python -m py_compile ...` | 18 passed, py_compile 통과. 이후 upstream patch/constructor forcing 방향 폐기 |
| 2026-06-10 | [superseded] Docker PnP object 고정 smoke | `docker exec ... make_env(... env_kwargs={'obj_groups': ...})` | object forcing은 동작했지만 reproducibility docs 기준과 맞지 않아 최종 수집 authority에서 제외 |
| 2026-06-10 | [superseded] 15-cell seed probe v4 resume | `tmux new-session -d -s n15_instruction_seedprobe_v4 "... select_instruction_seeds.py ... --target-per-cell 1 --resume"` | env forcing shortcut을 포함한 probe라 최종 manifest로 쓰지 않음 |
| 2026-06-10 | [superseded] 15-cell selected seed manifest 검산 | `python - <<'PY' ... selected_instruction_seeds_1.tsv ...` | rows 15, instruction mismatch 0이었지만 PnP 6개가 env forcing 기반이라 폐기 |
| 2026-06-10 | [superseded] 15-cell collect wrapper dry-run | `python scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py --selected-seeds ... --dry-run` | PnP 6개에 `--env-kwargs-json`을 전달하던 command라 폐기 |
| 2026-06-10 | instruction-fixed focused regression | `timeout 120 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_instruction_cell_summary.py tests/test_groot_n15_instruction_pathway_features.py tests/test_groot_n15_http_feature_collect.py tests/test_safe_groot_verify_rollout_collection.py tests/test_serve_lerobot.py::TestHealthEndpoint tests/test_serve_lerobot.py::TestSteeringRegistration -q` | 36 passed, 2 FastAPI deprecation warnings |
| 2026-06-10 | [superseded] 본수집용 50-seed selector 시작 | `tmux new-session -d -s n15_instruction_seedselect_50 "... select_instruction_seeds.py ... --target-per-cell 50 --max-seeds-per-cell 5000 --resume"` | object forcing 의존으로 중단. partial manifest는 최종 수집에 사용하지 않음 |
| 2026-06-10 | seed-only contract 복구 | `timeout 120 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py -q`, `python -m py_compile ...` | 17 passed, py_compile 통과. config/selected manifest에서 env forcing field 제거 |
| 2026-06-10 | natural PnP seed scan smoke | `docker exec ... select_instruction_seeds.py --cell-id ppcs_apple --target-per-cell 1 --max-seeds-per-cell 200` | seed `100050`에서 apple instruction match. selected manifest에 env forcing field 없음, ep_meta JSON/export와 scan samples 생성 |
| 2026-06-10 | 50-per-cell natural seed precollection 시작 | `tmux new-session -d -s n15_instruction_seedscan50_natural "... select_instruction_seeds.py ... --target-per-cell 50 --max-seeds-per-cell 10000 --resume"` | 실행 중. 초기 확인 기준 `ppcs_onion` 2/50 match, manifest schema seed-only |
| 2026-06-10 | superseded selector process 정리 | `ps -eo ...`, `kill 2155229` | old `target_instruction_fixed15_pathway_50ep` selector 중단. natural selector만 남음 |
| 2026-06-10 | seed scan audit script 추가 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_scan_summary.py -q`, `python -m py_compile ...` | 2 passed, py_compile 통과 |
| 2026-06-10 | live seed scan audit 생성 | `python scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_scan.py --selected-seeds ... --output-tsv .../analysis/seed_scan_status.tsv` | 15 rows, complete 0. `ppcs_onion` selected 3/50, mismatch 0, duplicate 0, ep_meta missing 0 |
| 2026-06-10 | seed shard merge gate 추가 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_shard_merge.py -q`, `python -m py_compile ...` | 4 passed, py_compile 통과. old `env_kwargs_json` schema reject와 `--require-complete` failure 확인 |
| 2026-06-10 | serial natural seed scan 중단 | `tmux kill-session -t n15_instruction_seedscan50_natural`, `kill <orphan-selector-pid>` | slow serial run 중단. partial artifact만 보존 |
| 2026-06-10 | 4-cell sharded natural seed scan 시작 | `tmux new-session -d -s n15_seedscan_{ppcs_onion,ppcs_apple,ppdc_tongs,ppdc_spoon} ... --cell-id ...` | 4개 shard process 실행 중. `ppcs_onion`/`ppdc_wooden_spoon` 초기 match 확인 |
| 2026-06-10 | sharded partial merge live probe | `python scripts/safe/groot_n15/robocasa/collect/merge_instruction_seed_shards.py --selected-seeds .../manifests/shards/*.tsv --output-tsv ...partial.tsv`, same with `--require-complete` | partial merge 2 rows 생성, `--require-complete`는 expected incomplete error |
| 2026-06-10 | shard merge 포함 focused regression | `timeout 120 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py -q`, `python -m py_compile ...`, `git diff --check -- ...` | 23 passed, py_compile/diff check 통과 |
| 2026-06-10 | seed shard launcher 추가 | `timeout 120 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py tests/test_groot_n15_instruction_seed_shard_launcher.py`, `git diff --check -- ...` | 29 passed, py_compile/diff check 통과 |
| 2026-06-10 | seed shard launcher dry-run/실행 | `python scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py --run-root ...seedscan50_sharded_natural --max-new-sessions 2 --dry-run`, same without `--dry-run` | active 4개 skip, `ppcc_potato`/`ppcc_bread` 두 shard 시작. sandbox tmux permission denial은 명시 실패로 확인 |
| 2026-06-10 | sharded seed scan summary 보강 | `python scripts/safe/groot_n15/robocasa/collect/merge_instruction_seed_shards.py --selected-seeds .../manifests/shards/*.tsv --output-tsv ...partial.tsv`, `python scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_scan.py --selected-seeds ...partial.tsv ...`, `timeout 120 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py -q`, `python -m py_compile ...`, `git diff --check -- ...` | partial merge 15 selected seed, status 0 complete, 30 passed, py_compile/diff check 통과 |
| 2026-06-10 | 추가 shard 실행 및 shard-dir merge 추가 | `python scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py --run-root ...seedscan50_sharded_natural --max-new-sessions 2 --dry-run`, same without `--dry-run`, `python scripts/safe/groot_n15/robocasa/collect/merge_instruction_seed_shards.py --shard-dir .../manifests/shards --output-tsv ...partial.tsv` | active 6개 skip, `open_cabinet_door`/`open_drawer_right` 두 shard 시작. partial merge 38 selected seed, status 0 complete |
| 2026-06-10 | 추가 shard 2개 실행 | `python scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py --run-root ...seedscan50_sharded_natural --max-new-sessions 2 --dry-run`, same without `--dry-run`, `python scripts/safe/groot_n15/robocasa/collect/merge_instruction_seed_shards.py --shard-dir .../manifests/shards --output-tsv ...partial.tsv` | active 8개 skip, `open_drawer_left`/`close_toaster_oven_door` 두 shard 시작. partial merge 63 selected seed, status 0 complete |
| 2026-06-10 | 추가 shard 2개 실행 | `python scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py --run-root ...seedscan50_sharded_natural --max-new-sessions 2 --dry-run`, same without `--dry-run`, `python scripts/safe/groot_n15/robocasa/collect/merge_instruction_seed_shards.py --shard-dir .../manifests/shards --output-tsv ...partial.tsv` | active 10개 skip, `turn_on_microwave`/`turn_on_sink_faucet` 두 shard 시작. partial merge 82 selected seed, status 0 complete |
| 2026-06-10 | 남은 shard 실행 | `python scripts/safe/groot_n15/robocasa/collect/launch_instruction_seed_shards.py --run-root ...seedscan50_sharded_natural --max-new-sessions 2 --dry-run`, same without `--dry-run`; 이후 `--max-new-sessions 1 --dry-run`, same without `--dry-run` | `navigate_fridge`, `navigate_coffee_machine`, `coffee_setup_mug` shard 시작. 15 cell 전체 active. partial merge 108 selected seed, status 0 complete |
| 2026-06-10 | sampled instruction audit 추가 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_seed_sample_summary.py -q`, `python scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_samples.py --shard-dir .../manifests/shards --cells-config configs/robocasa/n15_instruction_fixed_cells.tsv --top-n 5 --output-tsv .../analysis/seed_sample_summary.tsv` | 2 passed. partial merge 176 selected seed. sample summary 기준 `ppcc_potato`/`ppcc_bread`/Navigate 2개 match 0, `coffee_setup_mug` match rate 1.0 |
| 2026-06-10 | sharded seed scan live 갱신 및 0-match source 진단 | `merge_instruction_seed_shards.py --shard-dir ...`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `tmux list-sessions`, `nl/rg` source inspection | partial merge 279 selected seed, 3 cell complete. `ppcc_potato`/`ppcc_bread`는 `obj_groups="all"` sparse sampling으로 판단, source상 `potato`/`bread` 가능. `NavigateKitchen`은 source상 `CoffeeMachine`/fridge target 후보 포함, 두 navigate cell 모두 canonical match 1개 확인 |
| 2026-06-10 | sparse PnP range shard 추가 및 summary aggregation 보강 | `python -m pytest tests/test_groot_n15_instruction_seed_scan_summary.py::test_summarize_seed_scan_aggregates_multiple_range_shards -q`, `timeout 120 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_sample_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py -q`, `python -m py_compile ...`, `git diff --check -- ...` | RED 확인 후 수정. 23 passed, py_compile/diff check 통과. `ppcc_potato_r1`/`ppcc_bread_r1` seed 200000 range shard 시작. partial merge 312 selected seed, 3 cell complete, `--require-complete`는 expected incomplete |
| 2026-06-10 | range shard launcher 옵션 추가 및 sparse cell 추가 실행 | `python -m pytest tests/test_groot_n15_instruction_seed_shard_launcher.py::test_session_name_uses_stable_short_slug_for_long_cells ... -q` RED, `timeout 120 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_sample_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py -q`, `python -m py_compile ...`, `git diff --check -- ...`, `launch_instruction_seed_shards.py --shard-suffix _r1 --seed-start 200000 ...` | 25 passed, py_compile/diff check 통과. `ppcs_onion_r1`, `ppcs_apple_r1`, `navigate_fridge_r1`, `navigate_coffee_machine_r1` 시작. partial merge 320 selected seed, `--require-complete`는 expected incomplete |
| 2026-06-10 | object-sparse range shard 추가 실행 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py ... --require-complete`, `summarize_instruction_seed_samples.py ...`, `launch_instruction_seed_shards.py --shard-suffix _r2 --seed-start 300000 ...`, `launch_instruction_seed_shards.py --shard-suffix _r1 --seed-start 200000 ...`, `tmux list-sessions`, `tail logs/{ppcs_onion_r2,ppcc_bread_r2,ppdc_tongs_r1}.log` | partial merge 350 selected seed, 3 cell complete. `--require-complete`는 expected incomplete. `ppcs_onion_r2`/`ppcs_apple_r2`/`ppcc_potato_r2`/`ppcc_bread_r2`/`ppdc_tongs_r1`/`ppdc_wooden_spoon_r1` 시작 확인 |
| 2026-06-10 | incomplete shard 복구 및 Navigate range shard 추가 | `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py ... --require-complete`, `launch_instruction_seed_shards.py --shard-suffix _r2 --seed-start 300000 ...` | partial merge 366 selected seed, 3 cell complete. `--require-complete`는 expected incomplete. `ppcs_onion_r2` 재실행, `open_cabinet_door_r2`/`turn_on_sink_faucet_r2`/`navigate_fridge_r2`/`navigate_coffee_machine_r2` 시작 확인 |
| 2026-06-10 | sparse shard resume 및 load-gated 추가 shard 실행 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py ... --require-complete`, `summarize_instruction_seed_samples.py ...`, `launch_instruction_seed_shards.py --shard-suffix _r2 --seed-start 300000 ...`, `launch_instruction_seed_shards.py --shard-suffix _r3 --seed-start 400000`, `uptime`, `tmux list-sessions` | partial merge 388 selected seed, 3 cell complete. `--require-complete`는 expected incomplete. `ppcc_bread_r2`/`ppcc_potato_r2`/`ppcs_onion_r2`/`ppcs_apple_r2` 재실행, `ppcc_bread_r3` 시작 |
| 2026-06-10 | shard launcher log append 전환 및 live audit 갱신 | `python -m pytest tests/test_groot_n15_instruction_seed_shard_launcher.py -q`, `timeout 120 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_sample_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py -q`, `python -m py_compile ...`, `git diff --check -- ...`, `merge_instruction_seed_shards.py --shard-dir ...`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete` | RED 후 수정. 8 passed, focused 25 passed, py_compile/diff check 통과. partial merge 398 selected seed, 3 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 8/50, `ppcs_apple` 10/50, `ppdc_tongs` 20/50, `ppdc_wooden_spoon` 20/50, `ppcc_potato` 3/50, `ppcc_bread` 1/50, `open_cabinet_door` 35/50, `open_drawer_right` 46/50, `open_drawer_left` 31/50, `turn_on_sink_faucet` 44/50, `navigate_fridge` 14/50, `navigate_coffee_machine` 16/50 |
| 2026-06-10 | append-log contract로 `ppcs_onion` base shard resume | `launch_instruction_seed_shards.py --max-new-sessions 1 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ...ppcs_onion.tsv`, `tail logs/ppcs_onion.log`, `uptime` | sandbox tmux denial 후 host 권한 dry-run으로 append command 확인. `n15_seedscan_ppcs_onion` 시작, selector pid 14072 확인. log가 기존 scan 뒤에 append됨. load average 16.86/17.49/18.35 |
| 2026-06-10 | post-launch seed scan live merge | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `merge_instruction_seed_shards.py --shard-dir ... --require-complete`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `tmux list-sessions`, `uptime` | partial merge 410 selected seed, 3 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 8/50, `ppcs_apple` 10/50, `ppdc_tongs` 24/50, `ppdc_wooden_spoon` 20/50, `ppcc_potato` 3/50, `ppcc_bread` 2/50, `open_cabinet_door` 38/50, `open_drawer_right` 46/50, `open_drawer_left` 33/50, `turn_on_sink_faucet` 44/50, `navigate_fridge` 14/50, `navigate_coffee_machine` 18/50. load average 18.28/17.89/18.44 |
| 2026-06-10 | `ppcc_bread_r4` 병목 range shard 추가 | `launch_instruction_seed_shards.py --cell-id ppcc_bread --shard-suffix _r4 --seed-start 500000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ...ppcc_bread_r4.tsv`, `tail logs/ppcc_bread_r4.log`, `merge_instruction_seed_shards.py --shard-dir ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...` | `n15_seedscan_ppcc_bread_r4` 시작, selector pid 14149 확인. partial merge 417 selected seed, 3 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 9/50, `ppcs_apple` 10/50, `ppdc_tongs` 25/50, `ppdc_wooden_spoon` 21/50, `ppcc_potato` 3/50, `ppcc_bread` 2/50, `open_cabinet_door` 39/50, `open_drawer_right` 46/50, `open_drawer_left` 34/50, `turn_on_sink_faucet` 44/50, `navigate_fridge` 15/50, `navigate_coffee_machine` 19/50 |
| 2026-06-10 | `ppcc_potato_r3` 병목 range shard 추가 | `tail logs/ppcc_potato*.log`, `wc -l .../ppcc_potato*.tsv`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r3 --seed-start 400000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ...ppcc_potato_r3.tsv`, `tail logs/ppcc_potato_r3.log`, `merge_instruction_seed_shards.py --shard-dir ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...` | `ppcc_potato`은 base/r1 종료, `_r2`만 active라 `n15_seedscan_ppcc_potato_r3` 시작. selector pid 14226 확인. partial merge 422 selected seed, 3 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 9/50, `ppcs_apple` 10/50, `ppdc_tongs` 26/50, `ppdc_wooden_spoon` 22/50, `ppcc_potato` 3/50, `ppcc_bread` 4/50, `open_cabinet_door` 39/50, `open_drawer_right` 46/50, `open_drawer_left` 35/50, `turn_on_sink_faucet` 44/50, `navigate_fridge` 15/50, `navigate_coffee_machine` 19/50 |
| 2026-06-10 | high-load live audit 및 추가 shard 보류 | `uptime`, `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...` | partial merge 426 selected seed, 3 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 9/50, `ppcs_apple` 10/50, `ppdc_tongs` 28/50, `ppdc_wooden_spoon` 22/50, `ppcc_potato` 3/50, `ppcc_bread` 4/50, `open_cabinet_door` 39/50, `open_drawer_right` 46/50, `open_drawer_left` 36/50, `turn_on_sink_faucet` 44/50, `navigate_fridge` 16/50, `navigate_coffee_machine` 19/50. load average 20.99/19.39/18.75라 추가 shard 보류 |
| 2026-06-10 | live seed scan audit 갱신 및 collection gate 유지 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `merge_instruction_seed_shards.py --shard-dir ... --require-complete`, `summarize_instruction_seed_scan.py --selected-seeds ...partial.tsv`, `summarize_instruction_seed_samples.py --shard-dir ...`, `tmux list-sessions`, `uptime` | partial merge 451 selected seed, 5 cell complete. `open_drawer_right`와 `turn_on_sink_faucet`이 complete로 전환. `--require-complete`는 expected incomplete: `ppcs_onion` 9/50, `ppcs_apple` 10/50, `ppdc_tongs` 33/50, `ppdc_wooden_spoon` 24/50, `ppcc_potato` 4/50, `ppcc_bread` 4/50, `open_cabinet_door` 39/50, `open_drawer_left` 40/50, `navigate_fridge` 19/50, `navigate_coffee_machine` 19/50. load average 14.94/17.74/18.42이고 active shard 진행 중이라 collection은 시작하지 않음 |
| 2026-06-10 | `open_cabinet_door_r2` near-complete shard resume | `launch_instruction_seed_shards.py --cell-id open_cabinet_door --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `tail logs/open_cabinet_door_r2.log`, `uptime` | sandbox tmux denial 후 host 권한 dry-run으로 append command 확인. `n15_seedscan_open_cabinet_r2` 시작. log가 기존 `found=9/50` 뒤에 `existing=9 start_seed=300028`로 append됨. load average 15.77/17.13/18.12 |
| 2026-06-10 | post-resume focused verification 및 gate 재확인 | `timeout 120 python -m pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_sample_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py -q`, `python -m py_compile ...`, `git diff --check -- ...`, `merge_instruction_seed_shards.py --shard-dir ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...` | 25 passed, py_compile/diff check 통과. partial merge 456 selected seed, 5 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 9/50, `ppcs_apple` 11/50, `ppdc_tongs` 34/50, `ppdc_wooden_spoon` 25/50, `ppcc_potato` 4/50, `ppcc_bread` 4/50, `open_cabinet_door` 40/50, `open_drawer_left` 42/50, `navigate_fridge` 20/50, `navigate_coffee_machine` 19/50. collection은 시작하지 않음 |
| 2026-06-10 | `ppcs_onion_r1` sparse shard resume 및 gate 재확인 | `launch_instruction_seed_shards.py --cell-id ppcs_onion --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcs_onion_r1.tsv`, `merge_instruction_seed_shards.py --shard-dir ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `uptime` | `n15_seedscan_ppcs_onion_r1` 시작. log가 기존 `found=3/50` 뒤에 `existing=3 start_seed=200030`으로 append됨. partial merge 470 selected seed, 5 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 10/50, `ppcs_apple` 11/50, `ppdc_tongs` 36/50, `ppdc_wooden_spoon` 25/50, `ppcc_potato` 4/50, `ppcc_bread` 4/50, `open_cabinet_door` 45/50, `open_drawer_left` 43/50, `navigate_fridge` 22/50, `navigate_coffee_machine` 20/50. load average 15.93/16.27/17.47, collection은 시작하지 않음 |
| 2026-06-10 | `ppcc_potato_r4`/`ppcc_bread_r5` sparse range shard 추가 | `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r4 --seed-start 500000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id ppcc_bread --shard-suffix _r5 --seed-start 600000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `tail logs/{ppcc_potato_r4,ppcc_bread_r5}.log`, `merge_instruction_seed_shards.py --shard-dir ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `uptime` | `n15_seedscan_ppcc_potato_r4`, `n15_seedscan_ppcc_bread_r5` 시작. logs는 각각 `start_seed=500000`, `start_seed=600000`으로 시작. partial merge 480 selected seed, 5 cell complete. `ppcc_potato`은 5/50으로 증가. `--require-complete`는 expected incomplete: `ppcs_onion` 11/50, `ppcs_apple` 12/50, `ppdc_tongs` 36/50, `ppdc_wooden_spoon` 25/50, `ppcc_potato` 5/50, `ppcc_bread` 4/50, `open_cabinet_door` 47/50, `open_drawer_left` 45/50, `navigate_fridge` 25/50, `navigate_coffee_machine` 21/50. load average 16.90/16.66/17.33, collection은 시작하지 않음 |
| 2026-06-10 | live merge 482 및 active shard audit | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `tmux list-sessions`, `tail logs/{ppcs_onion_r1,ppcc_bread_r2}.log`, `uptime` | partial merge 482 selected seed, 5 cell complete. `ppcs_onion_r1`과 `ppcc_bread_r2`는 active tmux 목록에서 빠짐. `--require-complete`는 expected incomplete: `ppcs_onion` 11/50, `ppcs_apple` 12/50, `ppdc_tongs` 36/50, `ppdc_wooden_spoon` 25/50, `ppcc_potato` 5/50, `ppcc_bread` 4/50, `open_cabinet_door` 47/50, `open_drawer_left` 46/50, `navigate_fridge` 25/50, `navigate_coffee_machine` 22/50. load average 16.93/16.68/17.29, collection은 시작하지 않음 |
| 2026-06-10 | live merge 488 및 gate 재확인 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_scan.py ...`, `uptime` | partial merge 488 selected seed, 5 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 12/50, `ppcs_apple` 12/50, `ppdc_tongs` 37/50, `ppdc_wooden_spoon` 25/50, `ppcc_potato` 5/50, `ppcc_bread` 4/50, `open_cabinet_door` 48/50, `open_drawer_left` 46/50, `navigate_fridge` 26/50, `navigate_coffee_machine` 23/50. load average 16.68/16.60/17.20, collection은 시작하지 않음 |
| 2026-06-10 | live merge 506 및 완료 shard 정리 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete`, `tmux kill-session -t n15_seedscan_open_cabinet_r2`, `docker exec ... kill 14522` | partial merge 506 selected seed, 7 cell complete. `open_cabinet_door`/`open_drawer_left` complete 전환. 완료된 `open_cabinet_door_r2` 잔여 scanner 정리. `--require-complete`는 expected incomplete: `ppcs_onion` 12/50, `ppcs_apple` 13/50, `ppdc_tongs` 38/50, `ppdc_wooden_spoon` 27/50, `ppcc_potato` 5/50, `ppcc_bread` 7/50, `navigate_fridge` 30/50, `navigate_coffee_machine` 24/50. collection은 시작하지 않음 |
| 2026-06-10 | `ppcs_onion_r2`/`ppcs_apple_r2` resume | `launch_instruction_seed_shards.py --cell-id ppcs_onion --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id ppcs_apple --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af 'ppcs_(onion\|apple)_r2.tsv'`, `tail logs/{ppcs_onion_r2,ppcs_apple_r2}.log`, `uptime` | 두 range shard resume 확인. selector pid는 `ppcs_onion_r2=14830`, `ppcs_apple_r2=14895`. log는 기존 파일 뒤에 append됨. load average 15.50/15.94/16.57 |
| 2026-06-10 | `ppdc_wooden_spoon_r1`/`ppcc_potato_r1` resume 및 gate 재확인 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id ppdc_wooden_spoon --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ...`, `uptime` | partial merge 518 selected seed, 7 cell complete. `ppdc_wooden_spoon_r1` selector pid 15364, `ppcc_potato_r1` selector pid 15049 확인. `--require-complete`는 expected incomplete: `ppcs_onion` 13/50, `ppcs_apple` 14/50, `ppdc_tongs` 42/50, `ppdc_wooden_spoon` 30/50, `ppcc_potato` 6/50, `ppcc_bread` 8/50, `navigate_fridge` 31/50, `navigate_coffee_machine` 24/50. collection은 시작하지 않음 |
| 2026-06-10 | `navigate_fridge_r2`/`ppcs_apple_r2` inactive shard 복구 | `launch_instruction_seed_shards.py --cell-id navigate_fridge --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id ppcs_apple --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ...`, `tail logs/{navigate_fridge_r2,ppcs_apple_r2}.log`, `uptime` | 완료 전 빠진 두 shard를 append-log resume. selector pid는 `navigate_fridge_r2=15127`, `ppcs_apple_r2=15204`. `navigate_fridge_r2`는 `existing=9 start_seed=300114`, `ppcs_apple_r2`는 `existing=0 start_seed=300041`로 이어짐. load average 16.95/17.16/16.93 |
| 2026-06-10 | `ppcc_potato_r4` inactive shard 복구 | `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r4 --seed-start 500000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcc_potato_r4.tsv`, `tail logs/ppcc_potato_r4.log`, `uptime` | 완료 전 빠진 `ppcc_potato_r4`를 append-log resume. selector pid는 15281. log는 `existing=0 start_seed=500038`로 이어짐. load average 17.13/17.19/16.98 |
| 2026-06-10 | `ppcc_bread_r4` inactive shard 복구 | `launch_instruction_seed_shards.py --cell-id ppcc_bread --shard-suffix _r4 --seed-start 500000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcc_bread_r4.tsv`, `tail logs/ppcc_bread_r4.log`, `uptime` | 완료 전 빠진 `ppcc_bread_r4`를 append-log resume. selector pid는 15441. log는 `existing=0 start_seed=500067`로 이어짐. load average 17.16/17.49/17.24 |
| 2026-06-10 | live merge 527 및 `ppcc_bread_r5`/`navigate_coffee_machine_r1` 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id ppcc_bread --shard-suffix _r5 --seed-start 600000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id navigate_coffee_machine --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ...`, `tmux list-sessions`, `uptime` | partial merge 527 selected seed, 7 cell complete. `ppcc_bread_r5` selector pid 15518, `navigate_coffee_machine_r1` selector pid 15589 확인. `navigate_coffee_machine`은 26/50으로 증가. `--require-complete`는 expected incomplete: `ppcs_onion` 14/50, `ppcs_apple` 15/50, `ppdc_tongs` 43/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50, `ppcc_bread` 10/50, `navigate_fridge` 31/50, `navigate_coffee_machine` 26/50. collection은 시작하지 않음 |
| 2026-06-10 | live merge 536 및 `ppcc_potato_r2` 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ppcc_potato_r2.tsv`, `tail logs/ppcc_potato_r2.log`, `tmux list-sessions`, `uptime` | partial merge 536 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 15/50, `ppcs_apple` 16/50, `ppdc_tongs` 43/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50, `ppcc_bread` 11/50, `navigate_fridge` 33/50, `navigate_coffee_machine` 30/50. load가 16대로 내려와 inactive였던 `ppcc_potato_r2`를 append-log resume했고 selector pid 15684, log `existing=2 start_seed=300114` 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 539 및 `ppcs_onion_r1`/`ppcs_apple` 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id ppcs_onion --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id ppcs_apple --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ...`, `tail logs/{ppcs_onion_r1,ppcs_apple}.log`, `tmux list-sessions`, `uptime` | partial merge 539 selected seed, 7 cell complete. `ppcs_onion_r1` selector pid 15767, log `existing=4 start_seed=200047`; `ppcs_apple` selector pid 15838, log `existing=5 start_seed=100161`. `--require-complete`는 expected incomplete: `ppcs_onion` 17/50, `ppcs_apple` 16/50, `ppdc_tongs` 44/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50, `ppcc_bread` 11/50, `navigate_fridge` 33/50, `navigate_coffee_machine` 30/50. load average 21대라 추가 shard 확장 보류. collection은 시작하지 않음 |
| 2026-06-10 | live merge 544 및 `navigate_fridge_r2` 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id navigate_fridge --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af navigate_fridge_r2.tsv`, `tail logs/navigate_fridge_r2.log`, `tmux list-sessions`, `uptime` | partial merge 544 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 18/50, `ppcs_apple` 18/50, `ppdc_tongs` 46/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 6/50, `ppcc_bread` 11/50, `navigate_fridge` 33/50, `navigate_coffee_machine` 30/50. load가 16대로 내려와 inactive였던 `navigate_fridge_r2`를 append-log resume했고 selector pid 15927, log `existing=10 start_seed=300148` 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 552 및 inactive shard 3개 복구 | `git status -sb`, `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `summarize_instruction_seed_samples.py ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id ppdc_wooden_spoon --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id navigate_coffee_machine --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `launch_instruction_seed_shards.py --cell-id ppcs_apple --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `uptime` | partial merge 552 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 18/50, `ppcs_apple` 18/50, `ppdc_tongs` 47/50, `ppdc_wooden_spoon` 32/50, `ppcc_potato` 7/50, `ppcc_bread` 12/50, `navigate_fridge` 38/50, `navigate_coffee_machine` 30/50. `ppdc_wooden_spoon_r1`/`navigate_coffee_machine_r1`/`ppcs_apple_r1`을 append-log resume했고 selector pid는 16010/16081/16152. load average 18.75/17.62/17.88이라 추가 shard 확장은 보류. collection은 시작하지 않음 |
| 2026-06-10 | live merge 556 및 `ppcc_potato_r5` 병목 range shard 추가 | `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r5 --seed-start 600000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ppcc_potato_r5.tsv`, `tail logs/ppcc_potato_r5.log`, `uptime` | partial merge 556 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 19/50, `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 33/50, `ppcc_potato` 7/50, `ppcc_bread` 12/50, `navigate_fridge` 38/50, `navigate_coffee_machine` 31/50. `ppcc_potato_r5`를 seed 600000부터 시작했고 selector pid 16235 확인. load average 17.69/17.54/17.80, collection은 시작하지 않음 |
| 2026-06-10 | live merge 559 gate 재확인 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `uptime` | partial merge 559 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 19/50, `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 33/50, `ppcc_potato` 7/50, `ppcc_bread` 12/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 33/50. load average 19.23/18.26/18.04라 추가 shard 확장 보류. collection은 시작하지 않음 |
| 2026-06-10 | live merge 562 및 `ppcs_apple_r3` 병목 range shard 추가 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `tail logs/{ppcs_apple,ppcs_apple_r1,ppcs_apple_r2}.log`, `launch_instruction_seed_shards.py --cell-id ppcs_apple --shard-suffix _r3 --seed-start 400000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ppcs_apple_r3.tsv`, `tail logs/ppcs_apple_r3.log`, `uptime` | partial merge 562 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 20/50, `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 33/50, `ppcc_potato` 8/50, `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 33/50. `ppcs_apple_r3`를 seed 400000부터 시작했고 selector pid 16318 확인. load average 17.22/17.74/17.87, collection은 시작하지 않음 |
| 2026-06-10 | live merge 564 및 `ppcc_potato_r1` 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ppcc_potato_r1.tsv`, `tail logs/ppcc_potato_r1.log`, `uptime` | partial merge 564 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 20/50, `ppcs_apple` 18/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 34/50, `ppcc_potato` 8/50, `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 34/50. `ppcc_potato_r1`을 append-log resume했고 selector pid 16401, log `existing=4 start_seed=200077` 확인. load average 17.15/17.70/17.85, collection은 시작하지 않음 |
| 2026-06-10 | live merge 566 gate 재확인 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `uptime` | partial merge 566 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 20/50, `ppcs_apple` 19/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 34/50, `ppcc_potato` 8/50, `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. load average 20.35/18.78/18.23이라 추가 shard 확장 보류. collection은 시작하지 않음 |
| 2026-06-10 | live merge 569 및 `ppcc_bread_r3` 복구 | `git status -sb -- docs/steering/11_phase4_n15_instruction_fixed_plan.md src/benchmarks/robocasa`, `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `launch_instruction_seed_shards.py --cell-id ppcc_bread --shard-suffix _r3 --seed-start 400000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ppcc_bread_r3.tsv`, `tail logs/ppcc_bread_r3.log`, `uptime` | partial merge 569 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 20/50, `ppcs_apple` 19/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50, `ppcc_bread` 13/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. load average 16.67/17.20/17.67에서 inactive였던 `ppcc_bread_r3`를 append-log resume했고 selector pid 16496, log `existing=3 start_seed=400125` 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 570 및 추가 shard 보류 | `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `tail logs/{ppcc_bread_r3,ppcc_potato_r1}.log`, `uptime` | partial merge 570 selected seed, 7 cell complete. `ppcc_bread`가 14/50으로 증가. `--require-complete`는 expected incomplete: `ppcs_onion` 20/50, `ppcs_apple` 19/50, `ppdc_tongs` 48/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50, `ppcc_bread` 14/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. `ppcc_potato_r1`은 inactive지만 load average 18.28/18.10/17.95라 추가 복구 보류. collection은 시작하지 않음 |
| 2026-06-10 | live merge 574 및 `ppcc_potato_r1` 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r1 --seed-start 200000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcc_potato_r1.tsv`, `tail logs/ppcc_potato_r1.log`, `uptime` | partial merge 574 selected seed, 7 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 22/50, `ppcs_apple` 20/50, `ppdc_tongs` 49/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50, `ppcc_bread` 14/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. load average 16.67/17.54/17.76에서 inactive였던 `ppcc_potato_r1`을 append-log resume했고 selector pid 16597, log `existing=4 start_seed=200107` 확인. collection은 시작하지 않음 |
| 2026-06-10 | `ppcc_bread_r4` inactive shard 복구 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `launch_instruction_seed_shards.py --cell-id ppcc_bread --shard-suffix _r4 --seed-start 500000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcc_bread_r4.tsv`, `tail logs/ppcc_bread_r4.log`, `uptime` | partial merge는 574 selected seed로 유지. `--require-complete`는 expected incomplete: `ppcs_onion` 22/50, `ppcs_apple` 20/50, `ppdc_tongs` 49/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50, `ppcc_bread` 14/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. load average 16.45/17.10/17.54에서 inactive였던 `ppcc_bread_r4`를 append-log resume했고 selector pid 16686, log `existing=3 start_seed=500136` 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 576, 완료 shard 정리, `ppcc_potato_r6` 추가 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `tmux kill-session -t n15_seedscan_ppdc_tongs_r1`, `docker exec ... kill 12936`, `docker exec ... pgrep -af ppdc_tongs_r1.tsv`, `tail logs/ppcc_potato_r1.log`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r6 --seed-start 700000 --dry-run`, same without `--dry-run`, `docker exec ... pgrep -af ppcc_potato_r6.tsv`, `tail logs/ppcc_potato_r6.log`, `uptime` | partial merge 576 selected seed, 8 cell complete. `ppdc_tongs` complete 전환 후 잔여 session/orphan selector를 정리했다. `--require-complete`는 expected incomplete: `ppcs_onion` 22/50, `ppcs_apple` 20/50, `ppdc_wooden_spoon` 37/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. `ppcc_potato_r1`은 다시 inactive가 되어 새 `ppcc_potato_r6` seed 700000 range shard를 시작했고 selector pid 16793 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 583 gate 재확인 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `column -t -s $'\\t' .../seed_scan_status.tsv`, `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `uptime` | partial merge 583 selected seed, 8 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 23/50, `ppcs_apple` 24/50, `ppdc_wooden_spoon` 39/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50, `navigate_fridge` 39/50, `navigate_coffee_machine` 35/50. `ppcc_potato`는 8/691 sample로 가장 큰 병목이고, collection은 시작하지 않음 |
| 2026-06-10 | `navigate_fridge_r2` inactive shard 복구 | `launch_instruction_seed_shards.py --cell-id navigate_fridge --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af navigate_fridge_r2.tsv`, `tail logs/navigate_fridge_r2.log`, `uptime` | load average 16.19/16.14/16.80에서 inactive였던 `navigate_fridge_r2`를 append-log resume했다. selector pid 16882, log `existing=16 start_seed=300182` 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 587 및 `ppcc_potato_r7` 병목 range shard 추가 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `launch_instruction_seed_shards.py --cell-id ppcc_potato --shard-suffix _r7 --seed-start 800000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcc_potato_r7.tsv`, `tail logs/ppcc_potato_r7.log`, `uptime` | partial merge 587 selected seed, 8 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 24/50, `ppcs_apple` 25/50, `ppdc_wooden_spoon` 40/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50, `navigate_fridge` 40/50, `navigate_coffee_machine` 35/50. `ppcc_potato_r7` selector pid 16959, log `existing=0 start_seed=800000` 확인. collection은 시작하지 않음 |
| 2026-06-10 | live merge 587 유지 및 inactive shard 감사 | `git status -sb -- ...`, `tmux list-sessions`, `docker exec ... pgrep -af select_instruction_seeds.py`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `tail logs/{navigate_fridge_r2,ppcc_bread_r5,ppcc_potato_r7}.log`, `uptime` | partial merge는 587 selected seed로 유지. `ppcc_bread_r5`와 `navigate_fridge_r2`가 active list에서 빠졌고 새 match 없이 종료된 상태로 판단했다. load average 18.45/17.08/17.00이라 추가 재개는 보류. `ppcc_potato` sample은 8/730, `ppcc_bread`는 15/642. collection은 시작하지 않음 |
| 2026-06-10 | `navigate_fridge_r2` 재개 및 post-launch merge 확인 | `launch_instruction_seed_shards.py --cell-id navigate_fridge --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af navigate_fridge_r2.tsv`, `tail logs/navigate_fridge_r2.log`, `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `uptime` | load average가 15.99/16.60/16.83으로 내려와 `navigate_fridge_r2`를 append-log resume했다. selector pid 17042, log `existing=17 start_seed=300193` 확인. post-launch merge는 587 selected seed로 유지. collection은 시작하지 않음 |
| 2026-06-10 | live merge 589 및 `ppcs_onion_r2` 재개 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv ...partial.tsv`, `summarize_instruction_seed_scan.py ...`, `merge_instruction_seed_shards.py --require-complete`, `summarize_instruction_seed_samples.py ...`, `launch_instruction_seed_shards.py --cell-id ppcs_onion --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `docker exec ... pgrep -af ppcs_onion_r2.tsv`, `tail logs/ppcs_onion_r2.log`, `uptime` | partial merge 589 selected seed, 8 cell complete. `--require-complete`는 expected incomplete: `ppcs_onion` 24/50, `ppcs_apple` 25/50, `ppdc_wooden_spoon` 40/50, `ppcc_potato` 8/50, `ppcc_bread` 15/50, `navigate_fridge` 42/50, `navigate_coffee_machine` 35/50. `ppcs_onion_r2` selector pid 17125, log `existing=1 start_seed=300127` 확인. collection은 시작하지 않음 |
| 2026-06-10 | post-`ppcs_onion_r2` load check | `sleep 20`, `uptime`, `docker exec ... pgrep -af ppcs_onion_r2.tsv`, `git status -sb -- docs/steering/11_phase4_n15_instruction_fixed_plan.md src/benchmarks/robocasa` | 1분 load가 25.01까지 튄 뒤 20.23/18.99/17.69로 내려왔지만 여전히 높다. `ppcs_onion_r2` selector pid 17125는 살아 있고, 추가 shard 확장은 보류한다. `src/benchmarks/robocasa`는 status에 안 잡힘 |
| 2026-06-11 | 코드 안정화 재검증 및 `navigate_fridge_r2` 재개 | `python -m py_compile ...`, `timeout 180 python -m pytest tests/test_groot_n15_instruction_fixed_cells.py tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_scan_summary.py tests/test_groot_n15_instruction_seed_sample_summary.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_seed_shard_launcher.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py tests/test_groot_n15_instruction_cell_summary.py tests/test_groot_n15_instruction_pathway_features.py -q`, `merge_instruction_seed_shards.py --shard-dir ...`, `summarize_instruction_seed_scan.py --selected-seeds ...`, `merge_instruction_seed_shards.py --require-complete`, `launch_instruction_seed_shards.py --cell-id navigate_fridge --shard-suffix _r2 --seed-start 300000 --dry-run`, same without `--dry-run`, `tmux list-sessions`, `tail logs/navigate_fridge_r2.log`, `uptime` | py_compile 통과, focused regression 40 passed. partial merge 742 selected seed, 14 cell complete. `--require-complete`는 expected incomplete: `navigate_fridge=42/50`. load average 5대에서 `navigate_fridge_r2`를 append-log resume했고 session `n15_seedscan_navigate_fridge_r2`, log `existing=19 start_seed=300207` 확인. collection은 시작하지 않음 |
| 2026-06-11 | seed gate 완료 및 final manifest 배치 | `merge_instruction_seed_shards.py --shard-dir ... --output-tsv .../selected_instruction_seeds.tsv --require-complete`, `summarize_instruction_seed_scan.py --selected-seeds ...`, `wc -l .../selected_instruction_seeds.tsv` | 750 selected seeds, 15/15 cell complete. final collection manifest는 header 포함 751줄이며 `target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv`에 배치 |
| 2026-06-11 | N1.5 DiT/VL feature server 및 본수집 시작 | `tmux new-session -d -s lerobot_n15_instruction_feature_server ... --collect --capture-vl`, `curl --noproxy '*' -sS http://127.0.0.1:8400/health`, `collect_instruction_fixed_http_features.py --dry-run`, `tmux new-session -d -s n15_instruction_fixed50_collect ...` | `/health` 통과. DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool`, dry-run 750 commands. 실제 collection session 시작 |
| 2026-06-11 | [superseded] 본수집 max-step mismatch 발견 | `verify_rollout_collection.py ... --expected-max-episode-steps 720`, pkl payload audit | 첫 본수집 command가 `--max-episode-steps`를 넘기지 않아 task registry horizon으로 실행됨. pkl payload 기준 `PickPlaceCounterToStove=400`, `PickPlaceDrawerToCounter=500`이 확인되어 COAST/N1.6-aligned 기준과 불일치 |
| 2026-06-11 | [superseded] 잘못된 rollout 결과 삭제 및 seed manifest 보존 확인 | `tmux kill-session -t n15_instruction_fixed50_collect`, `rm -rf .../raw_rollouts .../instruction_success_rates.partial.tsv .../collect_instruction_fixed_http_features.log`, `wc -l .../selected_instruction_seeds.tsv` | 잘못된 collection 세션 중단. useless raw rollout 결과와 partial summary/log 삭제. final seed manifest는 header 포함 751줄, 15 cell x 50 seed 유지 |
| 2026-06-11 | 720 cap 본수집 재시작 및 `ppcs_onion` 완료 검증 | `collect_instruction_fixed_http_features.py --dry-run | awk ...`, `tmux new-session -d -s n15_instruction_fixed50_collect_720 ... --max-episode-steps 720`, pkl payload audit, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, ppcs_onion filename audit | dry-run 750 commands, missing 720 flag 0. 첫 산출물 `PickPlaceCounterToStove/ppcs_onion/task0--ep0--succ1`에서 `scenario_seed=100000`, `inference_seed=100000`, `max_episode_steps=720`, `n_action_steps=16` 확인. `ppcs_onion` 50/50 complete, success 41 / failure 9, ep0..49 missing 없음. 전체 raw_rollouts는 `ppcs_apple` 1개 포함 51개, verifier `status=partial-ok`. `ppcs_onion`은 failure < 10이라 top-up 후보 |
| 2026-06-11 | `ppcs_apple` 25+ checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, `collection_summary.tsv` 집계 | 전체 raw_rollouts 76개 기준 verifier `status=partial-ok`. `ppcs_apple`은 26/50, success 22 / failure 4. `ppcs_onion`은 50/50 유지 |
| 2026-06-11 | `ppcs_apple` 완료 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, ppcs_apple filename audit, `collection_summary.tsv` 집계 | `ppcs_apple` 50/50 complete, success 44 / failure 6, ep0..49 missing 없음. 전체 raw_rollouts는 `ppdc_tongs` 1개 포함 101개, verifier `status=partial-ok`. `ppcs_apple`은 failure < 10이라 top-up 후보 |
| 2026-06-11 | `ppdc_tongs` 32/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, selected seed manifest 대조, same task/seed duplicate audit | 전체 raw_rollouts 132개 기준 verifier `status=partial-ok`. `ppdc_tongs`는 32/50, success 9 / failure 23. manifest 대조 mismatch 0, same task/seed duplicate 0. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | `ppdc_tongs` 완료 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, ppdc_tongs filename audit, selected seed manifest 대조, duplicate audit | `ppdc_tongs` 50/50 complete, success 15 / failure 35, ep0..49 missing 없음. 전체 raw_rollouts는 `ppdc_wooden_spoon` 2개 포함 152개, verifier `status=partial-ok`. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0 |
| 2026-06-11 | `ppdc_wooden_spoon` 25/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, selected seed manifest 대조, duplicate audit | 전체 raw_rollouts 175개 기준 verifier `status=partial-ok`. `ppdc_wooden_spoon`은 25/50, success 5 / failure 20. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0 |
| 2026-06-11 | `ppdc_wooden_spoon` 완료 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, ppdc_wooden_spoon filename audit, selected seed manifest 대조, duplicate audit | `ppdc_wooden_spoon` 50/50 complete, success 8 / failure 42, ep0..49 missing 없음. 전체 raw_rollouts는 `ppcc_potato` 1개 포함 201개, verifier `status=partial-ok`. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. success < 10이라 top-up 후보 |
| 2026-06-11 | `ppcc_potato` 26/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, selected seed manifest 대조, duplicate audit | 전체 raw_rollouts 226개 기준 verifier `status=partial-ok`. `ppcc_potato`은 26/50, success 23 / failure 3. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0 |
| 2026-06-11 | `ppcc_potato` 완료 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, ppcc_potato filename audit, selected seed manifest 대조, duplicate audit | `ppcc_potato` 50/50 complete, success 43 / failure 7, ep0..49 missing 없음. 전체 raw_rollouts는 `ppcc_bread` 2개 포함 252개, verifier `status=partial-ok`. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. failure < 10이라 top-up 후보 |
| 2026-06-11 | `ppcc_bread` 28/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, selected seed manifest 대조, duplicate audit | 전체 raw_rollouts 278개 기준 verifier `status=partial-ok`. `ppcc_bread`는 28/50, success 20. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0 |
| 2026-06-11 | `ppcc_bread` 완료 및 `open_cabinet_door` 8/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, ppcc_bread filename audit, selected seed manifest 대조, duplicate audit, `/health` | 전체 raw_rollouts 308개 기준 verifier `status=partial-ok`. `ppcc_bread` 50/50 complete, success 39 / failure 11, ep0..49 missing 없음. `open_cabinet_door`는 8/50, success 2 / failure 6. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | `open_cabinet_door` 25/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, open_cabinet_door filename audit, selected seed manifest 대조, duplicate audit, `/health` | 전체 raw_rollouts 325개 기준 verifier `status=partial-ok`. `open_cabinet_door`는 25/50, success 9 / failure 16, ep0..24 missing 없음. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | `open_cabinet_door` 완료 및 `open_drawer_right` 시작 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, open_cabinet_door filename audit, selected seed manifest 대조, duplicate audit, `/health` | 전체 raw_rollouts 351개 기준 verifier `status=partial-ok`. `open_cabinet_door` 50/50 complete, success 20 / failure 30, ep0..49 missing 없음. `open_drawer_right`는 1/50, success 0. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | 본수집 pkl 내 `ep_meta` 저장 audit | raw rollout pkl 전수 검사, selected seed manifest 대조, run-root `ep_meta/*.json` count, selected manifest `ep_meta_path` 존재 확인 | 전체 pkl 365개 기준 `ep_meta` missing 0, empty 0, `ep_meta["lang"]` missing 0. pkl의 `scenario_seed`, `cell_id`, `canonical_instruction`, `task_description`, `ep_meta["lang"]`은 selected manifest와 mismatch 0. run root의 별도 `ep_meta/*.json`은 147개뿐이고, selected manifest의 seed-scan `ep_meta_path`는 750/750 missing이다. 현재 gate는 pkl 내 `ep_meta` 필드다 |
| 2026-06-11 | `ep_meta` JSON archive backfill 및 `open_drawer_right` 26/50 checkpoint 검증 | pkl 기반 run-root `ep_meta/*.json` backfill, selected manifest `ep_meta_path` rewrite, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, manifest/duplicate audit, `/health` | 전체 raw_rollouts 376개 기준 verifier `status=partial-ok`. `open_drawer_right`는 26/50, success 11 / failure 15, ep0..25 missing 없음. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. backfill snapshot pkl 377개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest의 `ep_meta_path`는 run-root `ep_meta/<task>/<cell_id>/...json`으로 갱신했고, 수집 완료분 377개는 존재, 미수집 373개는 아직 missing |
| 2026-06-11 | wrapper `ep_meta` export path 수정 및 최신 archive 검증 | `python -m pytest tests/test_groot_n15_instruction_collect_wrapper.py -q`, `python -m py_compile ...`, `git diff --check -- ...`, pkl 기반 backfill/audit | wrapper가 replay 여부와 무관하게 `--ep-meta-dir`를 collect script에 넘기도록 수정했다. replay load는 여전히 `--replay-ep-meta`일 때만 `--ep-meta-load-env-name`을 넘긴다. 테스트 4 passed, py_compile/diff check 통과. 최신 backfill snapshot pkl 380개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest `ep_meta_path` 존재 380, 미수집 370 |
| 2026-06-11 | `open_drawer_right` 완료 검증 및 `ep_meta` archive 최신화 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, open_drawer_right filename audit, selected seed manifest 대조, duplicate audit, pkl 기반 run-root `ep_meta/*.json` backfill/audit, `/health` | 전체 raw_rollouts 401개 기준 verifier `status=partial-ok`. `open_drawer_right` 50/50 complete, success 15 / failure 35, ep0..49 missing 없음. `open_drawer_left`는 1/50, success 0. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. backfill snapshot pkl 401개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest `ep_meta_path` 존재 401, 미수집 349. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | `open_drawer_left` 12/50 checkpoint 및 `ep_meta` archive 최신화 | `/health`, pkl 기반 run-root `ep_meta/*.json` backfill/audit, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, manifest/duplicate audit, instruction SR 집계 | 전체 raw_rollouts 412개 기준 verifier `status=partial-ok`. `open_drawer_left`는 verifier snapshot에서 12/50, success 4이고, 이후 live/backfill snapshot은 15/50, success 5 / failure 10이다. backfill snapshot pkl 415개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest `ep_meta_path` 존재 415, 미수집 335. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | `open_drawer_left` 28/50 checkpoint 및 partial SR summary 갱신 | pkl 기반 run-root `ep_meta/*.json` backfill/audit, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, manifest/duplicate audit, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv`, `/health` | 전체 raw_rollouts 428개 기준 verifier `status=partial-ok`. `open_drawer_left`는 verifier snapshot에서 28/50, success 10이고, 이후 partial summary snapshot은 31/50, success 13 / failure 18, SR 41.9%다. backfill snapshot pkl 431개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest `ep_meta_path` 존재 431, 미수집 319. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. partial SR TSV는 9개 instruction cell, instruction mismatch 0, `has_vl_hidden_states`는 각 collected count와 일치한다. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | `open_drawer_left` 완료 및 `close_toaster_oven_door` 시작 검증 | open_drawer_left filename audit, pkl 기반 run-root `ep_meta/*.json` backfill/audit, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, manifest/duplicate audit, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv`, `/health` | `open_drawer_left` 50/50 complete, success 23 / failure 27, ep0..49 missing 없음. 전체 raw_rollouts 452개 기준 verifier `status=partial-ok`: 10개 cell이 보이지만 완료 cell은 9개이고, `close_toaster_oven_door`는 verifier snapshot에서 2/50, success 0이다. 이후 live count는 `close_toaster_oven_door` 5/50, success 2 / failure 3까지 증가했다. backfill snapshot pkl 455개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest `ep_meta_path` 존재 455, 미수집 295. manifest 대조 mismatch 0, same task/seed duplicate 0, same task_id/episode/seed duplicate 0. partial SR TSV는 10개 instruction cell을 쓴다. feature server `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지 |
| 2026-06-11 | replay ep_meta hard gate 및 selected ep_meta materializer 추가 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_ep_meta_materialize.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py -q`, `timeout 120 python -m pytest tests/test_groot_n15_instruction_ep_meta_materialize.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_http_feature_collect.py tests/test_safe_groot_verify_rollout_collection.py tests/test_groot_n15_instruction_ep_meta_backfill.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/materialize_selected_ep_meta.py ...` | 20 passed, 이후 focused regression 28 passed, py_compile 통과. `--replay-ep-meta`가 켜졌는데 ep_meta manifest가 없으면 collector가 즉시 실패한다. wrapper는 `--skip-existing`를 지원한다. `materialize_selected_ep_meta.py`는 selected manifest의 missing JSON을 현재 env reset 기준으로 생성하고 instruction mismatch를 hard fail하며, `--continue-on-mismatch`로 valid seed만 회수할 수 있다. |
| 2026-06-11 | 남은 3개 cell seed/ep_meta repair 시작 | `materialize_selected_ep_meta.py --selected-seeds ... --cell-id navigate_fridge --cell-id navigate_coffee_machine --cell-id coffee_setup_mug --dry-run`, same in RoboCasa container, `launch_instruction_seed_shards.py ... --shard-suffix _r100/_r300/_r500/_r700` | dry-run 기준 150/150 ep_meta missing. container materialize는 `navigate_fridge` seed 200015가 현재 runtime에서 `Navigate to the shelves.`로 재현되어 중단됐다. 기존 selected manifest에 stale seed가 섞인 것으로 판단하고, `navigate_fridge`/`navigate_coffee_machine`/`coffee_setup_mug`를 repair shard로 재선별 중이다. |
| 2026-06-11 | selected manifest repair 완료 및 resume 재개 | selected manifest 후보 merge 스크립트, manifest audit, `collect_instruction_fixed_http_features.py --dry-run --cell-id navigate_fridge --cell-id navigate_coffee_machine --cell-id coffee_setup_mug --replay-ep-meta --skip-existing --max-episodes-per-cell 1`, `tmux new-session -d -s n15_instruction_fixed50_collect_720_resume_nav ...`, first pkl audit, `verify_rollout_collection.py ... --allow-partial`, `summarize_instruction_cells.py ...` | repaired `selected_instruction_seeds.tsv`는 header 포함 751줄, 15 cell x 50, ep_meta missing 0, instruction mismatch 0, duplicate cell/seed 0. 원본은 `selected_instruction_seeds.before_repair.tsv`로 보존했다. resume 첫 pkl `NavigateKitchen/navigate_fridge/task12--ep0--succ0.pkl`는 `task_description`/`ep_meta["lang"]` 모두 `Navigate to the fridge.`, DiT `(4,16,1024)`, VL `(2048,)`, `max_episode_steps=720`. verifier는 605/750 `status=partial-ok`, `navigate_fridge` 5/50 success 1. |
| 2026-06-11 | `close_toaster_oven_door` 26/50 checkpoint 검증 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, pkl 기반 run-root `ep_meta/*.json` backfill/audit, manifest audit, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv` | 전체 raw_rollouts 476개 기준 verifier `status=partial-ok`. `close_toaster_oven_door`는 verifier snapshot에서 26/50, success 14다. backfill snapshot pkl 477개 기준 pkl `ep_meta` missing 0, manifest row missing 0, JSON missing 0, JSON mismatch 0. selected manifest `ep_meta_path` 존재 477, 미수집 273. manifest 대조 mismatch 0. partial SR TSV는 10개 instruction cell을 쓴다. |
| 2026-06-11 | `close_toaster_oven_door` 완료 및 N1.5 feature shape 재확인 | `collection_summary.tsv` live aggregation, pkl sample shape audit, `/health` | `close_toaster_oven_door` 50/50 complete, success 24 / failure 26. shape audit snapshot에서 `turn_on_microwave`는 8/50, success 0 / failure 8이었다. pkl sample `CloseToasterOvenDoor/close_toaster_oven_door/task9--ep0--succ0.pkl` 기준 `hidden_states[0].shape == (4,16,1024)`, `vl_hidden_states[0].shape == (2048,)`, `feature_kind=groot_n15_dit_action_tokens_pre_decode`, `vl_feature_kind=groot_n15_vlln_seq_meanpool`, `max_episode_steps=720`이다. 따라서 현재 N1.5 본수집은 N1.6 Phase 3의 `hidden_states[step]=[7,51,1536]` block-residual scheme을 저장하지 않는다. |
| 2026-06-11 | `turn_on_microwave` partial verifier 및 archive 최신화 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, header 기반 manifest/`ep_meta_path` audit, pkl 내부 `ep_meta` audit, pkl 기반 JSON backfill, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv` | verifier snapshot은 raw_rollouts 511개 기준 `status=partial-ok`, 11 visible cells, `turn_on_microwave` 11/50 success 0, missing ep11..49. header 기반 manifest 대조는 backfill 후 513 rows 기준 missing manifest 0, seed/cell/task mismatch 0, collected `ep_meta_path` exists 513, missing 0. pkl 내부 `ep_meta`는 512개 snapshot에서 missing/empty/lang-missing 0. duplicate audit는 same task/seed 0, same task_id/episode/seed 0. partial SR TSV는 11개 instruction cell을 썼고, `turn_on_microwave` snapshot은 13/50 success 0 / failure 13, `has_vl_hidden_states=13`이다. 이후 live count는 14/50 success 0 / failure 14까지 증가했다. |
| 2026-06-11 | `turn_on_microwave` 18/50 checkpoint 및 schema 재확인 | `TurnOnMicrowave` pkl shape audit, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, duplicate audit, pkl 기반 JSON backfill, header 기반 manifest/`ep_meta_path` audit, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv` | `TurnOnMicrowave/turn_on_microwave` pkl sample 기준 `scenario_seed=inference_seed=100009`, `max_episode_steps=720`, `ep_meta.lang="Press the start button on the microwave."`, `hidden_states[0].shape == (4,16,1024)`, `vl_hidden_states[0].shape == (2048,)`, DiT/VL feature kind는 N1.5 expectation과 일치했다. verifier snapshot은 raw_rollouts 518개 기준 `status=partial-ok`, 11 visible cells, `turn_on_microwave` 18/50 success 0, missing ep18..49. duplicate audit는 same task/seed 0, same task_id/episode/seed 0. backfill은 JSON 4개를 새로 쓰고, 최종 manifest/`ep_meta_path` audit는 520 rows 기준 collected `ep_meta_path` exists 520, missing 0이다. 518-row 상세 audit에서는 missing manifest 0, seed/cell/task mismatch 0, pkl 내부 `ep_meta` missing/empty/lang-missing 0이었다. partial SR TSV는 11개 instruction cell을 썼고 `turn_on_microwave`는 0/18이다. 이후 live count는 20/50 success 0 / failure 20까지 증가했다. |
| 2026-06-11 | `turn_on_microwave` 22/50 checkpoint 및 첫 success 확인 | `TurnOnMicrowave` pkl shape audit, `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, duplicate audit, pkl 기반 JSON backfill, header 기반 manifest/`ep_meta_path` audit, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv` | `TurnOnMicrowave/turn_on_microwave` latest pkl sample 기준 `scenario_seed=inference_seed=100021`, `max_episode_steps=720`, canonical `ep_meta.lang`, `hidden_states[0].shape == (4,16,1024)`, `vl_hidden_states[0].shape == (2048,)`이다. verifier snapshot은 raw_rollouts 522개 기준 `status=partial-ok`, `turn_on_microwave` 22/50 success 0, missing ep22..49. duplicate audit는 same task/seed 0, same task_id/episode/seed 0. backfill은 JSON 3개를 새로 쓰고, full audit는 523 rows 기준 missing manifest 0, seed/cell/task mismatch 0, collected `ep_meta_path` exists 523, missing 0, pkl 내부 `ep_meta` missing/empty/lang-missing 0이다. partial SR TSV는 11개 instruction cell을 썼고, 이후 live/summary snapshot에서 `turn_on_microwave` 첫 success가 생겨 23/50 success 1 / failure 22, SR 4.35%가 됐다. 완료 후에도 success sample이 적으면 top-up 또는 fit 제외 기준을 판단해야 한다. |
| 2026-06-11 | `turn_on_microwave` 30/50 checkpoint 및 N1.6-aligned feature gap 정리 | `verify_rollout_collection.py ... --allow-partial --expected-max-episode-steps 720 --require-vl-hidden-states`, `collection_summary.tsv` live aggregation, pkl 기반 JSON backfill, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv`, N1.6 steering docs schema 확인 | verifier snapshot은 raw_rollouts 530개 기준 `status=partial-ok`, `turn_on_microwave` 30/50 success 2, missing ep30..49. processed 530 rows 기준 JSON backfill 5개, missing manifest 0, pkl `ep_meta` missing 0, processed rows `ep_meta_path` missing 0. partial SR TSV는 11개 cell을 썼고 `turn_on_microwave`는 2/30, SR 6.67%다. N1.6 steering 문서 기준 pathway run은 DiT `hidden_states[step]=[7,51,1536]` + VL `[2048]`이고, 현재 N1.5 본수집은 DiT final `hidden_states[step]=[4,16,1024]` + VL `[2048]`라 aligned scheme이 아니다. N1.5에서 N1.6-aligned pathway 수집을 하려면 active run과 별도로 `transformer_blocks[i]` residual capture hook, layer metadata, verifier expectation, smoke run을 추가해야 한다. |
| 2026-06-11 | N1.6-aligned N1.5 DiT block residual capture option 구현 | `pytest tests/test_serve_lerobot.py::{TestActWithFeaturesEndpoint,TestSafeHooks,TestHealthEndpoint} -q` 분리 실행, `pytest tests/test_vla_client.py::TestVLAClientPredictWithFeatures -q`, `pytest tests/test_groot_n15_http_feature_collect.py -q`, `pytest tests/test_safe_groot_collect.py::{...block_residual...,...writer...} -q`, `pytest tests/test_safe_metadata.py -q`, `py_compile`, `git diff --check` | `--groot-dit-capture-layers`를 추가해 GR00T N1.5 `action_head.model.transformer_blocks[i]` residual stream을 `[layer, model_token, feature_dim]`으로 캡처할 수 있게 했다. `/health`, `/act_with_features`, `VLAClient`, N1.5 HTTP collector, N1.6 공용 record mixin, pkl writer가 `capture_layers/layer_count/token_count` metadata를 보존한다. focused tests는 신규 RED->GREEN 후 최종 분리 실행 기준 server endpoint/hook/health 5/3/7 passed, VLAClient 9 passed, N1.5 collector 9 passed, writer/mixin subset 4 passed, safe metadata 8 passed. 실제 N1.5 model의 `T_model_token`/`D_block` runtime shape는 active final-DiT collection을 방해하지 않기 위해 아직 smoke하지 않았고, 별도 aligned smoke run에서 확정해야 한다. |
| 2026-06-11 | `turn_on_microwave` 완료 및 `turn_on_sink_faucet` 12/50 checkpoint 검증 | `verify_rollout_collection.py ... --expected-model-horizon 16 --expected-valid-horizon none --allow-partial --require-vl-hidden-states`, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.partial.tsv`, raw rollout count audit, `/health` | verifier snapshot은 raw_rollouts 562개 기준 `status=partial-ok`. 11개 cell이 50/50 complete이고 `turn_on_sink_faucet`은 12/50, success 5다. `turn_on_microwave`는 50/50 complete, success 5 / failure 45, SR 10.0%다. partial SR TSV는 12개 instruction cell을 쓰며 `turn_on_sink_faucet`은 5/12, SR 41.67%, instruction mismatch 0, `has_vl_hidden_states=12`다. N1.5 verifier는 N1.6 default horizon이 있으므로 `--expected-model-horizon 16 --expected-valid-horizon none`을 반드시 명시해야 한다. 현재 `/health`는 `collect_mode=true`, `capture_vl=true`, DiT `groot_n15_dit_action_tokens_pre_decode`, VL `groot_n15_vlln_seq_meanpool` 유지. |
| 2026-06-11 | pkl 기반 `ep_meta` backfill helper 추가 및 `turn_on_sink_faucet` 34/50 checkpoint 검증 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_ep_meta_backfill.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/backfill_instruction_ep_meta.py tests/test_groot_n15_instruction_ep_meta_backfill.py`, `git diff --check -- ...`, `backfill_instruction_ep_meta.py --run-root ... --rewrite-selected-seeds`, `summarize_instruction_seed_scan.py ...`, `verify_rollout_collection.py ... --expected-feature-axes denoising_step,action_step,feature_dim --expected-model-horizon 16 --expected-valid-horizon none --allow-partial --require-vl-hidden-states`, `/health` | backfill helper tests 2 passed, py_compile/diff check 통과. backfill snapshot은 processed pkl 584개, missing pkl `ep_meta` 0, missing manifest row 0, manifest mismatch 0이다. seed selection audit는 15 cell 모두 selected 50/50, instruction mismatch 0, duplicate seed 0이며 완료된 11개 cell의 `ep_meta_missing`은 0이다. verifier snapshot은 raw_rollouts 584개 기준 `status=partial-ok`, `turn_on_sink_faucet` 34/50 success 12다. partial SR TSV는 12개 instruction cell을 쓰며 `turn_on_sink_faucet`은 12/34, SR 35.29%, instruction mismatch 0, `has_vl_hidden_states=34`다. 현재 `/health`는 아직 final-DiT collection mode: DiT `groot_n15_dit_action_tokens_pre_decode` `[denoising_step, action_step, feature_dim]`, VL `groot_n15_vlln_seq_meanpool` 2048이다. N1.6-aligned DiT block residual collection은 별도 smoke/re-run이 필요하다. |
| 2026-06-11 | aligned block residual verifier metadata gate 추가 | `timeout 60 python -m pytest tests/test_safe_groot_verify_rollout_collection.py::test_verifier_accepts_block_residual_metadata_expectations -q`, `timeout 60 python -m pytest tests/test_safe_groot_verify_rollout_collection.py -q`, `python -m py_compile scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py tests/test_safe_groot_verify_rollout_collection.py`, `git diff --check -- ...`, current final-DiT verifier with `--expected-feature-axes denoising_step,action_step,feature_dim` | 신규 RED는 unknown `--expected-feature-axes/--expected-capture-layers/--expected-layer-count/--expected-token-count`로 실패했고, 구현 후 block residual metadata test 1 passed, verifier 전체 6 passed, py_compile/diff check 통과. 현재 final-DiT run verifier도 새 `--expected-feature-axes`를 포함해 `status=partial-ok`로 통과했다. aligned smoke에서는 shape와 함께 capture metadata까지 gate로 건다. |
| 2026-06-11 | aligned smoke per-cell limiter 및 최신 checkpoint 갱신 | `timeout 60 python -m pytest tests/test_groot_n15_instruction_collect_wrapper.py -q`, `collect_instruction_fixed_http_features.py ... --cell-id open_drawer_right --max-episodes-per-cell 1 --dry-run`, `backfill_instruction_ep_meta.py --run-root ...`, `verify_rollout_collection.py ... --expected-feature-axes denoising_step,action_step,feature_dim --allow-partial`, `summarize_instruction_cells.py ...` | wrapper tests 6 passed. dry-run은 `open_drawer_right` 첫 seed `100000` 1 command만 생성했고 `--max-episode-steps 720`을 포함했다. 최신 final-DiT checkpoint는 raw_rollouts 588개 기준 `status=partial-ok`, `turn_on_sink_faucet` 38/50 success 13, partial SR 34.21%다. backfill snapshot은 processed pkl 588개, pkl `ep_meta` missing 0, missing manifest row 0, manifest mismatch 0이다. |
| 2026-06-11 | N1.5 instruction-fixed final-DiT+VL 15-cell x 50 본수집 완료 | `verify_rollout_collection.py ... --episodes-per-task 50 --expected-feature-kind groot_n15_dit_action_tokens_pre_decode --expected-feature-axes denoising_step,action_step,feature_dim --expected-hidden-shape 4,16,1024 --expected-max-episode-steps 720 --require-vl-hidden-states --expected-vl-hidden-shape 2048`, `summarize_instruction_cells.py ... --output-tsv .../analysis/instruction_success_rates.tsv`, latest pkl spot checks | full verifier `completed=750 expected=750`, `status=ok`. `coffee_setup_mug`까지 15 cell 모두 50/50 complete. final SR TSV는 15 rows, instruction mismatch 0, `has_vl_hidden_states=50` for every cell. Low-success cells are `ppdc_wooden_spoon` 8/50, `turn_on_microwave` 5/50, `navigate_fridge` 3/50, `navigate_coffee_machine` 1/50, `coffee_setup_mug` 3/50; these need top-up or fit-exclude decision before conceptor fit. |
| 2026-06-11 | Phase C 입력 cache 및 top-up 판단표 생성 | `instruction_pathway_features.py ... --output-npz .../analysis/pathway_separation/final_dit_vl_step_features_mean_mean.npz --require-vl`, cache shape audit, `topup_recommendations.tsv` 생성 | feature cache는 25,272 step rows, DiT `(25272,1024)`, VL `(25272,2048)`, 15 cell coverage를 가진다. Top-up 판단표는 15 rows이며 10/10 fit 기준 미달 cell은 8개다: failure 부족 `ppcs_onion`, `ppcs_apple`, `ppcc_potato`; success 부족 `ppdc_wooden_spoon`, `turn_on_microwave`, `navigate_fridge`, `navigate_coffee_machine`, `coffee_setup_mug`. |
| 2026-06-11 | N1.5 HTTP collector의 ep_meta 자동 replay 차단 | `timeout 120 python -m pytest tests/test_groot_n15_http_feature_collect.py tests/test_groot_n15_instruction_collect_wrapper.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py tests/test_groot_n15_http_feature_collect.py` | 19 passed, py_compile 통과. `--ep-meta-load-env-name`이 없으면 기존 `--ep-meta-dir` JSON이 있어도 `env.set_ep_meta()`를 호출하지 않고 natural reset ep_meta를 새 authority로 저장한다. replay를 명시했는데 manifest가 없으면 즉시 `FileNotFoundError`로 실패한다. |
| 2026-06-11 | N1.6-aligned N1.5 block residual 1-cell smoke | `/health`, `docker exec -e MUJOCO_GL=egl temporal_vla-robocasa-run-3705634bbbf6 ... collect_instruction_fixed_http_features.py --cell-id open_drawer_right --max-episodes-per-cell 1 ...`, pkl shape audit, `verify_rollout_collection.py ... --allow-partial --expected-hidden-shape 7,49,1536 --expected-capture-layers 0,1,2,4,8,12,15 --expected-token-count 49 --require-vl-hidden-states` | server health 기준 `feature_kind=groot_n15_dit_block_residual_tokens`, `feature_axes=["layer","model_token","feature_dim"]`, capture layers `0,1,2,4,8,12,15`, VL dim 2048. Smoke pkl `OpenDrawer/open_drawer_right/task7--ep0--succ0.pkl`는 `hidden_states[0].shape == (7,49,1536)`, `vl_hidden_states[0].shape == (2048,)`, `max_episode_steps=720`, `scenario_seed=100000`, `task_description="Open the right drawer."`. Verifier는 `completed=1 expected=2`, `status=partial-ok`로 통과했다. |
| 2026-06-11 | N1.5 instruction-cell conceptor fit wrapper 추가 및 smoke fit | `timeout 120 python -m pytest tests/test_groot_n15_instruction_conceptor_fit.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/steer/fit_instruction_conceptors.py tests/test_groot_n15_instruction_conceptor_fit.py`, `fit_instruction_conceptors.py ... --pathway {dit,vl} --agg-mode truncated --max-len 10 --dry-run`, `fit_instruction_conceptors.py ... --pathway {dit,vl} --cell-id open_drawer_right --alpha 0.1` | fixture tests 3 passed, py_compile 통과. 실제 final-DiT/VL cache 기준 dry-run은 DiT/VL 모두 15 rows, 7 eligible / 8 skipped를 기록했다. Eligible cells: `close_toaster_oven_door`, `open_cabinet_door`, `open_drawer_left`, `open_drawer_right`, `ppcc_bread`, `ppdc_tongs`, `turn_on_sink_faucet`. `open_drawer_right` smoke fit은 DiT `C_*` matrices `(1024,1024)`, VL `C_*` matrices `(2048,2048)`를 생성했고 metadata는 `group_key=cell_id`, `agg_mode=truncated_w10`, samples success 150 / failure 350이다. |
| 2026-06-11 | 7 eligible cell DiT/VL conceptor full fit | `fit_instruction_conceptors.py --pathway dit --agg-mode truncated --max-len 10 --cell-id <7 eligible cells>`, `fit_instruction_conceptors.py --pathway vl --agg-mode truncated --max-len 10 --cell-id <7 eligible cells>`, artifact shape audit | DiT full fit은 7 rows 모두 `status=fit`, matrices `(1024,1024)`. VL full fit은 7 rows 모두 `status=fit`, matrices `(2048,2048)`. Fit roots: `conceptor/dit/truncated_w10/` and `conceptor/vl/truncated_w10/`. Artifact size after fit is about 1.2GB. Skipped 8 cells remain below 10/10 episode balance and require top-up or fit-exclude decision. |
| 2026-06-11 | N1.5 steering eval matrix planner 추가 | `timeout 120 python -m pytest tests/test_groot_n15_instruction_steering_eval_plan.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/steer/plan_instruction_steering_eval.py tests/test_groot_n15_instruction_steering_eval_plan.py`, `plan_instruction_steering_eval.py --run-tag steer_eval_instruction_fixed15_alpha_all_b01_b03`, eval matrix audit | planner tests 3 passed, py_compile 통과. 실제 conceptor fit 결과에서 eval matrix 97 rows 생성: baseline 7, DiT steered 50, VL steered 40. 모든 steered row의 host-side `steering_npz` 존재를 확인했고, generated command 내부 host absolute path count는 0이다. `server_command`와 `collect_command`는 `/temporal_vla/...` container paths를 사용하고, steered server command는 `--steering-alpha`를 명시한다. |
| 2026-06-11 | N1.5 steering eval row runner 추가 | `timeout 120 python -m pytest tests/test_groot_n15_instruction_steering_eval_runner.py -q`, `python -m py_compile scripts/safe/groot_n15/robocasa/steer/run_instruction_steering_eval.py tests/test_groot_n15_instruction_steering_eval_runner.py`, `run_instruction_steering_eval.py --eval-matrix ... --row-index 0 --dry-run --results-tsv .../results.dry_run.tsv`, sandbox runtime smoke attempt | runner tests 4 passed, py_compile 통과. 실제 matrix row 0 dry-run은 `results.dry_run.tsv` 1 row를 썼다. 첫 runtime smoke attempt는 sandbox Docker API permission denial로 server log에 `permission denied while trying to connect to the docker API`를 남겼다. 이 실패를 계기로 runner가 server process 조기 종료를 health wait 중 즉시 감지하도록 fast-fail gate를 추가했다. 실제 runtime smoke 재시도는 다음 행에서 완료됐다. |
| 2026-06-11 | N1.5 steering eval runtime smoke 및 cleanup 보강 | `pytest tests/test_groot_n15_instruction_steering_eval_plan.py tests/test_groot_n15_instruction_steering_eval_runner.py -q`, `plan_instruction_steering_eval.py --run-tag steer_eval_smoke_open_drawer_right_first_p8402_named --cell-id open_drawer_right --pathway dit --beta 0.1 --alpha-policy first --max-episodes-per-cell 1 --port 8402`, `run_instruction_steering_eval.py ... --row-index 0 --row-index 1`, pkl/action diff audit, verifier 2회, `ss -ltnp`, `docker ps` | planner/runner focused tests 7 passed, py_compile 통과. planner가 row별 deterministic `--name`을 server command에 넣고 runner가 finally에서 `docker rm -f`로 named container를 정리하도록 보강했다. runtime smoke는 baseline 1ep와 DiT `alpha=0.1,beta=0.1` 1ep 모두 `status=ok`, 각 pkl/csv/mp4 triplet과 `ep_meta` JSON 생성. 두 pkl은 `hidden_states[0].shape == (4,16,1024)`, `vl_hidden_states[0].shape == (2048,)`, `max_episode_steps=720`, `scenario_seed=100000`, `inference_seed=100000`. 같은 seed 기준 baseline vs DiT steered action L2는 `4.36035`라 steering이 action path에 반영됐다. verifier는 single-cell smoke 기준 `status=partial-ok`로 통과했고, 8402 port와 LeRobot 임시 container가 남지 않음을 확인했다. |
| 2026-06-11 | N1.6-aligned N1.5 block residual full collection 시작 | `tmux new-session -d -s lerobot_n15_block_residual_server ... --capture-vl --groot-dit-capture-layers 0,1,2,4,8,12,15`, `curl --noproxy '*' http://127.0.0.1:8400/health`, `tmux new-session -d -s n15_block_residual50_collect ... --max-episodes-per-cell 50`, first pkl shape audit, verifier partial | GPU free 상태에서 `target_instruction_fixed15_block_residual_50ep`를 시작했다. Server health는 DiT `groot_n15_dit_block_residual_tokens`, axes `[layer,model_token,feature_dim]`, layers `[0,1,2,4,8,12,15]`, VL dim 2048. 첫 pkl `ppcs_onion/task0--ep0--succ1.pkl`은 `hidden_states[0].shape == (7,49,1536)`, `vl_hidden_states[0].shape == (2048,)`, `scenario_seed == inference_seed == 100000`, `max_episode_steps=720`, `ep_meta` present. 초기 verifier는 `completed=2 expected=750`, `status=partial-ok`다. 수집 tmux는 running이다. |
| 2026-06-11 | aligned layer-preserved cache loader 및 layer key fit path 준비 | `pytest tests/test_groot_n15_instruction_pathway_features.py tests/test_groot_n15_instruction_conceptor_fit.py -q`, `py_compile`, `instruction_pathway_features.py ... --preserve-dit-layers`, `fit_instruction_conceptors.py --pathway dit_layer0 ... --dry-run`, `fit_instruction_conceptors.py --pathway vl ... --dry-run` | focused tests 7 passed, py_compile 통과. aligned smoke cache는 45 step rows를 만들었고 keys는 `dit (45,1536)`, `dit_layers (45,7,1536)`, `dit_capture_layers (7,)`, `dit_layer0/1/2/4/8/12/15 (45,1536)`, `vl (45,2048)`다. `fit_instruction_conceptors.py`는 `dit_layer0` 같은 path-safe cache key를 받을 수 있게 됐고, smoke dry-run은 feature_dim `1536`을 확인했다. |
| 2026-06-11 | aligned `dit_layer<i>` steering eval planner mapping 추가 | `pytest tests/test_groot_n15_instruction_steering_eval_plan.py tests/test_groot_n15_instruction_steering_eval_runner.py -q`, `py_compile` | focused tests 8 passed. planner는 conceptor fit pathway key `dit_layer0`를 matrix의 pathway/condition에는 유지하되 server command에서는 `--steering-pathway dit --steering-layer 0`으로 변환한다. runner의 `--pathway` filter도 `dit_layer0` 같은 aligned key를 받을 수 있다. |
| 2026-06-11 | aligned block residual full collection 16/750 checkpoint | `verify_rollout_collection.py ... --allow-partial --expected-hidden-shape 7,49,1536 --expected-token-count 49 --require-vl-hidden-states`, `df -h`, `tmux list-sessions`, `nvidia-smi` | verifier는 `completed=16 expected=750`, `status=partial-ok`. 현재 `ppcs_onion` 16/50, success 10. Disk free 131GB, tmux `lerobot_n15_block_residual_server`와 `n15_block_residual50_collect` running, GPU memory 약 7.8GB 사용. |
| 2026-06-11 | aligned conceptor set wrapper 및 29/750 checkpoint | `timeout 240 python -m pytest tests/test_groot_n15_aligned_conceptor_set_fit.py tests/test_groot_n15_instruction_conceptor_fit.py tests/test_groot_n15_instruction_steering_eval_plan.py tests/test_groot_n15_instruction_steering_eval_runner.py tests/test_groot_n15_instruction_pathway_features.py tests/test_groot_n15_http_feature_collect.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_safe_groot_verify_rollout_collection.py tests/test_serve_lerobot.py::TestSteeringRegistration -q`, `py_compile`, `git diff --check`, `/health`, aligned verifier partial | `fit_aligned_instruction_conceptors.py`를 추가해 aligned cache의 `dit_layer0/1/2/4/8/12/15`와 `vl`을 한 번에 dry-run/fit할 수 있게 했다. focused regression은 45 passed, 2 warnings이고 py_compile/diff check 통과. active aligned server health는 `feature_kind=groot_n15_dit_block_residual_tokens`, axes `[layer,model_token,feature_dim]`, layers `[0,1,2,4,8,12,15]`, VL dim 2048이다. verifier는 `completed=29 expected=750`, `status=partial-ok`, 현재 `ppcs_onion` 29/50, success 21이다. |
| 2026-06-11 | aligned post-processing runbook 갱신 및 36/750 checkpoint | `verify_rollout_collection.py ... --allow-partial --expected-hidden-shape 7,49,1536 --expected-token-count 49 --require-vl-hidden-states`, `pytest tests/test_groot_n15_aligned_conceptor_set_fit.py tests/test_groot_n15_instruction_conceptor_fit.py tests/test_groot_n15_instruction_pathway_features.py -q`, `py_compile`, `git diff --check` | runbook의 layer-specific dry-run 예시를 `fit_aligned_instruction_conceptors.py` 기본 경로로 바꿨고, 구현 표에 aligned set wrapper row를 추가했다. verifier는 `completed=36 expected=750`, `status=partial-ok`, `ppcs_onion` 36/50, success 28. focused tests는 9 passed이고 py_compile/diff check 통과. |
| 2026-06-11 | held-out eval seed exclusion gate 추가 및 51/750 checkpoint | `pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_seed_shard_merge.py tests/test_groot_n15_instruction_collect_wrapper.py tests/test_groot_n15_instruction_steering_eval_plan.py tests/test_groot_n15_instruction_steering_eval_runner.py -q`, `py_compile`, `select_instruction_seeds.py --help`, `verify_rollout_collection.py ... --allow-partial`, `git diff --check` | `select_instruction_seeds.py`에 `--exclude-selected-seeds`를 추가했다. collection manifest와 같은 `cell_id/scenario_seed`가 match되어도 held-out output에는 쓰지 않고 다음 matching seed를 찾는다. resume output이 exclude manifest와 겹치면 즉시 `ValueError`로 실패한다. focused regression은 29 passed이고 CLI help에 옵션이 노출된다. aligned verifier는 `completed=51 expected=750`, `status=partial-ok`, `ppcs_onion` 50/50 success 41, `ppcs_apple` 1/50 success 1. 실제 held-out eval manifest 생성은 아직 미실행이다. |
| 2026-06-11 | steering eval planner overlap gate 및 57/750 checkpoint | `pytest tests/test_groot_n15_instruction_seed_selector.py tests/test_groot_n15_instruction_steering_eval_plan.py tests/test_groot_n15_instruction_steering_eval_runner.py -q`, `py_compile`, `plan_instruction_steering_eval.py --help`, `verify_rollout_collection.py ... --allow-partial`, `git diff --check` | `plan_instruction_steering_eval.py`에 `--forbid-selected-overlap`을 추가했다. eval manifest와 collection/conceptor manifest가 같은 `cell_id/scenario_seed`를 공유하면 eval matrix를 쓰기 전에 실패한다. focused regression은 19 passed이고 CLI help에 옵션이 노출된다. aligned verifier는 `completed=57 expected=750`, `status=partial-ok`, `ppcs_onion` 50/50 success 41, `ppcs_apple` 7/50 success 7. 실제 held-out eval manifest 생성은 아직 미실행이다. |
| 2026-06-11 | N1.6-aligned contract 재확인 및 65/750 checkpoint | `verify_rollout_collection.py ... --allow-partial --expected-feature-kind groot_n15_dit_block_residual_tokens --expected-feature-axes layer,model_token,feature_dim --expected-hidden-shape 7,49,1536 --expected-capture-layers 0,1,2,4,8,12,15 --expected-token-count 49 --require-vl-hidden-states`, latest pkl spot check, `/health`, `tmux list-sessions` | N1.6 문서 기준 DiT residual stream + VL pathway contract에 맞춰 N1.5 aligned run을 재확인했다. verifier는 `completed=65 expected=750`, `status=partial-ok`, `ppcs_onion` 50/50 success 41, `ppcs_apple` 15/50 success 14다. active server health는 DiT `groot_n15_dit_block_residual_tokens`, axes `[layer,model_token,feature_dim]`, capture layers `[0,1,2,4,8,12,15]`, VL dim 2048이다. latest pkl spot check는 `scenario_seed == inference_seed`, `max_episode_steps=720`, `hidden_states[0].shape == (7,49,1536)`, `vl_hidden_states[0].shape == (2048,)`, `ep_meta` present를 확인했다. |
| 2026-06-11 | aligned ep_meta archive audit 및 73/750 checkpoint | `verify_rollout_collection.py ... --allow-partial --expected-hidden-shape 7,49,1536 --expected-capture-layers 0,1,2,4,8,12,15 --expected-token-count 49 --require-vl-hidden-states`, seed-keyed `ep_meta` JSON audit using `ep_meta_manifest_path(...)`, `/health`, `tmux list-sessions` | verifier는 `completed=73 expected=750`, `status=partial-ok`, `ppcs_onion` 50/50 success 41, `ppcs_apple` 23/50 success 20이다. active server는 aligned mode를 유지한다. pkl별 audit는 `rows=72`, pkl 내부 `ep_meta` missing 0, 대응 seed-keyed JSON missing 0, JSON env/seed/ep_meta mismatch 0이었다. JSON count는 live episode reset 직후 pkl보다 앞설 수 있으므로 archive 검증은 pkl별 `ep_meta_manifest_path(ep_meta/<task>/<cell_id>, env_name, scenario_seed)` 존재/내용 대조를 기준으로 한다. |
| 2026-06-11 | aligned selected-manifest 대조 및 78/750 checkpoint | pkl별 `(cell_id, episode_idx, scenario_seed, task, canonical_instruction)` vs `target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv`, `verify_rollout_collection.py ... --allow-partial --expected-hidden-shape 7,49,1536 --expected-token-count 49 --require-vl-hidden-states`, `git diff --check` | selected manifest는 750 rows, 15 cells, `(cell_id, scenario_seed)` duplicate 0, `ep_meta_path` missing 0이다. 현재 수집 pkl 77개를 selected manifest의 cell별 episode 순서와 대조했을 때 mismatch 0이다. 이후 verifier snapshot은 `completed=78 expected=750`, `status=partial-ok`, `ppcs_onion` 50/50 success 41, `ppcs_apple` 28/50 success 25다. 이 manifest가 held-out eval의 `--exclude-selected-seeds`/`--forbid-selected-overlap` 기준이다. |
| 2026-06-11 | aligned active collection 88/750 checkpoint | `/health`, `tmux list-sessions`, `verify_rollout_collection.py ... --allow-partial --expected-hidden-shape 7,49,1536 --expected-capture-layers 0,1,2,4,8,12,15 --expected-token-count 49 --require-vl-hidden-states` | active server health는 aligned mode를 유지하고, `lerobot_n15_block_residual_server`와 `n15_block_residual50_collect` tmux session은 running이다. verifier는 `completed=88 expected=750`, `status=partial-ok`, `ppcs_onion` 50/50 success 41, `ppcs_apple` 38/50 success 35다. |
| 2026-06-11 | wrong-layer block residual run purge and restart | stopped `lerobot_n15_block_residual_server` / `n15_block_residual50_collect`, removed `target_instruction_fixed15_block_residual_50ep`, preserved `target_instruction_fixed15_pathway_50ep/manifests/selected_instruction_seeds.tsv`, restarted server with `--groot-dit-capture-layers 0,2,4,8,10,12,15`, first-pkl shape audit, verifier partial | The previous full collection used capture layers `[0,1,2,4,8,12,15]` and was deleted. Seed-selection artifacts were not deleted: `selected_instruction_seeds.tsv` remains 750 rows plus header. Restarted server health reports layers `[0,2,4,8,10,12,15]`, VL dim 2048. First pkl samples show `hidden_states[0].shape == (7,49,1536)`, `vl_hidden_states[0].shape == (2048,)`, `token_count=49`, `scenario_seed == inference_seed`. Verifier reports `completed=3 expected=750`, `status=partial-ok`. |

## 산출물

최소 deliverable:

1. `target_instruction_fixed15_pathway_50ep` raw_rollouts와 verifier 통과 로그
2. cell별 SR table
3. cell별 VL vs DiT separation table
4. cell별 conceptor fit summary
5. baseline vs steered SR eval table

성공 claim 기준:

- "N1.5에서 COAST/VL steering이 된다"는 말은 cell-held-out eval에서 type-matched steering의
  ΔSR이 양수이고, Wilson CI까지 같이 제시된 뒤에만 쓴다.
- geometry AUROC만으로 SR 개선을 주장하지 않는다.
- 20 episode eval은 후보 선별이고, 최종 claim은 50 episode 이상에서 낸다.

## 현재 결론

이번 작업은 N1.6 Phase 3 결과를 그대로 반복하는 것이 아니라, NOTALL/COAST의 N1.5 조건으로
돌아가되 task-random confound를 제거하는 재검증이다. 핵심은 "N1.5 체크포인트로 다시 수집"이
아니라 **instruction cell을 고정한 VL+DiT 데이터셋을 먼저 만드는 것**이다.
