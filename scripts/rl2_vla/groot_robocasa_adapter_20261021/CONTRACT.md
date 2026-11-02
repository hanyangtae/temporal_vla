> 2026-10-21 v5: Primary baseline now uses `rl2_expert_rewards.py`: -1 before last3, zero last3,
> td_mask zero last3, gamma .99 (passed by QAM main). Raw next.reward/done are retained separately.
> Earlier native-reward runs remain diagnostic smokes, not RL2 reward reproductions.
> Synthetic terminal limits the critic effective prefix; this is not a claim of physical termination.
> Full H16 expert BC targets stay unchanged. Recorded absorbing decision rows remain terminal samples.
> `train_qam_baseline.py` writes both reward fields, a transition audit, and checkpoint reward provenance.
> Frozen102 extraction completed/verified; reward derivation does not modify that artifact.

# GR00T N1.5 RoboCasa ↔ QAM flow velocity 합성 계약 (2026-10-21, v4)

이 폴더는 계약·adapter·**QAM trainer 통합(prefix critic)**·OpenDrawer baseline 추출/학습 CLI·CPU 테스트를 담는다.
이 폴더 코드는 split 을 만들지 않는다 (split 은 메인이 만든 `OpenDrawer_aux_split_v2.json`).
실행 결과는 코드가 아니라 receipt 에 기록한다:
- v3 CPU 테스트·합성 smoke: `outputs/analysis/groot_robocasa_adapter_20261021/RECEIPT.md`
- 메인 세션 smoke 종합 (2026-10-21): `outputs/analysis/groot_robocasa_lineage_20261021/FINAL_FOLLOWUP.json`
  - 실제 GR00T GPU forward (2 episode / 12 결정, gate0 bit 동일, feature 캡처 일치): `lerobot_latent_smoke.json`
  - 실제 latent 30 update CPU 학습 smoke (축소 망): `outputs/rl2_vla/groot_prefix_qam_real_latent_smoke_20261021/result_portable.json`
  - production 망 GPU 10 update throughput: `qam_gpu_throughput_smoke.json`
  - reward/done 실측 audit (102 episode): `OpenDrawer_reward_audit.json`
- v4 CLI (§11) 는 코드와 CPU 테스트만 추가했다. 전체 추출·500k 학습은 **아직 실행하지 않았다**.

v4 = v3 + §11 (OpenDrawer 92/10 baseline 추출·학습 CLI) + `published_package_subset_for_baseline` purpose.
v3 = v2 + §10 (QAM prefix critic trainer) + §2/§8 membership 증거 모드.
v2는 `outputs/analysis/groot_robocasa_lineage_20261021/INTERFACE_REVIEW.md` 리뷰를 반영한 수정판이다
(v1은 QAM 관측을 DiT block residual 1536으로 잘못 잡았다).

| 파일 | 역할 |
|---|---|
| `contract.py` | checkpoint-120000 config/metadata 고정(sha), action codec, raw LeRobot12 → 모델 순서 재배열, QAM flags 검증 |
| `composition.py` | GR00T `get_action` 루프를 옮긴 합성 sampler + action_decoder 입력 feature 캡처(write-once) |
| `transitions.py` | episode 별 시간순 chunk transition 생성, expert target, train/val 분리 sampler |
| `data_contract.py` | finetuning 데이터·TRQAM rollout manifest를 fail-closed로 검증, membership 증거 모드, aux split 재계산 검증, record → feature 추출 |
| `qam_prefix.py` | upstream `QAMAgent` subclass (실행 prefix critic, 행별 discount) + `PrefixQAMTrainer` 학습 API·checkpoint·velocity callback |
| `smoke_prefix_qam.py` | **합성** transition CPU smoke CLI (데이터셋 학습 아님) |
| `tests/` | CPU 테스트. 실제 lerobot/upstream 소스를 AST로 꺼내 parity 비교, JAX trainer 검사 |

## 1. 고정 checkpoint

- 경로: HF cache `models--robocasa--robocasa365_checkpoints/snapshots/*/gr00t_n1-5/multitask_learning/checkpoint-120000`
  (`resolve_cached_checkpoint_dir`, `scripts.path_setup.DATA_ROOT` 기준. 다운로드 없음).
- 내용 고정: `config.json` sha256 `6713ae6e…0526`, `experiment_cfg/metadata.json` sha256 `8be0fc60…0b69`.
  두 snapshot(14895998…, c484448a…)이 같은 blob을 가리킨다.
- serve 경로: `configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml` →
  `lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py::get_action`
  (lerobot 1396b9f. Isaac-GR00T-N1.5 784fb74와 공백만 다름).
- config 거부 대상: action_dim≠32, horizon≠16, num_inference_timesteps≠4, DiT 내부 폭≠1536(32×48),
  DiT layer≠16, **DiT output_dim≠1024, hidden_size(action_decoder 입력)≠1024**,
  backbone_embedding_dim≠2048, model_type≠gr00t_n1_5, sha 불일치.
  내부 DiT 1536 검사는 그대로 둔다. 1536은 관측 폭이 아니라 모델 구조 검사용이다.

## 2. 학습 데이터 계보 (finetuning 데이터)

- checkpoint-120000(RoboCasa365 multitask)의 학습 데이터 = **`pretrain_human300`**.
  - 300 task (atomic 65 + composite 235), human/pretrain, filter_key `100_demos`, 공식 설명상 약 30,000 demo.
  - registry commit `4f8a2980…`. **Tabletop24k가 아니다.**
  - 출처: `outputs/analysis/checkpoint_training_data_20260916/` (report.md, provenance.json,
    pretrain_human300_manifest.tsv).
- trainer_state: global_step 120000, max_steps 300000, batch 128.
  300000은 계획 상한이다. 저장된 120000 checkpoint와 모순되지 않는다.
- 로컬 OpenDrawer LeRobot `v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot`:
  - 102 episode / 20488 frame, PandaOmron, split은 `train 0:102`뿐이다 (val/test 없음).
  - info.json sha256 `73367b37…`.
- **미해결** (`data_contract.FINETUNE_UNRESOLVED`): historical `100_demos` 필터가 로컬 102개 중
  정확히 어느 100개였는지. 이 목록은 여전히 모른다.
- downstream split: 메인이 `outputs/analysis/groot_robocasa_lineage_20261021/OpenDrawer_aux_split_v1.json`
  을 만들었고, 경로 이식성만 바꾼 `OpenDrawer_aux_split_v2.json` (sha256 `7959f295…af64`,
  `data_contract.AUX_SPLIT_V2_SHA256`) 이 현재 정본이다. membership·salt·parquet hash 는 v1 과 같다.
  baseline CLI (§11) 는 v2 sha 가 아니면 거부한다. 102개 전부를 inventory 하고 `sha256(salt|episode_id)` 순위 최저 10개를 val, 나머지 92개를
  train 으로 둔 **새 보조 split** 이다 (원래 GR00T split 아님, frozen GR00T 는 양쪽을 다 봤을 수 있다).
  `validate_aux_split` 가 rank hash·split·counts 를 독립 재계산하고, `verify_files=True` 면 package 의
  info.json·episodes.jsonl·parquet 102개 sha256 을 대조한다. 파일에 적힌 절대경로(`parquet_path`)는
  쓰지 않고 `DATA_ROOT` + info.json `data_path` 로 경로를 다시 만든다.
- **membership 증거 모드** (`load_finetune_data_manifest(..., purpose=...)`, purpose 필수):

  | purpose | 필요한 episode_membership | 결과 주장 |
  |---|---|---|
  | `published_package_subset_for_smoke` | `mode=published_package_subset_for_smoke`, `historical_membership_claimed=false` (명시 필수), package inventory sha·102개, `aux_split_sha256` | 공개 package 102개의 새 92/10 split. **원래 100_demos membership 아님**, smoke/보조 정책 전용 |
  | `published_package_subset_for_baseline` | smoke 와 같은 증거, `mode=published_package_subset_for_baseline` | 같은 split 으로 production QAM baseline 학습 (§11). **원래 100_demos membership 아님**. smoke manifest ↔ baseline purpose 교차 사용은 거부 |
  | `original_full_reproduction` | `mode=historical_exact_100_demos`, evidence, `historical_episode_list_sha256` = manifest episode 목록 지문 | 원래 membership. 현재는 목록 미상이라 **차단** |

  - package 모드 manifest 로 full reproduction 을 요청하면 "historical 100_demos 목록 미상"으로 차단된다.
  - 반환 dict 의 `membership_claim.original_membership_confirmed` 가 주장 범위를 기록한다.
  - `package_smoke_manifest(split_path, feature_extraction=..., mode=...)` 가 package 모드 manifest 를 조립한다
    (mode 기본 smoke, baseline CLI 는 `published_package_subset_for_baseline`).
    feature_extraction 블록은 메인이 실제 decoder-입력 latent 를 뽑은 뒤 확정해 넘긴다.

## 3. Action 공간

- 모델 출력은 `(B, 16, 32)`이다. 앞 12차원이 실제 action이고 12~31은 padding이다.
- **모델 순서** (PandaOmron data config): eef_pos 3 · eef_rot 3 · gripper_close 1 · base_motion 4 · control_mode 1.
  metadata.json의 key 순서(알파벳)는 학습 concat 순서가 아니다.
- **raw LeRobot12 순서** (OpenDrawer `meta/modality.json` 실측):
  base_motion `0:4` · control_mode `4:5` · eef_pos `5:8` · eef_rot `8:11` · gripper `11:12`.
  - `raw_to_model_index(modality["action"])`는 metadata slice로 gather index를 만든다
    (`[5,6,7,8,9,10,11,0,1,2,3,4]`). key 집합, 폭, 겹침이 틀리면 거부한다.
  - `reorder_raw_to_model` / `reorder_model_to_raw`는 순열만 수행한다. 양방향 roundtrip을 테스트한다.
- 정규화(upstream `Normalizer` 미러):
  - min_max: `2(x−min)/(max−min)−1`. inverse에서 clip하지 않는다.
  - binary(모델 idx 6, 11): `x>0.5` → {0,1}. raw 데이터의 binary 값은 ±1이다.
  - env 래퍼는 {0,1}을 0.5 임계로 {−1,+1}로 바꾼다.
  - degenerate 차원은 모델 idx 10 (base_motion[3], min=max=0)이다.
- **expert target 보존**: `expert_chunk_targets`는 데이터셋 action을 재배열하고 정규화한 값만 쓴다.
  GR00T 예측으로 대체하지 않는다. episode 끝을 넘는 위치는 0으로 채우고 `valid=False`로 표시한다.

## 4. Flow 시간과 합성

| | GR00T N1.5 | QAM (RL2 third_party) | pi0 (SIMPLER Bridge) |
|---|---|---|---|
| 시간 | t_k = k/4, 0→1 | t = i/flow_steps | 1→0, dt<0 |
| 보간 | x_t = (1−t)·noise + t·action | 동일 | x_t = t·noise + (1−t)·action |
| velocity | action − noise | 동일 | noise − action |

- 합성식 `x ← x + dt·v`:
  - `v = v_vla` (lam=0이면 QAM을 호출하지 않는다)
  - `v[..., :12] = (1−lam)·v_vla + lam·v_qam`
  - `v[..., 12:] = v_vla`
  - `lam = gate·qam_weight`, gate ∈ {0, 1}
- pi0 코드의 `w·v_vla − (1−w)·v_qam`은 시간 방향이 반대라서 생긴 부호다. GR00T에 복사하지 않는다.
- QAM flags 필수 조건: `ob_dims=[1024]`, `action_dim=12`, `horizon_length=16`, `flow_steps=4`,
  `action_chunking`, `best_of_n=1`, 그리고 `groot_robocasa_contract` provenance 블록.
  - provenance 항목: config/metadata sha, feature_signature,
    `feature_interface=action_decoder_input_last16_mean_denoise0`, 시간 규약, action 공간.
  - `ob_dims=[1536]`(block residual QAM)과 `[2048]`(VLM)은 "다른 인터페이스"로 거부한다.
  - pi0 Bridge QAM도 ob 1024라서 차원만으로는 구분되지 않는다. bridge 표식, 7D/H4/flow10,
    provenance 부재로 거부한다.
- QAM 입력 noisy action은 정규화 공간 active12를 flat하게 편 `(B, 192)`다.
  QAM velocity는 `(B, 192)` 또는 `(B, 16, 12)`만 받는다.

## 5. Gate0 baseline 보존

- RNG 소비는 upstream과 같다: `torch.randn((B, 16, 32))` 1회.
- gate=0 또는 qam_weight=0이면:
  - 출력이 upstream `get_action`과 bit 단위로 같다.
  - torch/numpy RNG 상태도 같고, QAM은 호출되지 않는다.
- gate=1이어도 torch RNG 소비는 같다 (`scripts/rl2_vla/PAIRED_EVAL.md` paired_rng_v2 전제).
- feature 캡처용 hook은 action_decoder **forward pre-hook**이다. 입력을 읽기만 하고 반환값은 None이라
  출력을 바꾸지 않는다. 호출이 끝나면 hook을 제거한다.

## 6. Feature 계약 (QAM 관측 1024) — 확정

- **정의**: `action_head.action_decoder`의 입력 (= DiT `proj_out_2` 출력, `(B, 49, 1024)`) 중
  **마지막 16개 action token만** 평균 → `(B, 1024)`.
  - 같은 `get_action`의 **첫 denoise call (flow step 0)** 에서만 캡처한다.
  - 4 flow step 동안 바꾸지 않고 그대로 쓴다 (numpy 배열 read-only).
  - state `[0:1]`, future `[1:33]` token은 포함하지 않는다.
  - pi0처럼 앞 token을 하나 버리는 slicing이 아니라, 정확히 `[-16:]` (= `[33:49]`)이다.
- `FeatureSpec()`에는 자유 파라미터가 없다. 다른 값(capture_point, token_selection, pool,
  denoise_step, dim)을 넣으면 생성 단계에서 거부한다.
- **write-once**: `FirstCallCapture`는 첫 decoder 호출 입력을 복사해 보관한다.
  - 이후 호출은 개수만 센다. 슬롯을 다시 쓰면 `ContractError`다.
  - sampler는 hook 값과 루프의 `model_output`이 `torch.equal`인지 확인한다.
  - decoder 호출 수가 정확히 4인지도 확인한다.
- **거부 대상**:
  - DiT block residual 1536 (`transformer_blocks[*]` 출력)
  - VLM context 2048
  - token 수≠49
  - step>0 feature, 미래/stale env_step, signature 불일치, 비유한값
  - 1536 → 1024 차원 변환이나 projection으로 "대체"하는 것도 금지다.
    다른 인터페이스이기 때문이다.
- gate_fn을 쓰면 같은 step-0 feature를 보고 gate를 정한다.
- 기존 rollout pkl의 `[7, 4, 49, 1536]` (layer, denoise_step, model_token, feature_dim) capture는
  이 계약의 입력이 **될 수 없다**. TRQAM에 쓰려면 decoder 입력을 다시 캡처해야 한다 (GPU 필요).
- **허용 record layout** (`check_capture_layout`, `feature_capture_point=action_head.action_decoder.input` 필수):

  | feature_axes | record_shape | 처리 |
  |---|---|---|
  | `[denoise_step, model_token, feature_dim]` | `[4, 49, 1024]` | step 0, `[-16:]` 평균 |
  | `[denoise_step, action_token, feature_dim]` | `[4, 16, 1024]` | step 0, 평균 |
  | `[model_token, feature_dim]` 등 denoise 축 없음 | `[49, 1024]` / `[16, 1024]` / `[1024]` | `denoise_step_captured=0` 명시 필수 |

  `layer` 축이 있거나, 마지막 폭이 1536/2048이면 거부한다.

## 7. Chunk transition (시간순 생성)

`transitions.build_episode_transitions`가 episode 하나씩 처리한다. 순서와 규칙은 다음과 같다.

1. 결정 시점은 `t_i = i·m`이다 (`0, m, 2m, … < T`). m은 실제 실행 step 수(1..16)이고,
   기존 수집은 m=5다. 결정 grid가 이와 다르면 거부한다.
2. 각 결정의 실행 길이는 `m_i = min(m, T − t_i)`다. 마지막 chunk는 짧을 수 있다(partial).
3. 출력 필드:
   - `rewards_i = Σ_{k<m_i} γ^k · r[t_i+k]` — 실제로 실행된 step만 합한다.
   - `next_observations_i = latent(t_i + m_i)` — 같은 episode의 다음 결정 latent다.
   - `discounts_i = γ^{m_i} · (1 − terminal_i)`
   - 그 외: `actions` (16, 12), `chunk_valid`, `executed_mask` (앞 m_i개 True), `executed_m`, `has_next`.
4. **terminal과 truncation** — `end_reason`을 반드시 명시한다:
   - `terminal`: env가 종료한 경우(성공 종료 등)다. 마지막 chunk의 discount는 0이고,
     next latent는 쓰지 않는다(zeros, `has_next=False`).
   - `truncated`: timeout인 경우다. terminal이 아니므로 마지막 chunk도 `γ^{m_i}`로 bootstrap한다.
     이때 `final_latent = latent(T)`가 **없으면 거부**한다.
5. episode 경계: 다음 latent는 해당 episode의 `latents`나 `final_latent`에서만 가져온다.
   따라서 다른 episode로 넘어가지 않는다.
6. 섞기: 모든 transition을 만든 뒤에만 `TrainSampler`가 행을 뽑는다. 다음 관측은 행에 저장된
   `next_observations`를 쓰며, 섞인 배열에서 `index+1`로 찾지 않는다.
7. train/val 분리:
   - `build_split`은 split이 섞이면 거부한다.
   - `TrainSampler`(optimizer용)는 `split=="train"`만 받는다. val과 episode_id나 run_group이 겹쳐도 거부한다.
   - val은 `val_batches`(순차, 평가 전용)로만 읽는다.
8. reward 계약은 manifest `reward_contract`에 이름과 γ로 명시한다.
   제공 helper는 `sparse_success_rewards`(`sparse_success_terminal`) 하나다.

### 원래 RL2/QAM·Octo 경로와의 차이 (동작 비교)

| 항목 | RL2 third_party QAM (`utils/datasets.py::sample_sequence`, `agents/qam.py`) | 이 adapter |
|---|---|---|
| 다음 관측 | flat buffer에서 랜덤 시작 index `idx`를 고르고, `idx+i` 연속 행의 저장된 `next_observations`를 쓴다. 저장 순서가 시간순이고 terminal이 표시돼 있다는 전제다. | episode 단위로 먼저 `latent(t+m)`을 붙여 저장한다. 그 뒤에 섞는다. |
| bootstrap 계수 | `discount**horizon_length * masks[..., -1]` (고정 H) | 행마다 `γ^{m_i}·(1−terminal_i)`. m_i는 실제 실행 길이다. |
| reward | H step 누적 할인합 | 실행된 m_i step만 할인합 |
| terminal/truncation | 온라인 수집 시 `masks = 1 − terminated`, `terminals = done`. truncation은 bootstrap을 유지한다. | 같은 의미를 `end_reason`으로 명시한다. truncation인데 latent(T)가 없으면 거부한다. |
| Bridge(Octo/V-GPS) | `next_observation`, `td_mask`는 외부 Octo/V-GPS 파이프라인(`subsample_length=100`, window 1, chunk N)에서 온다. 그 파이프라인 소스는 이 repo에 vendored되지 않았다. subsampling 이후 index 오프셋은 로컬에서 확인할 수 없다. | 그 방식을 따르지 않는다. 시간순으로 직접 생성한다. |
| Octo `skip_unlabeled` | language annotation이 없는 trajectory를 거르는 옵션이다 (실패 outcome 필터가 아니다). | outcome 필터는 manifest의 `source_role`과 `success`로 명시한다. |

- 행 필드 `next_executed_m` (v3 추가): has_next 면 명목 m, 종료 행이면 0. bootstrap 에서 다음 결정의
  실행 길이다 (§10.2).
- 이 차이들을 소비하는 trainer 는 §10 (`qam_prefix.py`) 이다. upstream 은 수정하지 않았다.

## 8. 데이터 manifest (fail-closed)

- **(A) finetuning 데이터 → QAM/RL2** (`load_finetune_data_manifest(path, feature_signature=, purpose=)`). 필수 항목:
  - `dataset_identity`: confirmed. `dataset_soup=pretrain_human300`, registry_commit, filter_key `100_demos`,
    evidence 필요. Tabletop24k, target500, MimicGen은 거부한다.
  - `episode_membership` — purpose 별 증거 모드 (§2 표). full reproduction 은 현재 차단.
  - `split_policy.status=confirmed` (decided_by, unit)
  - `normalization_source=checkpoint_metadata`
  - `action_target=expert_dataset_action`
  - `feature_extraction`: confirmed, capture_point는 decoder 입력, denoise 0
  - split은 train/val만 허용한다. 같은 run_group이 train과 val에 동시에 있으면 거부한다.
- **(B) 우리 rollout → TRQAM** (`load_trqam_rollout_manifest`):
  - `scenario`, `reward_contract`(name, gamma), `evaluation`이 모두 confirmed여야 한다.
    `transition.execute_steps`도 필요하다.
  - episode `source_role`:
    - `historical_base`, `historical_expert`: 기존 조건의 base rollout과 expert demo
    - `changed_failure`: 바뀐 조건의 실패. `success is False`만 허용한다.
    - `changed_success`는 adaptation 금지다. changed 조건에서 success가 True이거나
      미상(None/누락)이면 거부한다.
  - episode별 필수 항목: `end_reason`(terminal|truncated), `has_feature=true`(decoder 입력 캡처),
    `scenario_seed`, `run_group`, decoder 입력 layout.
  - `evaluation`의 episode_ids, run_groups, scenario_seeds, noise_ids, reset_ids는
    adaptation 쪽(train+val) 값과 서로소여야 한다. eval episode_ids와 scenario_seeds는 비어 있으면 안 된다.
  - unseen-scene 일반화는 주장하지 않는다.
- 이 코드는 split을 **만들지 않는다**. 미확정이면 `DataContractBlocked`를 낸다.

## 9. 검증 범위 (receipt 기준)

이미 한 것 (메인 세션 smoke, 근거 파일은 머리말 목록):
- 실제 GR00T 가중치 로드·GPU forward 와 gate0 bit 동일, decoder 입력 feature 캡처 일치:
  2 expert episode / 12 결정만 (`lerobot_latent_smoke.json`). 전체 102 episode 는 아님.
- 실제 latent 로 축소 망 30 update (CPU), production 망 GPU 10 update throughput. 둘 다 학습 품질 주장 아님.

아직 안 한 것:
- 102 episode 전체 추출 (§11 CLI), production 500k step 학습, validation 선택.
- 학습된 QAM 을 실제 GR00T 에 gate=1 로 합성한 rollout (bf16 gate=1 수치 포함), online 성공률 평가.
- lerobot MIN_MAX unnormalizer 와 이 codec(native 미러)의 일치 여부 실행 검증 (특히 degenerate idx 10 의 eps 처리).
- TRQAM 적응 (시나리오 답변 대기, §8 B 계약이 차단).

## 10. QAM trainer 통합: 실행 prefix critic (v3)

구현: `qam_prefix.py`. upstream `RL2-VLA/third_party/qam` 의 `QAMAgent`, `Value`, `ActorVectorField`,
`TrainState`, `save_agent`/`restore_agent_with_file` 를 **수정 없이** import 하고, critic 이 닿는 메서드만
subclass(`PrefixCriticQAMAgent`)에서 바꾼다. `agents/__init__.py`(다른 agent 전부 import)는 건너뛰고
`agents/qam.py` 만 파일 경로로 불러온다.

### 10.1 critic 정의

- actor 는 H16×12 = 192 차원 chunk 전체를 예측한다. 실행되는 것은 앞 m=5 step 이다.
- critic: `Q(s, a ⊙ M, M)`. `M` 은 prefix mask (16,), 앞 k 개만 1이다.
  - `PrefixMaskedValue` 가 upstream `Value` 를 감싼다. 입력 = `concat(obs, a·repeat(M, 12), M)`.
  - a 의 tail 은 critic 안에서 0 을 곱해 지운다 → tail 값과 무관하고, ∂Q/∂a_tail 은 정확히 0 이다.
  - M 을 입력에 붙여 duration 을 조건으로 준다. "tail 이 0 인 action" 과 "실행되지 않은 tail" 을 구분한다.
  - 의미: `Q(s, a, k) = E[Σ_{j<k} γ^j r_{t+j} + γ^k V^π(s_{t+k})]` (중간 종료 시 그 뒤 0).
    `V^π(s) = Q(s, π(s), M_m)` — 정책은 시간 한도를 관측하지 않으므로 배포 시 항상 명목 m 을 실행한다.
- mask 인자 없이 critic 을 부르면 `TypeError` 로 실패한다. upstream 의 `critic_loss`(γ^H 경로),
  fql/edit 경로가 우리 agent 에서 실수로 실행되어도 mask 없는 critic 접근은 일어나지 않는다.

### 10.2 어디에 어떤 mask 를 쓰는가

| 경로 | mask | 근거 |
|---|---|---|
| critic TD loss (데이터 행) | 행의 실제 `executed_m` (마지막 partial chunk 는 m_i < 5) | 실제로 실행된 action 만 결과에 영향 |
| target critic bootstrap | 행의 `next_executed_m` = 명목 5 (종료 행 0, discount 0 이라 무관) | V^π(s') 는 명목 m 실행. 데이터의 다음 행 m_{i+1} (종료·절단으로 짧아진 값)을 쓰지 않는다 |
| actor adjoint terminal 조건 | 명목 5 | 정책 개선 대상 = 배포 시 실행분 |
| sample ranking (best_of_n), 평가 | 명목 5 | 동일 |

- TD target: `y_i = r_i + d_i · (mean − ρ·std)_ens Q_target(s'_i, clip(π(s'_i)), M_{next_m_i})`.
  `d_i = γ^{m_i}(1 − terminal_i)` 는 transition 이 준 값을 **한 번만** 곱한다. `config.discount` 는 기록용이며
  target 에 쓰이지 않는다 (테스트로 확인).
- 종료(terminal) 행: d=0, next latent 는 0, `next_executed_m=0`.
- 절단(truncated) 마지막 행: d=γ^{m_i}, next = latent(T), `next_executed_m=5`.
- `to_qam_batch` 가 host 쪽에서 행 의미를 검사하고 어긋나면 거부한다:
  executed_mask = prefix(executed_m), 1 ≤ executed_m ≤ 5, 실행 prefix ⊆ chunk_valid,
  terminal ⇔ ¬has_next, `discounts == γ^{m_i}(1−terminal)` (γ^H 상수·재곱이면 거부),
  `next_executed_m == (has_next ? 5 : 0)`, 종료 행 next_observations = 0, 비유한값.

### 10.3 actor

- BC flow loss (actor_slow): target 은 **16 전체 expert chunk** 다. `chunk_valid` 로 loss 를 가린다.
  padding 위치는 target 을 0 으로 고정한 뒤 보간한다. loss 만 가리면 padding 값이 `x_t` 입력을 통해 valid
  차원 예측에 새어 들어간다 (첫 테스트에서 실제로 발견해 고쳤다). 모든 위치가 valid 면 upstream 식과 같다.
- adjoint matching: upstream `adj_matching` 과 같은 SDE/ODE·역전파이고, terminal adjoint
  `−inv_temp·∇_a Q(s, a, M_5)` 만 masked critic 에서 나온다 → terminal adjoint 의 tail 은 정확히 0 이다.
  - 역방향 전파 `adj ← adj + h·J^T adj` 에서 base flow(actor_slow) Jacobian 이 prefix·tail 을 섞으므로,
    중간 시간의 adjoint tail 은 0 이 아닐 수 있다. 이것은 critic 접근이 아니다. tail noise 궤적이 joint
    velocity field 를 통해 실행 prefix 에 주는 영향이며, prefix 만의 목적함수에 대한 정확한 adjoint 다.
  - 독립 대조: prefix 60 차원만 받는 함수의 gradient 와 terminal adjoint 가 같다 (테스트).
- 공개 기본 레시피 `fql_alpha=0`, `edit_scale=0`, `best_of_n=1` 경로만 지원한다.
  `fql_alpha>0`, `edit_scale>0`, `residual=True`, `action_chunking=False`, execute_steps ∉ [1,16] 은
  생성 단계에서 `ContractError` 로 거부한다 (masked critic 경로 미구현·미검증).
- target 갱신은 upstream 과 같다 (critic, actor_slow 만 Polyak. target_actor_fast 는 갱신되지 않고 쓰이지도 않음).

### 10.4 설계 비교

| 대안 | 문제 | 이 설계 |
|---|---|---|
| actor horizon 을 5 로 줄임 | GR00T 합성은 flow step 마다 16×12 velocity 가 필요하다. 5-step actor 는 tail velocity 를 줄 수 없어 prefix=QAM·tail=VLA 혼합 적분이 되고, DiT 가 16 token 을 함께 보는 chunk 구조가 깨진다. 16 전체 expert BC target 도 버리게 된다. partial chunk 문제는 그대로 남는다. | actor 는 16 유지, critic 만 prefix |
| full-chunk critic Q(s, a_192) | 데이터에서 tail(5..15)은 결과에 인과 영향이 없는 자유도인데, 수집 정책 때문에 결과와 상관될 수 있다 (expert tail ↔ 성공). critic 이 이 상관을 학습하면 adjoint 가 tail 을 밀고, joint flow 를 통해 실행 prefix 까지 왜곡할 수 있다. target critic 도 π 의 미실행 tail 을 평가한다. partial 행(m_i<5)을 표현하지 못한다. | tail 을 critic 입력에서 제거 + duration 조건 |
| 고정 prefix critic (입력 = 앞 60 차원) | m=5 고정이면 동치지만 partial 행을 표현하지 못한다. | mask 조건이 이를 일반화 |

### 10.5 원본 레시피 대비 변경

공개 레시피 = `RL2-VLA/third_party/qam/train_latents.sh` (Bridge latent) + `agents/qam.py` 기본값.

| 항목 | 공개/upstream | 이 trainer |
|---|---|---|
| 관측 | pi0 Bridge latent (외부 파이프라인) | GR00T decoder 입력 last16 평균 1024 (§6) |
| action / horizon / flow | 7D · H4 · flow 10 (기본값) | 12D · H16 · flow 4 (GR00T 고정) |
| 실행 길이 | chunk 전부 실행 | 16 예측 / 5 실행 |
| critic 입력 | obs + 전체 chunk | obs + masked prefix + mask (§10.1) |
| TD target | `rewards[-1] + γ^H·masks[-1]·next_q` | `r_i + d_i·next_q`, d_i 행별 (§10.2) |
| 다음 관측 | flat buffer `sample_sequence` 의 `next_observations[..., -1]` | episode 시간순 transition 행에 저장된 latent (§7) |
| critic 가중치 | `valid[..., -1]` | 모든 행 유효 (transition 단위로 생성) |
| BC loss | 192 차원 평균, `valid[-1]` 가중 | chunk_valid mask, padding target 0 고정 |
| adjoint 초기값 | ∇ full-chunk critic | ∇ masked critic (tail 0) |
| ranking / val | 전체 chunk critic, action_mse 192 전체 | 명목 m mask, mse 는 valid 전체·실행 prefix 따로 |
| fql / edit / residual | 지원 | 거부 (fail closed) |
| reward | Bridge streaming 경로는 외부 파이프라인 reward (`--sparse` 의 `process_train_dataset` 는 Bridge 경로에서 호출되지 않음) | manifest `reward_contract` (예: `sparse_success_terminal`) |
| 기타 hyperparam | inv_temp 0.1, lr 3e-4, batch 256, num_qs 10, ρ 0.5, τ 0.005, 512×4, clip_grad/adj, target_actor, use_target_grad | 같은 값 (production profile). smoke 는 망만 32×32·ensemble 2·batch 8 |
| 학습 길이 | offline 500k step, save 500k | 500k 기본, 0·50k 배수·마지막에 평가·저장 (§11, 아직 미실행) |

### 10.6 학습 API

```python
from transitions import build_split, TrainSampler
from qam_prefix import PrefixQAMTrainer
train = build_split(train_eps, split="train", execute_steps=5, gamma=γ)
val = build_split(val_eps, split="val", execute_steps=5, gamma=γ)
tr = PrefixQAMTrainer.create(profile="production", gamma=γ, seed=0, data_label="...")
tr.fit(TrainSampler(train, val=val, seed=0), steps=N, batch_size=256, val_set=val, eval_every=K)
tr.save(new_out_dir, feature_signature=SPEC.signature())       # params_<step>.pkl + flags_<step>.json
tr = PrefixQAMTrainer.load(new_out_dir, step, feature_signature=SPEC.signature())
ComposedFlowSampler(qam_fn=tr.velocity_fn(), qam_weight=w)        # GR00T 합성 (§4)
```

- profile: `production` (512×4, ensemble 10) / `smoke_reduced_network` (CPU smoke 전용, 32×32·2).
  두 profile 모두 관측 1024·action 12·H16·flow 4·m=5 계약을 바꿀 수 없다. 계약 key override 는 거부.
- `flags_<step>.json` 에 `groot_robocasa_contract` provenance 와 `prefix_critic_contract` 를 함께 저장한다.
  `load` 는 `validate_qam_flags` 와 prefix 계약 버전을 검사한다. prefix 계약이 없는 checkpoint
  (full-chunk critic QAM, Bridge QAM)는 거부한다.
- `velocity_fn()` 은 ComposedFlowSampler 의 `qam_fn(feature (B,1024), noisy (B,192), t)` 형식이다.
  velocity 는 `actor_fast(obs, x, t/4)` (inv_temp=0 이면 actor_slow) 로, upstream `compute_flow_actions` 와
  같은 망·시간이다. lam=1 이면 합성 결과 active12 가 QAM Euler 적분과 같다 (테스트).
- 실제 latent → `EpisodeInput` 변환은 `baseline_data.load_extraction` 이 한다 (§11).

## 11. OpenDrawer 92/10 baseline: 추출·학습 CLI (v4)

| 파일 | 실행 환경 | 역할 |
|---|---|---|
| `extract_opendrawer_latents.py` | 기존 lerobot image 컨테이너, GPU (발사·lease 는 메인) | 102 expert episode → 결정별 decoder-입력 latent |
| `baseline_data.py` | 양쪽 (numpy) | 추출 계약·episode commit·resume 검증·검증 loader |
| `train_qam_baseline.py` | host `.venvs/rl2_iid` (JAX) | prefix-critic QAM production 학습 + validation 선택 |

### 11.1 추출

- 모델: `robocasa/robocasa365_checkpoints` rev `c484448a…` `gr00t_n1-5/multitask_learning/checkpoint-120000`.
  HF cache 의 그 snapshot 경로를 직접 쓰고 (`snapshot_download` 호출 없음), config/metadata sha 를 검사한다.
  모델은 프로세스당 한 번 올린다.
- 전처리: serve 경로 그대로. `scripts/serve/lerobot.py` 를 importlib 로 별도 이름에 불러온다.
  **그 전에 lerobot 패키지를 import** 한다 (serve 파일 이름이 `lerobot.py` 라 패키지를 가릴 수 있다).
  `parse_payload` → `_apply_input_remap` → `GrootPolicyAdapter` preprocessor → `GrootPolicy.predict_action_chunk`.
  이미지는 무손실 PNG base64 로 넘긴다 (serve 가 decode). state 는 `modality.json` slice 를
  `GROOT_STATE_METADATA_KEYS` 역표로 payload key 에 대응시킨다. 카메라는 `GROOT_ENV_VIDEO_TO_UNIFIED_CAM`.
- 영상: episode 당 mp4 3개를 av 로 한 번씩 decode 한다. 프레임 수 = parquet 행 수, 프레임 시간이
  parquet timestamp 와 0.5/fps 안에 있어야 한다.
- feature: `FirstCallCapture` pre-hook 으로 첫 denoise call 의 decoder 입력만 잡아 last16 평균 (signature
  `ccbfb51ca01724cb`). decoder 호출 수 4 를 확인한다. 결정 시점 0,5,…<T.
- seed: 결정마다 `sha256(f"{base_seed}|{episode_id}|{t}")[:4] & 0x7fffffff` (base 42), serve
  `_apply_inference_seed` 로 적용한다. loader 가 다시 유도해 대조한다.
- 저장 (episode 별, immutable): `episodes/episode_<idx>__<sha16>.npz` (decision_steps, decision_seeds, latents,
  decoder_calls, expert_chunks (n,16,12), chunk_valid, raw_actions, rewards, done, frame_index) +
  `episodes/episode_<idx>.json` (commit marker: 계약 지문, npz sha·bytes, source 상대경로·sha (parquet, 영상 3개),
  task, split, reward audit, runtime). 모두 O_EXCL 로 만든다.
- resume: commit 된 episode 는 계약 지문·npz sha·episode_id·split 이 맞으면 건너뛴다. 하나라도 다르면 거부한다
  (덮어쓰지 않음). 계약 (`extraction_contract.json`) 에는 코드 sha·profile sha·image id·split sha 가 들어 있다.
  따라서 코드를 고친 뒤에는 새 출력 폴더가 필요하다.
- 102 개가 모두 commit 되면 `extraction_complete.json` 을 만든다.

### 11.2 reward·종료 규약

- reward = `next.reward` 원본 (T 행 전부), done = `next.done` 원본. 실측 패턴은 마지막 16 행 reward 1,
  그 앞 0, done 은 마지막 행만이다 (`OpenDrawer_reward_audit.json`). 추출과 loader 가 episode 마다 이 패턴을
  다시 확인하고, 다르면 거부한다.
- **offline dataset-end terminal 규약**: done 이 마지막 행이므로 `end_reason=terminal` 로 둔다.
  마지막 transition 의 discount 는 0 이다 (partial chunk 면 m_i<5). 이것은 데이터 끝 규약이다.
  실제 물리 env 가 그 시점에 종료했는지는 **모른다** (historical converter done_mode 미복원).
- Octo/V-GPS 의 last-3 성공 relabel 규약은 쓰지 않는다.

### 11.3 학습

- loader 검사: split sha = v2 + 독립 재계산·parquet sha (`validate_aux_split`), 추출 계약 (feature signature,
  checkpoint, split sha, H16/m5), episode 별 npz sha·split 일치 (누수 방지), seed 재유도, reward audit,
  expert target 재계산 일치, parquet 원본과 action/reward/done/frame_index byte 대조. 92/10 전부가 아니면
  거부한다 (`--allow-subset-smoke` 는 smoke 표시).
- data manifest 를 `published_package_subset_for_baseline` 으로 조립해 `load_finetune_data_manifest` 로 다시 검증한다.
- transition: γ=0.99, 시간순 다음 latent, 마지막 terminal discount 0 (§7). optimizer 는 `TrainSampler(train)` 만 쓴다.
- profile `production` (512×4, ensemble 10), batch 256, 500k step, seed 42.
- 사전 선언 val metric: `val/action_mse_exec_prefix` (전체 val set, 고정 val seed 42). 최소값 선택,
  동률은 이른 step. val 은 checkpoint 선택에만 쓴다 (gradient·hparam 탐색 없음).
- step 0, 50k 배수, 마지막 step 에서 평가하고 `checkpoints/step_<7자리>/` 에 저장한다 (tmp 폴더 → rename,
  기존 폴더면 거부). 파일별 sha256·bytes 를 기록한다. `selection/after_step_*.json` (평가마다 새 파일) 과
  `selection_final.json` 을 남긴다.
- `progress.jsonl` 은 log_every (기본 1000) 마다 flush+fsync 한다 (loss, steps/s, walltime; 평가 행에는 checkpoint
  bytes 와 예상 총 bytes). loss·val metric 이 비유한이면 즉시 멈추고 `FAILED.json` 을 남긴다 (exit 2).
- checkpoint 크기: 축소 망은 4.2MB 였다. production 은 실측 전이다. 파라미터 수로 보면 checkpoint 당
  수백 MB, 11 개 합계 수 GB 로 예상한다. 실제 값은 첫 평가 행 `checkpoint_bytes` 로 확인한다.

### 11.4 아직 없는 것

- 학습된 checkpoint 의 gate1 합성 rollout 과 small paired rollout 평가 CLI 는 이 폴더에 없다.
  §4·§5 `ComposedFlowSampler`·`PrefixQAMTrainer.velocity_fn` 과 `scripts/rl2_vla/PAIRED_EVAL.md` 를 써서 별도로 만들어야 한다.
- TRQAM 은 시나리오 답변 전까지 차단 상태다.


## Controller semantics correction (2026-10-21 saved-state diagnostic)

The executed PandaOmron arm commands are base-frame delta OSC inputs, not absolute end-effector poses. The 12 active dimensions also include gripper, base/torso and mode. Profile action_type=absolute and checkpoint modality absolute=true are retained source labels; they do not trigger pose conversion in this serving/collection path and must not be cited as proof of physical controller semantics. Expert targets are raw controller commands reordered and normalized; QAM composes velocity in the same normalized space; native inverse normalization and robot key mapping send delta commands unchanged to the controller. Evidence: outputs/analysis/groot_robocasa_lineage_20261021/expert_initial_state_diagnostic_v2/action_semantics_audit.json. No action conversion or retraining is warranted by this label discrepancy alone.
