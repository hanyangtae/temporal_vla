# exp4-3 핸드오프 — 분리도 지도(Separation Atlas) cross-model

작성 2026-07-24. 다음 세션이 이 문서 + 계획서(`~/.claude/plans/exp-4-3-luminous-truffle.md`)만 읽으면
자기완결이 되도록 씀. **exp4-1이 메인 라인이므로 그쪽 자원·트리를 방해하지 말 것(§6 격리 규칙 필수).**

---

## 0. 한 줄 요약 / 지금 어디까지

exp4-3 = "succ/fail 활성 **분리도**가 모델 × layer × phase 로 어떻게 다른가"의 기술 지도.
벤치=RoboCasa. 모델 = **N1.5(완료) → N1.6(수집 배선 완료·미검증) → π0.5(미착수) → Cosmos(미착수)**.

- **N1.5 완료**: atlas 코어·KL 통합진단·whitened 프로브 전부 산출. 결과 §2.
- **N1.6 진행 중**: 멀티레이어 활성 수집 인프라는 있으나 phase 라벨이 없었음 → `collect_rollout.py`에
  phase 라벨링 배선 **완료(커밋 d182e54… 최신 exp4-3 브랜치)**. **아직 smoke 미실행** — 다음
  세션 첫 작업. §3.
- **π0.5 / Cosmos**: 미착수. π0.5는 collector 자체가 없어 신규 작성 필요(SR>0 게이트 먼저). §4.

---

## 1. 브랜치·트리·자원 (반드시 지킬 것)

- **exp4-3 전용 worktree**: 로컬 `/home/dongkyu/pkt_ws/temporal_vla/.claude/worktrees/exp4-3-atlas`
  (브랜치 `exp/exp4-3-separation-atlas`). **여기서만 작업**. 본 트리(`~/pkt_ws/temporal_vla`)는
  exp4-1 전용이니 checkout/switch/편집 금지.
  - ⚠️ 사고 이력: N1.6 편집을 실수로 본 트리(exp4-1)에 했다가 worktree로 이관·본 트리 복원함.
    앞으로 파일 경로는 반드시 worktree 절대경로로.
- **승준 노드 worktree**: `kimseungjun@166.104.146.37:11112` → `~/workspace/exp4_3`
  (`~/workspace/temporal_vla`는 exp4-1 전용, 건드리지 말 것). 스크립트 배치는
  `rsync … kimseungjun@…:workspace/exp4_3/scripts/safe/exp4_3/`.
- **데이터 HDD**: 승준 `~/datasets/temporal_vla_outputs/…`(1.8T). **수집물은 episode마다 즉시
  승준 직송 + 로컬 삭제** (kanu 루트 디스크 여유 ~30GB뿐, full 모드 pkl 722MB/ep).
- **GPU 상한(전 exp 합산)**: kanu 최대 GPU **3개**·GPU당 serve **2개**; srv48/srv50 각 GPU **1개**·serve **6**.
  **완전히 빈 GPU만**(타인 프로세스 있으면 금지). 발사 전 `nvidia-smi --query-compute-apps`로 소유자 확인.
  현재 빈 GPU=4·5(가변). **포트는 8640~8659만**.
- **컨테이너 GPU 끊김 함정**: lerobot/robocasa/groot 컨테이너가 종종 NVML 초기화 실패(CPU 폴백,
  "FlashAttention2 not available on CPU"). 복구=`docker compose restart <container>`(유휴 확인 후).
- subagent 모델은 opus 이하만(fable 금지), 호출마다 model 명시.

---

## 2. N1.5 결과 (완료 — 산출물이 진실의 기준)

산출물: worktree `outputs/eval/robocasa/groot_n15/exp4_3/`
- `atlas/n15/<cell>.json` ×5 + `atlas/atlas_all.tsv`(210행) — model×cell×layer×phase 지표
- `figs/atlas_global.png`(5패널: mean_z·var_z·kl_z·mean_frac·quota), `figs/atlas_phase_{mean,var}_z.png`, `figs/atlas_quota.png`
- `probe_whitened/<cell>.json` ×5 — raw/lda/qda AUROC (24d 연결)

**셀 5개**: pq3_ppcc_bread/beer, pq3_drawer_left/right, exp41_mixer (exp4-1 fit30 데이터, beer는
오염 3판 제외 `task_PPCC_fit_beerclean`).

### 2.1 세 갈래 지표 (global, 선정 layer)
| cell | 평균분리 mean_z (peak) | 분산분리 var_z | 통합 KL kl_z (성분 mean_frac) |
|---|---|---|---|
| drawer_right | L8 +5.6 | 전 layer <2 (퇴화) | **L15 +12.0** (0.49) |
| drawer_left | L10 +5.6 | 퇴화 | L10 +4.6 (0.39) |
| beer | L10 +5.7 | L4 +3.2(단발) | L12 +4.2 (0.67) |
| bread | L10 +5.1 | 퇴화 | L10 +3.9 (0.53) |
| mixer | L15 +2.6(약) | 중간층 −3~−4 | L10 +6.2 (0.38) |

**판독**: ① 평균분리는 실재·중간층(L8~L12) peak (4/5 cell). ② conceptor형 분산분리는 전 cell
퇴화(exp4-1 재확인). ③ COAST quota 곡선은 **중간층 peak 없음** — drawer/mixer 단조감소, ppcc는
L2~L4 얕은 peak (COAST가 π0.5+LIBERO에서 본 L11 unimodal peak와 **모양 다름** = 조건 의존성 첫 실증).

### 2.2 통합 KL 진단 (kl_decomp.py) — 사용자 제안 반영
가우시안 KL을 **평균 성분 + 분산 성분**으로 정확 분해(둘 다 ≥0):
`2·KL = (μs−μf)ᵀΣs⁻¹(μs−μf)[mean] + [tr(Σs⁻¹Σf)−k+ln det비][cov]`.
축소공간 처리(사용자 동의): **라벨-무관 pooled PCA top-32** + Ledoit-Wolf shrinkage,
train에서 기저·모수 추정→held-out 평가. 각 항에 순열 null z + `mean_frac`(=mean/total).
검증: 합성 평균-only mean_frac 0.94 / 분산-only 0.05.

### 2.3 whitened 프로브 (probe_whitened.py) — 24d 문서 검증, ★결과가 24d와 어긋남
`docs/steering/24d_…`는 "whitening(LDA)이 raw 이기고 공분산(비선형)은 무용"이라 주장. 전-layer 재측정:
- **whitening(lda−raw)이 24d 재현 안 됨**: 24d는 drawer_R L4에서 raw0.752→LDA0.921(+0.169) 주장,
  우리 측정 L4 raw0.733→lda0.669(**−0.064**). whitening은 **late L15에서만 +0.09~0.16**, 중간층은 손해.
  → **원인 후보 = 부분공간/shrinkage 차이(24d는 PCA-50+λ0.1, 우리는 top-32+Ledoit-Wolf).**
  24d §2.1이 요구한 **k·λ 민감도 sweep 미실행** — whitening 결론은 이 sweep 전까지 보류.
- **QDA >> LDA (강한 cell)**: drawer/ppcc 전 layer에서 qda−lda **+0.10~0.25**, 순열 null 통과(qda_z 4~6).
  24d의 "공분산 무용"과 반대 — svm/mlp 대신 QDA(공분산 직접 모델)로 재면 공분산이 판별에 실재로 기여.
  단 mixer는 QDA도 못 살림(전 AUROC<0.7, qda_gain 최대 0.11).
- **★ "mixer=분산형" 판정 철회**: KL에서 mixer kl_cov_z=5.9로 커서 "분산형"이라 했으나, whitened
  프로브에서 mixer는 raw/lda/qda 전부 AUROC<0.7로 **그냥 신호가 약함**. kl_cov_z 강신호는
  판별에 안 쓰이는 공분산 divergence(scene/phase 잡음)였을 개연성. → mixer는 "약신호"로 등급.
- **함정(24d §3)**: 전부 read-out. succ/fail이 애초에 다른 scene이라 판별력이 "실패 인과"인지
  "scene 겉모습"인지 못 가름. 인과는 write(개입 SR)만 답함.

---

## 3. N1.6 (수집 배선 완료·미검증) — 다음 세션 첫 작업

### 3.1 확인된 사실
- **활성 인프라 존재**: serve `scripts/safe/groot_n16/robocasa/serve/feature_server.py`(ZMQ,
  `--capture-token-mode full`=[L=32,T=51,D=1536] per record, K는 mean됨), collect
  `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`, ckpt `groot__robocasa365_ckpt120000.yaml`.
  레퍼런스 러너 `scripts/safe/groot_n16/robocasa/steer/collect_multilayer_parallel.sh`(ZMQ ping·GPU 병렬).
- **기존 N1.6 데이터는 재사용 불가**: 승준 `~/datasets/…/groot_n16/`의 멀티레이어 collection
  (`multilayer_perT`·`pathway_pertoken`)은 계약은 맞으나 **feature_phases 빈값·seed 없음** → phase축 불가.
- **DiT 32블록**(N1.5는 ~16). 캡처 layer는 exp4-1처럼 `0,2,4,8,10,12,15` 쓰면 valid하나 **상대깊이가
  N1.5와 다름**(해석 주의) — 또는 전 32층 캡처가 더 정보적.
- **full 모드 pkl = 722MB/ep** (32×51×1536×144record). 사용자 지시: **토큰 pool 하지 말 것**(나중에
  per-token 분석). → episode마다 즉시 승준 직송+로컬 삭제 필수. (all/valid 모드는 토큰 pool이라 금지.)
- pkl에 `action_vectors`(144,12)·`scenario_seed` 존재 → 필요시 phase retrofit도 가능(현재는 불필요).

### 3.2 이번 세션에서 한 것 = phase 라벨링 배선 (커밋됨, exp4-3 브랜치)
`collect_env.run_single_rollout`에 `label_phases`/`proximity_phases` 인자 추가:
get_action마다 `make_robocasa_event_labeler(env.envs[0], env_name, proximity_phases).step()` 호출 →
per-record `feature_phases` 수집(N1.5 규약: 1 record=1 get_action, env.step 전 현재 state).
반환 튜플 끝에 feature_phases 추가·record 수 불일치 raise. `collect_rollout.py`에 `--label-phases
--proximity-phases` CLI → `write_safe_triplet(extra_metadata={"feature_phases": …})`.
**미검증** — smoke가 다음 첫 단계.

### 3.3 다음 세션 N1.6 실행 순서 (복붙 가능 수준)
```
# ⓐ GPU 확인 (빈 것만). 컨테이너 GPU 끊김 시 docker compose restart groot
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

# ⓑ serve (groot 컨테이너, GPU=빈것, PORT 8640, full 모드=토큰보존)
docker exec -d -e CUDA_VISIBLE_DEVICES=<GPU> -e NO_ALBUMENTATIONS_UPDATE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True groot bash -lc \
  "cd /temporal_vla && python scripts/safe/groot_n16/robocasa/serve/feature_server.py \
   --profile /temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
   --host '*' --port 8640 --device cuda --feature-slice valid \
   --capture-token-mode full --capture-vl"
# ready = ZMQ ping (collect_multilayer_parallel.sh:39-47 의 ping_server 함수 그대로)

# ⓒ smoke: OpenDrawer 1ep + --label-phases. PYTHONPATH 에 groot_n16 collect 경로 필수
docker exec -e MUJOCO_GL=egl -e ROBOCASA_ENV_SOURCE=robocasa365 \
  -e PYTHONPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla:/temporal_vla/scripts/safe/groot_n16/robocasa/collect" \
  robocasa python /temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py \
    --policy-client-host 127.0.0.1 --policy-client-port 8640 \
    --feature-endpoint get_action_with_multilayer_features \
    --env-name "robocasa_panda_omron/OpenDrawer_PandaOmron_Env" --robocasa-env-source robocasa365 \
    --output-dir <OUT>/OpenDrawer --task-id 8 --episode-start-idx 0 --n-episodes 1 --seed 100000 \
    --n_action_steps 5 --label-phases --proximity-phases

# ⓓ smoke 검증: pkl 열어 feature_phases n==n_rec(144?), record0=(32,51,1536),
#    atlas_loader.load_one() 통과(현재 [L,T,D] 계약은 로더 _load_slot이 못 읽음 → §3.4 주의)
```

### 3.4 ★ 미해결 코드 이슈 (다음 세션이 반드시 처리)
- **atlas_loader가 full 모드 [L=32,T=51,D=1536] per-record를 못 읽음**. 로더는 record ndim으로
  디스패치: ndim=3을 [L,K,D]로 해석. full 모드 record는 ndim=3 (L,T,D) → `_load_slot`(ndim=2용)이
  아니라 `_load_pooled`로 가야 하고, T를 K로 취급해 mean하면 [L,D](토큰평균)가 됨 = atlas OK.
  **하지만 로더가 phase 없으면 raise** — smoke pkl에 phase가 실제로 붙는지부터 확인. 그리고
  full 모드는 record가 (32,51,1536) ndim=3이라 `_load_pooled` 경로 → dit=mean(axis=1=T) → [L,D].
  **사용자가 "토큰 pool 하지 말라" 했으므로 raw pkl은 T보존으로 저장하되, atlas 분석 시점에만
  T-mean** (로더가 이미 그렇게 함). per-token 분석은 dit_k에 T가 남아 별도 가능. → 로더 재확인 필요.
- 즉 다음 세션: smoke pkl → `atlas_loader.load_one()` 실제로 태워보고 shape/phase 정합 확인 후 본수집.

### 3.5 N1.6 본수집 계획
- 셀 5종(우리 것과 겹치는 N1.6 task): OpenDrawer(좌우 구분은 cell-index로), PickPlaceCounterToCabinet,
  OpenStandMixerHead. 30ep/cell, seed 100000+ep, `--label-phases --proximity-phases`, full 모드.
- episode마다 승준 직송+로컬 삭제(722MB×150 = 105GB, 로컬 불가). 러너는
  `mixer_c0_collect.sh`의 ship_ep 패턴(rsync+검증+삭제) 참고.
- 수집 후 승준에서 `atlas_sweep.py --model n16` + `probe_whitened.py --model n16`.

---

## 4. π0.5 / Cosmos (미착수)

### π0.5 — collector 신규 작성 필요 (가장 큰 갭)
- serve는 됨: `lerobot.py --profile configs/checkpoints/lerobot_pi05__robocasa365_75000.yaml
  --collect --pi05-expert-capture-layers 0,5,11,17` (Gemma2 expert 18층 d=1024 K=10, [L,K,D]).
- **http_feature_collect.py 재사용 불가**: π0.5는 state 키가 다름(`base_to_eef_pos/quat` 필요,
  GR00T env 미노출). generic `RoboCasaObsProcessor(three_cameras=True)` 필요.
- **SR=0 gripper 버그 위험**: `.claude/agent-memory/vla-checkpoint-manager/pi05_robocasa_quirks.md` 필독.
  `--use-groot-env` action 경로가 안전하나 "미검증".
- **순서**: ① `scripts/eval/robocasa_eval.py --use-groot-env --three-cameras`로 **SR>0 먼저 확인**
  (통과 못하면 π0.5 보류) → ② 통과 시 collector 신작(obs generic + write_safe_triplet + 라벨러 +
  --label-phases 로직 이식) → ③ smoke → 수집.

### Cosmos Policy — 실행가능성 스파이크
- 채택 후보 = `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`(28블록 DiT, 추론 VRAM ~9GB, A4000 가능).
  14B·Cosmos3 RoboCasa policy는 미공개.
- 게이트: G-Cos-a(설치+1 rollout SR재현) → G-Cos-b(비디오·action 통합 latent에서 **action-슬롯
  인덱싱** 확정, WA-LQR 코드 선례) → 캡처 hook → phase는 env_step GT 소급 → 수집.
- 디스크 27GB 여유 문제로 승준 데이터 디스크(688GB) 설치 검토. 100GB+ GPU **불필요**(추론 9GB).

---

## 5. 열린 과학 질문 (우선순위)

1. **24d whitening 재현**: k·λ 민감도 sweep(24d §2.1). whitening 이득이 부분공간/shrinkage에
   민감하므로, top-k∈{16,32,50,전체}·λ∈{0.05,0.1,LW} 격자에서 lda−raw 안정성 확인. 이거 없이
   "whitening이 답"이라 말 못 함. (probe_whitened.py에 --k 인자 이미 있음.)
2. **QDA vs scene confound**: QDA 이득(공분산 판별)이 실패 인과인지 scene 겉모습인지 — read로
   불가. exp4-1 노이즈-재추첨 활성 재수집(같은 scene succ/fail)이 유일한 통제. 별건.
3. **cross-model**: N1.6·π0.5·Cosmos에서 세 지표(mean_z/kl_z/quota) layer 곡선이 N1.5와 어떻게
   다른가 = exp4-3 헤드라인. COAST의 π0.5 중간층 peak가 우리 π0.5에서 나오는지가 핵심 대조.
4. **처방(별건, exp4-1 라인 조율)**: mean_frac 지도가 셀·layer마다 setM(평균)/공분산 어느 쪽이
   필요한지 가리킴. whitened-setM(방향만 Σ⁻¹δ로 교체, rank-1 유지)이 24d 첫 처방. write-eval은 exp4-1 규약.

---

## 6. 산출물·파일 인덱스

worktree `scripts/safe/exp4_3/`:
- `atlas_loader.py` — 캡처계약 4종→공통 rolls (ndim 디스패치)
- `atlas_sweep.py` — (model,cell) layer×phase 지표(var_z·mean_z·kl_z·mean_frac·quota)
- `kl_decomp.py` — KL 분해(top-32·LW shrinkage)
- `probe_whitened.py` — raw/lda/qda AUROC (--k 인자)
- `atlas_emit.py` / `atlas_heatmap.py` (한글폰트 `assets/NanumGothic-Regular.ttf` 동봉)
- `run_atlas_remote.sh` / `run_probe_remote.sh` — 승준 러너

N1.6 배선(worktree): `scripts/safe/groot_n16/robocasa/collect/collect_env.py`·`collect_rollout.py`
(`--label-phases`).

exp4-3 브랜치 최신 커밋: N1.6 phase 배선. `git -C .claude/worktrees/exp4-3-atlas log --oneline`로 확인.

---

## 7. 새 세션 시작 프롬프트 (복붙용)

```
exp4-3(분리도 지도 cross-model) 이어서 한다. 핸드오프: worktree
.claude/worktrees/exp4-3-atlas/docs/steering/25_exp4-3_separation_atlas_handoff.md 정독.
반드시 exp4-3 worktree(.claude/worktrees/exp4-3-atlas, 브랜치 exp/exp4-3-separation-atlas)에서만
작업하고 본 트리(exp4-1)·승준 ~/workspace/temporal_vla 는 건드리지 마라. GPU는 빈 것만(전 exp 합산
kanu 3개·GPU당 serve 2), 포트 8640-8659, 수집물은 episode마다 승준 직송+로컬 삭제.
순서: ① N1.6 phase 배선 smoke(핸드오프 §3.3, OpenDrawer 1ep --label-phases) → pkl feature_phases·
atlas_loader 정합 확인(§3.4) → ② N1.6 5셀 30ep 본수집(full 모드, 토큰 pool 금지) → atlas_sweep+
probe_whitened(--model n16) → ③ 24d whitening k·λ 민감도 sweep(§5-1) → ④ π0.5 SR>0 게이트(§4).
문제·불확실 시 중단하고 보고.
```
