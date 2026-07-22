# GR00T Data Catalog (rollout / activation)

GR00T 계열(N1.5·N1.6) rollout·activation 데이터가 어느 경로에 무엇이 있는지 정리한 지도.

## 두 머신

| 머신 | 데이터 루트 |
|---|---|
| 로컬 (이 PC) | `outputs/eval/robocasa/{groot_n15,groot_n16}/` |
| 원격 (승준) | `kimseungjun@166.104.146.37:11112` → `~/workspace/temporal_vla/outputs/eval/robocasa/{groot_n15,groot_n16}/` |

아래 표의 `경로`는 `outputs/eval/robocasa/` 뒤의 상대경로다.

## N1.5 (`groot_n15/`) — instruction-fixed · phase 계열

| 경로 (`groot_n15/…`) | 위치 | 크기 | 어떤 데이터 |
|---|---|---|---|
| `target_instruction_fixed15_pathway_50ep` | 원격 | 7.0G / 467 pkl | N1.5 주력 셋. DiT action-token feature(`[K=4,16,1024]`) + VL meanpool(D=2048), per-step. instruction 고정 6~10 task |
| `target_instruction_fixed15_block_residual_50ep` | 원격 | 26G / 750 pkl | per-block residual(`[L=7,49,1536]`), 11 task |
| `phase_event_aligned_4cell` | 로컬 · 원격 | 로컬 2.9G / 96 pkl | phase-event 정렬 + succ/fail split. PPCC(bread)·PPCS 2 task. phase 분포·실패영상 주석 포함 |

## N1.6 (`groot_n16/`) — atomic · SAFE 계열

| 경로 (`groot_n16/…`) | 위치 | 크기 | 어떤 데이터 |
|---|---|---|---|
| `target_atomic_moderate10_multilayer_perT_100ep` | 원격 | 154G / 1000 pkl | 32-layer × per-timestep residual (최대 용량) |
| `target_atomic_moderate10_pathway_pertoken_100ep` | 원격 | 41G / 1000 pkl | 7-layer DiT + VL, per-token pathway |
| `target_atomic_seen18_ckpt120000_robocasa365_100ep` | 원격 | 16G | seen18 finetune(ckpt120000) rollout + feature (tar.zst 동봉) |
| `target_atomic_seen18_ckpt120000_..._ah8_100ep` | 원격 | 8.2G | 위의 action-horizon 8 변형 |
| `target_atomic_moderate10_multilayer_100ep` | 원격 | 5.5G | multilayer(perT 아닌 집계형) |
| `target_atomic_seen18_multilayer_15ep` | 원격 | 1.6G | seen18 multilayer 소량 |
| `rollouts_n16_seen5_20ep_upstream_video` | 원격 | 2.7G | upstream parity video rollout |
| `safe_split_seen4_unseen2_openDrawer_pnpCab_100ep` | 로컬 · 원격 | 2.7G | SAFE detector용 seen4/unseen2 split |
| `safe_seen18_4unseen_100ep` | 원격 | 1.7G | SAFE seen18+4unseen feature |
| `safe_split_seen18_4unseen_100ep` | 로컬 · 원격 | 22M | SAFE split 메타 |
| `conceptor_steering` · `safe_feature_vis` · `safe_train_logs` · `_ep_metas` | 로컬 · 원격 | 123M · 115M · 21M · 16K | 산출물(conceptor·시각화·로그·ep_meta) |

## 동료(kanu) COAST N1.5 — 원격 `~/datasets/kanu_archive/coast_n15/`

| 경로 (`kanu_archive/coast_n15/…`) | 크기 | 어떤 데이터 |
|---|---|---|
| `coast_faithful_7task_30ep_replan5` | 4.3G | 7 task raw_rollouts + ep_meta |
| `coast_exp1_steer_replan16` | 1.6G | baseline / steered + results.tsv |
| `coast_faithful_7task_30ep` | 882M | ep_meta · classified · conceptor_taskfit |
