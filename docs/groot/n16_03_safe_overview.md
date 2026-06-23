# GR00T N1.6 RoboCasa — SAFE Overview

이 문서 묶음은 GR00T N1.6 RoboCasa와 SAFE detector를 연결하는 canonical 위치다. 예전 단일 SAFE wiring 문서는 `n16_03`부터 `n16_12`까지의 번호 prefix 문서로 합쳐졌고(detector/visualization/report 3편은 `n16_07_safe_detector_report.md` 한 문서로 통합), 결론은 HTTP와 ZMQ를 모두 유지하되 현재 실험 기준선을 ZMQ로 두는 것이다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - **03 SAFE Overview (이 문서)**
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - [07 SAFE Detector + Visualization + Report](n16_07_safe_detector_report.md)
> - [09 SAFE Parity](n16_09_safe_parity.md)

## 현재 결론

- `ZMQ official eval`: GR00T N1.6 RoboCasa 성공률을 판단하는 기준 경로.
- `ZMQ SAFE feature server`: SAFE rollout 수집 기준 경로. official RoboCasa 클라이언트 환경을 쓰면서 action과 feature를 함께 저장한다.
- `HTTP /act` + `HTTP /act_with_features`: 프로젝트 공통 serving API로 유지한다 (port `:8500`, `scripts/serve/groot.py`). HTTP 경로는 일반 벤치마크가 같은 모델을 호출하거나 (`/act`) SAFE feature를 회수할 때 (`/act_with_features`) 사용한다. DiT pre-velocity feature 정의는 ZMQ feature server와 HTTP `/act_with_features` 가 `src/policies/groot/safe/features.py` 의 `capture_dit_features` 를 공유한다. Endpoint action parity, SAFE pkl/loader smoke, 짧은 closed-loop transport smoke까지 통과했다 ([09 SAFE Parity](n16_09_safe_parity.md#runtime-validation-2026-05-29), [07 SAFE Detector + Visualization + Report](n16_07_safe_detector_report.md#http-act_with_features-safe-collection-smoke-2026-05-29), [n16_11 변경 일지](n16_11_http_act_changes.md)).
- RoboCasa365 `target_atomic_seen18` 100ep/task collection은 완료됐다. `target_atomic_seen18_ckpt120000_robocasa365_100ep`는 `18 x 100 = 1800` episode triplet을 포함하고, seed range는 task별 `100000..100099`, verifier 기준 `status=ok`, total SR `967/1800 = 53.7%`다.

전체 task-set HTTP benchmark SR은 아직 산출하지 않았다. 현재 HTTP 검증 범위는 action parity, feature schema, transport smoke다.

## Checkpoint And Env

체크포인트:

- host: `/home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000`
- container: `/temporal_vla/outputs/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000`
- source: `Abhi03/grootn16_robocasa365_multitask_learning/checkpoint-120000`
- profile: `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml`

`groot__robocasa365_ckpt120000.yaml`은 사용자-facing 이름을 RoboCasa365 checkpoint 기준으로 둔다. 내부 `model_specific.embodiment_tag`는 checkpoint metadata에 맞춰 `NEW_EMBODIMENT`를 사용한다.

RoboCasa 환경:

- 기준 env: `/home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python`
- `robocasa_v02`: `/home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T/external_dependencies/robocasa`
- `robocasa365`: `/home/dongkyu/pdk_ws/temporal_vla/src/benchmarks/robocasa`

GR00T official eval과 기본 SAFE collection은 RoboCasa v0.2 (`robocasa_v02`) task name을 쓴다. RoboCasa365 수집은 `robocasa365` task name을 쓰므로 task mapping을 명시적으로 유지한다.

## Shared Run Config

SAFE N1.6 RoboCasa scripts는 per-script hardcoded path 대신 공통 run identity를 아래 두 파일에서 가져온다.

| file | consumer | role |
|---|---|---|
| `scripts/safe/groot_n16/robocasa/run_config.py` | Python scripts | `Path` 객체와 final detector selection의 canonical config |
| `scripts/safe/groot_n16/robocasa/run_config.sh` | Bash wrappers | shell 환경에서 쓰는 adapter config |

Python script는 `run_config.py`를 import하고, Bash wrapper는 `run_config.sh`를 source한다. 새 run/task set을 만들 때 개별 script 안의 output path를 직접 고치지 말고 아래 값을 override한다.

| variable | scope | meaning |
|---|---|---|
| `ROBOCASA_SAFE_RUN_ID` | Python/Bash shared run root | top-level SAFE run directory, default `safe_seen4_unseen2_100ep` |
| `ROBOCASA_SAFE_EXPERIMENT_ID` | feature visualization | visualization experiment directory name |
| `ROBOCASA_SAFE_FINAL_HORIZON_IDX_REL` | detector train/eval | final detector horizon aggregation |
| `ROBOCASA_SAFE_FINAL_DIFF_IDX_REL` | detector train/eval | final detector diffusion aggregation |
| `ROBOCASA_SAFE_HPARAM_SWEEP_ID` | detector train/eval | hparam sweep experiment directory |
| `RUN_ID` | collection wrappers only | per-collection output suffix under `experiments/collection_smoke/rollouts_${RUN_ID}` |

`ROBOCASA_SAFE_RUN_ID`는 전체 SAFE run tree를 바꾸고, `RUN_ID`는 한 번의 rollout collection 산출물 이름만 바꾼다.

현재 default run config 확인:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

python -c "import sys; sys.path.insert(0, 'scripts/safe/groot_n16/robocasa'); import run_config as c; print(c.RUN_ROOT); print(c.SPLIT_ROOT); print(c.FINAL_LSTM_RUN_DIR); print(c.FINAL_DETECTOR_DIR)"
```

## Script Pipeline Order

이 디렉터리의 파일은 아래 순서로 읽으면 된다. 각 step은 앞 step의 artifact를 입력으로 삼는다.

Layer boundary:

- `src/policies/groot/`: GR00T policy adapter library shared by serving, eval, and SAFE workflows. It must not import from `scripts/`; SAFE workflow pkl/CSV collection contracts do not live here.
- `src/policies/groot/robocasa/io.py`: GR00T RoboCasa IO adapter for `GrootRoboCasaEnv` native keys. GR00T HTTP eval and SAFE wiring use this instead of the generic benchmark processor pipeline.
- `src/processor/`: generic benchmark processor pipeline for Project FastAPI evaluation. It remains the path for non-GR00T-native RoboCasa/Calvin/LIBERO eval scripts, not the home for GR00T native key conversion.
- `scripts/`: executable workflow entrypoints and runtime wiring. Scripts may import `src/policies/groot/`.
- `scripts/safe/groot_n16/robocasa/collect/`: RoboCasa SAFE collection orchestration, transport clients, and collector pkl/CSV schema helpers.

| order | file | purpose | main input -> output |
|---:|---|---|---|
| 0 | `run_config.py` | Python-side shared run identity, paths, final detector selection | constants -> Python defaults |
| 0 | `run_config.sh` | Bash-side shared run identity adapter | env/defaults -> shell variables |
| 0 | `safe_feature_vectors.py` | `[K,H,D]` Flow-matching SAFE feature를 timestep-level SAFE feature vector로 aggregate | rollout pkl + aggregation command -> `[T,D]` features |
| 1 | `serve/feature_server.py` | ZMQ feature server exposing `get_action_with_features` | GR00T checkpoint -> action + unpooled feature |
| 2 | `collect/collect_rollout.py` | one-task/one-range rollout collector entrypoint | feature client + RoboCasa env -> SAFE-readable pkl/mp4/csv |
| 2 | `collect/collect_env.py` | RoboCasa env construction, video wrapper, one-episode rollout loop | env name + scenario replay -> rollout result |
| 2 | `collect/collect_artifacts.py` | SAFE rollout artifact writer | feature records + ep_meta + video -> pkl/mp4/csv |
| 2 | `collect/collect_policy_clients.py` | ZMQ/HTTP feature policy client transports | ZMQ `get_action_with_features` or HTTP `/act_with_features` -> action + feature records |
| 2 | `collect/collect_schema.py` | SAFE collection schema helpers derived from `src/policies/groot/core/schema.py` | observation/action/features -> collector pkl schema |
| 2 | `collect/collect_task_set_official_uv_host.sh` | preferred host-side task-set collection wrapper using official `robocasa_uv` env | task set + seeds -> raw rollout directories |
| 2 | `collect/collect_task_set_in_container.sh` | collection wrapper for already-running container shell context | task set + seeds -> raw rollout directories |
| 2 | `collect/collect_task_set_via_docker_exec.sh` | host wrapper that enters the Docker container for collection | task set + seeds -> raw rollout directories |
| 3 | `split/prepare_seen4_unseen2_split.py` | paper-faithful SAFE split construction | raw rollouts -> `train` / `val_seen` / `val_unseen` split tree |
| 4 | `train/train_lstm_mean_mean.sh` | legacy mean/mean SAFE-LSTM baseline | split tree -> baseline train logs |
| 4 | `train/train_lstm_aggregation_ablation.sh` | aggregation ablation over horizon/diffusion axes | split tree -> aggregation train logs |
| 5 | `analyze/summarize_lstm_aggregation_ablation.py` | select candidate aggregation from ablation logs | aggregation train logs -> json/md summary |
| 6 | `train/train_lstm_hparam_sweep.sh` | hparam sweep with selected final aggregation defaults | split tree -> hparam train logs |
| 7 | `analyze/summarize_lstm_hparam_sweep.py` | summarize hparam sweep and selection rule | hparam train logs -> json/md summary |
| 8 | `analyze/finalize_lstm_detector.py` | pin final checkpoint and generate fixed/CP/functional-CP artifacts | selected run dir -> `final_detector/` |
| 9 | `vis/plot_safe_conformal_curves.py` | SAFE Figure-8-style CP operating curves | `final_detector/*.csv` -> CP figures |
| 10 | `vis/run_feature_visualization.py` | adapter around SAFE's original feature visualizer | split tree -> SAFE-style t-SNE/UMAP artifacts |
| 10 | `vis/plot_safe_style_feature_space.py` | native SAFE-style feature-space plotter | split tree -> feature-space plots |
| 11 | `vis/plot_task_success_overlay.py` | overlay task/success labels on projected feature artifacts | feature visualization pkl -> overlay images |
| 12 | `vis/compute_feature_silhouette.py` | static feature-space separability diagnostics | split tree / projection pkl -> silhouette tables |
| 13 | `analyze/diagnose_rollout_mean_feature_separability.py` | rollout-mean aggregation separability diagnostic | split tree -> rollout-level separability tables/plots |

Default visualization and silhouette scripts now use the final detector aggregation from `run_config.py`. To reproduce the early `mean/mean` artifacts, pass `--horizon-idx-rel mean --diff-idx-rel mean` explicitly.

## Files

ZMQ SAFE:

- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/safe_feature_vectors.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_policy_clients.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_schema.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_task_set_in_container.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_task_set_official_uv_host.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_mean_mean.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/diagnose_rollout_mean_feature_separability.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_safe_style_feature_space.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py`

Final detector artifacts:

- `/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/README.md`
- `/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/model_final.ckpt`
- `/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/final_operating_point.json`

HTTP:

- `/home/dongkyu/pdk_ws/temporal_vla/scripts/serve/groot.py` (port 8500, `/act` + `/act_with_features`)
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/utils/vla_client.py` (`predict`, `predict_with_features`)
- `/home/dongkyu/pdk_ws/temporal_vla/src/policies/groot/safe/features.py` (HTTP/ZMQ 공유 DiT capture)
- `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml`
- `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml`

Validation utilities:

- `/home/dongkyu/pdk_ws/temporal_vla/docs/benchmarks/robocasa_task_name_mapping.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/adr/0001-dedicated-safe-groot-n16-zmq-server.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/groot/n16_03_safe_overview.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/groot/n16_04_safe_collection.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/groot/n16_05_safe_env_reproduction.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/groot/n16_06_safe_inference_semantics.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/groot/n16_07_safe_detector_report.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/groot/n16_09_safe_parity.md`

## 남은 항목

1. HTTP benchmark SR: 전체 task-set closed-loop 평가는 아직 남아 있다. 현재 HTTP/ZMQ SAFE transport smoke는 wiring과 schema 검증 범위다.
2. Proactive intervention: inference-step-level failure onset/intervention label protocol이 아직 없다.
3. Paired trace identity: `ep_meta`만으로는 부족하며 reset-time full sim-state replay (`qpos/qvel` 등)가 필요하다.
4. Feature axis ablation: `--feature-slice all` (`H=50`)과 current valid-horizon (`H=16`) export 비교가 남아 있다.
