# RL2 IID cluster reward 실험

## 2026-09-16 실행 변경 (이 절이 아래 초기 예시보다 우선)

- 사용자 지시로 kanu에서 GPU당 π0 하나씩 collection lane을 병렬 실행한다. GR00T 전용 슬롯 규칙은 적용하지 않는다. GPU 소유/lease와 실제 simulator device smoke를 먼저 확인한다.
- 수집은 `--stage2_save_activations True`: QAM context 외에 전체 `[chunk, candidate, denoise, token, feature]` activation을 lossless NPZ sidecar로 저장한다. 첫 redundant token도 보존하여 원본 SAFE loader와 같은 방식으로 제거할 수 있다. episode JSON은 sidecar 완료 후 마지막으로 atomic publish한다.
- `archive_shipper.py`는 완료 JSON을 감시하여 sidecar → JSON 순서로 승준에 전송하고 각각 크기/SHA-256을 검증한다. 작은 전송 영수증을 fsync한 뒤 동일한 로컬 파일만 삭제한다. 오류이면 로컬을 보존하고 재시도한다. 수집 중 파일·checkpoint·다른 실험은 삭제하지 않는다.
- archive root: `/home/kimseungjun/datasets/rl2_iid_cluster_reward_20260916`; 이 아래 local repo 상대 경로가 유지된다. local staging: `outputs/rl2_iid_20260916/rollouts`.
- 여유 디스크가 4 GiB 미만이면 다음 episode 시작을 대기한다. 전송 완료 영수증으로 lane 완료를 확인하므로 전송 후 삭제를 미수집으로 오인하지 않는다.
- **SAFE 고정 설정/통합 CP alpha .2의 아래 학습·평가 예시는 최신 승인 설정이 아니다. 현재 `train_safe.py`/eval 기본값도 아직 초기 고정 설정이므로 그대로 본 실험에 실행하지 않는다.** 사용자 지시에 따라 원본 SAFE feature/lr/regularization/epoch sweep → ROC-AUC 기반 선택 → IID task별 CP 보정 → balanced-accuracy alpha 선택 절차를 연결한 뒤 학습한다. 평가 holdout은 선택에 사용하지 않는다. 원본 스크립트: `submit_opi0_simpler.bash`, `train_optimal_opi0_simpler.bash`, `alpha_heuristic_opi0_simpler.bash`. 전체 activation 저장은 이 탐색을 재수집 없이 가능하게 한다.

```bash
REMOTE_REPO=/home/kimseungjun/datasets/rl2_iid_cluster_reward_20260916 \
  .venvs/rl2_iid/bin/python scripts/rl2_vla/stage2_cluster_reward/archive_shipper.py \
  --root outputs/rl2_iid_20260916/rollouts --interval 5

# 실제 소유권/빈 GPU 확인 후 --run; 초기 smoke는 별도 root와 reset 9000 사용.
.venvs/rl2_iid/bin/python scripts/rl2_vla/stage2_cluster_reward/run_experiment.py \
  --mode collect --rl2-root "$PWD/RL2-VLA" --python "$PWD/.venvs/rl2_iid/bin/python" \
  --gpus 2 3 4 5 6 7 --policy "$PWD/outputs/rl2_assets/pi0_bridge" \
  --output "$PWD/outputs/rl2_iid_20260916/rollouts" --seeds 42 0 7 --trials 100
```

## 목적·실행 계약

π0 동결, RL2-QAM flow composition, 우리가 학습한 SAFE만 사용한다. IID 네 task는 eggplant/spoon/stack cube/carrot이다. N=1, 원문 instruction, verifier 미호출, SAFE chunk별 gate, 혼합 비율 0.5, CP alpha 0.2다. phase 입력이나 OOD로 확장하지 않는다.

기본 reward는 RL2 학습 Octo `0af7423a9922e8b6be004257b954cc1454eb2deb`의 `dataset.py:447` 규칙이다: 성공 마지막 3 control step의 reward/mask=0/0, 이전=-1/1. 성공 rollout은 최초 성공까지 사용한다. 실패 전 구간은 -1, 정상 benchmark 종료의 bootstrap은 0. 수집 중단·손상은 실패로 relabel하지 않는다. 원본 성공 demonstration 전용 규칙에 실패 처리를 추가한 것이며, 성공 한 번 +1 규칙이 아니다.

K48, PCA16, task별 train fit, episode 균등 가중 표준화/PCA/KMeans를 사용한다. cluster 점수는 성공/실패 episode 평균 점유율의 `(s-f)/(s+f+1e-12)`에 독립 rollout 방문수 `n/(n+5)`를 곱한다. 한 클래스만 있으면 potential=0, 표본이 K/PCA 차원보다 적으면 중단한다. cluster는 상태의 인과적 좋고 나쁨 라벨이 아니다.

`R = R_base + 0.1*(0.99**actual_steps * bootstrap * Phi(next) - Phi(current))`.
기본 arm과 shaped arm은 split, 초기 actor, critic 초기화, minibatch 순서, seed, 학습량이 같다. 마지막 부분 chunk의 실제 prefix만 BC와 critic action 입력에 사용하고 terminal에서 bootstrap하지 않는다.

## 준비

실행 환경은 기존 RL2 Python 3.10 / Torch 2.5.1+cu121 / flax 0.10.5 환경이다. 원본 설치 지침은 `RL2-VLA/RL2_CoVer_VLA/env_simpler_pi.sh`에 있다. CPU 수치 smoke 환경은 별도로 검증했으며 GPU/SIMPLER 실행 검증을 대체하지 않는다. π0 checkpoint만 약 6.1GB이므로 저장 공간을 먼저 확보한다. GR00T GPU당 6모델 규칙을 적용하지 않는다.

```bash
# repo root
export STAGE2="$PWD/scripts/rl2_vla/stage2_cluster_reward"
export RL2_ROOT="$PWD/RL2-VLA"
export EXP_ROOT=/path/to/adequate/storage/rl2_iid_cluster_reward
export RL2_PY=/path/to/rl2/bin/python
export QAM_INIT=/path/to/rl2_vla_qam_bridge_500k.pkl
# QAM_INIT 옆 flags.json이 필수다.

git submodule update --init RL2-VLA
git -C RL2-VLA submodule update --init third_party/qam third_party/SAFE
# 새 checkout에서만 적용. 이미 반영됐다면 reverse --check로 확인한다.
git -C RL2-VLA apply --unidiff-zero --check ../scripts/rl2_vla/patches/rl2_vla_stage2_single_candidate.patch
git -C RL2-VLA apply --unidiff-zero ../scripts/rl2_vla/patches/rl2_vla_stage2_single_candidate.patch
```

Submodule 변경은 parent repo의 patch로 보존한다. 원본 evaluator와 rl2_utils 두 파일을 변경하며, 기존 Stage1 patch와 겹칠 경우 자동 덮어쓰기하지 않는다.

## 데이터 확보·분할

기존 SAFE pickle은 chunk 첫 step의 reward만 담을 수 있다. 실제 step별 실행 action/outcome/다음 context를 증명할 수 없으면 QAM transition으로 변환하지 않는다. SAFE 학습 자료로 재사용 가능 여부는 별도로 판단한다. 원본을 삭제하거나 새 수집으로 덮어쓰지 않는다.

신규 수집의 JSON contract는 `data.py`가 검증한다. 한 episode에 reset/policy seed, policy checkpoint, context, 실제 실행 action, 대응되는 π0 normalized action, step별 성공/종료, 정상 종료 원인이 필요하다. JSON은 전체 episode 완료 후만 atomic하게 생성된다. `request.json`으로 다른 config의 출력 혼합을 막는다.

```bash
# dry-run: 네 task ×100 initial states ×3 policy seeds =1200 episodes
"$RL2_PY" "$STAGE2/run_experiment.py" --mode collect \
  --rl2-root "$RL2_ROOT" --python "$RL2_PY" --gpu 0 \
  --output "$EXP_ROOT/rollouts" --seeds 42 0 7 --trials 100
# 검증한 실행 GPU로 --gpu를 지정하고 --run 추가. 한 번에 한 모델만 실행.

"$RL2_PY" "$STAGE2/prepare.py" --rollouts "$EXP_ROOT/rollouts" \
  --output "$EXP_ROOT/prepared"
```

같은 task/reset의 여러 policy seed는 함께 60/20/20 train/validation/holdout으로 나뉜다. train만 cluster 및 QAM/SAFE 가중치 학습에 사용한다. validation은 SAFE calibration, holdout은 별도 진단이다. 평가 초기 상태 기본값은 10000부터로 수집 1000부터와 분리하며 실제 manifest를 대조한다.

기존 완료 lane은 episode 수와 seed 집합 및 request 일치가 확인될 때만 건너뛴다. 부분 lane은 보존하고 누락 초기 상태 범위를 새 output에 실행한다. 불완전 결과를 DONE으로 간주하거나 최소 공통 개수로 잘라 분석하지 않는다.

## 우리 SAFE와 두 QAM 학습

```bash
"$RL2_PY" "$STAGE2/train_safe.py" --manifest "$EXP_ROOT/prepared/manifest.json" \
  --safe-root "$RL2_ROOT/third_party/SAFE" --output "$EXP_ROOT/safe" \
  --device cuda --seed 42 --epochs 2000 --batch-size 32

"$RL2_PY" "$STAGE2/safe_provenance.py" --safe-dir "$EXP_ROOT/safe" \
  --manifest "$EXP_ROOT/safe/provenance.json"

for reward in base cluster_potential; do
  "$RL2_PY" "$STAGE2/train_qam.py" --manifest "$EXP_ROOT/prepared/manifest.json" \
    --qam-root "$RL2_ROOT/third_party/qam" --checkpoint "$QAM_INIT" \
    --reward-mode "$reward" --cluster-bundle "$EXP_ROOT/prepared/clusters.json" \
    --seed 42 --batch-size 256 --warmup-steps 5000 --steps 50000 \
    --output "$EXP_ROOT/qam_$reward"
done
```

GPU 선택은 실행 전 `CUDA_VISIBLE_DEVICES`로 지정한다. 학습들은 이 예시대로 순차 실행한다. QAM pretrained actor/target actor를 복원하되 critic/target critic 및 optimizer는 두 arm에서 동일하게 새로 초기화한다. SAFE는 원본 LSTM과 손실을 사용하고 마지막 epoch를 저장한다. 기존 우리 SAFE가 있으면 provenance 및 split/추출 계약이 확인될 때만 재사용한다. 제공된 저자 SAFE 자동 fallback은 없다.

SAFE calibration의 평균/표준편차는 train 성공 기록으로 만들고 validation 성공의 실제 관측 구간에서 one-sided 최대 residual을 계산한다. 같은 reset의 policy seed는 최대 residual 한 개로 묶는다. alpha=.2의 유한 order statistic을 만들 수 없으면 중단한다. train 성공이 관측되지 않은 후기 timestep은 threshold>1로 비활성화하며, 성공 종료 후 padding을 관측된 성공으로 만들지 않는다. 이 구간은 검출이 보장되지 않는 제한으로 보고한다.

## 평가·분석

```bash
"$RL2_PY" "$STAGE2/run_experiment.py" --mode eval \
  --rl2-root "$RL2_ROOT" --python "$RL2_PY" --gpu 0 \
  --output "$EXP_ROOT/eval" --seeds 42 --trials 25 \
  --manifest "$EXP_ROOT/prepared/manifest.json" --safe-dir "$EXP_ROOT/safe" \
  --qam-original "$QAM_INIT" \
  --qam-base "$EXP_ROOT/qam_base/params_50000.pkl" \
  --qam-shaped "$EXP_ROOT/qam_cluster_potential/params_50000.pkl"
# 실제 실행은 --run. 첫 seed에서 실행 계약 통과 후 --seeds 0 7로 추가.

"$RL2_PY" "$STAGE2/analyze.py" --eval-root "$EXP_ROOT/eval" \
  --output "$EXP_ROOT/summary.json"
```

4 arm: vanilla / original QAM / base 추가학습 / cluster 추가학습. 25 초기 상태 ×4 task ×3 평가 seed=300판/arm, 총1200판. QAM 학습 seed는 위 예시에서42 하나다. 평가 seed 3개를 학습 seed 반복으로 주장하지 않는다. 학습 seed를 반복할 때는 C/D 쌍을 함께 학습하고 별도 실험 출력으로 평가한다.

`analyze.py`는 전체/task별 성공 수, vanilla 실패 중 구제/성공 중 파괴, 각각 SAFE가 발동한 episode 수, 개입 chunk 수/최초 시점, D-C reset-group bootstrap CI를 낸다. 개입 후 경로가 달라지므로 각 arm의 SAFE 발동은 해당 arm에서 관측한 값이며 반사실적 동일 상태 alarm이 아니다. 결과 해석·보고 전에 confound-audit skill을 적용한다.

## 검증

```bash
python -m pytest tests/test_rl2_cluster_reward.py tests/test_safe_provenance.py \
  tests/test_rl2_stage2_eval.py tests/test_rl2_stage2_loop.py tests/test_rl2_safe_training.py
bash -n scripts/rl2_vla/stage1_simpler/run_arm.sh
git diff --check
```

Synthetic CPU smoke는 원본 QAM update와 원본 SAFE training/save/load를 검증한다. 실험 성공률로 해석하지 않는다. 실제 배포 환경에서는 먼저 task 하나·episode 하나로 π0 정규화, CPU/GPU tensor device, SAFE CP 인덱스, QAM 합성, JSON contract를 확인한다.

### GPU당 모델 2개 확인 (2026-09-16)

kanu A4000 GPU1에서 격리된 π0 2개 ×10판 시험을 수행했다. 두 process 모두 exit0, 20판 완료, OOM 없음. 실측 합산 사용량 약15904MiB/16376MiB로 여유는 작다. 이번 수집에서만 GPU당 최대2개를 허용하며, GR00T 규칙의 일반 확장으로 해석하지 않는다. `run_experiment.py --models-per-gpu 2`가 두 고정 worker를 GPU별로 배치한다(기본값1).

진행 중이던 최초 1200판은 기존 scheduler만 SIGSTOP하여 미착수 lane 중복 발사를 방지하고, 잔여3lane(seed7 eggplant/stack/carrot)을 GPU2/3/4의 두 번째 slot에 배정했다. 기존 collector와 shipper는 계속 실행한다. `outputs/rl2_iid_20260916/setup/expand_pending.py`가 잔여lane 완료 후 기존 scheduler를 SIGCONT하며, 기존 scheduler는 동일 request/완료 결과를 확인해 skip한다. 추가 총평가량은 없으며 분리된20판 capacity smoke만 본 데이터에서 제외한다. 실제 운영PID와 로그는 같은 setup 디렉터리에 있다.

## 실제 학습·보정·비교 실행 (2026-09-16, 최신)

수집1200판/sidecar1200개 전송검증 완료. 원본은 승준 archive root에만 두고 `export_training_cache.py`로 derived SAFE16feature와 QAMtransition을 추출했다. 파생캐시880MiB는 kanu `outputs/rl2_iid_training_cache_20260916`에 있으며 모든 checksum을 확인했다. SAFE reset-group split은 train60/selection10/calibration10/holdout20; QAM과 cluster는 기존train60을 동일하게 사용한다.

`safe_sweep.py`는 원본512batch LSTM/loss/optimizer, feature16×lr3×reg4×seed3=576학습을2000epoch까지 수행하고1000/2000endpoint1152개를 비교한다. 원본 지표는 task별 shortest duration까지의 max-score earlyROC-AUC이다. task cutoff는 train+selection만으로 정해 원본 전체자료 cutoff의 holdout lookahead를 제거했다. 3seed평균으로 설정을 선택하며 배포seed는 원본처럼0이다.

CP는 원본 Tfunc modulation, 성공trajectory30/70 회귀/잔차 분할, terminal edge-padding, np.quantile을 사용한다. 다만 reset그룹은분리한다. task별alpha15개를 원본padded final-end BA로 순위화하고top3(동점포함)를 보존하며, 실제배포는 task합산BA상위1개(동점lowalpha)를 고정한다. 실제종료전BA도 병기한다. 이원본padding은 이미끝난성공의후기alarm을alpha선택에포함할수있다. 엄밀한유한표본coverage를 주장하지않는다. holdout은설정/보정에쓰지않는다.

```bash
.venvs/rl2_iid/bin/python scripts/rl2_vla/stage2_cluster_reward/run_training_pipeline.py \
 --cache outputs/rl2_iid_training_cache_20260916/cache.json --rl2-root RL2-VLA \
 --output outputs/rl2_iid_20260916/training \
 --policy "$PWD/outputs/rl2_assets/pi0_bridge" \
 --qam-initial outputs/rl2_assets/qam_bridge/rl2_vla_qam_bridge_500k.pkl \
 --python "$PWD/.venvs/rl2_iid/bin/python" --run
```

SAFE10workers(GPU2/3/4/6/7,각2개)와QAMbase/shaped(GPU0/1)를동시실행한다. QAM은동일초기actor/criticseed/720trainrollouts/미니배치순서,5kcriticwarmup+50kupdates이며reward만다르다. 원본QAM초기checkpoint의N=1/horizon4/discount.99를검증한다.

모든학습이성공하면SAFE선택·CP보정·provenance검증후4arm(vanilla/originalQAM/baseQAM/shapedQAM)×4task×25reset×3policyseed=1200eval을수행한다. 첫seed42의400판이정상완료된후0/7의800판으로진행한다. eval은GPU당1모델이며모든gatedarm이같은SAFE와task별선택alpha를쓴다. SAFE입력선택과QAM고정mean/denoise0입력을분리했다. 실제GPUforced-gate smoke1판(9개chunk/9개개입)에서모델load/feature/taskwiseband/QAMcomposition/action실행을검증했다. 이smoke의인위적band는실험에사용하지않는다.

실행상태는`training/status.json`,세부로그는`training/logs/training/worker_*.log`,최종비교는`training/summary.json`이다. 기존QAM완료checkpoint는설정·hash일치시만재사용하며부분QAM은덮어쓰지않는다. SAFE는100epoch마다중간재개파일을저장한다. 최종성과보고전confound-audit를적용하고, IID/단일QAM학습seed범위를넘어일반화하지않는다.
