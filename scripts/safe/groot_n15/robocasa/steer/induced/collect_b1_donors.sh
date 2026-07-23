#!/usr/bin/env bash
# exp4-2 B1 VL donor 수집 — 같은 scene(100084) 에 **타 instruction** 을 ep_meta.lang 편집
# replay 로 주입해 post_vl_sa_full 캡처 → extract_vl_donor_npz 로 [R,T_vl,D] donor 생산.
# 지시 2종(24b B1): "Open the drawer."(무관 과제) + receptacle 변경형(goal 변경).
# donor 는 성공 여부 무관 (--allow-fail — VL 임베딩은 record 0 부터 지시를 인코딩;
# PPCC 성공판정은 원 과제 기준이라 타 지시 rollout 의 succ 비트는 의미 없음. 보고서 명기).
# 사용: GPU=<idx> PORT=8492 bash collect_b1_donors.sh
set -uo pipefail

GPU="${GPU:?빈 GPU}"
PORT="${PORT:-8492}"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
TASK=PickPlaceCounterToCabinet
ENVN="robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
CELL_INDEX=5; NAS=5; MAXEP=720; SEED=100084
INSTR1="Open the drawer."
INSTR2="Pick the bread from the counter and place it in the sink."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WT_HOST="$(cd -- "${SCRIPT_DIR}/../../../../../.." && pwd)"
MAIN_HOST="$(cd -- "${WT_HOST}/../../.." && pwd)"
WT_CONT="/temporal_vla/.claude/worktrees/exp4-2-induced-failures"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
B1="${MAIN_HOST}/outputs/eval/robocasa/groot_n15/exp42_induced/b1_donors"
mkdir -p "$B1"

to_cont() { echo "${1/#${MAIN_HOST}//temporal_vla}"; }

# (07-23 정정) kitchen 은 lang 을 ep_meta 에서 읽지 않고 task 로직에서 재생성 — ep_meta
# lang 편집 replay 는 무효 (실측: 원 지시로 원복). 대신 collector 의 --instruction-override
# 로 모델에 보내는 instruction 만 교체한다 (env 는 원 과제 정상 실행 — scene/물리 동일).

# 2) serve (VL full 캡처)
docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -lc \
  "cd ${WT_CONT} && setsid nohup python scripts/serve/exp42_serve.py --profile ${PROFILE} \
     --host '*' --port ${PORT} --device cuda --collect --capture-vl \
     --groot-vl-capture-point post_vl_sa_full --groot-dit-capture-layers 15 \
     > /tmp/exp42_b1_${PORT}.log 2>&1 < /dev/null &"
ok=0
for _ in $(seq 1 150); do
  curl -s -m 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
  sleep 5
done
[ $ok = 1 ] || { echo ABORT-serve; docker exec lerobot bash -lc "tail -20 /tmp/exp42_b1_${PORT}.log"; exit 11; }

# 3) 지시별 2ep 수집 (inference seed 상이) + donor 추출
for i in 1 2; do
  eval "INSTR=\$INSTR$i"
  for ep in 0 1; do
    out="$B1/instr$i"
    if ! ls "$out/raw_rollouts/${TASK}/b1_instr$i/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then
      echo "[b1] instr$i ep$ep"
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python "${WT_CONT}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py" \
        --vla-server "http://127.0.0.1:${PORT}" --task "$TASK" --env-name "$ENVN" \
        --output-dir "$(to_cont "$out")/raw_rollouts" --cell-id "b1_instr$i" \
        --cell-index "$CELL_INDEX" --canonical-instruction "$INSTR" \
        --episode-start-idx "$ep" --n-episodes 1 --seed "$SEED" \
        --inference-seed "$((900000 + i * 1000 + ep))" --n-action-steps "$NAS" \
        --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
        --proximity-phases --instruction-override "$INSTR" 2>&1 \
        | grep -E "^wrote|Error|Traceback" || true
    fi
    pkl=$(ls "$out/raw_rollouts/${TASK}/b1_instr$i/task${CELL_INDEX}--ep${ep}--succ"*.pkl 2>/dev/null | head -1)
    if [ -n "$pkl" ] && [ ! -f "$B1/donor_instr${i}_ep${ep}.npz" ]; then
      docker exec lerobot python "${WT_CONT}/scripts/safe/groot_n15/robocasa/steer/induced/extract_vl_donor_npz.py" \
        --pkl "$(to_cont "$pkl")" --out "$(to_cont "$B1")/donor_instr${i}_ep${ep}.npz" --allow-fail
    fi
  done
done
docker exec lerobot bash -lc "pkill -f 'serve/exp42_serve.py.*--port ${PORT}' || true"
ls -la "$B1"/*.npz 2>/dev/null
echo "[b1] DONE"
