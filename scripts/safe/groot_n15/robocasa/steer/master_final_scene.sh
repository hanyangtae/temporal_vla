#!/usr/bin/env bash
# ★ 최종 라운드(고정seed·no-SAE 종결): scene-seed 일반화 대규모 매트릭스.
# 8 cell(bread×4seed, apple×4seed) × arm{base + (perseed|xseed|grand)×(perm|gated)×(15|30|60)} (+bread layer {4},{8,12})
# S0 데이터복원 → S1 신규6cell fit수집(ep0-59) → S2 전체 fits(CPU) → S3 cell체인(ep60-119) → S4 집계/Notion/rsync
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
LOG=outputs/eval/robocasa/groot_n15/pipeline_final.log
exec >>"$LOG" 2>&1
echo "=========== FINAL SCENE START $(date '+%F %T') ==========="
PY=~/miniconda3/envs/libero_bench/bin/python
FITPY=scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py
R=$(ls scripts/safe/groot_n15/robocasa/steer/heldout*_round_cell.sh)
SRC6=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts
AN=outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
CROOT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
FIT_GROUPS="global,reach-to-object,grasp,transport,place,insert-settle"
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

# cell|task|env|ci|seed|instr  (bread=B, apple=A)
CELLS=(
  "ppcc_bread|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|100084|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s300028|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|300028|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s300033|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|300033|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s400020|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|400020|Pick the bread from the counter and place it in the cabinet."
  "ppcs_apple|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100074|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100050|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100050|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100084|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100084|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100104|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100104|Pick the apple from the plate and place it in the pan."
)
BREADS="ppcc_bread ppcc_bread_s300028 ppcc_bread_s300033 ppcc_bread_s400020"
APPLES="ppcs_apple ppcs_apple_s100050 ppcs_apple_s100084 ppcs_apple_s100104"
row_of() { for c in "${CELLS[@]}"; do [[ "$c" == "$1|"* ]] && { echo "$c"; return; }; done; }

# ---------- S0: 기존 6p 데이터 복원 (S5 삭제분) ----------
if [ ! -f "$SRC6/PickPlaceCounterToCabinet/ppcc_bread/task5--ep0--succ0.pkl" ] && \
   ! ls $SRC6/PickPlaceCounterToCabinet/ppcc_bread/task5--ep0--succ*.pkl >/dev/null 2>&1; then
  echo "[S0] pull 6p raw from remote"
  rsync -az -e "ssh -o BatchMode=yes -p 11112" \
    kimseungjun@166.104.146.37:workspace/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts \
    outputs/eval/robocasa/groot_n15/phase_event_6p/ || { echo "[S0] FAIL"; exit 1; }
fi
echo "[S0] done"

# ---------- S1: 신규 6 cell fit 수집 (ep0-59, 6/7-phase) ----------
start_serve() { docker exec -d -e CUDA_VISIBLE_DEVICES="$1" lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port ${2} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     > /tmp/final_${2}.log 2>&1 < /dev/null &"; }
wait_health() { for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${1}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && return 0; sleep 5; done; return 1; }
collect_cell() {  # row portA portB e0 e1
  local row="$1" pA="$2" pB="$3" e0="$4" e1="$5"
  IFS='|' read -r cell task env ci seed instr <<<"$row"
  cw() { local wid=$1 port=$2
    for ep in $(seq $e0 $e1); do
      [ $((ep % 2)) -eq "$wid" ] || continue
      ls "$SRC6/${task}/${cell}/task${ci}--ep${ep}--succ"*.pkl >/dev/null 2>&1 && continue
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$env" \
        --output-dir "/temporal_vla/${SRC6}" --cell-id "$cell" --cell-index "$ci" \
        --canonical-instruction "$instr" --episode-start-idx "$ep" --n-episodes 1 \
        --seed "$seed" --inference-seed "$((ep * 1000))" --n-action-steps 5 \
        --max-episode-steps 720 --video-fps 20 --steps-per-render 2 --wait-ready \
        --proximity-phases 2>&1 | grep -E "^wrote|Traceback" || true
    done; }
  cw 0 "$pA" & cw 1 "$pB" & wait
  echo "[S1] $cell done: $(ls $SRC6/${task}/${cell}/*.pkl 2>/dev/null | wc -l)판"
}
NEWCELLS_L0="ppcc_bread_s300028 ppcc_bread_s300033"
NEWCELLS_L1="ppcs_apple_s100050 ppcs_apple_s100084"
NEWCELLS_L2="ppcc_bread_s400020 ppcs_apple_s100104"
if [ ! -f "$SRC6/.S1_FINAL_DONE" ]; then
  start_serve 0 8480; start_serve 0 8481; start_serve 4 8470; start_serve 4 8471; start_serve 6 8472; start_serve 6 8473
  for p in 8480 8481 8470 8471 8472 8473; do wait_health $p || { echo "[S1] serve $p TIMEOUT"; exit 11; }; done
  ( for c in $NEWCELLS_L0; do collect_cell "$(row_of $c)" 8480 8481 0 59; done ) &
  ( for c in $NEWCELLS_L1; do collect_cell "$(row_of $c)" 8470 8471 0 59; done ) &
  ( for c in $NEWCELLS_L2; do collect_cell "$(row_of $c)" 8472 8473 0 59; done ) &
  wait
  for p in 8480 8481 8470 8471 8472 8473; do docker exec lerobot bash -lc "pkill -9 -f 'lerobot.py.*${p}' || true" 2>/dev/null || true; done
  touch "$SRC6/.S1_FINAL_DONE"
fi
echo "[S1] done"

# ---------- S2: fits (CPU) — per-seed/xseed/grand × 15/30/60 ----------
fit_dir() {  # rundir cellpath outdir
  [ -f "$3/fit_summary.json" ] && return 0
  mkdir -p "$3"
  OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 $PY $FITPY --run-dir "$1" --cell "$2" \
    --groups "$FIT_GROUPS" --carve-window 0 --out-dir "$3" > "$3.fitlog" 2>&1 || true
  grep -m1 '\[done\]' "$3.fitlog" >/dev/null || { echo "[S2] FIT FAILED: $3"; tail -5 "$3.fitlog"; exit 2; }
}
# per-seed subsets
for row in "${CELLS[@]}"; do
  IFS='|' read -r cell task env ci seed instr <<<"$row"
  for N in 15 30 60; do
    SUBD=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_ps${N}/$task/$cell
    mkdir -p "$SUBD"
    for ep in $(seq 0 $((N - 1))); do
      src=$(ls $(pwd)/$SRC6/$task/$cell/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      [ -n "$src" ] && ln -sf "$src" "$SUBD/"
    done
    fit_dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_ps${N} "$task/$cell" "$AN/final_ps${N}/$cell"
  done
done
# cross-seed(instruction별) + grand: round-robin pool
pool_fit() {  # name cells... / uses N loop inside
  local name=$1; shift; local cells=("$@")
  for N in 15 30 60; do
    D=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_${name}${N}/Pool/all
    mkdir -p "$D"
    local i=0
    while [ $i -lt $N ]; do
      local cell=${cells[$((i % ${#cells[@]}))]}
      local ep=$((i / ${#cells[@]}))
      IFS='|' read -r c task env ci seed instr <<<"$(row_of $cell)"
      src=$(ls $(pwd)/$SRC6/$task/$c/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      [ -n "$src" ] && ln -sf "$src" "$D/${c}--ep${ep}.pkl"
      i=$((i + 1))
    done
    fit_dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_${name}${N} "Pool/all" "$AN/final_${name}${N}/all"
    for c2 in $BREADS $APPLES; do ln -sfn all "$AN/final_${name}${N}/$c2"; done
  done
}
pool_fit xb $BREADS
pool_fit xa $APPLES
pool_fit gx $BREADS $APPLES
echo "[S2] done"

# ---------- S3: cell별 arm 체인 (3 GPU lane) ----------
cell_chain() {  # row gpu pA pB
  IFS='|' read -r cell task env ci seed instr <<<"$1"
  local gpu=$2 pA=$3 pB=$4
  local XN=xb; [[ "$cell" == ppcs_* ]] && XN=xa
  run() {  # SUF ARMS NPZ_SUBDIR [layers]
    CELL_ID=$cell TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
      GPUS_L="$gpu $gpu" PORTS_L="$pA $pB" EP0=60 EP1=119 PROX=1 \
      SUF=$1 ARMS="$2" NPZ_ROOT=${CROOT}/$3 STEER_LAYERS="${4:-4,8,12}" \
      bash "$R" >> outputs/eval/robocasa/groot_n15/steer_eval/${cell}/final_chain.log 2>&1
  }
  mkdir -p outputs/eval/robocasa/groot_n15/steer_eval/${cell}
  for N in 15 30 60; do
    A="perm gated"; [ $N = 15 ] || A="perm gated"
    run ps${N} "$([ $N = 15 ] && echo base perm gated || echo perm gated)" final_ps${N}
    run ${XN}${N} "gated perm" final_${XN}${N}
    run gx${N} "gated perm" final_gx${N}
  done
  if [[ "$cell" == ppcc_bread* ]]; then
    run L4 "gated" final_ps60 "4"
    run L812 "gated" final_ps60 "8,12"
  fi
  echo "[S3 chain $cell] DONE $(date '+%T')"
}
lane() { for c in "$@"; do cell_chain "$(row_of $c)" "$LGPU" "$LPA" "$LPB"; done; }
LGPU=0 LPA=8480 LPB=8481 lane ppcc_bread ppcc_bread_s300028 ppcc_bread_s300033 &
LGPU=4 LPA=8470 LPB=8471 lane ppcc_bread_s400020 ppcs_apple ppcs_apple_s100050 &
LGPU=6 LPA=8472 LPB=8473 lane ppcs_apple_s100084 ppcs_apple_s100104 &
wait
echo "[S3] all done $(date '+%F %T')"

# ---------- S4: 집계 + Notion + rsync ----------
set -a; . ./.env; set +a
$PY - <<'PYEOF'
import os, glob, json, urllib.request
E="outputs/eval/robocasa/groot_n15/steer_eval"
CELLS={"ppcc_bread":(5,"PickPlaceCounterToCabinet"),"ppcc_bread_s300028":(5,"PickPlaceCounterToCabinet"),
"ppcc_bread_s300033":(5,"PickPlaceCounterToCabinet"),"ppcc_bread_s400020":(5,"PickPlaceCounterToCabinet"),
"ppcs_apple":(1,"PickPlaceCounterToStove"),"ppcs_apple_s100050":(1,"PickPlaceCounterToStove"),
"ppcs_apple_s100084":(1,"PickPlaceCounterToStove"),"ppcs_apple_s100104":(1,"PickPlaceCounterToStove")}
def sr(cell,tag):
    ci,T=CELLS[cell]; d=f"{E}/{cell}/{tag}/raw_rollouts/{T}/{cell}"
    s=f=0
    for ep in range(60,120):
        if glob.glob(f"{d}/task{ci}--ep{ep}--succ1.pkl"): s+=1
        elif glob.glob(f"{d}/task{ci}--ep{ep}--succ0.pkl"): f+=1
    return f"{s}/{s+f}" if s+f else "-"
arms=["ho_base"]
for k in ["ps","xb_or_xa","gx"]: pass
rows=[]
for cell in CELLS:
    xn="xb" if cell.startswith("ppcc") else "xa"
    r={"cell":cell,"base":sr(cell,"ho_base")}
    for N in [15,30,60]:
        r[f"perm{N}"]=sr(cell,f"ho_permps{N}"); r[f"gated{N}"]=sr(cell,f"ho_gatedps{N}")
        r[f"x{N}"]=sr(cell,f"ho_gated{xn}{N}"); r[f"xp{N}"]=sr(cell,f"ho_perm{xn}{N}")
        r[f"g{N}"]=sr(cell,f"ho_gatedgx{N}"); r[f"gp{N}"]=sr(cell,f"ho_permgx{N}")
    if cell.startswith("ppcc"):
        r["L4"]=sr(cell,"ho_gatedL4"); r["L812"]=sr(cell,"ho_gatedL812")
    rows.append(r)
json.dump(rows,open("outputs/eval/robocasa/groot_n15/steer_eval/RESULTS_final_scene.json","w"),indent=1)
print("results:",json.dumps(rows)[:800])
TOK=os.environ.get("NOTION_TOKEN")
if TOK:
    def rt(t,b=False): return [{"type":"text","text":{"content":t},"annotations":{"bold":b}}]
    def row(c,h=False): return {"type":"table_row","table_row":{"cells":[rt(x,h) for x in c]}}
    hdr=["cell","base","perm15/30/60","gated15/30/60","xseed-g15/30/60","grand-g15/30/60","L4","L812"]
    body=[]
    for r in rows:
        body.append(row([r["cell"],r["base"],
            "/".join(r[f"perm{N}"] for N in [15,30,60]),
            "/".join(r[f"gated{N}"] for N in [15,30,60]),
            "/".join(r[f"x{N}"] for N in [15,30,60]),
            "/".join(r[f"g{N}"] for N in [15,30,60]),
            r.get("L4","-"),r.get("L812","-")]))
    ch=[{"type":"heading_3","heading_3":{"rich_text":rt("★ scene-seed 일반화 최종 매트릭스 (8 cell, fit15/30/60, eval ep60-119)")}},
        {"type":"table","table":{"table_width":8,"has_column_header":True,"children":[row(hdr,True)]+body}}]
    req=urllib.request.Request("https://api.notion.com/v1/blocks/38e63918d42a80698ac2f193716c03a3/children",
        data=json.dumps({"children":ch}).encode(),
        headers={"Authorization":f"Bearer {TOK}","Notion-Version":"2022-06-28","Content-Type":"application/json"},method="PATCH")
    urllib.request.urlopen(req); print("notion appended")
PYEOF
echo "[S4] rsync"
rsync -az -e "ssh -o BatchMode=yes -p 11112" \
  outputs/eval/robocasa/groot_n15/steer_eval outputs/eval/robocasa/groot_n15/phase_event_6p \
  kimseungjun@166.104.146.37:workspace/temporal_vla/outputs/eval/robocasa/groot_n15/ && echo "[S4] rsync ok"
touch outputs/eval/robocasa/groot_n15/FINAL_SCENE_ALL_DONE
echo "=========== FINAL SCENE COMPLETE $(date '+%F %T') ==========="
