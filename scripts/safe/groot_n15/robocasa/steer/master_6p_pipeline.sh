#!/usr/bin/env bash
# ★ 6/7-phase 전면 재실험 무인 파이프라인 (사용자 승인: 전부 재수집, fit30+fit60 둘 다, 이후 cross까지)
# Stage1 fit재수집(3cell×ep0-59, proximity+wrong-grasp) → Stage2 fit30/fit60 + cross30/60/90 fit(CPU)
# → Stage3 cell별 arm 체인(base,perm30,gated30,perm60,gated60; ep60-119) → Stage4 cross gated(cr30/60/90)
# → Stage5 집계+Notion+rsync+정리. GPU 상한: 6 serve / 3 GPU (bread=0, apple=4, potato=6).
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
LOG=outputs/eval/robocasa/groot_n15/pipeline_6p.log
exec >>"$LOG" 2>&1
echo "=========== MASTER 6P START $(date '+%F %T') ==========="

PY=~/miniconda3/envs/libero_bench/bin/python
FITPY=scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py
R=$(ls scripts/safe/groot_n15/robocasa/steer/heldout*_round_cell.sh)
AN6=outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
SRC6=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts
FIT_GROUPS="global,reach-to-object,grasp,transport,place,insert-settle"

# ---------- Stage 1: fit 재수집 ----------
if [ ! -f outputs/eval/robocasa/groot_n15/phase_event_6p/logs/FIT6P_DONE ]; then
  echo "[S1] $(date '+%T') fit 재수집 시작"
  bash scripts/safe/groot_n15/robocasa/collect/collect_fit_6phase.sh
  [ -f outputs/eval/robocasa/groot_n15/phase_event_6p/logs/FIT6P_DONE ] || { echo "[S1] FAIL"; exit 1; }
fi
echo "[S1] done"

# ---------- Stage 2: conceptor fits (CPU) ----------
declare -A TASKOF=( [ppcc_bread]=PickPlaceCounterToCabinet [ppcs_apple]=PickPlaceCounterToStove [ppcc_potato]=PickPlaceCounterToCabinet )
declare -A CIOF=( [ppcc_bread]=5 [ppcs_apple]=1 [ppcc_potato]=4 )
for cell in ppcc_bread ppcs_apple ppcc_potato; do
  T=${TASKOF[$cell]}; ci=${CIOF[$cell]}
  # fit30 부분집합 심링크 (ep0-29)
  D30=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_fit30/$T/$cell
  mkdir -p "$D30"
  for ep in $(seq 0 29); do
    src=$(ls $(pwd)/$SRC6/$T/$cell/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
    [ -n "$src" ] && ln -sf "$src" "$D30/"
  done
  for N in 30 60; do
    OUT=$AN6/conceptor_6p_fit${N}/$cell
    [ -f "$OUT/fit_summary.json" ] && continue
    RUNDIR=$([ $N = 30 ] && echo outputs/eval/robocasa/groot_n15/phase_event_6p/raw_fit30 || echo $SRC6)
    echo "[S2] fit${N} $cell"
    mkdir -p "$OUT"
    OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 $PY $FITPY --run-dir "$RUNDIR" --cell "$T/$cell" \
      --groups "$FIT_GROUPS" --carve-window 0 --out-dir "$OUT" > "$OUT.fitlog" 2>&1 || true
    grep -E '^\[cell|\[done' "$OUT.fitlog" || true
    if [ ! -f "$OUT/fit_summary.json" ]; then
      echo "[S2] FIT FAILED: $cell fit${N} — tail:"; tail -20 "$OUT.fitlog"; exit 2
    fi
  done
done
# cross fits (총 30/60/90 = cell당 10/20/30)
for N in 30 60 90; do
  PER=$((N / 3))
  D=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_cross${N}/Cross/all
  mkdir -p "$D"
  for cell in ppcc_bread ppcs_apple ppcc_potato; do
    T=${TASKOF[$cell]}; ci=${CIOF[$cell]}
    for ep in $(seq 0 $((PER - 1))); do
      src=$(ls $(pwd)/$SRC6/$T/$cell/task${ci}--ep${ep}--succ*.pkl 2>/dev/null | head -1)
      [ -n "$src" ] && ln -sf "$src" "$D/"
    done
  done
  OUT=$AN6/conceptor_6p_cross${N}/all
  if [ ! -f "$OUT/fit_summary.json" ]; then
    echo "[S2] cross fit${N}: $(ls $D/*.pkl 2>/dev/null|wc -l)판"
    mkdir -p "$OUT"
    OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 $PY $FITPY \
      --run-dir outputs/eval/robocasa/groot_n15/phase_event_6p/raw_cross${N} --cell Cross/all \
      --groups "$FIT_GROUPS" --carve-window 0 --out-dir "$OUT" > "$OUT.fitlog" 2>&1 || true
    grep -E '^\[cell|\[done' "$OUT.fitlog" || true
    if [ ! -f "$OUT/fit_summary.json" ]; then
      echo "[S2] CROSS FIT FAILED: fit${N} — tail:"; tail -20 "$OUT.fitlog"; exit 2
    fi
  fi
  for cell in ppcc_bread ppcs_apple ppcc_potato; do
    ln -sfn all "$AN6/conceptor_6p_cross${N}/$cell"
  done
done
echo "[S2] done"

# ---------- Stage 3+4: cell별 arm 체인 (병렬, cell당 GPU1×serve2) ----------
CROOT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
cell_chain() {  # cell task envn ci seed instr gpu pA pB
  local cell=$1 task=$2 envn=$3 ci=$4 seed=$5 instr=$6 gpu=$7 pA=$8 pB=$9
  local CLOG=outputs/eval/robocasa/groot_n15/steer_eval/${cell}/pipeline6p.log
  mkdir -p outputs/eval/robocasa/groot_n15/steer_eval/${cell}
  run() {  # SUF ARMS NPZ_SUBDIR
    CELL_ID=$cell TASK=$task ENVN=$envn CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
      GPUS_L="$gpu $gpu" PORTS_L="$pA $pB" EP0=60 EP1=119 PROX=1 \
      SUF=$1 ARMS="$2" NPZ_ROOT=${CROOT}/$3 bash "$R" >> "$CLOG" 2>&1
  }
  echo "[S3 $cell] base+fit30 $(date '+%T')"
  run 6p30 "base perm gated" conceptor_6p_fit30
  echo "[S3 $cell] fit60 $(date '+%T')"
  run 6p60 "perm gated" conceptor_6p_fit60
  echo "[S4 $cell] cross $(date '+%T')"
  for N in 30 60 90; do run 6pcr${N} "gated" conceptor_6p_cross${N}; done
  echo "[chain $cell] DONE $(date '+%T')"
}
cell_chain ppcc_bread PickPlaceCounterToCabinet robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env 5 100084 "Pick the bread from the counter and place it in the cabinet." 0 8480 8481 &
cell_chain ppcs_apple PickPlaceCounterToStove robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env 1 100074 "Pick the apple from the plate and place it in the pan." 4 8470 8471 &
cell_chain ppcc_potato PickPlaceCounterToCabinet robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env 4 200019 "Pick the potato from the counter and place it in the cabinet." 6 8472 8473 &
wait
echo "[S3+4] all cells done $(date '+%T')"

# ---------- Stage 5: 집계 + Notion + rsync + 정리 ----------
set -a; . ./.env; set +a
$PY - <<'PYEOF'
import os, glob, json, urllib.request
E="outputs/eval/robocasa/groot_n15/steer_eval"
CI={"ppcc_bread":(5,"PickPlaceCounterToCabinet"),"ppcs_apple":(1,"PickPlaceCounterToStove"),"ppcc_potato":(4,"PickPlaceCounterToCabinet")}
def sr(cell,tag,e0=60,e1=119):
    ci,T=CI[cell]; d=f"{E}/{cell}/{tag}/raw_rollouts/{T}/{cell}"
    s=f=0
    for ep in range(e0,e1+1):
        if glob.glob(f"{d}/task{ci}--ep{ep}--succ1.pkl"): s+=1
        elif glob.glob(f"{d}/task{ci}--ep{ep}--succ0.pkl"): f+=1
    return s,s+f
rows=[]
for cell in CI:
    r={"cell":cell}
    for tag,name in [("ho_base","base"),("ho_perm6p30","perm30"),("ho_gated6p30","gated30"),
                     ("ho_perm6p60","perm60"),("ho_gated6p60","gated60"),
                     ("ho_gated6pcr30","cr30"),("ho_gated6pcr60","cr60"),("ho_gated6pcr90","cr90")]:
        s,n=sr(cell,tag); r[name]=f"{s}/{n}" if n else "-"
    rows.append(r)
out="outputs/eval/robocasa/groot_n15/steer_eval/RESULTS_6p_final.json"
json.dump(rows,open(out,"w"),indent=1)
print("results:",rows)
TOK=os.environ.get("NOTION_TOKEN")
if TOK:
    def rt(t,b=False): return [{"type":"text","text":{"content":t},"annotations":{"bold":b}}]
    def row(c,h=False): return {"type":"table_row","table_row":{"cells":[rt(x,h) for x in c]}}
    ch=[{"type":"heading_3","heading_3":{"rich_text":rt("★ 6/7-phase 전면 재실험 최종 (fit30/fit60/cross, eval ep60-119)")}},
        {"type":"table","table":{"table_width":9,"has_column_header":True,"children":
         [row(["cell","base","perm30","gated30","perm60","gated60","cr30","cr60","cr90"],True)]+
         [row([r["cell"],r["base"],r["perm30"],r["gated30"],r["perm60"],r["gated60"],r["cr30"],r["cr60"],r["cr90"]]) for r in rows]}}]
    req=urllib.request.Request("https://api.notion.com/v1/blocks/38e63918d42a80698ac2f193716c03a3/children",
        data=json.dumps({"children":ch}).encode(),
        headers={"Authorization":f"Bearer {TOK}","Notion-Version":"2022-06-28","Content-Type":"application/json"},method="PATCH")
    urllib.request.urlopen(req); print("notion appended")
PYEOF

echo "[S5] rsync"
rsync -az --stats -e "ssh -o BatchMode=yes -p 11112" \
  outputs/eval/robocasa/groot_n15/steer_eval \
  outputs/eval/robocasa/groot_n15/phase_event_strict \
  outputs/eval/robocasa/groot_n15/phase_event_6p \
  outputs/eval/robocasa/groot_n15/smoke_6phase \
  kimseungjun@166.104.146.37:workspace/temporal_vla/outputs/eval/robocasa/groot_n15/ \
  > /tmp/rsync_final_6p.log 2>&1
rc=$?
echo "[S5] rsync rc=$rc"
INCOMPLETE=$(grep -c '"-"' outputs/eval/robocasa/groot_n15/steer_eval/RESULTS_6p_final.json || true)
if [ $rc -eq 0 ] && [ "$INCOMPLETE" -eq 0 ]; then
  # 검증 후 대용량 로컬 정리 (요약물·NPZ·json·tsv·log 유지)
  for d in steer_eval phase_event_strict phase_event_6p phase_event_aligned_4cell smoke_6phase; do
    find outputs/eval/robocasa/groot_n15/$d -type f \( -name '*.pkl' -o -name '*.mp4' -o -name '*.csv' \) -delete 2>/dev/null
  done
  echo "[S5] local heavy files deleted (rsync verified rc=0)"
else
  echo "[S5] rsync rc=$rc / 미완성결과=$INCOMPLETE — 삭제 보류"
fi
touch outputs/eval/robocasa/groot_n15/PIPELINE_6P_ALL_DONE
echo "=========== MASTER 6P COMPLETE $(date '+%F %T') ==========="
