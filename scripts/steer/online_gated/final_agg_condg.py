"""condg 최종 셀단위 집계 — 쌍대응·분모(발화/실개입)·in-fit/held-out-noise 2분할.

usage: python final_agg_condg.py <og_condg_root> <slug> <arm1,arm2,...>
"""
import glob
import json
import sys

ROOT, SLUG = sys.argv[1], sys.argv[2]
ARMS = sys.argv[3].split(",")

# 셀 표(replay_cells.py: v6 11열 / legacy 9열) → {(평탄 cell id, noise): 수집 라벨}.
# 두 포맷 모두 **9번째 열(p[8])이 지터 좌표**다 — v6 는 jitter_idx(j), legacy 는
# jitter_reset_idx(k). v6 표는 p[9] 에 plan 유래 reset_idx 가 따로 있는데 좌표가 아니라
# 여기선 쓰지 않는다 (v6 표에서 p[8] 이 비면 legacy 행이라 p[9] 로 대체).
# 아래 scan() 이 ep 를 (ep//100, ep%100) 로 되접으므로 키도 러너와 같은 평탄 좌표여야
# 한다 (지터행 scene*100+j / 2축 legacy scene — docs/04 §3.1.1).
EMPTY = ("", "base", "NA", "None")
cells = {}
for ln in open(f"{ROOT}/logs/cells_{SLUG}.tsv"):
    p = ln.rstrip("\n").split("\t")
    k = p[8].strip() if len(p) > 8 else ""
    if k in EMPTY and len(p) > 9:
        k = p[9].strip()
    try:
        si = int(p[0])
        flat = si if k in EMPTY else si * 100 + int(k)
        cells[(flat, int(p[1]))] = int(p[4])
    except ValueError:
        continue

def scan(arm):
    out = {}
    for f in glob.glob(f"{ROOT}/{SLUG}/{arm}/**/task0--ep*.json", recursive=True):
        ep = int(f.split("--ep")[1].split("--")[0])
        d = json.load(open(f))
        g = d.get("phase_gated_flags") or []
        out[(ep // 100, ep % 100)] = {
            "succ": int(f.endswith("succ1.json")),
            "fired": d.get("trigger_step") is not None,
            "applied": sum(bool(x) for x in g),
        }
    return out

arm_data = {a: scan(a) for a in ARMS}

print(f"== {SLUG} (base=수집 라벨, n=발화/실개입/뒤집힘) ==")
for a in ARMS:
    D = arm_data[a]
    for grp, name in [((0, 1), "in-fit-noise(n0,1)"), ((5, 6), "held-out(n5,6)")]:
        rows = [(c, v) for c, v in D.items() if c[1] in grp and c in cells]
        nf = [r for r in rows if cells[r[0]] == 0]
        ns = [r for r in rows if cells[r[0]] == 1]
        resc = [c for c, v in nf if v["succ"] == 1]
        brk = [c for c, v in ns if v["succ"] == 0]
        fired_f = sum(1 for _, v in nf if v["fired"])
        app_f = sum(1 for _, v in nf if v["applied"] > 0)
        fired_s = sum(1 for _, v in ns if v["fired"])
        app_s = sum(1 for _, v in ns if v["applied"] > 0)
        sr = (len([1 for _, v in rows if v["succ"]]) / len(rows)) if rows else float("nan")
        print(f"{a:<12} {name:<18} n={len(rows):2d} SR={sr:.3f} "
              f"구제 {len(resc)}/{len(nf)} (발화{fired_f}·실개입{app_f}) "
              f"파손 {len(brk)}/{len(ns)} (오발화{fired_s}·개입{app_s})"
              + (f"  구제셀:{resc}" if resc else "") + (f" 파손셀:{brk}" if brk else ""))
