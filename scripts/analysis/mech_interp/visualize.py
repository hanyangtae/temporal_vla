"""Phase A 시각화: activation 추출 -> cluster -> injection -> 결과 를 그림/표/데이터로.

출력: ~/pkt_ws/mechanistic-steering-vlas/phaseA_viz/  (PNG + CSV)
"""
import os, re, json, yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
_fp = "/home/dongkyu/.fonts/NanumGothic-Regular.ttf"
if os.path.exists(_fp):
    fm.fontManager.addfont(_fp)
    matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/pkt_ws/mechanistic-steering-vlas")
ART = f"{ROOT}/openvla/ffn_value_vectors/artifacts/ffn_value_projection"
TXT = f"{ART}/top_tokens_output.txt"
DICT = f"{ROOT}/openvla/libero_experiments/configs/interventions/dictionaries_full.yaml"
LOGD = f"{ROOT}/openvla/libero_experiments/logs"
OUT = f"{ROOT}/phaseA_viz"
os.makedirs(OUT, exist_ok=True)
INTER = 11008

# ---------- load top tokens ----------
line_re = re.compile(r"\[(\d+)\]\s*(.*)")
def parse(rest):
    out = []
    for it in rest.split(", "):
        it = it.strip()
        if it.startswith("[action"):
            out.append(("action", it))
        elif len(it) >= 2 and it[0] == "'" and it[-1] == "'":
            out.append(("tok", it[1:-1]))
        elif it:
            out.append(("tok", it))
    return out
rows = {}
with open(TXT) as f:
    for line in f:
        m = line_re.match(line)
        if m: rows[int(m.group(1))] = parse(m.group(2))
nlayers = max(rows)//INTER + 1

# ====== (A) Fig 2b: action-token fraction per layer ======
frac = []
for L in range(nlayers):
    a = tot = 0
    for nrn in range(INTER):
        toks = rows.get(L*INTER+nrn, [])[:10]
        a += sum(1 for k,_ in toks if k=="action"); tot += len(toks)
    frac.append(a/tot if tot else 0)
plt.figure(figsize=(8,4))
plt.plot(range(nlayers), np.array(frac)*100, "o-", color="#c0392b")
plt.fill_between(range(nlayers), np.array(frac)*100, alpha=0.15, color="#c0392b")
plt.xlabel("transformer layer"); plt.ylabel("action-token % (top-10)")
plt.title("(A) Value vectors: action tokens concentrate in late layers  [paper Fig 2b]")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/A_action_token_per_layer.png", dpi=130); plt.close()
np.savetxt(f"{OUT}/A_action_token_per_layer.csv",
           np.c_[range(nlayers), np.array(frac)*100], delimiter=",",
           header="layer,action_token_pct", comments="")

# ====== (B) cluster layer distribution + injection table ======
dd = yaml.safe_load(open(DICT))
concepts = ["up_10","fast_10","slow_10"]
colors = {"up_10":"#2980b9","fast_10":"#27ae60","slow_10":"#8e44ad"}
plt.figure(figsize=(9,3.2))
for i,c in enumerate(concepts):
    flats = [int(k) for k in dd[c].keys()]
    layers = [f//INTER for f in flats]
    plt.scatter(layers, [i]*len(layers), s=80, color=colors[c], alpha=0.7, edgecolor="k")
plt.yticks(range(len(concepts)), concepts)
plt.xlabel("transformer layer (where the steered neuron lives)")
plt.title("(B) Which layers each concept-cluster's neurons come from")
plt.xlim(-1, nlayers); plt.grid(alpha=0.3, axis="x"); plt.tight_layout()
plt.savefig(f"{OUT}/B_cluster_layer_distribution.png", dpi=130); plt.close()

# injection table CSV (flat, layer, neuron, top tokens, coef)
import csv
with open(f"{OUT}/B_injection_points.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["concept","flat_idx","layer","neuron","coef","top5_tokens"])
    for c in ["up_10","up_20","fast_10","fast_20","slow_10","slow_14"]:
        if c not in dd: continue
        for flat in dd[c]:
            toks = [t[1] for t in rows.get(int(flat),[])[:5]]
            w.writerow([c,flat,int(flat)//INTER,int(flat)%INTER,6.0,"|".join(toks)])

# render fast_10/slow_10/up_10 token table as image
fig, axes = plt.subplots(3,1, figsize=(11,7.5));
for ax,c in zip(axes, concepts):
    ax.axis("off"); ax.set_title(f"(C) cluster '{c}' -> 주입 neuron + top tokens (coef=6)", loc="left", fontsize=11)
    table = [[f"L{int(fl)//INTER}", int(fl)%INTER, ", ".join(t[1] for t in rows[int(fl)][:6])] for fl in dd[c]]
    tb = ax.table(cellText=table, colLabels=["layer","neuron","top-6 tokens"],
                  cellLoc="left", colWidths=[0.07,0.1,0.83], loc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8)
plt.tight_layout(); plt.savefig(f"{OUT}/C_cluster_token_tables.png", dpi=130); plt.close()

# ====== (D) results: displacement ======
runs = {
 "baseline":"EVAL-libero_10-openvla-2026_06_25-09_00_49",
 "fast_10@6":"INTERVENTION-fast_10-coef6.0-2026_06_25-09_00_48",
 "slow_10@6":"INTERVENTION-slow_10-coef6.0-2026_06_25-09_16_51",
 "up_10@6":"INTERVENTION-up_10-coef6.0-2026_06_25-09_16_52",
}
per = {}
for name,run in runs.items():
    d = json.load(open(f"{LOGD}/{run}/actions.json"))
    vals = []
    for t,eps in d.items():
        for ep,a in eps.items():
            a = np.asarray(a,float)
            if a.ndim==2 and len(a): vals.append(float(np.linalg.norm(a[:,:3],axis=1).mean()))
    per[name] = vals
names = list(runs); means = [np.mean(per[n]) for n in names]; stds=[np.std(per[n]) for n in names]
plt.figure(figsize=(7,4.5))
bars = plt.bar(names, means, yerr=stds, capsize=5,
               color=["#7f8c8d","#27ae60","#8e44ad","#2980b9"])
for n in names:
    for v in per[n]:
        plt.scatter(n, v, color="k", alpha=0.5, s=20, zorder=3)
plt.ylabel("mean per-step EEF displacement  ||Δpos||")
ratio = means[1]/means[2]
plt.title(f"(D) Steering result: fast vs slow = +{(ratio-1)*100:.0f}%  (paper Fig5a: +27.7%)")
plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
plt.savefig(f"{OUT}/D_displacement_bar.png", dpi=130); plt.close()
with open(f"{OUT}/D_displacement.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["config","n_ep","mean_step_disp","std"])
    for n in names: w.writerow([n,len(per[n]),round(np.mean(per[n]),4),round(np.std(per[n]),4)])

# ====== (E) cumulative-z trajectory (height proxy), matched task 0 ======
plt.figure(figsize=(8,4.5))
for name,col in [("baseline","#7f8c8d"),("up_10@6","#2980b9"),("slow_10@6","#8e44ad")]:
    d = json.load(open(f"{LOGD}/{runs[name]}/actions.json"))
    task0 = list(d.keys())[0]; ep0 = list(d[task0].keys())[0]
    a = np.asarray(d[task0][ep0],float)
    plt.plot(np.cumsum(a[:,2]), color=col, label=name)
plt.xlabel("rollout step"); plt.ylabel("cumulative Δz (height proxy)")
plt.title(f"(E) Vertical trajectory on task: {task0[:45]}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/E_cumulative_z_task0.png", dpi=130); plt.close()

# manifest
print("=== 생성된 파일 ===")
for fn in sorted(os.listdir(OUT)):
    print(" ", f"{OUT}/{fn}")
print("\n=== displacement 요약 ===")
for n in names: print(f"  {n:12s} mean_step_disp={np.mean(per[n]):.4f} (n={len(per[n])})")
print(f"  fast/slow = +{(means[1]/means[2]-1)*100:.1f}%")
