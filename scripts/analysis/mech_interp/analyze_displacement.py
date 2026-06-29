"""5-trial baseline/fast/slow → 3가지 비교.
1) 전체 속도: 에피소드 전체 mean step disp
2) task 내 최단 길이로 truncate (길이통제)
3) 논문 방식: fast vs slow, task별 paired, 10 task paired t-test + % 개선 (Fig5a)
disp = mean_t ||action[:3]||  (OpenVLA LIBERO delta action = eef 속도)
"""
import json, glob, os, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
fp="/home/dongkyu/.fonts/NanumGothic-Regular.ttf"
if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams["font.family"]="NanumGothic"
matplotlib.rcParams["axes.unicode_minus"]=False

LOGD="/home/dongkyu/pkt_ws/mechanistic-steering-vlas/openvla/libero_experiments/logs"
OUT="/home/dongkyu/pkt_ws/mechanistic-steering-vlas/phaseA_viz"
runs={"baseline":sorted(glob.glob(f"{LOGD}/EVAL-libero_10-openvla-2026_06_29-02_*"))[-1],
      "fast":sorted(glob.glob(f"{LOGD}/INTERVENTION-fast_10-coef6.0-2026_06_29-02_*"))[-1],
      "slow":sorted(glob.glob(f"{LOGD}/INTERVENTION-slow_10-coef6.0-2026_06_29-02_*"))[-1]}
print("run dirs:", {k:os.path.basename(v) for k,v in runs.items()})
data={n:json.load(open(f"{r}/actions.json")) for n,r in runs.items()}
tasks=list(data["baseline"].keys())
arr=lambda actions: np.asarray(actions,float)
def disp(a, w=None):
    a=a[:w] if w else a
    return float(np.linalg.norm(a[:,:3],axis=1).mean())

# ---- (1) 전체 속도 ----
full={n:[] for n in runs}
for n in runs:
    for t in tasks:
        for ep,a in data[n][t].items(): full[n].append(disp(arr(a)))
print("\n=== (1) 전체 속도 (full episode, n=%d/cond) ===" % len(full["baseline"]))
m={n:np.mean(full[n]) for n in runs}
for n in runs: print(f"  {n:8s} {m[n]:.4f} ± {np.std(full[n]):.4f}")
print(f"  fast/slow = {(m['fast']/m['slow']-1)*100:+.1f}%")
print(f"  slow vs base {(m['slow']/m['baseline']-1)*100:+.1f}% | fast vs base {(m['fast']/m['baseline']-1)*100:+.1f}%")

# ---- (2) task 내 최단 길이로 truncate ----
trunc={n:[] for n in runs}
per_task_trunc={n:[] for n in runs}
for t in tasks:
    Lmin=min(len(arr(a)) for n in runs for a in data[n][t].values())
    for n in runs:
        ds=[disp(arr(a),Lmin) for a in data[n][t].values()]
        trunc[n]+=ds; per_task_trunc[n].append(np.mean(ds))
print("\n=== (2) task 내 최단 길이 truncate (길이통제) ===")
m2={n:np.mean(trunc[n]) for n in runs}
for n in runs: print(f"  {n:8s} {m2[n]:.4f} ± {np.std(trunc[n]):.4f}")
print(f"  fast/slow = {(m2['fast']/m2['slow']-1)*100:+.1f}% | slow vs base {(m2['slow']/m2['baseline']-1)*100:+.1f}% | fast vs base {(m2['fast']/m2['baseline']-1)*100:+.1f}%")

# ---- (3) 논문 방식: fast vs slow, task별 paired ----
ft=np.array([np.mean([disp(arr(a)) for a in data["fast"][t].values()]) for t in tasks])
st=np.array([np.mean([disp(arr(a)) for a in data["slow"][t].values()]) for t in tasks])
tstat,p=stats.ttest_rel(ft,st)
d=(ft-st).mean()/(ft-st).std(ddof=1)  # Cohen's dz (paired)
pct=np.mean((ft-st)/st)*100
print("\n=== (3) 논문 방식: fast vs slow, 10 task paired (full episode) ===")
print(f"  per-task fast: {np.round(ft,3)}")
print(f"  per-task slow: {np.round(st,3)}")
print(f"  fast가 slow보다 큰 task: {(ft>st).sum()}/10")
print(f"  평균 개선 (fast over slow) = +{pct:.1f}%  (논문 +27.73%)")
print(f"  paired t-test: t={tstat:.2f}, p={p:.4g}  (논문 p<0.001)")
print(f"  effect size (Cohen's dz) = {d:.2f}")

# ---- plots ----
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
for a_,(title,mm) in zip(ax[:2],[("(1) 전체 속도",m),("(2) task내 최단길이 truncate",m2)]):
    a_.bar(list(runs),[mm[n] for n in runs],color=["#7f8c8d","#27ae60","#8e44ad"])
    a_.set_title(title); a_.set_ylabel("mean step disp (=속도)"); a_.grid(alpha=.3,axis="y")
ax[2].scatter(np.zeros(10),st,color="#8e44ad",label="slow",zorder=3)
ax[2].scatter(np.ones(10),ft,color="#27ae60",label="fast",zorder=3)
for i in range(10): ax[2].plot([0,1],[st[i],ft[i]],color="gray",alpha=.5)
ax[2].set_xticks([0,1]); ax[2].set_xticklabels(["slow","fast"])
ax[2].set_title(f"(3) 논문방식 paired\n+{pct:.0f}% over slow, p={p:.3g}"); ax[2].set_ylabel("per-task mean disp"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/D_3way_5trial.png",dpi=130); plt.close()

import csv
with open(f"{OUT}/D_3way_5trial.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["comparison","baseline","fast","slow","fast/slow_%","note"])
    w.writerow(["1_full",round(m['baseline'],4),round(m['fast'],4),round(m['slow'],4),round((m['fast']/m['slow']-1)*100,1),"full episode"])
    w.writerow(["2_trunc_shortest",round(m2['baseline'],4),round(m2['fast'],4),round(m2['slow'],4),round((m2['fast']/m2['slow']-1)*100,1),"per-task min length"])
    w.writerow(["3_paper_paired","",round(ft.mean(),4),round(st.mean(),4),round(pct,1),f"paired t p={p:.3g} dz={d:.2f}"])
print("\nWROTE", f"{OUT}/D_3way_5trial.png", "+ D_3way_5trial.csv")
