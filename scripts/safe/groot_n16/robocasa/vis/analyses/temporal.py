"""temporal — quantitative comparison of temporal aggregation schemes.

Per scheme (pp/tp/tf/pad/win1-4) × time-index, full-dim Mahalanobis metrics:
  M1 succ/fail silhouette, M2 fail/succ centroid-spread ratio, M3 per-point a-vs-b.
Whitening by task ('whitenTask') and task_succ_fail ('whitenTaskFail'); x-axis as
absolute frame (byframe) and normalized time (bynorm).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import geometry, io, metrics
from core.schemes import SCHEME_LABEL, SCHEMES, iter_scheme
from analyses._common import add_common

NAME = "temporal"
HELP = "quantitative succ/fail separation & failure-zone vs time, per aggregation scheme"

FAIL_LABEL = 18
WHITEN_LABELS = ("task", "task_succ_fail")
WL_TAG = {"task": "whitenTask", "task_succ_fail": "whitenTaskFail"}


def add_args(p):
    add_common(p)
    p.add_argument("--t-d", type=int, default=16)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--min-class", type=int, default=8)
    p.add_argument("--sil-sample", type=int, default=1500)


def _r(v):
    return "-" if v is None else round(v, 3)


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    out_dir = args.out_root / "temporal_schemes"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {wl: {s: [] for s in SCHEMES} for wl in WHITEN_LABELS}
    for scheme in SCHEMES:
        print(f"== {scheme} ({SCHEME_LABEL[scheme]}) ==")
        for snap in iter_scheme(rollouts, scheme, t_d=args.t_d):
            X, succ, task = snap["X"], snap["succ"], snap["task"]
            task_sf = np.where(succ == 1, task, FAIL_LABEL)
            both = min(snap["n_succ"], snap["n_fail"]) >= 2
            for wl in WHITEN_LABELS:
                lab = task if wl == "task" else task_sf
                Xw = geometry.whiten_by(X, lab, reg=args.cov_reg)
                sil = metrics.silhouette_safe(Xw, succ, sample_size=args.sil_sample, random_state=args.seed) if both else None
                avsb = None
                if both:
                    own, other = metrics.per_point_ab(Xw, succ, metric="euclidean")
                    avsb = metrics.silhouette_proxy_mean(own, other)
                ss, sf, ratio, ntask = metrics.centroid_spread_ratio(Xw, succ, task, min_class=args.min_class)
                results[wl][scheme].append({
                    "x_native": snap["x_native"], "x_norm": snap["x_norm"], "x_abs": snap["x_abs"],
                    "n": snap["n"], "n_succ": snap["n_succ"], "n_fail": snap["n_fail"],
                    "sil_maha_sf": sil, "avsb_proxy": avsb, "spread_succ": ss, "spread_fail": sf,
                    "failzone_ratio": ratio, "n_task_used": ntask})

    cmap = mpl.cm.get_cmap("tab10", len(SCHEMES))
    colors = {s: cmap(i) for i, s in enumerate(SCHEMES)}

    for wl in WHITEN_LABELS:
        tag = WL_TAG[wl]; res = results[wl]

        def lineplot(key, ylabel, title, hline, stem, ylim=None):
            for xkey, xlabel, suffix, vline in (
                ("x_abs", "absolute frame (inference step); tp=nominal, truncated stop at 16", "byframe", 16),
                ("x_norm", "normalized time within each scheme's horizon", "bynorm", None),
            ):
                fig, ax = plt.subplots(figsize=(12, 6.2))
                for s in SCHEMES:
                    xs = [r[xkey] for r in res[s] if r[key] is not None]
                    ys = [r[key] for r in res[s] if r[key] is not None]
                    ls = "--" if s in ("tp", "pp") and xkey == "x_abs" else "-"
                    if xs:
                        ax.plot(xs, ys, "o" + ls, ms=4, color=colors[s], label=SCHEME_LABEL[s])
                if hline is not None:
                    ax.axhline(hline, color="grey", ls="--", lw=1)
                if vline is not None:
                    ax.axvline(vline, color="grey", ls=":", lw=0.8)
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                ax.set_title(f"{title}  [{tag}, full-dim, {suffix}]"); ax.legend(fontsize=8)
                if ylim:
                    ax.set_ylim(*ylim)
                fig.tight_layout(); fig.savefig(out_dir / f"{stem}_{suffix}_{tag}.png", dpi=140); plt.close(fig)

        lineplot("sil_maha_sf", "Mahalanobis silhouette (succ/fail)",
                 "M1: success/fail separation vs time", 0.0, "temporal_compare_separation")
        lineplot("failzone_ratio", "fail/succ cross-task centroid-spread ratio",
                 "M2: shared failure-zone vs time (ratio<1 => failures converge)", 1.0,
                 "temporal_compare_failzone", ylim=(0.4, 1.3))
        lineplot("avsb_proxy", "centroid-proxy silhouette mean((b-a)/max(a,b))",
                 "M3: per-point a-vs-b succ/fail", 0.0, "temporal_compare_avsb")
        lineplot("n_succ", "# success rollouts present",
                 "success count vs time", None, "temporal_compare_counts")

        summary = []
        for s in SCHEMES:
            rows = res[s]
            sils = [r["sil_maha_sf"] for r in rows if r["sil_maha_sf"] is not None]
            avsbs = [r["avsb_proxy"] for r in rows if r["avsb_proxy"] is not None]
            ratios = [r["failzone_ratio"] for r in rows if r["failzone_ratio"] is not None]
            summary.append({"scheme": s, "label": SCHEME_LABEL[s], "n_index": len(rows),
                            "peak_silhouette": max(sils) if sils else None,
                            "mean_silhouette": float(np.mean(sils)) if sils else None,
                            "peak_avsb_proxy": max(avsbs) if avsbs else None,
                            "min_failzone_ratio": min(ratios) if ratios else None,
                            "final_failzone_ratio": ratios[-1] if ratios else None})
        json.dump({"whiten_label": wl, "full_dim": True, "t_d": args.t_d, "summary": summary, "results": res},
                  open(out_dir / f"temporal_compare_summary_{tag}.json", "w"), indent=2)
        with (out_dir / f"temporal_compare_summary_{tag}.tsv").open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["scheme", "n_index", "peak_silhouette", "mean_silhouette",
                        "peak_avsb_proxy", "min_failzone_ratio", "final_failzone_ratio"])
            for r in summary:
                w.writerow([r["scheme"], r["n_index"], _r(r["peak_silhouette"]), _r(r["mean_silhouette"]),
                            _r(r["peak_avsb_proxy"]), _r(r["min_failzone_ratio"]), _r(r["final_failzone_ratio"])])
        print(f"  [{tag}] wrote temporal_compare_*_{tag}.png + summary")
