"""발표(PPT)용 검출 결과 figure/table 생성 (영문 라벨, 단일 detector).

pathway_lstm_detector.py(--split-instruction --per-task --n-perm) 의 JSON 하나(기본 MLP)를 읽어:
  fig_detection_headline.png : (a) functional-CP bal-acc vs α  (b) decision-time AUROC + length baseline
  fig_generalization.png     : DiT seen vs unseen (일반화)
  fig_per_task.png           : DiT bal-acc per task (일반화 robustness)
  fig_per_instruction.png    : OpenDrawer left/right (instruction-confound 통제)
  ppt_tables.md              : 헤드라인/length/decision-time/per-task/per-instruction 표
검증된 JSON 수치만 사용(재학습/재추론 없음). matplotlib 만 필요.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"dit": "#1f6fb4", "vl": "#d1492e"}
SPLITCOL = {"seen": "#8a8a8a", "unseen": "#1f6fb4"}
ALPHAS = ["0.05", "0.10", "0.20", "0.30", "0.50"]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 10})


def _xy_cp(cp):
    xs = [float(a) for a in ALPHAS if a in cp and cp[a].get("bal_acc") is not None]
    ys = [cp[a]["bal_acc"] for a in ALPHAS if a in cp and cp[a].get("bal_acc") is not None]
    return xs, ys


def _xy_dt(dt):
    ts = sorted(int(t) for t in dt)
    return ts, [dt[str(t)]["auroc"] for t in ts]


def fig_headline(d, out_png, det):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # (a) functional-CP bal-acc vs α — DiT vs VL
    ax = axes[0]
    for pw in ("dit", "vl"):
        x, y = _xy_cp(d["pathways"][pw]["cp_unseen"])
        ax.plot(x, y, "-o", color=COL[pw], label=pw.upper())
    ax.axhline(0.5, color="k", ls=":", lw=0.9, label="chance")
    sig = d["pathways"]["dit"].get("sig_unseen") or {}
    if sig:
        ci = sig.get("ci95"); txt = f"DiT @α={sig['alpha']}: bal-acc={sig['bal_acc']}"
        if ci:
            txt += f"\n95% CI [{ci[0]}, {ci[1]}]"
        if sig.get("p_value") is not None:
            txt += f"\nperm-null p={sig['p_value']}"
        ax.text(0.04, 0.06, txt, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="#1f6fb4", alpha=0.9))
    ax.set_xlabel("target FPR  α"); ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.45, 1.0); ax.set_title("(a) Functional-CP detection (unseen)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    # (b) decision-time AUROC + length baseline (length-fair 증거) — DiT 집중
    ax = axes[1]
    x, y = _xy_dt(d["pathways"]["dit"]["decision_time_unseen"])
    ax.plot(x, y, "-o", color=COL["dit"], label="DiT activation (causal)")
    lb = d.get("length_baseline", {}).get("unseen")  # {t_d: auroc_float}
    if lb:
        ts = sorted(int(t) for t in lb)
        ax.plot(ts, [lb[str(t)] for t in ts], "--^", color="#7a7a7a",
                label="length, total (non-causal oracle)")
    ax.axhline(0.5, color="k", ls=":", lw=0.9, label="length, causal = chance")
    ax.set_xlabel("decision time  t_d  (inference steps, causal)")
    ax.set_ylabel("AUROC (failure = positive)")
    ax.set_ylim(0.45, 1.0); ax.set_title("(b) Online & length-fair")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f"Online success/failure detection from VLA activations — {det.upper()} "
                 "(unseen, n_fail=93 / n_succ=107)", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


def fig_onepager(d, out_png):
    """1페이지용: (좌) DiT vs VL 검출(존재·pathway), (우) DiT seen vs unseen(일반화). bal-acc vs α 통일."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # (좌) existence + pathway: DiT vs VL, unseen
    ax = axes[0]
    for pw in ("dit", "vl"):
        x, y = _xy_cp(d["pathways"][pw]["cp_unseen"])
        ax.plot(x, y, "-o", color=COL[pw], label=pw.upper(), lw=2.2, ms=7)
    ax.axhline(0.5, color="k", ls=":", lw=0.9, label="chance")
    sig = d["pathways"]["dit"].get("sig_unseen") or {}
    if sig:
        ci = sig.get("ci95")
        txt = f"DiT @α={sig['alpha']}: bal-acc={sig['bal_acc']}"
        if ci:
            txt += f"\n95% CI [{ci[0]}, {ci[1]}]"
        if sig.get("p_value") is not None:
            txt += f"\nperm-null p={sig['p_value']}"
        ax.text(0.04, 0.06, txt, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec=COL["dit"], alpha=0.9))
    ax.set_xlabel("target FPR  α"); ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.45, 1.0); ax.set_title("(a) Failure signal in DiT activation (unseen)\nDiT ≫ VL")
    ax.legend(loc="center right"); ax.grid(alpha=0.3)
    # (우) generalization: DiT seen vs unseen
    ax = axes[1]
    for split in ("seen", "unseen"):
        x, y = _xy_cp(d["pathways"]["dit"][f"cp_{split}"])
        ax.plot(x, y, "-o", color=SPLITCOL[split], label=f"{split}-test", lw=2.2, ms=7)
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xlabel("target FPR  α"); ax.set_ylabel("balanced accuracy (DiT)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(b) Generalizes to unseen tasks\n(trained on 8, held-out 2; no drop)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.suptitle("Failure is decodable from DiT activations and generalizes to unseen tasks "
                 "(unseen n_fail=93 / n_succ=107)", fontsize=14, y=1.03)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


def fig_generalization(d, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for split in ("seen", "unseen"):
        x, y = _xy_cp(d["pathways"]["dit"][f"cp_{split}"])
        ax.plot(x, y, "-o", color=SPLITCOL[split], label=f"{split}-test")
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xlabel("target FPR  α"); ax.set_ylabel("balanced accuracy (DiT)")
    ax.set_ylim(0.45, 1.0); ax.set_title("(a) Functional-CP: seen vs unseen")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    ax = axes[1]
    for split in ("seen", "unseen"):
        x, y = _xy_dt(d["pathways"]["dit"][f"decision_time_{split}"])
        ax.plot(x, y, "-o", color=SPLITCOL[split], label=f"{split}-test")
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xlabel("decision time  t_d"); ax.set_ylabel("AUROC (DiT)")
    ax.set_ylim(0.45, 1.0); ax.set_title("(b) Decision-time AUROC: seen vs unseen")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.suptitle("DiT failure signal generalizes to unseen tasks (no degradation seen→unseen)",
                 fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


def fig_per_task(d, out_png, alpha="0.30"):
    pt = d.get("per_task", {}).get("pathways", {}).get("dit")
    if not pt:
        return
    rows = []  # (task, split, bal_acc)
    for split in ("unseen", "seen"):
        for task, cp in sorted(pt.get(split, {}).items()):
            c = cp.get(alpha)
            if c and c.get("bal_acc") is not None:
                rows.append((task, split, c["bal_acc"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(rows) + 3), 5))
    x = np.arange(len(rows))
    cols = [SPLITCOL[s] for _, s, _ in rows]
    bars = ax.bar(x, [v for _, _, v in rows], color=cols, edgecolor="k", linewidth=0.5)
    for b, (_, _, v) in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([t for t, _, _ in rows], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("balanced accuracy (DiT)"); ax.set_ylim(0.4, 1.05)
    ax.set_title(f"Per-task detection (DiT, α={alpha}) — not driven by one task")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPLITCOL[s]) for s in ("unseen", "seen")]
    ax.legend(handles, ["unseen", "seen"], loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


def fig_per_instruction(d, out_png, alpha="0.30"):
    pi = d.get("per_instruction", {}).get("pathways")
    if not pi:
        return
    instrs = ["Open the left drawer.", "Open the right drawer."]
    instrs = [i for i in instrs if i in pi.get("dit", {}).get("unseen", {})]
    if not instrs:
        return
    x = np.arange(len(instrs)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, pw in enumerate(("dit", "vl")):
        vals = [pi[pw]["unseen"].get(ins, {}).get(alpha, {}).get("bal_acc", np.nan) for ins in instrs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=pw.upper(), color=COL[pw],
                      edgecolor="k", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["left drawer", "right drawer"])
    ax.set_ylabel("balanced accuracy"); ax.set_ylim(0.4, 1.05)
    ax.set_title(f"Per-instruction (OpenDrawer, unseen, α={alpha})\nDiT holds within instruction; VL collapses")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


def write_tables(d, out_md, det):
    L = [f"# 발표용 검출 결과 표 ({det.upper()})", "",
         "## (A) Pooled 헤드라인 — DiT, unseen (n_fail=93 / n_succ=107)", "",
         "| α | TPR | FPR | bal-acc | norm. T-det |", "|---|---|---|---|---|"]
    cp = d["pathways"]["dit"]["cp_unseen"]
    for a in ALPHAS:
        if a in cp:
            c = cp[a]
            L.append(f"| {a} | {c['tpr']} | {c['fpr']} | **{c['bal_acc']}** | {c['mean_tdet_fired']} |")
    sig = d["pathways"]["dit"].get("sig_unseen")
    if sig:
        L += ["", f"유의성 @α={sig['alpha']}: bal-acc **{sig['bal_acc']}**, 95% CI {sig.get('ci95')}, "
              f"permutation-null mean {sig.get('null_mean')}, **p={sig.get('p_value')}**."]

    lb = d.get("length_baseline", {})
    if lb:
        L += ["", "## (B) Length-fair: length-only baseline vs activation (decision-time AUROC)", "",
              lb.get("note", ""), "",
              "| feature | " + " | ".join(f"t_d={t}" for t in (3, 5, 8, 11, 15, 20)) + " |",
              "|" + "---|" * 7]
        dtd = d["pathways"]["dit"]["decision_time_unseen"]
        L.append("| DiT activation (causal) | " + " | ".join(
            f"{dtd.get(str(t),{}).get('auroc', float('nan')):.3f}" for t in (3, 5, 8, 11, 15, 20)) + " |")
        un = lb.get("unseen", {})
        L.append("| length total (non-causal oracle) | " + " | ".join(
            f"{un.get(str(t), float('nan')):.3f}" if str(t) in un else "—" for t in (3, 5, 8, 11, 15, 20)) + " |")
        L.append("| length causal | " + " | ".join("0.500" for _ in (3, 5, 8, 11, 15, 20)) + " |")

    pt = d.get("per_task", {}).get("pathways", {}).get("dit")
    if pt:
        L += ["", "## (C) Per-task (DiT, α=0.30) — 일반화 robustness", "",
              "| task | split | TPR | FPR | bal-acc | n_fail | n_succ |", "|---|---|---|---|---|---|---|"]
        for split in ("unseen", "seen"):
            for task, cp2 in sorted(pt.get(split, {}).items()):
                c = cp2.get("0.30")
                if c:
                    L.append(f"| {task} | {split} | {c['tpr']} | {c['fpr']} | **{c['bal_acc']}** | "
                             f"{c['n_fail']} | {c['n_succ']} |")

    pi = d.get("per_instruction", {}).get("pathways")
    if pi:
        L += ["", "## (D) Per-instruction (OpenDrawer, unseen, α=0.30) — instruction-confound 통제", "",
              "| instruction | pathway | TPR | FPR | bal-acc | n_fail | n_succ |",
              "|---|---|---|---|---|---|---|"]
        for ins in ["Open the left drawer.", "Open the right drawer."]:
            for pw in ("dit", "vl"):
                c = pi.get(pw, {}).get("unseen", {}).get(ins, {}).get("0.30")
                if c:
                    L.append(f"| {ins} | {pw.upper()} | {c['tpr']} | {c['fpr']} | {c['bal_acc']} | "
                             f"{c['n_fail']} | {c['n_succ']} |")
    out_md.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="analysis 디렉터리(pathway_instr_* 상위)")
    ap.add_argument("--detector", default="mlp", choices=("mlp", "lstm"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "ppt_figures"
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads((run_dir / f"pathway_instr_{args.detector}" / "pathway_lstm_detector.json").read_text())
    fig_onepager(d, out / "fig_onepager.png")
    fig_headline(d, out / "fig_detection_headline.png", args.detector)
    fig_generalization(d, out / "fig_generalization.png")
    fig_per_task(d, out / "fig_per_task.png")
    fig_per_instruction(d, out / "fig_per_instruction.png")
    write_tables(d, out / "ppt_tables.md", args.detector)
    print(f"[done] -> {out}/")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
