"""GR00T N1.5 RoboCasa — DiT succ/fail 분리가 진짜인가 confound(길이/phase/task)인가?

질문: t-SNE/LDA plot 에서 DiT latent 가 특정 layer 에서 succ/fail(그리고 task)로
갈라져 *보인다*. 이게 진짜 succ/fail 신호인가, 아니면 action-phase·rollout 길이·task
confound 인가? — DiT layer/phase 별로 판정한다.

이 스크립트는 ``phase_separation.py`` 의 검증된 primitive 를 **import 재사용**한다
(load_rollout, pool_denoise, equal_budget_pool, rank_auroc, loo_auroc,
perm_null_upper, _lda_project, phase_records). phase_separation.py 는 수정하지 않는다.

분석 4가지:
  1. 결정적 검정 — within-cell, within-phase succ/fail 분리 (layer별): loo_auroc +
     length-only baseline(phase record 수) + perm_null_upper. 판정:
       GREEN(genuine)   = |auroc-.5| > null margin  AND  |auroc-.5| > |length-.5| + 0.07
       LENGTH-CONFOUND  = null 은 넘지만 length 를 0.07 이상 못 이김 (둘 다 높음)
       NO-SIGNAL        = null 못 넘음 (chance 근처)
     insert-settle 는 success-only → succ/fail 비교 skip.
  2. 길이 분리 — transport early-window: budget = 첫 3~4 transport record (succ·fail
     dwell 동일). 유지되면 genuine early-divergence, chance 로 붕괴하면 length 였다.
  3. task vs outcome — (a) within-cell 가 task 제거함을 명시. (b) cross-cell POOLED
     succ/fail AUROC(layer별) — task 섞임. (c) TASK 분류 AUROC(layer별): bread-succ
     vs onion-succ (outcome=succ 고정) → 각 layer 가 TASK 를 얼마나 인코딩하나.
  4. layer 구조 귀속 — cell·layer별 record 전체 pool 에서 3 factor 각각 독립으로
     supervised 1D 분리(_lda_project→rank_auroc, in-sample 서술적 max-분리도):
     PHASE(reach vs transport), OUTCOME(succ vs fail), NORM-STEP(early vs late).
     각 layer 에서 어느 factor 가 지배적인지 → plot 이 왜 그렇게 보이는지 설명.

저표본(fail 3~4/cell) → LOO + permutation null, null 상단 ~0.9 검정력 낮음을 정직히 병기.
READ-ONLY on raw_rollouts. 출력만 analysis/dit_succfail_investigation/ 아래.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase_separation as ps  # noqa: E402  (검증된 primitive 재사용)

REPO = Path(__file__).resolve().parents[5]
RUN_DIR = REPO / "outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/raw_rollouts"
OUT_DIR = REPO / "outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/analysis/dit_succfail_investigation"

CELLS = {
    "bread": RUN_DIR / "PickPlaceCounterToCabinet/ppcc_bread",
    "onion": RUN_DIR / "PickPlaceCounterToStove/ppcs_onion",
}
GREEN_MARGIN = 0.07  # |auroc-.5| 가 |length-.5| 를 이 이상 이겨야 genuine 판정
N_PERM = 300


# --------------------------------------------------------------------------- #
# 공통 헬퍼
# --------------------------------------------------------------------------- #
def load_cell(cell_dir: Path) -> list[dict]:
    return [ps.load_rollout(p) for p in sorted(cell_dir.glob("*.pkl"))]


def layer_setup(rolls: list[dict]):
    cap = rolls[0]["capture_layers"]  # [0,2,4,8,10,12,15]
    keys = list(range(rolls[0]["dit"].shape[1])) + (["VL"] if rolls[0]["vl"] is not None else [])
    labels = {str(i): f"DiT-L{cap[i]}" for i in range(len(cap))}
    labels["VL"] = "VL"
    return cap, keys, labels


def pooled_matrix(rolls, phase, layer_key, budget):
    """budget record 이상 phase 에 존재하는 rollout 만: equal-budget pool + y(success)."""
    X, y = [], []
    for r in rolls:
        recs = ps.phase_records(r, phase, layer_key)
        if len(recs) < budget:
            continue
        X.append(ps.equal_budget_pool(recs, budget))
        y.append(r["success"])
    if not X:
        return None, None
    return np.stack(X, axis=0), np.asarray(y)


def present_counts(rolls, phase, layer_key0):
    """phase 에 record>=1 인 rollout 의 (record수, success) 리스트."""
    out = []
    for r in rolls:
        recs = ps.phase_records(r, phase, layer_key0)
        if len(recs) > 0:
            out.append((len(recs), r["success"]))
    return out


def verdict(auroc, null_hi, length_auroc):
    sep = abs(auroc - 0.5)
    lensep = abs(length_auroc - 0.5)
    null_margin = null_hi - 0.5
    if sep <= null_margin:
        return "NO-SIGNAL"
    if sep > lensep + GREEN_MARGIN:
        return "GENUINE"
    return "LENGTH-CONFOUNDED"


# --------------------------------------------------------------------------- #
# Analysis 1 — 결정적 검정: within-cell, within-phase, layer별
# --------------------------------------------------------------------------- #
def analysis1(cells_rolls, layer_keys, labels, phases=("reach-to-object", "transport")):
    res = {}
    for cid, rolls in cells_rolls.items():
        res[cid] = {}
        for phase in phases:
            counts = present_counts(rolls, phase, layer_keys[0])
            ys = np.array([s for _, s in counts])
            n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
            entry = {"n_succ": n1, "n_fail": n0,
                     "record_counts": {"succ": [c for c, s in counts if s],
                                       "fail": [c for c, s in counts if not s]}}
            if n1 < 3 or n0 < 3:
                entry["status"] = f"skip: succ={n1} fail={n0} < 3"
                res[cid][phase] = entry
                continue
            budget = min(c for c, _ in counts)
            entry["budget"] = int(budget)
            lens = np.array([c for c, _ in counts], dtype=np.float64)
            length_auroc = ps.rank_auroc(lens, ys)
            entry["length_auroc"] = length_auroc
            layers = {}
            for lk in layer_keys:
                X, y = pooled_matrix(rolls, phase, lk, budget)
                if X is None:
                    continue
                a = ps.loo_auroc(X, y)
                if a is None:
                    continue
                null_hi = ps.perm_null_upper(X, y, n_perm=N_PERM)
                layers[str(lk)] = {"label": labels[str(lk)], "auroc": a,
                                   "null95_upper": null_hi,
                                   "verdict": verdict(a, null_hi, length_auroc)}
            entry["layers"] = layers
            res[cid][phase] = entry
    return res


# --------------------------------------------------------------------------- #
# Analysis 2 — transport early-window (dwell-matched)
# --------------------------------------------------------------------------- #
def analysis2(cells_rolls, layer_keys, labels, budgets=(3, 4)):
    res = {}
    for cid, rolls in cells_rolls.items():
        res[cid] = {}
        for budget in budgets:
            # phase='transport' 에서 budget record 이상인 rollout 만
            counts = present_counts(rolls, "transport", layer_keys[0])
            ys_full = np.array([s for c, s in counts if c >= budget])
            n1 = int((ys_full == 1).sum())
            n0 = int((ys_full == 0).sum())
            entry = {"budget": budget, "n_succ": n1, "n_fail": n0}
            if n1 < 3 or n0 < 3:
                entry["status"] = f"skip: succ={n1} fail={n0} < 3 at budget={budget}"
                res[cid][f"budget{budget}"] = entry
                continue
            layers = {}
            for lk in layer_keys:
                X, y = pooled_matrix(rolls, "transport", lk, budget)
                if X is None:
                    continue
                a = ps.loo_auroc(X, y)
                if a is None:
                    continue
                null_hi = ps.perm_null_upper(X, y, n_perm=N_PERM)
                # early-window 에서 pooled window 길이는 succ·fail 동일(=budget).
                # length_auroc(full count) 는 여전히 ~ceiling 이므로 참고용만.
                layers[str(lk)] = {"label": labels[str(lk)], "auroc": a,
                                   "null95_upper": null_hi,
                                   "clears_null": bool(abs(a - 0.5) > null_hi - 0.5)}
            entry["layers"] = layers
            res[cid][f"budget{budget}"] = entry
    return res


# --------------------------------------------------------------------------- #
# Analysis 3 — task vs outcome
# --------------------------------------------------------------------------- #
def analysis3(cells_rolls, layer_keys, labels):
    res = {"note": "within-cell(A1)=고정 seed/instruction → task 이미 제거됨. 아래는 대조군."}

    # (b) cross-cell POOLED succ/fail (phase별) — task 섞임
    pooled = {}
    all_rolls = cells_rolls["bread"] + cells_rolls["onion"]
    for phase in ("reach-to-object", "transport"):
        counts = present_counts(all_rolls, phase, layer_keys[0])
        ys = np.array([s for _, s in counts])
        n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
        if n1 < 3 or n0 < 3:
            pooled[phase] = {"status": f"skip succ={n1} fail={n0}"}
            continue
        budget = min(c for c, _ in counts)
        lens = np.array([c for c, _ in counts], dtype=np.float64)
        length_auroc = ps.rank_auroc(lens, ys)
        layers = {}
        for lk in layer_keys:
            X, y = pooled_matrix(all_rolls, phase, lk, budget)
            if X is None:
                continue
            a = ps.loo_auroc(X, y)
            if a is None:
                continue
            layers[str(lk)] = {"label": labels[str(lk)], "auroc": a}
        pooled[phase] = {"n_succ": n1, "n_fail": n0, "budget": int(budget),
                         "length_auroc": length_auroc, "layers": layers}
    res["cross_cell_pooled_succfail"] = pooled

    # (c) TASK 분류: bread-succ vs onion-succ (outcome=succ 고정), phase=reach
    #     reach 는 두 cell 성공 rollout 전부에 존재 → 표본 넉넉.
    task_rolls = ([r for r in cells_rolls["bread"] if r["success"]]
                  + [r for r in cells_rolls["onion"] if r["success"]])
    task_y = np.array([1] * sum(r["success"] for r in cells_rolls["bread"])
                      + [0] * sum(r["success"] for r in cells_rolls["onion"]))  # bread=1 onion=0
    phase = "reach-to-object"
    counts = [len(ps.phase_records(r, phase, layer_keys[0])) for r in task_rolls]
    budget = min(counts)
    lens = np.array(counts, dtype=np.float64)
    task_length_auroc = ps.rank_auroc(lens, task_y)
    layers = {}
    for lk in layer_keys:
        X = []
        ok = True
        for r in task_rolls:
            recs = ps.phase_records(r, phase, lk)
            if len(recs) < budget:
                ok = False
                break
            X.append(ps.equal_budget_pool(recs, budget))
        if not ok:
            continue
        X = np.stack(X, axis=0)
        a = ps.loo_auroc(X, task_y)
        if a is None:
            continue
        null_hi = ps.perm_null_upper(X, task_y, n_perm=N_PERM)
        layers[str(lk)] = {"label": labels[str(lk)], "auroc": a, "null95_upper": null_hi}
    res["task_classification_reach"] = {
        "desc": "bread-succ(1) vs onion-succ(0), reach-to-object, equal-budget",
        "n_bread": int((task_y == 1).sum()), "n_onion": int((task_y == 0).sum()),
        "budget": int(budget), "length_auroc": task_length_auroc, "layers": layers}
    return res


# --------------------------------------------------------------------------- #
# Analysis 4 — layer 구조 귀속 (record-level, in-sample 서술적)
# --------------------------------------------------------------------------- #
def analysis4(cells_rolls, layer_keys, labels):
    res = {"note": "record-level, in-sample _lda_project→rank_auroc = 서술적 max 선형분리도"
                   "(일반화 아님). 각 layer 에서 어느 factor 가 지배적인지 비교용."}
    for cid, rolls in cells_rolls.items():
        # record 단위 factor 라벨 수집
        outcome, phase_lab, norm_step = [], [], []
        for r in rolls:
            n = r["length"]
            for i, ph in enumerate(r["phases"]):
                outcome.append(r["success"])
                phase_lab.append(ph)
                norm_step.append(i / max(1, n - 1))
        outcome = np.asarray(outcome)
        phase_lab = np.asarray(phase_lab)
        norm_step = np.asarray(norm_step)
        # phase: reach vs transport (insert-settle 제외)
        pmask = np.isin(phase_lab, ["reach-to-object", "transport"])
        y_phase = (phase_lab[pmask] == "transport").astype(int)
        # step: early vs late (전체 record median split)
        y_step = (norm_step >= 0.5).astype(int)

        cell_out = {}
        for lk in layer_keys:
            # record-level X for this layer
            rows = []
            for r in rolls:
                if lk == "VL":
                    if r["vl"] is None:
                        rows = None
                        break
                    rows.append(r["vl"])
                else:
                    rows.append(r["dit"][:, lk, :])
            if rows is None:
                continue
            X = np.concatenate(rows, axis=0)  # [Nrec, D]
            def sep(Xa, ya):
                if len(np.unique(ya)) < 2:
                    return None
                s = ps._lda_project(Xa, ya, Xa)
                return ps.rank_auroc(s, ya)
            cell_out[str(lk)] = {
                "label": labels[str(lk)],
                "phase_reach_vs_transport": sep(X[pmask], y_phase),
                "outcome_succ_vs_fail": sep(X, outcome),
                "step_early_vs_late": sep(X, y_step),
            }
        res[cid] = {"n_records": int(len(outcome)), "layers": cell_out}
    return res


# --------------------------------------------------------------------------- #
# 플롯
# --------------------------------------------------------------------------- #
def make_plots(a1, a2, a3, a4, cap, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab_order = [str(i) for i in range(len(cap))] + ["VL"]
    xlabels = [f"L{c}" for c in cap] + ["VL"]

    # A1: per (cell,phase) auroc vs length, colored by verdict
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (cid, phase) in zip(axes.ravel(),
                                [("bread", "reach-to-object"), ("bread", "transport"),
                                 ("onion", "reach-to-object"), ("onion", "transport")]):
        e = a1.get(cid, {}).get(phase, {})
        if "layers" not in e:
            ax.set_title(f"{cid} / {phase}\n{e.get('status','')}")
            ax.axis("off")
            continue
        vals, cols = [], []
        cmap = {"GENUINE": "tab:green", "LENGTH-CONFOUNDED": "tab:orange", "NO-SIGNAL": "tab:gray"}
        for lk in lab_order:
            v = e["layers"].get(lk)
            vals.append(v["auroc"] if v else np.nan)
            cols.append(cmap.get(v["verdict"], "k") if v else "k")
        xs = np.arange(len(lab_order))
        ax.bar(xs, vals, color=cols)
        ax.axhline(0.5, color="k", lw=0.6)
        la = e.get("length_auroc", 0.5)
        ax.axhline(la, color="red", ls="--", lw=1.2, label=f"length_auroc={la:.2f}")
        # null band (per-layer varies; draw mean)
        nulls = [e["layers"][lk]["null95_upper"] for lk in lab_order if e["layers"].get(lk)]
        if nulls:
            ax.axhline(float(np.mean(nulls)), color="purple", ls=":", lw=1, label="mean null95")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{cid} / {phase}  (succ={e['n_succ']} fail={e['n_fail']} budget={e.get('budget')})",
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("A1: within-cell within-phase succ/fail LOO-AUROC (green=genuine, orange=length, gray=no-signal)")
    fig.tight_layout()
    fig.savefig(out_dir / "A1_within_phase_verdict.png", dpi=110)
    plt.close(fig)

    # A2: transport early-window (bread) auroc vs budget per layer
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, cid in zip(axes, ["bread", "onion"]):
        for lk in lab_order:
            bs, ys = [], []
            for bkey in sorted(a2.get(cid, {}).keys()):
                e = a2[cid][bkey]
                if "layers" not in e or lk not in e["layers"]:
                    continue
                bs.append(e["budget"]); ys.append(e["layers"][lk]["auroc"])
            if bs:
                ax.plot(bs, ys, marker="o", label=(f"L{cap[int(lk)]}" if lk != "VL" else "VL"))
        ax.axhline(0.5, color="k", lw=0.6)
        ax.set_xlabel("transport early-window budget (records)")
        ax.set_ylabel("succ/fail LOO-AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"A2 {cid}: dwell-matched early window")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "A2_transport_early_window.png", dpi=110)
    plt.close(fig)

    # A3: task classification vs outcome per layer
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tc = a3["task_classification_reach"]["layers"]
    xs = np.arange(len(lab_order))
    task_vals = [tc[lk]["auroc"] if lk in tc else np.nan for lk in lab_order]
    ax.bar(xs - 0.2, [abs(v - 0.5) + 0.5 if v == v else np.nan for v in task_vals],
           width=0.4, label="TASK (bread vs onion, |·| folded)", color="tab:blue")
    # cross-cell pooled succ/fail (reach)
    cc = a3["cross_cell_pooled_succfail"].get("reach-to-object", {}).get("layers", {})
    cc_vals = [cc[lk]["auroc"] if lk in cc else np.nan for lk in lab_order]
    ax.bar(xs + 0.2, cc_vals, width=0.4, label="cross-cell succ/fail (reach)", color="tab:orange")
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_title("A3: TASK encoding vs cross-cell outcome per layer (reach)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "A3_task_vs_outcome.png", dpi=110)
    plt.close(fig)

    # A4: factor domination heatmap per cell
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    factors = ["phase_reach_vs_transport", "outcome_succ_vs_fail", "step_early_vs_late"]
    fnames = ["PHASE", "OUTCOME", "STEP"]
    for ax, cid in zip(axes, ["bread", "onion"]):
        cell = a4[cid]["layers"]
        M = np.full((len(factors), len(lab_order)), np.nan)
        for j, lk in enumerate(lab_order):
            if lk not in cell:
                continue
            for i, f in enumerate(factors):
                v = cell[lk][f]
                if v is not None:
                    M[i, j] = abs(v - 0.5) + 0.5  # folded
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(lab_order))); ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
        ax.set_yticks(range(len(factors))); ax.set_yticklabels(fnames)
        for i in range(len(factors)):
            for j in range(len(lab_order)):
                if M[i, j] == M[i, j]:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.8 else "k", fontsize=7)
        ax.set_title(f"A4 {cid}: record-level factor separability (|AUROC-.5| folded)")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "A4_factor_domination.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells_rolls = {cid: load_cell(d) for cid, d in CELLS.items()}
    cap, layer_keys, labels = layer_setup(cells_rolls["bread"])

    print("[A1] within-cell within-phase succ/fail ...")
    a1 = analysis1(cells_rolls, layer_keys, labels)
    print("[A2] transport early-window ...")
    a2 = analysis2(cells_rolls, layer_keys, labels)
    print("[A3] task vs outcome ...")
    a3 = analysis3(cells_rolls, layer_keys, labels)
    print("[A4] record-level factor domination ...")
    a4 = analysis4(cells_rolls, layer_keys, labels)

    results = {
        "run_dir": str(RUN_DIR),
        "capture_layers": cap,
        "layer_labels": labels,
        "green_margin": GREEN_MARGIN,
        "n_perm": N_PERM,
        "A1_within_phase": a1,
        "A2_transport_early_window": a2,
        "A3_task_vs_outcome": a3,
        "A4_factor_domination": a4,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[done] results -> {out_dir/'results.json'}")

    if not args.no_plots:
        make_plots(a1, a2, a3, a4, cap, out_dir)
        print(f"[done] plots -> {out_dir}")

    _print_summary(results)


def _print_summary(r):
    lab = r["layer_labels"]
    order = [str(i) for i in range(len(r["capture_layers"]))] + ["VL"]
    print("\n" + "=" * 70)
    print("A1 VERDICT TABLE")
    for cid, phases in r["A1_within_phase"].items():
        for phase, e in phases.items():
            if "layers" not in e:
                print(f"  {cid}/{phase}: {e.get('status')}")
                continue
            print(f"  {cid}/{phase}  succ={e['n_succ']} fail={e['n_fail']} "
                  f"budget={e['budget']} length_auroc={e['length_auroc']:.3f}")
            for lk in order:
                v = e["layers"].get(lk)
                if not v:
                    continue
                print(f"      {v['label']:8s} auroc={v['auroc']:.3f} "
                      f"null95={v['null95_upper']:.3f}  {v['verdict']}")
    print("\nA2 EARLY-WINDOW (transport, dwell-matched)")
    for cid, bud in r["A2_transport_early_window"].items():
        for bkey, e in bud.items():
            if "layers" not in e:
                print(f"  {cid}/{bkey}: {e.get('status')}")
                continue
            best = max(e["layers"].items(), key=lambda kv: abs(kv[1]["auroc"] - 0.5))
            print(f"  {cid}/budget{e['budget']} succ={e['n_succ']} fail={e['n_fail']} "
                  f"best={best[1]['label']} auroc={best[1]['auroc']:.3f} "
                  f"clears_null={best[1]['clears_null']}")
    print("\nA3 TASK classification (reach)")
    tc = r["A3_task_vs_outcome"]["task_classification_reach"]
    print(f"  budget={tc['budget']} length_auroc={tc['length_auroc']:.3f}")
    for lk in order:
        v = tc["layers"].get(lk)
        if v:
            print(f"      {v['label']:8s} task_auroc={v['auroc']:.3f} null95={v['null95_upper']:.3f}")


if __name__ == "__main__":
    main()
