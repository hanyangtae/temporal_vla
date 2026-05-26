#!/usr/bin/env python3
"""Per-progress cluster evolution (consolidates prior plot_progress_* variants).

For each milestone p in [0, 1]:
  - take z[round(p * (T-1))] per rollout
  - compute pooled within-class cov V_p (per-progress), whiten
  - LDA 2D, t-SNE Euclidean, t-SNE Mahalanobis
  - silhouette + centroid-proxy a-vs-b
Frames share the same global axis bounds across all milestones, so the
sequence of PNGs can be assembled into a video that shows cluster evolution.

Two label modes are produced when --label-mode is `both`:
  - all_task (7-label: 6 task-success + 1 failure)
  - per_task (per-task success-vs-failure binary)

Both modes write PNG frames, an mp4 video (via OpenCV), and a TSV summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow `from core import ...` when invoked as a script (the script's parent
# directory contains the core/ package). This keeps the module importable
# without requiring an editable install or PYTHONPATH munging.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.distance import pooled_within_cov, whiten
from core.features import frame_at, pooled_hidden_states
from core.io import load_manifest, load_pkl, pkl_path
from core.metrics import per_point_ab, silhouette_safe
from core.plotting import (
    global_ab_bounds,
    global_bounds_2d,
    make_video,
    plot_ab,
    scatter_with_centroids,
)
from core.projection import lda_2d, run_tsne


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_split_seen4_unseen2_openDrawer_pnpCab_100ep"
)
BASE_OUT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_feature_vis"
    / "seen4_unseen2_openDrawer_pnpCab_100ep"
)

TASKS = (
    "CoffeeSetupMug",
    "OpenSingleDoor",
    "PnPCounterToCab",
    "PnPSinkToCounter",
    "PnPCounterToStove",
    "OpenDrawer",
)
TASK_TO_SUCC_LABEL = {t: i for i, t in enumerate(TASKS)}
FAIL_LABEL = 6
SEVEN_LABEL_NAMES = {
    0: "Coffee · succ",
    1: "OpenSingleDoor · succ",
    2: "PnPCounterToCab · succ",
    3: "PnPSinkToCounter · succ",
    4: "PnPCounterToStove · succ",
    5: "OpenDrawer · succ",
    6: "Failure (all tasks)",
}
SF_NAMES = {0: "failure", 1: "success"}
SF_COLORS = {0: (0.85, 0.32, 0.27, 1.0), 1: (0.27, 0.55, 0.85, 1.0)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-root", type=Path, default=BASE_OUT)
    p.add_argument("--label-mode", choices=("all_task", "per_task", "both"), default="both")
    p.add_argument("--out-tag", default="progress_v4_scaled",
                   help="suffix used to name output subdirectories")
    p.add_argument("--n-milestones", type=int, default=11)
    p.add_argument("--tsne-perplexity-alltask", type=int, default=30)
    p.add_argument("--tsne-perplexity-pertask", type=int, default=15)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--cov-reg", type=float, default=1e-3)
    p.add_argument("--skip-video", action="store_true", help="skip mp4 generation")
    return p.parse_args()


def seven_label_colormap() -> dict[int, tuple]:
    cmap = plt.get_cmap("tab10")
    colors = {i: cmap(i) for i in range(6)}
    colors[6] = (0.2, 0.2, 0.2, 1.0)
    return colors


def gather(split_root: Path) -> list[dict]:
    rows = load_manifest(split_root)
    records = []
    for r in rows:
        rec = load_pkl(pkl_path(split_root, r))
        X = pooled_hidden_states(rec)
        if X.shape[0] == 0:
            continue
        succ = int(r["success"])
        records.append({
            "z": X,
            "T": int(X.shape[0]),
            "task": r["task"],
            "success": succ,
            "seven_label": TASK_TO_SUCC_LABEL[r["task"]] if succ == 1 else FAIL_LABEL,
        })
    return records


def compute_per_milestone(
    features_list: list[np.ndarray],
    labels: np.ndarray,
    *,
    cov_reg: float,
    tsne_perp: int,
    random_state: int,
    cov_source_features: list[np.ndarray] | None = None,
    cov_source_labels: np.ndarray | None = None,
) -> list[dict]:
    """Per-milestone analysis. If a cov_source pool is given, the whitening
    transform is computed from that (typically all 600 rollouts with 7-label)
    and then applied to the (smaller) per-task features."""
    out = []
    for k, X in enumerate(features_list):
        if cov_source_features is not None and cov_source_labels is not None:
            cov = pooled_within_cov(cov_source_features[k], cov_source_labels, reg=cov_reg)
        else:
            cov = pooled_within_cov(X, labels, reg=cov_reg)
        Xw = whiten(X, cov)
        sil_eu = silhouette_safe(X, labels, "euclidean", random_state=random_state)
        sil_mh = silhouette_safe(Xw, labels, "euclidean", random_state=random_state)
        Z_lda = lda_2d(X, labels, random_state=random_state)
        Z_eu = run_tsne(X, metric="euclidean", perplexity=tsne_perp,
                        random_state=random_state, init="pca")
        Z_mh = run_tsne(Xw, metric="euclidean", perplexity=tsne_perp,
                        random_state=random_state, init="pca")
        own, other = per_point_ab(Xw, labels, metric="euclidean")
        out.append({
            "Z_lda": Z_lda, "Z_eu": Z_eu, "Z_mh": Z_mh,
            "own": own, "other": other,
            "sil_eu": sil_eu, "sil_mh": sil_mh,
        })
        sil_eu_s = "NA" if sil_eu is None else f"{sil_eu:+.3f}"
        sil_mh_s = "NA" if sil_mh is None else f"{sil_mh:+.3f}"
        print(f"    milestone {k:2d}: sil_eu={sil_eu_s}  sil_mh={sil_mh_s}")
    return out


def render_frames(
    per_milestone: list[dict],
    milestones: np.ndarray,
    labels: np.ndarray,
    *,
    label_kind: str,
    out_dir: Path,
    title_prefix: str,
    n_label_pts: int,
) -> list[Path]:
    """Render one 4-panel PNG per milestone with global axis bounds locked."""
    bounds_lda = global_bounds_2d([m["Z_lda"] for m in per_milestone])
    bounds_eu = global_bounds_2d([m["Z_eu"] for m in per_milestone])
    bounds_mh = global_bounds_2d([m["Z_mh"] for m in per_milestone])
    bounds_ab = global_ab_bounds([m["own"] for m in per_milestone],
                                 [m["other"] for m in per_milestone])

    if label_kind == "sf":
        colors = SF_COLORS
        names = SF_NAMES
        fail_label = None
    else:
        colors = seven_label_colormap()
        names = SEVEN_LABEL_NAMES
        fail_label = FAIL_LABEL

    out_paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(milestones):
        d = per_milestone[k]
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        scatter_with_centroids(axes[0, 0], d["Z_lda"], labels, colors, names,
                               title="LDA 2D (per-progress)",
                               fail_label=fail_label, xlim=bounds_lda[0], ylim=bounds_lda[1])
        scatter_with_centroids(axes[0, 1], d["Z_eu"], labels, colors, names,
                               title="t-SNE Euclidean (per-progress)",
                               fail_label=fail_label, xlim=bounds_eu[0], ylim=bounds_eu[1])
        scatter_with_centroids(axes[1, 0], d["Z_mh"], labels, colors, names,
                               title="t-SNE Mahalanobis (per-progress)",
                               fail_label=fail_label, xlim=bounds_mh[0], ylim=bounds_mh[1])
        plot_ab(axes[1, 1], d["own"], d["other"], labels, colors, names,
                xlim=bounds_ab[0], ylim=bounds_ab[1],
                metric_name="mahalanobis", fail_label=fail_label)

        sil_eu_s = "NA" if d["sil_eu"] is None else f"{d['sil_eu']:+.4f}"
        sil_mh_s = "NA" if d["sil_mh"] is None else f"{d['sil_mh']:+.4f}"
        for col in (0, 1):
            axes[0, col].text(
                0.5, -0.10, f"Eu silhouette = {sil_eu_s}",
                transform=axes[0, col].transAxes, ha="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="none"),
            )
        axes[1, 0].text(
            0.5, -0.10, f"Maha silhouette = {sil_mh_s}",
            transform=axes[1, 0].transAxes, ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="none"),
        )

        if label_kind == "sf":
            n_succ = int((labels == 1).sum())
            n_fail = int((labels == 0).sum())
            subtitle = f"  |  n_succ={n_succ}, n_fail={n_fail}"
        else:
            subtitle = f"  |  n={n_label_pts}"
        fig.suptitle(f"{title_prefix}  |  progress p={p:.3f}{subtitle}",
                     fontsize=14, y=0.995)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95), h_pad=3.5, w_pad=2.5)

        out_path = out_dir / f"frame_p{int(round(float(p) * 100)):03d}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        wr.writeheader()
        for r in rows:
            for k in fields:
                r.setdefault(k, "")
            wr.writerow(r)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    print("loading rollouts...")
    records = gather(args.split_root)
    n = len(records)
    print(f"  n_rollouts={n}")

    milestones = np.linspace(0.0, 1.0, args.n_milestones)
    print(f"  milestones = {[f'{p:.3f}' for p in milestones]}")

    tasks_arr = np.asarray([r["task"] for r in records])
    succ_arr = np.asarray([r["success"] for r in records], dtype=np.int64)
    seven_arr = np.asarray([r["seven_label"] for r in records], dtype=np.int64)

    feats_per_p = [
        np.stack([frame_at(r["z"], float(p)) for r in records], axis=0).astype(np.float32)
        for p in milestones
    ]

    run_all = args.label_mode in ("all_task", "both")
    run_per = args.label_mode in ("per_task", "both")

    if run_all:
        all_task_dir = args.out_root / f"rollout_progress_7label_{args.out_tag}"
        print("\n========== ALL-TASK 7-label =========")
        pm_all = compute_per_milestone(
            feats_per_p, seven_arr,
            cov_reg=args.cov_reg,
            tsne_perp=args.tsne_perplexity_alltask,
            random_state=args.random_state,
        )
        paths_all = render_frames(
            pm_all, milestones, seven_arr,
            label_kind="7label",
            out_dir=all_task_dir,
            title_prefix="z at progress  |  7-label  |  per-progress projection",
            n_label_pts=n,
        )
        if not args.skip_video:
            make_video(paths_all, all_task_dir / "progress_7label_progression.mp4", args.fps)
        write_tsv(
            all_task_dir / "progress_7label_summary.tsv",
            [{"progress": float(p),
              "sil_eu": pm_all[k]["sil_eu"], "sil_mh": pm_all[k]["sil_mh"]}
             for k, p in enumerate(milestones)],
            ["progress", "sil_eu", "sil_mh"],
        )
        print(f"  saved {all_task_dir}")

    if run_per:
        per_task_dir = args.out_root / f"rollout_per_task_progress_{args.out_tag}"
        per_task_dir.mkdir(parents=True, exist_ok=True)
        print("\n========== PER-TASK 2-label =========")
        summary_pt = []
        for task in TASKS:
            print(f"\n--- {task} ---")
            mask = tasks_arr == task
            feats_per_p_task = [Xp[mask] for Xp in feats_per_p]
            labels_t = succ_arr[mask]
            pm_t = compute_per_milestone(
                feats_per_p_task, labels_t,
                cov_reg=args.cov_reg,
                tsne_perp=args.tsne_perplexity_pertask,
                random_state=args.random_state,
                cov_source_features=feats_per_p,
                cov_source_labels=seven_arr,
            )
            out_dir_t = per_task_dir / task
            paths_t = render_frames(
                pm_t, milestones, labels_t,
                label_kind="sf",
                out_dir=out_dir_t,
                title_prefix=f"{task}  |  success vs failure  |  per-progress projection",
                n_label_pts=int(mask.sum()),
            )
            for p, src in zip(milestones, paths_t):
                dest = per_task_dir / f"{task}_p{int(round(float(p) * 100)):03d}.png"
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(src.relative_to(per_task_dir))
            if not args.skip_video:
                make_video(paths_t, per_task_dir / f"{task}_progression.mp4", args.fps)
            for k, p in enumerate(milestones):
                summary_pt.append({
                    "task": task, "progress": float(p),
                    "sil_eu": pm_t[k]["sil_eu"], "sil_mh": pm_t[k]["sil_mh"],
                })
        write_tsv(per_task_dir / "per_task_progress_summary.tsv", summary_pt,
                  ["task", "progress", "sil_eu", "sil_mh"])
        print(f"  saved {per_task_dir}")


if __name__ == "__main__":
    main()
