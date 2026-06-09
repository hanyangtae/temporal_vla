#!/usr/bin/env python3
"""Online failure detection: onset distribution + per-decision-time ROC-AUC.

Two questions answered:
  (1) When does the LSTM detector actually fire (t_fail) vs T_max=45?
      If t_fail ≪ T_max for failures → real early detection, not a T detector.
  (2) Using only the first t_d frames per rollout (causal features), can a
      linear classifier predict eventual success/failure? Compare:
        - h_{t_d}    : LSTM causal hidden state at t_d  (nonlinear+temporal)
        - mean(z_{1..t_d}) : causal time average        (linear+temporal)
        - z_{t_d}    : single-frame at t_d              (frame-level baseline)

Outputs:
  - onset_distribution.png    : histogram of t_fail by GT label & scope
  - decision_time_curve.png   : ROC-AUC vs t_d for the three feature families
  - online_detection.{json,tsv}
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_split_seen4_unseen2_openDrawer_pnpCab_100ep"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_train_logs"
    / "groot_n16-robocasa_seen4_unseen2_openDrawer_pnpCab_100ep-lstm-seed2_mean_mean"
    / "20260520/110644/model_final.ckpt"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep"
    / "online_detection"
)
DEFAULT_THRESHOLD = 0.9978993169999711  # val_seen-best from model_perf_vs_det.csv
DEFAULT_DECISION_TIMES = (3, 5, 8, 11, 15, 20, 30, 45)

NAME = "online_detection"
HELP = "detector onset distribution and per-decision-time ROC-AUC"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--decision-times", type=int, nargs="+", default=list(DEFAULT_DECISION_TIMES))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--random-state", type=int, default=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return p.parse_args()


def tensor_to_numpy(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def load_manifest(split_root):
    with (split_root / "manifest.tsv").open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows.sort(key=lambda r: (int(r["task_id"]), int(r["episode_idx"])))
    return rows


def pkl_path(split_root, row):
    return (split_root / row["split"] / row["task"]
            / f"task{int(row['task_id'])}--ep{int(row['episode_idx'])}--succ{int(row['success'])}.pkl")


def pooled_hidden_states(record):
    feats = []
    for h in record["hidden_states"]:
        a = tensor_to_numpy(h).astype(np.float32, copy=False)
        feats.append(a.mean(axis=(0, 1)))
    return np.stack(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


class LSTMDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


def load_detector(ckpt_path: Path, hidden_dim: int, device: str) -> LSTMDetector:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    model = LSTMDetector(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_lstm(model, X, device):
    """Return both per-frame failure prob and per-frame hidden state.

    LSTM is causal: out[t] depends only on x_1..x_t.
    """
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, D)
    out, _ = model.lstm(x)
    scores = torch.sigmoid(model.fc(out)).squeeze(-1).squeeze(0).cpu().numpy()  # (T,)
    hiddens = out.squeeze(0).cpu().numpy()  # (T, hidden)
    return scores, hiddens


def gather_per_rollout(split_root, ckpt, device, hidden_dim):
    model = load_detector(ckpt, hidden_dim, device)
    rows = load_manifest(split_root)
    records = []
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        X = pooled_hidden_states(rec)
        if X.shape[0] == 0:
            continue
        scores, hiddens = run_lstm(model, X, device)
        records.append({
            "task": r["task"], "split": r["split"],
            "episode_idx": int(r["episode_idx"]), "success": int(r["success"]),
            "T": int(X.shape[0]),
            "z": X, "h": hiddens, "p": scores,
        })
    return records


def onset_distribution(records, threshold):
    """For each rollout, find first t where p_t >= threshold. -1 if never."""
    rows = []
    for r in records:
        cross = np.where(r["p"] >= threshold)[0]
        onset = int(cross[0]) if len(cross) else -1
        rows.append({
            "task": r["task"], "split": r["split"], "episode_idx": r["episode_idx"],
            "success": r["success"], "T": r["T"], "onset": onset,
            "fired": onset >= 0,
            "fired_within_full": (onset >= 0) and (onset < r["T"] - 1),
            "max_p": float(r["p"].max()),
        })
    return rows


def causal_features_at(record, t_d):
    """Return causal features computed using only frames 0..t_d-1 (inclusive)."""
    t_eff = min(t_d, record["T"])  # cap at actual rollout length
    z_seg = record["z"][:t_eff]
    h_at = record["h"][t_eff - 1]
    return {
        "h_at": h_at,
        "z_mean_causal": z_seg.mean(axis=0),
        "z_at": record["z"][t_eff - 1],
        "t_eff": int(t_eff),
        "truncated": int(t_eff < t_d),
    }


def per_decision_time_auc(records, decision_times, random_state):
    """For each decision time t_d, train logistic regression on train+val_seen,
    test on val_unseen using only first t_d frames."""
    splits = np.asarray([r["split"] for r in records])
    labels = np.asarray([r["success"] for r in records], dtype=np.int64)  # 1=success, 0=failure
    train_mask = np.isin(splits, ["train", "val_seen"])
    test_mask = splits == "val_unseen"

    results = []
    for t_d in decision_times:
        feats = {"h_at": [], "z_mean_causal": [], "z_at": []}
        truncated_count = 0
        for r in records:
            f = causal_features_at(r, t_d)
            feats["h_at"].append(f["h_at"])
            feats["z_mean_causal"].append(f["z_mean_causal"])
            feats["z_at"].append(f["z_at"])
            truncated_count += f["truncated"]
        feats = {k: np.stack(v, axis=0) for k, v in feats.items()}

        row = {"t_d": t_d, "n_truncated": truncated_count, "n_rollouts": len(records)}
        for name, X in feats.items():
            scaler = StandardScaler().fit(X[train_mask])
            Xtr = scaler.transform(X[train_mask])
            Xte = scaler.transform(X[test_mask])
            try:
                clf = LogisticRegression(max_iter=5000, C=1.0, random_state=random_state).fit(Xtr, labels[train_mask])
                proba = clf.predict_proba(Xte)[:, 1]
                auc = float(roc_auc_score(labels[test_mask], proba))
            except Exception:
                auc = None
            row[f"auc_{name}"] = auc
        results.append(row)
    return results


def plot_onset(records, onset_rows, threshold, out_path):
    fired_fail = [r["onset"] for r in onset_rows if r["fired"] and r["success"] == 0]
    fired_succ = [r["onset"] for r in onset_rows if r["fired"] and r["success"] == 1]
    not_fired_fail = sum(1 for r in onset_rows if (not r["fired"]) and r["success"] == 0)
    not_fired_succ = sum(1 for r in onset_rows if (not r["fired"]) and r["success"] == 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(0, 46)
    axes[0].hist([fired_fail, fired_succ], bins=bins, stacked=False,
                 color=["#d95234", "#458cd6"], alpha=0.7,
                 label=[f"GT failure (fired, n={len(fired_fail)})",
                        f"GT success (fired, n={len(fired_succ)})"])
    axes[0].axvline(45, color="grey", linestyle="--", alpha=0.6, label="T_max")
    axes[0].set_xlabel("onset frame t_fail")
    axes[0].set_ylabel("rollout count")
    axes[0].set_title(f"Detector onset distribution (τ={threshold:.4f})\n"
                      f"fired: fail {len(fired_fail)} / succ {len(fired_succ)}, "
                      f"never: fail {not_fired_fail} / succ {not_fired_succ}")
    axes[0].legend(loc="best", fontsize=8)

    # Box plot of onset/T_max ratio (only fired failures)
    if len(fired_fail):
        axes[1].hist(np.asarray(fired_fail) / 45, bins=20, color="#d95234", alpha=0.7)
        axes[1].axvline(1.0, color="grey", linestyle="--", alpha=0.6)
        axes[1].set_xlabel("t_fail / T_max")
        axes[1].set_ylabel("rollout count")
        axes[1].set_title("Failure-rollout onset (fired only)\nrelative to T_max=45")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_auc_curve(results, out_path):
    t_ds = [r["t_d"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    series = {
        "h_at (LSTM hidden, causal)": "auc_h_at",
        "mean z_{1..t_d} (linear+temporal)": "auc_z_mean_causal",
        "z_{t_d} (single frame)": "auc_z_at",
    }
    colors = {"h_at (LSTM hidden, causal)": "#458cd6",
              "mean z_{1..t_d} (linear+temporal)": "#43a047",
              "z_{t_d} (single frame)": "#d95234"}
    for label, key in series.items():
        vals = [r[key] for r in results]
        ax.plot(t_ds, vals, marker="o", lw=2, label=label, color=colors[label])
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.6, label="chance")
    ax.set_xlabel("decision time t_d (frames seen so far)")
    ax.set_ylabel("ROC-AUC on val_unseen")
    ax.set_title("Per-decision-time success/failure ROC-AUC\n(causal features from first t_d frames; train on train+val_seen)")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            for k in fields:
                r.setdefault(k, "")
            writer.writerow(r)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading detector + scoring on {args.device}...")
    records = gather_per_rollout(args.split_root, args.ckpt, args.device, args.hidden_dim)
    print(f"  n_rollouts={len(records)}")

    # 1) Onset distribution
    onset_rows = onset_distribution(records, args.threshold)
    write_tsv(args.out_dir / "onset_distribution.tsv", onset_rows,
              ["task", "split", "episode_idx", "success", "T",
               "onset", "fired", "fired_within_full", "max_p"])

    # quick summary
    n_fail = sum(1 for r in onset_rows if r["success"] == 0)
    n_succ = sum(1 for r in onset_rows if r["success"] == 1)
    fired_fail_onsets = np.asarray([r["onset"] for r in onset_rows if r["fired"] and r["success"] == 0])
    fired_succ_onsets = np.asarray([r["onset"] for r in onset_rows if r["fired"] and r["success"] == 1])
    print(f"  GT failure: {len(fired_fail_onsets)}/{n_fail} fired. onset median={np.median(fired_fail_onsets) if len(fired_fail_onsets) else 'NA'}")
    print(f"  GT success: {len(fired_succ_onsets)}/{n_succ} fired (false positive).")

    plot_onset(records, onset_rows, args.threshold, args.out_dir / "onset_distribution.png")

    # 2) Per-decision-time ROC-AUC
    print("\nrunning per-decision-time evaluation...")
    auc_rows = per_decision_time_auc(records, args.decision_times, args.random_state)
    write_tsv(args.out_dir / "decision_time_auc.tsv", auc_rows,
              ["t_d", "n_truncated", "n_rollouts", "auc_h_at", "auc_z_mean_causal", "auc_z_at"])
    for row in auc_rows:
        print(f"  t_d={row['t_d']:2d}: "
              f"AUROC h_at={row['auc_h_at']:.4f}  "
              f"z_mean={row['auc_z_mean_causal']:.4f}  "
              f"z_at={row['auc_z_at']:.4f}  "
              f"(truncated={row['n_truncated']})")

    plot_auc_curve(auc_rows, args.out_dir / "decision_time_curve.png")

    with (args.out_dir / "online_detection.json").open("w") as f:
        json.dump({
            "threshold": args.threshold,
            "decision_times": list(args.decision_times),
            "n_rollouts": len(records),
            "fired_fail": int(len(fired_fail_onsets)),
            "fired_succ": int(len(fired_succ_onsets)),
            "fired_fail_onset_mean": float(fired_fail_onsets.mean()) if len(fired_fail_onsets) else None,
            "fired_fail_onset_median": float(np.median(fired_fail_onsets)) if len(fired_fail_onsets) else None,
            "auc_rows": auc_rows,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    print(args.out_dir)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
