#!/usr/bin/env python3
"""CPU-only, manifest-selected equal-budget LR detector and setM fitting.

No raw collection, model serving, or GPU allocation occurs in this entrypoint.
All statistics, including length caps and score calibration, use selected data.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1] / "failure_detector_sim.py"
_spec = importlib.util.spec_from_file_location("lr_detector_base", BASE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return (str(row["slug"]), int(row["scene"]), int(row["jitter"]), int(row["noise"]))


@dataclass
class Record:
    row: dict
    X: np.ndarray
    phase: np.ndarray  # semantic phase names, never shard-local integer IDs
    fingerprint: dict

    @property
    def identity(self):
        return key(self.row)

    @property
    def success(self):
        return int(self.row["success"])

    @property
    def side(self):
        return str(self.row["side"])


class Shards:
    """Explicit shard mapping: {"<canonical-slug>__s<scene>": path or descriptor}.

    Descriptor accepts path, optional source_instruction, sha256. Canonical naming
    is never inferred from legacy swapped oven/washer directory names.
    """
    def __init__(self, mapping):
        self.mapping = mapping
        self.cache = {}

    def load(self, row):
        slug, scene, jitter, noise = key(row)
        mapkey = f"{slug}__s{scene}"
        descriptor = self.mapping[mapkey]
        descriptor = {"path": descriptor} if isinstance(descriptor, str) else descriptor
        path = Path(descriptor["path"]).expanduser().resolve()
        if mapkey not in self.cache:
            sha = digest(path)
            if descriptor.get("sha256") and sha != descriptor["sha256"]:
                raise ValueError(f"source hash mismatch: {mapkey}")
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
                if descriptor.get("source_instruction") and meta.get("instruction") != descriptor["source_instruction"]:
                    raise ValueError(f"source instruction mismatch: {mapkey}")
                if meta.get("jitter_axis") is not True:
                    raise ValueError(f"no verified jitter axis: {mapkey}")
                layers = [int(x) for x in meta.get("capture_layers", meta.get("layers", []))]
                segments = meta["segment_names"]
                axis = "jitter_idx" if "jitter_idx" in z.files else "jitter"
                cols = {name: np.asarray(z[name]) for name in
                        ("ep_id", "scene", "noise", "rec_idx", "succ", "phase_code", "ep_len", axis)}
                tensor = z["X"]
                if tensor.ndim != 5:
                    raise ValueError("Tier A [record,layer,denoise,segment,dim] required")
                X = np.asarray(tensor[:, layers.index(12), -1, segments.index("all"), :], dtype=np.float32)
                del tensor
                if not np.isfinite(X).all():
                    raise ValueError(f"nonfinite source features: {mapkey}")
            phase_names = {int(v): k for k, v in meta["phase_codebook"].items()}
            if any(not name or Path(name).name != name or name in (".", "..") for name in phase_names.values()):
                raise ValueError(f"invalid semantic phase path: {mapkey}")
            episodes = {}
            for ep_id in np.unique(cols["ep_id"]):
                idx = np.flatnonzero(cols["ep_id"] == ep_id)
                idx = idx[np.argsort(cols["rec_idx"][idx], kind="stable")]
                for name in ("scene", "noise", "succ", "ep_len", axis):
                    if len(np.unique(cols[name][idx])) != 1:
                        raise ValueError(f"inconsistent episode column {name}: {mapkey}/{ep_id}")
                if int(cols["ep_len"][idx[0]]) != len(idx) or not np.array_equal(cols["rec_idx"][idx], np.arange(len(idx))):
                    raise ValueError(f"incomplete record sequence: {mapkey}/{ep_id}")
                coord = (int(cols["scene"][idx[0]]), int(cols[axis][idx[0]]), int(cols["noise"][idx[0]]))
                if coord in episodes:
                    raise ValueError(f"duplicate shard coordinate: {mapkey}/{coord}")
                sigs = meta.get("sigs") or []
                if int(ep_id) >= len(sigs):
                    raise ValueError(f"missing source fingerprint: {mapkey}/{ep_id}")
                episodes[coord] = (X[idx], np.asarray([phase_names[int(c)] for c in cols["phase_code"][idx]]),
                                   int(cols["succ"][idx[0]]), {"shard": path.name, "sha256": sha,
                                   "ep_id": int(ep_id), "sig": sigs[int(ep_id)],
                                   "source_instruction": meta.get("instruction")})
            self.cache[mapkey] = episodes
        X, phase, success, fingerprint = self.cache[mapkey][(scene, jitter, noise)]
        if success != int(row["success"]):
            raise ValueError(f"manifest/shard label mismatch: {key(row)}")
        if row.get("sig") is not None and row["sig"] != fingerprint["sig"]:
            raise ValueError(f"episode source fingerprint mismatch: {key(row)}")
        return Record(dict(row), X, phase, fingerprint)


def common_caps(arms):
    by_arm = [{e.identity: e for e in eps if e.success} for eps in arms]
    shared = set.intersection(*(set(d) for d in by_arm))
    if not shared:
        raise ValueError("no common selected success episodes for common phase caps")
    eps = [by_arm[0][k] for k in sorted(shared)]
    caps = {}
    for phase in sorted({p for e in eps for p in e.phase.tolist()}):
        dwell = np.array([np.sum(e.phase == phase) for e in eps], dtype=float)
        dwell = dwell[dwell > 0]
        caps[phase] = int(np.ceil(dwell.mean() + dwell.std()))
    return caps, [list(k) for k in sorted(shared)]


def truncate(e, caps):
    idx = sorted(i for phase, cap in caps.items() for i in np.flatnonzero(e.phase == phase)[:cap])
    if len(idx) < 2:
        raise ValueError(f"episode too short after common cap; budget would change: {e.identity}")
    return replace(e, X=e.X[idx], phase=e.phase[idx])


def weights(eps):
    """Equal mass per unique episode; the planner already matches marginals.

    Rebalancing observed side/jitter groups again would change the shared current
    failure anchor's mass and undo the planner's exact (class,jitter) match.
    """
    if not eps:
        raise ValueError("cannot weight an empty selection")
    return np.full(len(eps), 1.0 / len(eps), dtype=np.float64)


def standardize(eps, w):
    mu = sum(a * e.X.astype(np.float64).mean(0) for a, e in zip(w, eps))
    variance = sum(a * ((e.X.astype(np.float64) - mu) ** 2).mean(0) for a, e in zip(w, eps))
    sd = np.sqrt(np.maximum(variance, 0))
    sd[sd < 1e-6] = 1
    return mu.astype(np.float32), sd.astype(np.float32)


def train(eps, mu, sd, w, args):
    torch.manual_seed(args.seed)
    model = base.build_detector("lstm", len(mu), args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    updates = 0
    for _ in range(args.epochs):
        order = rng.permutation(len(eps))
        for start in range(0, len(eps), args.batch_size):
            ids = order[start:start + args.batch_size]
            n, T = len(ids), max(len(eps[i].X) for i in ids)
            X = torch.zeros(n, T, len(mu))
            mask = torch.zeros(n, T)
            y = torch.zeros(n, T)
            for k, i in enumerate(ids):
                t = len(eps[i].X)
                X[k, :t] = torch.from_numpy((eps[i].X - mu) / sd)
                mask[k, :t] = 1
                y[k, :t] = 1 - eps[i].success
            pred = model(X).clamp(1e-6, 1 - 1e-6)
            bce = -(y * pred.log() + (1 - y) * (1 - pred).log())
            per_episode = (bce * mask).sum(1) / mask.sum(1)
            # Uniform episode permutation, importance weights give the desired
            # full-data expectation without batch composition renormalization.
            loss = (per_episode * torch.as_tensor(w[ids] * len(eps), dtype=torch.float32)).mean()
            loss += args.lambda_reg * sum((p ** 2).sum() for name, p in model.named_parameters() if "bias" not in name)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updates += 1
    return model.eval(), updates


def loo_band(scores, alpha=0.1):
    """LOO score-band calibration (not LOO model refitting or held-out CP)."""
    if len(scores) < 10:
        raise ValueError("alpha .1 calibration requires at least 10 selected success episodes")
    L = max(map(len, scores))
    A = np.stack([np.pad(s, (0, L - len(s)), mode="edge") for s in scores])
    excess = []
    for i in range(len(A)):
        other = np.delete(A, i, axis=0)
        excess.append(float(np.max((A[i] - other.mean(0)) / (other.std(0, ddof=1) + 1e-8))))
    mu, sd = A.mean(0), A.std(0, ddof=1) + 1e-8
    bw = float(np.quantile(excess, 1 - alpha))
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "bw": bw,
            "delta": (mu + bw * sd).astype(np.float32)}


def phase_mean(eps, phase):
    present = [e for e in eps if np.any(e.phase == phase)]
    if not present:
        return None
    # Equal episode mass within a side/jitter/class, retain all capped records.
    return np.mean([e.X[e.phase == phase].astype(np.float64).mean(0) for e in present], axis=0)


def operators(eps, target_side, out):
    sides = sorted({e.side for e in eps})
    phases = sorted({p for e in eps for p in e.phase.tolist()})
    report = []
    for phase in phases:
        means, deltas = {}, {}
        for side in sides:
            side_eps = [e for e in eps if e.side == side]
            for success in (0, 1):
                by_j = [phase_mean([e for e in side_eps if e.success == success and int(e.row["jitter"]) == j], phase)
                        for j in sorted({int(e.row["jitter"]) for e in side_eps})]
                by_j = [m for m in by_j if m is not None]
                means[side, success] = np.mean(by_j, axis=0) if by_j else None
            deltas[side] = []
            for j in sorted({int(e.row["jitter"]) for e in side_eps}):
                a = phase_mean([e for e in side_eps if e.success == 1 and int(e.row["jitter"]) == j], phase)
                b = phase_mean([e for e in side_eps if e.success == 0 and int(e.row["jitter"]) == j], phase)
                if a is not None and b is not None:
                    deltas[side].append(a - b)
        anchor_eps = [e for e in eps if e.side == target_side and e.row["role"] == "current_fail"]
        anchor = phase_mean(anchor_eps, phase)
        for variant in ("plain", "lr_jfair"):
            reason, delta, setpoint = None, None, None
            if variant == "plain":
                if any(m is None for m in means.values()):
                    reason = "missing_side_class_phase"
                else:
                    mu_s = np.mean([means[s, 1] for s in sides], axis=0)
                    delta = mu_s - np.mean([means[s, 0] for s in sides], axis=0)
            elif any(len(deltas[s]) < 2 for s in sides):
                reason = "fewer_than_two_mixed_jitters_per_side"
            elif anchor is None:
                reason = "missing_current_target_failure_phase"
            else:
                delta = np.mean([np.mean(deltas[s], axis=0) for s in sides], axis=0)
            if reason is None:
                norm = np.linalg.norm(delta)
                if not np.isfinite(norm) or norm < 1e-10:
                    reason = "zero_or_nonfinite_direction"
                else:
                    v = delta / norm
                    setpoint = float(mu_s @ v) if variant == "plain" else float(anchor @ v + delta @ v)
                    directory = out / variant / phase / "dit_L12"
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory / "conceptors.npz", alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(setpoint))
                    (directory / "metadata.json").write_text(json.dumps({"op": "setpoint", "selected_alpha": 0,
                        "phase": phase, "variant": variant, "target_side": target_side,
                        "n_mixed_jitters": {s: len(deltas[s]) for s in sides}, "episode_weighted": True,
                        "phase_label_source": "gt", "layer": 12}, indent=2) + "\n")
            report.append({"phase": phase, "variant": variant, "registered": reason is None, "reason": reason})
    return report


def normalize_row(row):
    """Planner TSV adapter; preserve complete original provenance in source_row."""
    return {"slug": row["canonical_instruction"].replace("/", "_"),
            "scene": int(row["scene_idx"]), "jitter": int(row["jitter_idx"]),
            "noise": int(row["noise_idx"]), "success": int(row["success"]),
            "side": row["canonical_side"],
            "role": "current_fail" if int(row["is_current_failure"]) else "historical",
            "episode_uid": row["episode_uid"], "sig": row.get("sig") or None, "source_row": row}


def validate_pair(rows, plan):
    request = plan["request"]
    arms = defaultdict(list)
    for row in rows:
        arms[row["arm"]].append(normalize_row(row))
    if len(arms) != 2:
        raise ValueError("paired solo/mix experiment requires exactly two arms")
    budget, S = int(request["budget"]), int(request["n_success"])
    if S < 10:
        raise ValueError("selected success budget must be >=10 for alpha .1")
    if int(request.get("calibration_count", -1)) != S:
        raise ValueError("calibration_count must equal the selected success budget")
    for row in rows:
        if int(row.get("train", 0)) != 1 or int(row.get("calibration", -1)) != int(row["success"]):
            raise ValueError("calibration flags must select exactly all selected training successes")
    for arm, selected in arms.items():
        if len(selected) != budget or len({key(r) for r in selected}) != budget or len({r["episode_uid"] for r in selected}) != budget:
            raise ValueError(f"incorrect or duplicated unique episode budget: {arm}")
        if sum(r["success"] for r in selected) != S:
            raise ValueError(f"success quota mismatch: {arm}")
        if any(r["success"] and r["jitter"] == int(request["target_jitter"]) for r in selected):
            raise ValueError("target-jitter success is not authorized fit input")
        current = [r for r in selected if r["role"] == "current_fail"]
        if len(current) != int(request["target_fail_count"]) or any(r["success"] or r["side"] != request["target_side"] or r["jitter"] != int(request["target_jitter"]) for r in current):
            raise ValueError(f"current failure anchor quota mismatch: {arm}")
        if "LRmix" in arm:
            for label, count in ((1, S), (0, budget - S)):
                if any(sum(r["success"] == label and r["side"] == side for r in selected) != count // 2 for side in ("L", "R")):
                    raise ValueError("mixed side/class quotas must be exact halves")
        elif any(r["side"] != request["target_side"] for r in selected):
            raise ValueError("solo contains opposite-side episode")
    anchor_sets = [{key(r) for r in selected if r["role"] == "current_fail"} for selected in arms.values()]
    if anchor_sets[0] != anchor_sets[1]:
        raise ValueError("paired current failure anchors differ")
    class_jitter = [Counter((r["success"], r["jitter"]) for r in selected) for selected in arms.values()]
    if class_jitter[0] != class_jitter[1]:
        raise ValueError("paired (success,jitter) episode counts differ")
    return dict(arms)


def fit_pair(rows, plan, shards, out, args, provenance):
    pair_id = plan["pair_id"]
    root = out / pair_id
    if root.exists():
        raise FileExistsError(f"refuse overwriting prior/partial fit: {root}")
    selected = validate_pair(rows, plan)
    records = {arm: [shards.load(row) for row in arm_rows] for arm, arm_rows in selected.items()}
    dims = {e.X.shape[1] for eps in records.values() for e in eps}
    if dims != {args.expected_dim}:
        raise ValueError(f"feature dimension mismatch: {dims}, expected {args.expected_dim}")
    caps, cap_keys = common_caps(list(records.values()))
    records = {arm: [truncate(e, caps) for e in eps] for arm, eps in records.items()}
    root.mkdir(parents=True)
    result = {"pair_id": pair_id, "status": "fitting", "request": plan["request"],
              "common_phase_caps": caps, "cap_source_keys": cap_keys,
              "provenance": provenance, "arms": {}}
    for arm, eps in records.items():
        dest = root / arm
        dest.mkdir()
        w = weights(eps)
        mu, sd = standardize(eps, w)
        model, updates = train(eps, mu, sd, w, args)
        successes = [e for e in eps if e.success]
        scores = [base.score_seq(model, (e.X - mu) / sd) for e in successes]
        band = loo_band(scores, 0.1)
        payload = {"arm": arm, "model": "lstm", "group": pair_id,
                   "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                   "input_dim": len(mu), "hidden": args.hidden, "std_mean": mu, "std_std": sd,
                   "feature": {"layer": 12, "denoise": -1, "seg": "all", "dim": len(mu)},
                   "cp_bands": {pair_id: {"0.10": band}}, "tasks": [pair_id],
                   "shards": sorted({e.fingerprint["shard"] for e in eps}),
                   "train": {"epochs": args.epochs, "lr": args.lr, "hidden": args.hidden,
                             "lambda_reg": args.lambda_reg, "batch_size": args.batch_size,
                             "seed": args.seed, "updates": updates, "weighting": "uniform_unique_episode/mean_timestep",
                             "n_unique": len(eps)},
                   "truncate": {"mode": "phase-gt", "phase_caps": caps, "cap_source_keys": cap_keys,
                                "eval": "full", "n_dropped": 0},
                   "calibration": {"method": "loo_score_band_selected_train_success", "alpha": 0.1,
                                   "n_success": len(successes), "held_out_model": False,
                                   "note": "LOO score-band calibration, not finite-sample held-out conformal guarantee"},
                   "provenance": provenance,
                   "selected_episodes": [{**e.row, "fingerprint": e.fingerprint} for e in eps]}
        torch.save(payload, dest / "detector.pt")
        registry = operators(eps, plan["request"]["target_side"], dest / "operators")
        report = {"arm": arm, "pair_id": pair_id, "failure_task": pair_id,
                  "n_unique": len(eps), "n_success": len(successes), "n_failure": len(eps) - len(successes),
                  "updates": updates, "detector": "detector.pt", "detector_sha256": digest(dest / "detector.pt"),
                  "registered": {variant: [r["phase"] for r in registry if r["variant"] == variant and r["registered"]]
                                 for variant in ("plain", "lr_jfair")},
                  "registry": registry, "phase_caps": caps,
                  "episodes": [{**e.row, "weight": float(a), "n_capped_records": len(e.X),
                                "fingerprint": e.fingerprint} for a, e in zip(w, eps)]}
        (dest / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
        result["arms"][arm] = {k: report[k] for k in ("n_unique", "n_success", "n_failure", "updates", "registered", "detector_sha256")}
    if len({r["updates"] for r in result["arms"].values()}) != 1:
        raise AssertionError("paired update budget mismatch")
    result["status"] = "complete"
    (root / "pair_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pair_id": pair_id, "status": "complete", "arms": result["arms"]}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--shard-map", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair-id", action="append", help="fit only specified pair IDs (repeatable)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0, help="network/permutation seed; selection seed remains planner provenance")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--expected-dim", type=int, default=1536, help="override only for synthetic contract smoke")
    return p


def main():
    args = parser().parse_args()
    if min(args.epochs, args.hidden, args.batch_size, args.threads) < 1:
        raise ValueError("positive epochs, hidden, batch size and threads required")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    if any(not row.get("sig") for row in rows):
        raise ValueError("source episode sig required for every manifest row")
    plans = json.loads(args.plans.read_text())["plans"]
    mapping = json.loads(args.shard_map.read_text())
    for descriptor in mapping.values():
        if not isinstance(descriptor, dict) or not descriptor.get("source_instruction"):
            raise ValueError("explicit source_instruction descriptor required for every shard (legacy LR alias ambiguity)")
    shards = Shards(mapping)
    provenance = {"manifest_sha256": digest(args.manifest), "plans_sha256": digest(args.plans),
                  "shard_map_sha256": digest(args.shard_map), "fit_source_sha256": digest(__file__),
                  "detector_source_sha256": digest(BASE), "torch": torch.__version__, "numpy": np.__version__,
                  "device": "cpu", "threads": args.threads}
    selected = [p for p in plans if p["status"] == "planned" and (not args.pair_id or p["pair_id"] in args.pair_id)]
    if not selected or (args.pair_id and set(args.pair_id) != {p["pair_id"] for p in selected}):
        raise ValueError("requested pair absent or infeasible")
    for plan in selected:
        fit_pair([r for r in rows if r["pair_id"] == plan["pair_id"]], plan, shards, args.out, args, provenance)


if __name__ == "__main__":
    main()
