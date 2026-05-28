"""Temporal aggregation schemes over per-rollout z[T, D] (seen18 GR00T N1.6).

Two interfaces, both consuming rollouts = [{"z":[T,D], "succ", "task"}, ...]
(from core.io.reconstruct_rollouts):

  build_at(rollouts, mode, spec) -> (X, succ, task)
    one snapshot at a single time spec. modes:
      "abs"    spec=t      : z[t]                (length-controlled, alive only)
      "mean"   spec=t      : mean(z[0:t+1])      (cumulative mean)
      "concat" spec=t      : concat(z[0:t+1])    (D=(t+1)*Dz)
      "window" spec="lo-hi": mean(z[lo:hi+1])    (non-cumulative)

  iter_scheme(rollouts, scheme) -> yields per-time-index snapshot dicts
    for the evolution/comparison schemes (tp/tf/pad/winW); each dict has
    x_native, x_norm (∈[0,1]), x_abs (representative absolute frame), X, succ,
    task, n, n_succ, n_fail.
"""

from __future__ import annotations

import numpy as np

from .features import frame_at, z_mean_over  # noqa: F401 (re-exported)

SCHEMES = ("pp", "tp", "tf", "pad", "win1", "win2", "win3", "win4")
SCHEME_LABEL = {
    "pp": "plain-progress (full length)",
    "tp": "truncated-progress (t_d=16)",
    "tf": "truncated-frame (t_d=16)",
    "pad": "padded (L_max=45)",
    "win1": "window W=1 (per-frame)",
    "win2": "window W=2",
    "win3": "window W=3",
    "win4": "window W=4",
}


def parse_window(spec: str) -> tuple[int, int]:
    lo, hi = (int(x) for x in str(spec).split("-"))
    return lo, hi


def feat_at(z: np.ndarray, mode: str, spec) -> np.ndarray | None:
    """Single per-rollout feature for a (mode, spec); None if rollout too short."""
    if mode == "abs":
        t = int(spec)
        return z[t] if len(z) > t else None
    if mode == "mean":
        t = int(spec)
        return z[: t + 1].mean(0) if len(z) > t else None
    if mode == "concat":
        t = int(spec)
        return z[: t + 1].ravel() if len(z) > t else None
    if mode == "window":
        lo, hi = parse_window(spec)
        return z[lo:hi + 1].mean(0) if len(z) > hi else None
    raise ValueError(f"unknown mode {mode!r}")


def build_at(rollouts, mode: str, spec):
    """Stack one snapshot across rollouts. Returns (X[n,D], succ[n], task[n]) or None."""
    X, succ, task = [], [], []
    for r in rollouts:
        v = feat_at(r["z"], mode, spec)
        if v is None:
            continue
        X.append(v); succ.append(r["succ"]); task.append(r["task"])
    if not X:
        return None
    return np.stack(X).astype(np.float64), np.asarray(succ), np.asarray(task)


def _emit(X, succ, task, x_native, x_norm, x_abs) -> dict:
    X = np.stack(X).astype(np.float64)
    succ = np.asarray(succ); task = np.asarray(task)
    return {"x_native": x_native, "x_norm": float(x_norm), "x_abs": float(x_abs),
            "X": X, "succ": succ, "task": task, "n": len(succ),
            "n_succ": int((succ == 1).sum()), "n_fail": int((succ == 0).sum())}


def iter_scheme(rollouts, scheme: str, *, t_d=16, l_max=45, n_milestones=11, pad_step=2):
    """Yield per-time-index snapshots for an evolution/comparison scheme.

    tp  : truncated-progress  - z'=z[:t_d]; at progress p, cummean over first
          round(p*(len-1))+1 frames. x_abs = nominal p*(t_d-1).
    tf  : truncated-frame      - z'=z[:t_d]; at frame t, cummean(z', t+1) (alive).
    pad : padded               - z padded to l_max (repeat last); cummean(z, t+1).
    winW: window W (non-cumulative) - mean(z[wW:(w+1)W]) for full windows.
    """
    if scheme == "pp":  # plain-progress: cumulative mean over FULL length
        for p in np.linspace(0.0, 1.0, n_milestones):
            X, succ, task = [], [], []
            for r in rollouts:
                z = r["z"]
                k = int(round(p * (len(z) - 1))) + 1
                X.append(z_mean_over(z, k)); succ.append(r["succ"]); task.append(r["task"])
            yield _emit(X, succ, task, float(p), float(p), float(p) * (l_max - 1))

    elif scheme == "tp":
        for p in np.linspace(0.0, 1.0, n_milestones):
            X, succ, task = [], [], []
            for r in rollouts:
                z = r["z"][:t_d]
                k = int(round(p * (len(z) - 1))) + 1
                X.append(z_mean_over(z, k)); succ.append(r["succ"]); task.append(r["task"])
            yield _emit(X, succ, task, float(p), float(p), float(p) * (t_d - 1))

    elif scheme == "tf":
        for t in range(t_d):
            X, succ, task = [], [], []
            for r in rollouts:
                z = r["z"][:t_d]
                if len(z) < t + 1:
                    continue
                X.append(z_mean_over(z, t + 1)); succ.append(r["succ"]); task.append(r["task"])
            if X:
                yield _emit(X, succ, task, t, t / (t_d - 1), float(t))

    elif scheme == "pad":
        for t in range(0, l_max, pad_step):
            X, succ, task = [], [], []
            for r in rollouts:
                z = r["z"][:l_max]
                if len(z) < l_max:
                    z = np.vstack([z, np.repeat(z[-1:], l_max - len(z), axis=0)])
                X.append(z_mean_over(z, t + 1)); succ.append(r["succ"]); task.append(r["task"])
            yield _emit(X, succ, task, t, t / (l_max - 1), float(t))

    elif scheme.startswith("win"):
        W = int(scheme[3:])
        for w in range(l_max // W):
            lo, hi = w * W, (w + 1) * W
            X, succ, task = [], [], []
            for r in rollouts:
                if len(r["z"]) < hi:
                    continue
                X.append(r["z"][lo:hi].mean(0)); succ.append(r["succ"]); task.append(r["task"])
            if X:
                snap = _emit(X, succ, task, w, (lo + W / 2) / l_max, w * W + (W - 1) / 2)
                if min(snap["n_succ"], snap["n_fail"]) >= 1:
                    yield snap
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
