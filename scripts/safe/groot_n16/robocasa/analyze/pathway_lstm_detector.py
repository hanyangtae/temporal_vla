"""SAFE-LSTM 실패 detector — per-pathway, task별 80/20 seen split + unseen.

SAFE-LSTM(NeurIPS'25, `vis/core/lstm.py` LSTMDetector): per-step LSTM scalar score
(단층 LSTM→linear→sigmoid), per-step BCE 로 학습. inference-step 순차 처리(causal:
score[t]는 x[:t+1]만 의존). 우리 pathway feature(VL/DiT)에 직접 학습.

split (사용자 지시 [[detector-eval-seen-and-unseen-8020]]):
  - seen task: 각 task 80% train / 20% seen-test
  - unseen task: 전체 unseen-test
eval:
  - **decision-time AUROC**(seen-test·unseen-test, t_d별, living=length>t_d): LSTM score[t_d-1] 로.
  - **per-step score 궤적**(성공 vs 실패 평균): "LSTM이 언제 점수를 올리나" = 검출 동역학.
  - onset(score>τ 첫 시점). length baseline(full length→fail) 대조.
주의: x축은 **inference step(action-chunk)** — env-step 아님. 성공 중앙값 ~13추론이라 t_d=11은 성공의 ~85%
([[n16-online-detection-feasible]]). 입력 feature 표준화(train stats) 필수. pkl 원격(torch) → ${REMOTE_PYTHON}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "vis"))
from pathway_separation import load_rollout_features  # noqa: E402
from core.lstm import LSTMDetector  # noqa: E402  (SAFE-LSTM 아키텍처 단일 출처)

CAPTURE_LAYERS = [0, 2, 4, 8, 16, 24, 31]
DEFAULT_TDS = (3, 5, 8, 11, 15, 20)


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    s = np.zeros(cnt.size); np.add.at(s, inv, ranks); ranks = (s / cnt)[inv]
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pathway_seq(r: dict, pathway: str, dit_idx: int) -> np.ndarray | None:
    """rollout → per-step feature seq [T, D] for pathway."""
    dit = r["dit"][:, dit_idx, :]
    if pathway == "dit":
        return dit
    if r["vl"] is None:
        return None
    if pathway == "vl":
        return r["vl"]
    return np.concatenate([r["vl"], dit], axis=1)


def load_all(run_dir: Path, token_pool: str):
    rolls = []
    for td in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for p in sorted(td.glob("*.pkl")):
            r = load_rollout_features(p, token_pool)
            if r is None:
                continue
            r["task"] = td.name
            rolls.append(r)
    return rolls


def split_8020(rolls, seen, unseen, frac, seed):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rolls:
        if r["task"] in seen:
            by[r["task"]].append(r)
    train, seen_test = [], []
    for task, rs in by.items():
        idx = np.arange(len(rs)); rng.shuffle(idx)
        n = int(round(frac * len(rs)))
        train += [rs[i] for i in idx[:n]]
        seen_test += [rs[i] for i in idx[n:]]
    unseen_test = [r for r in rolls if r["task"] in unseen]
    return train, seen_test, unseen_test


def standardizer(train_seqs):
    allf = np.concatenate([s[0] for s in train_seqs], axis=0)
    mu = allf.mean(axis=0); sd = allf.std(axis=0); sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def train_lstm(train_seqs, input_dim, epochs, lr, hidden, device, seed,
               lambda_reg=1e-2, grad_clip=1.0):
    """SAFE-LSTM 학습. SAFE식: BCE + lambda_reg·L2(bias 제외) + grad clip."""
    torch.manual_seed(seed)
    model = LSTMDetector(input_dim=input_dim, hidden_dim=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    for ep in range(epochs):
        random.shuffle(train_seqs)
        tot = 0.0
        for X, y, _L, _t in train_seqs:
            xb = torch.from_numpy(X).float().unsqueeze(0).to(device)  # [1,T,D]
            sc = model(xb).squeeze(0)                                 # [T]
            loss = bce(sc, torch.full_like(sc, float(y)))
            if lambda_reg > 0:
                l2 = sum((p ** 2).sum() for n, p in model.named_parameters() if "bias" not in n)
                loss = loss + lambda_reg * l2
            opt.zero_grad(); loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += loss.item()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{epochs} loss={tot/max(1,len(train_seqs)):.4f}")
    model.eval()
    return model


@torch.no_grad()
def score_seq(model, X, device):
    xb = torch.from_numpy(X).float().unsqueeze(0).to(device)
    return model(xb).squeeze(0).cpu().numpy()  # [T]


def make_seqs(rolls, pathway, dit_idx, mu=None, sd=None):
    out = []
    for r in rolls:
        X = pathway_seq(r, pathway, dit_idx)
        if X is None:
            continue
        Xn = ((X - mu) / sd).astype(np.float32) if mu is not None else X.astype(np.float32)
        out.append((Xn, 1 - int(r["success"]), r["length"], r["task"]))
    return out


def decision_time_auroc(model, seqs, tds, device):
    """t_d별 living rollout(length>t_d)에서 LSTM score[t_d-1] AUROC."""
    res = {}
    pre = [(score_seq(model, X, device), y, L) for X, y, L, _ in seqs]
    for t in tds:
        s, yy = [], []
        for sc, y, L in pre:
            if L > t:
                s.append(sc[t - 1]); yy.append(y)
        s, yy = np.asarray(s), np.asarray(yy)
        if len(np.unique(yy)) == 2 and min(np.bincount(yy)) >= 3:
            res[str(t)] = {"auroc": auroc(s, yy), "n": int(len(yy)), "n_fail": int(yy.sum())}
    return res


def length_auroc(seqs, tds):
    res = {}
    for t in tds:
        L, y = [], []
        for _, yy, ln, _ in seqs:
            if ln > t:
                L.append(ln); y.append(yy)
        L, y = np.asarray(L, float), np.asarray(y)
        if len(L) and len(np.unique(y)) == 2:
            res[str(t)] = auroc(L, y)
    return res


def cp_metrics(model, cal_seqs, eval_seqs, device, alphas):
    """constant-threshold functional-CP + normalized T-det (SAFE식).

    cal 성공 rollout 들의 **max-score** 의 (1-α) 분위로 임계 δ 보정 → 성공의 α 만 넘김(FPR≈α 보장).
    eval 에서: failure 가 δ 를 넘으면 검출(TPR), 성공이 넘으면 오경보(FPR),
    **normalized T-det = 첫 crossing step / 그 rollout 길이**(낮을수록 일찍, 안 넘으면 1).
    """
    cal_max = np.array([score_seq(model, X, device).max() for X, y, _L, _t in cal_seqs if y == 0])
    if len(cal_max) < 3:
        return {}
    ev = [(score_seq(model, X, device), y, L) for X, y, L, _t in eval_seqs]
    out = {}
    for a in alphas:
        delta = float(np.quantile(cal_max, 1.0 - a))
        fail = succ = tp = fp = 0
        tdet_fail, tdet_fired = [], []
        for sc, y, L in ev:
            cross = np.where(sc > delta)[0]
            fired = len(cross) > 0
            if y == 1:
                fail += 1
                if fired:
                    tp += 1; tdet_fired.append(cross[0] / L)
                tdet_fail.append(cross[0] / L if fired else 1.0)
            else:
                succ += 1
                fp += int(fired)
        tpr = tp / fail if fail else float("nan")
        fpr = fp / succ if succ else float("nan")
        out[f"{a:.2f}"] = {
            "delta": round(delta, 4), "tpr": round(tpr, 3), "fpr": round(fpr, 3),
            "bal_acc": round((tpr + (1 - fpr)) / 2, 3) if (fail and succ) else None,
            "mean_tdet_fail": round(float(np.mean(tdet_fail)), 3) if tdet_fail else None,
            "mean_tdet_fired": round(float(np.mean(tdet_fired)), 3) if tdet_fired else None,
            "n_fail": fail, "n_succ": succ,
        }
    return out


def _pad_to(s, L):
    return s[:L] if len(s) >= L else np.concatenate([s, np.full(L - len(s), s[-1])])


def functional_cp_metrics(model, train_succ, cal_succ, eval_seqs, device, alphas, L):
    """SAFE functional CP (시간가변 밴드). 성공 궤적 per-step μ_t,σ_t →
    δ_t = μ_t + bw·σ_t, bw = 보정성공의 max_t(s_t-μ_t)/σ_t 의 (1-α)분위.
    실패 = δ_t 위로 이탈하는 첫 t. 궤적은 L 로 forward-fill 패딩(성공 종료후 plateau).
    """
    tr = [_pad_to(score_seq(model, X, device), L) for X, y, _L, _t in train_succ if y == 0]
    cal = [_pad_to(score_seq(model, X, device), L) for X, y, _L, _t in cal_succ if y == 0]
    if len(tr) < 3 or len(cal) < 3:
        return {}
    tr, cal = np.stack(tr), np.stack(cal)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0, ddof=1) + 1e-8
    excursion = np.max((cal - mu) / sd, axis=1)
    ev = [(score_seq(model, X, device), y, Lr) for X, y, Lr, _t in eval_seqs]
    out = {}
    for a in alphas:
        bw = float(np.quantile(excursion, 1 - a))
        delta = mu + bw * sd  # [L] time-varying band
        fail = succ = tp = fp = 0
        tdet_fail, tdet_fired = [], []
        for sc, y, Lr in ev:
            n = min(len(sc), L)
            cross = np.where(sc[:n] > delta[:n])[0]
            fired = len(cross) > 0
            if y == 1:
                fail += 1
                if fired:
                    tp += 1; tdet_fired.append(cross[0] / Lr)
                tdet_fail.append(cross[0] / Lr if fired else 1.0)
            else:
                succ += 1; fp += int(fired)
        tpr = tp / fail if fail else float("nan")
        fpr = fp / succ if succ else float("nan")
        out[f"{a:.2f}"] = {
            "band_width": round(bw, 3),
            "delta_t_range": [round(float(delta.min()), 3), round(float(delta.max()), 3)],
            "tpr": round(tpr, 3), "fpr": round(fpr, 3),
            "bal_acc": round((tpr + (1 - fpr)) / 2, 3) if (fail and succ) else None,
            "mean_tdet_fail": round(float(np.mean(tdet_fail)), 3) if tdet_fail else None,
            "mean_tdet_fired": round(float(np.mean(tdet_fired)), 3) if tdet_fired else None,
            "n_fail": fail, "n_succ": succ,
        }
    return out


def mean_trajectory(model, seqs, device, max_t=40):
    """성공/실패 평균 per-step score 궤적 (padding: 마지막 score 유지)."""
    succ, fail = [], []
    for X, y, L, _ in seqs:
        sc = score_seq(model, X, device)
        pad = np.full(max_t, sc[-1]); pad[: min(len(sc), max_t)] = sc[: max_t]
        (fail if y == 1 else succ).append(pad)
    f = np.stack(fail).mean(0) if fail else None
    s = np.stack(succ).mean(0) if succ else None
    return s, f


def main():
    ap = argparse.ArgumentParser(description="SAFE-LSTM per-pathway detector (80/20 seen + unseen)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token-pool", default="valid16")
    ap.add_argument("--dit-block", type=int, default=31)
    ap.add_argument("--pathways", default="dit,vl")
    ap.add_argument("--t-ds", default=",".join(map(str, DEFAULT_TDS)))
    ap.add_argument("--seen", default="CloseToasterOvenDoor,NavigateKitchen,OpenCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,SlideDishwasherRack,TurnOnMicrowave,TurnOnSinkFaucet")
    ap.add_argument("--unseen", default="OpenDrawer,PickPlaceCounterToCabinet")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lambda-reg", type=float, default=1e-2, help="SAFE식 L2 정규화 계수")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="grad clip max-norm(0=off)")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5", help="CP 유의수준(FPR 목표)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / "analysis" / "pathway_lstm_detector"
    dit_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
    pathways = args.pathways.split(",")
    tds = [int(x) for x in args.t_ds.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    seen, unseen = set(args.seen.split(",")), set(args.unseen.split(","))

    rolls = load_all(run_dir, args.token_pool)
    train, seen_test, unseen_test = split_8020(rolls, seen, unseen, args.train_frac, args.seed)
    L_max = max((r["length"] for r in rolls), default=45)
    print(f"[load] {len(rolls)} rollouts | train={len(train)} seen_test={len(seen_test)} "
          f"unseen_test={len(unseen_test)} | DiT block={CAPTURE_LAYERS[dit_idx]} | L_max={L_max}")
    if args.smoke:
        print("  smoke: shapes only");  return

    results = {"dit_block": CAPTURE_LAYERS[dit_idx], "t_ds": tds, "seen": sorted(seen),
               "unseen": sorted(unseen), "n_train": len(train), "pathways": {}}
    traj = {}
    for pw in pathways:
        tr = make_seqs(train, pw, dit_idx)
        if not tr:
            print(f"  [skip] {pw}: no features"); continue
        mu, sd = standardizer(tr)
        tr = make_seqs(train, pw, dit_idx, mu, sd)
        st = make_seqs(seen_test, pw, dit_idx, mu, sd)
        ut = make_seqs(unseen_test, pw, dit_idx, mu, sd)
        input_dim = tr[0][0].shape[1]
        print(f"[train] pathway={pw} dim={input_dim} n_train={len(tr)}")
        model = train_lstm(tr, input_dim, args.epochs, args.lr, args.hidden, device, args.seed,
                            args.lambda_reg, args.grad_clip)
        dt_seen = decision_time_auroc(model, st, tds, device)
        dt_unseen = decision_time_auroc(model, ut, tds, device)
        # CP: seen-test 성공 절반으로 보정, 나머지+실패로 seen-eval. functional(시간가변)이 헤드라인.
        st_succ = [s for s in st if s[1] == 0]
        st_fail = [s for s in st if s[1] == 1]
        order = np.random.default_rng(args.seed + 1).permutation(len(st_succ))
        half = len(order) // 2
        cal = [st_succ[i] for i in order[:half]]
        seen_eval = [st_succ[i] for i in order[half:]] + st_fail
        fcp_seen = functional_cp_metrics(model, tr, cal, seen_eval, device, alphas, L_max)
        fcp_unseen = functional_cp_metrics(model, tr, cal, ut, device, alphas, L_max)
        cp_unseen_const = cp_metrics(model, cal, ut, device, alphas)  # 상수임계(참고)
        results["pathways"][pw] = {
            "decision_time_seen": dt_seen,
            "decision_time_unseen": dt_unseen,
            "cp_seen": fcp_seen,          # functional (헤드라인; _plot이 사용)
            "cp_unseen": fcp_unseen,
            "cp_unseen_const": cp_unseen_const,
        }
        traj[pw] = {"seen": mean_trajectory(model, st, device),
                    "unseen": mean_trajectory(model, ut, device)}
        print(f"    [{pw}] AUROC " + " ".join(
            f"t{t}:un={dt_unseen.get(str(t),{}).get('auroc',float('nan')):.2f}" for t in tds))
        for a in sorted(fcp_unseen):
            r = fcp_unseen[a]
            print(f"    [{pw}] funcCP α={a} unseen: TPR={r['tpr']} FPR={r['fpr']} bal-acc={r['bal_acc']} "
                  f"T-det={r['mean_tdet_fired']} | δ_t∈{r['delta_t_range']}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "pathway_lstm_detector.json").write_text(json.dumps(results, indent=2))
    _plot(results, traj, tds, out)
    print(f"[done] -> {out}/")


def _plot(results, traj, tds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pws = list(results["pathways"].keys())
    # 1) decision-time AUROC (seen vs unseen, per pathway)
    fig, ax = plt.subplots(figsize=(9, 6))
    col = {"dit": "#458cd6", "vl": "#d95234", "both": "#43a047"}
    for pw in pws:
        for split, ls in (("decision_time_seen", "-"), ("decision_time_unseen", "--")):
            d = results["pathways"][pw][split]
            xs = [t for t in tds if str(t) in d]; ys = [d[str(t)]["auroc"] for t in xs]
            if xs:
                ax.plot(xs, ys, ls=ls, marker="o", color=col.get(pw, "k"),
                        label=f"{pw} {'seen-test' if 'seen' in split else 'unseen'}")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("decision time t_d (inference steps seen, causal)")
    ax.set_ylabel("LSTM AUROC (failure=positive)")
    ax.set_ylim(0.3, 1.0); ax.set_title("SAFE-LSTM decision-time detection (80/20 seen + unseen)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "lstm_decision_time.png", dpi=140); plt.close()
    # 2) per-step score trajectory (success vs failure), unseen
    fig, axes = plt.subplots(1, len(pws), figsize=(6 * len(pws), 5), squeeze=False)
    for ax, pw in zip(axes[0], pws):
        s, f = traj[pw]["unseen"]
        if s is not None: ax.plot(s, color="green", lw=2, label="success (unseen)")
        if f is not None: ax.plot(f, color="red", lw=2, label="failure (unseen)")
        ax.set_xlabel("inference step"); ax.set_ylabel("LSTM failure score")
        ax.set_ylim(0, 1); ax.set_title(f"{pw}: per-step score"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "lstm_score_trajectory.png", dpi=140); plt.close()
    # 3) CP trade-off: balanced-acc vs normalized T-det (SAFE Fig.4 style; 좌상단=좋음)
    fig, ax = plt.subplots(figsize=(8, 6))
    for pw in pws:
        for split, ls, mk in (("cp_seen", "-", "o"), ("cp_unseen", "--", "s")):
            d = results["pathways"][pw].get(split, {})
            pts = [(d[a]["mean_tdet_fail"], d[a]["bal_acc"]) for a in sorted(d)
                   if d[a].get("mean_tdet_fail") is not None and d[a].get("bal_acc") is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, ls=ls, marker=mk, color=col.get(pw, "k"),
                        label=f"{pw} {'seen' if split == 'cp_seen' else 'unseen'}")
    ax.set_xlabel("normalized T-det (failures; 낮을수록 일찍)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("SAFE-style CP trade-off (점=α; 좌상단이 좋음)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "lstm_cp_tradeoff.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
