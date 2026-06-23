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


class MLPDetector(nn.Module):
    """SAFE-MLP: per-step MLP→scalar→sigmoid (step 독립, memoryless).
    검출 score는 per-step 출력의 **시간 누적평균**(score_seq 에서 적용). cumulative=True 표식."""
    cumulative = True

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):  # [B,T,D] → [B,T] per-step sigmoid (각 step 독립 처리)
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_detector(detector_type: str, input_dim: int, hidden: int):
    return (MLPDetector(input_dim, hidden) if detector_type == "mlp"
            else LSTMDetector(input_dim=input_dim, hidden_dim=hidden))


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
               lambda_reg=1e-2, grad_clip=1.0, detector_type="lstm"):
    """SAFE 검출기 학습(lstm/mlp). per-step BCE + lambda_reg·L2(bias 제외) + grad clip."""
    torch.manual_seed(seed)
    model = build_detector(detector_type, input_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    for ep in range(epochs):
        random.shuffle(train_seqs)
        tot = 0.0
        for X, y, _L, *_ in train_seqs:
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
    raw = model(xb).squeeze(0).cpu().numpy()  # [T] per-step sigmoid
    if getattr(model, "cumulative", False):   # SAFE-MLP: 검출 score = 출력 누적평균
        raw = np.cumsum(raw) / np.arange(1, len(raw) + 1)
    return raw


def make_seqs(rolls, pathway, dit_idx, mu=None, sd=None):
    """rollout → (Xn, y, length, task, lang) per-step seq. y=1-success(실패=positive).
    lang = ep_meta.lang(instruction). 소비자는 task/lang 을 무시할 때 `*_` 로 언팩."""
    out = []
    for r in rolls:
        X = pathway_seq(r, pathway, dit_idx)
        if X is None:
            continue
        Xn = ((X - mu) / sd).astype(np.float32) if mu is not None else X.astype(np.float32)
        out.append((Xn, 1 - int(r["success"]), r["length"], r["task"], r.get("lang", "")))
    return out


def group_by_lang(seqs, min_fail=8, min_succ=8):
    """eval seq 들을 instruction(lang)별로 묶고 fail>=min_fail & succ>=min_succ 만 남김.
    seq = (Xn, y, length, task, lang), y: 1=fail/0=succ. 작은 subset(예: SlideDW in=fail4)은 제외."""
    by = defaultdict(list)
    for s in seqs:
        by[s[4]].append(s)
    out = {}
    for lang, grp in by.items():
        n_fail = sum(1 for s in grp if s[1] == 1)
        n_succ = sum(1 for s in grp if s[1] == 0)
        if n_fail >= min_fail and n_succ >= min_succ:
            out[lang] = grp
    return out


def group_by_task(seqs, min_fail=8, min_succ=8):
    """eval seq 들을 task(tuple index 3)별로 묶고 fail>=min & succ>=min 만. per-task 일반화 검증용."""
    by = defaultdict(list)
    for s in seqs:
        by[s[3]].append(s)
    out = {}
    for task, grp in by.items():
        n_fail = sum(1 for s in grp if s[1] == 1)
        n_succ = sum(1 for s in grp if s[1] == 0)
        if n_fail >= min_fail and n_succ >= min_succ:
            out[task] = grp
    return out


def _balacc(fired, y):
    """balanced accuracy = (TPR + (1-FPR))/2. fired/y: 1-D bool/int array."""
    fired = np.asarray(fired); y = np.asarray(y)
    f1, f0 = fired[y == 1], fired[y == 0]
    if f1.size == 0 or f0.size == 0:
        return float("nan")
    tpr = float(f1.mean()); fpr = float(f0.mean())
    return (tpr + (1 - fpr)) / 2


def who_first(cp_vl, cp_dit, alpha):
    """instruction별 VL vs DiT 검출 선후. cp_* = {lang: {alpha_str: {mean_tdet_fired, ...}}}.
    낮은 normalized T-det(=먼저 발화)이 'first'. 미발화(None)는 진 것으로 본다. 두 pathway 공통 lang만."""
    akey = f"{alpha:.2f}"
    out = {}
    for lang in cp_vl:
        if lang not in cp_dit:
            continue
        v = cp_vl[lang].get(akey, {}).get("mean_tdet_fired")
        d = cp_dit[lang].get(akey, {}).get("mean_tdet_fired")
        if v is None and d is None:
            first = "neither"
        elif v is None:
            first = "DiT"
        elif d is None:
            first = "VL"
        elif v < d:
            first = "VL"
        elif d < v:
            first = "DiT"
        else:
            first = "tie"
        out[lang] = {"vl_tdet": v, "dit_tdet": d, "first": first}
    return out


def _md_metric_rows(per_pathway, split, alpha_key):
    """split('seen'/'unseen') 의 instruction × pathway 행들을 markdown table 문자열로."""
    pws = [pw for pw in ("dit", "vl") if pw in per_pathway]
    langs = sorted({lang for pw in pws for lang in per_pathway[pw].get(split, {})})
    if not langs:
        return ""
    lines = [
        f"#### {split} (α={alpha_key})",
        "",
        "| instruction | pathway | TPR | FPR | bal-acc | T-det | n_fail | n_succ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lang in langs:
        for pw in pws:
            cell = per_pathway[pw].get(split, {}).get(lang, {}).get(alpha_key)
            if not cell:
                continue
            lines.append(
                f"| {lang} | {pw} | {cell.get('tpr')} | {cell.get('fpr')} | "
                f"{cell.get('bal_acc')} | {cell.get('mean_tdet_fired')} | "
                f"{cell.get('n_fail')} | {cell.get('n_succ')} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_per_instruction_md(per_instruction, detector_type):
    """per-instruction 결과 → detector_results_per_instruction.md 본문.

    per_instruction = {"alpha_whofirst": a, "pathways": {pw: {"seen":{lang:fcp}, "unseen":{lang:fcp}}}}.
    fcp = {alpha_str: {tpr, fpr, bal_acc, mean_tdet_fired, n_fail, n_succ}}.
    """
    pw = per_instruction["pathways"]
    a = per_instruction.get("alpha_whofirst", 0.3)
    akey = f"{a:.2f}"
    parts = [
        f"# Per-instruction detection ({detector_type.upper()})",
        "",
        "통짜 multi-task 검출기 그대로(재학습 X), functional-CP 평가만 `ep_meta.lang`별 분해. "
        "instruction당 fail≥8 & succ≥8 만(작은 subset 제외). 표본 작으면 노이즈 — 단정 금지.",
        "",
    ]
    for split in ("unseen", "seen"):
        block = _md_metric_rows(pw, split, akey)
        if block:
            parts.append(block)
    # who-first (VL vs DiT) — unseen 우선, 없으면 seen
    for split in ("unseen", "seen"):
        cp_vl = pw.get("vl", {}).get(split, {})
        cp_dit = pw.get("dit", {}).get(split, {})
        wf = who_first(cp_vl, cp_dit, a)
        if not wf:
            continue
        parts += [
            f"#### who fires first — {split} (α={akey}, normalized T-det)",
            "",
            "| instruction | VL T-det | DiT T-det | first |",
            "|---|---|---|---|",
        ]
        for lang in sorted(wf):
            w = wf[lang]
            parts.append(f"| {lang} | {w['vl_tdet']} | {w['dit_tdet']} | {w['first']} |")
        parts.append("")
    return "\n".join(parts)


def decision_time_auroc(model, seqs, tds, device):
    """t_d별 living rollout(length>t_d)에서 LSTM score[t_d-1] AUROC."""
    res = {}
    pre = [(score_seq(model, X, device), y, L) for X, y, L, *_ in seqs]
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
        for _, yy, ln, *_ in seqs:
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
    cal_max = np.array([score_seq(model, X, device).max() for X, y, _L, *_ in cal_seqs if y == 0])
    if len(cal_max) < 3:
        return {}
    ev = [(score_seq(model, X, device), y, L) for X, y, L, *_ in eval_seqs]
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
    tr = [_pad_to(score_seq(model, X, device), L) for X, y, _L, *_ in train_succ if y == 0]
    cal = [_pad_to(score_seq(model, X, device), L) for X, y, _L, *_ in cal_succ if y == 0]
    if len(tr) < 3 or len(cal) < 3:
        return {}
    tr, cal = np.stack(tr), np.stack(cal)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0, ddof=1) + 1e-8
    excursion = np.max((cal - mu) / sd, axis=1)
    ev = [(score_seq(model, X, device), y, Lr) for X, y, Lr, *_ in eval_seqs]
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


def functional_cp_band(model, train_succ, cal_succ, device, alpha, L):
    """SAFE functional CP 밴드 δ_t (시간가변) [L] 반환 (영상 검출 임계용)."""
    tr = [_pad_to(score_seq(model, X, device), L) for X, y, _L, *_ in train_succ if y == 0]
    cal = [_pad_to(score_seq(model, X, device), L) for X, y, _L, *_ in cal_succ if y == 0]
    if len(tr) < 3 or len(cal) < 3:
        return np.full(L, 0.5)
    tr, cal = np.stack(tr), np.stack(cal)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0, ddof=1) + 1e-8
    bw = float(np.quantile(np.max((cal - mu) / sd, axis=1), 1 - alpha))
    return mu + bw * sd


def per_instruction_cp(model, train_succ, cal_succ, eval_seqs, device, alphas, L,
                       min_fail=8, min_succ=8):
    """평가만 instruction(lang)별로 분해. 통짜 model·pooled cal 그대로,
    eval_seqs 만 lang group → 각 subset 에 functional_cp_metrics. {lang: fcp}."""
    out = {}
    for lang, grp in group_by_lang(eval_seqs, min_fail, min_succ).items():
        out[lang] = functional_cp_metrics(model, train_succ, cal_succ, grp, device, alphas, L)
    return out


def per_task_cp(model, train_succ, cal_succ, eval_seqs, device, alphas, L,
                min_fail=8, min_succ=8):
    """평가만 task별 분해(통짜 model·pooled cal 그대로). {task: fcp}. per-task 일반화 검증."""
    out = {}
    for task, grp in group_by_task(eval_seqs, min_fail, min_succ).items():
        out[task] = functional_cp_metrics(model, train_succ, cal_succ, grp, device, alphas, L)
    return out


def _fired_labels(model, train_succ, cal_succ, eval_seqs, device, alpha, L):
    """functional-CP 밴드(alpha) 하에서 eval rollout별 (fired bool, y). 유의성 계산용.
    functional_cp_metrics 와 동일한 밴드 정의(δ_t = μ_t + bw·σ_t)."""
    tr = [_pad_to(score_seq(model, X, device), L) for X, y, _L, *_ in train_succ if y == 0]
    cal = [_pad_to(score_seq(model, X, device), L) for X, y, _L, *_ in cal_succ if y == 0]
    if len(tr) < 3 or len(cal) < 3:
        return None
    tr, cal = np.stack(tr), np.stack(cal)
    mu = tr.mean(axis=0); sd = tr.std(axis=0, ddof=1) + 1e-8
    bw = float(np.quantile(np.max((cal - mu) / sd, axis=1), 1 - alpha))
    delta = mu + bw * sd
    fired, ys = [], []
    for X, y, _Lr, *_ in eval_seqs:
        sc = score_seq(model, X, device); n = min(len(sc), L)
        fired.append(bool(np.any(sc[:n] > delta[:n]))); ys.append(int(y))
    return np.array(fired), np.array(ys)


def cp_significance(model, train_succ, cal_succ, eval_seqs, device, alpha, L,
                    n_perm=200, n_boot=1000, seed=0):
    """헤드라인 bal-acc 의 bootstrap 95% CI + label-permutation null/p-value.
    "우연·노이즈 아님"을 못박기 위함. fired/label 은 한 번만 계산 후 재표집."""
    fl = _fired_labels(model, train_succ, cal_succ, eval_seqs, device, alpha, L)
    if fl is None:
        return {}
    fired, y = fl
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return {}
    obs = _balacc(fired, y)
    rng = np.random.default_rng(seed)
    boots = np.array([b for b in (_balacc(fired[i], y[i]) for i in
                      (rng.integers(0, len(y), len(y)) for _ in range(n_boot))) if b == b])
    nulls = np.array([b for b in (_balacc(fired, rng.permutation(y)) for _ in range(n_perm)) if b == b])
    return {
        "alpha": alpha, "bal_acc": round(float(obs), 3),
        "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                 round(float(np.percentile(boots, 97.5)), 3)] if boots.size else None,
        "null_mean": round(float(nulls.mean()), 3) if nulls.size else None,
        "null_p95": round(float(np.percentile(nulls, 95)), 3) if nulls.size else None,
        "p_value": round(float((nulls >= obs).mean()), 4) if nulls.size else None,
        "n_fail": int((y == 1).sum()), "n_succ": int((y == 0).sum()),
    }


def mean_trajectory(model, seqs, device, max_t=40):
    """성공/실패 평균 per-step score 궤적 (padding: 마지막 score 유지)."""
    succ, fail = [], []
    for X, y, L, *_ in seqs:
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
    ap.add_argument("--detector-type", default="lstm", choices=("lstm", "mlp"),
                    help="lstm=recurrent / mlp=per-step MLP+출력 누적평균(SAFE-MLP)")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5", help="CP 유의수준(FPR 목표)")
    ap.add_argument("--split-instruction", action="store_true",
                    help="통짜 검출기 그대로, functional-CP 평가만 ep_meta.lang(instruction)별 분해")
    ap.add_argument("--min-instr-fail", type=int, default=8,
                    help="instruction 평가 포함 최소 fail rollout 수")
    ap.add_argument("--min-instr-succ", type=int, default=8,
                    help="instruction 평가 포함 최소 succ rollout 수")
    ap.add_argument("--whofirst-alpha", type=float, default=0.3,
                    help="VL vs DiT 검출 선후 비교에 쓸 α (T-det)")
    ap.add_argument("--per-task", action="store_true",
                    help="평가만 task별 분해(per-task 일반화 검증)")
    ap.add_argument("--min-task-fail", type=int, default=8, help="per-task 포함 최소 fail")
    ap.add_argument("--min-task-succ", type=int, default=8, help="per-task 포함 최소 succ")
    ap.add_argument("--n-perm", type=int, default=0,
                    help="헤드라인 bal-acc label-permutation null 횟수(0=off)")
    ap.add_argument("--null-alpha", type=float, default=0.1,
                    help="유의성(null/CI) 계산에 쓸 헤드라인 α")
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
    if args.split_instruction:
        results["per_instruction"] = {"alpha_whofirst": args.whofirst_alpha,
                                      "min_fail": args.min_instr_fail,
                                      "min_succ": args.min_instr_succ, "pathways": {}}
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
                            args.lambda_reg, args.grad_clip, args.detector_type)
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
        # length-only baseline(total length = confound 천장; pathway 무관 → 한 번만)
        if "length_baseline" not in results:
            results["length_baseline"] = {
                "note": "total-length AUROC(non-causal, 미래정보 사용 = confound 상한). causal 길이예측은 fixed-t_d서 chance.",
                "seen": length_auroc(seen_eval, tds), "unseen": length_auroc(ut, tds)}
        # 유의성: 헤드라인 α 에서 bal-acc bootstrap CI + permutation null
        if args.n_perm > 0:
            results["pathways"][pw]["sig_unseen"] = cp_significance(
                model, tr, cal, ut, device, args.null_alpha, L_max, args.n_perm, seed=args.seed)
            results["pathways"][pw]["sig_seen"] = cp_significance(
                model, tr, cal, seen_eval, device, args.null_alpha, L_max, args.n_perm, seed=args.seed)
            s = results["pathways"][pw]["sig_unseen"]
            if s:
                print(f"    [{pw}] sig(unseen,α={args.null_alpha}): bal-acc={s['bal_acc']} "
                      f"CI95={s['ci95']} null={s['null_mean']} p={s['p_value']}")
        if args.per_task:
            pt_seen = per_task_cp(model, tr, cal, seen_eval, device, alphas, L_max,
                                  args.min_task_fail, args.min_task_succ)
            pt_unseen = per_task_cp(model, tr, cal, ut, device, alphas, L_max,
                                    args.min_task_fail, args.min_task_succ)
            results.setdefault("per_task", {"min_fail": args.min_task_fail,
                                            "min_succ": args.min_task_succ, "pathways": {}})
            results["per_task"]["pathways"][pw] = {"seen": pt_seen, "unseen": pt_unseen}
            print(f"    [{pw}] per-task: seen={len(pt_seen)} unseen={len(pt_unseen)}")
        if args.split_instruction:
            pi_seen = per_instruction_cp(model, tr, cal, seen_eval, device, alphas, L_max,
                                         args.min_instr_fail, args.min_instr_succ)
            pi_unseen = per_instruction_cp(model, tr, cal, ut, device, alphas, L_max,
                                           args.min_instr_fail, args.min_instr_succ)
            results["per_instruction"]["pathways"][pw] = {"seen": pi_seen, "unseen": pi_unseen}
            print(f"    [{pw}] per-instruction: seen={len(pi_seen)} unseen={len(pi_unseen)} "
                  f"(fail≥{args.min_instr_fail} & succ≥{args.min_instr_succ})")
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
    if args.split_instruction and results.get("per_instruction", {}).get("pathways"):
        md = render_per_instruction_md(results["per_instruction"], args.detector_type)
        (out / "detector_results_per_instruction.md").write_text(md)
        print(f"[done] per-instruction md -> {out}/detector_results_per_instruction.md")
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
