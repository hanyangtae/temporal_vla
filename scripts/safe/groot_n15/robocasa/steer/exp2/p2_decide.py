"""P2 선택 판정 (하방-위험 규칙, Gate1 합의) — 수치는 selection_report 에만 (본문 보고 금지).

base = select-half 의 수집 시 라벨 (split.json — fresh rollout 불필요: 수집 자체가 base).
후보 탈락: succ(후보) ≤ succ(base) − 2 episode.
잔여 동률(SE 이내): 보수 우선 = 작은 β → 적은 layer 수.

usage: p2_decide.py --scene <scene> [--arms p2_single_L4_b01 ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
G15 = REPO / "outputs/eval/robocasa/groot_n15"


def arm_sr(scene: str, tag: str, task: str):
    d = G15 / "steer_eval_exp2/p2" / scene / tag / "raw_rollouts" / task / scene
    s = len(list(d.glob("*succ1.pkl"))); f = len(list(d.glob("*succ0.pkl")))
    return s, s + f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()
    split = json.loads((G15 / "exp2_manifests" / args.scene / "split.json").read_text())
    sel = split["select_half"]
    labels = split["labels"]
    base_s = sum(int(labels[str(e)]) for e in sel)
    task = args.task or ("PickPlaceCounterToCabinet" if args.scene.startswith("ppcc") else "PickPlaceCounterToStove")
    arms = args.arms or sorted(p.name for p in (G15 / "steer_eval_exp2/p2" / args.scene).glob("p2_*"))
    rows = []
    for tag in arms:
        s, n = arm_sr(args.scene, tag, task)
        m = re.match(r"p2_(single_L(\d+)|multi_[\d_]+)_b(\d+)", tag)
        beta = float("0." + m.group(3).lstrip("0")) if m else None
        nlayer = 1 if (m and m.group(1).startswith("single")) else 3
        rows.append({"tag": tag, "succ": s, "n": n, "beta": beta, "n_layers": nlayer,
                     "eliminated": s <= base_s - 2})
    survivors = [r for r in rows if not r["eliminated"] and r["n"] > 0]
    # 보수 tie-break: 최고 succ 에서 SE(≈√(npq)) 이내인 후보 중 (작은 β, 적은 layer)
    pick = None
    if survivors:
        top = max(r["succ"] for r in survivors)
        n = survivors[0]["n"]
        p = top / n if n else 0
        se_eps = (n * p * (1 - p)) ** 0.5
        tied = [r for r in survivors if r["succ"] >= top - se_eps]
        pick = sorted(tied, key=lambda r: (r["beta"], r["n_layers"]))[0]
    report = {"scene": args.scene, "base_select_half": {"succ": base_s, "n": len(sel)},
              "rows": rows, "selected": pick, "rule": "하방-위험(base−2 탈락)→SE내 동률 보수(β↓,layer↓)"}
    out = G15 / "exp2_analysis" / f"selection_report_{args.scene}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[{args.scene}] base(select-half)={base_s}/{len(sel)} → "
          + (f"선택: {pick['tag']}" if pick else "생존 후보 없음(전부 base−2 이하 — positive-only/재검 필요)"))
    print(f"  (수치 상세는 {out.name} — 본문 보고 금지)")


if __name__ == "__main__":
    main()
