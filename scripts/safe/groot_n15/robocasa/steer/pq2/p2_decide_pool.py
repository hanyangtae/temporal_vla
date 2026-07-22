"""P2 pool(cross_scene/grand) 선택 판정 — p2_decide 와 동일 규칙, scene 분배 합산.

base = 각 소속 scene select-half 앞 K판의 수집 라벨 합 (rollout 불필요).
arm  = p2_pool/<scope>/<arm> 아래 전 scene pkl 합산.

usage: p2_decide_pool.py --scope cross_scene_bread --scenes A B C D --k 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
G15 = REPO / "outputs/eval/robocasa/groot_n15"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    base_s = base_n = 0
    for sc in args.scenes:
        split = json.loads((G15 / "pq2_manifests" / sc / "split.json").read_text())
        eps = sorted(split["select_half"])[: args.k]
        base_s += sum(int(split["labels"][str(e)]) for e in eps)
        base_n += len(eps)
    root = G15 / "steer_eval_pq2/p2_pool" / args.scope
    rows = []
    for armd in sorted(root.glob("p2_*")):
        s = len(list(armd.rglob("*succ1.pkl"))); f = len(list(armd.rglob("*succ0.pkl")))
        m = re.match(r"p2_(single_L(\d+)|multi_[\d_]+)_b(\d+)", armd.name)
        beta = float("0." + m.group(3).lstrip("0")) if m else None
        nlayer = 1 if (m and m.group(1).startswith("single")) else 3
        rows.append({"tag": armd.name, "succ": s, "n": s + f, "beta": beta, "n_layers": nlayer,
                     "eliminated": s <= base_s - 2})
    survivors = [r for r in rows if not r["eliminated"] and r["n"] >= base_n]
    pick, fallback = None, False
    if survivors:
        top = max(r["succ"] for r in survivors)
        n = survivors[0]["n"]
        p = top / n if n else 0
        se_eps = (n * p * (1 - p)) ** 0.5
        tied = [r for r in survivors if r["succ"] >= top - se_eps]
        pick = sorted(tied, key=lambda r: (r["beta"], r["n_layers"]))[0]
    else:
        # 생존 0 = 전 후보가 base 대비 해악 — 보수 규칙의 연장으로 가장 보수적 후보를
        # fallback 선택 (선택 실패 자체가 이 tier 의 정보; held-out 이 최종 판정)
        full = [r for r in rows if r["n"] >= base_n]
        if full:
            pick = sorted(full, key=lambda r: (r["beta"], r["n_layers"]))[0]
            fallback = True
    report = {"scope": args.scope, "scenes": args.scenes, "k": args.k,
              "base": {"succ": base_s, "n": base_n}, "rows": rows, "selected": pick,
              "fallback_conservative": fallback,
              "rule": "하방-위험(base−2 탈락, n 미달 제외)→SE내 동률 보수(β↓,layer↓); 생존 0 이면 최보수 fallback"}
    out = G15 / "pq2_analysis" / f"selection_report_pool_{args.scope}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[{args.scope}] base={base_s}/{base_n} → "
          + (f"선택: {pick['tag']}{' (최보수 fallback — 전 후보 base 미달)' if fallback else ''}"
             if pick else "판정 불가 (표본 미달)"))


if __name__ == "__main__":
    main()
