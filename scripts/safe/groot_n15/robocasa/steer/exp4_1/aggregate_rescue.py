#!/usr/bin/env python3
"""exp4-1: rescue rate 집계 + primary 검정 (setM vs setM_pl paired exact McNemar, 4-task Holm).

ITT 규칙 (24a §1): 분모 = t0 manifest 의 feasible 행 전체. 미주석(t0_record=NA) episode 는
**비구제로 계상** (별도로 주석 부분집합 rate 병기). 구제 = arm rollout episode_success==1.

pairing: 같은 (cell, episode_idx, inference_seed) 의 setM vs setM_pl 결과 쌍.
McNemar exact: discordant b(setM만 성공)·c(pl만 성공), p = 2·P(Bin(b+c,½) ≤ min(b,c)) capped 1.
Holm: 사전등록 task family 4건 = bread / beer / drawer(좌우 pooled) / mixer (--task-map).

사용:
  python aggregate_rescue.py --t0-manifest <t0_manifest.tsv> \
      --arm setM:<루트> --arm setM_pl:<루트> [--arm conceptor:<루트> --arm A0:<루트>] \
      [--setm-npz-root <npz 루트 — seen/unseen 층화용 metadata>] --out <json>
  python aggregate_rescue.py --self-test   # McNemar 단위검증
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """양측 exact McNemar p (discordant only). b+c==0 이면 p=1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k), X~Bin(n, 1/2)
    p_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_tail)


def holm(pvals: dict) -> dict:
    """Holm step-down 보정 (key→adj_p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run_max = {}, 0.0
    for i, (k, p) in enumerate(items):
        run_max = max(run_max, (m - i) * p)
        adj[k] = min(1.0, run_max)
    return adj


def load_t0(path: Path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(header, ln.split("\t")))
        if str(r.get("feasible", "1")) != "1":
            continue  # 기하 불가 seed 는 분모 제외 (fit·eval 양쪽 동일 — 공유문서 §5)
        rows.append(r)
    return rows


def scan_arm(root: Path):
    """arm rollout 루트 → {(cell, episode_idx, inference_seed): success}."""
    out = {}
    for sc in sorted(root.rglob("*--succ*.json")):
        if "quarantine" in sc.parts:
            continue
        d = json.loads(sc.read_text())
        cell = d.get("cell_id") or sc.parent.name
        key = (cell, int(d.get("episode_idx", -1)), int(d.get("inference_seed", -1)))
        out[key] = int(d.get("episode_success", 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-manifest", type=Path, action="append",
                    help="t0_manifest.tsv (반복 가능 — legacy + mixer 별도 파일)")
    ap.add_argument("--arm", action="append", default=[], help="'이름:rollout루트' 반복")
    # 사전등록 task 4종 = bread / beer / OpenDrawer(좌우 cell 풀링) / mixer (Gate2 P1-4 —
    # drawer 를 cell 별로 나누면 Holm family 5가 되고 mixer 가 누락됨)
    ap.add_argument("--task-map", default="pq3_ppcc_bread:bread,pq3_ppcc_beer:beer,"
                    "pq3_drawer_left:drawer,pq3_drawer_right:drawer,exp41_mixer:mixer")
    ap.add_argument("--setm-npz-root", type=Path, default=None,
                    help="setM metadata 의 eval_targets seen/unseen 층화용")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        # 알려진 값: b=8, c=2 → p = 2*P(Bin(10,.5)<=2) = 2*(1+10+45)/1024 = 0.109375
        assert abs(mcnemar_exact(8, 2) - 0.109375) < 1e-9
        assert mcnemar_exact(0, 0) == 1.0
        assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-9  # 대칭 → p=2*P(<=5)>1 → cap 1
        # Holm: p=[.01,.04,.03,.5] → adj=[.04,.09,.09,.5]
        adj = holm({"a": .01, "b": .04, "c": .03, "d": .5})
        assert abs(adj["a"] - .04) < 1e-9 and abs(adj["d"] - .5) < 1e-9
        assert abs(adj["b"] - .09) < 1e-9 and abs(adj["c"] - .09) < 1e-9
        print("SELF-TEST PASS")
        return

    assert args.t0_manifest and args.arm, "--t0-manifest / --arm 필요"
    t0rows = [r for p in args.t0_manifest for r in load_t0(p)]
    task_of = dict(kv.split(":") for kv in args.task_map.split(","))
    arms = {}
    for spec in args.arm:
        name, root = spec.split(":", 1)
        arms[name] = scan_arm(Path(root))

    # seen/unseen 층화 플래그 (setM metadata)
    seen_flag = {}
    if args.setm_npz_root:
        for meta_p in args.setm_npz_root.glob("*/setM_permanent/steer/dit_L*/metadata.json"):
            meta = json.loads(meta_p.read_text())
            for t in meta.get("eval_targets", []):
                seen_flag[(meta["cell"], int(t["episode_idx"]), int(t["inference_seed"]))] = \
                    bool(t["seen_scene"])

    report = {"tasks": {}, "arms": sorted(arms)}
    pvals = {}
    for task in sorted(set(task_of.values())):
        cells = [c for c, t in task_of.items() if t == task]
        rows = [r for r in t0rows if r["cell"] in cells]
        keys = [(r["cell"], int(r["episode_idx"]), int(r["inference_seed"])) for r in rows]
        annotated = [k for r, k in zip(rows, keys) if r["t0_record"] != "NA"]
        entry = {"n_itt": len(rows), "n_annotated": len(annotated), "per_arm": {}}
        for name, res in arms.items():
            resc_itt = sum(res.get(k, 0) for k in keys)  # 미주석/미실행 = 0(비구제)
            resc_ann = sum(res.get(k, 0) for k in annotated)
            per = {
                "rescued_itt": resc_itt, "rate_itt": resc_itt / len(rows) if rows else None,
                "rescued_annotated": resc_ann,
                "rate_annotated": resc_ann / len(annotated) if annotated else None,
                "n_rolled": sum(k in res for k in keys),
            }
            if seen_flag:
                for lab, want in (("seen", True), ("unseen", False)):
                    sk = [k for k in annotated if seen_flag.get(k) is want]
                    per[f"rescued_{lab}"] = sum(res.get(k, 0) for k in sk)
                    per[f"n_{lab}"] = len(sk)
            entry["per_arm"][name] = per
        if "setM" in arms and "setM_pl" in arms:
            b = sum(1 for k in annotated
                    if arms["setM"].get(k, 0) == 1 and arms["setM_pl"].get(k, 0) == 0)
            c = sum(1 for k in annotated
                    if arms["setM"].get(k, 0) == 0 and arms["setM_pl"].get(k, 0) == 1)
            entry["mcnemar"] = {"b_setM_only": b, "c_pl_only": c, "p": mcnemar_exact(b, c)}
            pvals[task] = entry["mcnemar"]["p"]
        report["tasks"][task] = entry
    if pvals:
        adj = holm(pvals)
        for task in pvals:
            report["tasks"][task]["mcnemar"]["p_holm"] = adj[task]

    txt = json.dumps(report, indent=2, ensure_ascii=False)
    print(txt)
    if args.out:
        args.out.write_text(txt)
        print(f"[wrote] {args.out}")
    print("[주의] 보고 전 confound-audit skill 경유 (24 공유문서 §5) — "
          "primary contrast=setM vs setM_pl paired McNemar, 4-task Holm 만 pre-registered")


if __name__ == "__main__":
    main()
