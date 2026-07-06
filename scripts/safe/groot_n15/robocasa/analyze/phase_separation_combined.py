"""potato/apple 두 seed(scene) 합친 combined succ/fail 분리도 + **seed(scene) confound 진단**.

사용자 요청: potato={ppcc_potato(seed200010) + ppcc_potato_s2(seed200019)},
apple={ppcs_apple(seed100050) + ppcs_apple_s2(seed100074)} 를 각 30 rollout 으로 묶어 분석.

⚠️ 두 seed 를 합치면 scene(layout/object placement)이 2개 섞인다. 특히 한 seed 가 all-succ,
다른 seed 가 all-fail 이면 succ/fail 분리 = **seed(scene) 분리**가 되어 outcome 이 아니다.
그래서 여기서는 (1) combined succ/fail AUROC 와 함께 (2) **succ↔seed 교차표** + (3) 성공만으로의
**seed 분리 AUROC**(scene 가 latent 에서 얼마나 갈라지나)를 같이 출력해 confound 를 노출한다.

phase_separation.py 의 검증된 primitive 재사용.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from phase_separation import (  # noqa: E402
    equal_budget_pool,
    load_rollout,
    loo_auroc,
    perm_null_upper,
    phase_records,
    rank_auroc,
)

RUN = "outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/raw_rollouts"
GROUPS = {
    "potato_combined": [
        ("PickPlaceCounterToCabinet/ppcc_potato", 200010),
        ("PickPlaceCounterToCabinet/ppcc_potato_s2", 200019),
    ],
    "apple_combined": [
        ("PickPlaceCounterToStove/ppcs_apple", 100050),
        ("PickPlaceCounterToStove/ppcs_apple_s2", 100074),
    ],
}
PHASES = ("reach-to-object", "transport")
MIN_PER_CLASS = 3


def load_group(specs):
    rolls = []
    for subdir, seed in specs:
        for p in sorted((Path(RUN) / subdir).glob("*.pkl")):
            r = load_rollout(p)
            r["seed"] = seed
            rolls.append(r)
    return rolls


def sep_per_phase(rolls, layer_keys, label_fn, min_per_class=MIN_PER_CLASS):
    """label_fn(roll)->0/1. per phase per layer: equal-budget LOO-AUROC + null."""
    out = {}
    for phase in PHASES:
        present = [(r, phase_records(r, phase, layer_keys[0])) for r in rolls]
        present = [(r, rc) for r, rc in present if len(rc) > 0]
        ys = np.array([label_fn(r) for r, _ in present])
        if len(np.unique(ys)) < 2 or np.bincount(ys).min() < min_per_class:
            out[phase] = {"skip": f"n1={int((ys==1).sum())} n0={int((ys==0).sum())}"}
            continue
        budget = min(len(rc) for _, rc in present)
        lens = np.array([len(rc) for _, rc in present], float)
        entry = {"n1": int((ys == 1).sum()), "n0": int((ys == 0).sum()),
                 "budget": int(budget), "length_auroc": rank_auroc(lens, ys), "layers": {}}
        rp = [r for r, _ in present]
        for lk in layer_keys:
            X, ok = [], True
            for r in rp:
                rc = phase_records(r, phase, lk)
                if len(rc) < budget:
                    ok = False; break
                X.append(equal_budget_pool(rc, budget))
            if not ok:
                continue
            X = np.stack(X)
            a = loo_auroc(X, ys)
            if a is None:
                continue
            entry["layers"][str(lk)] = {"auroc": a, "null95": perm_null_upper(X, ys)}
        out[phase] = entry
    return out


def main():
    sample = load_rollout(sorted((Path(RUN) / "PickPlaceCounterToCabinet/ppcc_potato").glob("*.pkl"))[0])
    cap = sample["capture_layers"]
    layer_keys = list(range(sample["dit"].shape[1])) + (["VL"] if sample["vl"] is not None else [])
    labels = {i: f"DiT-L{cap[i]}" for i in range(len(cap))}; labels["VL"] = "VL"

    results = {"capture_layers": cap, "groups": {}}
    for gname, specs in GROUPS.items():
        rolls = load_group(specs)
        seeds = sorted({r["seed"] for r in rolls})
        succ = np.array([r["success"] for r in rolls])
        seed_arr = np.array([r["seed"] for r in rolls])
        # succ↔seed 교차표 (scene confound 노출)
        xtab = {int(s): {"succ": int(((seed_arr == s) & (succ == 1)).sum()),
                         "fail": int(((seed_arr == s) & (succ == 0)).sum())} for s in seeds}
        print(f"\n=== {gname}: {len(rolls)} rollouts, succ={int(succ.sum())} fail={int((1-succ).sum())} ===")
        print(f"  succ↔seed 교차표: {xtab}")

        # (1) combined succ/fail 분리도
        succfail = sep_per_phase(rolls, layer_keys, lambda r: r["success"])
        # (3) 성공만으로의 seed(scene) 분리도 — scene 가 latent 에서 갈라지나
        succ_only = [r for r in rolls if r["success"] == 1]
        seed_sep = None
        if len(seeds) == 2 and sum(r["seed"] == seeds[0] for r in succ_only) >= MIN_PER_CLASS \
           and sum(r["seed"] == seeds[1] for r in succ_only) >= MIN_PER_CLASS:
            seed_sep = sep_per_phase(succ_only, layer_keys, lambda r: int(r["seed"] == seeds[1]))

        results["groups"][gname] = {"xtab": xtab, "succ_fail": succfail, "seed_scene_sep_succ_only": seed_sep}

        for phase in PHASES:
            e = succfail.get(phase, {})
            if "layers" not in e:
                print(f"  [succ/fail {phase}] {e.get('skip')}"); continue
            best = max(e["layers"].items(), key=lambda kv: abs(kv[1]["auroc"] - 0.5))
            print(f"  [succ/fail {phase}] budget={e['budget']} length_auroc={e['length_auroc']:.3f} "
                  f"best={labels[int(best[0]) if best[0]!='VL' else 'VL'] if best[0]!='VL' else 'VL'} "
                  f"auroc={best[1]['auroc']:.3f} null={best[1]['null95']:.3f}")
        if seed_sep:
            for phase in PHASES:
                e = seed_sep.get(phase, {})
                if "layers" in e:
                    best = max(e["layers"].items(), key=lambda kv: abs(kv[1]["auroc"] - 0.5))
                    print(f"  [SEED(scene) sep, succ-only, {phase}] best auroc={best[1]['auroc']:.3f} "
                          f"null={best[1]['null95']:.3f}  (높으면 latent가 scene을 강하게 인코딩=confound 증거)")

    out = Path(RUN).parent / "analysis" / "phase_separation_combined"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {out/'results.json'}")


if __name__ == "__main__":
    main()
