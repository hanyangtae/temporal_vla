#!/usr/bin/env python3
"""Plan target-conditioned LR comparisons with identical unique episode budgets.

No model, GPU, or third-party dependency is needed. Inputs use the rollout index
schema, plus an explicit original-instruction -> canonical family/side mapping.
Only the target side's current failed episodes enter either arm. All successes
at the target jitter and the other side's current failures are excluded.

The two arms have identical success/failure x jitter counts; the mixed arm has
equal L/R counts within each outcome. Target-side mixed examples are also in the
single-side arm. Calibration reuses each arm's selected successes under the same rule and count;
it is not held out, and does not add episodes to the declared budget.
Shard ep_id is positional: this planner NEVER guesses it from an index ordering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class InfeasiblePlan(ValueError):
    """A requested exact allocation has insufficient unique support."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]):
        super().__init__(reason)
        self.diagnostics = {"status": "infeasible", "reason": reason, **diagnostics}


def _rank(seed: int, repeat: int, salt: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{repeat}|{salt}|{value}".encode()).hexdigest()


def normalize_records(rows: list[dict[str, Any]], mapping: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Keep mapped base rollouts; preserve source keys and require explicit sides.

    Mapping entries require family, side ('L'/'R'), canonical_instruction. Unmapped
    instructions are ignored so a full collection index can be supplied. Missing
    PKLs and non-base arms are ignored. Duplicate coordinate identities raise.
    Additional input fields, including provenance checksums, are preserved.
    """
    for original, entry in mapping.items():
        if entry.get("side") not in ("L", "R") or not entry.get("family") or not entry.get("canonical_instruction"):
            raise ValueError(f"Explicit family/side/canonical_instruction required: {original}")
    result, seen = [], set()
    for row in rows:
        original = str(row.get("original_key", row.get("grid_instruction", "")))
        if original not in mapping or str(row.get("armsig", "base")) != "base":
            continue
        if str(row.get("has_pkl", "1")) not in ("1", "True", "true"):
            continue
        entry = mapping[original]
        rec = dict(row)
        for field in ("scene_idx", "jitter_idx", "noise_idx", "layout_id", "style_id", "env_seed", "success"):
            try:
                rec[field] = int(row[field])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid/missing {field}: {original}") from exc
        if rec["success"] not in (0, 1):
            raise ValueError(f"Nonbinary success: {original}")
        rec.update(original_key=original, family=entry["family"], canonical_side=entry["side"],
                   canonical_instruction=entry["canonical_instruction"])
        identity = [str(row.get("plan_id", "")), original, rec["scene_idx"], rec["jitter_idx"], rec["noise_idx"]]
        uid = json.dumps(identity, separators=(",", ":"))
        if uid in seen:
            raise ValueError(f"Duplicate unique episode identity: {uid}")
        seen.add(uid)
        rec["episode_uid"] = uid
        rec["coordinate_key"] = f"{original}/s{rec['scene_idx']}/j{rec['jitter_idx']}/n{rec['noise_idx']}"
        rec["fit_ep_id"] = row.get("fit_ep_id", "")
        rec["fit_ep_id_status"] = "provided_unverified" if rec["fit_ep_id"] != "" else "requires_shard_lookup"
        result.append(rec)
    return sorted(result, key=lambda r: r["episode_uid"])


def _allocate(capacity: dict[tuple[str, int, int], int], target_side: str,
              target_jitter: int, success: int, total: int, anchor_count: int,
              seed: int, repeat: int) -> dict[int, tuple[int, int]] | None:
    """DP exact equal-side outcome allocation, minimizing jitter concentration.

    Values are (target-side mixed count, other-side mixed count). Their sum is
    also the corresponding single-side quota, bounded by target-side capacity.
    """
    other = "R" if target_side == "L" else "L"
    half = total // 2
    if total % 2 or anchor_count > half or anchor_count > capacity.get((target_side, success, target_jitter), 0):
        return None
    historical = sorted({j for _, label, j in capacity if label == success and j != target_jitter})
    # Fixed current-failure count is identical in both arms; no current successes.
    initial = {target_jitter: (anchor_count, 0)} if anchor_count else {}
    states: dict[tuple[int, int], tuple[int, dict[int, tuple[int, int]]]] = {(anchor_count, 0): (0, initial)}
    for jitter in historical:
        a = capacity.get((target_side, success, jitter), 0)
        b = capacity.get((other, success, jitter), 0)
        options = [(x, y) for x in range(min(a, half) + 1)
                   for y in range(min(b, half, a - x) + 1)]
        options.sort(key=lambda xy: _rank(seed, repeat, f"quota:{success}:{jitter}", str(xy)))
        updated = {}
        for (used_a, used_b), (cost, allocation) in states.items():
            for x, y in options:
                key = used_a + x, used_b + y
                if max(key) > half:
                    continue
                score = cost + (x + y) ** 2
                previous = updated.get(key)
                if previous is None or score < previous[0]:
                    updated[key] = score, {**allocation, jitter: (x, y)}
        states = updated
    found = states.get((half, half))
    return found[1] if found else None


def plan_pair(records: list[dict[str, Any]], *, family: str, layout_id: int,
              style_id: int, target_side: str, target_jitter: int, budget: int,
              n_success: int, target_fail_count: int = 1, calibration_count: int | None = None,
              repeat: int = 0, seed: int = 20260907) -> dict[str, Any]:
    """Return {manifest, diagnostics}; raise InfeasiblePlan on exact shortages.

    A mixed model is specific to this target side/jitter, NOT a common model for
    both evaluation directions. Calibration examples are reused training successes selected in each arm
    (default: all n_success). IDs may differ; rule and count are identical.
    """
    if target_side not in ("L", "R"):
        raise ValueError("target_side must be L or R")
    if budget <= 0 or budget % 2 or n_success <= 0 or n_success >= budget or n_success % 2:
        raise ValueError("budget and n_success must be positive even integers, 0 < n_success < budget")
    if target_fail_count < 0:
        raise ValueError("target_fail_count must be nonnegative")
    calibration_count = n_success if calibration_count is None else calibration_count
    if calibration_count < 0:
        raise ValueError("calibration_count must be nonnegative")
    target_key = f"{family}/layout{layout_id}/style{style_id}/{target_side}/j{target_jitter}"
    request = dict(family=family, layout_id=layout_id, style_id=style_id,
                   target_side=target_side, target_jitter=target_jitter, budget=budget,
                   n_success=n_success, target_fail_count=target_fail_count,
                   calibration_count=calibration_count, repeat=repeat, seed=seed)
    scoped = [r for r in records if r["family"] == family and r["layout_id"] == layout_id and r["style_id"] == style_id]
    # A layout/style cannot silently join multiple scene indices per instruction.
    scene_sets = defaultdict(set)
    for row in scoped:
        scene_sets[row["canonical_side"]].add((row["original_key"], row["scene_idx"]))
    if any(len(scenes) > 1 for scenes in scene_sets.values()):
        raise InfeasiblePlan("ambiguous scene pairing; multiple source scenes per side", {"request": request, "scenes": {s: sorted(v) for s, v in scene_sets.items()}})
    groups = defaultdict(list)
    exclusions = Counter()
    for row in scoped:
        current = row["jitter_idx"] == target_jitter
        if current and row["success"]:
            exclusions["target_jitter_success"] += 1
        elif current and row["canonical_side"] != target_side:
            exclusions["other_side_current_failure"] += 1
        else:
            groups[(row["canonical_side"], row["success"], row["jitter_idx"])].append(row)
    capacity = {key: len(value) for key, value in groups.items()}
    diagnostics = {"request": request, "target_key": target_key,
                   "available": {f"{s}/success{label}/j{j}": count for (s, label, j), count in sorted(capacity.items())},
                   "exclusions": dict(exclusions)}
    if calibration_count > n_success:
        raise InfeasiblePlan("calibration exceeds selected success budget", diagnostics)
    quotas = {}
    for success, total in ((0, budget - n_success), (1, n_success)):
        quota = _allocate(capacity, target_side, target_jitter, success, total,
                          target_fail_count if success == 0 else 0, seed, repeat)
        if quota is None:
            raise InfeasiblePlan(f"no exact side/outcome/jitter allocation for success={success}", diagnostics)
        quotas[success] = quota
    single, mixed = [], []
    other = "R" if target_side == "L" else "L"
    for success, by_jitter in quotas.items():
        for jitter, (own_count, other_count) in sorted(by_jitter.items()):
            own = sorted(groups.get((target_side, success, jitter), []), key=lambda r: _rank(seed, repeat, target_key, r["episode_uid"]))
            opposite = sorted(groups.get((other, success, jitter), []), key=lambda r: _rank(seed, repeat, target_key, r["episode_uid"]))
            mixed.extend(own[:own_count] + opposite[:other_count])
            single.extend(own[:own_count + other_count])
    pair_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    manifest = []
    for arm, selected in ((f"{target_side}_only", single), (f"LRmix_for_{target_side}", mixed)):
        successes = [r for r in selected if r["success"]]
        calibration = {r["episode_uid"] for r in sorted(successes, key=lambda r: _rank(seed, repeat, "calibration:" + target_key, r["episode_uid"]))[:calibration_count]}
        for row in sorted(selected, key=lambda r: r["episode_uid"]):
            is_cal = row["episode_uid"] in calibration
            manifest.append({**row, "pair_id": pair_id, "arm": arm, "repeat": repeat,
                             "target_key": target_key, "target_side": target_side,
                             "target_jitter": target_jitter, "budget": budget,
                             "n_success_budget": n_success, "train": 1,
                             "calibration": int(is_cal), "role": "train+calibration" if is_cal else "train",
                             "is_current_failure": int(row["jitter_idx"] == target_jitter),
                             "selection_seed": seed})
    # Validate invariants independently of DP, so future sampling edits fail loud.
    assert len(single) == len(mixed) == budget
    assert len({r["episode_uid"] for r in single}) == len({r["episode_uid"] for r in mixed}) == budget
    assert Counter((r["success"], r["jitter_idx"]) for r in single) == Counter((r["success"], r["jitter_idx"]) for r in mixed)
    assert all(sum(r["canonical_side"] == side and r["success"] == label for r in mixed) == count // 2
               for side in ("L", "R") for label, count in ((0, budget - n_success), (1, n_success)))
    assert {r["episode_uid"] for r in single if r["jitter_idx"] == target_jitter} == {r["episode_uid"] for r in mixed if r["jitter_idx"] == target_jitter}
    diagnostics.update(status="planned", pair_id=pair_id,
                       quotas={str(label): {str(j): {"target_side_mixed": a, "other_side_mixed": b, "single": a + b} for j, (a, b) in q.items()} for label, q in quotas.items()},
                       unique_per_arm=budget, calibration_unique_per_arm=calibration_count,
                       calibration_rule="selected_training_success_same_count_per_arm",
                       calibration_is_held_out=False,
                       shared_training_episodes=len({r["episode_uid"] for r in single} & {r["episode_uid"] for r in mixed}),
                       fit_ep_ids_resolved=all(r["fit_ep_id"] != "" for r in single + mixed))
    return {"manifest": manifest, "diagnostics": diagnostics}


def feasible_budgets(records: list[dict[str, Any]], *, budgets: list[int],
                     success_counts: list[int] | None = None, **request: Any) -> list[dict[str, Any]]:
    """Enumerate feasible exact (N,S) allocations; no oversampling or auto shrink.

    Returns quota/support diagnostics for each feasible allocation. If an explicit
    request is infeasible, call plan_pair to obtain its detailed exclusion reason.
    """
    result = []
    for budget in sorted(set(budgets)):
        for n_success in sorted(set(success_counts if success_counts is not None else range(2, budget, 2))):
            if budget <= 0 or budget % 2 or not 0 < n_success < budget or n_success % 2:
                continue
            try:
                result.append(plan_pair(records, budget=budget, n_success=n_success, **request)["diagnostics"])
            except InfeasiblePlan:
                continue
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-tsv", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--requests-json", required=True, help="List of plan_pair kwargs; explicit budgets are required")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    with open(args.index_tsv, newline="") as stream:
        raw = list(csv.DictReader(stream, delimiter="\t"))
    mapping = json.loads(Path(args.mapping_json).read_text())
    records = normalize_records(raw, mapping)
    requests = json.loads(Path(args.requests_json).read_text())
    if not isinstance(requests, list):
        raise ValueError("requests JSON must contain a list")
    manifest, diagnostics = [], []
    for request in requests:
        try:
            plan = plan_pair(records, **request)
            manifest.extend(plan["manifest"])
            diagnostics.append(plan["diagnostics"])
        except InfeasiblePlan as exc:
            diagnostics.append(exc.diagnostics)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in manifest for k in row}) or ["pair_id", "arm", "episode_uid"]
    with (out / "manifest.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    source_hash = hashlib.sha256(Path(args.index_tsv).read_bytes()).hexdigest()
    payload = {"schema": "lr_equal_budget/1", "source_index_sha256": source_hash,
               "mapping": mapping, "input_rows": len(raw), "normalized_rows": len(records),
               "plans": diagnostics}
    (out / "diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"planned": sum(d["status"] == "planned" for d in diagnostics),
                      "infeasible": sum(d["status"] != "planned" for d in diagnostics),
                      "manifest_rows": len(manifest), "out_dir": str(out)}))
    if not manifest:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
