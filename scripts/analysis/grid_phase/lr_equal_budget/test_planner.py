"""Synthetic contract checks for the equal-unique-episode planner."""

import unittest
from collections import Counter

try:
    from .planner import InfeasiblePlan, feasible_budgets, normalize_records, plan_pair
except ImportError:
    from planner import InfeasiblePlan, feasible_budgets, normalize_records, plan_pair


MAPPING = {f"old-{side}": {"family": "drawer", "side": side,
                          "canonical_instruction": f"OpenDrawer/{side}"} for side in ("L", "R")}


def synthetic(n_each=12):
    rows = []
    for side in ("L", "R"):
        for jitter in range(3):
            for success in (0, 1):
                for index in range(n_each):
                    rows.append(dict(grid_instruction=f"old-{side}", scene_idx=0 if side == "L" else 2,
                                     jitter_idx=jitter, noise_idx=success * n_each + index,
                                     success=success, plan_id="p", layout_id=2, style_id=3,
                                     env_seed=100 + jitter, rel_path=f"{side}/j{jitter}/s{success}/n{index}",
                                     pkl_sha256=f"hash-{side}-{jitter}-{success}-{index}"))
    return rows


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.records = normalize_records(synthetic(), MAPPING)
        self.request = dict(family="drawer", layout_id=2, style_id=3, target_side="L",
                            target_jitter=0, budget=24, n_success=12, target_fail_count=2,
                            calibration_count=5, seed=42, repeat=0)

    def test_exact_budget_labels_jitters_and_common_current_data(self):
        result = plan_pair(self.records, **self.request)
        rows = result["manifest"]
        single = [r for r in rows if r["arm"] == "L_only"]
        mixed = [r for r in rows if r["arm"] == "LRmix_for_L"]
        self.assertEqual(len({r["episode_uid"] for r in single}), 24)
        self.assertEqual(len({r["episode_uid"] for r in mixed}), 24)
        self.assertEqual(Counter((r["success"], r["jitter_idx"]) for r in single),
                         Counter((r["success"], r["jitter_idx"]) for r in mixed))
        self.assertEqual(Counter((r["canonical_side"], r["success"]) for r in mixed),
                         {("L", 0): 6, ("L", 1): 6, ("R", 0): 6, ("R", 1): 6})
        current = lambda arm: {r["episode_uid"] for r in arm if r["is_current_failure"]}
        self.assertEqual(current(single), current(mixed))
        self.assertEqual(len(current(single)), 2)
        self.assertFalse(any(r["success"] and r["jitter_idx"] == 0 for r in rows))
        self.assertFalse(any(r["canonical_side"] == "R" and r["jitter_idx"] == 0 for r in rows))

    def test_same_rule_calibration_is_inside_budget_and_not_heldout(self):
        result = plan_pair(self.records, **self.request)
        by_arm = {}
        for row in result["manifest"]:
            if row["calibration"]:
                by_arm.setdefault(row["arm"], set()).add(row["episode_uid"])
                self.assertEqual((row["train"], row["success"]), (1, 1))
        self.assertEqual(len(by_arm["L_only"]), len(by_arm["LRmix_for_L"]))
        self.assertEqual(len(by_arm["L_only"]), 5)
        self.assertFalse(result["diagnostics"]["calibration_is_held_out"])

    def test_selection_is_reproducible_and_input_order_invariant(self):
        first = plan_pair(self.records, **self.request)
        reordered = plan_pair(list(reversed(self.records)), **self.request)
        self.assertEqual(first, reordered)
        repeated = plan_pair(self.records, **{**self.request, "repeat": 1})
        self.assertNotEqual({r["episode_uid"] for r in first["manifest"]},
                            {r["episode_uid"] for r in repeated["manifest"]})

    def test_no_opposite_success_means_explicit_shortage_not_oversampling(self):
        reduced = [r for r in self.records if not (r["canonical_side"] == "R" and r["success"])]
        with self.assertRaises(InfeasiblePlan) as caught:
            plan_pair(reduced, **self.request)
        self.assertIn("success=1", str(caught.exception))
        self.assertIn("available", caught.exception.diagnostics)

    def test_total_support_does_not_hide_impossible_matching_jitter_quotas(self):
        # Both sides have sufficient total successes, but R's six required
        # successes can only come from j1, where L-only has just two examples.
        limits = {("L", 1): 2, ("L", 2): 10, ("R", 1): 8, ("R", 2): 0}
        reduced = [r for r in self.records if not r["success"] or r["jitter_idx"] == 0
                   or r["noise_idx"] - 12 < limits[(r["canonical_side"], r["jitter_idx"])]]
        with self.assertRaisesRegex(InfeasiblePlan, "success=1"):
            plan_pair(reduced, **self.request)

    def test_matching_uses_layout_style_not_scene_index(self):
        result = plan_pair(self.records, **self.request)
        self.assertEqual({r["scene_idx"] for r in result["manifest"]}, {0, 2})
        altered = [dict(r, layout_id=9) if r["canonical_side"] == "R" else r for r in self.records]
        with self.assertRaises(InfeasiblePlan):
            plan_pair(altered, **self.request)

    def test_duplicate_coordinate_and_ambiguous_scene_fail_loud(self):
        rows = synthetic()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            normalize_records(rows + [rows[0]], MAPPING)
        extra = dict(self.records[0], scene_idx=3, episode_uid="different")
        with self.assertRaisesRegex(InfeasiblePlan, "ambiguous scene"):
            plan_pair(self.records + [extra], **self.request)

    def test_current_failure_and_calibration_caps_cannot_be_evaded(self):
        with self.assertRaises(InfeasiblePlan):
            plan_pair(self.records, **{**self.request, "target_fail_count": 7})
        with self.assertRaises(InfeasiblePlan):
            plan_pair(self.records, **{**self.request, "calibration_count": 13})

    def test_class_ratio_not_forced_balanced_and_feasibility_sweep(self):
        result = plan_pair(self.records, **{**self.request, "n_success": 16})
        mixed = [r for r in result["manifest"] if r["arm"] == "LRmix_for_L"]
        self.assertEqual(sum(r["success"] for r in mixed), 16)
        request = {k: v for k, v in self.request.items() if k not in ("budget", "n_success")}
        feasible = feasible_budgets(self.records, budgets=[12, 24], success_counts=[8, 12, 16], **request)
        self.assertEqual({(r["request"]["budget"], r["request"]["n_success"]) for r in feasible}, {(12, 8), (24, 8), (24, 12), (24, 16)})

    def test_mapping_explicit_and_shard_id_never_guessed(self):
        with self.assertRaises(ValueError):
            normalize_records(synthetic(), {"old-L": {"family": "drawer", "side": "L"}})
        self.assertTrue(all(r["fit_ep_id_status"] == "requires_shard_lookup" for r in self.records))
        self.assertTrue(all(r["fit_ep_id"] == "" for r in self.records))
        reverse_mapping = {key: {**val, "side": "R" if val["side"] == "L" else "L"} for key, val in MAPPING.items()}
        swapped = normalize_records(synthetic(), reverse_mapping)
        self.assertTrue(all(r["canonical_side"] == "R" for r in swapped if r["original_key"] == "old-L"))


if __name__ == "__main__":
    unittest.main()
