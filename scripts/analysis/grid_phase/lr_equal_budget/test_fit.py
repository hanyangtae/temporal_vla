"""Synthetic CPU tests; run in robocasa container with numpy and torch."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lr_equal_budget_fit", HERE / "fit.py")
fit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fit
spec.loader.exec_module(fit)
sys.path.insert(0, str(HERE.parents[3]))
from src.failure_online.online_failure import OnlineFailureDetector


def synthetic(root):
    rng = np.random.default_rng(7)
    source = {}
    mapping = {}
    for side in ("L", "R"):
        rows = []
        for success in (1, 0):
            for j in (1, 2):
                for n in range(6):
                    rows.append({"success": success, "jitter_idx": j, "noise_idx": success * 10 + n})
        for n in range(2):
            rows.append({"success": 0, "jitter_idx": 0, "noise_idx": n})
        codebook = {"approach": 0, "pull": 1} if side == "L" else {"approach": 1, "pull": 0}
        columns = {k: [] for k in ("X", "ep_id", "scene", "noise", "rec_idx", "succ", "phase_code", "ep_len", "jitter")}
        for ep, row in enumerate(rows):
            T = 6 + (ep % 3)
            # Failure dwell is longer and phase integer IDs differ across LR.
            T += 3 * (1 - row["success"])
            feature = rng.normal(size=(T, 4)).astype(np.float32)
            feature[:, 0] += row["success"] * 2
            values = {"X": feature[:, None, None, None, :], "ep_id": np.full(T, ep),
                      "scene": np.zeros(T), "noise": np.full(T, row["noise_idx"]),
                      "rec_idx": np.arange(T), "succ": np.full(T, row["success"]),
                      "phase_code": np.array([codebook["approach"]] * 3 + [codebook["pull"]] * (T - 3)),
                      "ep_len": np.full(T, T), "jitter": np.full(T, row["jitter_idx"])}
            for name, value in values.items():
                columns[name].append(value)
            row.update(slug=f"Task_{side}", canonical_instruction=f"Task/{side}", canonical_side=side,
                       original_key=f"Task/{side}", scene_idx=0, episode_uid=f"{side}-{ep}",
                       is_current_failure=int(row["jitter_idx"] == 0))
        path = root / f"{side}.npz"
        meta = {"instruction": f"Task/{side}", "capture_layers": [12], "segment_names": ["all"],
                "phase_codebook": codebook, "jitter_axis": True, "sigs": [f"{side}-{i}" for i in range(len(rows))]}
        np.savez_compressed(path, **{k: np.concatenate(v) for k, v in columns.items()}, meta_json=json.dumps(meta))
        mapping[f"Task_{side}__s0"] = {"path": str(path), "source_instruction": f"Task/{side}"}
        source[side] = rows
    def choose(side, success, n_per_j):
        result = []
        for j in (1, 2):
            result.extend([r for r in source[side] if r["success"] == success and r["jitter_idx"] == j][:n_per_j])
        return result
    anchor = [r for r in source["L"] if r["jitter_idx"] == 0]
    solo = choose("L", 1, 6) + choose("L", 0, 5) + anchor
    mixed = choose("L", 1, 3) + choose("R", 1, 3) + choose("L", 0, 2) + choose("R", 0, 3) + anchor
    manifest = [{**r, "pair_id": "pair", "arm": arm, "train": 1, "calibration": r["success"]}
                for arm, rows in (("L_only", solo), ("LRmix_for_L", mixed)) for r in rows]
    plan = {"pair_id": "pair", "status": "planned", "request": {"budget": 24, "n_success": 12,
            "calibration_count": 12, "target_side": "L", "target_jitter": 0, "target_fail_count": 2, "seed": 7, "repeat": 0}}
    return manifest, plan, mapping


class FitTests(unittest.TestCase):
    def test_cpu_fit_export_and_online_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows, plan, mapping = synthetic(root)
            args = argparse.Namespace(epochs=1, hidden=4, batch_size=8, seed=2, lr=.001,
                                      lambda_reg=.01, expected_dim=4)
            torch.set_num_threads(1)
            result = fit.fit_pair(rows, plan, fit.Shards(mapping), root / "output", args, {"synthetic": True})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["cap_source_keys"]), 6)
            for arm in ("L_only", "LRmix_for_L"):
                path = root / "output" / "pair" / arm
                report = json.loads((path / "fit_report.json").read_text())
                self.assertEqual(report["n_unique"], 24)
                self.assertEqual(report["updates"], 3)
                self.assertEqual(report["registered"]["lr_jfair"], ["approach", "pull"])
                detector = OnlineFailureDetector.from_checkpoint(path / "detector.pt", alpha=.1, task="pair")
                out = detector.step(np.zeros(4, dtype=np.float32))
                self.assertTrue(np.isfinite(out["score"]))
                trajectory = np.random.default_rng(9).normal(size=(7, 4)).astype(np.float32)
                expected = fit.base.score_seq(detector.model, (trajectory - detector.std_mean) / detector.std_std)
                detector.reset()
                online = [detector.step(x)["score"] for x in trajectory]
                np.testing.assert_allclose(online, expected, atol=1e-6)
                for variant in ("plain", "lr_jfair"):
                    z = np.load(path / "operators" / variant / "pull" / "dit_L12" / "conceptors.npz")
                    self.assertAlmostEqual(float(np.linalg.norm(z["alpha0_v_steer"])), 1, places=5)
            with self.assertRaises(FileExistsError):
                fit.fit_pair(rows, plan, fit.Shards(mapping), root / "output", args, {})

    def test_budget_duplicate_and_target_success_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows, plan, _ = synthetic(Path(d))
            bad = [dict(r) for r in rows]
            bad[1] = dict(bad[0])
            with self.assertRaisesRegex(ValueError, "duplicated"):
                fit.validate_pair(bad, plan)
            rows[0]["jitter_idx"] = 0
            with self.assertRaisesRegex(ValueError, "target-jitter success"):
                fit.validate_pair(rows, plan)

    def test_source_label_and_instruction_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows, _, mapping = synthetic(Path(d))
            row = fit.normalize_row(rows[0])
            row["success"] = 0
            with self.assertRaisesRegex(ValueError, "label mismatch"):
                fit.Shards(mapping).load(row)
            mapping["Task_L__s0"]["source_instruction"] = "Task/R"
            with self.assertRaisesRegex(ValueError, "instruction mismatch"):
                fit.Shards(mapping).load(row)

    def test_pair_anchor_mass_and_class_jitter_match(self):
        with tempfile.TemporaryDirectory() as d:
            rows, plan, mapping = synthetic(Path(d))
            arms = fit.validate_pair(rows, plan)
            shard = fit.Shards(mapping)
            masses = []
            for selected in arms.values():
                eps = [shard.load(r) for r in selected]
                masses.append({e.identity: w for e, w in zip(eps, fit.weights(eps)) if e.row["role"] == "current_fail"})
            self.assertEqual(masses[0], masses[1])
            self.assertTrue(all(w == 1 / 24 for w in masses[0].values()))
            changed = [dict(r) for r in rows]
            # Keep unique coordinates and S/F counts, but change jitter marginal.
            mixed_success = next(r for r in changed if r["arm"] == "LRmix_for_L" and r["success"])
            mixed_success["jitter_idx"] = 3
            with self.assertRaisesRegex(ValueError, r"\(success,jitter\)"):
                fit.validate_pair(changed, plan)

    def test_calibration_subset_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows, plan, _ = synthetic(Path(d))
            rows[0]["calibration"] = 0
            with self.assertRaisesRegex(ValueError, "calibration flags"):
                fit.validate_pair(rows, plan)
            rows[0]["calibration"] = 1
            plan["request"]["calibration_count"] = 10
            with self.assertRaisesRegex(ValueError, "calibration_count"):
                fit.validate_pair(rows, plan)

    def test_standardization_is_episode_not_dwell_weighted(self):
        eps = [fit.Record({"success": 1, "side": "L", "jitter": 1}, np.full((t, 2), x, dtype=np.float32),
                          np.repeat("p", t), {}) for t, x in ((2, 0), (20, 10))]
        mu, _ = fit.standardize(eps, fit.weights(eps))
        np.testing.assert_allclose(mu, [5, 5])

    def test_strict_operator_does_not_use_single_side_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows, _, mapping = synthetic(root)
            # Remove all R failures in second jitter: strict mixed is unsupported.
            eps = [fit.Shards(mapping).load(fit.normalize_row(r)) for r in rows if r["arm"] == "LRmix_for_L"
                   and not (r["canonical_side"] == "R" and not r["success"] and r["jitter_idx"] == 2)]
            registry = fit.operators(eps, "L", root / "operators")
            self.assertTrue(all(not r["registered"] for r in registry if r["variant"] == "lr_jfair"))
            self.assertTrue(all(r["registered"] for r in registry if r["variant"] == "plain"))


if __name__ == "__main__":
    unittest.main()
