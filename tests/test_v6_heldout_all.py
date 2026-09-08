"""Unit tests for the stdlib helpers in ``run_v6_heldout_all.py``."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "steer" / "online_gated" / "run_v6_heldout_all.py"
SPEC = importlib.util.spec_from_file_location("run_v6_heldout_all", SCRIPT)
assert SPEC and SPEC.loader
v6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v6)


def _row(noise, success, *, collection_success=None, ep=None):
    return {
        "noise_idx": str(noise),
        "scene_idx": "0",
        "jitter_idx": "0",
        "env_seed": str(100 + noise),
        "inference_seed": str(1000 + noise),
        "collection_success": str(success if collection_success is None else collection_success),
        "success": str(success),
        "ep": str(noise if ep is None else ep),
    }


class TestCheckRows(unittest.TestCase):
    def setUp(self):
        self.expected = [_row(0, 0), _row(1, 1)]

    def test_accepts_expected_noise_and_labels(self):
        v6.check_rows([_row(0, 0), _row(1, 1)], self.expected)

    def test_rejects_duplicate_noise(self):
        with self.assertRaisesRegex(ValueError, "noise set mismatch"):
            v6.check_rows([_row(0, 0), _row(0, 0)], self.expected)

    def test_rejects_missing_noise(self):
        with self.assertRaisesRegex(ValueError, "noise set mismatch"):
            v6.check_rows([_row(0, 0)], self.expected)

    def test_rejects_inference_seed_mismatch(self):
        bad = [_row(0, 0), _row(1, 1)]
        bad[1]["inference_seed"] = "9999"
        with self.assertRaisesRegex(ValueError, "inference_seed"):
            v6.check_rows(bad, self.expected)


class TestSummarize(unittest.TestCase):
    def test_paired_rescue_destruction_and_completion(self):
        cell = ("cell", 0, 0)
        expected = [_row(0, 0), _row(1, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for arm, rows in {
                "base": [_row(0, 0), _row(1, 1)],
                "reseed_jfair_b09": [_row(0, 1, collection_success=0), _row(1, 0, collection_success=1)],
            }.items():
                path = out / arm / "cell_s0_j0" / "cell" / v6.ARMS[arm][0] / "per_episode.tsv"
                path.parent.mkdir(parents=True)
                v6.write_tsv(path, rows, list(rows[0]))

            result = v6.summarize(out, {cell: expected}, ["reseed_jfair_b09"])

            self.assertTrue(result["complete"])
            self.assertEqual(result["baseline_collection_mismatches"], 0)
            total = result["arms"][0]
            self.assertEqual(total["completed"], 2)
            self.assertEqual(total["expected"], 2)
            self.assertEqual(total["paired"], 2)
            self.assertEqual(total["baseline_fail"], 1)
            self.assertEqual(total["rescued"], 1)
            self.assertEqual(total["baseline_success"], 1)
            self.assertEqual(total["destroyed"], 1)
            self.assertIn("rescued", (out / "summary.tsv").read_text().splitlines()[0])


class TestCheckSidecars(unittest.TestCase):
    def test_reseed_setm_requires_offset_seed(self):
        cell = ("cell", 0, 0)
        row = _row(0, 1, ep=7)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            folder = out / "reseed_jfair_b09" / "cell_s0_j0" / "cell" / "ps_reseed_setm" / "raw_rollouts"
            folder.mkdir(parents=True)
            sidecar = folder / "task0--ep7--succ1.json"
            sidecar.write_text(json.dumps({
                "perstep_op": "reseed_setm",
                "episode_success": 1,
                "perstep_seed2": [901000, 901001],
            }))

            v6.check_sidecars(out, "reseed_jfair_b09", cell, [row])

            sidecar.write_text(json.dumps({
                "perstep_op": "reseed_setm",
                "episode_success": 1,
                "perstep_seed2": [1000, 1001],
            }))
            with self.assertRaisesRegex(ValueError, "seed2 mismatch"):
                v6.check_sidecars(out, "reseed_jfair_b09", cell, [row])


if __name__ == "__main__":
    unittest.main()
