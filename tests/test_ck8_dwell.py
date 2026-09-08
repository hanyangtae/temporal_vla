"""Bounded tests for CK8 dwell caps and episode-level weighting."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit = _load("fit_ck8_dwell_setm", ROOT / "scripts/analysis/grid_phase/fit_ck8_dwell_setm.py")
detector = _load("failure_detector_sim", ROOT / "scripts/analysis/grid_phase/failure_detector_sim.py")


class TestCk8Dwell(unittest.TestCase):
    def test_episode_mean_is_invariant_to_long_episode_record_replication(self):
        X = np.array([[0.0], [2.0], [10.0], [14.0], [20.0]])
        ep = np.array([1, 1, 2, 2, 2])
        got = fit.episode_mean(X, ep, np.ones(len(ep), dtype=bool))
        X2 = np.concatenate([X, np.array([[10.0], [14.0], [20.0]])])
        ep2 = np.concatenate([ep, np.array([2, 2, 2])])
        got2 = fit.episode_mean(X2, ep2, np.ones(len(ep2), dtype=bool))
        np.testing.assert_allclose(got, got2)
        np.testing.assert_allclose(got, [((0 + 2) / 2 + (10 + 14 + 20) / 3) / 2])

    def test_capped_rows_preserves_cluster_missing_from_success_pool(self):
        ep = np.array([1, 1, 2, 2, 2])
        cluster = np.array([0, 0, 1, 1, 1])
        succ = np.array([1, 1, 0, 0, 0])
        keep, caps, contributions = fit.capped_rows(
            ep, cluster, succ, np.ones(len(ep), dtype=bool)
        )
        self.assertIn(0, caps)
        self.assertNotIn(1, caps)
        self.assertTrue(np.all(keep))
        missing = [x for x in contributions if x["cluster"] == 1][0]
        self.assertEqual(missing["raw"], missing["used"])
        self.assertEqual(missing["episode_weight"], 1.0)

    def test_phase_ck8_keeps_short_episode_and_unknown_cluster(self):
        e = detector.Episode(
            "task", 7, 0, 0, 0, np.ones((1, 3), dtype=np.float32), np.array([99])
        )
        kept = detector.truncate_episode(e, "phase-ck8", None, {0: 2})
        self.assertIsNotNone(kept)
        self.assertEqual(kept.T, 1)
        np.testing.assert_array_equal(kept.phase, [99])


if __name__ == "__main__":
    unittest.main()
