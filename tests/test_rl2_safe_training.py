import importlib.util
import unittest

import numpy as np

SCRIPT = "scripts/rl2_vla/stage2_cluster_reward/train_safe.py"
spec = importlib.util.spec_from_file_location("train_safe", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SafeTrainingCalibrationTest(unittest.TestCase):
    def test_mask_excludes_padded_frames(self):
        train = np.array([[0.2, 0.2, 0.0, 0.0], [0.4, 0.4, 0.0, 0.0]])
        train_mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
        cal = np.array([[0.3, 0.3, 99.0, 99.0]] * 4)
        cal_mask = np.array([[1, 1, 0, 0]] * 4, dtype=bool)
        band = module.grouped_conformal_band(train, train_mask, cal, cal_mask, ["a", "b", "c", "d"])
        self.assertTrue(np.all(band < 10.0))

    def test_reset_group_aggregates_policy_seed_maximum(self):
        train = np.ones((2, 2), dtype=float)
        masks = np.ones_like(train, dtype=bool)
        cal = np.array([[1.1, 1.1], [1.8, 1.8], [1.2, 1.2], [1.3, 1.3], [1.4, 1.4]])
        cal_masks = np.ones_like(cal, dtype=bool)
        band_grouped = module.grouped_conformal_band(train, masks, cal, cal_masks, ["r", "r", "s", "t", "u"], alpha=0.4)
        band_single = module.grouped_conformal_band(train, masks, cal[[1, 2, 3, 4]], cal_masks[[1, 2, 3, 4]], ["r2", "s", "t", "u"], alpha=0.4)
        np.testing.assert_allclose(band_grouped, band_single)

    def test_calibration_cannot_change_training_center(self):
        train = np.array([[0.1, 0.2], [0.3, 0.4]])
        train_mask = np.ones_like(train, dtype=bool)
        cal = np.array([[0.2, 0.3]] * 4)
        cal_mask = np.ones_like(cal, dtype=bool)
        first = module.grouped_conformal_band(train, train_mask, cal, cal_mask, ["a", "b", "c", "d"])
        changed = module.grouped_conformal_band(train, train_mask, cal + 0.1, cal_mask, ["a", "b", "c", "d"])
        self.assertFalse(np.allclose(first, changed))
        self.assertAlmostEqual(float(np.mean(train, axis=0)[0]), 0.2)

    def test_insufficient_grouped_calibration_fails(self):
        data = np.ones((2, 2), dtype=float)
        masks = np.ones_like(data, dtype=bool)
        with self.assertRaises(ValueError):
            module.grouped_conformal_band(data, masks, data[:1], masks[:1], ["only"])


if __name__ == "__main__":
    unittest.main()
