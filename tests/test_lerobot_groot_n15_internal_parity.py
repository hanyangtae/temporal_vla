"""Unit tests for GR00T N1.5 internal parity diagnostics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from lerobot_groot_n15_internal_parity import (  # noqa: E402
    CheckpointPrefixSpec,
    DEFAULT_CHECKPOINT_PREFIX_SPECS,
    compare_checkpoint_prefixes,
)


class TestCheckpointPrefixParity(unittest.TestCase):
    def test_default_prefixes_cover_loaded_vision_projector_and_action_head(self):
        self.assertEqual(
            [spec.name for spec in DEFAULT_CHECKPOINT_PREFIX_SPECS],
            ["eagle_vision", "eagle_projector", "action_head"],
        )

    def test_reports_exact_loaded_checkpoint_prefixes(self):
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            save_file(
                {
                    "backbone.eagle_model.mlp1.0.weight": torch.arange(4, dtype=torch.float32),
                    "backbone.eagle_model.mlp1.0.bias": torch.ones(2),
                    "backbone.eagle_model.vision_model.layer.weight": torch.full((2,), 3.0),
                    "other.weight": torch.full((1,), 9.0),
                },
                str(checkpoint_dir / "model-00001-of-00001.safetensors"),
            )
            model_state = {
                "backbone.eagle_model.mlp1.0.weight": torch.arange(4, dtype=torch.float32),
                "backbone.eagle_model.mlp1.0.bias": torch.ones(2),
                "backbone.eagle_model.vision_model.layer.weight": torch.full((2,), 3.0),
            }

            report = compare_checkpoint_prefixes(
                model_state,
                checkpoint_dir,
                [
                    CheckpointPrefixSpec("projector", "backbone.eagle_model.mlp1."),
                    CheckpointPrefixSpec("vision", "backbone.eagle_model.vision_model."),
                ],
            )

        self.assertEqual(report["checked_count"], 3)
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["shape_mismatch_count"], 0)
        self.assertEqual(report["value_mismatch_count"], 0)
        self.assertEqual(report["prefixes"]["projector"]["checked_count"], 2)
        self.assertEqual(report["prefixes"]["vision"]["checked_count"], 1)

    def test_supports_stripping_checkpoint_prefix_for_module_state_dict(self):
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            save_file(
                {
                    "backbone.eagle_model.mlp1.0.weight": torch.tensor([1.0, 2.0]),
                },
                str(checkpoint_dir / "model-00001-of-00001.safetensors"),
            )
            model_state = {"0.weight": torch.tensor([1.0, 2.0])}

            report = compare_checkpoint_prefixes(
                model_state,
                checkpoint_dir,
                [
                    CheckpointPrefixSpec(
                        "projector_module",
                        "backbone.eagle_model.mlp1.",
                        model_prefix="",
                    ),
                ],
            )

        self.assertEqual(report["checked_count"], 1)
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["value_mismatch_count"], 0)
        self.assertEqual(report["prefixes"]["projector_module"]["max_abs"], 0.0)

    def test_separates_missing_shape_and_value_mismatches(self):
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            save_file(
                {
                    "foo.good": torch.tensor([1.0, 2.0]),
                    "foo.missing": torch.tensor([3.0]),
                    "foo.shape": torch.tensor([4.0, 5.0]),
                    "foo.value": torch.tensor([6.0, 7.0]),
                },
                str(checkpoint_dir / "model-00001-of-00001.safetensors"),
            )
            model_state = {
                "foo.good": torch.tensor([1.0, 2.0]),
                "foo.shape": torch.tensor([[4.0, 5.0]]),
                "foo.value": torch.tensor([6.0, 8.0]),
            }

            report = compare_checkpoint_prefixes(
                model_state,
                checkpoint_dir,
                [CheckpointPrefixSpec("foo", "foo.")],
            )

        prefix = report["prefixes"]["foo"]
        self.assertEqual(report["checked_count"], 2)
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["shape_mismatch_count"], 1)
        self.assertEqual(report["value_mismatch_count"], 1)
        self.assertEqual(prefix["missing"][0]["checkpoint_key"], "foo.missing")
        self.assertEqual(prefix["shape_mismatches"][0]["checkpoint_shape"], [2])
        self.assertEqual(prefix["shape_mismatches"][0]["model_shape"], [1, 2])
        self.assertEqual(prefix["value_mismatches"][0]["checkpoint_key"], "foo.value")
        self.assertEqual(prefix["max_abs"], 1.0)

    def test_reports_value_match_after_casting_checkpoint_to_model_dtype(self):
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"safetensors unavailable: {exc}")

        values = torch.tensor([0.1001, -0.2002, 1.2345], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            save_file(
                {"action_head.weight": values},
                str(checkpoint_dir / "model-00001-of-00001.safetensors"),
            )
            model_state = {"action_head.weight": values.to(torch.bfloat16)}

            report = compare_checkpoint_prefixes(
                model_state,
                checkpoint_dir,
                [CheckpointPrefixSpec("action_head", "action_head.")],
            )

        prefix = report["prefixes"]["action_head"]
        self.assertEqual(report["value_mismatch_count"], 1)
        self.assertEqual(report["dtype_cast_value_mismatch_count"], 0)
        self.assertEqual(prefix["dtype_cast_value_mismatch_count"], 0)
        self.assertEqual(prefix["dtype_cast_max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
