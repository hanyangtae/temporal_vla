import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/rl2_vla/stage2_cluster_reward/safe_provenance.py"


class SafeProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.safe = self.root / "safe"
        self.safe.mkdir()
        self.train = self.root / "training.json"
        self.calibration = self.root / "calibration.json"
        self.train.write_text(json.dumps({"episodes": [{"episode_id": "train-1", "reset_id": "reset-train-1", "task_id": "task", "split": "train", "path": "train/episode.json"}]}), encoding="utf-8")
        self.calibration.write_text(json.dumps({"episodes": [{"episode_id": "cal-1", "reset_id": "reset-cal-1", "task_id": "task", "split": "calibration", "path": "cal/episode.json"}]}), encoding="utf-8")
        (self.safe / "checkpoint.pt").write_bytes(b"checkpoint")
        (self.safe / "model_final.ckpt").write_bytes(b"model")
        (self.safe / "config.yaml").write_text("model: safe\n", encoding="utf-8")
        (self.safe / "cp_band_by_alpha.npy").write_bytes(b"cp-band")
        (self.safe / "nested").mkdir()
        (self.safe / "nested" / "metadata.json").write_text('{"seed":42}\n', encoding="utf-8")
        self.manifest = self.safe / "provenance.json"
        self._write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_manifest(self, **updates):
        payload = {
            "owner": "temporal_vla",
            "feature_contract": "pi0_denoise0_action_mean_1024",
            "training_manifest": {"path": "../training.json", "sha256": self.digest(self.train)},
            "calibration_manifest": {"path": "../calibration.json", "sha256": self.digest(self.calibration)},
            "artifacts": [
                {"path": "checkpoint.pt", "sha256": self.digest(self.safe / "checkpoint.pt")},
                {"path": "model_final.ckpt", "sha256": self.digest(self.safe / "model_final.ckpt")},
                {"path": "config.yaml", "sha256": self.digest(self.safe / "config.yaml")},
                {"path": "cp_band_by_alpha.npy", "sha256": self.digest(self.safe / "cp_band_by_alpha.npy")},
                {"path": "nested/metadata.json", "sha256": self.digest(self.safe / "nested/metadata.json")},
            ],
        }
        payload.update(updates)
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")

    def run_cli(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--safe-dir", str(self.safe), "--manifest", str(self.manifest)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_valid_manifest_and_recursive_artifacts(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_wrong_owner(self):
        self._write_manifest(owner="author")
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("owner", result.stderr)

    def test_rejects_changed_checkpoint(self):
        (self.safe / "checkpoint.pt").write_bytes(b"changed")
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash mismatch", result.stderr)

    def test_rejects_unlisted_json_artifact(self):
        (self.safe / "extra.json").write_text("{}\n", encoding="utf-8")
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing from provenance", result.stderr)

    def test_rejects_missing_loader_required_file(self):
        (self.safe / "model_final.ckpt").unlink()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_final.ckpt", result.stderr)

    def test_rejects_overlapping_reset_id(self):
        self.calibration.write_text(json.dumps({"episodes": [{"episode_id": "cal-1", "reset_id": "reset-train-1"}]}), encoding="utf-8")
        self._write_manifest()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reset_id overlap", result.stderr)

    def test_rejects_bundled_saved_path(self):
        self.calibration.write_text(json.dumps({"episodes": [{"episode_id": "cal-1", "reset_id": "reset-cal-1", "path": "SAVED/author.json"}]}), encoding="utf-8")
        self._write_manifest()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bundled SAVED path", result.stderr)

    def test_rejects_missing_training_manifest(self):
        self.train.unlink()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
