import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("lr_remote_stage", Path(__file__).with_name("remote_stage.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class DiscoveryTest(unittest.TestCase):
    def test_source_fingerprint_wins_over_legacy_lr_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "segA_scene").mkdir()
            manifest = root / "manifest.tsv"
            with manifest.open("w") as f:
                w = csv.DictWriter(f, ["canonical_instruction", "scene_idx", "sig"], delimiter="\t")
                w.writeheader()
                w.writerow(dict(canonical_instruction="OvenRack/out-right", scene_idx=0, sig="actual"))
            np.savez(root / "segA_scene/OvenRack_out-right__s0.npz",
                     meta_json=json.dumps({"instruction": "OvenRack/out-right", "sigs": ["wrong"]}))
            np.savez(root / "segA_scene/OvenRack_out-left__s0.npz",
                     meta_json=json.dumps({"instruction": "OvenRack/out-left", "sigs": ["actual"]}))
            mapping = stage.discover_shards(manifest, root)
            self.assertTrue(mapping["OvenRack_out-right__s0"]["path"].endswith("OvenRack_out-left__s0.npz"))
            self.assertEqual(mapping["OvenRack_out-right__s0"]["source_instruction"], "OvenRack/out-left")
            manifest.write_text(manifest.read_text().replace("actual", "absent"))
            with self.assertRaisesRegex(ValueError, "unresolved"):
                stage.discover_shards(manifest, root)

    def test_refuse_unpinned_revision(self):
        with self.assertRaisesRegex(ValueError, "SHA"):
            stage.remote_code("archive", "main", "main")

    def test_only_output_data_paths(self):
        with self.assertRaises(ValueError):
            stage.data_files(stage.CODE / "scripts")


if __name__ == "__main__":
    unittest.main()
