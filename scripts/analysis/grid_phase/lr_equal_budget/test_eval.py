import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from .eval import build_jobs, completed, check_idle
except ImportError:
    from eval import build_jobs, completed, check_idle


def write_tsv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.cases = [dict(scene_idx="1", jitter_idx="2", noise_idx=str(n), env_seed="102", inference_seed=str(1300000+n), success=str(n)) for n in (0, 1)]

    def result_rows(self):
        return [{**r, "collection_success": r["success"]} for r in self.cases]

    def test_exact_keyset_accepts_correct_table_and_rejects_duplicate_equal_rowcount(self):
        p = self.root / "results.tsv"
        write_tsv(p, self.result_rows())
        self.assertTrue(completed(p, self.cases))
        write_tsv(p, [self.result_rows()[0]] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            completed(p, self.cases)

    def test_missing_wrong_seed_and_baseline_gate_fail_loud(self):
        p = self.root / "results.tsv"
        self.assertFalse(completed(p, self.cases))
        write_tsv(p, self.result_rows()[:1])
        with self.assertRaisesRegex(ValueError, "partial"):
            completed(p, self.cases)
        wrong = self.result_rows()
        wrong[0]["env_seed"] = "999"
        write_tsv(p, wrong)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            completed(p, self.cases)
        write_tsv(p, [{**self.result_rows()[0], "success": "1"}])
        with self.assertRaisesRegex(ValueError, "reproduce failure"):
            completed(p, self.cases[:1], require_failure=True)

    def test_dry_plan_uses_original_replay_keys_all_noise_and_skips_unsupported(self):
        runner = self.root / "scripts/steer/online_gated/run_online_gated_eval.sh"
        runner.parent.mkdir(parents=True)
        runner.write_text("#!/bin/bash\n")
        plan = {"status": "planned", "pair_id": "abc", "target_key": "dish/layout9/L/j2", "request": {"repeat": 0}}
        (self.root / "diagnostics.json").write_text(json.dumps({"plans": [plan]}))
        records = [{**r, "pair_id": "abc", "original_key": "DishwasherRack/out-right", "canonical_instruction": "DishwasherRack/out-left", "machine": "worker1"} for r in self.cases]
        write_tsv(self.root / "eval_cases.tsv", records[:1])
        write_tsv(self.root / "preservation_eval.tsv", records[1:])
        write_tsv(self.root / "manifest.tsv", [{"pair_id": "abc", "arm": a, "noise_idx": "0"} for a in ("L_only", "LRmix_for_L")])
        for arm in ("L_only", "LRmix_for_L"):
            dest = self.root / "fit/abc" / arm
            dest.mkdir(parents=True)
            (dest / "detector.pt").write_bytes(b"synthetic")
            (dest / "fit_report.json").write_text(json.dumps({"registered": {"plain": ["grasp-handle"], "lr_jfair": []}}))
        with patch("subprocess.run", side_effect=AssertionError("dry planning must not invoke GPU/runner")):
            jobs, coverage = build_jobs(self.root, self.root, "srv48", 2)
        self.assertEqual(len(jobs), 5)  # one gate, two reseed, two plain
        self.assertEqual(sum(len(j["expected"]) for j in jobs), 9)
        self.assertEqual(sum(c["status"] == "unsupported_no_phase" for c in coverage), 2)
        for job in jobs:
            self.assertEqual(job["env"]["SLUGS"], "DishwasherRack_out-right")
            self.assertEqual(job["env"]["FAILURE_TASK"], "abc")
            self.assertEqual(job["env"]["SERVES_PER_GPU"], "1")
            self.assertEqual(job["env"]["EVAL_NOISES"], "0" if job["gate"] else "0,1")
        self.assertEqual(build_jobs(self.root, self.root, "kanu", 4)[0], [])

    def test_gpu_with_any_compute_pid_is_rejected(self):
        class Result:
            stdout = "999\n"
        with patch("subprocess.run", return_value=Result()):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                check_idle(2, 8892)


if __name__ == "__main__":
    unittest.main()
