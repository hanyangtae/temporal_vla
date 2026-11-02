"""CPU tests for the atomic65 common-QAM pipeline (no GPU, no network, no remote).

Run: .venvs/rl2_iid/bin/python -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_common_qam.py -v
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CQ = Path(__file__).resolve().parents[1] / "common_qam"
sys.path.insert(0, str(CQ))
import experiment as X  # noqa: E402
import staging as S  # noqa: E402
import manifest as M  # noqa: E402
import extract as E  # noqa: E402
import train as T  # noqa: E402
import cli as C  # noqa: E402

GROOT_DATASET = X.REPO / "src/benchmarks/robocasa/robocasa/utils/groot_utils/groot_dataset.py"
ACTION_SLICES = {"base_motion": {"start": 0, "end": 4}, "control_mode": {"start": 4, "end": 5},
                 "end_effector_position": {"start": 5, "end": 8}, "end_effector_rotation": {"start": 8, "end": 11},
                 "gripper_close": {"start": 11, "end": 12}}


def upstream_filter_fn():
    tree = ast.parse(GROOT_DATASET.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_subset_demos_filter_key")
    ns = {"os": os, "json": json, "random": random, "Sequence": Sequence,
          "LE_ROBOT_EPISODE_FILENAME": "meta/episodes.jsonl"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(GROOT_DATASET), "exec"), ns)
    return ns["get_subset_demos_filter_key"]


def make_package(root: Path, n_eps: int, *, seed: int = 7, robot: str = "PandaOmron", bad_reward: int | None = None,
                 action_slices=None) -> Path:
    pkg = root / "lerobot"
    (pkg / "meta").mkdir(parents=True)
    ids = list(range(n_eps)); random.Random(seed).shuffle(ids)
    eps = [{"episode_index": i, "length": 12 + i % 4, "tasks": ["open the thing"]} for i in ids]
    info = {"robot_type": robot, "codebase_version": "v2.1", "fps": 20, "chunks_size": 1000, "total_episodes": n_eps,
            "total_frames": sum(e["length"] for e in eps),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.cam": {"dtype": "video"}, "action": {"dtype": "float32"}}}
    mod = {"action": action_slices or ACTION_SLICES, "state": {"s": {"start": 0, "end": 3}},
           "video": {"cam": {"original_key": "observation.images.cam"}}}
    (pkg / "meta/info.json").write_text(json.dumps(info))
    (pkg / "meta/modality.json").write_text(json.dumps(mod))
    (pkg / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (pkg / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "open the thing"}\n')
    rng = np.random.default_rng(seed)
    for e in eps:
        i, L = e["episode_index"], e["length"]
        rew = np.zeros(L); rew[-1] = 1
        if i == bad_reward:
            rew[-1] = 0
        done = np.zeros(L, bool); done[-1] = True
        tab = pa.table({"action": [list(map(float, r)) for r in rng.uniform(-1, 1, (L, 12))],
                        "next.reward": rew.tolist(), "next.done": done.tolist(), "frame_index": list(range(L))})
        f = pkg / info["data_path"].format(episode_chunk=0, episode_index=i)
        f.parent.mkdir(parents=True, exist_ok=True); pq.write_table(tab, f)
        v = pkg / info["video_path"].format(episode_chunk=0, video_key="observation.images.cam", episode_index=i)
        v.parent.mkdir(parents=True, exist_ok=True); v.write_bytes(os.urandom(32))
    return pkg


class Fixture(unittest.TestCase):
    TASKS = {"TaskA": 104, "TaskB": 100}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cq_"))
        self.src = self.tmp / "seungjun"           # stands in for the 승준 stager root
        self.staging = self.tmp / "staging"
        self.kanu = self.tmp / "kanu"
        self.rows, self.receipts = [], {}
        for t, n in self.TASKS.items():
            self.add_task(t, n)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_task(self, t, n, **kw):
        rel = f"v1.0/pretrain/atomic/{t}/20250101/lerobot"
        pkg = make_package((self.src / rel).parent, n, seed=len(self.rows) + 7, **kw)
        order = [json.loads(l)["episode_index"] for l in (pkg / "meta/episodes.jsonl").read_text().splitlines()]
        rec = {"status": "COMPLETE", "task": t, "package_manifest_sha256": S.manifest_sha(pkg),
               "filter_member_episode_indices": sorted(X.filter_members(order)), "episodes_total": n,
               "package_bytes": sum(p.stat().st_size for p in pkg.rglob("*") if p.is_file()),
               "archive_sha256": "a" * 64}
        rp = self.tmp / "stager_receipts" / f"{t}.json"
        rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(json.dumps(rec))
        self.rows.append({"task": t, "rel_path": rel, "_receipt_path": str(rp)})
        self.receipts[t] = rec
        return pkg

    def stage(self, **kw):
        args = dict(tasks=None, jobs=2, reserve=0, transport="local", source_root=str(self.src), kanu_root=self.kanu,
                    rows=self.rows, receipts=self.receipts)
        args.update(kw)
        return S.stage(self.staging, **args)


class FilterRuleTests(unittest.TestCase):
    def test_matches_robocasa_upstream(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            pkg = make_package(tmp, 107)
            order = [json.loads(l)["episode_index"] for l in (pkg / "meta/episodes.jsonl").read_text().splitlines()]
            self.assertEqual(X.filter_members(order), upstream_filter_fn()("100_demos", 0, str(pkg)))
            self.assertEqual(len(X.filter_members(order)), 100)
            self.assertEqual(X.filter_members(order[:100]), order[:100])   # <=100: upstream returns None = all
            self.assertIsNone(upstream_filter_fn()("100_demos", 0, str(make_package(tmp / "b", 100))))
        finally:
            shutil.rmtree(tmp)


class StagingTests(Fixture):
    def test_stage_verifies_hash_and_restarts_as_noop(self):
        s = self.stage()
        self.assertEqual((s["failed"], sorted(s["staged_or_present"])), ({}, ["TaskA", "TaskB"]))
        for t in self.TASKS:
            r = json.loads((self.staging / "receipts" / f"{t}.json").read_text())
            self.assertEqual(r["source"], "seungjun_tar_stream")
            self.assertEqual(S.manifest_sha(self.staging / "robocasa" / self.rows[0]["rel_path"].replace("TaskA", t)),
                             self.receipts[t]["package_manifest_sha256"])
        s2 = self.stage(reverify=True)
        self.assertEqual(s2["failed"], {})
        last = [json.loads(l) for l in Path(s2["log"]).read_text().splitlines() if '"task"' in l]
        self.assertEqual({x["status"] for x in last if "status" in x}, {"ALREADY_STAGED"})

    def test_kanu_copy_reused_only_if_identical(self):
        rel_a, rel_b = self.rows[0]["rel_path"], self.rows[1]["rel_path"]
        shutil.copytree(self.src / rel_a, self.kanu / rel_a)                 # identical
        shutil.copytree(self.src / rel_b, self.kanu / rel_b)
        (self.kanu / rel_b / "meta/tasks.jsonl").write_text("tampered\n")     # differs
        self.assertEqual(self.stage()["failed"], {})
        src = {t: json.loads((self.staging / "receipts" / f"{t}.json").read_text())["source"] for t in self.TASKS}
        self.assertTrue(src["TaskA"].startswith("kanu_cache_"))
        self.assertEqual(src["TaskB"], "seungjun_tar_stream")
        self.assertEqual((self.kanu / rel_b / "meta/tasks.jsonl").read_text(), "tampered\n")   # untouched

    def test_existing_different_package_untouched_and_error_propagates(self):
        dest = self.staging / "robocasa" / self.rows[0]["rel_path"]
        dest.mkdir(parents=True); (dest / "foreign").write_text("x")
        s = self.stage(jobs=1)
        self.assertIn("TaskA", s["failed"])
        self.assertEqual(s["not_started"], ["TaskB"])                       # fail-fast
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["foreign"])

    def test_space_gate_and_waiting_tasks(self):
        s = self.stage(reserve=10 ** 18, tasks=["TaskA"])
        self.assertIn("INSUFFICIENT_SPACE", s["failed"]["TaskA"])
        self.assertFalse((self.staging / "robocasa" / self.rows[0]["rel_path"]).exists())
        receipts = dict(self.receipts); receipts["TaskB"] = None
        s = self.stage(receipts=receipts)
        self.assertEqual(s["waiting_for_stager"], ["TaskB"])

    def test_bad_source_hash_moved_aside(self):
        (self.src / self.rows[0]["rel_path"] / "meta/tasks.jsonl").write_text("changed on source\n")
        s = self.stage(tasks=["TaskA"])
        self.assertIn("hash", s["failed"]["TaskA"])
        self.assertFalse((self.staging / "robocasa" / self.rows[0]["rel_path"]).exists())
        self.assertTrue(list((self.staging / ".partial").glob("TaskA.badhash.*")))


class ManifestTests(Fixture):
    def build(self, **kw):
        return M.build(self.staging, tasks_subset=list(self.TASKS), stager_receipts=self.receipts,
                       rows_override=self.rows, **kw)

    def test_members_only_balanced_split_and_contract(self):
        self.stage()
        m, errors = self.build()
        self.assertEqual(errors, [])
        per = {t["task"]: t for t in m["tasks"]}
        self.assertEqual((per["TaskA"]["package_episodes"], per["TaskA"]["episodes"], per["TaskA"]["outside_filter"]), (104, 100, 4))
        self.assertEqual((per["TaskA"]["train"], per["TaskA"]["val"]), (90, 10))
        a_ids = sorted(e["episode_index"] for e in m["episodes"] if e["task"] == "TaskA")
        self.assertEqual(a_ids, self.receipts["TaskA"]["filter_member_episode_indices"])
        e0 = m["episodes"][0]
        self.assertTrue(e0["episode_id"].startswith("robocasa/v1.0/pretrain/atomic/TaskA/20250101/episode_"))
        self.assertTrue(e0["parquet_rel"].startswith("robocasa/v1.0/pretrain/"))
        self.assertEqual(m["contract"]["raw_index"], [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])
        self.assertEqual(m["mode"], "subset")
        self.assertEqual(m["totals"]["outside_filter"], 4)
        # package on disk still has all 104 episodes (no truncation)
        self.assertEqual(len(list((self.staging / "robocasa" / self.rows[0]["rel_path"] / "data").rglob("*.parquet"))), 104)
        out = self.tmp / "meta_out"
        with self.assertRaises(SystemExit):
            M.write(m, [], out, "manifest.json")                         # subset may not pose as full
        self.assertEqual(M.write(m, [], out, "manifest_smoke.json"), 0)
        self.assertEqual(M.write(m, [], out, "manifest_smoke.json"), 0)  # identical rewrite ok

    def test_violation_is_reported_not_dropped(self):
        pkg = self.src / self.rows[0]["rel_path"]
        order = [json.loads(l)["episode_index"] for l in (pkg / "meta/episodes.jsonl").read_text().splitlines()]
        victim = X.filter_members(order)[3]
        shutil.rmtree(self.src); self.rows, self.receipts = [], {}
        self.add_task("TaskA", 104, bad_reward=victim); self.add_task("TaskB", 100)
        self.stage()
        m, errors = self.build()
        self.assertTrue(any(f"episode {victim}" in e for e in errors), errors)
        self.assertEqual(M.write(m, errors, self.tmp / "o", "manifest_smoke.json"), 1)
        self.assertFalse((self.tmp / "o" / "manifest_smoke.json").exists())

    def test_signature_mismatch_and_full_mode_requirements(self):
        self.add_task("TaskC", 100, robot="OtherRobot")
        self.stage()
        m, errors = M.build(self.staging, tasks_subset=None, stager_receipts=self.receipts, rows_override=self.rows)
        self.assertTrue(any("robot_type" in e for e in errors))
        self.assertTrue(any("full manifest needs 65" in e or "signature" in e for e in errors))

    def test_membership_must_match_stager_receipt(self):
        self.receipts["TaskA"] = dict(self.receipts["TaskA"], filter_member_episode_indices=list(range(100)))
        self.stage()
        _, errors = self.build()
        self.assertTrue(any("membership differs" in e for e in errors))


class ExtractFinalizeTests(Fixture):
    def setUp(self):
        super().setUp()
        self.stage()
        m, errors = M.build(self.staging, tasks_subset=list(self.TASKS), stager_receipts=self.receipts, rows_override=self.rows)
        assert not errors
        self.man = self.tmp / "manifest_smoke.json"; self.man.write_text(json.dumps(m))
        self.m = m
        self.out = self.tmp / "smoke_latents"; self.out.mkdir()
        (self.out / "contract.json").write_text(json.dumps(E.contract_for(self.man, m, True, {"x": "y"})))
        for e in E.smoke_episodes(m):
            npz = self.out / (E.ident(e) + ".npz"); np.savez(npz, latents=np.zeros((1, 1024), np.float32))
            rec = {**e, "npz": npz.name, "npz_sha256": X.sha256_path(npz),
                   "decisions": (e["length"] + 4) // 5}
            (self.out / (E.ident(e) + ".json")).write_text(json.dumps(rec))

    def test_finalize_ok_then_blocks_tamper_and_missing(self):
        self.assertEqual(E.finalize(self.man, self.out, True), 0)
        c = json.loads((self.out / "COMPLETE.json").read_text())
        self.assertEqual((c["episodes"], c["contract"]["mode"]), (4, "atomic65_smoke"))
        self.assertEqual({r["split"] for r in c["records"]}, {"train", "val"})
        victim = c["records"][0]
        (self.out / victim["npz"]).write_bytes(b"tampered")
        (self.out / "COMPLETE.json").unlink()
        self.assertEqual(E.finalize(self.man, self.out, True), 1)
        self.assertFalse((self.out / "COMPLETE.json").exists())
        (self.out / (E.ident(victim) + ".json")).unlink()
        self.assertEqual(E.finalize(self.man, self.out, True), 1)

    def test_contract_mode_must_match(self):
        self.assertEqual(E.finalize(self.man, self.out, False), 1)      # smoke contract cannot finalize full

    def test_shards_partition_all_episodes(self):
        got = [e["episode_id"] for k in range(3) for e in E.selected(self.m, False, k, 3)]
        self.assertEqual(sorted(got), sorted(e["episode_id"] for e in self.m["episodes"]))
        self.assertEqual(len(set(got)), len(got))


class TrainAndCliTests(Fixture):
    def test_production_guard_uses_manifest_counts(self):
        m = {"experiment": X.EXPERIMENT, "mode": "subset", "n_tasks": 2, "tasks": [], "episodes": []}
        with self.assertRaises(SystemExit):
            T.check_manifest(m, 0, smoke=False)
        T.check_manifest(m, 0, smoke=True)
        saved = T.N_TASKS
        try:
            T.N_TASKS = 2
            tasks = [{"task": "A", "package_episodes": 104}, {"task": "B", "package_episodes": 99}]
            eps = [{"task": "A"}] * 100 + [{"task": "B"}] * 99
            full = {"experiment": X.EXPERIMENT, "mode": "full", "n_tasks": 2, "tasks": tasks, "episodes": eps}
            T.check_manifest(full, 199, smoke=False)
            with self.assertRaises(SystemExit):
                T.check_manifest(dict(full, episodes=eps[:-1]), 198, smoke=False)
            with self.assertRaises(SystemExit):
                T.check_manifest(full, 150, smoke=False)
        finally:
            T.N_TASKS = saved

    def test_archive_manifest_text_matches_stager_rule(self):
        pkg = self.src / self.rows[0]["rel_path"]
        from atomic65_stage import file_manifest
        self.assertEqual(T.tree_manifest_text(pkg), file_manifest(pkg))

    def test_extract_launcher_cmd(self):
        import run_extract as RX
        cmd = RX.build_cmd(gpu=3, smoke=False, shard=1, num_shards=2, work=self.tmp, image="sha256:img")
        s = " ".join(cmd)
        self.assertIn("scripts/rl2_vla/groot_fold27_20261028/common_qam/extract.py extract", s)
        self.assertIn("--package-root /cache/datasets/atomic65_staging_20261029", s)
        self.assertIn("--shard 1 --num-shards 2", s)
        self.assertIn("device=3", s)
        self.assertNotIn("bash -s", s)

    def test_preflight_and_cleanup_fail_closed(self):
        errs = C.preflight("train", self.tmp / "work", self.staging)
        self.assertTrue(any("missing" in e for e in errs))
        rc = C.main(["--staging", str(self.staging), "--work", str(self.tmp / "work"), "cleanup-plan"])
        self.assertEqual(rc, 1)

    def test_no_legacy_hardcoded_scope(self):
        for p in CQ.glob("*.py"):
            src = p.read_text()
            for bad in ("1061", "target13_work", "groot_multitask_work", "len(m['tasks'])==13", "local10task"):
                self.assertNotIn(bad, src, f"{p.name} contains {bad}")


class LockTests(Fixture):
    def test_second_stager_refused_while_lock_held(self):
        import fcntl
        self.staging.mkdir(parents=True)
        with open(self.staging / ".stage.lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(S.StageLocked):
                self.stage()
            self.assertFalse((self.staging / "receipts").exists())
        self.assertEqual(self.stage()["failed"], {})                 # released -> proceeds


class SelectorTests(Fixture):
    def test_unknown_task_selectors_rejected(self):
        with self.assertRaises(SystemExit):
            self.stage(tasks=["AdjustToaster"])            # typo must not be silently ignored
        with self.assertRaises(SystemExit):
            M.build(self.staging, tasks_subset=["TaskA", "Nope"], stager_receipts=self.receipts, rows_override=self.rows)


class TrainResumeTests(unittest.TestCase):
    """Real PrefixQAMTrainer/TaskBalancedSampler on CPU (reduced net): interrupted+resumed == uninterrupted."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    def build(self):
        from transitions import EpisodeInput
        from multitask_data import TaskBalancedSampler, build_multitask_rl2_split, macro_validation
        from transitions import TransitionSet
        rng = np.random.default_rng(0)
        eps, mapping = [], {}
        for t in ("TaskA", "TaskB"):
            for j in range(3):
                L = 23 + j
                steps = np.arange(0, L, 5)
                eid = f"robocasa/v1.0/pretrain/atomic/{t}/20250101/episode_{j:06d}"
                rew = np.zeros(L); rew[-1] = 1
                eps.append(EpisodeInput(eid, eid, "val" if j == 2 else "train", L, "terminal", steps,
                                        rng.normal(size=(len(steps), 1024)).astype(np.float32),
                                        rng.uniform(-1, 1, (len(steps), 16, 12)).astype(np.float32), rew,
                                        np.ones((len(steps), 16), bool)))
                mapping[eid] = t
        train, _ = build_multitask_rl2_split([e for e in eps if e.split == "train"], split="train")
        val, _ = build_multitask_rl2_split([e for e in eps if e.split == "val"], split="val")
        sam = TaskBalancedSampler(train, val=val, episode_to_task=mapping, seed=42)
        vals = {}
        for task in sam.tasks:
            mask = np.asarray([mapping[str(e)] == task for e in val.data["episode_id"]])
            vals[task] = TransitionSet("val", {k: v[mask] for k, v in val.data.items()})
        return sam, vals, macro_validation

    def trainer(self):
        from qam_prefix import PrefixQAMTrainer
        return PrefixQAMTrainer.create(profile="smoke_reduced_network", gamma=0.99, seed=42,
                                       data_label="resume_test", execute_steps=5, batch_size=8)

    def params_bytes(self, tr):
        from flax import serialization
        return serialization.to_bytes(serialization.to_state_dict(tr.agent))

    def test_resume_is_bitwise_continuation(self):
        tmp = Path(tempfile.mkdtemp(prefix="cqr_"))
        try:
            fake_archive = lambda ck: {"verified": True, "relative": str(ck)}  # noqa: E731
            sig = X.FEATURE_SIGNATURE
            # uninterrupted 6 steps, checkpoint every 3
            sam, vals, macro = self.build(); tr = self.trainer()
            ref = tmp / "ref"; ref.mkdir()
            T.train_loop(tr, sam, vals, ref, steps=6, every=3, bs=8, eval_seed=1, start=0, results=[],
                         feature_signature=sig, archive_fn=fake_archive, macro_fn=macro, reserve_bytes=0)
            ref_bytes, ref_draws = self.params_bytes(tr), sam.draw_counts.copy()
            # interrupted: run to 3, then a "crash", then resume in a fresh process state
            sam, vals, macro = self.build(); tr = self.trainer()
            run = tmp / "run"; run.mkdir()
            T.train_loop(tr, sam, vals, run, steps=3, every=3, bs=8, eval_seed=1, start=0, results=[],
                         feature_signature=sig, archive_fn=fake_archive, macro_fn=macro, reserve_bytes=0)
            self.assertEqual(T.resumable_steps(run), [0, 3])
            sam2, vals2, macro = self.build()
            tr2, results = T.restore(run, sam2, 3, sig)
            self.assertEqual(tr2.steps_done, 3)
            self.assertEqual([r["step"] for r in results], [0, 3])
            T.train_loop(tr2, sam2, vals2, run, steps=6, every=3, bs=8, eval_seed=1, start=4, results=results,
                         feature_signature=sig, archive_fn=fake_archive, macro_fn=macro, reserve_bytes=0)
            self.assertEqual(self.params_bytes(tr2), ref_bytes)          # params + optimizer + agent RNG
            self.assertTrue(np.array_equal(sam2.draw_counts, ref_draws))  # sampler RNG continued
            self.assertEqual(T.resumable_steps(run), [0, 3, 6])
            # an uncommitted checkpoint (archive not verified) is never a resume point
            v = json.loads((run / "validation_6.json").read_text()); v["archive"]["verified"] = False
            (run / "validation_6.json").write_text(json.dumps(v))
            self.assertEqual(T.resumable_steps(run), [0, 3])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class FakeProc:
    """poll() returns None for `polls` calls, then rc (and runs side effect once)."""
    def __init__(self, rc, side=None, polls=0):
        self.rc, self.side, self.polls, self.pid, self.fired = rc, side, polls, 4242, False

    def poll(self):
        if self.polls > 0:
            self.polls -= 1; return None
        if not self.fired:
            self.fired = True
            if self.side:
                self.side()
        return self.rc

    def wait(self):
        while self.poll() is None:
            pass
        return self.rc


class PipelineTests(unittest.TestCase):
    """Queue/admission/resume logic with fakes (no GPU, no harness)."""

    def setUp(self):
        import pipeline as P
        self.P = P
        self.tmp = Path(tempfile.mkdtemp(prefix="cqp_"))
        self.work, self.staging = self.tmp / "work", self.tmp / "staging"
        (self.work / "metadata").mkdir(parents=True)
        src = self.tmp / "src"; src.mkdir()
        (src / "a.py").write_text("x = 1\n")
        self.saved = P.FROZEN_GLOBS
        P.FROZEN_GLOBS = [(src, "*.py")]
        self.src = src
        self.calls = []
        self.missing = {0: 0, 1: 5, 2: 5}         # shard0 already complete (like the real run)
        self.busy_plan = []                       # list of busy dicts returned per probe, last repeats
        self.shard_script = {}                    # shard -> list of outcomes: "ok" | "refused" | "fail"
        self.train_script = ["ok"]

    def tearDown(self):
        self.P.FROZEN_GLOBS = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, *extra):
        return self.P.parse(["--work", str(self.work), "--staging", str(self.staging), "--gpu-pool", "2", "3", "5",
                             "--session", "t", "--poll-s", "0", *extra])

    def stage_fn(self):
        (self.staging / "receipts").mkdir(parents=True, exist_ok=True)
        for k in range(65):
            (self.staging / "receipts" / f"T{k}.json").write_text("{}")
        return {"failed": {}, "waiting_for_stager": []}

    def runner(self, name, argv, env=None, wait=True):
        self.calls.append((name, dict(env or {}), list(argv)))
        log = self.work / "pipeline" / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("extract_shard"):
            k = int(argv[argv.index("--shard") + 1])
            outcome = (self.shard_script.get(k) or ["ok"]).pop(0) if self.shard_script.get(k) else "ok"
            if outcome == "refused":
                with open(log, "a") as f:
                    f.write("[harness] BLOCKED: kanu: aggregate GPU limit exceeded across sessions\n")
                return FakeProc(2), open(os.devnull, "w")
            if outcome == "fail":
                def side():
                    self.missing[k] = max(0, self.missing[k] - 2)   # partial progress kept
                return FakeProc(1, side, polls=1), open(os.devnull, "w")

            def side_ok():
                self.missing[k] = 0
            return FakeProc(0, side_ok, polls=1), open(os.devnull, "w")
        if name == "train":
            outcome = self.train_script.pop(0)
            out = self.work / "scratch_run"
            if outcome == "refused":
                with open(log, "a") as f:
                    f.write("[with_gpu_lease] kanu gpu2 점유 — 발사 중단(잡은 것 되돌림)\n")
                return FakeProc(3), open(os.devnull, "w")

            def crash():
                out.mkdir(exist_ok=True)
                (out / "run_manifest.json").write_text("{}")
                ck = out / "checkpoints/step_0050000"; ck.mkdir(parents=True, exist_ok=True)
                for fn in ("params_50000.pkl", "flags_50000.json", "sampler_state.json"):
                    (ck / fn).write_text("x")
                (out / "validation_50000.json").write_text(json.dumps({"archive": {"verified": True}}))

            def finish():
                out.mkdir(exist_ok=True)
                (out / "COMPLETE.json").write_text(json.dumps({"steps": 500000, "selection": {
                    "step": 500000, "macro": 0.1, "archive": {"verified": True, "relative": "r"}}}))
            return FakeProc(1 if outcome == "crash" else 0, crash if outcome == "crash" else finish), open(os.devnull, "w")
        return 0   # preflight

    def make(self, a):
        def manifest_fn():
            (self.work / "metadata/manifest.json").write_text("{}"); return 0

        def finalize_fn():
            (self.work / "latents").mkdir(exist_ok=True)
            (self.work / "latents/COMPLETE.json").write_text("{}"); return 0
        p = self.P.Pipeline(a, runner=self.runner, stage_fn=self.stage_fn, manifest_fn=manifest_fn,
                            finalize_fn=finalize_fn, sleep=lambda s: None)
        p.shard_missing = lambda k, n: self.missing[k]

        def busy():
            return self.busy_plan.pop(0) if len(self.busy_plan) > 1 else (self.busy_plan[0] if self.busy_plan else
                                                                            {g: [] for g in range(8)})
        p.gpu_busy = busy
        return p

    def smoke_done(self):
        (self.work / "smoke_latents").mkdir(); (self.work / "smoke_latents/COMPLETE.json").write_text("{}")
        (self.work / "scratch_smoke").mkdir(); (self.work / "scratch_smoke/COMPLETE.json").write_text('{"steps": 20}')

    def status(self):
        return json.loads((self.work / "pipeline/STATUS.json").read_text())

    def events(self):
        return [json.loads(l) for l in (self.work / "pipeline/events.jsonl").read_text().splitlines()]

    def launches(self, prefix):
        return [c for c in self.calls if c[0].startswith(prefix)]

    def test_missing_only_shards_then_train_and_resume_skips(self):
        self.smoke_done()
        self.assertEqual(self.make(self.args()).run(), 0)
        shards = sorted(int(c[2][c[2].index("--shard") + 1]) for c in self.launches("extract_shard"))
        self.assertEqual(shards, [1, 2])                                 # completed shard0 never relaunched
        self.assertTrue(all("--num-shards" in c[2] and c[2][c[2].index("--num-shards") + 1] == "3"
                            for c in self.launches("extract_shard")))
        self.assertEqual(self.status()["stage"], "done")
        self.calls.clear()
        self.assertEqual(self.make(self.args()).run(), 0)               # rerun: nothing relaunched
        self.assertFalse(self.launches("extract") + self.launches("train"))

    def test_waits_for_foreign_gpu_processes(self):
        self.smoke_done()
        all_busy = {g: [999] for g in range(8)}
        self.busy_plan = [all_busy, all_busy, {**all_busy, 3: []}, {**all_busy, 3: []}, {g: [] for g in range(8)}]
        self.assertEqual(self.make(self.args()).run(), 0)
        first = self.launches("extract_shard")[0]
        self.assertEqual(first[1]["GPUS"], "3")                          # only the GPU that became free
        self.assertTrue(any(e["stage"] == "extract_waiting_for_gpu" for e in self.events()))

    def test_gpu_outside_pool_never_used_and_cap(self):
        self.smoke_done()
        self.busy_plan = [{0: [], 1: [], 2: [999], 3: [999], 4: [], 5: [999], 6: [], 7: []},
                          {g: [] for g in range(8)}]
        self.assertEqual(self.make(self.args("--max-gpus", "1")).run(), 0)
        used = {c[1]["GPUS"] for c in self.launches("extract_shard")}
        self.assertTrue(used <= {"2", "3", "5"})

    def test_harness_refusal_is_retried_not_failed(self):
        self.smoke_done()
        self.shard_script = {1: ["refused", "refused", "ok"], 2: ["refused", "ok"]}
        self.assertEqual(self.make(self.args("--max-retries", "0")).run(), 0)
        refused = [e for e in self.events() if e["stage"] == "extract_admission_refused"]
        self.assertEqual(len(refused), 3)
        self.assertFalse((self.work / "pipeline/ALERTS.jsonl").exists() and
                         "rc=" in (self.work / "pipeline/ALERTS.jsonl").read_text())

    def test_real_failure_retried_missing_only_then_blocks(self):
        self.smoke_done()
        self.shard_script = {1: ["fail", "ok"]}
        self.assertEqual(self.make(self.args("--max-retries", "1")).run(), 0)
        self.assertIn("rc=1", (self.work / "pipeline/ALERTS.jsonl").read_text())
        shutil.rmtree(self.work / "latents"); shutil.rmtree(self.work / "scratch_run")
        (self.work / "pipeline/DONE.json").unlink()
        self.missing = {0: 0, 1: 5, 2: 0}
        self.shard_script = {1: ["fail", "fail", "fail"]}
        self.calls.clear()
        self.assertEqual(self.make(self.args("--max-retries", "1")).run(), 1)
        self.assertIn("failed 2x", self.status()["error"])
        self.assertFalse(self.launches("train"))

    def test_train_crash_resumes_and_lease_refusal_waits(self):
        self.smoke_done()
        self.train_script = ["refused", "crash", "ok"]
        self.assertEqual(self.make(self.args()).run(), 0)
        trains = self.launches("train")
        self.assertEqual(len(trains), 3)
        self.assertNotIn("--resume", trains[0][2]); self.assertNotIn("--resume", trains[1][2])
        self.assertIn("--resume", trains[2][2])                          # after crash: resume, never fresh
        self.assertTrue(any(e["stage"] == "train_admission_refused" for e in self.events()))

    def test_train_dir_without_verified_checkpoint_blocks(self):
        self.smoke_done()
        (self.work / "scratch_run").mkdir()
        self.assertEqual(self.make(self.args()).run(), 1)
        self.assertIn("no verified checkpoint", self.status()["error"])

    def test_smoke_gate_and_alert(self):
        self.assertEqual(self.make(self.args()).run(), 1)
        self.assertIn("smoke extraction", self.status()["error"])
        self.assertTrue((self.work / "pipeline/ALERTS.jsonl").read_text())

    def test_source_change_blocks(self):
        self.smoke_done()
        p = self.make(self.args()); p.freeze()
        (self.src / "a.py").write_text("x = 2\n")
        self.assertEqual(self.make(self.args()).run(), 1)
        self.assertIn("source changed", self.status()["error"])

    def test_refreeze_is_one_time(self):
        self.smoke_done()
        self.make(self.args()).freeze()                    # old freeze
        (self.src / "a.py").write_text("x = 2\n")          # reviewed change
        p = self.make(self.args("--refreeze"))
        orig_extract = p.step_extract

        def edit_mid_run():
            (self.src / "a.py").write_text("x = 3\n")      # unreviewed edit while running
            return orig_extract()
        p.step_extract = edit_mid_run
        self.assertEqual(p.run(), 1)
        self.assertIn("source changed", self.status()["error"])
        self.assertTrue(any(e["stage"] == "source_refrozen" for e in self.events()))

    def test_waits_while_stage_lock_held(self):
        self.smoke_done()
        p = self.make(self.args())
        state = {"n": 0}

        def locked_then_ok():
            state["n"] += 1
            if state["n"] <= 2:
                raise S.StageLocked("held by main")
            return self.stage_fn()
        p.stage_fn = locked_then_ok
        self.assertEqual(p.run(), 0)
        self.assertEqual(sum("waiting_for_lock" in e for e in self.events()), 2)

    def test_nvidia_smi_failure_is_fail_closed(self):
        p = self.P.Pipeline(self.args())
        import subprocess as sp
        orig = sp.run
        try:
            sp.run = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))
            self.assertEqual(p.free_gpus(set()), [])
        finally:
            sp.run = orig

    def test_gpu_cap(self):
        with self.assertRaises(SystemExit):
            self.P.parse(["--gpu-pool", "1", "2", "--max-gpus", "4", "--session", "s"])


if __name__ == "__main__":
    unittest.main()
