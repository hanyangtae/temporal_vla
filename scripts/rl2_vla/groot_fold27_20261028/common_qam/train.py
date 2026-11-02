"""atomic65 common QAM: fresh task-balanced prefix-QAM training on frozen-GR00T features.

Recipe, architecture and budget are those of groot_target13_20261022/train_scratch.py
(production profile, seed 42, Adam 3e-4, batch 256, 500k steps, eval/checkpoint every 50k,
fixed final-step selection, RL2 final3 reward, H16/execute5/flow4, normalized active12).
Differences: data = atomic65 manifest (counts from the manifest, not hardcoded),
experiment labels, archive destination, a shell-only archive hash check, and resume.

Resume (--resume): every checkpoint stores the full agent state (params, optimizer state,
network step, agent RNG; PrefixQAMTrainer.save) plus sampler_state.json (numpy PCG64 state
and per-task draw counts). Resuming from the newest checkpoint whose archive was verified
continues the identical update/sample sequence (CPU test: resumed params == uninterrupted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import (ADAPTER, ARCHIVE_REMOTE, DECODER_CALLS, EXPERIMENT, FILTER_N, LATENT_DIM,  # noqa: E402
                        N_TASKS, RECIPE, REPO, TARGET13, sha256_path)
for p in (TARGET13, ADAPTER):
    sys.path.insert(0, str(p))

WRAPPER = REPO / "scripts/utils/remote_compute.sh"


def check_manifest(m: dict, eps_loaded: int, smoke: bool) -> None:
    """Production run only on the full atomic65 manifest; counts come from the manifest."""
    if smoke:
        return
    if m.get("experiment") != EXPERIMENT or m.get("mode") != "full":
        raise SystemExit("production training needs the full atomic65 manifest")
    if m["n_tasks"] != N_TASKS or len(m["tasks"]) != N_TASKS:
        raise SystemExit(f"manifest tasks {m['n_tasks']} != {N_TASKS}")
    if eps_loaded != len(m["episodes"]):
        raise SystemExit(f"extraction episodes {eps_loaded} != manifest {len(m['episodes'])}")
    per = {}
    for e in m["episodes"]:
        per[e["task"]] = per.get(e["task"], 0) + 1
    expected = {t["task"]: min(FILTER_N, t["package_episodes"]) for t in m["tasks"]}
    if per != expected:
        raise SystemExit(f"per-task episode counts differ from 100_demos rule: {per} vs {expected}")


def tree_manifest_text(root: Path) -> str:
    rels = sorted(("./" + str(p.relative_to(root))).encode() for p in root.rglob("*") if p.is_file())
    return "".join(f"{sha256_path(root / r.decode()[2:])}  {r.decode()}\n" for r in rels)


def archive(path: Path, work_alias: str, remote_root: str = ARCHIVE_REMOTE) -> dict:
    """Mirror `path` (inside the repo-relative work alias) to 승준 and verify every file hash.
    Same push-data/REMOTE_REPO mechanism as target13; verification is a read-only shell pipeline."""
    alias = REPO / work_alias
    rel = Path(work_alias) / path.resolve().relative_to(alias.resolve())
    local = tree_manifest_text(path)
    subprocess.run(["bash", str(WRAPPER), "run", f"mkdir -p {remote_root}"], cwd=REPO, check=True, timeout=120,
                   capture_output=True)
    env = dict(os.environ, REMOTE_REPO=remote_root)
    subprocess.run(["bash", str(WRAPPER), "push-data", str(rel)], cwd=REPO, env=env, check=True, timeout=3600,
                   capture_output=True)
    target = f"{remote_root}/{rel}"
    p = subprocess.run(["bash", str(WRAPPER), "run",
                        f"cd {target} && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum"],
                       cwd=REPO, check=True, timeout=600, capture_output=True, text=True)
    if p.stdout != local:
        raise RuntimeError(f"archive hash mismatch for {rel}")
    return {"archive_root": "seungjun_checkpoints_atomic65_common_qam", "relative": str(rel),
            "manifest_sha256": hashlib.sha256(local.encode()).hexdigest(), "verified": True}


def load(extraction: Path, manifest: Path, checkpoint_root: Path):
    import numpy as np
    from transitions import EpisodeInput, expert_chunk_targets
    from contract import load_checkpoint_contract, resolve_cached_checkpoint_dir
    complete = json.loads((extraction / "COMPLETE.json").read_text())
    assert complete["contract"]["manifest_sha256"] == sha256_path(manifest)
    m = json.loads(manifest.read_text())
    identity = {e["episode_id"]: e for e in m["episodes"]}
    cc = load_checkpoint_contract(resolve_cached_checkpoint_dir(checkpoint_root))
    eps = []
    for rec in complete["records"]:
        e = identity[rec["episode_id"]]
        assert all(rec[k] == e[k] for k in ["split", "task", "parquet_sha256", "length"])
        assert sha256_path(extraction / rec["npz"]) == rec["npz_sha256"]
        with np.load(extraction / rec["npz"], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        assert d["latents"].shape == (len(d["decision_steps"]), LATENT_DIM) and np.isfinite(d["latents"]).all()
        assert np.all(d["decoder_calls"] == DECODER_CALLS)
        chunks, valid = expert_chunk_targets(d["raw_actions"], d["decision_steps"], codec=cc.codec,
                                             raw_index=np.asarray(m["contract"]["raw_index"]))
        assert np.array_equal(chunks, d["expert_chunks"]) and np.array_equal(valid, d["chunk_valid"])
        assert np.flatnonzero(d["done"]).tolist() == [e["length"] - 1]
        eps.append(EpisodeInput(e["episode_id"], e["episode_id"], e["split"], e["length"], "terminal",
                                d["decision_steps"], d["latents"], chunks, d["rewards"], valid))
    return m, eps


def save_checkpoint(tr, sam, out: Path, step: int, feature_signature: str) -> Path:
    """Agent pickle (params + optimizer + agent RNG) + flags + sampler RNG/draw counts."""
    ck = out / "checkpoints" / f"step_{step:07d}"
    tr.save(ck, feature_signature=feature_signature)
    state = {"step": int(step), "steps_done": int(tr.steps_done), "tasks": list(sam.tasks),
             "sampler_bit_generator_state": sam._rng.bit_generator.state,
             "draw_counts": [int(x) for x in sam.draw_counts]}
    tmp = ck / "sampler_state.json.tmp"
    tmp.write_text(json.dumps(state)); tmp.rename(ck / "sampler_state.json")
    return ck


def resumable_steps(out: Path) -> list[int]:
    """Checkpoints usable for resume: agent/flags/sampler files present and a validation
    record with a verified archive (i.e. the checkpoint was fully committed)."""
    ok = []
    for vp in out.glob("validation_*.json"):
        s = int(vp.stem.split("_")[1])
        ck = out / "checkpoints" / f"step_{s:07d}"
        v = json.loads(vp.read_text())
        if (v.get("archive", {}).get("verified") and (ck / f"params_{s}.pkl").is_file()
                and (ck / f"flags_{s}.json").is_file() and (ck / "sampler_state.json").is_file()):
            ok.append(s)
    return sorted(ok)


def restore(out: Path, sam, step: int, feature_signature: str):
    from qam_prefix import PrefixQAMTrainer
    ck = out / "checkpoints" / f"step_{step:07d}"
    tr = PrefixQAMTrainer.load(ck, step, feature_signature=feature_signature)
    st = json.loads((ck / "sampler_state.json").read_text())
    if st["tasks"] != list(sam.tasks) or tr.steps_done != step or st["steps_done"] != step:
        raise SystemExit(f"checkpoint {step} does not match this run (tasks/steps)")
    sam._rng.bit_generator.state = st["sampler_bit_generator_state"]
    sam.draw_counts[:] = st["draw_counts"]
    results = [json.loads((out / f"validation_{s}.json").read_text())
               for s in sorted(int(p.stem.split("_")[1]) for p in out.glob("validation_*.json")) if s <= step]
    return tr, results


def train_loop(tr, sam, vals, out: Path, *, steps: int, every: int, bs: int, eval_seed: int, start: int,
               results: list, feature_signature: str, archive_fn, macro_fn, reserve_bytes: int = 6 * 1024 ** 3):
    """Steps start..steps. Step `start` is evaluated/checkpointed only when start == 0 (fresh run)."""
    t0 = time.time()
    for step in range(start, steps + 1):
        if step:
            loss = tr.step(sam.sample(bs))
            if step % 1000 == 0 or step == steps:
                with (out / "progress.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "elapsed": time.time() - t0, **loss}) + "\n")
        if step % every == 0 or step == steps:
            metrics = {t: tr.evaluate(v, batch_size=bs, seed=eval_seed)["val/action_mse_exec_prefix"]
                       for t, v in vals.items()}
            macro = macro_fn(metrics)
            if shutil.disk_usage(out).free < reserve_bytes:
                raise RuntimeError("output filesystem reserve below 6GiB")
            ck = save_checkpoint(tr, sam, out, step, feature_signature)
            receipt = archive_fn(ck)
            result = {"step": step, "macro": macro, "tasks": metrics, "archive": receipt,
                      "task_draw_counts": dict(zip(sam.tasks, map(int, sam.draw_counts))),
                      "task_effective_passes": {t: float(n) / len(ix) for t, n, ix in zip(sam.tasks, sam.draw_counts, sam.indices)}}
            results.append(result)
            (out / f"validation_{step}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"step": step, "macro": macro, "durable_checkpoint": True}), flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--extraction", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-alias", default="outputs/rl2_vla/atomic65_common_qam_work_20261029",
                    help="repo-relative dir containing --out (archive path namespace)")
    ap.add_argument("--checkpoint-root", type=Path, default=None, help="datasets root holding the HF cache")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing --out from its newest verified checkpoint (never a fresh restart)")
    a = ap.parse_args(argv)
    if a.resume:
        if not (a.out / "run_manifest.json").exists():
            raise SystemExit(f"--resume: {a.out} has no run_manifest.json")
        if (a.out / "COMPLETE.json").exists():
            raise SystemExit(f"--resume: {a.out} is already COMPLETE")
    else:
        a.out.mkdir(exist_ok=False)
    import numpy as np
    import jax
    from flax import serialization
    from multitask_data import TaskBalancedSampler, build_multitask_rl2_split, macro_validation
    from transitions import TransitionSet
    from qam_prefix import PrefixQAMTrainer
    assert jax.default_backend() == "gpu", jax.devices()
    print("GPU_DEVICES", jax.devices(), flush=True)
    m, eps = load(a.extraction, a.manifest, a.checkpoint_root)
    check_manifest(m, len(eps), a.smoke)
    R = RECIPE
    train, side = build_multitask_rl2_split([e for e in eps if e.split == "train"], split="train")
    val, _ = build_multitask_rl2_split([e for e in eps if e.split == "val"], split="val")
    mapping = {e["episode_id"]: e["task"] for e in m["episodes"]}
    sam = TaskBalancedSampler(train, val=val, episode_to_task=mapping, seed=R["seed"])
    steps = R["smoke_steps"] if a.smoke else R["steps"]
    every = R["smoke_eval_every"] if a.smoke else R["eval_every"]
    bs = R["batch_size"]
    vals = {}
    for task in sam.tasks:
        mask = np.asarray([mapping[str(e)] == task for e in val.data["episode_id"]]); assert mask.any()
        vals[task] = TransitionSet("val", {k: v[mask] for k, v in val.data.items()})
    run = {"experiment": EXPERIMENT, "mode": "smoke" if a.smoke else "production_atomic65",
           "manifest_sha256": sha256_path(a.manifest), "episodes": len(eps), "tasks": sam.tasks,
           "train_decisions": len(train.data["observations"]), "val_decisions": len(val.data["observations"]),
           "train_decisions_by_task": {t: len(ix) for t, ix in zip(sam.tasks, sam.indices)},
           "recipe": R, "steps": steps, "warm_start_step": None, "optimizer_state": "fresh"}
    if a.resume:
        if json.loads((a.out / "run_manifest.json").read_text()) != json.loads(json.dumps(run)):
            raise SystemExit("--resume: run_manifest differs (data/recipe changed); refusing")
        ok = resumable_steps(a.out)
        if not ok:
            raise SystemExit("--resume: no verified checkpoint with sampler_state.json")
        tr, results = restore(a.out, sam, ok[-1], m["feature_signature"])
        start = ok[-1] + 1
        with (a.out / "progress.jsonl").open("a") as f:
            f.write(json.dumps({"resume_from_step": ok[-1], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "trainer_sha256": sha256_path(Path(__file__))}) + "\n")
        print(json.dumps({"resumed_from": ok[-1]}), flush=True)
    else:
        tr = PrefixQAMTrainer.create(profile=R["profile"], gamma=R["gamma"], seed=R["seed"],
                                     data_label=R["data_label"] + ("_smoke" if a.smoke else ""),
                                     execute_steps=R["execute_steps"], batch_size=R["batch_size"])
        initial = serialization.to_state_dict(tr.agent)
        counts = []

        def check_opt(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    counts.append(int(v)) if k == "count" else check_opt(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    check_opt(v)
        check_opt(initial["network"]["opt_state"])
        assert counts and all(x == 0 for x in counts) and tr.steps_done == 0
        assert all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(initial["network"]["opt_state"]))
        params = initial["network"]["params"]
        for name in ["critic", "actor_slow"]:
            l1 = jax.tree_util.tree_leaves(params["modules_" + name]); l2 = jax.tree_util.tree_leaves(params["modules_target_" + name])
            assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(l1, l2))
        (a.out / "initialization.json").write_text(json.dumps({
            "source": "fresh PrefixQAMTrainer.create production seed42; no restore; no warm start",
            "optimizer_counts": counts, "network_step": int(initial["network"]["step"]),
            "params_sha256": hashlib.sha256(serialization.to_bytes(tr.agent.network.params)).hexdigest(),
            "trainer_sha256": sha256_path(Path(__file__)), "prior_updates": 0}, indent=2))
        (a.out / "run_manifest.json").write_text(json.dumps(run, indent=2))
        np.savez_compressed(a.out / "raw_rl2_rewards.npz", **side)
        results, start = [], 0
    results = train_loop(tr, sam, vals, a.out, steps=steps, every=every, bs=bs, eval_seed=R["eval_seed"],
                         start=start, results=results, feature_signature=m["feature_signature"],
                         archive_fn=lambda ck: archive(ck, a.work_alias), macro_fn=macro_validation)
    (a.out / "COMPLETE.json").write_text(json.dumps({"steps": steps, "selection": results[-1], "evaluations": results}, indent=2))
    archive(a.out, a.work_alias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
