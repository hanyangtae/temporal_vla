"""Frozen-GR00T feature extraction for the atomic65 common-QAM manifest.

Per-episode loop is the groot_multitask_20261021/extract.py loop unchanged (same model,
preprocessing, decision steps/seeds, 1024 latent, expert chunk targets, record format);
only the experiment labels, package root, sharding and the separate finalize step differ.

  extract  --manifest M --out D [--shard k --num-shards n] [--smoke]   (GPU, in model container)
  finalize --manifest M --out D [--smoke]                              (CPU) -> D/COMPLETE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import (ADAPTER, CHECKPOINT_REVISION, DECODER_CALLS, EXECUTE_STEPS, EXPERIMENT,  # noqa: E402
                        LATENT_DIM, sha256_path)
sys.path.insert(0, str(ADAPTER))


def smoke_episodes(m: dict) -> list[dict]:
    """First train + first val episode of the first two manifest tasks (manifest order)."""
    tasks = [t["task"] for t in m["tasks"]][:2]
    return [next(e for e in m["episodes"] if e["task"] == t and e["split"] == s) for t in tasks for s in ("train", "val")]


def selected(m: dict, smoke: bool, shard: int, num_shards: int) -> list[dict]:
    eps = smoke_episodes(m) if smoke else m["episodes"]
    if not 0 <= shard < num_shards:
        raise SystemExit("bad shard")
    return [e for j, e in enumerate(eps) if j % num_shards == shard]


def contract_for(manifest: Path, m: dict, smoke: bool, dep_hashes: dict) -> dict:
    return {"experiment": EXPERIMENT, "manifest_sha256": sha256_path(manifest),
            "feature_signature": m["feature_signature"], "checkpoint_revision": CHECKPOINT_REVISION,
            "extractor_sha256": sha256_path(Path(__file__)), "dependency_code_sha256": dep_hashes,
            "mode": "atomic65_smoke" if smoke else "atomic65_full"}


def ident(e: dict) -> str:
    return hashlib.sha256(e["episode_id"].encode()).hexdigest()


def finalize(manifest: Path, out: Path, smoke: bool) -> int:
    m = json.loads(manifest.read_text())
    contract = json.loads((out / "contract.json").read_text())
    if contract["manifest_sha256"] != sha256_path(manifest) or contract["mode"] != ("atomic65_smoke" if smoke else "atomic65_full"):
        print("FINALIZE BLOCKED: contract does not match manifest/mode"); return 1
    eps = smoke_episodes(m) if smoke else m["episodes"]
    records, missing, bad = [], [], []
    for e in eps:
        jp = out / (ident(e) + ".json")
        if not jp.exists():
            missing.append(e["episode_id"]); continue
        rec = json.loads(jp.read_text())
        if any(rec.get(k) != e[k] for k in ("episode_id", "split", "task", "parquet_sha256", "length")):
            bad.append(e["episode_id"] + ": identity"); continue
        if rec["decisions"] != (e["length"] + EXECUTE_STEPS - 1) // EXECUTE_STEPS:
            bad.append(e["episode_id"] + ": decisions"); continue
        if sha256_path(out / rec["npz"]) != rec["npz_sha256"]:
            bad.append(e["episode_id"] + ": npz hash"); continue
        records.append(rec)
    if missing or bad:
        print(f"FINALIZE BLOCKED: {len(records)}/{len(eps)} ok, missing={len(missing)} bad={bad[:5]}"); return 1
    complete = {"contract": contract, "episodes": len(records), "decisions": sum(r["decisions"] for r in records),
                "records": records}
    p = out / "COMPLETE.json"
    text = json.dumps(complete, indent=2)
    if p.exists() and p.read_text() != text:
        print(f"FINALIZE BLOCKED: different {p} exists"); return 1
    p.write_text(text)
    print(f"FINALIZE OK episodes={len(records)} decisions={complete['decisions']} -> {p}")
    return 0


def extract(a) -> int:
    import numpy as np
    from extract_opendrawer_latents import (FrozenGroot, PROFILE_REL, REPO_ROOT, checkpoint_dir, code_hashes,
                                            decode_video, import_serve_module, png_b64, state_payload_map,
                                            video_payload_map)
    from baseline_data import decision_seed, decision_steps
    from contract import load_checkpoint_contract, raw_to_model_index
    from transitions import expert_chunk_targets
    import importlib.util   # this module is also named extract -> load the 10-task one by path
    spec = importlib.util.spec_from_file_location(
        "groot_multitask_extract_20261021", ADAPTER.parent / "groot_multitask_20261021" / "extract.py")
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    package = mt.package   # groot_multitask_20261021 package reader, unchanged

    m = json.loads(a.manifest.read_text())
    eps = selected(m, a.smoke, a.shard, a.num_shards)
    a.out.mkdir(parents=True, exist_ok=True)
    deps = dict(code_hashes())
    for p in (ADAPTER.parent / "groot_multitask_20261021" / "extract.py", HERE / "experiment.py"):
        deps[str(p.relative_to(REPO_ROOT))] = sha256_path(p)
    contract = contract_for(a.manifest, m, a.smoke, deps)
    cp = a.out / "contract.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == contract, "contract mismatch: use a new --out"
    else:
        cp.write_text(json.dumps(contract, indent=2))
    cc = load_checkpoint_contract(checkpoint_dir(a.checkpoint_root))
    model = FrozenGroot(import_serve_module(), profile_path=REPO_ROOT / PROFILE_REL,
                        pretrained=checkpoint_dir(a.checkpoint_root), device="cuda:0")
    packages, done = {}, 0
    progress = a.out / f"progress_shard{a.shard}of{a.num_shards}.jsonl"
    for e in eps:
        jp = a.out / (ident(e) + ".json")
        if jp.exists():   # resume: verify and skip
            rec = json.loads(jp.read_text())
            assert sha256_path(a.out / rec["npz"]) == rec["npz_sha256"]
            done += 1
            continue
        t0 = time.time(); task = e["task"]
        if task not in packages:
            packages[task] = package((a.package_root / e["parquet_rel"]).parents[2])
        pkg = packages[task]; i = e["episode_index"]
        assert sha256_path(pkg.root / pkg.rel(i)) == e["parquet_sha256"]
        d = pkg.read(i); T = e["length"]; assert len(d["raw_actions"]) == T
        tid = set(d["task_desc"].tolist()); assert len(tid) == 1
        language = pkg.tasks[tid.pop()]; assert [language] == e["tasks"]
        sm = state_payload_map(pkg.modality, model.profile); vm = video_payload_map(pkg.modality)
        videos, vhash = {}, {}
        for key, dest in vm.items():
            vp = pkg.root / pkg.rel(i, key)
            videos[dest] = decode_video(vp, T, d["timestamp"], pkg.fps); vhash[key] = sha256_path(vp)
        steps = decision_steps(T); lat, seeds, calls = [], [], []
        for step in steps:
            payload = {k: png_b64(v[step]) for k, v in videos.items()}
            payload.update({k: d["state"][step, lo:hi].tolist() for k, (lo, hi) in sm.items()})
            payload["task"] = language
            seed = decision_seed(42, e["episode_id"], int(step))
            v, n = model.latent(payload, seed); lat.append(v); calls.append(n); seeds.append(seed)
        raw = d["raw_actions"]; rew = d["rewards"]; done_ = d["done"]
        assert raw.shape == (T, 12) and np.isfinite(raw).all()
        assert set(np.unique(rew)).issubset({0, 1}) and rew[-1] == 1 and np.flatnonzero(done_).tolist() == [T - 1]
        chunks, valid = expert_chunk_targets(raw, steps, codec=cc.codec, raw_index=raw_to_model_index(pkg.modality["action"]))
        arrays = dict(latents=np.stack(lat), decision_steps=steps, decision_seeds=np.asarray(seeds),
                      decoder_calls=np.asarray(calls), expert_chunks=chunks, chunk_valid=valid, raw_actions=raw,
                      rewards=rew, done=done_, frame_index=d["frame_index"])
        assert arrays["latents"].shape == (len(steps), LATENT_DIM) and np.isfinite(arrays["latents"]).all()
        assert np.all(arrays["decoder_calls"] == DECODER_CALLS)
        npz = a.out / (ident(e) + ".npz")
        tmp = a.out / (ident(e) + ".tmp.npz")
        np.savez(tmp, **arrays); tmp.rename(npz)          # a crash never leaves a half npz under the final name
        rec = {**e, "npz": npz.name, "npz_sha256": sha256_path(npz), "video_sha256": vhash, "language": language,
               "decisions": len(steps), "wall_seconds": time.time() - t0}
        tj = jp.with_suffix(".json.tmp"); tj.write_text(json.dumps(rec, indent=2)); tj.rename(jp)
        done += 1
        with progress.open("a") as f:
            f.write(json.dumps({"task": task, "episode": i, "done": done, "total": len(eps),
                                "seconds": rec["wall_seconds"]}) + "\n")
        print(json.dumps({"task": task, "episode": i, "done": done, "total": len(eps)}), flush=True)
    print(f"SHARD_DONE shard={a.shard}/{a.num_shards} episodes={done}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--smoke", action="store_true")
    p = sub.choices["extract"]
    p.add_argument("--package-root", type=Path, required=True, help="staging root holding robocasa/v1.0/...")
    p.add_argument("--checkpoint-root", type=Path, default=Path("/cache/datasets"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args(argv)
    return extract(a) if a.cmd == "extract" else finalize(a.manifest, a.out, a.smoke)


if __name__ == "__main__":
    sys.exit(main())
