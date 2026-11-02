"""Build the atomic65 common-QAM episode manifest from compute-staged packages.

Per task: the registry 100_demos members (seed 0 rule) are recomputed from the staged
meta/episodes.jsonl and must equal the stager receipt; every member episode is audited
(actual 12-D expert actions, successful-expert reward/done, frame index, videos). Any
violation fails the build with a report -- no episode is silently dropped or added.
The package itself is never modified; non-member episodes are simply not listed.

Output format is the groot_multitask_20261021 manifest format (episode_id, parquet_rel,
parquet_sha256, split, tasks, length, ...) so the frozen-GR00T extractor/loader contract
is unchanged.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiment import (ADAPTER, DECISION_SEED_BASE, ENV_ACTION_DIM, EXECUTE_STEPS, EXPERIMENT,
                        FEATURE_SIGNATURE, FILTER_N, FILTER_SEED, N_TASKS, RECIPE, ROBOT, SPLIT_SALT,
                        VAL_FRACTION, episode_id, filter_members, load_stage_rows, sha256_path,
                        val_episodes)

import sys
if str(ADAPTER) not in sys.path:
    sys.path.insert(0, str(ADAPTER))


def raw_index(action_slices) -> list[int]:
    from contract import raw_to_model_index   # adapter contract (numpy only)
    return raw_to_model_index(action_slices).tolist()


def audit_task(task: str, rel: str, staging: Path, staging_receipt: dict, stager_receipt: dict):
    import pyarrow.parquet as pq
    root = staging / "robocasa" / rel
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    mod = json.loads((meta / "modality.json").read_text())
    eps = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
    errors = []
    if info.get("robot_type") != ROBOT:
        errors.append(f"robot_type {info.get('robot_type')} != {ROBOT}")
    signature = {"robot": info.get("robot_type"), "action": mod["action"], "state": mod["state"],
                 "video": mod["video"], "raw_index": raw_index(mod["action"])}
    order = [int(e["episode_index"]) for e in eps]
    members = filter_members(order)
    if sorted(members) != stager_receipt["filter_member_episode_indices"]:
        errors.append("100_demos membership differs from stager receipt")
    if len(order) != stager_receipt["episodes_total"]:
        errors.append("episodes.jsonl rows != stager receipt episodes_total")
    by_index = {int(e["episode_index"]): e for e in eps}
    video_keys = sorted(k for k, v in info["features"].items() if v["dtype"] == "video")
    rows = []
    for rank, i in enumerate(members):
        e = by_index[i]
        drel = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
        f = root / drel
        tab = pq.read_table(f, columns=["action", "next.reward", "next.done", "frame_index"]).to_pydict()
        act = np.asarray(tab["action"], dtype=np.float64)
        rew = np.asarray(tab["next.reward"], dtype=np.float64).reshape(-1)
        done = np.asarray(tab["next.done"]).reshape(-1).astype(bool)
        L = int(e["length"])
        bad = []
        if act.shape != (L, ENV_ACTION_DIM) or not np.isfinite(act).all():
            bad.append(f"action shape {act.shape} / nonfinite")
        if len(rew) != L or not set(np.unique(rew).tolist()) <= {0.0, 1.0} or rew[-1] != 1:
            bad.append("reward not successful-expert {0,1} with last=1")
        if np.flatnonzero(done).tolist() != [L - 1]:
            bad.append("done not only at last step")
        if not np.array_equal(np.asarray(tab["frame_index"]).reshape(-1), np.arange(L)):
            bad.append("frame_index != arange(length)")
        if L <= 3:
            bad.append("length <= 3 (RL2 final3 needs > 3)")
        for k in video_keys:
            vp = root / info["video_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i, video_key=k)
            if not vp.is_file():
                bad.append(f"missing video {k}")
        if bad:
            errors.append(f"episode {i}: {bad}")
            continue
        eid = episode_id(rel, i)
        rows.append({"episode_id": eid, "episode_index": i, "task": task, "tasks": e["tasks"], "length": L,
                     "parquet_rel": "robocasa/" + rel + "/" + drel, "parquet_sha256": sha256_path(f),
                     "filter_rank": rank,
                     "raw_reward_audit": {"positive_steps": int((rew > 0).sum()),
                                          "first_positive": int(np.flatnonzero(rew > 0)[0]),
                                          "unique": np.unique(rew).tolist()}})
    val = val_episodes([r["episode_id"] for r in rows])
    for r in rows:
        r["split"] = "val" if r["episode_id"] in val else "train"
    summary = {"task": task, "package_episodes": len(order), "filter_members": len(members),
               "outside_filter": len(order) - len(members), "episodes": len(rows),
               "frames": sum(r["length"] for r in rows),
               "train": sum(r["split"] == "train" for r in rows), "val": sum(r["split"] == "val" for r in rows),
               "package_manifest_sha256": staging_receipt["package_manifest_sha256"],
               "stager_archive_sha256": stager_receipt["archive_sha256"],
               "info_sha256": sha256_path(meta / "info.json"), "modality_sha256": sha256_path(meta / "modality.json"),
               "episodes_sha256": sha256_path(meta / "episodes.jsonl")}
    return signature, rows, summary, errors


def build(staging: Path, *, tasks_subset: list[str] | None = None, stager_receipts: dict | None = None,
          rows_override: list[dict] | None = None) -> tuple[dict, list[str]]:
    """Returns (manifest, errors). Full mode requires all 65 tasks staged."""
    from experiment import load_stager_receipt
    stage_rows = rows_override if rows_override is not None else load_stage_rows()
    unknown = sorted(set(tasks_subset or []) - {r["task"] for r in stage_rows})
    if unknown:
        raise SystemExit(f"unknown task selector(s): {unknown}")
    sel = [r for r in stage_rows if not tasks_subset or r["task"] in tasks_subset]
    errors, episodes, tasks, reference = [], [], [], None
    for r in sel:
        t = r["task"]
        srec = stager_receipts[t] if stager_receipts is not None else load_stager_receipt(t)
        cp = staging / "receipts" / f"{t}.json"
        if srec is None or not cp.exists():
            errors.append(f"{t}: not staged yet (stager receipt={srec is not None}, staging receipt={cp.exists()})")
            continue
        crec = json.loads(cp.read_text())
        if crec["package_manifest_sha256"] != srec["package_manifest_sha256"]:
            errors.append(f"{t}: staging hash != stager hash")
            continue
        sig, rows, summ, errs = audit_task(t, r["rel_path"], staging, crec, srec)
        if reference is None:
            reference = sig
        if sig != reference:
            errs.append("action/state/video/robot signature differs from first task")
        errors += [f"{t}: {e}" for e in errs]
        episodes += rows
        tasks.append(summ)
        print("audited", t, summ["episodes"], "errors", len(errs), flush=True)
    full = tasks_subset is None
    if full and len(tasks) != N_TASKS:
        errors.append(f"full manifest needs {N_TASKS} tasks, have {len(tasks)}")
    if len({e["episode_id"] for e in episodes}) != len(episodes):
        errors.append("duplicate episode_id")
    m = {
        "version": "atomic65_common_qam_v1", "experiment": EXPERIMENT,
        "mode": "full" if full else "subset", "n_tasks": len(tasks), "n_tasks_expected": N_TASKS,
        "status": "data_preparation_only_not_trained",
        "source_family": "RoboCasa pretrain_human300 atomic65 (official Box archives via 승준 stager)",
        "filter": {"key": f"{FILTER_N}_demos", "seed": FILTER_SEED,
                   "rule": "robocasa groot_dataset.get_subset_demos_filter_key (episodes.jsonl order, "
                           "random.seed(seed).shuffle, first N)",
                   "historical_training_membership_verified": False,
                   "packages_untouched": "all package episodes kept on disk; non-members are not listed"},
        "split_rule": f"per task, sha256('{SPLIT_SALT}'+episode_id) rank, {VAL_FRACTION:.0%} val of members",
        "contract": reference, "feature_signature": FEATURE_SIGNATURE,
        "decision": {"execute_steps": EXECUTE_STEPS, "seed_base": DECISION_SEED_BASE},
        "language": "actual episode task string enters frozen GR00T preprocessing; QAM conditioned on the "
                    "resulting 1024 latent; no task id appended",
        "action": "dataset expert actions (raw LeRobot 12-D) -> model order -> checkpoint min_max/binary "
                  "normalisation (active12 of 32); next observation = next decision latent of the same episode",
        "recipe": RECIPE, "warm_start": None,
        "tasks": tasks, "episodes": episodes,
        "totals": {"episodes": len(episodes), "frames": sum(e["length"] for e in episodes),
                   "decisions": sum((e["length"] + EXECUTE_STEPS - 1) // EXECUTE_STEPS for e in episodes),
                   "train": sum(e["split"] == "train" for e in episodes),
                   "val": sum(e["split"] == "val" for e in episodes),
                   "package_episodes": sum(t["package_episodes"] for t in tasks),
                   "outside_filter": sum(t["outside_filter"] for t in tasks)},
    }
    per_task = Counter(e["task"] for e in episodes)
    short = {t: n for t, n in per_task.items() if n != FILTER_N}
    if short and not errors:
        m["tasks_with_fewer_than_filter_n"] = short   # package had <=100 episodes: rule keeps all
    return m, errors


def write(m: dict, errors: list[str], out: Path, name: str) -> int:
    from experiment import write_json_once
    out.mkdir(parents=True, exist_ok=True)
    if errors:
        (out / (name + ".errors.json")).write_text(json.dumps(errors, indent=1) + "\n")
        print(f"MANIFEST BLOCKED: {len(errors)} errors -> {out / (name + '.errors.json')}")
        return 1
    if (m["mode"] == "full") != (name == "manifest.json"):
        raise SystemExit("only the full 65-task manifest may be named manifest.json")
    write_json_once(out / name, m)
    print(f"MANIFEST OK {m['mode']} tasks={m['n_tasks']} episodes={m['totals']['episodes']} "
          f"decisions={m['totals']['decisions']} -> {out / name} sha256={sha256_path(out / name)[:16]}")
    return 0
