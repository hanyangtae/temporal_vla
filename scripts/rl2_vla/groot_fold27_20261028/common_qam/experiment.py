"""atomic65 common QAM experiment definition (single source for every common_qam module).

Data: the official RoboCasa pretrain human atomic65 packages staged by
scripts/rl2_vla/atomic65_stage_20261028 (Box archive -> 승준 -> compute staging).
Training set: per task the registry `100_demos` members (seed 0 rule), task-balanced.
This is a new, independent experiment; it is NOT the earlier local 10-task or target13
runs, and it never warm-starts from them.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ADAPTER = REPO / "scripts/rl2_vla/groot_robocasa_adapter_20261021"
TARGET13 = REPO / "scripts/rl2_vla/groot_target13_20261022"   # TaskBalancedSampler/RL2 split (read-only reuse)
STAGER = REPO / "scripts/rl2_vla/atomic65_stage_20261028"
for p in (STAGER, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from scripts.path_setup import DATA_ROOT  # noqa: E402  (kanu cache single source)

EXPERIMENT = "atomic65_common_qam_20261029"
STAGER_OUT = REPO / "outputs/analysis/atomic65_stage_20261028"
STAGER_MANIFEST = STAGER_OUT / "atomic65_manifest.tsv"
STAGER_RECEIPTS = STAGER_OUT / "receipts"
SEUNGJUN_ROOT = "/home/kimseungjun/datasets/robocasa_atomic65_20261028"
KANU_ROBOCASA = DATA_ROOT / "robocasa"                     # pre-existing kanu copies (reuse only if hash-identical)
STAGING_ROOT = DATA_ROOT / "atomic65_staging_20261029"     # <root>/robocasa/v1.0/... ; container: /cache/datasets/...
WORK = REPO / "outputs/rl2_vla/atomic65_common_qam_work_20261029"   # gitignored; manifest/extraction/train
ARCHIVE_REMOTE = "/home/kimseungjun/datasets/temporal_vla_store/checkpoints/atomic65_common_qam_20261029"

N_TASKS = 65
FILTER_N, FILTER_SEED = 100, 0           # registry pretrain human -> "100_demos"; LeRobotSingleDataset seed default 0
VAL_FRACTION = 0.1                       # per task, of the filter members
SPLIT_SALT = "atomic65_common_qam_v1_seed42|"

# Feature / action contract: identical to the frozen-GR00T QAM line (groot_robocasa_adapter_20261021).
FEATURE_SIGNATURE = "ccbfb51ca01724cb"   # action_decoder_input_last16_mean_denoise0, 1024-d
CHECKPOINT_REVISION = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
LATENT_DIM = 1024
DECODER_CALLS = 4                        # flow steps per decision
ENV_ACTION_DIM = 12                      # PandaOmron: eef pos3, rot3, gripper1, base4, control_mode1
EXECUTE_STEPS = 5
DECISION_SEED_BASE = 42
ROBOT = "PandaOmron"

# Recipe: unchanged from groot_target13_20261022/train_scratch.py (fresh production prefix QAM).
RECIPE = {
    "profile": "production", "gamma": 0.99, "seed": 42, "execute_steps": EXECUTE_STEPS,
    "batch_size": 256, "steps": 500_000, "eval_every": 50_000,
    "smoke_steps": 20, "smoke_eval_every": 10, "eval_seed": 20261021,
    "selection": "fixed final step (no SR/MSE selection); MSE diagnostic only",
    "sampling": "uniform task, then uniform train transition within task",
    "reward": "RL2 successful-expert final3 (-1, last3 zero); raw rewards preserved in sidecar",
    "init": "fresh PrefixQAMTrainer.create; no restore / no warm start",
    "data_label": "official_atomic65_human_100demos_scratch",
}


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def filter_members(episode_indices_in_file_order: list[int], n: int = FILTER_N, seed: int = FILTER_SEED):
    """RoboCasa groot_dataset.get_subset_demos_filter_key: shuffle file order with
    random.seed(seed), keep the first n; n >= total means every episode."""
    ids = list(episode_indices_in_file_order)
    state = random.getstate()
    random.seed(seed)
    random.shuffle(ids)
    random.setstate(state)
    return list(episode_indices_in_file_order) if n >= len(ids) else ids[:n]


def episode_id(rel_path: str, index: int) -> str:
    """Same identity format as groot_multitask_20261021 (trainer requires robocasa/v1.0/pretrain/)."""
    return "robocasa/" + rel_path.removesuffix("/lerobot") + "/episode_%06d" % index


def val_episodes(eids: list[str]) -> set[str]:
    k = round(len(eids) * VAL_FRACTION)
    ranked = sorted(eids, key=lambda e: hashlib.sha256((SPLIT_SALT + e).encode()).hexdigest())
    return set(ranked[:k])


def load_stage_rows() -> list[dict]:
    import csv
    with open(STAGER_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != N_TASKS:
        raise SystemExit(f"{STAGER_MANIFEST}: {len(rows)} rows != {N_TASKS}")
    return rows


def load_stager_receipt(task: str) -> dict | None:
    p = STAGER_RECEIPTS / f"{task}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("status") == "COMPLETE" else None


def write_json_once(p: Path, obj) -> None:
    """Create, or accept an identical existing file; never replace a different one."""
    text = json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
    if p.exists():
        if p.read_text() != text:
            raise SystemExit(f"refusing to overwrite different {p}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(p)
