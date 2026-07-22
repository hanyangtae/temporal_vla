"""patchceil — 승준 zst pkl 에서 t0 라벨·donor 선정·action-replay 재료만 추출 (원격 실행).

hidden_states(무거움)는 버리고 meta JSON + actions NPZ 만 남긴다:
  /tmp/patchceil_meta/<cell>/ep{N}.json      — phases/event_steps/타임라인/시드
  /tmp/patchceil_meta/<cell>/ep{N}_actions.npz — action key 별 [R,16,dim]

실행 (로컬에서 stdin 파이프 — scp 금지 규약 준수, provenance 는 이 파일):
  ssh -p 11112 kimseungjun@166.104.146.37 'nice -n 10 ~/anaconda3/bin/python -' \
    < scripts/safe/groot_n15/robocasa/steer/patchceil/extract_ep_meta_remote.py
"""
from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path.home() / (
    "datasets/temporal_vla_outputs/eval/robocasa/groot_n15/"
    "phase_event_6p/raw_rollouts/PickPlaceCounterToCabinet"
)
OUT = Path("/tmp/patchceil_meta")
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]

META_KEYS = [
    "cell_id", "episode_idx", "episode_success", "scenario_seed", "inference_seed",
    "n_action_steps", "num_inference_timesteps", "max_episode_steps",
    "feature_phases", "phase_timeline", "phase_scheme", "event_steps", "event_order",
    "grasp_steps", "drop_steps", "wrong_grasp_steps", "grasp_count",
    "model_action_horizon", "valid_action_horizon", "capture_layers", "feature_kind",
]


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def main() -> None:
    for cell in CELLS:
        out_dir = OUT / cell
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((ROOT / cell).glob("task5--ep*--succ*.pkl.zst"))
        assert len(files) == 60, f"{cell}: {len(files)} != 60"
        for f in files:
            raw = subprocess.run(
                ["zstd", "-dcq", str(f)], check=True, capture_output=True
            ).stdout
            d = pickle.loads(raw)
            ep = int(d["episode_idx"])
            meta = {k: json_safe(d.get(k)) for k in META_KEYS}
            meta["n_records"] = len(d.get("actions") or [])
            meta["source"] = f.name
            (out_dir / f"ep{ep}.json").write_text(
                json.dumps(meta, ensure_ascii=False)
            )
            acts = d.get("actions") or []
            arrays = {}
            for key in acts[0].keys():
                arrays[key.replace("action.", "a_")] = np.stack(
                    [np.asarray(r[key], dtype=np.float32) for r in acts], axis=0
                )
            np.savez(out_dir / f"ep{ep}_actions.npz", **arrays)
        print(f"{cell}: {len(files)} eps -> {out_dir}", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
