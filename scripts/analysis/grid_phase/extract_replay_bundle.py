#!/usr/bin/env python
"""rollout.pkl 에서 **replay 에 필요한 최소 키만** 뽑아 소용량 번들로 저장 (원격 실행용).

동기: grid rollout.pkl 은 hidden_states 때문에 600MB/개다. 클린 영상 재생성(replay)에
필요한 것은 actions·scenario_seed·env_name·ep_meta·states 뿐 (수 MB). 원격(승준 노드)에서
이 스크립트로 번들을 만들고 그것만 회수한다 — pkl 통째 전송 금지 (docs/04).

사용 (원격, torch 있는 python 필요 — pkl 이 torch 텐서를 담고 있어 unpickle 에 필요):
    ~/anaconda3/bin/python scripts/analysis/grid_phase/extract_replay_bundle.py \
        --pkl <abs path to rollout.pkl> --out <abs path to bundle.pkl>

번들 키: env_name, scenario_seed, seed, ep_meta, n_action_steps, steps_per_render,
video_fps, episode_success, task_description, actions(각 record 의 앞 n_action_steps 만),
states(record 별 observation.state.*), feature_phases, machine, grid_instruction,
scene_idx, noise_idx, pkl_sha256(원본 지문 — docs/04 sig 규약).
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

KEEP_SCALAR = (
    "env_name", "scenario_seed", "seed", "ep_meta", "n_action_steps", "steps_per_render",
    "video_fps", "episode_success", "task_description", "inference_seed", "machine",
    "grid_instruction", "scene_idx", "noise_idx", "plan_id", "armsig", "max_episode_steps",
    "feature_phases", "action_keys",
)


def _to_np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu()
    return np.asarray(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.pkl)
    d = pickle.load(open(src, "rb"))
    nas = int(d.get("n_action_steps", 5))

    # 실행된 것은 chunk 앞 n_action_steps 뿐 (MultiStepWrapper) → 나머지는 버린다.
    actions = [{k: _to_np(v)[:nas] for k, v in rec.items()} for rec in d["actions"]]
    states = [{k: _to_np(v) for k, v in st.items()} for st in (d.get("states") or [])]

    bundle = {k: d.get(k) for k in KEEP_SCALAR if k in d}
    bundle["actions"] = actions
    bundle["states"] = states
    bundle["source_pkl_name"] = src.name
    bundle["pkl_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[bundle] {src} -> {out}  records={len(actions)} states={len(states)} "
          f"size={out.stat().st_size / 1e6:.2f}MB sig={bundle['pkl_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
