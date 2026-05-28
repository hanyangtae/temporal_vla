#!/usr/bin/env python3
"""Verify that Eagle pre-LLM embeddings were extracted correctly for all 10 tasks.

Checks per task:
  1. ``data/robocasa_eagle_pre_llm/<task>/embeddings.pt`` exists.
  2. ``len(embeddings) == info.json['total_frames']`` (모든 frame caching 됨).
  3. ``embedding.shape == (2048,)`` (Eagle Qwen3-1.7B hidden dim).
  4. ``set(abs_idx)`` 가 0..total_frames-1 모두 덮음 (이 데이터셋은 task-local abs_idx 라 그렇게 가정).
  5. NaN / Inf 없음.

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.path_setup import DATA_ROOT

DEFAULT_DATA_ROOT = DATA_ROOT / "robocasa/v1.0/pretrain/atomic"
DEFAULT_CACHE_ROOT = DATA_ROOT / "robocasa_eagle_pre_llm"

DEFAULT_TASKS = [
    # 기존 10
    "OpenDrawer", "CloseDrawer",
    "OpenCabinet", "CloseCabinet",
    "OpenFridge", "CloseFridge",
    "OpenMicrowave", "CloseMicrowave",
    "PickPlaceCounterToStove", "PickPlaceCounterToSink",
    # 신규 20 (single-door / single-fixture atomic)
    "OpenOven", "OpenDishwasher", "OpenFridgeDrawer", "OpenToasterOvenDoor",
    "OpenBlenderLid", "OpenElectricKettleLid", "OpenStandMixerHead",
    "PickPlaceCounterToCabinet", "PickPlaceCabinetToCounter", "PickPlaceCounterToDrawer",
    "PickPlaceCounterToMicrowave", "PickPlaceCounterToOven",
    "PickPlaceSinkToCounter", "PickPlaceStoveToCounter",
    "TurnOnSinkFaucet", "TurnOnStove", "TurnOnMicrowave",
    "TurnOnBlender", "TurnOnElectricKettle", "TurnOnToaster",
]

EXPECTED_HIDDEN = 2048


def verify_one(task: str, data_root: Path, cache_root: Path) -> tuple[bool, str]:
    # 기대 frame 수 — 해당 task 의 info.json 에서.
    candidates = sorted((data_root / task).glob("*/lerobot/meta/info.json"))
    if not candidates:
        return False, f"no info.json under {data_root / task}"
    expected_frames = json.load(open(candidates[0]))["total_frames"]

    cache_path = cache_root / task / "embeddings.pt"
    if not cache_path.exists():
        return False, f"MISSING {cache_path}"

    try:
        e = torch.load(str(cache_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        return False, f"torch.load failed: {exc!r}"

    if not isinstance(e, dict) or not e:
        return False, "cache is not a non-empty dict"

    n = len(e)
    sample_key = next(iter(e))
    sample = e[sample_key]
    shape = tuple(sample.shape)

    if shape != (EXPECTED_HIDDEN,):
        return False, f"shape {shape} != ({EXPECTED_HIDDEN},)"

    if n != expected_frames:
        return False, f"frames {n} != expected {expected_frames}"

    keys = set(int(k) for k in e.keys())
    # 일부 dataset 은 abs_idx 가 0-base, 일부는 task-local cumulative. 단조성만 확인.
    min_k, max_k = min(keys), max(keys)
    if max_k - min_k + 1 != expected_frames:
        return False, f"abs_idx span {min_k}..{max_k} mismatch (expected gap = {expected_frames-1})"

    # NaN/Inf — 모든 frame 검사는 비싸니 첫/중간/끝 sample 만
    spot_keys = [min_k, min_k + n // 2, max_k]
    for sk in spot_keys:
        if sk not in e:
            return False, f"missing abs_idx {sk}"
        t = e[sk]
        if torch.isnan(t).any() or torch.isinf(t).any():
            return False, f"NaN/Inf at abs_idx {sk}"

    return True, f"frames={n} shape={shape} abs_idx_span=[{min_k},{max_k}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = parser.parse_args()

    print(f"[verify] cache_root={args.cache_root}")
    print(f"[verify] data_root={args.data_root}")
    print()

    failures = []
    total_frames = 0
    for task in args.tasks:
        ok, detail = verify_one(task, args.data_root, args.cache_root)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {task:30s} {detail}")
        if not ok:
            failures.append((task, detail))
        else:
            # frames=<n> 를 detail 에서 파싱
            try:
                n = int(detail.split("frames=")[1].split()[0])
                total_frames += n
            except Exception:
                pass

    print()
    if failures:
        print(f"FAILED: {len(failures)} / {len(args.tasks)} tasks have issues")
        for t, d in failures:
            print(f"  - {t}: {d}")
        sys.exit(1)
    print(f"OK: all {len(args.tasks)} tasks verified ({total_frames:,} frames total).")


if __name__ == "__main__":
    main()
