#!/usr/bin/env python3
"""exp5-4 sanity: probe npz 활성 ↔ capture-ON rollout pkl 의 record 0 활성 대조.

probe 는 env.reset 직후 t=0 관측으로 inference_seed=s 를 1회 호출한다. 수집 경로의
record 0 도 같은 관측·같은 seed(call_inference_seed = inference_seed + 0)이므로 두
텐서는 **bit 단위로 같아야** 한다. 다르면 (a) 관측 재현 실패 (b) 서버 상태 오염
(c) 비결정 커널 중 하나 — 어느 쪽이든 Phase A 판정 전에 잡아야 한다.

pkl 은 torch 텐서를 담고 있어 unpickle 에 torch 가 필요하다 (원격 분석은
~/anaconda3/bin/python, srv50 은 conda env). numpy 만 있는 인터프리터에서는 실패한다.

사용:
  python check_probe_identity.py --probe probe/scene100000.npz \
      --pkl .../task7--ep0--succ0.pkl [--seed 1056329423]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, type=Path, help="probe npz")
    ap.add_argument("--pkl", required=True, type=Path, help="capture-ON rollout pkl")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="대조할 inference_seed (기본 = pkl 의 inference_seed)",
    )
    args = ap.parse_args()

    npz = np.load(args.probe)
    payload = _load_pkl(args.pkl)

    pkl_seed = payload.get("inference_seed")
    seed = args.seed if args.seed is not None else pkl_seed
    if seed is None:
        raise SystemExit("pkl 에 inference_seed 가 없고 --seed 도 미지정")
    if pkl_seed is not None and int(pkl_seed) != int(seed):
        print(f"[warn] pkl inference_seed={pkl_seed} != 요청 {seed}")

    scene_probe = int(npz["scenario_seed"])
    scene_pkl = payload.get("scenario_seed", payload.get("seed"))
    if scene_pkl is not None and int(scene_pkl) != scene_probe:
        raise SystemExit(f"scene 불일치: probe {scene_probe} vs pkl {scene_pkl}")

    seeds = [int(s) for s in npz["seeds"]]
    if int(seed) not in seeds:
        raise SystemExit(f"probe npz 에 seed {seed} 없음 (있는 값: {seeds})")
    idx = seeds.index(int(seed))

    probe_h = np.asarray(npz["hidden"][idx], dtype=np.float32)
    hs = payload.get("hidden_states")
    if not hs:
        raise SystemExit("pkl 에 hidden_states 가 없음 (capture-ON 으로 수집했는지 확인)")
    rec0 = hs[0]
    rec0 = np.asarray(rec0.cpu().numpy() if hasattr(rec0, "cpu") else rec0, dtype=np.float32)

    print(f"scene={scene_probe} seed={seed} probe{probe_h.shape} vs record0{rec0.shape}")
    if probe_h.shape != rec0.shape:
        raise SystemExit(f"[FAIL] shape 불일치 {probe_h.shape} != {rec0.shape}")

    bit_equal = bool(np.array_equal(probe_h, rec0))
    diff = np.abs(probe_h.astype(np.float64) - rec0.astype(np.float64))
    print(f"hidden : bit_equal={bit_equal} maxabs={diff.max():.3e} mean={diff.mean():.3e}")

    # action chunk 대조 — pkl 의 actions[0] = **첫 inference 의 전체 chunk**(sub-key dict)
    # 를 probe 가 쓴 키 순서로 재구성해 값까지 비교한다 (action_vectors 는 replan 별
    # 첫 스텝 벡터라 대조 대상이 아님 — Gate2 P2 지적 반영).
    action_ok: bool | None = None
    if "action_chunk" in npz.files:
        probe_chunk = np.asarray(npz["action_chunk"][idx], dtype=np.float32)
        keys = json.loads(str(npz["action_chunk_keys"])) if "action_chunk_keys" in npz.files else []
        actions0 = (payload.get("actions") or [None])[0]
        if not keys or not isinstance(actions0, dict):
            print("action : pkl actions[0] 또는 probe 키 목록 없음 — 비교 생략")
        else:
            missing = [k for k in keys if k not in actions0]
            if missing:
                print(f"action : pkl actions[0] 에 키 없음 {missing} (키셋={sorted(actions0)})")
                action_ok = False
            else:
                pieces = []
                for k in keys:
                    arr = np.asarray(actions0[k], dtype=np.float32)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    pieces.append(arr)
                pkl_chunk = np.concatenate(pieces, axis=1)
                if pkl_chunk.shape != probe_chunk.shape:
                    print(f"action : shape 불일치 {probe_chunk.shape} != {pkl_chunk.shape}")
                    action_ok = False
                else:
                    action_ok = bool(np.array_equal(probe_chunk, pkl_chunk))
                    adiff = np.abs(
                        probe_chunk.astype(np.float64) - pkl_chunk.astype(np.float64)
                    ).max()
                    print(
                        f"action : bit_equal={action_ok} maxabs={adiff:.3e} "
                        f"shape={probe_chunk.shape} keys={keys}"
                    )

    # 층별 최대 오차 (어느 layer 부터 갈리는지 — 비결정 커널 위치 힌트)
    if probe_h.ndim >= 1 and not bit_equal:
        layers = [int(x) for x in npz["capture_layers"]] if "capture_layers" in npz.files else []
        per_layer = diff.reshape(diff.shape[0], -1).max(axis=1)
        for i, m in enumerate(per_layer):
            name = layers[i] if i < len(layers) else i
            print(f"  L{name}: maxabs={m:.3e}")
    ok = bit_equal and (action_ok is not False)
    print("VERDICT:", "PASS(bit-identical)" if ok else "FAIL(not bit-identical)")
    if not ok:
        raise SystemExit(1)   # ★비트 불일치는 자동 파이프라인이 반드시 멈춰야 하는 사건


if __name__ == "__main__":
    main()
