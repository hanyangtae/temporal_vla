#!/usr/bin/env python3
# 유래: exp5-4 (2026-08-10 기능명 재배치 — docs/review/RENAME_PLAN.md)
"""exp5-4 probe 결정성 smoke (robocasa 컨테이너 안에서 실행).

Codex Gate1 리뷰의 smoke 4건 중 서버측 3건을 담당한다 (④ 동일 rollout 2회는
probe_collect.sh --stage smoke 가 collector 를 2회 돌려 처리):

  ① A-B-A : 같은 (obs, seed) 로 capture-ON 3회 (A, B=다른 seed, A 재호출)
             → A vs A' 의 hidden·action 이 bit 단위로 같은가 (서버 상태 오염 검출)
  ② capture-ON vs skip_features : 같은 (obs, seed) 의 action chunk 동일성
             (캡처 hook 이 추론 결과를 바꾸지 않는가)
  ③ cross-server : 같은 요청을 serve 전부에 → 0번 서버 대비 동일성
             (worker 간 결과 차이 = 판 배정에 따른 confound 검출)

각 항목은 bit-equal 여부 + maxabs 를 표로 출력한다.

사용(컨테이너):
  python smoke_probe.py --task OpenDrawer \
     --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
     --seed 100000 --inference-seed 1056329423 \
     --servers http://127.0.0.1:8620,http://127.0.0.1:8621
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "collect"))

import http_feature_collect as hfc  # noqa: E402  (sys.path 세팅 후 — make_env 등 본류 재사용)
import probe_lib  # noqa: E402  (같은 디렉토리 — probe 전용 클라이언트/해시/평탄화)


def _cmp(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    return bool(np.array_equal(a, b)), float(
        np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
    )


def _call(client, images, states, instruction, seed, *, skip_features=False):
    client.reset()
    extra = {"skip_features": 1} if skip_features else None
    actions, features, _lat = client.predict_with_features(
        images, states, instruction, inference_seed=int(seed), extra_payload=extra
    )
    chunk, _keys = probe_lib._flatten_action_chunk(actions)
    hidden = None
    if features and features.get("hidden_states") is not None:
        hidden = np.asarray(features["hidden_states"], dtype=np.float32)
        if hidden.ndim == 5 and hidden.shape[0] == 1:
            hidden = hidden[0]
    return hidden, chunk.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--split", default="target")
    ap.add_argument("--seed", type=int, required=True, help="scenario seed (scene)")
    ap.add_argument("--inference-seed", type=int, required=True, help="후보 base seed")
    ap.add_argument("--alt-inference-seed", type=int, default=None, help="①의 B 호출 seed")
    ap.add_argument("--servers", required=True, help="콤마 구분 serve URL 목록")
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--max-episode-steps", type=int, default=720)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    if not servers:
        raise SystemExit("--servers 비어 있음")
    seed_a = int(args.inference_seed)
    seed_b = int(args.alt_inference_seed if args.alt_inference_seed is not None else seed_a + 7)

    env = hfc.make_env(
        args.task,
        args.split,
        env_name=args.env_name,
        scenario_seed=args.seed,
        video_dir=None,
        overlay_text=False,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    rows: list[tuple[str, str, str]] = []
    try:
        obs, _info = env.reset(seed=args.seed)
        images, states, instruction = hfc.official_obs_to_lerobot_inputs(obs)
        obs_hash = probe_lib._obs_hash(images, states)
        print(f"scene={args.seed} instr={instruction!r} obs_hash={obs_hash[:16]}")

        clients = [probe_lib._ProbeClient(url, timeout=args.timeout) for url in servers]
        for c in clients:
            c.wait_until_ready(max_wait=args.timeout)
        c0 = clients[0]

        # ① A - B - A
        hA, aA = _call(c0, images, states, instruction, seed_a)
        hB, aB = _call(c0, images, states, instruction, seed_b)
        hA2, aA2 = _call(c0, images, states, instruction, seed_a)
        eq, mx = _cmp(hA, hA2)
        rows.append(("① A-B-A hidden", str(eq), f"{mx:.3e}"))
        eq, mx = _cmp(aA, aA2)
        rows.append(("① A-B-A action", str(eq), f"{mx:.3e}"))
        eq, mx = _cmp(hA, hB)
        rows.append(("① A vs B(다른 seed, 달라야 정상)", str(eq), f"{mx:.3e}"))

        # ② capture-ON vs skip_features
        hS, aS = _call(c0, images, states, instruction, seed_a, skip_features=True)
        if hS is not None:
            rows.append(("② skip_features 인데 features 반환", "ANOMALY", "-"))
        eq, mx = _cmp(aA, aS)
        rows.append(("② capture-ON vs skip_features action", str(eq), f"{mx:.3e}"))

        # ③ cross-server
        for url, c in zip(servers[1:], clients[1:]):
            h, a = _call(c, images, states, instruction, seed_a)
            eq, mx = _cmp(hA, h)
            rows.append((f"③ {url} hidden vs srv0", str(eq), f"{mx:.3e}"))
            eq, mx = _cmp(aA, a)
            rows.append((f"③ {url} action vs srv0", str(eq), f"{mx:.3e}"))
    finally:
        env.close()

    width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'check'.ljust(width)}{'bit_equal'.ljust(12)}maxabs")
    for name, eq, mx in rows:
        print(f"{name.ljust(width)}{eq.ljust(12)}{mx}")


if __name__ == "__main__":
    main()
