#!/usr/bin/env python3
"""seed → instruction 레지스트리 스캔 (정책·GPU 불필요, env reset 만).

RoboCasa 는 scenario_seed 마다 task variant 가 바뀐다:
  - OpenDrawer            : "Open the left/right/top ... drawer." (좌우 변이)
  - PickPlaceCounterToCabinet : 물체가 beer/can/mug/... 로 변이
따라서 cell 을 고정하려면 seed 를 미리 선별해야 한다 (exp4-1 사이드카 실측으로 확인된 사실).
출력: TSV (seed, instruction) — canonical instruction 매칭은 호출측에서.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _extract_lang(env, obs) -> str:
    """instruction 정본 추출. 수집기는 obs 의 language 필드를 쓰므로 obs 를 우선한다."""
    if isinstance(obs, dict):
        for k in ("lang", "language", "task_description", "instruction"):
            v = obs.get(k)
            if isinstance(v, bytes):
                v = v.decode("utf-8", "ignore")
            if isinstance(v, str) and v.strip():
                return v.strip()
    # fallback: ep_meta 계층 탐색 (wrapper 깊이가 버전마다 다름)
    getters = []
    try:
        getters.append(env.get_wrapper_attr("get_ep_meta"))
    except Exception:  # noqa: BLE001
        pass
    node = getattr(env, "unwrapped", env)
    for _ in range(4):
        fn = getattr(node, "get_ep_meta", None)
        if callable(fn):
            getters.append(fn)
        node = getattr(node, "env", None)
        if node is None:
            break
    for fn in getters:
        try:
            meta = fn() or {}
            v = meta.get("lang", "")
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:  # noqa: BLE001
            continue
    return "__NOLANG__"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=0)  # exclusive
    ap.add_argument("--seed-list", default="", help="쉼표구분 seed 목록 (range 대신 명시 지정)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401

    if args.seed_list.strip():
        seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
        seeds = seeds[args.offset :: args.stride]
    else:
        seeds = list(range(args.seed_start + args.offset, args.seed_end, args.stride))
    with open(args.out, "w") as fh:
        for s in seeds:
            lang = ""
            try:
                # 수집기와 동일: gym.make(seed=...) 로 env.rng 고정 후 reset(seed=...)
                env = gym.make(args.env_name, enable_render=False, seed=s)
                obs, _info = env.reset(seed=s)
                lang = _extract_lang(env, obs)
                env.close()
            except Exception as exc:  # noqa: BLE001
                lang = f"__ERROR__ {type(exc).__name__}: {exc}"
            fh.write(f"{s}\t{lang}\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
