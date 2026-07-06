"""seed 재현성 확인용 단발 스크립트 (robocasa 컨테이너 안에서 실행).

같은 (env_name, seed) 로 reset 한 뒤 초기 scene 이미지 + ep_meta(layout/style/objects/lang)
를 덤프한다. 서로 다른 process 로 두 번 돌려 산출물을 비교하면 "seed 고정 시 같은 환경이
불려오는가" 를 확인할 수 있다.

usage: python repro_check.py <env_name> <seed> <run_tag> <out_dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/temporal_vla/scripts/safe/groot_n16/robocasa/collect")
sys.path.insert(0, "/temporal_vla")

import imageio.v2 as imageio  # noqa: E402

from collect_env import make_robocasa_env  # noqa: E402
from src.policies.groot.robocasa.scenario_replay import get_robocasa_ep_meta  # noqa: E402


def find_images(o):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            try:
                a = np.asarray(v)
            except Exception:
                continue
            if a.ndim == 3 and a.shape[-1] == 3 and a.shape[0] > 8:
                out[k] = a
            elif a.ndim == 4 and a.shape[-1] == 3:
                out[k] = a[-1]
            elif a.ndim == 5 and a.shape[-1] == 3:
                out[k] = a[0, -1]
    return out


def main():
    env_name, seed, run_tag, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)

    env = make_robocasa_env(env_name, scenario_seed=seed)
    obs, _info = env.reset(seed=seed)

    # ep_meta = 가장 결정적인 재현성 지표 (layout/style/object/lang)
    try:
        ep = get_robocasa_ep_meta(env)
    except Exception as e:  # noqa: BLE001
        ep = {"_error": repr(e)}
    lang = ep.get("lang") if isinstance(ep, dict) else None
    print(f"[{run_tag}] lang = {lang!r}")
    if isinstance(ep, dict):
        print(f"[{run_tag}] ep_meta keys = {list(ep.keys())}")
        for k in ("layout_id", "style_id"):
            if k in ep:
                print(f"[{run_tag}] {k} = {ep[k]}")
    json.dump(ep, open(f"{out_dir}/ep_meta_{run_tag}.json", "w"), default=str, indent=1)

    imgs = find_images(obs)
    print(f"[{run_tag}] obs={type(obs).__name__} image_candidates="
          f"{ {k: tuple(v.shape) for k, v in imgs.items()} }")
    saved = []
    for k, v in imgs.items():
        sk = k.replace("/", "_").replace(".", "_")
        p = f"{out_dir}/{run_tag}__{sk}.png"
        imageio.imwrite(p, np.asarray(v).astype("uint8"))
        saved.append(p)
    if not saved:
        try:
            r = env.render()
            a = np.asarray(r)
            print(f"[{run_tag}] env.render() -> {type(r).__name__} shape={getattr(a, 'shape', None)}")
            if a.ndim == 3 and a.shape[-1] == 3:
                p = f"{out_dir}/{run_tag}__render.png"
                imageio.imwrite(p, a.astype("uint8"))
                saved.append(p)
        except Exception as e:  # noqa: BLE001
            print(f"[{run_tag}] render fallback failed: {e!r}")
    print(f"[{run_tag}] saved: {saved}")
    env.close()


if __name__ == "__main__":
    main()
