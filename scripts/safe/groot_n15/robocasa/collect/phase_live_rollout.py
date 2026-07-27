"""라이브 robocasa 롤아웃 + 라이브 phase readout 수집 (렌더용 산출물 저장).

robocasa env 를 실제로 굴리며 매 inference 마다 라이브 gr00t serve(/act_with_features)를
호출한다. 응답의 features.phase (serve --phase-readout 가 DiT layer-12 residual 로 산출한
AE/SAE action-phase) 를 프레임과 함께 저장한다 → phase_live_render.py 가 참고영상 스타일로
렌더한다.

삭제된 gr00t.eval 래퍼(http_feature_collect 경로)를 우회 — robocasa_eval 의 GrootRoboCasaEnv
경로만 쓴다.

    python phase_live_rollout.py --env-id robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env \
        --instruction "Pick the bread ..." --seed 100084 --inference-seed 0 \
        --max-steps 720 --out /temporal_vla/outputs/phase_live_demo/bread_seen
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

# GT 라벨러 의존 경로: env_step_phase(n15 collect) 는 robocasa_event_labeler(n16 collect) 를 import.
_REPO = Path(__file__).resolve().parents[5]
for _p in (_REPO / "scripts/safe/groot_n16/robocasa/collect",
           _REPO / "scripts/safe/groot_n15/robocasa/collect",
           _REPO / "scripts/utils", _REPO / "scripts", _REPO):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from path_setup import configure_repo_paths  # noqa
configure_repo_paths()

from src.processor.factory import make_groot_robocasa_processors
from src.processor.types import TransitionKey
from src.policies.groot.robocasa.io import latest_image_frame

# robocasa_eval 의 프레임 선택 키 (obs 에서 렌더용 뷰 3개)
try:
    from scripts.eval.robocasa_eval import GROOT_HTTP_VIDEO_FRAME_KEYS
except Exception:
    GROOT_HTTP_VIDEO_FRAME_KEYS = ("video.image_side_0", "video.image_side_1", "video.image_wrist_0")


def split_obs(processed_obs):
    images, states, instruction = {}, {}, ""
    for k, v in processed_obs.items():
        if k.startswith("observation.images."):
            images[k.split(".")[-1]] = v
        elif k.startswith("observation.state."):
            states[k.split(".")[-1]] = v
        elif k in ("task", "instruction", "prompt"):
            instruction = v if isinstance(v, str) else instruction
    return images, states, instruction


def frame_of(obs):
    # 3-뷰 가로 스택 (참고영상처럼)
    views = []
    for key in GROOT_HTTP_VIDEO_FRAME_KEYS:
        if key in obs:
            f = latest_image_frame(obs[key])
            if f is not None:
                views.append(np.asarray(f))
    if not views:
        return None
    h = min(v.shape[0] for v in views)
    views = [v[:h] for v in views]
    return np.concatenate(views, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--instruction", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--inference-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--vla-server", default="http://127.0.0.1:8400")
    ap.add_argument("--n-action-steps", type=int, default=5,
                    help="get_action 당 실행할 env-step 수 (execute-n). 학습/평가 정합=5.")
    ap.add_argument("--no-proximity", action="store_true",
                    help="GT 라벨러 proximity sub-phase(grasp/place) 끄기. 기본은 pq3 와 동일하게 켬.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vla_client import VLAClient
    import gymnasium as gym
    import robocasa  # noqa: F401  (env 등록)
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401
    from env_step_phase import EnvStepGT, StepPhaseProbeWrapper  # env-step GT 라벨러

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    obs_pipeline, action_pipeline = make_groot_robocasa_processors(strict=False, action_mode="step")
    client = VLAClient(url=a.vla_server, timeout=300.0)

    env_kwargs = {"enable_render": True}
    if a.seed is not None:
        env_kwargs["seed"] = a.seed
    env = gym.make(a.env_id, **env_kwargs)
    env = StepPhaseProbeWrapper(env)   # 매 env.step 후 GT 라벨러 콜백 (execute-1 → 프레임당 1 GT)

    reset_out = env.reset(seed=a.seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    client.reset()

    env_gt = None
    try:
        env_gt = EnvStepGT(env, a.env_id, not a.no_proximity)
        env.set_gt(env_gt)
        env_gt.start()   # s_0
    except Exception as e:
        print(f"[warn] GT 라벨러 비활성: {e}", flush=True)

    frames, phases, latencies = [], [], []
    success = False
    step = 0          # env-step 수 (프레임 수)
    n_inf = 0         # get_action 호출 수 (execute-n 이므로 step != n_inf)
    done = False
    try:
        while step < a.max_steps and not done:
            processed = obs_pipeline({TransitionKey.OBSERVATION: obs})[TransitionKey.OBSERVATION]
            images, states, instr = split_obs(processed)
            instr = instr or a.instruction
            # get_action 1회 = 16-step action chunk. 학습/평가와 동일하게 execute-n(기본 5) 실행.
            actions, features, latency = client.predict_with_features(
                images, states or None, instr,
                inference_seed=a.inference_seed + n_inf,
            )
            n_inf += 1
            latencies.append(latency)
            phase = (features or {}).get("phase")   # serve --phase-readout 산출 (라이브, 이 chunk 기준)

            chunk_len = len(np.asarray(next(iter(actions.values()))))
            for j in range(min(a.n_action_steps, chunk_len)):
                aj = {k: np.asarray(v)[j:j+1] for k, v in actions.items()}   # j번째 스텝만 [1,dim]
                ea = action_pipeline({TransitionKey.ACTION: aj})[TransitionKey.ACTION]
                obs, _r, terminated, truncated, info = env.step(ea)
                if info.get("success", False):
                    success = True
                gt_phase = env_gt.phases[-1] if (env_gt is not None and env_gt.phases) else None
                fr = frame_of(obs)
                if fr is not None:
                    frames.append(fr)
                    phases.append({"live": phase, "gt": gt_phase})   # phase 는 chunk 단위(5프레임 공유)
                step += 1
                if terminated or truncated or success or step >= a.max_steps:
                    done = True
                    break
            if n_inf % 10 == 0:
                p = phase.get("ae", {}) if isinstance(phase, dict) else {}
                print(f"[inf{n_inf} step{step}] ae={p.get('phase')} succ={success} lat={latency:.0f}ms", flush=True)
    finally:
        env.close()

    np.savez_compressed(out / "frames.npz", frames=np.asarray(frames, dtype=np.uint8))
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({"phases": phases, "instruction": a.instruction, "env_id": a.env_id,
                     "seed": a.seed, "success": bool(success), "n_frames": len(frames),
                     "latency_ms_mean": float(np.mean(latencies)) if latencies else None}, f)
    print(f"wrote {len(frames)} frames, success={success} -> {out}", flush=True)


if __name__ == "__main__":
    main()
