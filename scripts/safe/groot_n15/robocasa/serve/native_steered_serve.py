#!/usr/bin/env python3
"""Native GR00T N1.5 ZMQ serve + optional COAST conceptor steering.

Mirrors ``Isaac-GR00T-N1.5/scripts/inference_service.py`` (Gr00tPolicy +
RobotInferenceServer ZMQ) but registers a ``ConceptorSteering`` forward hook on
``policy.model.action_head.model.transformer_blocks[layer]`` — the same injection
point the lerobot path uses (``steering_hooks.ConceptorSteering``). No submodule edit.

Baseline serve = run without ``--steering-npz`` (identical serve, hook absent), so
baseline-vs-steered differ only by the conceptor gate.
"""

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402
from gr00t.eval.robot import RobotInferenceServer  # noqa: E402

from scripts.serve.steering_hooks import ConceptorSteering, load_steering_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--api-token", default=None)
    # COAST steering (optional). Omit --steering-npz for the baseline serve.
    ap.add_argument("--steering-npz", default=None)
    ap.add_argument("--steering-beta", type=float, default=0.1)
    ap.add_argument("--steering-alpha", type=float, default=None)
    ap.add_argument("--steering-layer", type=int, default=None,
                    help="DiT transformer_block index. None = action_head.model output.")
    ap.add_argument("--steering-key", default="C_steer",
                    choices=("C_steer", "C_success", "C_failure"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_data_config(args.data_config)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    if args.steering_npz:
        M = load_steering_matrix(
            args.steering_npz, args.steering_beta,
            alpha=args.steering_alpha, key=args.steering_key,
        )
        ConceptorSteering(
            policy.model, M, pathway="dit", layer=args.steering_layer,
        ).register()
        print(
            f"[steer] REGISTERED npz={args.steering_npz} layer={args.steering_layer} "
            f"alpha={args.steering_alpha} beta={args.steering_beta} key={args.steering_key} "
            f"M={M.shape}",
            flush=True,
        )
    else:
        print("[steer] BASELINE (no steering hook)", flush=True)

    print(f"[serve] RobotInferenceServer port={args.port}", flush=True)
    RobotInferenceServer(policy, port=args.port, api_token=args.api_token).run()


if __name__ == "__main__":
    main()
