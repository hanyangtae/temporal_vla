"""GR00T N1.6 inference server for the project FastAPI interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Keep direct script execution aligned with the docker PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from path_setup import configure_repo_paths  # noqa: E402

configure_repo_paths(include_script_utils=True, include_groot=True)

from checkpoint_profile import load_profile  # noqa: E402
from src.utils.common.serving import (  # noqa: E402
    add_server_args,
    run_uvicorn,
    setup_serve_logging,
)
from src.policies.groot.safe.features import FEATURE_DTYPES, FEATURE_SLICES  # noqa: E402
from src.policies.groot.core.schema import build_video_mapping  # noqa: E402
from src.policies.groot.core.service import (  # noqa: E402
    GrootFeatureConfig,
    GrootModelNotLoadedError,
    GrootPolicyService,
    temporary_inference_seed,
)


logger = logging.getLogger(__name__)

app = FastAPI(title="GR00T N1.6 Inference Server")
_service = GrootPolicyService()


def _apply_runtime_args() -> None:
    _service.feature_config = GrootFeatureConfig.from_args(getattr(app.state, "args", None))


def _build_groot_obs(payload: dict) -> dict:
    return _service.build_groot_obs(payload)


def _log_debug_call(out: dict, payload: dict) -> None:
    _service._log_debug_call(out, payload)


_temporary_inference_seed = temporary_inference_seed


@app.on_event("startup")
def load_model() -> None:
    try:
        _load_model_impl()
    except Exception:
        import traceback

        sys.stderr.write("=== load_model FAILED ===\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _load_model_impl() -> None:
    args = app.state.args
    _service.load_policy(device=args.device)


@app.post("/reset")
async def reset():
    return _service.reset()


@app.post("/act")
async def predict_action(payload: dict):
    _apply_runtime_args()
    try:
        return _service.act(payload)
    except GrootModelNotLoadedError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/act_with_features")
async def predict_action_with_features(payload: dict):
    _apply_runtime_args()
    try:
        return _service.act_with_features(payload)
    except GrootModelNotLoadedError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/health")
async def health():
    _apply_runtime_args()
    return _service.health()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GR00T N1.6 inference server (port 8500)")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Checkpoint profile YAML path (configs/checkpoints/*.yaml)",
    )
    add_server_args(parser, default_port=8500)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override profile model_specific.device only when needed",
    )
    parser.add_argument(
        "--model-path-override",
        type=str,
        default=None,
        help="Override profile.checkpoint_source.id with an explicit path",
    )
    parser.add_argument(
        "--feature-slice",
        type=str,
        choices=FEATURE_SLICES,
        default="valid",
        help=(
            "/act_with_features DiT pre-velocity slice. "
            "valid=embodiment decoded horizon, all=model horizon"
        ),
    )
    parser.add_argument(
        "--feature-dtype",
        type=str,
        choices=FEATURE_DTYPES,
        default="float16",
        help="/act_with_features hidden_states serialization dtype",
    )
    parser.add_argument(
        "--feature-action-horizon",
        type=int,
        default=None,
        help="Leading action-token count to export. Defaults to the feature-slice horizon.",
    )
    return parser


def main() -> None:
    setup_serve_logging("groot_serve")

    args = build_parser().parse_args()

    profile = load_profile(args.profile)
    if args.model_path_override is not None:
        profile.checkpoint_source.id = args.model_path_override
    logger.info("Loaded profile %s from %s", profile.name, args.profile)
    assert profile.base_model == "groot", (
        f"profile.base_model={profile.base_model!r}, but this server is groot"
    )

    _service.profile = profile
    _service.feature_config = GrootFeatureConfig.from_args(args)
    _service.step_count = 0
    app.state.args = args
    run_uvicorn(app, args)


if __name__ == "__main__":
    main()
