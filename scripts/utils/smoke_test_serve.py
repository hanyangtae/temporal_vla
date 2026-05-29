"""VLA serve 서버 공용 smoke test.

사용:
  python scripts/utils/smoke_test_serve.py \\
      --profile configs/checkpoints/<name>.yaml \\
      --url http://localhost:8400

동작:
  1. /health 를 폴링해 서버가 뜰 때까지 대기
  2. /health 응답이 프로파일의 action_type, emits_subkeys, name 과 일치하는지 검증
  3. 프로파일 observation_requirements 기반으로 dummy payload 생성 후 /act 호출
  4. 응답에 프로파일의 emits_subkeys 가 모두 있고 각 sub-key 의 dim 이 일치하는지 검증

서버 기동까지 자동으로 하지는 않는다. 컨테이너 안에서 serve 스크립트를 먼저 띄운 뒤 이걸 실행.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from checkpoint_profile import CheckpointProfile, load_profile
from vla_client import VLAClient


logger = logging.getLogger(__name__)

# sub-key emit 시 기대하는 차원
_EMIT_DIM = {
    "action.eef_pos": 3,
    "action.eef_euler": 3,
    "action.eef_axisangle": 3,
    "action.eef_quat": 4,
    "action.eef_rot6d": 6,
    "action.gripper": 1,
    "action.joint_pos": 7,
    "action.base_motion": 4,
    "action.control_mode": 1,
}

# state sub-key 에 대한 dummy 값 (payload 조립용)
_DUMMY_STATE = {
    "eef_pos": [0.0, 0.0, 0.0],
    "eef_euler": [0.0, 0.0, 0.0],
    "eef_quat": [0.0, 0.0, 0.0, 1.0],
    "eef_axisangle": [0.0, 0.0, 0.0],
    "eef_pos_rel": [0.0, 0.0, 0.0],
    "eef_quat_rel": [0.0, 0.0, 0.0, 1.0],
    "gripper_qpos": [0.0, 0.0],
    "gripper_action": [0.0],
    "gripper_opening": [0.0],
    "joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "base_position": [0.0, 0.0, 0.0],
    "base_rotation": [0.0, 0.0, 0.0, 1.0],
}


class SmokeTestFailure(AssertionError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SmokeTestFailure(msg)


def _build_dummy_payload(profile: CheckpointProfile) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """프로파일 요구사항 기반 dummy images / states 생성."""
    res = profile.image_preprocess.resolution
    images = {
        view: np.zeros((res, res, 3), dtype=np.uint8)
        for view in profile.observation_requirements.images
    }
    states = {}
    for key in profile.observation_requirements.state:
        if key not in _DUMMY_STATE:
            logger.warning("No dummy value for state key %s; using zeros", key)
            states[f"observation.state.{key}"] = np.zeros(1, dtype=np.float32)
        else:
            states[f"observation.state.{key}"] = np.array(_DUMMY_STATE[key], dtype=np.float32)
    return images, states


def _check_health(info: dict, profile: CheckpointProfile) -> None:
    logger.info("/health = %s", info)
    _require(info.get("status") == "ok", f"/health status != ok: {info}")
    _require(
        info.get("action_type") == profile.action_type,
        f"action_type mismatch: profile={profile.action_type!r} vs health={info.get('action_type')!r}",
    )
    health_keys = set(info.get("action_keys", []))
    profile_keys = set(profile.emits_subkeys)
    _require(
        health_keys == profile_keys,
        f"action_keys mismatch: profile={sorted(profile_keys)} vs health={sorted(health_keys)}",
    )
    if "profile" in info:
        _require(
            info["profile"] == profile.name,
            f"profile name mismatch: expected {profile.name!r}, got {info['profile']!r}",
        )


def _check_action(action_dict: dict, profile: CheckpointProfile) -> None:
    logger.info("/act returned keys: %s", sorted(action_dict.keys()))
    missing = [sk for sk in profile.emits_subkeys if sk not in action_dict]
    _require(not missing, f"response missing sub-keys: {missing}")

    for sk in profile.emits_subkeys:
        arr = action_dict[sk]
        _require(
            arr.ndim == 2,
            f"{sk} must be 2D [n_steps, dim], got shape {arr.shape}",
        )
        n_steps, dim = arr.shape
        expected_dim = _EMIT_DIM.get(sk)
        if expected_dim is not None:
            _require(
                dim == expected_dim,
                f"{sk} dim mismatch: expected {expected_dim}, got {dim}",
            )
        logger.info("  %s: shape=%s, first=%s", sk, arr.shape, arr[0].tolist())


def run_smoke_test(profile_path: str, url: str, max_wait: float) -> int:
    profile = load_profile(profile_path)
    logger.info("Loaded profile %s (base_model=%s)", profile.name, profile.base_model)

    client = VLAClient(url)
    logger.info("Waiting for %s to become ready (max %.0fs) ...", url, max_wait)
    info = client.wait_until_ready(max_wait=max_wait, poll_interval=3.0)

    try:
        _check_health(info, profile)
    except SmokeTestFailure as e:
        logger.error("[/health FAIL] %s", e)
        return 1
    logger.info("[/health OK]")

    images, states = _build_dummy_payload(profile)
    logger.info(
        "Dummy payload: images=%s, states=%s",
        list(images.keys()),
        list(states.keys()),
    )
    client.reset()
    action_dict, latency_ms = client.predict(
        images=images, states=states, instruction="pick up the object"
    )
    _require(
        isinstance(action_dict, dict),
        f"expected sub-keyed response (dict), got {type(action_dict).__name__}",
    )
    logger.info("/act latency: %.1fms", latency_ms)

    try:
        _check_action(action_dict, profile)
    except SmokeTestFailure as e:
        logger.error("[/act FAIL] %s", e)
        return 1
    logger.info("[/act OK]")
    logger.info("[SMOKE TEST PASS] profile=%s", profile.name)
    return 0


def main():
    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger("smoke_test_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="VLA serve 서버 smoke test")
    parser.add_argument("--profile", type=str, required=True,
                        help="체크포인트 프로파일 YAML 경로")
    parser.add_argument("--url", type=str, default="http://localhost:8400",
                        help="serve URL (default: :8400)")
    parser.add_argument("--max-wait", type=float, default=600.0,
                        help="/health ready 폴링 최대 시간(초). 초기 모델 로드가 길면 증가.")
    args = parser.parse_args()

    _require(Path(args.profile).is_file(), f"profile not found: {args.profile}")
    sys.exit(run_smoke_test(args.profile, args.url, args.max_wait))


if __name__ == "__main__":
    main()
