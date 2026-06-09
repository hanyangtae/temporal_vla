#!/usr/bin/env python
"""Internal parity checks for LeRobot GR00T N1.5 RoboCasa365 loading.

This diagnostic targets evidence below rollout success rate:

1. selected raw Isaac-GR00T checkpoint prefixes are present in the loaded
   LeRobot-wrapped GR00T model;
2. tensor shapes match;
3. tensor values match within a configured tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PROFILE = REPO_ROOT / "configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"


@dataclass(frozen=True)
class CheckpointPrefixSpec:
    name: str
    checkpoint_prefix: str
    model_prefix: str | None = None

    def model_key_for(self, checkpoint_key: str) -> str:
        if self.model_prefix is None:
            return checkpoint_key
        suffix = checkpoint_key[len(self.checkpoint_prefix) :]
        return f"{self.model_prefix}{suffix}"


DEFAULT_CHECKPOINT_PREFIX_SPECS: tuple[CheckpointPrefixSpec, ...] = (
    CheckpointPrefixSpec("eagle_vision", "backbone.eagle_model.vision_model."),
    CheckpointPrefixSpec("eagle_projector", "backbone.eagle_model.mlp1."),
    CheckpointPrefixSpec("action_head", "action_head."),
)


def _checkpoint_shards(checkpoint_dir: Path) -> list[Path]:
    shards = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found under {checkpoint_dir}")
    return shards


def _load_checkpoint_tensors_for_specs(
    checkpoint_dir: str | Path,
    specs: Sequence[CheckpointPrefixSpec],
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    prefixes = tuple(spec.checkpoint_prefix for spec in specs)
    tensors: dict[str, torch.Tensor] = {}
    for shard_path in _checkpoint_shards(checkpoint_dir):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key.startswith(prefixes):
                    tensors[key] = shard.get_tensor(key)
    return tensors


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().to("cpu").flatten()
    if flat.dtype == torch.bfloat16:
        flat = flat.to(torch.float32)
    if not flat.dtype.is_floating_point:
        flat = flat.to(torch.float32)
    head = flat[:8].to(torch.float32).tolist()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "head8": head,
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    left_cpu = left.detach().to("cpu")
    right_cpu = right.detach().to("cpu")
    if left_cpu.numel() == 0:
        return 0.0, 0.0
    diff = left_cpu.to(torch.float32) - right_cpu.to(torch.float32)
    abs_diff = diff.abs()
    return float(abs_diff.max().item()), float(abs_diff.mean().item())


def _is_close(left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float) -> bool:
    if left.dtype == right.dtype and torch.equal(left.detach().to("cpu"), right.detach().to("cpu")):
        return True
    return bool(
        torch.allclose(
            left.detach().to("cpu").to(torch.float32),
            right.detach().to("cpu").to(torch.float32),
            atol=atol,
            rtol=rtol,
        )
    )


def _unexpected_model_keys(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_keys: set[str],
    spec: CheckpointPrefixSpec,
) -> list[str]:
    if spec.model_prefix == "":
        return []
    model_prefix = spec.checkpoint_prefix if spec.model_prefix is None else spec.model_prefix
    unexpected = []
    for model_key in model_state:
        if not model_key.startswith(model_prefix):
            continue
        suffix = model_key[len(model_prefix) :]
        checkpoint_key = f"{spec.checkpoint_prefix}{suffix}"
        if checkpoint_key not in checkpoint_keys:
            unexpected.append(model_key)
    return sorted(unexpected)


def compare_checkpoint_prefixes(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_dir: str | Path,
    specs: Sequence[CheckpointPrefixSpec] = DEFAULT_CHECKPOINT_PREFIX_SPECS,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare raw checkpoint tensors with a loaded model state dict.

    ``model_prefix=None`` means checkpoint and model keys are identical.
    ``model_prefix=""`` means the checkpoint prefix is stripped before lookup,
    which is useful for comparing a submodule state dict such as ``mlp1``.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_tensors = _load_checkpoint_tensors_for_specs(checkpoint_dir, specs)
    checkpoint_keys = set(checkpoint_tensors)
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "checked_count": 0,
        "missing_count": 0,
        "shape_mismatch_count": 0,
        "value_mismatch_count": 0,
        "dtype_cast_value_mismatch_count": 0,
        "unexpected_count": 0,
        "max_abs": 0.0,
        "dtype_cast_max_abs": 0.0,
        "prefixes": {},
    }

    for spec in specs:
        prefix_keys = sorted(key for key in checkpoint_tensors if key.startswith(spec.checkpoint_prefix))
        prefix_report: dict[str, Any] = {
            "checkpoint_prefix": spec.checkpoint_prefix,
            "model_prefix": spec.model_prefix,
            "checkpoint_count": len(prefix_keys),
            "checked_count": 0,
            "missing": [],
            "shape_mismatches": [],
            "value_mismatches": [],
            "unexpected": _unexpected_model_keys(model_state, checkpoint_keys, spec),
            "max_abs": 0.0,
            "dtype_cast_value_mismatch_count": 0,
            "dtype_cast_max_abs": 0.0,
        }

        for checkpoint_key in prefix_keys:
            expected = checkpoint_tensors[checkpoint_key]
            model_key = spec.model_key_for(checkpoint_key)
            actual = model_state.get(model_key)
            if actual is None:
                prefix_report["missing"].append(
                    {"checkpoint_key": checkpoint_key, "model_key": model_key}
                )
                continue
            if tuple(actual.shape) != tuple(expected.shape):
                prefix_report["shape_mismatches"].append(
                    {
                        "checkpoint_key": checkpoint_key,
                        "model_key": model_key,
                        "checkpoint_shape": list(expected.shape),
                        "model_shape": list(actual.shape),
                        "checkpoint": _tensor_summary(expected),
                        "model": _tensor_summary(actual),
                    }
                )
                continue

            prefix_report["checked_count"] += 1
            max_abs, mean_abs = _max_abs_diff(actual, expected)
            expected_cast = expected.detach().to("cpu").to(dtype=actual.detach().to("cpu").dtype)
            cast_max_abs, cast_mean_abs = _max_abs_diff(actual, expected_cast)
            prefix_report["max_abs"] = max(float(prefix_report["max_abs"]), max_abs)
            prefix_report["dtype_cast_max_abs"] = max(
                float(prefix_report["dtype_cast_max_abs"]),
                cast_max_abs,
            )
            if not _is_close(actual, expected_cast, atol=atol, rtol=rtol):
                prefix_report["dtype_cast_value_mismatch_count"] += 1
            if not _is_close(actual, expected, atol=atol, rtol=rtol):
                prefix_report["value_mismatches"].append(
                    {
                        "checkpoint_key": checkpoint_key,
                        "model_key": model_key,
                        "max_abs": max_abs,
                        "mean_abs": mean_abs,
                        "dtype_cast_max_abs": cast_max_abs,
                        "dtype_cast_mean_abs": cast_mean_abs,
                        "checkpoint": _tensor_summary(expected),
                        "model": _tensor_summary(actual),
                    }
                )

        report["checked_count"] += int(prefix_report["checked_count"])
        report["missing_count"] += len(prefix_report["missing"])
        report["shape_mismatch_count"] += len(prefix_report["shape_mismatches"])
        report["value_mismatch_count"] += len(prefix_report["value_mismatches"])
        report["dtype_cast_value_mismatch_count"] += int(
            prefix_report["dtype_cast_value_mismatch_count"]
        )
        report["unexpected_count"] += len(prefix_report["unexpected"])
        report["max_abs"] = max(float(report["max_abs"]), float(prefix_report["max_abs"]))
        report["dtype_cast_max_abs"] = max(
            float(report["dtype_cast_max_abs"]),
            float(prefix_report["dtype_cast_max_abs"]),
        )
        report["prefixes"][spec.name] = prefix_report

    report["mismatch_count"] = (
        report["missing_count"]
        + report["shape_mismatch_count"]
        + report["value_mismatch_count"]
        + report["unexpected_count"]
    )
    return report


def _configure_import_paths() -> None:
    for path in (
        REPO_ROOT,
        REPO_ROOT / "scripts",
        REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "utils",
        REPO_ROOT / "scripts" / "utils",
        REPO_ROOT / "scripts" / "serve",
        REPO_ROOT / "configs" / "policies",
        REPO_ROOT / "src" / "policies" / "Isaac-GR00T-N1.5",
        REPO_ROOT / "lerobot" / "src",
    ):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def _resolve_pretrained_path(profile: Any) -> Path:
    if profile.checkpoint_source.type == "local":
        return Path(profile.checkpoint_source.id)

    from huggingface_hub import snapshot_download

    subfolder = profile.model_specific.get("checkpoint_subfolder")
    if not subfolder:
        raise ValueError("hf_repo profile requires model_specific.checkpoint_subfolder")
    snapshot = snapshot_download(
        repo_id=profile.checkpoint_source.id,
        repo_type="model",
        local_files_only=True,
    )
    return Path(snapshot) / subfolder


def load_lerobot_groot_model_state(
    *,
    profile_path: str | Path = DEFAULT_PROFILE,
    checkpoint_dir: str | Path | None = None,
    device: str = "cuda",
) -> tuple[dict[str, torch.Tensor], Path]:
    """Load the adapter path and return ``policy._groot_model.state_dict()``."""

    _configure_import_paths()

    from checkpoint_profile import load_profile
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot_adapters.groot import GrootPolicyAdapter

    profile = load_profile(Path(profile_path))
    pretrained_path = Path(checkpoint_dir) if checkpoint_dir is not None else _resolve_pretrained_path(profile)
    loaded = GrootPolicyAdapter().load(
        profile=profile,
        policy_cls=GrootPolicy,
        pretrained_path=pretrained_path,
        dataset_stats=None,
        device=device,
    )
    model = getattr(loaded.policy, "_groot_model", None)
    if model is None:
        raise ValueError("Loaded LeRobot policy does not expose _groot_model")
    return dict(model.state_dict()), pretrained_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/debug/lerobot_groot_n15_internal_parity/checkpoint_load.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_state, checkpoint_dir = load_lerobot_groot_model_state(
        profile_path=args.profile,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    report = compare_checkpoint_prefixes(
        model_state,
        checkpoint_dir,
        DEFAULT_CHECKPOINT_PREFIX_SPECS,
        atol=args.atol,
        rtol=args.rtol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("checked_count", "mismatch_count", "max_abs")}, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
