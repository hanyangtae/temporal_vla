#!/usr/bin/env python3
"""Load N1.5 instruction-fixed DiT and VL features without N1.6 shape assumptions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
N16_ROBOCASA_ROOT = REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa"
if str(N16_ROBOCASA_ROOT) not in sys.path:
    sys.path.insert(0, str(N16_ROBOCASA_ROOT))

from safe_feature_vectors import (  # noqa: E402
    parse_aggregation_command,
    pooled_hidden_states,
    process_tensor_idx_rel,
    tensor_to_numpy,
)


@dataclass(frozen=True)
class InstructionFeatureExample:
    path: Path
    task: str
    cell_id: str
    cell_index: int
    episode_idx: int
    scenario_seed: int
    success: int
    dit_features: np.ndarray
    dit_layer_features: np.ndarray | None
    dit_capture_layers: tuple[int, ...] | None
    vl_features: np.ndarray | None
    # COAST denoise format only: per-row denoise step index and originating env step.
    denoise_idx: np.ndarray | None = None
    env_step_idx: np.ndarray | None = None


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_vl_features(payload: dict[str, Any], *, require_vl: bool) -> np.ndarray | None:
    vl_hidden_states = payload.get("vl_hidden_states") or []
    if not vl_hidden_states:
        if require_vl:
            raise ValueError("missing vl_hidden_states")
        return None

    vectors = []
    for hidden in vl_hidden_states:
        hidden_np = tensor_to_numpy(hidden).astype(np.float32, copy=False)
        if hidden_np.ndim != 1:
            raise ValueError(f"Expected VL hidden state [D], got {hidden_np.shape}")
        vectors.append(hidden_np)
    return np.stack(vectors, axis=0)


def _load_dit_layer_features(
    payload: dict[str, Any],
    *,
    token_idx_rel: float | str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    feature_axes = list(payload.get("feature_axes") or [])
    if feature_axes[:2] != ["layer", "model_token"]:
        raise ValueError(
            "preserve_dit_layers requires block-residual feature axes "
            f"['layer', 'model_token', ...], got {feature_axes}"
        )
    capture_layers = tuple(int(layer) for layer in (payload.get("capture_layers") or []))
    if not capture_layers:
        raise ValueError("preserve_dit_layers requires capture_layers metadata")

    features = []
    for hidden in payload["hidden_states"]:
        hidden_np = tensor_to_numpy(hidden).astype(np.float32, copy=False)
        if hidden_np.ndim != 3:
            raise ValueError(f"Expected hidden state [L, T, D], got {hidden_np.shape}")
        token_pooled = process_tensor_idx_rel(hidden_np, token_idx_rel)
        if token_pooled.ndim != 2:
            raise ValueError(f"Expected token-pooled hidden state [L, D], got {token_pooled.shape}")
        if token_pooled.shape[0] != len(capture_layers):
            raise ValueError(
                "capture_layers length does not match hidden state layer axis: "
                f"layers={len(capture_layers)} hidden={token_pooled.shape[0]}"
            )
        features.append(token_pooled)
    if not features:
        return np.empty((0, len(capture_layers), 0), dtype=np.float32), capture_layers
    return np.stack(features, axis=0), capture_layers


def _load_denoise_features(
    payload: dict[str, Any],
    *,
    require_vl: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
]:
    """COAST-faithful loader for ``["layer","denoise_step","feature_dim"]`` features.

    Each env-step hidden is ``[L, K, D]`` (action-token mean, denoise K preserved). For the
    global strategy COAST treats every denoising step as an independent sample, so we expand
    K into rows: env-step -> K rows of ``[L, D]``. Returns ``(dit_features [N,D] (layer-mean),
    dit_layer_features [N,L,D], capture_layers, vl_features [N,Dvl]|None, denoise_idx [N],
    env_step_idx [N])`` with ``N = sum_env(K)``. The pooled ``dit`` is only an alignment/summary
    pathway; COAST steering uses a single ``dit_layer<id>``. VL fires once per env-step, so it
    is repeated x K to align (duplication is fine — VL is not used for COAST DiT steering).
    """
    feature_axes = list(payload.get("feature_axes") or [])
    if feature_axes[:2] != ["layer", "denoise_step"]:
        raise ValueError(
            f"_load_denoise_features requires ['layer','denoise_step',...] axes, got {feature_axes}"
        )
    capture_layers = tuple(int(layer) for layer in (payload.get("capture_layers") or []))
    if not capture_layers:
        raise ValueError("denoise features require capture_layers metadata")

    dit_rows: list[np.ndarray] = []
    layer_rows: list[np.ndarray] = []
    denoise_idx: list[int] = []
    env_step_idx: list[int] = []
    for env_i, hidden in enumerate(payload["hidden_states"]):
        h = tensor_to_numpy(hidden).astype(np.float32, copy=False)
        if h.ndim != 3:
            raise ValueError(f"Expected denoise hidden [L, K, D], got {h.shape}")
        n_layers, n_denoise, _ = h.shape
        if n_layers != len(capture_layers):
            raise ValueError(
                f"capture_layers ({len(capture_layers)}) != hidden layer axis ({n_layers})"
            )
        for k in range(n_denoise):
            layer_vec = h[:, k, :]  # [L, D]
            layer_rows.append(layer_vec)
            dit_rows.append(layer_vec.mean(axis=0))  # [D] layer-mean (summary pathway)
            denoise_idx.append(k)
            env_step_idx.append(env_i)

    denoise_idx_arr = np.asarray(denoise_idx, dtype=np.int64)
    env_step_idx_arr = np.asarray(env_step_idx, dtype=np.int64)
    if not dit_rows:
        empty_layers = np.empty((0, len(capture_layers), 0), dtype=np.float32)
        return (
            np.empty((0, 0), dtype=np.float32),
            empty_layers,
            capture_layers,
            None,
            denoise_idx_arr,
            env_step_idx_arr,
        )
    dit_features = np.stack(dit_rows, axis=0)  # [N, D]
    dit_layer_features = np.stack(layer_rows, axis=0)  # [N, L, D]

    vl_features = None
    vl_raw = _load_vl_features(payload, require_vl=require_vl)  # [n_env, Dvl] | None
    if vl_raw is not None:
        n_env = len(payload["hidden_states"])
        if len(vl_raw) != n_env:
            raise ValueError(f"VL/env-step mismatch: vl={len(vl_raw)} env={n_env}")
        counts = np.bincount(env_step_idx_arr, minlength=n_env)
        vl_features = np.repeat(vl_raw, counts, axis=0)  # [N, Dvl] (env VL repeated x K)

    return (
        dit_features,
        dit_layer_features,
        capture_layers,
        vl_features,
        denoise_idx_arr,
        env_step_idx_arr,
    )


def load_instruction_feature_examples(
    root: Path,
    *,
    horizon_idx_rel: str = "mean",
    diff_idx_rel: str = "mean",
    require_vl: bool = False,
    preserve_dit_layers: bool = False,
    cell_ids: set[str] | None = None,
) -> list[InstructionFeatureExample]:
    horizon_command = parse_aggregation_command(horizon_idx_rel)
    diff_command = parse_aggregation_command(diff_idx_rel)
    examples: list[InstructionFeatureExample] = []

    for path in sorted(root.glob("*/*/*.pkl")):
        payload = _load_pickle(path)
        cell_id = str(payload.get("cell_id") or path.parent.name)
        if cell_ids is not None and cell_id not in cell_ids:
            continue

        feature_axes = list(payload.get("feature_axes") or [])
        is_denoise = feature_axes[:2] == ["layer", "denoise_step"]
        denoise_idx = None
        env_step_idx = None
        if preserve_dit_layers and is_denoise:
            # COAST-faithful [L,K,D]: expand denoise K into sample rows.
            (
                dit_features,
                dit_layer_features,
                dit_capture_layers,
                vl_features,
                denoise_idx,
                env_step_idx,
            ) = _load_denoise_features(payload, require_vl=require_vl)
        else:
            dit_features = pooled_hidden_states(
                payload,
                horizon_idx_rel=horizon_command,
                diff_idx_rel=diff_command,
            )
            dit_layer_features = None
            dit_capture_layers = None
            if preserve_dit_layers:
                dit_layer_features, dit_capture_layers = _load_dit_layer_features(
                    payload,
                    token_idx_rel=horizon_command,
                )
                if len(dit_layer_features) != len(dit_features):
                    raise ValueError(
                        f"Layer-preserved/pooled step count mismatch for {path}: "
                        f"layers={len(dit_layer_features)} pooled={len(dit_features)}"
                    )
            vl_features = _load_vl_features(payload, require_vl=require_vl)
            if vl_features is not None and len(vl_features) != len(dit_features):
                raise ValueError(
                    f"VL/DiT step count mismatch for {path}: "
                    f"vl={len(vl_features)} dit={len(dit_features)}"
                )

        examples.append(
            InstructionFeatureExample(
                path=path,
                task=str(payload.get("robocasa_task") or path.parent.parent.name),
                cell_id=cell_id,
                cell_index=int(payload.get("cell_index", payload.get("task_id", -1))),
                episode_idx=int(payload.get("episode_idx", -1)),
                scenario_seed=int(payload.get("scenario_seed", -1)),
                success=int(payload.get("episode_success", 0)),
                dit_features=dit_features,
                dit_layer_features=dit_layer_features,
                dit_capture_layers=dit_capture_layers,
                vl_features=vl_features,
                denoise_idx=denoise_idx,
                env_step_idx=env_step_idx,
            )
        )

    return examples


def flatten_examples(examples: list[InstructionFeatureExample]) -> dict[str, np.ndarray]:
    dit_chunks: list[np.ndarray] = []
    dit_layer_chunks: list[np.ndarray] = []
    vl_chunks: list[np.ndarray] = []
    capture_layers: tuple[int, ...] | None = None
    success: list[int] = []
    task: list[str] = []
    cell_id: list[str] = []
    cell_index: list[int] = []
    episode_idx: list[int] = []
    scenario_seed: list[int] = []
    step_idx: list[int] = []
    denoise_idx: list[int] = []
    has_denoise = True

    for example in examples:
        n_steps = int(example.dit_features.shape[0])
        dit_chunks.append(example.dit_features)
        if example.dit_layer_features is not None:
            if capture_layers is None:
                capture_layers = example.dit_capture_layers
            elif capture_layers != example.dit_capture_layers:
                raise ValueError(
                    "Mismatched capture_layers across examples: "
                    f"{capture_layers} != {example.dit_capture_layers}"
                )
            dit_layer_chunks.append(example.dit_layer_features)
        if example.vl_features is not None:
            vl_chunks.append(example.vl_features)
        success.extend([example.success] * n_steps)
        task.extend([example.task] * n_steps)
        cell_id.extend([example.cell_id] * n_steps)
        cell_index.extend([example.cell_index] * n_steps)
        episode_idx.extend([example.episode_idx] * n_steps)
        scenario_seed.extend([example.scenario_seed] * n_steps)
        # COAST denoise format: step_idx is the originating env step (K rows share it);
        # otherwise step_idx is the per-env-step running index.
        if example.env_step_idx is not None:
            step_idx.extend(int(s) for s in example.env_step_idx.tolist())
        else:
            step_idx.extend(range(n_steps))
        if example.denoise_idx is not None:
            denoise_idx.extend(int(d) for d in example.denoise_idx.tolist())
        else:
            has_denoise = False

    flattened = {
        "dit": np.concatenate(dit_chunks, axis=0) if dit_chunks else np.empty((0, 0)),
        "success": np.asarray(success, dtype=np.int64),
        "task": np.asarray(task),
        "cell_id": np.asarray(cell_id),
        "cell_index": np.asarray(cell_index, dtype=np.int64),
        "episode_idx": np.asarray(episode_idx, dtype=np.int64),
        "scenario_seed": np.asarray(scenario_seed, dtype=np.int64),
        "step_idx": np.asarray(step_idx, dtype=np.int64),
    }
    if has_denoise and len(denoise_idx) == len(success):
        flattened["denoise_idx"] = np.asarray(denoise_idx, dtype=np.int64)
    if vl_chunks:
        flattened["vl"] = np.concatenate(vl_chunks, axis=0)
    if dit_layer_chunks:
        dit_layers = np.concatenate(dit_layer_chunks, axis=0)
        flattened["dit_layers"] = dit_layers
        assert capture_layers is not None
        flattened["dit_capture_layers"] = np.asarray(capture_layers, dtype=np.int64)
        for local_idx, layer_id in enumerate(capture_layers):
            flattened[f"dit_layer{layer_id}"] = dit_layers[:, local_idx, :]
    return flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="raw_rollouts root with task/cell pkl layout")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--horizon-idx-rel", default="mean")
    parser.add_argument("--diff-idx-rel", default="mean")
    parser.add_argument("--require-vl", action="store_true")
    parser.add_argument(
        "--preserve-dit-layers",
        action="store_true",
        help=(
            "For block-residual features [layer, model_token, feature_dim], "
            "also store dit_layers [row, layer, dim] and dit_layer<id> keys."
        ),
    )
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = load_instruction_feature_examples(
        args.root,
        horizon_idx_rel=args.horizon_idx_rel,
        diff_idx_rel=args.diff_idx_rel,
        require_vl=args.require_vl,
        preserve_dit_layers=bool(args.preserve_dit_layers),
        cell_ids=None if args.cell_ids is None else set(args.cell_ids),
    )
    arrays = flatten_examples(examples)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    print(f"wrote {sum(example.dit_features.shape[0] for example in examples)} feature rows -> {args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
