#!/usr/bin/env python3
"""Validate provenance and file hashes for a locally trained SAFE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


FEATURE_CONTRACT = "pi0_denoise0_action_mean_1024"
ARTIFACT_SUFFIXES = {".pt", ".pth", ".pkl", ".ckpt", ".json", ".yaml", ".yml", ".npy", ".npz"}


class ProvenanceError(ValueError):
    """Raised when a SAFE provenance document or its files are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProvenanceError(f"{label}.sha256 must be a 64-character hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProvenanceError(f"{label}.sha256 is not hexadecimal") from exc
    return value.lower()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{label} must contain a JSON object: {path}")
    return payload


def _checked_relative(path_value: Any, base: Path, label: str, *, confine: bool = True) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ProvenanceError(f"{label}.path must be a non-empty relative path")
    candidate = Path(path_value)
    if candidate.is_absolute():
        raise ProvenanceError(f"{label}.path must be relative: {path_value}")
    resolved = (base / candidate).resolve()
    if confine:
        try:
            resolved.relative_to(base.resolve())
        except ValueError as exc:
            raise ProvenanceError(f"{label}.path escapes its base directory: {path_value}") from exc
    return resolved


def _check_hashed_file(entry: Any, base: Path, label: str, *, confine: bool = True) -> Path:
    if not isinstance(entry, dict):
        raise ProvenanceError(f"{label} must be an object with path and sha256")
    path = _checked_relative(entry.get("path"), base, label, confine=confine)
    expected = _require_sha256(entry.get("sha256"), label)
    if not path.is_file():
        raise ProvenanceError(f"{label} file does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ProvenanceError(f"{label} hash mismatch for {path}: expected {expected}, got {actual}")
    return path


def _load_manifest_rows(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot parse {label} as JSON: {path}: {exc}") from exc


def _episode_values(payload: Any, key: str) -> set[str]:
    values: set[str] = set()
    if isinstance(payload, dict):
        if key in payload and payload[key] is not None:
            values.add(str(payload[key]))
        for value in payload.values():
            values.update(_episode_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.update(_episode_values(value, key))
    return values


def _reject_bundled_paths(payload: Any, label: str) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "path" and isinstance(value, str):
                parts = {part.upper() for part in Path(value).parts}
                if "SAVED" in parts:
                    raise ProvenanceError(f"{label} contains bundled SAVED path: {value}")
            _reject_bundled_paths(value, label)
    elif isinstance(payload, list):
        for value in payload:
            _reject_bundled_paths(value, label)


def validate(safe_dir: Path, manifest_path: Path) -> dict[str, Any]:
    safe_dir = safe_dir.resolve()
    manifest_path = manifest_path.resolve()
    if not safe_dir.is_dir():
        raise ProvenanceError(f"SAFE directory does not exist: {safe_dir}")
    provenance = _load_json(manifest_path, "provenance manifest")
    if provenance.get("owner") != "temporal_vla":
        raise ProvenanceError("provenance.owner must be 'temporal_vla'")
    if provenance.get("feature_contract") != FEATURE_CONTRACT:
        raise ProvenanceError(f"provenance.feature_contract must be '{FEATURE_CONTRACT}'")

    manifest_payloads: dict[str, Any] = {}
    for name in ("training_manifest", "calibration_manifest"):
        entry = provenance.get(name)
        if not isinstance(entry, dict):
            raise ProvenanceError(f"provenance.{name} must be an object with path and sha256")
        manifest_file = _check_hashed_file(entry, manifest_path.parent, f"provenance.{name}", confine=False)
        payload = _load_manifest_rows(manifest_file, f"provenance.{name}")
        if not isinstance(payload, dict) or not payload.get('episodes'):
            raise ProvenanceError(f'provenance.{name} must contain actual episode records')
        if any(not row.get('episode_id') or not row.get('reset_id') for row in payload['episodes']):
            raise ProvenanceError(f'provenance.{name} lacks episode/reset identities')
        _reject_bundled_paths(payload, f"provenance.{name}")
        manifest_payloads[name] = payload
    for key in ("episode_id", "reset_id"):
        training_values = _episode_values(manifest_payloads["training_manifest"], key)
        calibration_values = _episode_values(manifest_payloads["calibration_manifest"], key)
        overlap = training_values & calibration_values
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ProvenanceError(f"training/calibration {key} overlap: {sample}")

    entries = provenance.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ProvenanceError("provenance.artifacts must be a non-empty list")
    listed: dict[Path, str] = {}
    for index, entry in enumerate(entries):
        path = _check_hashed_file(entry, safe_dir, f"provenance.artifacts[{index}]")
        if path == manifest_path:
            raise ProvenanceError("provenance manifest must not list itself as an artifact")
        if path in listed:
            raise ProvenanceError(f"duplicate artifact path: {path}")
        listed[path] = str(entry["path"])

    required_names = ("config.yaml", "model_final.ckpt")
    for required_name in required_names:
        required_path = safe_dir / required_name
        if not required_path.is_file():
            raise ProvenanceError(f"required SAFE file does not exist: {required_path}")
    if not ((safe_dir / "cp_band_by_alpha.npy").is_file() or (safe_dir / "cp_bands_by_task.npy").is_file()):
        raise ProvenanceError(
            "required SAFE CP-band file does not exist: expected cp_band_by_alpha.npy or cp_bands_by_task.npy"
        )

    required = {
        path.resolve()
        for path in safe_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES and path.resolve() != manifest_path
    }
    missing = sorted(required - set(listed), key=str)
    if missing:
        names = ", ".join(str(path.relative_to(safe_dir)) for path in missing)
        raise ProvenanceError(f"SAFE artifact files missing from provenance.artifacts: {names}")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        validate(args.safe_dir, args.manifest)
    except (OSError, ProvenanceError) as exc:
        print(f"safe provenance validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"safe provenance valid: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
