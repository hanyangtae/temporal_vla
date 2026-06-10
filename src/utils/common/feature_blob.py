"""Shared JSON blob helpers for feature tensors."""

from __future__ import annotations

import base64
import io
from typing import Any, Mapping

import numpy as np


def encode_feature_blob(arr: np.ndarray) -> dict[str, Any]:
    """Encode an ndarray as a JSON-safe ``{data, shape, dtype}`` blob."""
    arr = np.asarray(arr)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def decode_feature_blob(blob: Mapping[str, Any]) -> np.ndarray:
    """Decode a blob produced by ``encode_feature_blob``."""
    raw = base64.b64decode(blob["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(blob["dtype"]))
    return arr.reshape(blob["shape"]).copy()


def encode_legacy_feature_array(arr: np.ndarray) -> str:
    """Encode an ndarray as legacy ``base64(np.save(...))`` bytes."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_legacy_feature_array(b64_str: str) -> np.ndarray:
    """Decode legacy ``base64(np.save(...))`` feature bytes."""
    return np.load(io.BytesIO(base64.b64decode(b64_str)), allow_pickle=False)
