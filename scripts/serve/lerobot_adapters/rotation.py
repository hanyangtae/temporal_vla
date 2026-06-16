"""Rotation helpers for LeRobot serving adapters."""

from __future__ import annotations

from typing import Any
import math

import numpy as np


def quat_xyzw_to_axisangle(quat: Any) -> np.ndarray:
    """Quaternion (x, y, z, w) -> axis-angle (3D)."""
    quat = np.asarray(quat, dtype=np.float64)
    w = max(-1.0, min(1.0, float(quat[3])))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def quat_wxyz_to_rotation_6d(value: Any) -> np.ndarray:
    """Quaternion (w, x, y, z) -> first two rows of a rotation matrix."""
    quat = np.asarray(value, dtype=np.float64).flatten()
    if quat.size < 4:
        quat = np.pad(quat, (0, 4 - quat.size))
        quat[0] = 1.0
    w, x, y, z = quat[:4]
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 0:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return matrix[:2, :].reshape(6)
