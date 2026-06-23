"""Rotation conversion helpers shared by action processors."""

from __future__ import annotations

import numpy as np


def rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation -> euler (roll, pitch, yaw)."""
    a1 = rot6d[..., 0::2]
    a2 = rot6d[..., 1::2]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    pitch = -np.arcsin(np.clip(b1[..., 2], -1.0, 1.0))
    roll = np.arctan2(b2[..., 2], b3[..., 2])
    yaw = np.arctan2(b1[..., 1], b1[..., 0])

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def quat_xyzw_to_euler(quat: np.ndarray) -> np.ndarray:
    """Quaternion -> euler (roll, pitch, yaw).

    Convention: ``(x, y, z, w)`` (scalar-last). wxyz(scalar-first) 가 필요하면
    인덱스만 ``(w, x, y, z)`` 로 맞춰 동일 공식을 적용한다.
    """
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)
