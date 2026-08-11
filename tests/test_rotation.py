"""Quaternion→euler 컨버터의 컨벤션(wxyz vs xyzw) 회귀 테스트.

두 컨버터는 같은 수학을 element 순서만 다르게 구현한다. 이름이 곧 가드레일
(`quat_wxyz_to_euler` / `quat_xyzw_to_euler`)이므로, 각 컨버터가 자기 컨벤션의
scipy 결과와 일치하고 다른 컨벤션과는 구별됨을 고정한다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scipy.spatial.transform import Rotation

    _HAS_SCIPY = True
except ImportError:  # scipy 없는 환경에서는 skip
    _HAS_SCIPY = False


def quat_wxyz_to_euler(q):
    """구 DreamVLA adapter(2869696 에서 백본과 함께 삭제)의 wxyz 컨벤션 참조 구현.
    이 테스트의 목적이 '두 컨벤션 혼동 검출'이라 로컬 참조로 유지한다."""
    import numpy as _np
    return _rotation.quat_xyzw_to_euler(_np.asarray(q)[[1, 2, 3, 0]])

from src.processor.action import _rotation


def _random_unit_quats_xyzw(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    # w>=0 로 정규화 (이중 커버 제거)
    q[q[:, 3] < 0] *= -1.0
    return q


def _same_rotation(euler_xyz: np.ndarray, ref: "Rotation") -> bool:
    """euler(xyz)가 ref 와 동일 회전을 나타내는지 (gimbal 모호성 회피)."""
    back = Rotation.from_euler("xyz", euler_xyz)
    rel = (ref * back.inv()).magnitude()  # 0 이면 동일 회전
    return bool(np.all(rel < 1e-5))


@unittest.skipUnless(_HAS_SCIPY, "scipy 필요")
class TestQuatToEulerConventions(unittest.TestCase):
    def setUp(self) -> None:
        self.q_xyzw = _random_unit_quats_xyzw(256, seed=12345)

    def test_xyzw_matches_scipy(self) -> None:
        for q in self.q_xyzw:
            euler = _rotation.quat_xyzw_to_euler(q)
            ref = Rotation.from_quat(q)  # scipy 는 xyzw 입력
            self.assertTrue(_same_rotation(euler, ref), msg=f"xyzw mismatch q={q}")

    def test_wxyz_matches_scipy(self) -> None:
        for q_xyzw in self.q_xyzw:
            q_wxyz = q_xyzw[[3, 0, 1, 2]]  # xyzw → wxyz
            euler = quat_wxyz_to_euler(q_wxyz)
            ref = Rotation.from_quat(q_xyzw)  # 같은 회전
            self.assertTrue(_same_rotation(euler, ref), msg=f"wxyz mismatch q={q_wxyz}")

    def test_conventions_are_distinct(self) -> None:
        """같은 raw 4-vector 를 두 컨벤션으로 해석하면 (일반적으로) 다른 회전.

        컨벤션 혼동이 '우연히 같은 값'이 아니라 실제로 틀린 결과를 내는지 확인.
        """
        diff_count = 0
        for q in self.q_xyzw:
            e_xyzw = _rotation.quat_xyzw_to_euler(q)
            e_wxyz = quat_wxyz_to_euler(q)  # 같은 vector 를 wxyz 로 오해석
            if not np.allclose(e_xyzw, e_wxyz, atol=1e-4):
                diff_count += 1
        # 대부분의 랜덤 quat 에서 두 해석이 달라야 한다 (가드레일의 의미).
        self.assertGreater(diff_count, len(self.q_xyzw) * 0.9)


if __name__ == "__main__":
    unittest.main()
