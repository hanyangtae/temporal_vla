"""exp4-2 Track P — 물리/관측 섭동으로 실패 유도 (docs/steering/24b §1, 2026-07-22 메뉴 개정판).

4모드 (WAM_Steer arXiv:2607.14943 정렬 + 기존 유지분):
- **C1_camera**: reset 직후 agentview 좌/우 카메라 `cam_pos/cam_quat/cam_fovy` 를 seeded δ로
  변조 (WAM: δpos σ=0.10m, δrot σ=8°, δfov σ=5°). 에피소드 내내 지속(렌더 전용, 물리 비접촉).
  변조 후 첫 obs 를 재취득해 policy 첫 추론부터 perturbed 관측을 보게 한다.
- **G1_gripper_init**: reset 직후 scripted OSC delta chunk 로 EE 를 δxyz(σ=10cm) 실이동 후
  policy 시작 (WAM gripper 초기위치 방식). record↔env-step 정렬이 `env_step_offset` 만큼 밀림.
- **P1_displace**: trigger record 에서 target free-joint qpos xy 에 δ 가산 + `sim.forward()`
  (pre-grasp, RePO E3 계열). teleport 시각 불연속 → fit 은 창끝+4 부터 (manifest builder 규칙).
- **P2_force**: trigger record 에서 `xfrc_applied[bid,:3]` 수평 wrench set, duration record
  경계에서 명시적 0 해제 (mujoco 가 매 물리스텝 자동 적용 — per-step 훅 불필요).

결정론 규약: 모든 확률량은 `spec_seed` 의 `np.random.default_rng` 로만 샘플하고, `resolve()`
가 확정한 resolved dict 를 사이드카/pkl 에 기록한다 (spec+seed 의 순수 함수).

sham 규약 (S1 배선 무오염 증명):
- C1-sham: 카메라 값을 **원값 그대로 재기록** + obs 재취득까지 실행 (렌더 결정론 검증 겸용,
  csv bitwise ≡ baseline 기대).
- G1-sham: scripted 구간 전체 skip (`sham_skipped_sim_write`) — bitwise ≡ baseline 기대.
- P1-sham: sim write skip (여분 `mj_forward()` 의 warmstart 재계산이 bitwise 를 깰 수 있어
  기본 skip; S1 에서 "δ=0 + forward 실행" 판을 1회 실측해 승격 여부 결정).
- P2-sham: `xfrc_applied = 0` 을 **실기록** (이미 0 → 증명 가능하게 inert).

접속 지점 (http_feature_collect.py):
- 에피소드 시작: ``obs = perturber.on_episode_start(obs)`` (env.reset + env_gt.start 후).
- 루프: ``official_action = perturber.maybe_apply(record_idx, official_action)``
  (get_action 직후, env.step 직전 — record_idx = progress_before).
- 저장: ``extra_metadata.update(perturber.export())``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("C1_camera", "G1_gripper_init", "P1_displace", "P2_force")

# WAM_Steer 기본 σ (C1) — scale 배수로 grid 스윕.
C1_SIGMA_POS_M = 0.10
C1_SIGMA_ROT_DEG = 8.0
C1_SIGMA_FOVY_DEG = 5.0
DEFAULT_C1_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")

# OSC_POSE (default_pandaomron.json): 정규화 입력 ±1 → ±0.05 m/step.
OSC_POS_SCALE_M = 0.05
# G1 per-step 이동 상한 (정규화 0.8 = 4 cm — 포화 여유).
G1_MAX_STEP_DELTA_M = 0.04


@dataclass
class PerturbSpec:
    """단일 에피소드 섭동 사양. json 직렬화 그대로 --perturb-spec 으로 전달."""

    mode: str
    spec_seed: int = 0
    sham: bool = False
    # P1-sham 변형: sim write 를 skip 하지 않고 **원값 재기록 + sim.forward() 실행**
    # (S1 에서 warmstart 재계산이 bitwise 를 깨는지 실측하는 용도 — 24b sham 규약).
    sham_forward: bool = False
    tag: str = ""
    # C1
    scale: float = 1.0
    cameras: tuple[str, ...] = DEFAULT_C1_CAMERAS
    # G1
    sigma_xyz_m: float = 0.10
    # P1/P2 공통
    target: str = "obj"
    trigger_record: int = -1
    magnitude: float = 0.0  # P1: |δ| m / P2: |F| N
    direction: tuple[float, float] | None = None  # 수평 단위벡터 (미지정 → seeded 샘플)
    # P2
    duration_records: int = 2

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown perturb mode: {self.mode} (expected {MODES})")
        if self.mode in ("P1_displace", "P2_force"):
            if not self.sham and self.trigger_record < 0:
                raise ValueError(f"{self.mode}: trigger_record 필수 (got {self.trigger_record})")
            if not self.sham and self.magnitude <= 0:
                raise ValueError(f"{self.mode}: magnitude > 0 필수 (got {self.magnitude})")
        if self.mode == "P2_force" and self.duration_records <= 0:
            raise ValueError(f"P2_force: duration_records > 0 필수 (got {self.duration_records})")

    @classmethod
    def from_json(cls, text: str) -> "PerturbSpec":
        """inline json(`{`로 시작) 또는 파일 경로를 파싱."""
        raw = text.strip()
        if not raw.startswith("{"):
            raw = Path(raw).read_text()
        d = json.loads(raw)
        if "cameras" in d:
            d["cameras"] = tuple(d["cameras"])
        if d.get("direction") is not None:
            d["direction"] = tuple(float(v) for v in d["direction"])
        return cls(**d)

    def resolve(self) -> dict[str, Any]:
        """spec_seed 로 확률량을 확정한 resolved dict (로깅의 단일 출처)."""
        rng = np.random.default_rng(self.spec_seed)
        resolved: dict[str, Any] = {
            "mode": self.mode,
            "spec_seed": self.spec_seed,
            "sham": self.sham,
            "tag": self.tag,
        }
        if self.mode == "C1_camera":
            cams = {}
            for cam in self.cameras:
                # 카메라별 독립 δ. sham 이어도 같은 개수의 rng 소비 → sham/실섭동 창 산술 동일.
                dpos = rng.normal(0.0, C1_SIGMA_POS_M * self.scale, size=3)
                axis = rng.normal(0.0, 1.0, size=3)
                axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
                angle_deg = float(rng.normal(0.0, C1_SIGMA_ROT_DEG * self.scale))
                dfovy = float(rng.normal(0.0, C1_SIGMA_FOVY_DEG * self.scale))
                cams[cam] = {
                    "dpos": [float(v) for v in dpos],
                    "rot_axis": [float(v) for v in axis],
                    "rot_angle_deg": angle_deg,
                    "dfovy": dfovy,
                }
            resolved.update(
                {
                    "scale": self.scale,
                    "sigma": {
                        "pos_m": C1_SIGMA_POS_M,
                        "rot_deg": C1_SIGMA_ROT_DEG,
                        "fovy_deg": C1_SIGMA_FOVY_DEG,
                    },
                    "cameras": cams,
                }
            )
        elif self.mode == "G1_gripper_init":
            delta = rng.normal(0.0, self.sigma_xyz_m, size=3)
            resolved.update(
                {
                    "sigma_xyz_m": self.sigma_xyz_m,
                    "delta_xyz_m": [float(v) for v in delta],
                    "max_step_delta_m": G1_MAX_STEP_DELTA_M,
                }
            )
        else:  # P1/P2
            if self.direction is not None:
                dxy = np.asarray(self.direction, dtype=np.float64)
                dxy = dxy / max(float(np.linalg.norm(dxy)), 1e-12)
            else:
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                dxy = np.array([math.cos(theta), math.sin(theta)])
            resolved.update(
                {
                    "target": self.target,
                    "trigger_record": self.trigger_record,
                    "magnitude": self.magnitude,
                    "direction_xy": [float(dxy[0]), float(dxy[1])],
                }
            )
            if self.mode == "P2_force":
                resolved["duration_records"] = self.duration_records
        sha = hashlib.sha256(
            json.dumps(resolved, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        resolved["spec_sha12"] = sha
        return resolved


def _axis_angle_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """축-각 → wxyz quaternion."""
    half = math.radians(angle_deg) * 0.5
    q = np.zeros(4)
    q[0] = math.cos(half)
    q[1:] = np.asarray(axis, dtype=np.float64) * math.sin(half)
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz quaternion 곱 a∘b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _find_groot_env(env: Any) -> Any:
    """wrapper 체인에서 **실제** GrootRoboCasaEnv 인스턴스를 찾는다.

    주의: 인스턴스 hasattr 는 gymnasium Wrapper 의 __getattr__ 전달 때문에 wrapper 에서도
    True 라서 쓰면 안 됨 (private 접근 시 "accessing private attribute ... is prohibited").
    type-레벨 hasattr 로 메서드를 직접 정의/상속한 클래스만 매칭.
    """
    e = env
    for _ in range(24):
        if hasattr(type(e), "get_groot_observation"):
            return e
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    raise RuntimeError("GrootRoboCasaEnv 를 wrapper 체인에서 찾지 못함")


class Perturber:
    """에피소드 1개에 spec 1개를 적용하는 상태 기계 (fresh process 규약과 함께 사용)."""

    def __init__(self, env: Any, spec: PerturbSpec, *, n_action_steps: int):
        # http_feature_collect 가 sys.path 를 이미 구성한 뒤 생성됨 — 지연 import.
        from robocasa_event_labeler import find_robocasa_env

        self.env = env
        self.spec = spec
        self.nas = int(n_action_steps)
        self.genv = _find_groot_env(env)
        self.kenv = find_robocasa_env(env)
        if self.kenv is None:
            raise RuntimeError("robosuite kitchen env 를 찾지 못함 (find_robocasa_env)")
        self.resolved = spec.resolve()
        # 실행 기록
        self._applied_env_steps: list[int] = []
        self._env_step_offset = 0
        self._released: bool | None = None
        self._sham_skipped = False
        self._achieved_delta: list[float] | None = None
        self._cam_before_after: dict[str, Any] = {}
        self._scripted_terminated = False

    # -- 내부 헬퍼 ---------------------------------------------------------------
    def _refresh_obs(self) -> dict[str, Any]:
        """카메라 변조 후 reset 과 동일한 경로로 obs 재취득 (gymnasium_basic 재현)."""
        rsenv = self.genv.env
        raw = (
            rsenv.viewer._get_observations(force_update=True)
            if getattr(rsenv, "viewer_get_obs", False)
            else rsenv._get_observations(force_update=True)
        )
        basic = self.genv.get_basic_observation(raw)
        return self.genv.get_groot_observation(basic)

    def _eef_pos(self) -> np.ndarray:
        kenv = self.kenv
        sid = kenv.robots[0].eef_site_id["right"]
        return np.array(kenv.sim.data.site_xpos[sid], dtype=np.float64)

    # -- 에피소드 시작 (C1/G1) ----------------------------------------------------
    def on_episode_start(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.spec.mode == "C1_camera":
            return self._apply_camera(obs)
        if self.spec.mode == "G1_gripper_init":
            return self._apply_gripper_init(obs)
        return obs

    def _apply_camera(self, obs: dict[str, Any]) -> dict[str, Any]:
        sim = self.kenv.sim
        for cam, delta in self.resolved["cameras"].items():
            camid = sim.model.camera_name2id(cam)
            pos0 = np.array(sim.model.cam_pos[camid], dtype=np.float64)
            quat0 = np.array(sim.model.cam_quat[camid], dtype=np.float64)
            fovy0 = float(sim.model.cam_fovy[camid])
            if self.spec.sham:
                # 완전 no-op (write·obs 재취득 모두 skip): 재취득 렌더는 reset 첫 렌더와
                # 체계적으로 다름 (EGL 첫-프레임 warmup — S1 실측: sham 재취득 시 발산,
                # 재취득끼리는 결정적). 실섭동의 처치 정의 = shift+재취득이며 perturbed
                # fail/succ 양 클래스 공통이라 primary 대조에서 confound 아님.
                pos1, quat1, fovy1 = pos0, quat0, fovy0
            else:
                pos1 = pos0 + np.asarray(delta["dpos"])
                dq = _axis_angle_quat(np.asarray(delta["rot_axis"]), delta["rot_angle_deg"])
                quat1 = _quat_mul(dq, quat0)
                quat1 = quat1 / max(float(np.linalg.norm(quat1)), 1e-12)
                fovy1 = min(max(fovy0 + delta["dfovy"], 1.0), 179.0)
                sim.model.cam_pos[camid] = pos1
                sim.model.cam_quat[camid] = quat1
                sim.model.cam_fovy[camid] = fovy1
            self._cam_before_after[cam] = {
                "pos_before": [float(v) for v in pos0],
                "pos_after": [float(v) for v in pos1],
                "quat_before": [float(v) for v in quat0],
                "quat_after": [float(v) for v in quat1],
                "fovy_before": fovy0,
                "fovy_after": float(fovy1),
            }
        # sim.forward() 호출 금지 — cam_pos/quat/fovy 는 렌더 전용 모델 필드라 물리 재계산이
        # 불필요하고, forward() 는 warmstart 재계산으로 bitwise 를 깬다 (S1 sham_p1f 실측).
        if self.spec.sham:
            self._sham_skipped = True
            return obs
        return self._refresh_obs()

    def _base_rot(self) -> np.ndarray:
        """robot 루트(base) body 의 world 회전행렬 — OSC delta 는 base 프레임 입력."""
        kenv = self.kenv
        root = kenv.robots[0].robot_model.root_body
        bid = kenv.sim.model.body_name2id(root)
        return np.array(kenv.sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)

    def _apply_gripper_init(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.spec.sham:
            self._sham_skipped = True
            return obs
        # WAM 방식 closed-loop: OSC delta 는 base 프레임 + 관성으로 open-loop 추종이 ~20%
        # 에 그침 (S1 실측) → 매 chunk 잔여 오차(world)를 base 프레임으로 회전해 명령,
        # 도달(1cm) 또는 max_chunks 에서 중단. 전부 sim 상태의 함수라 결정적.
        delta = np.asarray(self.resolved["delta_xyz_m"], dtype=np.float64)
        eef0 = self._eef_pos()
        target = eef0 + delta
        max_chunks = 8
        n_chunks = 0
        for _ in range(max_chunks):
            err_w = target - self._eef_pos()
            if float(np.linalg.norm(err_w)) < 0.01:
                break
            step_w = err_w / self.nas
            n = float(np.linalg.norm(step_w))
            if n > G1_MAX_STEP_DELTA_M:
                step_w = step_w * (G1_MAX_STEP_DELTA_M / n)
            step_b = self._base_rot().T @ step_w
            step_norm = np.clip(step_b / OSC_POS_SCALE_M, -1.0, 1.0)
            chunk = {
                "action.end_effector_position": np.tile(
                    step_norm.astype(np.float32), (self.nas, 1)
                ),
                "action.end_effector_rotation": np.zeros((self.nas, 3), dtype=np.float32),
                "action.gripper_close": np.zeros((self.nas, 1), dtype=np.float32),
                "action.base_motion": np.zeros((self.nas, 4), dtype=np.float32),
                "action.control_mode": np.zeros((self.nas, 1), dtype=np.float32),
            }
            obs, _reward, terminated, truncated, _info = self.env.step(chunk)
            n_chunks += 1
            if terminated or truncated:
                self._scripted_terminated = True
                raise RuntimeError(
                    "G1 scripted 구간에서 episode 종료 — spec 재검토 필요 "
                    f"(terminated={terminated} truncated={truncated})"
                )
        eef1 = self._eef_pos()
        self._achieved_delta = [float(v) for v in (eef1 - eef0)]
        self._env_step_offset = n_chunks * self.nas
        self._applied_env_steps = list(range(self._env_step_offset))
        return obs

    # -- 루프 (P1/P2) --------------------------------------------------------------
    def maybe_apply(self, record_idx: int, official_action: dict[str, Any]) -> dict[str, Any]:
        if self.spec.mode == "P1_displace":
            self._maybe_p1(record_idx)
        elif self.spec.mode == "P2_force":
            self._maybe_p2(record_idx)
        return official_action

    def _maybe_p1(self, record_idx: int) -> None:
        if record_idx != self.resolved.get("trigger_record", -10):
            return
        kenv = self.kenv
        jname = kenv.objects[self.spec.target].joints[0]
        if self.spec.sham:
            if self.spec.sham_forward:
                # δ=0 + 원값 재기록 + forward — warmstart bitwise 실측 변형.
                q = np.array(kenv.sim.data.get_joint_qpos(jname), dtype=np.float64)
                kenv.sim.data.set_joint_qpos(jname, q)
                kenv.sim.forward()
            else:
                self._sham_skipped = True
            return
        q = np.array(kenv.sim.data.get_joint_qpos(jname), dtype=np.float64)
        dxy = np.asarray(self.resolved["direction_xy"])
        q[0:2] += self.spec.magnitude * dxy
        kenv.sim.data.set_joint_qpos(jname, q)
        kenv.sim.forward()
        self._applied_env_steps.append(record_idx * self.nas + self._env_step_offset)

    def _maybe_p2(self, record_idx: int) -> None:
        trig = self.resolved.get("trigger_record", -10)
        dur = self.resolved.get("duration_records", 0)
        kenv = self.kenv
        bid = kenv.obj_body_id[self.spec.target]
        if record_idx == trig:
            dxy = np.asarray(self.resolved["direction_xy"])
            force = 0.0 if self.spec.sham else self.spec.magnitude
            kenv.sim.data.xfrc_applied[bid, :3] = force * np.array([dxy[0], dxy[1], 0.0])
            self._released = False
            if not self.spec.sham:
                self._applied_env_steps.extend(
                    range(
                        trig * self.nas + self._env_step_offset,
                        (trig + dur) * self.nas + self._env_step_offset,
                    )
                )
        elif record_idx == trig + dur and self._released is False:
            kenv.sim.data.xfrc_applied[bid, :] = 0.0
            self._released = True

    # -- 저장 -----------------------------------------------------------------------
    def record_window(self) -> dict[str, Any]:
        mode = self.spec.mode
        if mode == "C1_camera":
            return {"perturb_record_window": [0, -1], "perturb_persistent": True}
        if mode == "G1_gripper_init":
            return {"perturb_record_window": [-1, -1], "perturb_persistent": False}
        if mode == "P1_displace":
            t = self.resolved["trigger_record"]
            return {"perturb_record_window": [t, t], "perturb_persistent": False}
        t = self.resolved["trigger_record"]
        return {
            "perturb_record_window": [t, t + self.resolved["duration_records"] - 1],
            "perturb_persistent": False,
        }

    def export(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "perturb_spec": self.resolved,
            "perturb_applied_env_steps": list(self._applied_env_steps),
            "perturb_env_step_offset": self._env_step_offset,
            **self.record_window(),
        }
        if self.spec.mode == "C1_camera":
            out["perturb_cameras"] = self._cam_before_after
        if self.spec.mode == "G1_gripper_init":
            out["perturb_achieved_delta"] = self._achieved_delta
        if self.spec.mode == "P2_force":
            out["perturb_released"] = self._released
        if self._sham_skipped:
            out["sham_skipped_sim_write"] = True
        return out
