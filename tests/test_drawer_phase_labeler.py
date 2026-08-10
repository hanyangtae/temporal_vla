"""DrawerPhaseLabeler 유닛테스트 (pure layer; robocasa/env 불필요 — stub 사용)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root

from src.collect.robocasa.event_labeler import (  # noqa: E402
    DrawerPhaseLabeler,
    make_robocasa_event_labeler,
)


class StubModel:
    def body_name2id(self, name):
        if name == "d_door_handle_handle":
            return 0
        raise ValueError(name)

    def geom_name2id(self, name):
        raise ValueError(name)

    def site_name2id(self, name):
        raise ValueError(name)


class StubData:
    def __init__(self):
        self.body_xpos = [np.array([1.0, 0.0, 1.0])]  # handle 위치
        self.site_xpos = [np.array([0.0, 0.0, 1.0])]  # gripper 위치 (초기: 멀리)


class StubSim:
    def __init__(self):
        self.model = StubModel()
        self.data = StubData()


class StubDrawer:
    def __init__(self):
        self.q = 0.0

    def get_door_state(self, env):
        return {"drawer": env._stub_q}

    @property
    def handle_name(self):  # 실물 fixture 와 동일하게 property
        return "d_door_handle_handle"


class StubRobot:
    eef_site_id = {"right": 0}


class StubEnv:
    def __init__(self):
        self.sim = StubSim()
        self.objects = {}
        self.robots = [StubRobot()]
        self.drawer = StubDrawer()
        self._stub_q = 0.0

    def set_q(self, q):
        self._stub_q = q

    def set_gripper(self, pos):
        self.sim.data.site_xpos[0] = np.array(pos, dtype=float)


def make(env, behavior="open"):
    return DrawerPhaseLabeler(env, behavior=behavior, grasp_hold=2)


def test_dispatch_returns_drawer_labeler():
    env = StubEnv()
    lab = make_robocasa_event_labeler(env, "robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    assert isinstance(lab, DrawerPhaseLabeler)
    assert lab._behavior == "open"
    lab_c = make_robocasa_event_labeler(env, "CloseDrawer", proximity_phases=True)
    assert lab_c._behavior == "close"


def test_phase_progression_open():
    env = StubEnv()
    lab = make(env)
    assert lab.step() == "reach-to-handle"          # 멀고 q=0
    env.set_gripper([0.95, 0.0, 1.0])               # handle 5cm 이내
    assert lab.step() == "grasp-handle"
    env.set_q(0.02)                                  # Δq>ε 1스텝째 (debounce 미달)
    assert lab.step() == "grasp-handle"
    env.set_q(0.06)                                  # 2스텝 연속 증가 → pull
    ph = lab.step()
    assert ph == "pull"
    assert lab.event_steps.get("open-start") == 3    # q>0.05 첫 스텝
    env.set_q(0.96)
    assert lab.step() == "open-done"
    assert "open-done" in lab.event_steps
    assert lab.max_phase_label == "open-done"


def test_non_monotonic_and_pushback():
    env = StubEnv()
    lab = make(env)
    env.set_q(0.96); lab.step()
    assert lab.phase_timeline[-1] == "open-done"
    env.set_q(0.80); lab.step()                      # 되밀림 1스텝 (debounce 미달)
    env.set_q(0.60); ph = lab.step()                 # 2스텝 연속 감소
    assert ph == "push-back"
    assert lab.max_phase_label == "open-done"        # 최대 도달 phase 는 유지


def test_wrong_grasp_override_and_rising_edge():
    env = StubEnv()
    lab = make(env)
    script = iter([False, True, True, True, False])
    lab._wrong_grasped = lambda: next(script)        # OU 스텁 (streak 은 스크립트로 대체)
    lab.step()
    lab.step()                                       # wg True 1번째
    ph = lab.step()                                  # wg True 2번째
    assert ph == "wrong-grasp"
    assert lab.wrong_grasp_steps[0] == 1             # rising edge 시점
    env.set_q(0.0)
    assert lab.step() == "wrong-grasp"
    assert lab.step() == "reach-to-handle"           # 놓으면 복귀


def test_disengage_after_failed_grasp():
    """손잡이 붙었다가 못 열고 후퇴 → disengage. 다시 다가가면 reach 복귀."""
    env = StubEnv()
    lab = make(env)
    # handle at [1,0,1]. gripper x=0.9 → 거리 0.1 (near), 작을수록 멀다.
    env.set_gripper([0.30, 0.0, 1.0]); assert lab.step() == "reach-to-handle"  # 첫 접근
    env.set_gripper([0.90, 0.0, 1.0]); assert lab.step() == "grasp-handle"     # 근처 → engaged
    # 후퇴: 거리 증가(0.1→0.3→0.5→0.7) 3스텝 연속
    env.set_gripper([0.70, 0.0, 1.0]); lab.step()   # 멀어짐 streak 1 (아직 grasp)
    env.set_gripper([0.50, 0.0, 1.0]); assert lab.step() == "disengage"  # streak 2 → 후퇴
    env.set_gripper([0.30, 0.0, 1.0]); assert lab.step() == "disengage"  # 계속 멀어짐
    assert lab.event_steps.get("disengage:handle") is not None
    assert lab.max_phase_label == "grasp-handle"    # 최고 도달 유지
    # 재접근: 거리 감소(0.7→0.5→0.3) → reach 복귀
    env.set_gripper([0.50, 0.0, 1.0]); lab.step()   # 다가옴 streak 1 (아직 disengage)
    env.set_gripper([0.70, 0.0, 1.0]); assert lab.step() == "reach-to-handle"  # streak 2 → 재접근
    env.set_gripper([0.90, 0.0, 1.0]); assert lab.step() == "grasp-handle"     # 다시 근처


def test_close_behavior():
    env = StubEnv()
    env.set_q(1.0)
    lab = make(env, behavior="close")
    assert lab.step() == "reach-to-handle"           # close: q=1.0 은 미완
    env.set_q(0.5); lab.step()
    env.set_q(0.2); ph = lab.step()                  # 2스텝 연속 감소 = close 방향 pull
    assert ph == "pull"
    env.set_q(0.03)
    assert lab.step() == "open-done"                 # close 성공역 (q≤0.05)


def test_interface_contract_fields():
    env = StubEnv()
    lab = make(env)
    lab.step()
    assert isinstance(lab.phase_timeline, list)
    assert isinstance(lab.event_steps, dict)
    assert lab.event_order_keys == sorted(lab.event_steps, key=lab.event_steps.get)
    assert lab.grasp_steps == [] and lab.drop_steps == []
    assert len(lab.wrong_grasp_timeline) == 1
