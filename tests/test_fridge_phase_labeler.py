"""FridgePhaseLabeler 유닛테스트 (pure layer; robocasa/env 불필요 — stub 사용)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root

from src.collect.robocasa.event_labeler import (  # noqa: E402
    FridgePhaseLabeler,
    make_robocasa_event_labeler,
)

# 실물 이름 규약 그대로 (probe 실측): 관절 "<door>_joint", geom "<door>_handle_main" 등
LEFT = "fr_fridge_left_door"
RIGHT = "fr_fridge_right_door"
GEOMS = {  # geom name -> 위치
    LEFT + "_handle_main": np.array([1.0, 0.0, 1.0]),
    RIGHT + "_handle_main": np.array([1.0, 2.0, 1.0]),
}


class StubModel:
    # 점 크기 박스(half-extent 0) → surface distance == 중심거리 (테스트 기대값 유지)
    geom_type = [6] * len(GEOMS)
    geom_size = [np.zeros(3) for _ in GEOMS]

    def geom_name2id(self, name):
        if name in GEOMS:
            return list(GEOMS).index(name)
        raise ValueError(name)

    def body_name2id(self, name):
        raise ValueError(name)

    def site_name2id(self, name):
        raise ValueError(name)


class StubData:
    def __init__(self):
        self.geom_xpos = [GEOMS[k] for k in GEOMS]
        self.geom_xmat = [np.eye(3).ravel() for _ in GEOMS]  # 회전 없음
        self.site_xpos = [np.array([0.0, 0.0, 1.0])]  # gripper (초기: 멀리)


class StubSim:
    def __init__(self):
        self.model = StubModel()
        self.data = StubData()


class StubFxtr:
    """Fridge fixture 대역 — get_door_state/handle_name 없음(실물과 동일)."""

    def __init__(self, joints):
        self._fridge_door_joint_names = list(joints)
        self._freezer_door_joint_names = ["fr_freezer_door_joint"]  # 무시되어야 함

    def get_joint_state(self, env, joint_names):
        return {j: env._stub_q[j] for j in joint_names}


class StubRobot:
    eef_site_id = {"right": 0}


class StubEnv:
    def __init__(self, joints=(LEFT + "_joint",)):
        self.sim = StubSim()
        self.objects = {}
        self.robots = [StubRobot()]
        self.fxtr = StubFxtr(joints)
        self._stub_q = {j: 0.95 for j in joints}

    def set_q(self, joint, q):
        self._stub_q[joint] = q

    def set_gripper(self, pos):
        self.sim.data.site_xpos[0] = np.array(pos, dtype=float)


def make(env, behavior="close"):
    return FridgePhaseLabeler(env, behavior=behavior, grasp_hold=2)


def test_dispatch_returns_fridge_labeler():
    env = StubEnv()
    lab = make_robocasa_event_labeler(
        env, "robocasa_panda_omron/CloseFridge_PandaOmron_Env"
    )
    assert isinstance(lab, FridgePhaseLabeler)
    assert lab._behavior == "close"
    lab2 = make_robocasa_event_labeler(env, "OpenFridge")
    assert isinstance(lab2, FridgePhaseLabeler) and lab2._behavior == "open"


def test_fridge_drawer_task_is_not_routed_here():
    """OpenFridgeDrawer 는 서랍 태스크 — Fridge 도어 라벨러가 가로채면 안 된다."""
    from src.collect.robocasa.event_labeler import DrawerPhaseLabeler

    class DrawerEnv(StubEnv):
        class _D:
            handle_name = "d_handle"

            def get_door_state(self, env):
                return {"d": 0.0}

        def __init__(self):
            super().__init__()
            self.drawer = DrawerEnv._D()

    lab = make_robocasa_event_labeler(DrawerEnv(), "OpenFridgeDrawer")
    assert isinstance(lab, DrawerPhaseLabeler)


def test_phase_progression_reach_contact_push_done():
    env = StubEnv()
    j = LEFT + "_joint"
    lab = make(env)
    assert lab.step() == "reach-to-door"          # 멀리, 정지
    env.set_gripper([0.95, 0.0, 1.0])             # 손잡이 근처(0.05m)
    lab.step()
    assert lab.step() == "contact-door"           # 근접, 관절 변화 없음
    for q in (0.80, 0.60, 0.40, 0.20):            # 밀어서 닫는 중
        env.set_q(j, q)
        ph = lab.step()
    assert ph == "push-close"
    env.set_q(j, 0.0)                             # 완전히 닫힘
    lab.step()
    assert lab.step() == "close-done"
    assert "close-start" in lab.event_steps and "close-done" in lab.event_steps
    assert lab.max_phase_label == "close-done"


def test_swing_open_is_distinct_from_push():
    """되열림(반대 방향)은 push-close 가 아니라 swing-open."""
    env = StubEnv()
    j = LEFT + "_joint"
    lab = make(env)
    env.set_gripper([0.95, 0.0, 1.0])
    for q in (0.90, 0.80, 0.70):
        env.set_q(j, q)
        lab.step()
    assert lab.phase_timeline[-1] == "push-close"
    for q in (0.75, 0.80, 0.85):
        env.set_q(j, q)
        ph = lab.step()
    assert ph == "swing-open"


def test_disengage_then_return_to_reach():
    """접촉 후 멀어지면 disengage, 다시 다가오면 reach-to-door 로 복귀."""
    env = StubEnv()
    lab = make(env)
    env.set_gripper([0.95, 0.0, 1.0])
    lab.step()
    assert lab.step() == "contact-door"
    for x in (0.80, 0.65, 0.50):                  # 후퇴
        env.set_gripper([x, 0.0, 1.0])
        ph = lab.step()
    assert ph == "disengage"
    assert "disengage:door" in lab.event_steps
    for x in (0.60, 0.70, 0.80):                  # 재접근 (아직 NEAR 밖)
        env.set_gripper([x, 0.0, 1.0])
        ph = lab.step()
    assert ph == "reach-to-door"


def test_french_door_target_is_nearest_unfinished_door():
    """대상 = 아직 안 닫힌 도어 중 그리퍼에 가장 가까운 쪽. 한 짝 닫으면 반대편으로 넘어간다."""
    env = StubEnv(joints=(LEFT + "_joint", RIGHT + "_joint"))
    lab = make(env)
    env.set_gripper([0.95, 0.0, 1.0])             # 왼쪽 손잡이 옆 (둘 다 열림)
    assert lab._q_and_target()[1] == LEFT + "_joint"
    env.set_q(LEFT + "_joint", 0.0)               # 왼쪽 닫힘 → 남은 건 오른쪽뿐
    q, target = lab._q_and_target()
    assert target == RIGHT + "_joint" and q == 0.95
    lab.step()
    assert lab.step() == "reach-to-door"          # 왼쪽에 붙어 있어도 대상은 먼 오른쪽
    env.set_gripper([1.0, 1.95, 1.0])             # 오른쪽 손잡이 옆
    lab.step()
    assert lab.step() == "contact-door"


def test_french_door_progress_on_the_less_open_door_is_detected():
    """덜 열린 짝을 먼저 닫아도 push-close 가 잡혀야 한다 (구 max-q 대상 선택의 버그)."""
    env = StubEnv(joints=(LEFT + "_joint", RIGHT + "_joint"))
    env.set_q(LEFT + "_joint", 0.90)              # 왼쪽이 '덜' 열림
    env.set_q(RIGHT + "_joint", 0.96)             # 오른쪽이 더 열림
    lab = make(env)
    env.set_gripper([0.95, 0.0, 1.0])             # 로봇은 왼쪽에 붙어 있다
    lab.step()
    for q in (0.70, 0.50, 0.30):                  # 왼쪽을 밀어 닫는 중
        env.set_q(LEFT + "_joint", q)
        ph = lab.step()
    assert ph == "push-close"
    assert lab.door_worst_timeline[-1] == 0.96    # 전체 진행도는 오른쪽이 지배


def test_success_threshold_matches_env_predicate():
    """close-done 은 env 의 is_closed(th=0.005) 와 동일 — 모든 fridge 도어가 닫혀야."""
    env = StubEnv(joints=(LEFT + "_joint", RIGHT + "_joint"))
    lab = make(env)
    env.set_q(LEFT + "_joint", 0.0)
    env.set_q(RIGHT + "_joint", 0.006)            # 살짝 열림 → 실패
    assert lab.step() != "close-done"
    env.set_q(RIGHT + "_joint", 0.004)
    assert lab.step() == "close-done"


def test_wrong_grasp_takes_priority_and_records_rising_edge():
    import types

    env = StubEnv()
    env.objects = {"door_obj": object()}
    lab = make(env)

    grasped = {"v": False}
    ou = types.ModuleType("robocasa.utils.object_utils")
    ou.check_obj_grasped = lambda _env, _name: grasped["v"]
    pkg = types.ModuleType("robocasa")
    utils = types.ModuleType("robocasa.utils")
    utils.object_utils = ou
    pkg.utils = utils
    saved = {k: sys.modules.get(k) for k in
             ("robocasa", "robocasa.utils", "robocasa.utils.object_utils")}
    sys.modules.update({"robocasa": pkg, "robocasa.utils": utils,
                        "robocasa.utils.object_utils": ou})
    try:
        lab.step()
        grasped["v"] = True
        lab.step()                                 # HOLD=2 → 아직
        assert lab.step() == "wrong-grasp"
        assert lab.wrong_grasp_steps == [2]
        grasped["v"] = False
        assert lab.step() != "wrong-grasp"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_geom_surface_distance_uses_box_extent_not_center():
    """도어 패널은 큰 박스 — 표면까지 거리를 써야 한다(중심거리면 가장자리 접촉을 놓침)."""
    env = StubEnv()
    lab = make(env)
    # 실측 SideBySide 패널 크기(half-extent)로 geom 0 을 바꿔치기
    env.sim.model.geom_type = [6, 6]
    env.sim.model.geom_size = [np.array([0.224, 0.005, 0.861]), np.zeros(3)]
    p = np.array([1.0, 0.0, 1.80])   # 패널 중심 바로 위 0.80m → 표면까지는 0.80-0.861<0
    assert lab._geom_surface_dist(0, p) == 0.0        # 박스 내부
    p = np.array([1.0, 0.03, 1.0])   # 판 두께(0.005) 밖 0.025m
    assert abs(lab._geom_surface_dist(0, p) - 0.025) < 1e-9
    # 중심거리였다면 패널 끝(z+0.86)에서 0.86 이 나와 근접 판정이 죽는다
    p_edge = np.array([1.0, 0.0, 1.85])
    assert lab._geom_surface_dist(0, p_edge) < 0.02
    assert np.linalg.norm(p_edge - GEOMS[LEFT + "_handle_main"]) > 0.8


def test_interface_contract_matches_collector_expectations():
    env = StubEnv()
    lab = make(env)
    for _ in range(3):
        lab.step()
    for attr in (
        "phase_timeline", "event_steps", "event_order_keys", "max_phase_label",
        "wrong_grasp_steps", "wrong_grasp_timeline", "grasp_steps", "drop_steps",
    ):
        assert hasattr(lab, attr), attr
    assert len(lab.phase_timeline) == 3
    assert len(lab.wrong_grasp_timeline) == 3
    lab.reset()
    assert lab.phase_timeline == [] and lab.event_steps == {}


# ── StandMixerPhaseLabeler (FridgePhaseLabeler 상속; 관절 1개 + head body geom) ──────
HEAD_J = "sm_head_joint"


class StubMixerModel(StubModel):
    joint_names = (HEAD_J,)
    ngeom = len(GEOMS)
    geom_bodyid = [0, 1]   # geom0 만 head body 소속

    def body_name2id(self, name):
        if name == "sm_head":
            return 0
        raise ValueError(name)

    def geom_id2name(self, gid):
        return list(GEOMS)[gid]


class StubMixerSim(StubSim):
    def __init__(self):
        super().__init__()
        self.model = StubMixerModel()


class StubMixerFxtr:
    _joint_names = {"head": HEAD_J, "bowl": "sm_bowl_joint"}

    def get_joint_state(self, env, joint_names):
        return {j: env._stub_q[j] for j in joint_names}


class StubMixerEnv(StubEnv):
    def __init__(self):
        super().__init__()
        self.sim = StubMixerSim()
        self.stand_mixer = StubMixerFxtr()
        self._stub_q = {HEAD_J: 0.0}

    def set_head(self, q):
        self._stub_q[HEAD_J] = q


def test_mixer_dispatch_and_thresholds():
    from src.collect.robocasa.event_labeler import StandMixerPhaseLabeler

    env = StubMixerEnv()
    lab = make_robocasa_event_labeler(
        env, "robocasa_panda_omron/OpenStandMixerHead_PandaOmron_Env"
    )
    assert isinstance(lab, StandMixerPhaseLabeler) and lab._behavior == "open"
    lab2 = make_robocasa_event_labeler(env, "CloseStandMixerHead")
    assert lab2._behavior == "close"
    # env 판정과 동일: open 성공 = head > 0.99
    assert lab._door_joints() == [HEAD_J]
    env.set_head(0.985)
    assert lab.step() != "open-done"
    env.set_head(0.995)
    assert lab.step() == "open-done"


def test_mixer_phase_names_and_progression():
    env = StubMixerEnv()
    lab = make_robocasa_event_labeler(env, "OpenStandMixerHead")
    assert lab.step() == "reach-to-head"
    env.set_gripper([0.98, 0.0, 1.0])       # head geom(=GEOMS[0]) 근처
    lab.step()
    assert lab.step() == "contact-head"
    for q in (0.2, 0.4, 0.6):               # 머리를 들어올리는 중
        env.set_head(q)
        ph = lab.step()
    assert ph == "lift-open"
    for q in (0.5, 0.3, 0.1):               # 다시 눌러 내림
        env.set_head(q)
        ph = lab.step()
    assert ph == "push-down"
    assert "open-start" in lab.event_steps and "near:head" in lab.event_steps
    assert lab.max_phase_label == "lift-open"
