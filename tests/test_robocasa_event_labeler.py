"""Unit tests for RoboCasa event-anchored labeler (pure layer; no robocasa/env)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root

from src.collect.libero.event_phase_labeler import (  # noqa: E402
    INSERT_SETTLE,
    REACH_OBJECT,
    TERMINAL,
    TRANSPORT,
    EventPhaseLabeler,
)
from src.collect.robocasa.event_labeler import (  # noqa: E402
    GRASP_PHASE,
    PLACE_PHASE,
    TASK_EVENTS,
    TASK_NEAR_TARGET,
    WRONG_GRASP_PHASE,
    ProximityEventPhaseLabeler,
    RoboCasaEventDetector,
    _near_target_cabinet,
    _near_target_stove,
    lookup_near_target_pred,
    lookup_task_events,
    make_robocasa_event_labeler,
)


class ScriptedProbe:
    def __init__(self, script):
        self.script = script
        self.t = -1

    def active_events(self):
        self.t += 1
        return set(self.script[min(self.t, len(self.script) - 1)])


def test_registry_pnp_tasks_distinct_place_pred():
    ev_c, pred_c = lookup_task_events(
        "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
    )
    ev_s, pred_s = lookup_task_events("PickPlaceCounterToStove")
    ev_m, pred_m = lookup_task_events(
        "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env"
    )
    assert [e.detect for e in ev_c] == ["grasp", "place", "release"]
    assert [e.gap_before for e in ev_c] == [REACH_OBJECT, TRANSPORT, INSERT_SETTLE]
    assert [e.detect for e in ev_m] == ["grasp", "place", "release"]
    assert callable(pred_c) and callable(pred_s) and callable(pred_m)
    # 태스크마다 place 술어가 달라야 한다 (cabinet / stove / coffee machine)
    assert len({id(pred_c), id(pred_s), id(pred_m)}) == 3
    assert set(TASK_EVENTS) == {
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "CoffeeSetupMug",
    }
    # 근접 서브페이즈 레지스트리도 같은 태스크 집합을 덮어야 한다
    assert set(TASK_NEAR_TARGET) == set(TASK_EVENTS)


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        lookup_task_events("PickPlaceSinkToCounter")


def test_pnp_phase_progression():
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    script = [
        set(),                                  # reach-to-object
        {"grasp:obj"},                          # transport
        {"grasp:obj", "place:obj"},             # release-settle
        {"place:obj", "release:obj"},           # terminal
        {"release:obj"},                        # terminal
    ]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=events)
    lab.reset()
    for _ in range(len(script)):
        lab.step()
    assert lab.phase_timeline == [
        REACH_OBJECT, TRANSPORT, INSERT_SETTLE, TERMINAL, TERMINAL
    ]
    assert lab.event_steps == {"grasp:obj": 1, "place:obj": 2, "release:obj": 3}


def test_stall_in_transport_when_place_never_fires():
    events, _ = lookup_task_events("PickPlaceCounterToStove")
    # grasps but never places → held in transport (current milestone = grasp, progress = 1).
    lab = EventPhaseLabeler(probe=ScriptedProbe([set(), {"grasp:obj"}, {"grasp:obj"}]), events=events)
    lab.reset()
    for _ in range(3):
        lab.step()
    assert lab.phase_timeline == [REACH_OBJECT, TRANSPORT, TRANSPORT]
    assert lab.progress == 1


def test_pnp_drop_reverts_then_regrasp():
    # State-based / non-monotone: a grasp dropped before placing reverts transport →
    # reach-to-object; a re-grasp returns to transport. Env-coupled OU predicates faked
    # via ScriptedProbe (the RoboCasaEventDetector already reports current-step state).
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    script = [
        set(),                                  # 0 reach
        {"grasp:obj"},                          # 1 grasp -> transport
        {"grasp:obj"},                          # 2 transport
        set(),                                  # 3 DROP -> revert to reach
        {"grasp:obj"},                          # 4 re-grasp -> transport
        {"grasp:obj", "place:obj"},             # 5 place -> insert-settle
        {"place:obj", "release:obj"},           # 6 release -> terminal
    ]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=events)
    lab.reset()
    for _ in range(len(script)):
        lab.step()
    assert lab.phase_timeline == [
        REACH_OBJECT, TRANSPORT, TRANSPORT, REACH_OBJECT, TRANSPORT, INSERT_SETTLE, TERMINAL
    ]
    assert lab.phase_timeline[2] == TRANSPORT and lab.phase_timeline[3] == REACH_OBJECT  # revert
    assert lab.phase_timeline[4] == TRANSPORT                                            # re-grasp
    assert lab.grasp_steps == [1, 4]
    assert lab.grasp_count == 2
    assert lab.drop_steps == [3]
    assert lab.grasp_timeline == [False, True, True, False, True, True, False]
    # first-fire timings preserved for backward-compatible event_steps reads
    assert lab.event_steps == {"grasp:obj": 1, "place:obj": 5, "release:obj": 6}
    assert lab.max_phase_label == TERMINAL


# ── opt-in proximity sub-phases ("grasp"/"place"; causal current-state refinement) ──
class ProximityScriptedProbe(ScriptedProbe):
    """ScriptedProbe + per-step near_obj/near_target flags (last entry repeats).

    ``active_events`` (called first inside ``step``) advances ``self.t``; the near
    scripts are read at that same index, so all three are aligned per step.
    """

    def __init__(self, script, near_obj_script, near_target_script):
        super().__init__(script)
        self.near_obj_script = near_obj_script
        self.near_target_script = near_target_script

    def near_obj(self):
        return bool(self.near_obj_script[min(self.t, len(self.near_obj_script) - 1)])

    def near_target(self):
        return bool(self.near_target_script[min(self.t, len(self.near_target_script) - 1)])


# one shared script: far reach → near(¬grasped) → carry far → carry near-target →
# placed(held) → released. Base 4-phase labels vs proximity 6-phase labels.
_PROX_SCRIPT = [
    set(),                        # 0 far, empty hand         -> reach-to-object
    set(),                        # 1 near obj, not grasped   -> "grasp" (proximity only)
    {"grasp:obj"},                # 2 grasped, far from target-> transport
    {"grasp:obj"},                # 3 grasped, near target    -> "place" (proximity only)
    {"grasp:obj", "place:obj"},   # 4 placed (still held)     -> insert-settle
    {"place:obj", "release:obj"},  # 5 released               -> terminal
]
_PROX_NEAR_OBJ = [False, True, True, False, False, False]
_PROX_NEAR_TGT = [False, False, False, True, True, True]


def test_proximity_full_six_phase_progression():
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = ProximityEventPhaseLabeler(
        probe=ProximityScriptedProbe(_PROX_SCRIPT, _PROX_NEAR_OBJ, _PROX_NEAR_TGT),
        events=events,
    )
    lab.reset()
    for _ in range(len(_PROX_SCRIPT)):
        lab.step()
    assert lab.phase_timeline == [
        REACH_OBJECT, GRASP_PHASE, TRANSPORT, PLACE_PHASE, INSERT_SETTLE, TERMINAL
    ]
    # base-labeler bookkeeping is untouched by the refinement
    assert lab.event_steps == {"grasp:obj": 2, "place:obj": 4, "release:obj": 5}
    assert lab.grasp_steps == [2] and lab.grasp_count == 1 and lab.drop_steps == []
    assert lab.grasp_timeline == [False, False, True, True, True, False]
    assert lab.max_phase_label == TERMINAL


def test_proximity_off_same_script_gives_old_four_phase_timeline():
    # Regression: proximity_phases=False semantics — the plain EventPhaseLabeler on the
    # SAME script/probe must produce the OLD 4-phase timeline (near flags ignored).
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = EventPhaseLabeler(
        probe=ProximityScriptedProbe(_PROX_SCRIPT, _PROX_NEAR_OBJ, _PROX_NEAR_TGT),
        events=events,
    )
    lab.reset()
    for _ in range(len(_PROX_SCRIPT)):
        lab.step()
    assert lab.phase_timeline == [
        REACH_OBJECT, REACH_OBJECT, TRANSPORT, TRANSPORT, INSERT_SETTLE, TERMINAL
    ]
    assert GRASP_PHASE not in lab.phase_timeline and PLACE_PHASE not in lab.phase_timeline


def test_proximity_drop_during_place_reverts_by_current_state():
    # Non-monotone consistency: a drop while in "place" goes back to "grasp" when the
    # gripper is still near the (fallen) object, then to reach-to-object once it moves
    # away, and a re-grasp returns to transport.
    events, _ = lookup_task_events("PickPlaceCounterToStove")
    script = [
        {"grasp:obj"},   # 0 grasped, near target -> "place"
        set(),           # 1 DROP, still near obj -> "grasp" (not bare reach)
        set(),           # 2 moved away           -> reach-to-object
        {"grasp:obj"},   # 3 re-grasp, far target -> transport
    ]
    near_obj = [False, True, False, False]
    near_tgt = [True, True, False, False]
    lab = ProximityEventPhaseLabeler(
        probe=ProximityScriptedProbe(script, near_obj, near_tgt), events=events
    )
    lab.reset()
    for _ in range(len(script)):
        lab.step()
    assert lab.phase_timeline == [PLACE_PHASE, GRASP_PHASE, REACH_OBJECT, TRANSPORT]
    assert lab.drop_steps == [1]           # drop bookkeeping unchanged by refinement
    assert lab.grasp_steps == [0, 3]
    assert lab.max_phase_label == PLACE_PHASE  # furthest reached (place > transport)


def test_proximity_phase_rank_orders_six_phases():
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = ProximityEventPhaseLabeler(
        probe=ProximityScriptedProbe([set()], [False], [False]), events=events
    )
    ranks = [
        lab.phase_rank(p)
        for p in (REACH_OBJECT, GRASP_PHASE, TRANSPORT, PLACE_PHASE, INSERT_SETTLE, TERMINAL)
    ]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)  # strictly increasing
    assert lab.phase_rank("not-a-phase") == -1


def test_make_labeler_proximity_flag_default_off():
    class _FakeKitchenEnv:  # passes find_robocasa_env's .sim/.objects duck check
        sim = object()
        objects = {}

    env = _FakeKitchenEnv()
    lab = make_robocasa_event_labeler(env, "PickPlaceCounterToCabinet")
    assert type(lab) is EventPhaseLabeler                 # default: EXACT old labeler
    assert lab.probe.near_target_pred is None             # detector stays inert
    lab6 = make_robocasa_event_labeler(env, "PickPlaceCounterToCabinet", proximity_phases=True)
    assert isinstance(lab6, ProximityEventPhaseLabeler)
    assert lab6.probe.near_target_pred is _near_target_cabinet
    labs = make_robocasa_event_labeler(env, "PickPlaceCounterToStove", proximity_phases=True)
    assert labs.probe.near_target_pred is _near_target_stove
    assert lookup_near_target_pred("PickPlaceCounterToStove") is _near_target_stove
    with pytest.raises(KeyError):
        lookup_near_target_pred("PickPlaceSinkToCounter")
    # env-coupled near predicates fail closed (no OU / no real sim -> False, no raise)
    assert lab6.probe.near_obj() is False
    assert lab6.probe.near_target() is False
    assert lab.probe.near_target() is False


# ── "wrong-grasp" (distractor grasped instead of target; failure detour) ──────────
class WrongGraspScriptedProbe(ProximityScriptedProbe):
    """ProximityScriptedProbe + per-step wrong_grasped flags (last entry repeats)."""

    def __init__(self, script, near_obj_script, near_target_script, wrong_grasp_script):
        super().__init__(script, near_obj_script, near_target_script)
        self.wrong_grasp_script = wrong_grasp_script

    def wrong_grasped(self):
        return bool(self.wrong_grasp_script[min(self.t, len(self.wrong_grasp_script) - 1)])


# observed failure motif: drop-free reach → grab the PEAR (distractor) → drop it →
# recover: reach → grasp bread → transport → place → insert-settle. t5 additionally
# scripts a wrong_grasped=True blip WHILE the target is held (closed-gripper bump into
# the distractor) to pin target-grasp priority: phase must stay transport.
_WG_SCRIPT = [
    set(),                        # 0 far, empty hand              -> reach-to-object
    set(),                        # 1 DISTRACTOR grasped (near obj)-> wrong-grasp (beats "grasp")
    set(),                        # 2 carrying distractor          -> wrong-grasp
    set(),                        # 3 dropped distractor, far      -> reach-to-object
    set(),                        # 4 near target obj, empty hand  -> "grasp"
    {"grasp:obj"},                # 5 TARGET grasped (+wrong blip) -> transport (priority)
    {"grasp:obj"},                # 6 carrying, near target        -> "place"
    {"grasp:obj", "place:obj"},   # 7 placed (still held)          -> insert-settle
]
_WG_NEAR_OBJ = [False, True, False, False, True, False, False, False]
_WG_NEAR_TGT = [False, False, False, False, False, False, True, True]
_WG_WRONG = [False, True, True, False, False, True, False, False]


def test_wrong_grasp_full_sequence_with_recovery():
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = ProximityEventPhaseLabeler(
        probe=WrongGraspScriptedProbe(_WG_SCRIPT, _WG_NEAR_OBJ, _WG_NEAR_TGT, _WG_WRONG),
        events=events,
    )
    lab.reset()
    for _ in range(len(_WG_SCRIPT)):
        lab.step()
    assert lab.phase_timeline == [
        REACH_OBJECT, WRONG_GRASP_PHASE, WRONG_GRASP_PHASE, REACH_OBJECT,
        GRASP_PHASE, TRANSPORT, PLACE_PHASE, INSERT_SETTLE,
    ]
    # per-step raw wrong-grasp record (parallel to grasp_timeline) + rising edges;
    # t5's blip is recorded but does NOT relabel the phase (target grasped -> transport)
    assert lab.wrong_grasp_timeline == _WG_WRONG
    assert lab.wrong_grasp_steps == [1, 5]
    assert lab.phase_timeline[5] == TRANSPORT
    # base-labeler bookkeeping untouched: wrong-grasp is NOT a target grasp/drop
    assert lab.grasp_timeline == [False, False, False, False, False, True, True, True]
    assert lab.grasp_steps == [5] and lab.drop_steps == []
    assert lab.event_steps == {"grasp:obj": 5, "place:obj": 7}
    assert lab.max_phase_label == INSERT_SETTLE


def test_wrong_grasp_default_path_same_script_old_four_phase():
    # Regression: proximity_phases=False semantics — plain EventPhaseLabeler on the SAME
    # script must give the OLD 4-phase labels; wrong/near flags ignored, no bookkeeping.
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = EventPhaseLabeler(
        probe=WrongGraspScriptedProbe(_WG_SCRIPT, _WG_NEAR_OBJ, _WG_NEAR_TGT, _WG_WRONG),
        events=events,
    )
    lab.reset()
    for _ in range(len(_WG_SCRIPT)):
        lab.step()
    assert lab.phase_timeline == [
        REACH_OBJECT, REACH_OBJECT, REACH_OBJECT, REACH_OBJECT,
        REACH_OBJECT, TRANSPORT, TRANSPORT, INSERT_SETTLE,
    ]
    assert WRONG_GRASP_PHASE not in lab.phase_timeline
    assert GRASP_PHASE not in lab.phase_timeline and PLACE_PHASE not in lab.phase_timeline
    assert not hasattr(lab, "wrong_grasp_timeline")


def test_wrong_grasp_rank_between_reach_and_grasp_and_never_max_over_progress():
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = ProximityEventPhaseLabeler(
        probe=WrongGraspScriptedProbe([set()], [False], [False], [False]), events=events
    )
    assert (
        lab.phase_rank(REACH_OBJECT)
        < lab.phase_rank(WRONG_GRASP_PHASE)
        < lab.phase_rank(GRASP_PHASE)
        < lab.phase_rank(TRANSPORT)
    )
    # wrong-grasp then genuine near-object grasp: max is the genuine progress phase
    lab2 = ProximityEventPhaseLabeler(
        probe=WrongGraspScriptedProbe(
            [set(), set()], [False, True], [False, False], [True, False]
        ),
        events=events,
    )
    lab2.reset()
    lab2.step(); lab2.step()
    assert lab2.phase_timeline == [WRONG_GRASP_PHASE, GRASP_PHASE]
    assert lab2.max_phase_label == GRASP_PHASE
    # ...but over bare reach-to-object the detour IS the furthest point reached
    lab3 = ProximityEventPhaseLabeler(
        probe=WrongGraspScriptedProbe([set(), set()], [False], [False], [True, False]),
        events=events,
    )
    lab3.reset()
    lab3.step(); lab3.step()
    assert lab3.max_phase_label == WRONG_GRASP_PHASE
    # reset clears wrong-grasp bookkeeping
    lab3.reset()
    assert lab3.wrong_grasp_timeline == [] and lab3.wrong_grasp_steps == []


def test_wrong_grasp_probe_without_predicate_is_inert():
    # older probes (no wrong_grasped attr) keep working: never fires, timeline all-False
    events, _ = lookup_task_events("PickPlaceCounterToCabinet")
    lab = ProximityEventPhaseLabeler(
        probe=ProximityScriptedProbe(_PROX_SCRIPT, _PROX_NEAR_OBJ, _PROX_NEAR_TGT),
        events=events,
    )
    lab.reset()
    for _ in range(len(_PROX_SCRIPT)):
        lab.step()
    assert WRONG_GRASP_PHASE not in lab.phase_timeline
    assert lab.wrong_grasp_timeline == [False] * len(_PROX_SCRIPT)
    assert lab.wrong_grasp_steps == []


class _FakeKitchenEnv:  # passes find_robocasa_env's .sim/.objects duck check
    sim = object()

    def __init__(self, objects):
        self.objects = objects


def test_detector_distractor_name_discovery():
    # env.objects keys (kitchen.py _create_objects) minus target "obj" minus receptacles
    # ("container" + auto try_to_place_in "<name>_container"); fixtures are never in
    # env.objects so they are excluded structurally.
    events, place = lookup_task_events("PickPlaceCounterToStove")
    det = RoboCasaEventDetector(
        _FakeKitchenEnv(
            {"obj": 0, "obj_container": 1, "container": 2, "distr_counter": 3, "distr_cab": 4}
        ),
        events,
        place,
    )
    assert det._distractor_names() == ["distr_counter", "distr_cab"]
    # fail-closed without a real sim/OU: False, no raise (same style as near_obj/near_target)
    assert det.wrong_grasped() is False


def test_detector_wrong_grasped_debounce_and_ou_convention(monkeypatch):
    # fake robocasa.utils.object_utils: check_obj_grasped(env, obj_name) with obj_name a
    # key of env.objects — the exact convention the detector must mirror.
    import sys as _sys
    import types as _types

    held = {"names": set()}
    queried = []

    fake_ou = _types.ModuleType("robocasa.utils.object_utils")

    def check_obj_grasped(env, obj_name, threshold=0.035):
        assert obj_name in env.objects  # convention: obj_name indexes env.objects
        queried.append(obj_name)
        return obj_name in held["names"]

    fake_ou.check_obj_grasped = check_obj_grasped
    fake_pkg = _types.ModuleType("robocasa")
    fake_utils = _types.ModuleType("robocasa.utils")
    fake_pkg.utils = fake_utils
    fake_utils.object_utils = fake_ou
    monkeypatch.setitem(_sys.modules, "robocasa", fake_pkg)
    monkeypatch.setitem(_sys.modules, "robocasa.utils", fake_utils)
    monkeypatch.setitem(_sys.modules, "robocasa.utils.object_utils", fake_ou)

    events, place = lookup_task_events("PickPlaceCounterToCabinet")
    det = RoboCasaEventDetector(
        _FakeKitchenEnv({"obj": 0, "container": 1, "distr_counter": 2}),
        events,
        place,
        grasp_hold=2,
    )
    assert det.wrong_grasped() is False          # nothing held
    held["names"] = {"distr_counter"}
    assert det.wrong_grasped() is False          # streak 1 < grasp_hold (contact flicker)
    assert det.wrong_grasped() is True           # streak 2 -> debounced fire
    held["names"] = set()
    assert det.wrong_grasped() is False          # release resets the streak
    held["names"] = {"distr_counter"}
    assert det.wrong_grasped() is False          # must re-earn the streak
    # only distractor names ever queried — never the target or the receptacle
    assert set(queried) == {"distr_counter"}
    # target-grasp debounce is a SEPARATE counter: untouched by wrong-grasp streaks
    assert det._grasp_streak == 0
