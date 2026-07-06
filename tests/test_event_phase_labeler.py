"""Unit tests for state-based phase segmentation (pure layer, no env).

The scheme is NON-MONOTONE: the phase is a pure function of the CURRENT per-step
active-event set, so losing a held grasp before placing reverts transport ->
reach-to-object and a re-grasp returns to transport (the drop/re-grasp behaviour).
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "scripts", "safe", "groot_n16", "libero"
    ),
)

from event_phase_labeler import (  # noqa: E402
    INSERT_SETTLE,
    REACH_DOOR,
    REACH_OBJECT,
    TERMINAL,
    TRANSPORT,
    EventPhaseLabeler,
    PhaseSegmenter,
    TaskEvent,
    lookup_task_events,
)

EVENTS = [
    TaskEvent("grasp", REACH_OBJECT, "grasp", obj="o"),
    TaskEvent("place", TRANSPORT, "in", obj="o", region="r"),
    TaskEvent("release", INSERT_SETTLE, "release", obj="o", region="r"),
    TaskEvent("close", REACH_DOOR, "close", region="r"),
]


class ScriptedProbe:
    """Reports a scripted set of active event keys per step (last entry repeats)."""

    def __init__(self, script):
        self.script = script
        self.t = -1

    def active_events(self):
        self.t += 1
        return set(self.script[min(self.t, len(self.script) - 1)])


def _run(labeler, n):
    labeler.reset()
    for _ in range(n):
        labeler.step()


# ── PhaseSegmenter (pure, state-based) ────────────────────────────────────────
def test_segmenter_state_based_labels():
    seg = PhaseSegmenter(EVENTS)
    # phase = gap_before of the event AFTER the highest currently-active event.
    assert seg.phase_for(set()) == REACH_OBJECT
    assert seg.phase_for({"grasp"}) == TRANSPORT
    assert seg.phase_for({"grasp", "place"}) == INSERT_SETTLE
    assert seg.phase_for({"grasp", "place", "release"}) == REACH_DOOR
    assert seg.phase_for({"grasp", "place", "release", "close"}) == TERMINAL
    # highest_active_index tracks the furthest currently-true milestone (-1 = none).
    assert seg.highest_active_index(set()) == -1
    assert seg.highest_active_index({"grasp"}) == 0
    assert seg.highest_active_index({"grasp", "place", "release", "close"}) == 3


def test_segmenter_is_non_monotone_no_prefix_requirement():
    # State-based: a later predicate true WITHOUT its predecessor reads as that later
    # stage (no forced-order prefix). placed-without-grasp -> insert-settle.
    seg = PhaseSegmenter(EVENTS)
    assert seg.phase_for({"place"}) == INSERT_SETTLE
    assert seg.phase_for({"place", "release"}) == REACH_DOOR
    # Losing a milestone reverts the phase (the reason the scheme is non-monotone).
    assert seg.phase_for({"grasp"}) == TRANSPORT
    assert seg.phase_for(set()) == REACH_OBJECT  # grasp lost -> back to reach


def test_segmenter_phase_rank_orders_pipeline():
    seg = PhaseSegmenter(EVENTS)
    ranks = [seg.phase_rank(p) for p in (REACH_OBJECT, TRANSPORT, INSERT_SETTLE, REACH_DOOR, TERMINAL)]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)  # strictly increasing
    assert seg.phase_rank(TERMINAL) == seg.n_events
    assert seg.phase_rank("not-a-phase") == -1


def test_segmenter_rejects_empty():
    with pytest.raises(ValueError):
        PhaseSegmenter([])


# ── EventPhaseLabeler (state-based, non-monotone) ─────────────────────────────
def test_happy_path_phase_timeline_and_event_steps():
    script = [
        set(), set(),               # 0,1: reach
        {"grasp"},                  # 2: grasp -> transport
        {"grasp", "place"},         # 3: place -> insert-settle
        {"place", "release"},       # 4: release -> reach-to-door
        {"release", "close"},       # 5: close -> terminal
        {"close"},                  # 6: terminal
    ]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=EVENTS)
    _run(lab, len(script))
    assert lab.phase_timeline == [
        REACH_OBJECT, REACH_OBJECT, TRANSPORT, INSERT_SETTLE, REACH_DOOR, TERMINAL, TERMINAL
    ]
    assert lab.event_steps == {"grasp": 2, "place": 3, "release": 4, "close": 5}
    assert lab.progress == 4
    assert lab.max_phase_label == TERMINAL
    # single clean grasp, no drop
    assert lab.grasp_steps == [2]
    assert lab.grasp_count == 1
    assert lab.drop_steps == []


def test_drop_reverts_transport_to_reach():
    # grasp fires once then the probe stops reporting it (a drop before placing).
    # Old running-max scheme kept "transport"; state-based reverts to reach-to-object.
    script = [{"grasp"}, set(), set()]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=EVENTS)
    _run(lab, 3)
    assert lab.phase_timeline == [TRANSPORT, REACH_OBJECT, REACH_OBJECT]
    assert lab.grasp_steps == [0]
    assert lab.drop_steps == [1]           # grasp lost at step 1, not placed -> a drop
    assert lab.grasp_timeline == [True, False, False]
    assert lab.max_phase_label == TRANSPORT  # furthest reached, even after the revert


def test_placed_without_grasp_reads_as_insert_settle():
    # Non-monotone / no prefix requirement: a placed predicate (no grasp) -> insert-settle.
    script = [set(), {"place"}, {"place"}, {"place"}]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=EVENTS)
    _run(lab, 4)
    assert lab.phase_timeline == [REACH_OBJECT, INSERT_SETTLE, INSERT_SETTLE, INSERT_SETTLE]
    assert lab.progress == 2               # highest active (place=idx1) + 1
    assert lab.grasp_steps == [] and lab.drop_steps == []  # grasp never fired -> no drop


def test_drop_then_reach_then_regrasp_sequence():
    # Explicit spec case (3-event PnP, so release -> terminal): reach, reach, grasp,
    # grasp(transport), DROP, reach, grasp-again(transport), place(insert-settle),
    # release(terminal).
    pnp_events = EVENTS[:3]  # grasp, place(in), release — no actuate/close stage
    script = [
        set(),                     # 0 reach
        set(),                     # 1 reach
        {"grasp"},                 # 2 grasp -> transport
        {"grasp"},                 # 3 transport
        set(),                     # 4 DROP -> revert to reach
        set(),                     # 5 reach
        {"grasp"},                 # 6 re-grasp -> transport again
        {"grasp", "place"},        # 7 place -> insert-settle (placed priority over grasp)
        {"place", "release"},      # 8 release -> terminal
    ]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=pnp_events)
    _run(lab, len(script))
    assert lab.phase_timeline == [
        REACH_OBJECT, REACH_OBJECT,          # 0,1
        TRANSPORT, TRANSPORT,                # 2,3
        REACH_OBJECT, REACH_OBJECT,          # 4 (drop revert), 5
        TRANSPORT,                           # 6 re-grasp
        INSERT_SETTLE,                       # 7
        TERMINAL,                            # 8
    ]
    # revert on drop, return to transport on re-grasp
    assert lab.phase_timeline[3] == TRANSPORT and lab.phase_timeline[4] == REACH_OBJECT
    assert lab.phase_timeline[6] == TRANSPORT
    # grasp fires twice; the single drop is logged
    assert lab.grasp_steps == [2, 6]
    assert lab.grasp_count == 2
    assert lab.drop_steps == [4]
    # first-fire timings preserved (backward compatible)
    assert lab.event_steps == {"grasp": 2, "place": 7, "release": 8}
    assert lab.grasp_timeline == [
        False, False, True, True, False, False, True, True, False
    ]
    assert lab.max_phase_label == TERMINAL


def test_release_after_place_is_not_a_drop():
    # A grasp lost WHILE placed is a normal release, not a drop.
    script = [{"grasp"}, {"grasp", "place"}, {"place"}, {"place", "release"}]
    lab = EventPhaseLabeler(probe=ScriptedProbe(script), events=EVENTS)
    _run(lab, len(script))
    assert lab.drop_steps == []            # grasp lost at step 2 but object was placed
    assert lab.phase_timeline == [TRANSPORT, INSERT_SETTLE, INSERT_SETTLE, REACH_DOOR]


# ── task registry / lookup ────────────────────────────────────────────────────
def test_lookup_forced_order_tasks():
    ev4 = lookup_task_events(
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    )
    assert [e.detect for e in ev4] == ["grasp", "in", "release", "close"]
    assert ev4[0].obj == "akita_black_bowl_1"
    ev6 = lookup_task_events(
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
    )
    assert ev6[-1].region == "microwave_1"


def test_lookup_unknown_raises():
    with pytest.raises(KeyError):
        lookup_task_events("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup")
