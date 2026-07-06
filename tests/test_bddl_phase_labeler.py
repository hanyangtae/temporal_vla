"""Unit tests for the LIBERO BDDL phase labeler (no mujoco/GPU).

Uses a mock domain that exposes the same surface the labeler depends on:
``parsed_problem["goal_state"]`` and ``_eval_predicate(state)``.
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

from bddl_phase_labeler import (  # noqa: E402
    BddlPhaseLabeler,
    entered_episode_mask,
    find_domain,
    phase_local_labels,
)


class MockDomain:
    """Mock libero domain. ``script`` is a per-step dict {subgoal_index: bool}."""

    def __init__(self, goal_state, script):
        self.parsed_problem = {"goal_state": goal_state}
        self._script = script
        self._t = -1

    def advance(self):
        self._t += 1

    def _eval_predicate(self, state):
        idx = self._goal_index(state)
        step = self._script[min(self._t, len(self._script) - 1)]
        return step.get(idx, False)

    def _goal_index(self, state):
        return self.parsed_problem["goal_state"].index(state)


class Wrapper:
    """Mimics OffScreenRenderEnv exposing the domain as ``.env``."""

    def __init__(self, inner):
        self.env = inner


KITCHEN4_GOAL = [
    ["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"],
    ["close", "white_cabinet_1_bottom_region"],
]


def _run(labeler, domain, n_steps):
    labeler.reset()
    for _ in range(n_steps):
        domain.advance()
        labeler.step()


def test_find_domain_walks_wrapper_chain():
    dom = MockDomain(KITCHEN4_GOAL, [{}])
    env = Wrapper(Wrapper(dom))  # double-wrapped
    assert find_domain(env) is dom


def test_find_domain_raises_when_absent():
    with pytest.raises(AttributeError):
        find_domain(Wrapper(object()))


def test_forced_order_monotone_phase_timeline():
    # subgoal 0 (in) true from step 3; subgoal 1 (close) true from step 6.
    script = [
        {}, {}, {},                 # steps 0-2: nothing -> phase 0
        {0: True}, {0: True}, {0: True},   # steps 3-5: in -> phase 1
        {0: True, 1: True}, {0: True, 1: True},  # steps 6-7: both -> phase 2
    ]
    dom = MockDomain(KITCHEN4_GOAL, script)
    lab = BddlPhaseLabeler(Wrapper(dom))
    assert lab.n_subgoals == 2
    _run(lab, dom, len(script))
    assert lab.phase_timeline == [0, 0, 0, 1, 1, 1, 2, 2]
    assert lab.max_phase == 2
    assert lab.phase_step_indices(0) == [0, 1, 2]
    assert lab.phase_step_indices(1) == [3, 4, 5]
    assert lab.phase_step_indices(2) == [6, 7]


def test_running_max_prevents_regression():
    # subgoal 0 toggles true then false; phase must not drop back to 0.
    script = [{}, {0: True}, {}, {0: True}]
    dom = MockDomain(KITCHEN4_GOAL, script)
    lab = BddlPhaseLabeler(Wrapper(dom))
    _run(lab, dom, len(script))
    assert lab.phase_timeline == [0, 1, 1, 1]


def test_stalled_episode_never_reaches_terminal():
    # Only first subgoal ever completes -> stalls in phase 1.
    script = [{}, {0: True}, {0: True}, {0: True}]
    dom = MockDomain(KITCHEN4_GOAL, script)
    lab = BddlPhaseLabeler(Wrapper(dom))
    _run(lab, dom, len(script))
    assert lab.max_phase == 1


def test_phase_local_labels_and_mask():
    # 5 episodes with max_phase: 0,1,1,2,2  (n_subgoals = 2)
    max_phases = [0, 1, 1, 2, 2]
    labels = phase_local_labels(max_phases, n_subgoals=2)
    # phase 0: all 5 entered (>=0). advanced (>=1) = [0,1,1,1,1]
    assert labels[0] == [0, 1, 1, 1, 1]
    # phase 1: episodes with max>=1 = [1,1,2,2]. advanced (>=2) = [0,0,1,1]
    assert labels[1] == [0, 0, 1, 1]
    # masks select the same episodes the feature pooling must keep
    assert entered_episode_mask(max_phases, 0) == [True] * 5
    assert entered_episode_mask(max_phases, 1) == [False, True, True, True, True]
    assert entered_episode_mask(max_phases, 2) == [False, False, False, True, True]


def test_empty_goal_state_rejected():
    dom = MockDomain([], [{}])
    with pytest.raises(ValueError):
        BddlPhaseLabeler(Wrapper(dom))
