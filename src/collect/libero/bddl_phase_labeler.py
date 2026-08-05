"""LIBERO BDDL sim-state phase labeler (oracle offline phase labels).

A libero task goal is a *conjunction* of atomic subgoals stored in
``domain.parsed_problem["goal_state"]`` (e.g. KITCHEN_SCENE4 ->
``[["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"],
   ["close", "white_cabinet_1_bottom_region"]]``). The domain exposes
``_eval_predicate(state) -> bool`` to evaluate ONE subgoal from current sim
state (the same call ``_check_success`` ANDs over).

This labeler turns that per-subgoal sim-state signal into:
  * a per-env-step **phase index** (monotone subtask progress), and
  * the data needed for **phase-local outcome** labels used by the offline
    per-phase success/failure separability analysis.

Phase definition
----------------
``phase_t = running_max(#subgoals satisfied at step t)``. For *forced-order*
tasks (put bowl in drawer -> then close it) the satisfied-count rises 0->1->2
in execution order, so the phase index is exactly "index of the next
unsatisfied subgoal" — a clean monotone subtask timeline. ``running_max``
enforces monotonicity so transient predicate toggling cannot regress a label.

Phase-local outcome (load-bearing: avoids leaking the global success/length
label). For phase ``p``, across episodes that *entered* p (``max_phase >= p``):
  * positive = **advanced past p** (``max_phase >= p + 1``)
  * negative = **stalled in p** (``max_phase == p``)
The per-phase feature is pooled over the steps an episode spent IN phase p,
so positives (transited through) and negatives (stuck) are compared at the
same subtask — not "did the whole episode eventually succeed" (= length
artifact).

The labeler does NO mujoco/torch work — it only calls the domain's predicate
evaluator — so it is unit-testable against a mock domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple


def find_domain(env: Any) -> Any:
    """Walk the ``.env`` wrapper chain to the BDDL domain that owns the goal.

    ``OffScreenRenderEnv`` wraps the problem domain as ``env.env``; the domain
    is the first object exposing both ``parsed_problem`` and ``_eval_predicate``.
    """
    cur = env
    for _ in range(16):  # bounded walk; chains are short
        if hasattr(cur, "parsed_problem") and hasattr(cur, "_eval_predicate"):
            return cur
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise AttributeError(
        "Could not locate a BDDL domain (with parsed_problem & _eval_predicate) "
        "in the env wrapper chain."
    )


def _subgoal_str(state: Sequence[Any]) -> str:
    return f"{state[0]}(" + ", ".join(str(a) for a in state[1:]) + ")"


@dataclass
class BddlPhaseLabeler:
    """Online per-episode phase labeler bound to a single libero env.

    Call :meth:`reset` at episode start, then :meth:`step` once per env-step
    (after ``env.step``) to record the current subgoal-satisfaction vector and
    get the monotone phase index for that step.
    """

    env: Any
    domain: Any = field(init=False)
    goal_state: List[List[Any]] = field(init=False)
    subgoal_strs: List[str] = field(init=False)
    # per-step records (filled by step())
    subgoal_timeline: List[Tuple[bool, ...]] = field(default_factory=list, init=False)
    phase_timeline: List[int] = field(default_factory=list, init=False)
    _running_max: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.domain = find_domain(self.env)
        self.goal_state = list(self.domain.parsed_problem["goal_state"])
        if not self.goal_state:
            raise ValueError("Task goal_state is empty; no subgoals to label.")
        self.subgoal_strs = [_subgoal_str(s) for s in self.goal_state]

    @property
    def n_subgoals(self) -> int:
        return len(self.goal_state)

    def reset(self) -> None:
        self.subgoal_timeline = []
        self.phase_timeline = []
        self._running_max = 0

    def eval_subgoals(self) -> Tuple[bool, ...]:
        """Evaluate each goal subgoal independently from current sim state."""
        return tuple(bool(self.domain._eval_predicate(s)) for s in self.goal_state)

    def step(self) -> int:
        """Record subgoal satisfaction for the current step; return phase index."""
        sat = self.eval_subgoals()
        count = int(sum(sat))
        self._running_max = max(self._running_max, count)
        self.subgoal_timeline.append(sat)
        self.phase_timeline.append(self._running_max)
        return self._running_max

    @property
    def max_phase(self) -> int:
        """Deepest phase reached this episode (0 .. n_subgoals)."""
        return self._running_max

    def phase_step_indices(self, p: int) -> List[int]:
        """Indices of env-steps spent in phase ``p`` (phase_timeline == p)."""
        return [i for i, ph in enumerate(self.phase_timeline) if ph == p]


def phase_local_labels(
    max_phases: Sequence[int], n_subgoals: int
) -> List[List[int]]:
    """Cross-episode phase-local outcome labels.

    Given each episode's ``max_phase``, return for each phase ``p`` in
    ``0 .. n_subgoals-1`` a list of labels over the episodes that ENTERED p
    (max_phase >= p): 1 = advanced past p (max_phase >= p+1), 0 = stalled at p.
    Episodes that never entered p are omitted (label = None placeholder dropped).

    Returns ``labels[p]`` = list of 0/1 aligned to ``[e for e in episodes if
    max_phases[e] >= p]`` (same filtering the caller must apply to features).
    """
    labels: List[List[int]] = []
    for p in range(n_subgoals):
        lab = [1 if m >= p + 1 else 0 for m in max_phases if m >= p]
        labels.append(lab)
    return labels


def entered_episode_mask(max_phases: Sequence[int], p: int) -> List[bool]:
    """Boolean mask over episodes that entered phase ``p`` (for feature filtering)."""
    return [m >= p for m in max_phases]
