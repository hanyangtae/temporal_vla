"""Event-anchored phase segmentation for RoboCasa atomic PnP (GR00T N1.5).

Reuses the pure ``PhaseSegmenter`` / ``EventPhaseLabeler`` core from the LIBERO event
labeler (shared segmentation logic) and adds a RoboCasa-specific ``RoboCasaEventDetector``
that reads ``robocasa.utils.object_utils`` (OU) sim predicates each env-step.

PnP canonical order: reach → grasp → transport → place → release.
  events (boundaries): grasp("obj") → place("obj" in target) → release(placed & gripper far)
  phases (gaps):       reach-to-object → transport → insert-settle → terminal

The shared pure core is now STATE-BASED / non-monotone: the phase is a pure function of the
CURRENT per-step predicates (grasp/place/release) this detector reports, so losing a held
grasp before placing reverts transport → reach-to-object, and a re-grasp returns to
transport. ``RoboCasaEventDetector`` is unchanged (it already reports current-step state);
the drop/re-grasp behaviour lives entirely in ``EventPhaseLabeler``/``PhaseSegmenter``.

Per-task place target (from each PnP ``_check_success``):
  PickPlaceCounterToCabinet : OU.obj_inside_of(env, "obj", env.cab)
  PickPlaceCounterToStove   : OU.check_obj_in_receptacle(env, "obj", "container", th=0.07)

OPT-IN proximity sub-phases (``make_robocasa_event_labeler(..., proximity_phases=True)``):
splits the two long approach phases by CURRENT-state proximity predicates (causal — no
future info — so the same labels are usable online for oracle gating at inference):

  "grasp" = ¬grasped ∧ near-object   (near-object = NOT OU.gripper_obj_far, the same
             predicate release already uses, negated; reach-to-object keeps ¬grasped ∧ far)
  "place" = grasped ∧ ¬placed ∧ near-target   (transport keeps grasped ∧ ¬placed ∧ ¬near)
             near-target: cabinet = OU.obj_inside_of margin th=NEAR_CABINET_TH (bounds
             expanded ±th vs strict 0.05); stove = xy-dist(obj, container) < NEAR_STOVE_TH
             (distance component of check_obj_in_receptacle, contact requirement dropped —
             the strict check requires contact so a relaxed th alone cannot fire pre-touch).

  6-phase timeline: reach-to-object / grasp / transport / place / insert-settle / terminal.

  "wrong-grasp" (7th label, failure detour): target NOT grasped ∧ some DISTRACTOR grasped
  (``RoboCasaEventDetector.wrong_grasped()`` = OU.check_obj_grasped over every
  ``env.objects`` key ≠ "obj", receptacles excluded, debounced like grasp). Overrides the
  reach-to-object→"grasp" refinement so distractor-holding steps never pool with genuine
  reach/grasp records in conceptor fits (observed failure: drop bread, pick up pear).
  Target-grasp has priority by construction — the base label is only reach-to-object when
  ¬grasped. After the distractor is dropped the phase falls back to reach/"grasp" per
  current state. Ranked reach+0.25 (< "grasp") so it never claims ``max_phase_label`` over
  genuine progress. Per-step ``wrong_grasp_timeline`` / rising-edge ``wrong_grasp_steps``
  are logged on ``ProximityEventPhaseLabeler`` for offline analysis.

Implemented as a POST-REFINEMENT over the unchanged pure core (base phase computed as
today, then reach-to-object→"grasp" when near-object, transport→"place" when near-target),
so event-order logic, ``grasp_steps``/``drop_steps``/``grasp_timeline`` and the default
(``proximity_phases=False``) 4-phase behaviour are untouched. Non-monotone consistency is
automatic: a drop during "place" reverts to reach-to-object or "grasp" per current state.

NOTE on activation alignment: GR00T inference fires every n_action_steps (~5) env-steps,
so each captured activation covers a 5-env-step block. The collection wiring must label
each activation with the phase at the env-step the activation was COMPUTED from (the obs
the policy acted on = block start), exactly like the LIBERO feature_phases handling.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

# reuse the pure, env-agnostic segmentation core from the LIBERO event labeler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libero"))
from event_phase_labeler import (  # noqa: E402
    INSERT_SETTLE,
    REACH_OBJECT,
    TRANSPORT,
    EventPhaseLabeler,
    PhaseSegmenter,  # noqa: F401  (re-exported for callers/tests)
    TaskEvent,
)

OBJ = "obj"  # RoboCasa PnP references the manipulated object as "obj"

# ── opt-in proximity sub-phase labels (between the base gap labels in pipeline order) ──
GRASP_PHASE = "grasp"   # ¬grasped ∧ near-object   (refines reach-to-object)
PLACE_PHASE = "place"   # grasped ∧ ¬placed ∧ near-target (refines transport)
WRONG_GRASP_PHASE = "wrong-grasp"  # ¬target-grasped ∧ distractor grasped (failure detour)

# ``env.objects`` keys that are task RECEPTACLES, not distractors: "container" (stove/sink
# PnP cfg name) and auto-generated ``try_to_place_in`` companions (``<name>_container``,
# kitchen.py ``_create_objects``). Excluded from wrong-grasp: closed-gripper contact with
# the receptacle during insertion would fire ``check_obj_grasped`` spuriously.
_RECEPTACLE_NAME = "container"
_RECEPTACLE_SUFFIX = "_container"

# near-target thresholds (metres). Strict place preds: cabinet obj_inside_of th=0.05,
# stove check_obj_in_receptacle th=0.07 (+contact). Relaxed values are heuristics — one
# runtime smoke in the robocasa container should confirm they fire during the approach.
NEAR_CABINET_TH = 0.20  # margin added to the cabinet interior-region bounds (±th per axis)
NEAR_STOVE_TH = 0.20    # xy distance obj→container (contact requirement dropped)


# ── per-task place predicates (OU lazy-imported; only runs in robocasa container) ──
def _place_cabinet(env: Any) -> bool:
    import robocasa.utils.object_utils as OU  # noqa: PLC0415

    return bool(OU.obj_inside_of(env, OBJ, env.cab))


def _place_stove(env: Any) -> bool:
    import robocasa.utils.object_utils as OU  # noqa: PLC0415

    return bool(OU.check_obj_in_receptacle(env, OBJ, "container", th=0.07))


# ── per-task NEAR-target predicates (relaxed place; opt-in proximity phases only) ──
def _near_target_cabinet(env: Any) -> bool:
    """obj bbox within the cabinet interior region expanded by ±NEAR_CABINET_TH per axis.

    ``OU.obj_inside_of`` natively supports a margin: its ``th`` is added to both ends of
    the interior-region bound checks (strict place uses the default 0.05), so a large th
    is exactly "near/at the opening" — no custom geometry needed.
    """
    import robocasa.utils.object_utils as OU  # noqa: PLC0415

    return bool(OU.obj_inside_of(env, OBJ, env.cab, th=NEAR_CABINET_TH))


def _near_target_stove(env: Any) -> bool:
    """xy distance obj→container < NEAR_STOVE_TH (relaxed check_obj_in_receptacle).

    The strict predicate is ``env.check_contact(obj, recep) AND xy-dist < th`` — the
    contact term can only fire at touchdown, so a relaxed th alone would still miss the
    hover/descend approach. The near version keeps only the distance component (which
    strictly subsumes the relaxed-contact variant).
    """
    import numpy as np  # noqa: PLC0415

    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[OBJ]])
    recep_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["container"]])
    return bool(np.linalg.norm(obj_pos[:2] - recep_pos[:2]) < NEAR_STOVE_TH)


# task tag -> relaxed near-target predicate (proximity_phases=True only)
TASK_NEAR_TARGET: Dict[str, Callable[[Any], bool]] = {
    "PickPlaceCounterToCabinet": _near_target_cabinet,
    "PickPlaceCounterToStove": _near_target_stove,
}


def lookup_near_target_pred(task_name: str) -> Callable[[Any], bool]:
    for tag, pred in TASK_NEAR_TARGET.items():
        if tag in task_name:
            return pred
    raise KeyError(
        f"No RoboCasa near-target predicate for task '{task_name}'. "
        f"known: {sorted(TASK_NEAR_TARGET)}"
    )


def _pnp_events() -> List[TaskEvent]:
    return [
        TaskEvent("grasp:obj", gap_before=REACH_OBJECT, detect="grasp", obj=OBJ),
        TaskEvent("place:obj", gap_before=TRANSPORT, detect="place", obj=OBJ),
        TaskEvent("release:obj", gap_before=INSERT_SETTLE, detect="release", obj=OBJ),
    ]


# task tag (substring of env/task name) -> (events, place_predicate)
TASK_EVENTS: Dict[str, Tuple[List[TaskEvent], Callable[[Any], bool]]] = {
    "PickPlaceCounterToCabinet": (_pnp_events(), _place_cabinet),
    "PickPlaceCounterToStove": (_pnp_events(), _place_stove),
}


def lookup_task_events(task_name: str) -> Tuple[List[TaskEvent], Callable[[Any], bool]]:
    for tag, spec in TASK_EVENTS.items():
        if tag in task_name:
            return spec
    raise KeyError(
        f"No RoboCasa event spec for task '{task_name}'. known: {sorted(TASK_EVENTS)}"
    )


def find_robocasa_env(env: Any) -> Any:
    """Walk the wrapper/vector chain to the robosuite kitchen env (OU-compatible).

    OU.* predicates need the kitchen env (``.sim`` + ``.objects``; cabinet tasks also
    ``.cab``). Walks ``.env`` / ``.unwrapped`` and a single-element ``.envs`` (vector env).
    """
    cur = env
    for _ in range(24):
        if hasattr(cur, "sim") and hasattr(cur, "objects"):
            return cur
        nxt = (
            getattr(cur, "env", None)
            or getattr(cur, "unwrapped", None)
        )
        if nxt is None and getattr(cur, "envs", None):
            nxt = cur.envs[0]
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise AttributeError("Could not locate the robosuite kitchen env (.sim + .objects).")


class RoboCasaEventDetector:
    """Reads OU.* sim predicates each env-step → currently-true event keys.

    grasp   = OU.check_obj_grasped(env, "obj"), debounced ``grasp_hold`` steps.
    place   = task place predicate (cabinet/stove).
    release = placed AND OU.gripper_obj_far(env, "obj").

    Proximity extras (only queried by the opt-in ``ProximityEventPhaseLabeler``; inert
    otherwise — ``active_events`` never reads them):
    near_obj      = NOT OU.gripper_obj_far(env, "obj")  (default th=0.25).
    near_target   = ``near_target_pred`` (relaxed per-task place predicate), False if unset.
    wrong_grasped = OU.check_obj_grasped(env, name) for ANY distractor name (env.objects
                    keys ≠ "obj", receptacles excluded), debounced ``grasp_hold`` steps on
                    its own streak counter. Call exactly once per env-step.
    """

    def __init__(
        self,
        env: Any,
        events: Sequence[TaskEvent],
        place_pred: Callable[[Any], bool],
        grasp_hold: int = 2,
        near_target_pred: Callable[[Any], bool] = None,
    ):
        self.env = find_robocasa_env(env)
        self.events = list(events)
        self.place_pred = place_pred
        self.grasp_hold = int(grasp_hold)
        self.near_target_pred = near_target_pred
        self._grasp_streak = 0
        self._wrong_grasp_streak = 0

    def _grasped(self) -> bool:
        try:
            import robocasa.utils.object_utils as OU  # noqa: PLC0415

            raw = bool(OU.check_obj_grasped(self.env, OBJ))
        except Exception:
            raw = False
        self._grasp_streak = self._grasp_streak + 1 if raw else 0
        return self._grasp_streak >= self.grasp_hold

    def _placed(self) -> bool:
        try:
            return bool(self.place_pred(self.env))
        except Exception:
            return False

    def _gripper_far(self) -> bool:
        try:
            import robocasa.utils.object_utils as OU  # noqa: PLC0415

            return bool(OU.gripper_obj_far(self.env, OBJ))
        except Exception:
            return False

    def near_obj(self) -> bool:
        """CURRENT-state: gripper within OU.gripper_obj_far's th (0.25 m) of the object.

        Deliberately its own try/except (not ``not _gripper_far()``): both predicates
        must default to False when OU/sim reads fail, and negating ``_gripper_far``'s
        False fallback would fabricate near=True.
        """
        try:
            import robocasa.utils.object_utils as OU  # noqa: PLC0415

            return not bool(OU.gripper_obj_far(self.env, OBJ))
        except Exception:
            return False

    def near_target(self) -> bool:
        """CURRENT-state relaxed place predicate (False when no pred configured)."""
        if self.near_target_pred is None:
            return False
        try:
            return bool(self.near_target_pred(self.env))
        except Exception:
            return False

    def _distractor_names(self) -> List[str]:
        """DISTRACTOR names = ``env.objects`` keys minus target/receptacles.

        Mirrors ``OU.check_obj_grasped``'s convention: its ``obj_name`` indexes
        ``env.objects`` (kitchen.py ``_create_objects`` dict, name → movable MJCF model),
        so every key here is a valid argument. Fixtures (cabinet/counter/stove) live in
        ``env.fixtures``, not ``env.objects`` — excluded for free. Read fresh per call:
        the dict is rebuilt each env reset (episode-dependent object set).
        """
        try:
            names = list(self.env.objects.keys())
        except Exception:
            return []
        return [
            n for n in names
            if n != OBJ and n != _RECEPTACLE_NAME and not n.endswith(_RECEPTACLE_SUFFIX)
        ]

    def wrong_grasped(self) -> bool:
        """CURRENT-state: some distractor grasped, debounced ``grasp_hold`` steps.

        Separate streak counter from the target-grasp debounce (the two can overlap
        during hand-offs and must not corrupt each other). Only the opt-in
        ``ProximityEventPhaseLabeler`` calls this (once per step); the default path
        never does, so its behaviour is byte-identical.
        """
        try:
            import robocasa.utils.object_utils as OU  # noqa: PLC0415

            raw = any(
                bool(OU.check_obj_grasped(self.env, name))
                for name in self._distractor_names()
            )
        except Exception:
            raw = False
        self._wrong_grasp_streak = self._wrong_grasp_streak + 1 if raw else 0
        return self._wrong_grasp_streak >= self.grasp_hold

    def active_events(self) -> Set[str]:
        grasped = self._grasped()      # call once/step to keep debounce streak correct
        placed = self._placed()
        far = self._gripper_far()
        out: Set[str] = set()
        for e in self.events:
            if e.detect == "grasp" and grasped:
                out.add(e.key)
            elif e.detect == "place" and placed:
                out.add(e.key)
            elif e.detect == "release" and placed and far:
                out.add(e.key)
        return out


# ── opt-in proximity refinement (post-refine over the unchanged pure core) ────────
class ProximityEventPhaseLabeler(EventPhaseLabeler):
    """EventPhaseLabeler + causal proximity sub-phases ("grasp" / "place" / "wrong-grasp").

    ``step()`` first runs the base state-based labeler unchanged (event_steps,
    grasp_steps/drop_steps/grasp_timeline semantics identical), then refines ONLY the
    phase label from CURRENT-state predicates read off the probe:

      base reach-to-object (¬grasped) ∧ probe.wrong_grasped() -> "wrong-grasp"
      base reach-to-object (¬grasped) ∧ probe.near_obj()      -> "grasp"
      base transport (grasped ∧ ¬placed) ∧ probe.near_target() -> "place"

    "wrong-grasp" (failure detour: holding a DISTRACTOR) overrides the near-object
    "grasp" refinement; it can only appear while the TARGET is not grasped because the
    base label is reach-to-object only then (target-grasp priority by construction).
    All predicates are current-state → fully causal (online-usable). Non-monotone
    reverts fall out automatically: dropping the distractor lands back on
    reach-to-object or "grasp" per current state. Probe must expose ``near_obj``/
    ``near_target`` (``RoboCasaEventDetector`` does); ``wrong_grasped`` is optional
    (missing → never fires, timeline stays all-False).

    Wrong-grasp bookkeeping (offline analysis; RAW predicate, not gated by base phase):
      * ``wrong_grasp_timeline`` -- per-step bool, 1:1 with ``grasp_timeline``.
      * ``wrong_grasp_steps``    -- rising-edge steps (mirrors ``grasp_steps``).
    """

    # sub-phase -> (base phase it refines, rank offset). "wrong-grasp" sits at reach+0.25
    # (< "grasp"'s reach+0.5): a failure detour, not progress — it must never claim
    # ``max_phase_label`` over a genuine "grasp"/transport step.
    _REFINED_AFTER = {
        GRASP_PHASE: (REACH_OBJECT, 0.5),
        PLACE_PHASE: (TRANSPORT, 0.5),
        WRONG_GRASP_PHASE: (REACH_OBJECT, 0.25),
    }

    def __post_init__(self) -> None:
        super().__post_init__()
        self._reset_wrong_grasp()

    def _reset_wrong_grasp(self) -> None:
        self.wrong_grasp_timeline: List[bool] = []
        self.wrong_grasp_steps: List[int] = []
        self._prev_wrong_grasped = False

    def reset(self) -> None:
        super().reset()
        self._reset_wrong_grasp()

    def _wrong_grasped_now(self) -> bool:
        fn = getattr(self.probe, "wrong_grasped", None)
        return bool(fn()) if fn is not None else False

    def step(self) -> str:
        base = super().step()
        wrong = self._wrong_grasped_now()   # once per step (detector keeps a debounce streak)
        self.wrong_grasp_timeline.append(wrong)
        if wrong and not self._prev_wrong_grasped:
            self.wrong_grasp_steps.append(self._t)
        self._prev_wrong_grasped = wrong
        refined = self._refine(base, wrong)
        if refined != base:
            self.phase_timeline[-1] = refined
        return refined

    def _refine(self, base: str, wrong_grasped: bool) -> str:
        if base == REACH_OBJECT:
            if wrong_grasped:               # distractor in hand beats near-object "grasp"
                return WRONG_GRASP_PHASE
            if self.probe.near_obj():
                return GRASP_PHASE
        if base == TRANSPORT and self.probe.near_target():
            return PLACE_PHASE
        return base

    def phase_rank(self, phase: str) -> float:
        """Pipeline order incl. sub-phases: reach < wrong-grasp < grasp < transport < place < ..."""
        spec = self._REFINED_AFTER.get(phase)
        if spec is not None:
            anchor, offset = spec
            return self.segmenter.phase_rank(anchor) + offset
        return float(self.segmenter.phase_rank(phase))

    @property
    def max_phase_label(self) -> str:
        """FURTHEST phase reached (sub-phases ranked between their neighbours)."""
        if not self.phase_timeline:
            return self.segmenter.gap_labels[0]
        return max(self.phase_timeline, key=self.phase_rank)


def make_robocasa_event_labeler(
    env: Any,
    task_name: str,
    grasp_hold: int = 2,
    proximity_phases: bool = False,
) -> EventPhaseLabeler:
    events, place_pred = lookup_task_events(task_name)
    if not proximity_phases:  # default: EXACTLY the original 4-phase labeler
        detector = RoboCasaEventDetector(env, events, place_pred, grasp_hold=grasp_hold)
        return EventPhaseLabeler(probe=detector, events=events)
    detector = RoboCasaEventDetector(
        env,
        events,
        place_pred,
        grasp_hold=grasp_hold,
        near_target_pred=lookup_near_target_pred(task_name),
    )
    return ProximityEventPhaseLabeler(probe=detector, events=events)
