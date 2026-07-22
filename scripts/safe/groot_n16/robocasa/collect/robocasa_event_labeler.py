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


# ── Drawer 계열 (OpenDrawer/CloseDrawer) 전용 라벨러 ─────────────────────────────
# PnP 술어(obj 파지·container 근접)가 안 맞아 별도 구현. EventPhaseLabeler 와 동일한
# 공개 인터페이스(step/reset/phase_timeline/event_steps/event_order_keys/max_phase_label
# + wrong_grasp_steps/timeline)를 제공해 collector·EnvStepGT 가 무수정으로 사용.
# 신호: q = min(drawer.get_door_state(env).values()) ∈ [0,1] (성공역: open ≥0.95/close ≤0.05),
#       gripper–handle 거리, Δq (debounce). phase 는 현재 상태의 순수 함수(비단조).
_DRAWER_PHASE_RANK = {
    "reach-to-handle": 0.0, "wrong-grasp": 0.25, "disengage": 0.5,
    "push-back": 0.75, "grasp-handle": 1.0, "pull": 2.0, "open-done": 3.0,
}
# fit/분석에서 제외 권장: open-done(=success 종결, 프레임 극소 [[terminal 동치]]).
DRAWER_TERMINAL_PHASES = frozenset({"open-done"})


class DrawerPhaseLabeler:
    NEAR_HANDLE_TH = 0.10   # m — gripper site ↔ handle body
    DQ_EPS = 1e-3           # 스텝당 열림비율 변화 (당김/되밀림 판정)
    DD_EPS = 3e-3           # 스텝당 gripper-handle 거리 변화 (재접근/후퇴 판정)
    HOLD = 2                # debounce (호출 단위: env-step 구동이면 env-step 스케일)

    def __init__(self, env: Any, behavior: str = "open", grasp_hold: int = 2):
        self._env = find_robocasa_env(env)
        self._behavior = behavior
        self.HOLD = int(grasp_hold)
        self._handle_body = None
        self.reset()

    # -- 내부 술어 --------------------------------------------------------------
    def _q(self) -> float:
        st = self._env.drawer.get_door_state(env=self._env)
        vals = list(st.values()) or [0.0]
        return min(vals) if self._behavior == "open" else max(vals)

    def _done(self, q: float) -> bool:
        return q >= 0.95 if self._behavior == "open" else q <= 0.05

    def _handle_pos(self):
        import numpy as np
        sim = self._env.sim
        if self._handle_body is None:
            name = self._env.drawer.handle_name  # 실물 fixture 에선 property
            if callable(name):
                name = name()
            # Drawer 는 광고된 "*_handle_handle" 이 모델에 없음(robocasa 불일치) —
            # 실물 body "*_handle_main" / geom "*_handle_g0" 로 fallback.
            cands = [name]
            if name.endswith("_handle"):
                cands.append(name[: -len("_handle")] + "_main")
                cands.append(name[: -len("_handle")] + "_g0")
            for nm in cands:
                for kind in ("body", "geom", "site"):
                    try:
                        getattr(sim.model, f"{kind}_name2id")(nm)
                        self._handle_body = (kind, nm)
                        break
                    except Exception:
                        continue
                if self._handle_body is not None:
                    break
            if self._handle_body is None:
                return None
        kind, name = self._handle_body
        arr = getattr(sim.data, f"{kind}_xpos")
        return np.array(arr[getattr(sim.model, f"{kind}_name2id")(name)])

    def _grip_handle_dist(self) -> float:
        import numpy as np
        hp = self._handle_pos()
        if hp is None:
            return float("inf")
        gp = self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id["right"]]
        return float(np.linalg.norm(gp - hp))

    def _near_handle(self) -> bool:
        return self._grip_handle_dist() < self.NEAR_HANDLE_TH

    def _wrong_grasped(self) -> bool:
        try:
            import robocasa.utils.object_utils as OU
            raw = any(OU.check_obj_grasped(self._env, n) for n in self._env.objects)
        except Exception:
            raw = False
        self._wg_streak = self._wg_streak + 1 if raw else 0
        return self._wg_streak >= self.HOLD

    # -- 인터페이스 --------------------------------------------------------------
    def reset(self) -> None:
        self.phase_timeline: List[str] = []
        self.event_steps: dict = {}
        self.wrong_grasp_steps: List[int] = []
        self.wrong_grasp_timeline: List[bool] = []
        self.grasp_steps: List[int] = []   # PnP 호환용 (drawer 에선 미사용)
        self.drop_steps: List[int] = []
        self._t = 0
        self._q_prev = None
        self._pull_streak = 0
        self._push_streak = 0
        self._wg_streak = 0
        self._wg_prev = False
        self._engaged = False   # 한 번이라도 grasp-handle/pull 에 든 적 있나 (후퇴 구분용)
        self._disengaged_prev = False
        self._d_prev = None      # 직전 gripper-handle 거리
        self._retreat_streak = 0
        self._approach_streak = 0
        self._diseng_state = False  # 후퇴 후 멀어짐(True) vs 재접근(False) 상태

    def step(self) -> str:
        q = self._q()
        dq = 0.0 if self._q_prev is None else q - self._q_prev
        opening = dq > self.DQ_EPS if self._behavior == "open" else dq < -self.DQ_EPS
        closing = dq < -self.DQ_EPS if self._behavior == "open" else dq > self.DQ_EPS
        self._pull_streak = self._pull_streak + 1 if opening else 0
        self._push_streak = self._push_streak + 1 if closing else 0
        wg = self._wrong_grasped()

        d = self._grip_handle_dist()
        dd = 0.0 if self._d_prev is None or d == float("inf") else d - self._d_prev
        self._retreat_streak = self._retreat_streak + 1 if dd > self.DD_EPS else 0
        self._approach_streak = self._approach_streak + 1 if dd < -self.DD_EPS else 0
        # 후퇴/재접근 상태 전이 (debounce): 멀어지면 disengage, 다시 다가오면 reach 로 복귀
        if self._retreat_streak >= self.HOLD:
            self._diseng_state = True
        elif self._approach_streak >= self.HOLD:
            self._diseng_state = False

        near = self._near_handle()
        if self._done(q):
            ph = "open-done"
        elif wg:
            ph = "wrong-grasp"
        elif self._pull_streak >= self.HOLD:
            ph = "pull"
        elif self._push_streak >= self.HOLD:
            ph = "push-back"
        elif near:
            ph = "grasp-handle"
        elif self._engaged and self._diseng_state:
            ph = "disengage"        # 파지 실패 후 손잡이에서 멀어지는 중
        else:
            ph = "reach-to-handle"  # 첫 접근 또는 후퇴 뒤 재접근

        # engaged: grasp-handle/pull 에 들면 set (이후 멀어짐은 reach 아니라 disengage)
        if ph in ("grasp-handle", "pull"):
            self._engaged = True
        self._d_prev = d if d != float("inf") else self._d_prev

        if wg and not self._wg_prev:
            self.wrong_grasp_steps.append(self._t)
        self.wrong_grasp_timeline.append(wg)
        self._wg_prev = wg
        if ph == "disengage" and not self._disengaged_prev:
            self.event_steps.setdefault("disengage:handle", self._t)
        self._disengaged_prev = ph == "disengage"
        thresh_start = 0.05 if self._behavior == "open" else 0.95
        started = q > thresh_start if self._behavior == "open" else q < thresh_start
        if started:
            self.event_steps.setdefault("open-start", self._t)
        if self._done(q):
            self.event_steps.setdefault("open-done", self._t)
        if ph == "grasp-handle":
            self.event_steps.setdefault("near:handle", self._t)
        self.phase_timeline.append(ph)
        self._q_prev = q
        self._t += 1
        return ph

    @property
    def max_phase_label(self) -> str:
        if not self.phase_timeline:
            return "reach-to-handle"
        return max(self.phase_timeline, key=lambda p: _DRAWER_PHASE_RANK.get(p, 0.0))

    @property
    def event_order_keys(self) -> List[str]:
        return sorted(self.event_steps, key=self.event_steps.get)


# ── Fridge 도어 계열 (OpenFridge/CloseFridge) 전용 라벨러 ────────────────────────
# Drawer 와 같은 구조(연속 관절 + 손잡이 근접)지만 세 가지가 다르다:
#  (1) 도어가 여러 개일 수 있다 — FridgeFrenchDoor 는 fridge 도어 2개이고 env 의
#      _check_success = fxtr.is_closed(compartment="fridge") 는 **모든** fridge 도어가
#      닫혀야 True. 따라서 q 는 "아직 남은 최악의 도어"로 잡고(close=max, open=min),
#      거리도 그 도어 기준으로 재 계산한다(한 짝 닫고 다른 짝으로 이동이 자연히 잡힘).
#      freezer 도어/서랍 관절은 성공 판정에 안 들어가므로 제외한다.
#  (2) Fridge fixture 에는 get_door_state 도 handle_name 도 없다(=None). 관절은
#      fxtr._fridge_door_joint_names, 손잡이는 geom "<door>_handle_main|_1|_2",
#      패널은 geom "<door>_main" 으로 직접 해석한다(실측 확인, 3 모델 공통).
#  (3) 닫기는 파지가 아니라 밀기 — grasp 대신 contact-door(근접) 를 쓴다.
# 임계 q: env 와 동일하게 close 성공 = 정규화 열림도 ≤ 0.005 (Fixture.is_closed th),
#         open 성공 = ≥ 0.90 (Fixture.is_open th).
_FRIDGE_PHASE_RANK = {
    "reach-to-door": 0.0, "wrong-grasp": 0.25, "disengage": 0.5,
    "swing-open": 0.75, "contact-door": 1.0, "push-close": 2.0, "close-done": 3.0,
}
# fit/분석에서 제외 권장: close-done(=success 종결, 프레임 극소 [[terminal 동치]]).
FRIDGE_TERMINAL_PHASES = frozenset({"close-done"})


class FridgePhaseLabeler:
    NEAR_DOOR_TH = 0.06     # m — gripper site ↔ 대상 도어 표면(손잡이/패널, surface distance)
    DQ_EPS = 1e-3           # 스텝당 열림비율 변화 (밀기/되열림 판정)
    DD_EPS = 3e-3           # 스텝당 gripper-door 거리 변화 (재접근/후퇴 판정)
    HOLD = 2                # debounce (호출 단위: env-step 구동이면 env-step 스케일)
    CLOSED_TH = 0.005       # Fixture.is_closed 기본 th 와 동일
    OPEN_TH = 0.90          # Fixture.is_open 기본 th 와 동일
    TARGET_MARGIN = 0.03    # m — 다짝 도어 대상 전환 히스테리시스
    # phase/event 라벨 (StandMixer 등 같은 구조의 fixture 태스크가 이름만 교체해 재사용)
    PH_REACH = "reach-to-door"
    PH_CONTACT = "contact-door"
    PH_PROGRESS = "push-close"
    PH_REGRESS = "swing-open"
    PH_DISENGAGE = "disengage"
    PH_WRONG = "wrong-grasp"
    PH_DONE = "close-done"
    EV_NEAR = "near:door"
    EV_START = "close-start"
    EV_DONE = "close-done"
    EV_DISENGAGE = "disengage:door"
    PHASE_RANK = _FRIDGE_PHASE_RANK

    def __init__(self, env: Any, behavior: str = "close", grasp_hold: int = 2):
        self._env = find_robocasa_env(env)
        self._behavior = behavior
        self.HOLD = int(grasp_hold)
        self._geom_cache: dict = {}
        self.reset()

    # -- 내부 술어 --------------------------------------------------------------
    def _fxtr(self):
        return getattr(self._env, "fxtr", None)

    def _door_joints(self) -> List[str]:
        fx = self._fxtr()
        if fx is None:
            return []
        # fridge 칸 도어만 (freezer/drawer 는 성공 판정 대상 아님)
        return list(getattr(fx, "_fridge_door_joint_names", []) or [])

    def _door_states(self) -> dict:
        fx = self._fxtr()
        joints = self._door_joints()
        if fx is None or not joints:
            return {}
        try:
            return dict(fx.get_joint_state(self._env, joints))
        except Exception:
            return {}

    def _q_and_target(self):
        """(q, 대상 도어 관절명) — **아직 안 끝났고 그리퍼에 가장 가까운** 도어.

        French door(2짝)에서 "가장 열린 짝"을 대상으로 잡으면, 로봇이 덜 열린 짝을 먼저
        닫는 동안 q 가 안 변해 push-close 가 아예 안 잡힌다(실측: ep0 에서 push 5 step).
        실제로 로봇이 지금 다루는 도어 = 손이 가 있는 도어이므로 거리로 고른다.
        """
        st = self._door_states()
        if not st:
            return (0.0 if self._behavior == "close" else 1.0), None
        remaining = [n for n, v in st.items() if not self._joint_done(v)]
        if not remaining:  # 전부 목표 달성 → 마지막으로 다룬 도어를 유지
            name = self._target_prev if self._target_prev in st else next(iter(st))
            return float(st[name]), name
        if len(remaining) == 1:
            name = remaining[0]
        else:
            name = min(remaining, key=self._grip_door_dist)
            # 히스테리시스: 대상이 바뀌면 Δq·Δd 연속성이 끊겨 push 검출이 죽는다.
            # 직전 대상이 아직 남아 있으면 새 후보가 TARGET_MARGIN 이상 가까울 때만 교체.
            prev = self._target_prev
            if prev in remaining and name != prev:
                if self._grip_door_dist(prev) - self._grip_door_dist(name) < self.TARGET_MARGIN:
                    name = prev
        return float(st[name]), name

    def _joint_done(self, v: float) -> bool:
        return v <= self.CLOSED_TH if self._behavior == "close" else v >= self.OPEN_TH

    def _done(self, _q: float) -> bool:
        """env 판정과 동일: **모든** fridge 도어가 목표 상태여야 성공."""
        st = self._door_states()
        if not st:
            return False
        return all(self._joint_done(v) for v in st.values())

    def _door_geoms(self, joint_name: str):
        """대상 도어의 손잡이/패널 geom (kind, name) 목록 — 최초 1회 해석 후 캐시."""
        if joint_name in self._geom_cache:
            return self._geom_cache[joint_name]
        base = joint_name[: -len("_joint")] if joint_name.endswith("_joint") else joint_name
        sim = self._env.sim
        found = []
        for suffix in ("_handle_main", "_handle_1", "_handle_2", "_main"):
            nm = base + suffix
            for kind in ("geom", "body", "site"):
                try:
                    getattr(sim.model, f"{kind}_name2id")(nm)
                    found.append((kind, nm))
                    break
                except Exception:
                    continue
        if not found:  # 최후 fallback: 도어 body 자체
            for kind in ("body", "geom"):
                try:
                    getattr(sim.model, f"{kind}_name2id")(base)
                    found.append((kind, base))
                    break
                except Exception:
                    continue
        self._geom_cache[joint_name] = found
        return found

    def _geom_surface_dist(self, gid: int, p) -> float:
        """점 p 에서 geom **표면**까지의 거리.

        중심거리를 쓰면 안 된다 — 도어 패널 geom 은 큰 박스(실측 half-extent 0.22×0.005×0.86,
        = 0.45m×1.7m 판)라 가장자리를 밀어도 중심에서 0.8m 넘게 나온다(초안에서 contact 미검출
        원인). 박스/원통/구는 정확히, 그 외(mesh 등)는 중심거리로 fallback.
        """
        import numpy as np
        sim = self._env.sim
        c = np.asarray(sim.data.geom_xpos[gid])
        gt = int(sim.model.geom_type[gid])
        s = np.asarray(sim.model.geom_size[gid], dtype=float)
        d = np.asarray(p) - c
        try:
            local = np.asarray(sim.data.geom_xmat[gid]).reshape(3, 3).T @ d
        except Exception:
            return float(np.linalg.norm(d))
        if gt == 6:      # box: size = half extents
            return float(np.linalg.norm(np.maximum(np.abs(local) - s[:3], 0.0)))
        if gt == 5:      # cylinder: size = (radius, half-length) along local z
            rad = max(float(np.linalg.norm(local[:2])) - s[0], 0.0)
            ax = max(abs(float(local[2])) - s[1], 0.0)
            return float((rad * rad + ax * ax) ** 0.5)
        if gt == 3:      # capsule: size = (radius, half-length) along local z
            z = min(max(float(local[2]), -s[1]), s[1])
            return max(float(np.linalg.norm(local - np.array([0.0, 0.0, z]))) - s[0], 0.0)
        if gt == 2:      # sphere
            return max(float(np.linalg.norm(local)) - s[0], 0.0)
        return float(np.linalg.norm(d))

    def _grip_door_dist(self, joint_name) -> float:
        import numpy as np
        if joint_name is None:
            return float("inf")
        sim = self._env.sim
        try:
            gp = np.asarray(sim.data.site_xpos[self._env.robots[0].eef_site_id["right"]])
        except Exception:
            return float("inf")
        best = float("inf")
        for kind, nm in self._door_geoms(joint_name):
            try:
                gid = getattr(sim.model, f"{kind}_name2id")(nm)
                if kind == "geom":
                    best = min(best, self._geom_surface_dist(gid, gp))
                else:  # body/site 는 크기 정보가 없어 중심거리
                    arr = getattr(sim.data, f"{kind}_xpos")
                    best = min(best, float(np.linalg.norm(gp - arr[gid])))
            except Exception:
                continue
        return best

    def _wrong_grasped(self) -> bool:
        """도어와 무관한 물체(door_obj/distractor)를 집고 있으면 wrong-grasp."""
        try:
            import robocasa.utils.object_utils as OU
            raw = any(OU.check_obj_grasped(self._env, n) for n in self._env.objects)
        except Exception:
            raw = False
        self._wg_streak = self._wg_streak + 1 if raw else 0
        return self._wg_streak >= self.HOLD

    # -- 인터페이스 --------------------------------------------------------------
    def reset(self) -> None:
        self.phase_timeline: List[str] = []
        self.event_steps: dict = {}
        self.wrong_grasp_steps: List[int] = []
        self.wrong_grasp_timeline: List[bool] = []
        self.grasp_steps: List[int] = []   # PnP 호환용 (fridge 에선 미사용)
        self.drop_steps: List[int] = []
        self.door_timeline: List[float] = []        # 대상 도어 정규화 열림도 (진단용)
        self.door_worst_timeline: List[float] = []  # 전체 도어 중 목표에서 가장 먼 값
        #   = 성공 판정을 지배하는 값. th 0.005 근처 near-miss 판별에 쓴다.
        self._t = 0
        self._q_prev = None
        self._target_prev = None
        self._push_streak = 0
        self._swing_streak = 0
        self._wg_streak = 0
        self._wg_prev = False
        self._engaged = False   # 한 번이라도 contact-door/push-close 였나 (후퇴 구분용)
        self._disengaged_prev = False
        self._d_prev = None
        self._retreat_streak = 0
        self._approach_streak = 0
        self._diseng_state = False

    def step(self) -> str:
        q, target = self._q_and_target()
        # 대상 도어가 바뀌면(한 짝 닫고 다음 짝) Δq·Δd 연속성이 깨지므로 리셋
        if target != self._target_prev:
            self._q_prev = None
            self._d_prev = None
            self._push_streak = self._swing_streak = 0
            self._retreat_streak = self._approach_streak = 0
        dq = 0.0 if self._q_prev is None else q - self._q_prev
        # progress = 목표 방향으로의 관절 변화
        progressing = dq < -self.DQ_EPS if self._behavior == "close" else dq > self.DQ_EPS
        regressing = dq > self.DQ_EPS if self._behavior == "close" else dq < -self.DQ_EPS
        self._push_streak = self._push_streak + 1 if progressing else 0
        self._swing_streak = self._swing_streak + 1 if regressing else 0
        wg = self._wrong_grasped()

        d = self._grip_door_dist(target)
        dd = 0.0 if self._d_prev is None or d == float("inf") else d - self._d_prev
        self._retreat_streak = self._retreat_streak + 1 if dd > self.DD_EPS else 0
        self._approach_streak = self._approach_streak + 1 if dd < -self.DD_EPS else 0
        if self._retreat_streak >= self.HOLD:
            self._diseng_state = True
        elif self._approach_streak >= self.HOLD:
            self._diseng_state = False

        near = d < self.NEAR_DOOR_TH
        if self._done(q):
            ph = self.PH_DONE
        elif wg:
            ph = self.PH_WRONG
        elif self._push_streak >= self.HOLD:
            ph = self.PH_PROGRESS
        elif self._swing_streak >= self.HOLD:
            ph = self.PH_REGRESS
        elif near:
            ph = self.PH_CONTACT
        elif self._engaged and self._diseng_state:
            ph = self.PH_DISENGAGE  # 접촉 실패 후 대상에서 멀어지는 중
        else:
            ph = self.PH_REACH      # 첫 접근 또는 후퇴 뒤 재접근

        if ph in (self.PH_CONTACT, self.PH_PROGRESS):
            self._engaged = True
        self._d_prev = d if d != float("inf") else self._d_prev

        if wg and not self._wg_prev:
            self.wrong_grasp_steps.append(self._t)
        self.wrong_grasp_timeline.append(wg)
        self._wg_prev = wg
        if ph == self.PH_DISENGAGE and not self._disengaged_prev:
            self.event_steps.setdefault(self.EV_DISENGAGE, self._t)
        self._disengaged_prev = ph == self.PH_DISENGAGE
        thresh_start = 0.95 if self._behavior == "close" else 0.05
        started = q < thresh_start if self._behavior == "close" else q > thresh_start
        if started:
            self.event_steps.setdefault(self.EV_START, self._t)
        if self._done(q):
            self.event_steps.setdefault(self.EV_DONE, self._t)
        if ph == self.PH_CONTACT:
            self.event_steps.setdefault(self.EV_NEAR, self._t)
        self.phase_timeline.append(ph)
        self.door_timeline.append(q)
        _st = self._door_states()
        if _st:
            self.door_worst_timeline.append(
                max(_st.values()) if self._behavior == "close" else min(_st.values())
            )
        self._q_prev = q
        self._target_prev = target
        self._t += 1
        return ph

    @property
    def max_phase_label(self) -> str:
        if not self.phase_timeline:
            return self.PH_REACH
        return max(self.phase_timeline, key=lambda p: self.PHASE_RANK.get(p, 0.0))

    @property
    def event_order_keys(self) -> List[str]:
        return sorted(self.event_steps, key=self.event_steps.get)


# ── StandMixer head 계열 (OpenStandMixerHead/CloseStandMixerHead) ──────────────────
# 구조는 Fridge 와 동일(연속 관절 1개 + 그 관절이 움직이는 body 표면 근접)이라
# FridgePhaseLabeler 를 상속하고 ① fixture 참조 ② 관절 ③ 대상 geom ④ 라벨 이름만 바꾼다.
# 차이점(실측):
#  - fixture 참조는 `env.stand_mixer` (도어 태스크의 `env.fxtr` 아님).
#  - 판정 관절은 `fxtr._joint_names["head"]` 하나뿐 — 다짝 문제 없음.
#    env 판정: OpenStandMixerHead = head > 0.99, Close = head < 0.01.
#  - 손잡이 geom 이름이 없다(head body 하위가 전부 g0..gN 익명) → head **body 소속 geom
#    전체**에 대해 표면거리를 재고 그 최소값을 쓴다.
#  - `env.objects` 가 비어 있어(distractor 없음) wrong-grasp 은 구조상 발생하지 않는다.
_MIXER_PHASE_RANK = {
    "reach-to-head": 0.0, "wrong-grasp": 0.25, "disengage": 0.5,
    "push-down": 0.75, "contact-head": 1.0, "lift-open": 2.0, "open-done": 3.0,
}
MIXER_TERMINAL_PHASES = frozenset({"open-done"})


class StandMixerPhaseLabeler(FridgePhaseLabeler):
    NEAR_DOOR_TH = 0.06
    OPEN_TH = 0.99          # OpenStandMixerHead._check_success 와 동일
    CLOSED_TH = 0.01        # CloseStandMixerHead._check_success 와 동일
    PH_REACH = "reach-to-head"
    PH_CONTACT = "contact-head"
    PH_PROGRESS = "lift-open"
    PH_REGRESS = "push-down"
    PH_DONE = "open-done"
    EV_NEAR = "near:head"
    EV_START = "open-start"
    EV_DONE = "open-done"
    EV_DISENGAGE = "disengage:head"
    PHASE_RANK = _MIXER_PHASE_RANK

    def __init__(self, env: Any, behavior: str = "open", grasp_hold: int = 2):
        super().__init__(env, behavior=behavior, grasp_hold=grasp_hold)

    def _fxtr(self):
        return getattr(self._env, "stand_mixer", None)

    def _door_joints(self) -> List[str]:
        fx = self._fxtr()
        if fx is None:
            return []
        jn = (getattr(fx, "_joint_names", {}) or {}).get("head")
        if not jn:
            return []
        try:
            if jn not in self._env.sim.model.joint_names:
                return []
        except Exception:
            pass
        return [jn]

    def _door_geoms(self, joint_name: str):
        """head body 소속 geom 전부 (이름 붙은 손잡이 geom 이 없음)."""
        if joint_name in self._geom_cache:
            return self._geom_cache[joint_name]
        sim = self._env.sim
        base = joint_name[: -len("_joint")] if joint_name.endswith("_joint") else joint_name
        found = []
        try:
            bid = sim.model.body_name2id(base)
            found = [
                ("geom", sim.model.geom_id2name(g))
                for g in range(sim.model.ngeom)
                if int(sim.model.geom_bodyid[g]) == bid and sim.model.geom_id2name(g)
            ]
        except Exception:
            found = []
        if not found:
            found = [("body", base)]
        self._geom_cache[joint_name] = found
        return found


def make_robocasa_event_labeler(
    env: Any,
    task_name: str,
    grasp_hold: int = 2,
    proximity_phases: bool = False,
) -> EventPhaseLabeler:
    if "StandMixerHead" in task_name:
        return StandMixerPhaseLabeler(
            env,
            behavior="close" if "CloseStandMixerHead" in task_name else "open",
            grasp_hold=grasp_hold,
        )
    # Fridge 도어 조작(OpenFridge/CloseFridge). OpenFridgeDrawer/CloseFridgeDrawer 는
    # 서랍 태스크이므로 제외하고 아래 Drawer 분기로 보낸다.
    if ("OpenFridge" in task_name or "CloseFridge" in task_name) and (
        "FridgeDrawer" not in task_name
    ):
        return FridgePhaseLabeler(
            env,
            behavior="close" if "CloseFridge" in task_name else "open",
            grasp_hold=grasp_hold,
        )
    if "Drawer" in task_name:
        return DrawerPhaseLabeler(
            env, behavior="close" if "Close" in task_name else "open", grasp_hold=grasp_hold
        )
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
