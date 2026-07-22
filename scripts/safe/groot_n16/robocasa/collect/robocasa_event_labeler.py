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


def make_robocasa_event_labeler(
    env: Any,
    task_name: str,
    grasp_hold: int = 2,
    proximity_phases: bool = False,
) -> EventPhaseLabeler:
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
