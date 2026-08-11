"""State-based phase segmentation for LIBERO (supersedes coarse BDDL-subgoal phase).

Per ``HANDOFF_event_phase_segmentation.md``. Discrete sim events (grasp / in / release /
close ...) are PHASE BOUNDARIES; the moving segments *between* them (reach / transport /
reach-to-door) are the phases. Cross-episode alignment is by phase-LABEL (event type),
not timestep — which removes the length confound the coarse labeler could not control.

The phase is a pure function of the CURRENT manipulation state each step and is therefore
NON-MONOTONE (it may revert): a grasp that is dropped before placing sends transport ->
reach-to-object, and a fresh grasp returns to transport. This fixes the old running-max
prefix scheme, under which a grasp-then-drop looked identical to "still carrying" (the
label stayed "transport" until timeout).

순수 코어만 남긴 파일이다 (2026-08-07 S2 판정): LIBERO env 결합층(EventDetector·
make_event_labeler·bddl_phase_labeler)은 libero 축 폐기로 절제됨. 남은 것은 env 무관 —
  * ``PhaseSegmenter`` — PURE: CURRENT-step active-event set -> phase label (비단조).
  * ``EventPhaseLabeler`` — probe(Protocol) 주입식 소비자 API (reset/step/phase_timeline).
  * ``TASK_EVENTS`` — forced-order 이벤트 시퀀스 예시 (LIBERO scene 태그, 순수 데이터 —
    테스트 fixture 겸 신규 task 등록 예제).
robocasa 라벨러(src/collect/robocasa/event_labeler.py)가 이 코어를 재사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Set


# ── phase (gap) labels — the moving segments between events ───────────────────
REACH_OBJECT = "reach-to-object"   # start → grasp (empty-hand approach)
TRANSPORT = "transport"            # grasp → placed (carrying object)
INSERT_SETTLE = "insert-settle"    # placed(In/On) → release (let go)
REACH_DOOR = "reach-to-door"       # release → actuate (empty-hand approach to door/switch)
TERMINAL = "terminal"              # all events fired


@dataclass(frozen=True)
class TaskEvent:
    """One expected event in a task's canonical order.

    ``key`` is a per-task-unique id. ``gap_before`` is the phase label for the segment
    leading UP TO this event (= what the robot is doing while working toward it).
    ``detect`` + ``obj``/``region`` describe how ``EventDetector`` recognizes it.
    """

    key: str
    gap_before: str
    detect: str               # "grasp" | "in" | "on" | "release" | "close" | "turnon"
    obj: str = ""
    region: str = ""          # container/region (in/on/release) or articulation (close/turnon)


def _pnp_then_actuate(obj: str, region: str, articulation: str, actuate: str) -> List[TaskEvent]:
    """Canonical forced-order: reach→grasp→transport→place→release→reach-to-door→actuate."""
    return [
        TaskEvent(f"grasp:{obj}", gap_before=REACH_OBJECT, detect="grasp", obj=obj),
        TaskEvent(f"place:{obj}", gap_before=TRANSPORT, detect="in", obj=obj, region=region),
        TaskEvent(f"release:{obj}", gap_before=INSERT_SETTLE, detect="release", obj=obj, region=region),
        TaskEvent(f"{actuate}:{articulation}", gap_before=REACH_DOOR, detect=actuate, region=articulation),
    ]


# ── task registry (forced-order first; exchangeable/single deferred) ──────────
# keyed by a stable scene tag found as a substring of the libero task name.
TASK_EVENTS: Dict[str, List[TaskEvent]] = {
    # KITCHEN_SCENE4: put bowl in bottom drawer, then close it (drawer starts open).
    "KITCHEN_SCENE4": _pnp_then_actuate(
        obj="akita_black_bowl_1",
        region="white_cabinet_1_bottom_region",
        articulation="white_cabinet_1_bottom_region",
        actuate="close",
    ),
    # KITCHEN_SCENE6: put mug in microwave, then close it (microwave starts open).
    "KITCHEN_SCENE6": _pnp_then_actuate(
        obj="white_yellow_mug_1",
        region="microwave_heating_region",
        articulation="microwave_1",
        actuate="close",
    ),
}


def lookup_task_events(task_name: str) -> List[TaskEvent]:
    """Find the event spec whose scene tag is a substring of ``task_name``."""
    for tag, events in TASK_EVENTS.items():
        if tag in task_name:
            return events
    raise KeyError(
        f"No event-anchored spec for task '{task_name}'. "
        f"Known forced-order tags: {sorted(TASK_EVENTS)}"
    )


# ── PURE segmenter (STATE-BASED, non-monotone) ───────────────────────────────
class PhaseSegmenter:
    """CURRENT-step active-event set -> phase label (state-based, non-monotone).

    Handoff decision #2: this pure layer maps the current manipulation STATE to a phase and
    is unit-tested with a mock probe (no env, no torch). Unlike the earlier running-max
    *prefix* scheme, the phase is a pure function of which milestone events are true THIS
    step — the accumulated ever-fired set is NOT consulted:

        phase = ``gap_before`` of the event *after* the highest-order event currently active

      * no event active             -> the first gap label (reach-to-object)
      * highest active = event i     -> ``gap_before`` of event i+1
      * highest active = last event  -> ``TERMINAL``

    Reading the CURRENT set makes the phase revert when a milestone is lost: a grasp dropped
    before placing (grasp event no longer active) sends transport -> reach-to-object, and a
    fresh grasp returns it to transport. The prefix scheme could not express this — there a
    grasp-then-drop stayed "transport" until timeout.

    Note this drops the forced-order *prefix* requirement: a later predicate that is true
    without its predecessor (e.g. ``placed`` reported without ``grasped``) reads as that
    later stage (insert-settle), not the first gap. The mapping trusts the detector's
    per-step predicates; ordering robustness is the detector's responsibility.
    """

    def __init__(self, events: Sequence[TaskEvent]):
        if not events:
            raise ValueError("PhaseSegmenter needs a non-empty event list")
        self.events: List[TaskEvent] = list(events)
        self.gap_labels: List[str] = [e.gap_before for e in self.events]

    def highest_active_index(self, active: Set[str]) -> int:
        """Index of the highest-order event whose key is currently active (-1 if none)."""
        hi = -1
        for i, e in enumerate(self.events):
            if e.key in active:
                hi = i
        return hi

    def phase_for(self, active: Set[str]) -> str:
        """State-based phase for the events true THIS step (non-monotone; may revert)."""
        hi = self.highest_active_index(active)
        if hi < 0:
            return self.gap_labels[0]
        if hi >= len(self.events) - 1:
            return TERMINAL
        return self.gap_labels[hi + 1]

    def phase_rank(self, phase: str) -> int:
        """Pipeline order of a phase label (``gap_labels[i]`` -> i; ``TERMINAL`` highest; else -1).

        Used to pick the FURTHEST phase an episode reached even after a non-monotone revert.
        """
        if phase == TERMINAL:
            return len(self.events)
        try:
            return self.gap_labels.index(phase)
        except ValueError:
            return -1

    @property
    def n_events(self) -> int:
        return len(self.events)


# ── event probe (injectable so the labeler is testable with a mock) ───────────
class EventProbe(Protocol):
    """Reports which task-event keys are currently true (this step)."""

    def active_events(self) -> Set[str]:
        ...


# ── consumer-facing labeler (mirrors BddlPhaseLabeler API) ────────────────────
@dataclass
class EventPhaseLabeler:
    """Per-step, state-based phase labeler (non-monotone; phases may revert on a drop).

    Consumer API preserved for ``scripts/eval/libero.py`` and the N1.5 collector
    (``http_feature_collect.py``): ``reset`` / ``step`` (returns AND appends the current
    phase) / ``phase_timeline`` / ``event_steps`` / ``event_order_keys`` / ``max_phase_label``.

    Recurrence logging for offline drop analysis (grasp can fire many times now):
      * ``event_steps``    -- key -> FIRST-fire step (backward compatible; place/release too).
      * ``grasp_steps``    -- every step a grasp rising-edge fired (list, may be > 1).
      * ``drop_steps``     -- every step a held grasp was lost while NOT placed (a drop).
      * ``grasp_count``    -- number of grasp fires (= ``len(grasp_steps)``).
      * ``grasp_timeline`` -- per-step ``grasped`` bool (1:1 with ``phase_timeline``) so future
                              data can be re-analyzed for drops without re-running the sim.
    """

    probe: EventProbe
    events: List[TaskEvent]
    segmenter: PhaseSegmenter = field(init=False)
    phase_timeline: List[str] = field(default_factory=list, init=False)
    event_steps: Dict[str, int] = field(default_factory=dict, init=False)   # key -> first-fire step
    grasp_timeline: List[bool] = field(default_factory=list, init=False)    # per-step grasped bool
    grasp_steps: List[int] = field(default_factory=list, init=False)        # grasp rising-edge steps
    drop_steps: List[int] = field(default_factory=list, init=False)         # grasp-lost-before-place steps
    _grasp_keys: Set[str] = field(default_factory=set, init=False)
    _place_keys: Set[str] = field(default_factory=set, init=False)
    _last_active: Set[str] = field(default_factory=set, init=False)
    _prev_grasped: bool = field(default=False, init=False)
    _t: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self.segmenter = PhaseSegmenter(self.events)
        # grasp keys drive drop/re-grasp detection; place keys ("place"/"in"/"on") gate a
        # falling grasp edge as a genuine DROP vs a normal release-after-placement.
        self._grasp_keys = {e.key for e in self.events if e.detect == "grasp"}
        self._place_keys = {e.key for e in self.events if e.detect in ("place", "in", "on")}

    def reset(self) -> None:
        self.phase_timeline = []
        self.event_steps = {}
        self.grasp_timeline = []
        self.grasp_steps = []
        self.drop_steps = []
        self._last_active = set()
        self._prev_grasped = False
        self._t = -1

    def step(self) -> str:
        self._t += 1
        active = set(self.probe.active_events())     # CURRENT per-step predicates (not first-fire)
        self._last_active = active
        for key in active:
            self.event_steps.setdefault(key, self._t)  # first-fire step (recurrence-safe)
        grasped = bool(active & self._grasp_keys)
        placed = bool(active & self._place_keys)
        if grasped and not self._prev_grasped:           # rising edge = a grasp fire
            self.grasp_steps.append(self._t)
        elif self._prev_grasped and not grasped and not placed:  # lost grasp before placing = drop
            self.drop_steps.append(self._t)
        self._prev_grasped = grasped
        self.grasp_timeline.append(grasped)
        ph = self.segmenter.phase_for(active)            # STATE-BASED (non-monotone)
        self.phase_timeline.append(ph)
        return ph

    @property
    def grasp_count(self) -> int:
        return len(self.grasp_steps)

    @property
    def progress(self) -> int:
        """CURRENT-state milestone count (highest active event index + 1; non-monotone)."""
        return self.segmenter.highest_active_index(self._last_active) + 1

    @property
    def max_phase_label(self) -> str:
        """FURTHEST phase reached over the episode (by pipeline order), NOT the last phase.

        With non-monotone reverts the last phase can be an earlier gap (e.g. reach-to-object
        after a drop); ``max_phase_label`` reports how far the episode actually got.
        """
        if not self.phase_timeline:
            return self.segmenter.gap_labels[0]
        return max(self.phase_timeline, key=self.segmenter.phase_rank)

    @property
    def event_order_keys(self) -> List[str]:
        return [e.key for e in self.events]
