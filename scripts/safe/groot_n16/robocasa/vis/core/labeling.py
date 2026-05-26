"""Labeling schemes for SAFE rollout cluster analysis.

Each scheme maps a list of per-rollout records (with keys: task, success,
task_description) to (labels, names, fail_label). Labels are int IDs; names
are display strings; fail_label (optional) tells plotting code which label
should be styled as the failure-cluster (e.g., square marker + grey).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TASKS_ORDERED = (
    "CoffeeSetupMug",
    "OpenSingleDoor",
    "PnPCounterToCab",
    "PnPSinkToCounter",
    "PnPCounterToStove",
    "OpenDrawer",
)
_TASK_TO_ID = {t: i for i, t in enumerate(TASKS_ORDERED)}
PNP_TASKS = {"PnPCounterToCab", "PnPSinkToCounter", "PnPCounterToStove"}


@dataclass
class LabelingResult:
    labels: np.ndarray          # shape (n_rollouts,), int64
    names: dict[int, str]       # label id -> display string
    fail_label: int | None      # label id treated as 'failure' for plotting (or None)
    scheme: str                 # scheme name (e.g., 'task7')
    extra: dict[str, Any]       # auxiliary info per scheme (e.g., instruction-to-task map)


def _open_sub_instruction(task: str, instruction: str) -> str | None:
    if task == "OpenSingleDoor":
        if "microwave" in instruction:
            return "microwave"
        if "cabinet" in instruction:
            return "cabinet"
    elif task == "OpenDrawer":
        if "right" in instruction:
            return "right"
        if "left" in instruction:
            return "left"
    return None


def label_task(records: list[dict]) -> LabelingResult:
    """6-class: task id 0..5 (CoffeeSetupMug ... OpenDrawer)."""
    labels = np.asarray([_TASK_TO_ID[r["task"]] for r in records], dtype=np.int64)
    names = {i: t for t, i in _TASK_TO_ID.items()}
    return LabelingResult(labels, names, fail_label=None, scheme="task", extra={})


def label_success_fail(records: list[dict]) -> LabelingResult:
    """2-class: 0 = failure, 1 = success."""
    labels = np.asarray([int(r["success"]) for r in records], dtype=np.int64)
    names = {0: "failure", 1: "success"}
    return LabelingResult(labels, names, fail_label=0, scheme="success_fail", extra={})


def label_task_failure(records: list[dict]) -> LabelingResult:
    """12-class: task * 2 + (1 - succ)."""
    labels = np.asarray(
        [_TASK_TO_ID[r["task"]] * 2 + (1 - int(r["success"])) for r in records],
        dtype=np.int64,
    )
    names: dict[int, str] = {}
    for t, tid in _TASK_TO_ID.items():
        names[tid * 2] = f"{t} · success"
        names[tid * 2 + 1] = f"{t} · failure"
    return LabelingResult(labels, names, fail_label=None, scheme="task_failure", extra={})


def label_task7(records: list[dict]) -> LabelingResult:
    """7-class: 6 task-success clusters + 1 unified failure cluster (id 6)."""
    labels = []
    for r in records:
        if int(r["success"]) == 1:
            labels.append(_TASK_TO_ID[r["task"]])
        else:
            labels.append(6)
    names = {i: f"{t} · succ" for t, i in _TASK_TO_ID.items()}
    names[6] = "Failure (all tasks)"
    return LabelingResult(
        labels=np.asarray(labels, dtype=np.int64),
        names=names,
        fail_label=6,
        scheme="task7",
        extra={},
    )


def label_fine_open(records: list[dict]) -> LabelingResult:
    """8-class: same as `task` but Open* split by sub-instruction.

    0 Coffee, 1 OpenSingleDoor/microwave, 2 .../cabinet,
    3 PnPCounterToCab, 4 PnPSinkToCounter, 5 PnPCounterToStove,
    6 OpenDrawer/right, 7 OpenDrawer/left.
    """
    key_to_id = {
        ("CoffeeSetupMug", None): 0,
        ("OpenSingleDoor", "microwave"): 1,
        ("OpenSingleDoor", "cabinet"): 2,
        ("PnPCounterToCab", None): 3,
        ("PnPSinkToCounter", None): 4,
        ("PnPCounterToStove", None): 5,
        ("OpenDrawer", "right"): 6,
        ("OpenDrawer", "left"): 7,
    }
    names = {
        0: "CoffeeSetupMug",
        1: "OpenSingleDoor / microwave",
        2: "OpenSingleDoor / cabinet",
        3: "PnPCounterToCab",
        4: "PnPSinkToCounter",
        5: "PnPCounterToStove",
        6: "OpenDrawer / right",
        7: "OpenDrawer / left",
    }
    labels = []
    for r in records:
        task = r["task"]
        instr = r.get("task_description", "")
        sub = _open_sub_instruction(task, instr) if task in ("OpenSingleDoor", "OpenDrawer") else None
        key = (task, sub)
        if key not in key_to_id:
            raise ValueError(f"fine_open: unknown (task, sub) = {key} (instr={instr!r})")
        labels.append(key_to_id[key])
    return LabelingResult(
        labels=np.asarray(labels, dtype=np.int64),
        names=names,
        fail_label=None,
        scheme="fine_open",
        extra={},
    )


def label_coarse_group(records: list[dict]) -> LabelingResult:
    """3-class: Coffee (0) / Open (1) / PnP (2)."""
    group_id = {
        "CoffeeSetupMug": 0,
        "OpenSingleDoor": 1,
        "OpenDrawer": 1,
        "PnPCounterToCab": 2,
        "PnPSinkToCounter": 2,
        "PnPCounterToStove": 2,
    }
    labels = np.asarray([group_id[r["task"]] for r in records], dtype=np.int64)
    names = {0: "Coffee", 1: "Open", 2: "PnP"}
    return LabelingResult(labels, names, fail_label=None, scheme="coarse_group", extra={})


def label_pnp_instruction(records: list[dict]) -> LabelingResult:
    """Variable-class: one cluster per unique task_description (PnP-only).

    Caller is expected to pre-filter records to PnP tasks; this function
    raises ValueError if any non-PnP record is present.
    """
    if any(r["task"] not in PNP_TASKS for r in records):
        raise ValueError("label_pnp_instruction requires PnP-only records (filter scope=pnp_only)")
    instrs = [r.get("task_description", "") for r in records]
    uniq = sorted(set(instrs))
    id_map = {s: i for i, s in enumerate(uniq)}
    labels = np.asarray([id_map[s] for s in instrs], dtype=np.int64)
    names = {i: s for s, i in id_map.items()}
    instruction_task: dict[int, str] = {}
    for r in records:
        instruction_task[id_map[r.get("task_description", "")]] = r["task"]
    return LabelingResult(
        labels=labels,
        names=names,
        fail_label=None,
        scheme="pnp_instruction",
        extra={"instruction_task": instruction_task},
    )


SCHEMES = {
    "task": label_task,
    "success_fail": label_success_fail,
    "task_failure": label_task_failure,
    "task7": label_task7,
    "fine_open": label_fine_open,
    "coarse_group": label_coarse_group,
    "pnp_instruction": label_pnp_instruction,
}


def apply(scheme: str, records: list[dict]) -> LabelingResult:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown labeling scheme: {scheme!r}; choices = {list(SCHEMES)}")
    return SCHEMES[scheme](records)
