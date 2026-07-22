"""Normalized loader for our RoboCasa GR00T rollout files (read-only).

This adapter turns one rollout triplet ``<dir>/task<id>--ep<idx>--succ<0|1>.{pkl,mp4,csv}``
into a single normalized ``EpisodeData`` dict that a downstream segmentation module
consumes. It is **read-only** with respect to the data: it never writes, mutates, or
re-renders the rollout artifacts.

OUR DATA SCHEMA (verified across two on-disk variants)
------------------------------------------------------
Rollouts live as filename triplets, e.g. under::

    outputs/eval/robocasa/groot_n16/<run>/{train,val_seen,val_unseen}/<TaskName>/
    outputs/eval/robocasa/groot_n15/<run>/raw_rollouts/<TaskName>/

The ``.pkl`` (load with ``pickle``; needs ``torch``+``numpy`` importable — run under
``~/miniconda3/envs/lerobot_safe/bin/python``) carries, among others:

  - ``task_description`` (str)
  - ``ep_meta`` (dict; ``ep_meta['lang']`` is the human instruction when present)
  - ``episode_success`` (int 0/1)
  - ``env_name`` (str, e.g. ``robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env``)
  - ``states``: ``list[dict]`` per executed step OR ``None`` (feature-export variant).
    Each step dict (when present) has unified keys:
    ``observation.state.eef_pos_rel`` [3], ``observation.state.eef_quat_rel`` [4],
    ``observation.state.gripper_qpos`` [2], ``observation.state.base_position`` [3],
    ``observation.state.base_rotation`` [4].
  - ``action_vectors``: ``np[N, 12]`` =
    ``[eef_pos dx,dy,dz | eef_rot drx,dry,drz | gripper | base_motion x4 | control_mode]``.
    ``N`` is the number of executed action-steps for the executed-rollout variant.
  - ``actions``: ``list`` of native action-chunk dicts (per executed step), each with
    ``action.end_effector_position`` [.,16,3], ``action.end_effector_rotation`` [.,16,3],
    ``action.gripper_close`` [.,16,1], ``action.base_motion`` [.,16,4],
    ``action.control_mode`` [.,16,1]. (The leading axis may be absent: shapes are
    seen as both ``(16,3)`` and ``(1,16,3)`` across runs.)
  - ``video_fps`` (20 when present), ``steps_per_render``, ``n_action_steps``,
    ``max_episode_steps`` — any of these may be ``None`` in the feature-export variant.

Two on-disk variants both satisfy the contract below:
  * Executed-rollout (e.g. groot_n15 ``coast*/raw_rollouts/``): ``states`` populated,
    ``ep_meta['lang']`` set, ``action_vectors`` = [N_steps, 12], with a sibling ``.mp4``.
  * Feature-export (e.g. groot_n16 ``safe_split_*``): ``states is None``,
    ``ep_meta`` empty; we fall back to ``task_description`` and derive geometry from
    ``action_vectors`` only.

The ``.mp4`` is **3 camera views concatenated horizontally**: total width 3*256 = 768,
height 256. Panel order is ``[side_0 | side_1 | wrist_0]``. Per
``src/policies/groot/core/schema.py`` ``GROOT_ENV_VIDEO_TO_UNIFIED_CAM``,
``side_0 -> left``, ``side_1 -> right``, ``wrist_0 -> wrist``. So panel0 = left exterior,
panel1 = right exterior, panel2 = wrist. Decoded with ``cv2.VideoCapture`` (no ffmpeg);
cv2 returns BGR, which we convert to RGB.

The ``.csv`` holds per-step action delta columns (``action/dx`` .. ``action/dgripper``),
redundant with ``action_vectors``; ignored here (available for cross-check).

EpisodeData CONTRACT (exact keys returned by :func:`load_episode`)
-----------------------------------------------------------------
``{``
  ``"task": str,``                 # ep_meta['lang'] if present else task_description
  ``"success": int,``              # episode_success
  ``"n_steps": int,``              # len(states) / action_vectors.shape[0]
  ``"frames": {"exterior": uint8[T,H,W,3], "exterior2": uint8[T,H,W,3],``
  ``           "wrist": uint8[T,H,W,3]},``  # RGB, 768-wide mp4 split into 3 panels
  ``"frame_to_step": int[T],``     # round(f * n_steps / T) per decoded frame
  ``"ee_pos": float[n_steps,3] | None,``   # from states eef_pos_rel (None if no states)
  ``"ee_quat": float[n_steps,4] | None,``  # from states eef_quat_rel (None if no states)
  ``"ee_delta": float[n_steps,6],``        # action_vectors[:, 0:6]
  ``"gripper": float[n_steps],``           # action_vectors[:, 6] (executed-per-step)
  ``"has_gripper": bool,``                 # max |diff(gripper)| > ~0.1
  ``"meta": {"pkl": str, "mp4": str, "task_name": <dir name>, "env_name": ...},``
``}``

GRIPPER SIGNAL CHOICE
---------------------
``gripper`` uses ``action_vectors[:, 6]``. This is the executed-per-step gripper command
(the first action of each executed step's chunk), and we verified it is identical to
``actions[i]['action.gripper_close'][..., 0, 0]`` for every step. ``action_vectors`` is
preferred because it is always a clean ``[N]`` series, whereas the native chunk axis
layout varies across runs. ``has_gripper`` is True iff the per-step gripper command ever
makes a real open/close transition (``max |diff| > GRIPPER_TRANSITION_THRESHOLD``):
grasping tasks (PickPlace / CoffeeSetupMug) show ~1.0 jumps; pure door/drawer tasks
(CloseFridge / OpenDrawer) stay flat.
"""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # noqa: F401  (used in _decode_mp4_panels)

    _HAVE_CV2 = True
except Exception:  # pragma: no cover - cv2 expected in lerobot_safe env
    _HAVE_CV2 = False


# ── constants ─────────────────────────────────────────────────────────────────

# Real open/close gripper transitions exceed this; flat door/drawer tasks stay below.
GRIPPER_TRANSITION_THRESHOLD = 0.1

# action_vectors[:, 0:6] = eef_pos delta (3) + eef_rot delta (3); col 6 = gripper.
EE_DELTA_SLICE = slice(0, 6)
GRIPPER_COL = 6

# mp4 panel order [side_0 | side_1 | wrist_0] -> our view names, per
# src/policies/groot/core/schema.py GROOT_ENV_VIDEO_TO_UNIFIED_CAM
# (side_0->left, side_1->right, wrist_0->wrist).
DEFAULT_VIEWS = ("exterior", "exterior2", "wrist")
PANEL_VIEW_ORDER = ("exterior", "exterior2", "wrist")  # panel0, panel1, panel2

# state keys (unified) inside each per-step dict
STATE_EEF_POS = "observation.state.eef_pos_rel"
STATE_EEF_QUAT = "observation.state.eef_quat_rel"

_FILENAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])")


# ── pkl loading ───────────────────────────────────────────────────────────────


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Detach/CPU/numpy any torch tensor; passthrough for array-likes.

    Mirrors the idiom in scripts/safe/groot_n16/robocasa/safe_feature_vectors.py.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pkl(pkl_path: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def _instruction(record: dict[str, Any]) -> str:
    ep_meta = record.get("ep_meta") or {}
    lang = ep_meta.get("lang") if isinstance(ep_meta, dict) else None
    if lang:
        return str(lang)
    desc = record.get("task_description")
    return str(desc) if desc is not None else ""


def _states_geometry(
    states: list[dict[str, Any]] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stack per-step eef_pos_rel [n,3] and eef_quat_rel [n,4] from states list.

    Returns (None, None) when states is absent (feature-export variant).
    """
    if not states:
        return None, None
    pos, quat = [], []
    for step in states:
        pos.append(np.asarray(step[STATE_EEF_POS], dtype=np.float32).reshape(-1))
        quat.append(np.asarray(step[STATE_EEF_QUAT], dtype=np.float32).reshape(-1))
    return (
        np.stack(pos, axis=0).astype(np.float32),
        np.stack(quat, axis=0).astype(np.float32),
    )


# ── mp4 decoding ──────────────────────────────────────────────────────────────


def _split_panels(frame_rgb: np.ndarray, n_panels: int = 3) -> list[np.ndarray]:
    """Split a [H, W, 3] frame into ``n_panels`` horizontal panels.

    If W is a clean multiple of n_panels, split evenly. Otherwise log a warning and
    split as evenly as possible (last panel absorbs the remainder).
    """
    h, w = frame_rgb.shape[:2]
    if w % n_panels == 0:
        pw = w // n_panels
        return [frame_rgb[:, i * pw : (i + 1) * pw] for i in range(n_panels)]
    pw = w // n_panels
    warnings.warn(
        f"mp4 width {w} is not a multiple of {n_panels}; splitting at {pw}px "
        f"with the last panel taking the remainder.",
        stacklevel=2,
    )
    bounds = [i * pw for i in range(n_panels)] + [w]
    return [frame_rgb[:, bounds[i] : bounds[i + 1]] for i in range(n_panels)]


def _decode_mp4_panels(
    mp4_path: Path, view_names: tuple[str, str, str]
) -> dict[str, np.ndarray]:
    """Decode the 3-panel mp4 into per-view uint8 RGB stacks [T, H, W, 3].

    Panel0 -> view_names[0] (left exterior), panel1 -> view_names[1] (right exterior),
    panel2 -> view_names[2] (wrist). cv2 yields BGR; we convert to RGB.
    """
    if not _HAVE_CV2:  # pragma: no cover
        raise RuntimeError(
            "cv2 unavailable: run under ~/miniconda3/envs/lerobot_safe/bin/python"
        )
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cv2 could not open mp4: {mp4_path}")

    panels: list[list[np.ndarray]] = [[], [], []]
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            for i, panel in enumerate(_split_panels(frame_rgb, n_panels=3)):
                panels[i].append(panel)
    finally:
        cap.release()

    frames: dict[str, np.ndarray] = {}
    for view, panel_list in zip(view_names, panels):
        if panel_list:
            frames[view] = np.stack(panel_list, axis=0).astype(np.uint8)
        else:
            frames[view] = np.empty((0, 0, 0, 3), dtype=np.uint8)
    return frames


def _frame_to_step(n_frames: int, n_steps: int) -> np.ndarray:
    """Map each decoded video frame index f -> nearest action-step idx.

    round(f * n_steps / T), clamped to [0, n_steps - 1].
    """
    if n_frames <= 0 or n_steps <= 0:
        return np.zeros((max(n_frames, 0),), dtype=np.int64)
    idx = np.round(np.arange(n_frames) * n_steps / n_frames).astype(np.int64)
    return np.clip(idx, 0, n_steps - 1)


# ── public API ────────────────────────────────────────────────────────────────


def load_episode(
    pkl_path: str | Path,
    views: tuple[str, str, str] = DEFAULT_VIEWS,
) -> dict[str, Any]:
    """Load one rollout triplet into the normalized EpisodeData dict.

    See the module docstring for the exact returned contract. ``views`` names the three
    decoded panels in panel order ``[side_0, side_1, wrist_0]`` (left, right, wrist).
    The sibling mp4 is ``pkl_path`` with a ``.mp4`` suffix; if it is missing, ``frames``
    is an empty dict and ``frame_to_step`` is empty (noted in ``meta``). When ``states``
    is absent, ``ee_pos``/``ee_quat`` are ``None`` and geometry is derived from
    ``action_vectors`` only.
    """
    if len(views) != 3:
        raise ValueError(f"views must name 3 panels, got {views!r}")

    pkl_path = Path(pkl_path)
    record = _load_pkl(pkl_path)

    action_vectors = _tensor_to_numpy(record["action_vectors"]).astype(np.float32)
    if action_vectors.ndim != 2 or action_vectors.shape[1] < 7:
        raise ValueError(
            f"Expected action_vectors [N, >=7], got {action_vectors.shape}"
        )

    states = record.get("states")
    ee_pos, ee_quat = _states_geometry(states)

    # n_steps: states length when present, else action_vectors row count.
    n_steps = len(states) if states else int(action_vectors.shape[0])
    if ee_pos is not None and ee_pos.shape[0] != action_vectors.shape[0]:
        # Defensive: keep them consistent; trust action_vectors for delta/gripper length.
        n_steps = int(action_vectors.shape[0])

    ee_delta = action_vectors[:, EE_DELTA_SLICE].astype(np.float32)
    gripper = action_vectors[:, GRIPPER_COL].astype(np.float32)
    has_gripper = bool(
        gripper.shape[0] > 1
        and np.max(np.abs(np.diff(gripper))) > GRIPPER_TRANSITION_THRESHOLD
    )

    meta: dict[str, Any] = {
        "pkl": str(pkl_path),
        "mp4": "",
        "task_name": pkl_path.parent.name,
        "env_name": record.get("env_name"),
    }

    mp4_path = pkl_path.with_suffix(".mp4")
    frames: dict[str, np.ndarray] = {}
    frame_to_step = np.zeros((0,), dtype=np.int64)
    if mp4_path.exists():
        meta["mp4"] = str(mp4_path)
        frames = _decode_mp4_panels(mp4_path, views)
        n_frames = next((v.shape[0] for v in frames.values() if v.size), 0)
        frame_to_step = _frame_to_step(n_frames, n_steps)
    else:
        meta["mp4_missing"] = True

    return {
        "task": _instruction(record),
        "success": int(record.get("episode_success", 0)),
        "n_steps": int(n_steps),
        "frames": frames,
        "frame_to_step": frame_to_step,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "ee_delta": ee_delta,
        "gripper": gripper,
        "has_gripper": has_gripper,
        "meta": meta,
    }


def list_rollouts(
    root: str | Path,
    task: str | None = None,
    success: int | None = None,
) -> list[Path]:
    """Glob rollout pkls under ``root``, optionally filtered by task and success.

    Matches ``**/task*--ep*--succ*.pkl``. ``task`` is a case-insensitive substring
    matched against the immediate parent directory name (the TaskName dir). ``success``
    filters by the ``succ<0|1>`` flag parsed from the filename. Returns a sorted list of
    resolvable (existing) pkl paths.
    """
    root = Path(root)
    task_lc = task.lower() if task else None
    out: list[Path] = []
    for pkl in sorted(root.glob("**/task*--ep*--succ*.pkl")):
        m = _FILENAME_RE.search(pkl.name)
        if not m:
            continue
        if success is not None and int(m.group(3)) != int(success):
            continue
        if task_lc is not None and task_lc not in pkl.parent.name.lower():
            continue
        # skip broken symlinks (some splits symlink to archived sources)
        if not pkl.exists():
            continue
        out.append(pkl)
    return out


def select_pilot_set(
    root: str | Path,
    specs: list[tuple[str, int, int]],
) -> list[Path]:
    """Build a balanced pilot list of pkl paths.

    ``specs`` is a list of ``(task_substring, n_success, n_fail)``. For each spec we take
    up to ``n_success`` successful and ``n_fail`` failed rollouts whose parent TaskName
    dir contains ``task_substring`` (case-insensitive). Returns the concatenated list in
    spec order (success rollouts first within each task), de-duplicated.
    """
    selected: list[Path] = []
    seen: set[Path] = set()
    for task_substring, n_success, n_fail in specs:
        succ = list_rollouts(root, task=task_substring, success=1)[: max(n_success, 0)]
        fail = list_rollouts(root, task=task_substring, success=0)[: max(n_fail, 0)]
        for p in (*succ, *fail):
            if p not in seen:
                seen.add(p)
                selected.append(p)
    return selected


# ── self-test ─────────────────────────────────────────────────────────────────


def _print_episode(label: str, ep: dict[str, Any]) -> None:
    fshapes = {v: arr.shape for v, arr in ep["frames"].items()}
    f2s = ep["frame_to_step"]
    f2s_range = (int(f2s.min()), int(f2s.max())) if f2s.size else None
    print(f"\n[{label}]  {ep['meta']['pkl']}")
    print(f"  task        : {ep['task']!r}")
    print(f"  success     : {ep['success']}")
    print(f"  n_steps     : {ep['n_steps']}")
    print(f"  frames      : {fshapes}")
    print(f"  frame_to_step: len={f2s.size} range={f2s_range}")
    print(
        "  ee_pos      : "
        + (str(ep["ee_pos"].shape) if ep["ee_pos"] is not None else "None")
    )
    print(
        "  ee_quat     : "
        + (str(ep["ee_quat"].shape) if ep["ee_quat"] is not None else "None")
    )
    print(f"  ee_delta    : {ep['ee_delta'].shape}")
    print(f"  gripper     : {ep['gripper'].shape}")
    print(f"  has_gripper : {ep['has_gripper']}")
    print(f"  env_name    : {ep['meta']['env_name']}")


def _selftest() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    n15 = repo_root / "outputs/eval/robocasa/groot_n15"

    # Grasp task (expect has_gripper=True): CoffeeSetupMug = "pick the mug ...".
    grasp_dir = n15 / "coast4_instruction_pathway_50ep/raw_rollouts/CoffeeSetupMug"
    # No-grasp task (expect has_gripper=False): CloseFridge = "close the fridge door".
    nograsp_dir = n15 / "coast4_instruction_pathway_50ep/raw_rollouts/CloseFridge"

    grasp = list_rollouts(grasp_dir, success=1)
    nograsp = list_rollouts(nograsp_dir, success=1)
    if not grasp or not nograsp:
        print(
            "Self-test rollouts not found under groot_n15; checked:\n"
            f"  grasp_dir  = {grasp_dir} ({len(grasp)} found)\n"
            f"  nograsp_dir= {nograsp_dir} ({len(nograsp)} found)\n"
            "Adjust paths to a run with full-schema rollouts (states + mp4)."
        )
        return

    ep_grasp = load_episode(grasp[0])
    ep_nograsp = load_episode(nograsp[0])
    _print_episode("GRASP / PickPlace-like (CoffeeSetupMug)", ep_grasp)
    _print_episode("NO-GRASP (CloseFridge)", ep_nograsp)

    print("\n=== has_gripper assertions ===")
    print(
        f"  grasp   has_gripper={ep_grasp['has_gripper']}  "
        f"(expect True)  -> {'OK' if ep_grasp['has_gripper'] else 'FAIL'}"
    )
    print(
        f"  nograsp has_gripper={ep_nograsp['has_gripper']}  "
        f"(expect False) -> {'OK' if not ep_nograsp['has_gripper'] else 'FAIL'}"
    )


if __name__ == "__main__":
    _selftest()
