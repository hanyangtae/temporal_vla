"""INSIGHT primitive-segmentation, adapted to in-memory robocasa GR00T rollouts.

PROVENANCE
----------
Ported from the INSIGHT VLA project (https://github.com/insight-vla/insight,
Apache-2.0), file ``sim/libero_flywheel/data_processing/densely_label_dataset.py``.
Each ported function below cites the original function it derives from. The core
logic (gripper-command segmentation, recursive variance-reduction changepoints,
two-pass VLM video labeling with EE-caption boundary refinement, frame
annotation) is preserved; only the DATA SOURCE changed: the original consumed a
LeRobot ``dataset`` indexed by global frame id, this version consumes an
in-memory ``EpisodeData`` dict of numpy arrays. The verbatim VLM system prompts
live in ``prompts.py`` (also ported, with their own provenance header).

EpisodeData CONTRACT (the upstream adapter produces exactly this; depend on it)
------------------------------------------------------------------------------
::

    ep = {
      "task": str,                       # language instruction
      "success": int,                    # 0/1
      "n_steps": int,                    # number of executed env/action steps
      "frames": {                        # decoded RGB frames per view, np.uint8 [T,H,W,3]
          "exterior": np.ndarray, "exterior2": np.ndarray, "wrist": np.ndarray },
      "frame_to_step": np.ndarray,       # [T] int, video-frame idx -> action-step idx
      "ee_pos": np.ndarray,              # [n_steps,3] eef position (base-relative)
      "ee_quat": np.ndarray,             # [n_steps,4]
      "ee_delta": np.ndarray,            # [n_steps,6] per-step [dx,dy,dz,drx,dry,drz]
      "gripper": np.ndarray,             # [n_steps] gripper signal
      "has_gripper": bool,               # gripper shows real transitions
      "meta": dict,                      # task name, paths, etc.
    }

``segment_episode(ep, vlm, cfg)`` returns ordered segments that COVER THE WHOLE
EPISODE in step units::

    [{"start_step": int, "end_step": int, "label": str}, ...]

with ``segments[0]["start_step"] == 0`` and
``segments[-1]["end_step"] == ep["n_steps"]``, contiguous and non-overlapping.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from prompts import (
    ASSIGN_PLAN_EE_NOTE,
    ASSIGN_PLAN_TO_FRAMES_PROMPT,
    DEFAULT_KNOWN_PRIMITIVES,
    LABEL_EPISODE_FROM_VIDEO_PROMPT,
    REFINE_BOUNDARY_PROMPT,
)

logger = logging.getLogger(__name__)

# Default primary view used for VLM keyframes when cfg.primary_view is unset.
_DEFAULT_PRIMARY_VIEW = "exterior"


# ---------------------------------------------------------------------------
# Config — mirrors the subset of INSIGHT densely_label_dataset.py :: Args that
# this in-memory port actually uses.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SegConfig:
    # Method selection: "video" (VLM labels from keyframes — INSIGHT primary
    # path), "gripper" (segment_episode_by_gripper), or "action_change"
    # (find_action_changepoints on the EE trajectory; for gripperless tasks).
    segment_method: str = "video"
    has_gripper: bool = True            # gripper shows real transitions (Args.has_gripper)
    model: Optional[str] = None         # VLM model override (None -> client default)

    # Video / VLM labeling
    num_keyframes: int = 24             # keyframes sampled from the episode for the VLM
    primary_view: str = _DEFAULT_PRIMARY_VIEW   # ep["frames"][view] used for keyframes
    extra_views: Tuple[str, ...] = ()   # additional camera views to interleave
    known_primitives: Tuple[str, ...] = tuple(DEFAULT_KNOWN_PRIMITIVES)
    plan: Optional[Tuple[str, ...]] = None  # fixed primitive_sequence (skip plan decomposition)
    refine_boundaries: bool = True      # second pass: snap boundaries to state changepoints
    refine_window: int = 15             # half-window (frames) around each coarse boundary
    use_ee_captions: bool = True        # inject per-frame EE caption + dom-axis tag

    # Gripper segmentation (segment_episode_by_gripper)
    gripper_motion_threshold: float = 0.002   # |Δgripper_cmd| velocity threshold
    gripper_close_threshold: float = 0.0      # direction tiebreaker (kept for parity)
    min_segment_length: int = 5

    # Changepoint tuning (find_action_changepoints)
    changepoint_window: int = 5         # smoothing window for variance-reduction
    changepoint_normalize: bool = False  # z-score features per-dim before variance
    changepoint_backtrack: str = "backward"  # "backward" | "forward" | "none"
    changepoint_n_splits: int = 4       # number of split points for action_change method


# ===========================================================================
# State-changepoint helper.
# Ported from densely_label_dataset.py :: find_action_changepoints().
# Original used torch tensors indexed over the full episode; this version is a
# faithful numpy reimplementation operating on a [T, D] feature array. The
# recursive variance-reduction binary-split algorithm, effect-size gate (0.3),
# and backtrack refinement are preserved.
# ===========================================================================
def find_action_changepoints(
    features: np.ndarray,
    seg_start: int,
    seg_end: int,
    n_splits: int,
    window: int = 5,
    normalize_features: bool = False,
    min_seg_len: Optional[int] = None,
    backtrack: str = "backward",
) -> List[int]:
    """Find changepoints in a [T, D] feature profile within [seg_start, seg_end).

    Recursive binary splitting: at each step pick the sub-segment whose
    best within-segment variance-reduction split scores highest, split it, and
    repeat until ``n_splits`` boundaries are placed (or no meaningful reduction
    remains). Returns split indices in original (episode) numbering.
    """
    seg_source = np.asarray(features[seg_start:seg_end], dtype=np.float64)
    seg_len = len(seg_source)
    min_seg = min_seg_len if min_seg_len is not None else max(3, window)
    if seg_len < min_seg * 2:
        return [seg_start + int(seg_len * (i + 1) / (n_splits + 1)) for i in range(n_splits)]

    # Feature: up to first 6 dims (translation + rotation).
    feat_dim = min(6, seg_source.shape[-1])
    full_features = seg_source[:, :feat_dim]

    # Smooth each dim with a moving average (replicate padding), matching the
    # original conv1d-with-replicate-pad behaviour.
    if seg_len > window:
        kernel = np.ones(window) / window
        pad = window // 2
        smoothed_cols = []
        for dim in range(feat_dim):
            sig = full_features[:, dim]
            padded = np.pad(sig, (pad, pad), mode="edge")
            s = np.convolve(padded, kernel, mode="valid")
            # conv1d valid output length = len(padded) - window + 1; trim/extend to seg_len.
            s = s[:seg_len]
            if len(s) < seg_len:
                s = np.pad(s, (0, seg_len - len(s)), mode="edge")
            smoothed_cols.append(s)
        full_features = np.stack(smoothed_cols, axis=1)

    # Optional per-dim z-score.
    if normalize_features:
        mu = full_features.mean(axis=0, keepdims=True)
        sd = full_features.std(axis=0, keepdims=True) + 1e-6
        full_features = (full_features - mu) / sd

    def best_split_in(start: int, end: int) -> Tuple[int, float]:
        sub_len = end - start
        if sub_len < min_seg * 2:
            return -1, float("-inf")
        sub = full_features[start:end]
        total_var = float(sub.var(axis=0).sum())
        if total_var < 1e-6:
            return -1, float("-inf")
        best_idx = -1
        best_score = float("-inf")
        for t in range(min_seg, sub_len - min_seg):
            left_var = float(sub[:t].var(axis=0).sum())
            right_var = float(sub[t:].var(axis=0).sum())
            weighted_var = (t * left_var + (sub_len - t) * right_var) / sub_len
            score = total_var - weighted_var
            if score > best_score:
                best_score = score
                best_idx = t
        return (start + best_idx if best_idx >= 0 else -1), best_score

    def apply_backtrack(local_idx: int, start: int, end: int) -> int:
        if backtrack == "none":
            return local_idx
        sub = full_features[start:end]
        split_rel = local_idx - start
        left_mean = sub[:split_rel].mean(axis=0)
        right_mean = sub[split_rel:].mean(axis=0)
        idx = split_rel
        if backtrack == "backward":
            for bi in range(split_rel - 1, 0, -1):
                if np.abs(sub[bi] - right_mean).mean() < np.abs(sub[bi] - left_mean).mean():
                    idx = bi
                else:
                    break
        elif backtrack == "forward":
            for fi in range(split_rel + 1, len(sub)):
                if np.abs(sub[fi] - left_mean).mean() < np.abs(sub[fi] - right_mean).mean():
                    idx = fi
                else:
                    break
        return start + idx

    sub_segments: List[Tuple[int, int]] = [(0, seg_len)]
    cached_candidates = {sub_segments[0]: best_split_in(0, seg_len)}
    split_points: List[int] = []

    for _ in range(n_splits):
        best_sub = None
        best_split_idx = -1
        best_score = float("-inf")
        for sub in sub_segments:
            cand_idx, cand_score = cached_candidates[sub]
            if cand_score > best_score and cand_idx >= 0:
                best_score = cand_score
                best_sub = sub
                best_split_idx = cand_idx
        if best_sub is None or best_score <= 0:
            break
        sub_start, sub_end = best_sub
        # Effect-size gate.
        sub_feats = full_features[sub_start:sub_end]
        left_mean = full_features[sub_start:best_split_idx].mean(axis=0)
        right_mean = full_features[best_split_idx:sub_end].mean(axis=0)
        pooled_std = sub_feats.std(axis=0) + 1e-8
        effect_size = float((np.abs(left_mean - right_mean) / pooled_std).mean())
        if effect_size < 0.3:
            cached_candidates[best_sub] = (-1, float("-inf"))
            continue
        refined_split = apply_backtrack(best_split_idx, sub_start, sub_end)
        split_points.append(seg_start + refined_split)
        sub_segments.remove(best_sub)
        del cached_candidates[best_sub]
        left_sub = (sub_start, refined_split)
        right_sub = (refined_split, sub_end)
        sub_segments.append(left_sub)
        sub_segments.append(right_sub)
        cached_candidates[left_sub] = best_split_in(*left_sub)
        cached_candidates[right_sub] = best_split_in(*right_sub)

    return sorted(split_points)


# ===========================================================================
# Gripper-command segmentation.
# Ported from densely_label_dataset.py :: segment_episode_by_gripper().
# Operates on the per-step gripper command (ep["gripper"]) and EE delta z-axis
# for the long-"move" sub-division. Returns (start, end, type) in step units.
# ===========================================================================
def segment_episode_by_gripper(
    gripper_cmd: np.ndarray,
    ee_delta: np.ndarray,
    min_segment_length: int = 5,
    gripper_motion_threshold: float = 0.002,
    gripper_close_threshold: float = 0.0,  # direction tiebreaker only (parity)
) -> List[Tuple[int, int, str]]:
    """Segment based on gripper ACTION COMMAND transitions.

    A step belongs to a gripper segment iff the command is moving at that step
    (``|Δgripper_cmd| > gripper_motion_threshold``); connected moving regions
    are gripper transitions, everything else is "move". Direction comes from the
    sign of the command change across the region. Long "move" segments are
    sub-divided on vertical (Δz) motion-direction reversals.

    Returns list of (start, end, segment_type) with segment_type in
    {"move", "close_gripper", "open_gripper"}.
    """
    gripper_cmd = np.asarray(gripper_cmd, dtype=np.float64).reshape(-1)
    n_frames = len(gripper_cmd)
    if n_frames == 0:
        return [(0, 0, "move")]

    # Frame-to-frame velocity; prepend first sample so lengths align.
    gripper_vel = np.abs(np.diff(gripper_cmd, prepend=gripper_cmd[:1]))
    in_transition = gripper_vel > gripper_motion_threshold

    transitions: List[Tuple[int, int, str]] = []
    i = 0
    while i < n_frames:
        if not in_transition[i]:
            i += 1
            continue
        start = i
        while i < n_frames and in_transition[i]:
            i += 1
        end = i  # exclusive
        delta = gripper_cmd[end - 1] - gripper_cmd[max(start - 1, 0)]
        kind = "close_gripper" if delta > 0 else "open_gripper"
        if end - start >= 3:  # ignore single-frame noise spikes
            transitions.append((start, end, kind))

    segments: List[Tuple[int, int, str]] = []
    prev_end = 0
    for start, end, kind in transitions:
        if start > prev_end + min_segment_length:
            segments.append((prev_end, start, "move"))
        segments.append((start, end, kind))
        prev_end = end
    if prev_end < n_frames - min_segment_length:
        segments.append((prev_end, n_frames, "move"))
    if not segments:
        segments = [(0, n_frames, "move")]

    # Merge consecutive "move" segments.
    merged: List[Tuple[int, int, str]] = []
    for seg in segments:
        if merged and seg[2] == "move" and merged[-1][2] == "move":
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], "move")
        else:
            merged.append(seg)

    # Sub-divide long "move" segments on vertical-motion direction changes.
    ee_delta = np.asarray(ee_delta, dtype=np.float64)
    final_segments: List[Tuple[int, int, str]] = []
    for seg in merged:
        start, end, seg_type = seg
        seg_length = end - start
        if seg_type == "move" and seg_length > 30 and ee_delta.shape[0] >= end:
            z_motion = ee_delta[start:end, 2]  # per-step Δz
            sub_segs: List[Tuple[int, int]] = []
            sub_start = 0
            for k in range(10, len(z_motion), 5):
                recent_z = float(z_motion[max(0, k - 10):k].sum())
                upcoming_z = float(z_motion[k:min(len(z_motion), k + 10)].sum())
                if (recent_z > 0.05 and upcoming_z < -0.05) or (recent_z < -0.05 and upcoming_z > 0.05):
                    if k - sub_start >= 10:
                        sub_segs.append((start + sub_start, start + k))
                        sub_start = k
            if len(z_motion) - sub_start >= 10:
                sub_segs.append((start + sub_start, end))
            if len(sub_segs) > 1:
                for ss, se in sub_segs:
                    final_segments.append((ss, se, "move"))
            else:
                final_segments.append(seg)
        else:
            final_segments.append(seg)

    return final_segments if final_segments else merged


# ===========================================================================
# Frame helpers.
# ===========================================================================
def _encode_frame_b64(arr: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB array as a base64 JPEG string.

    Ported from densely_label_dataset.py :: _encode_frame_b64().
    """
    from PIL import Image as PILImage  # lazy
    pil_img = PILImage.fromarray(np.asarray(arr, dtype=np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(resp: str) -> dict:
    """Extract the last top-level JSON object in a VLM response.

    Ported from densely_label_dataset.py :: _extract_json(). Handles reasoning
    prose / code fences around the JSON block.
    """
    resp = (resp or "").strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", resp, re.DOTALL)
    if fenced:
        return json.loads(fenced[-1])
    depth = 0
    end = -1
    for i in range(len(resp) - 1, -1, -1):
        c = resp[i]
        if c == "}":
            if depth == 0:
                end = i
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0 and end > i:
                return json.loads(resp[i:end + 1])
    return json.loads(resp)


def _sample_keyframe_indices(n_frames: int, num_keyframes: int) -> List[int]:
    """Evenly spaced keyframe indices through [0, n_frames)."""
    if n_frames <= num_keyframes:
        return list(range(n_frames))
    return [int(round(i * (n_frames - 1) / (num_keyframes - 1))) for i in range(num_keyframes)]


# ===========================================================================
# Per-frame EE caption (dominant-axis tag).
# Ported from densely_label_dataset.py :: assign_plan_to_frames._ee_caption().
# `states` here is the per-frame EE pose array [T, >=6] (pos + rpy). For
# robocasa we build the [T,6] pose from ep["ee_pos"] + integrated rotation, or
# fall back to position-only when rpy is unavailable.
# ===========================================================================
def _ee_caption(states: Optional[np.ndarray], fi: int) -> str:
    if states is None or len(states) == 0 or fi < 0 or fi >= len(states):
        return ""
    pose = np.asarray(states[fi], dtype=np.float64).tolist()
    sdim = states.shape[-1]
    s = f" | EE x={pose[0]:+.3f} y={pose[1]:+.3f} z={pose[2]:+.3f}"
    if sdim >= 6:
        s += f" rx={pose[3]:+.2f} ry={pose[4]:+.2f} rz={pose[5]:+.2f}"
    left = max(0, fi - 1)
    right = min(len(states) - 1, fi + 1)
    dv = (np.asarray(states[right], dtype=np.float64) - np.asarray(states[left], dtype=np.float64)) / max(right - left, 1)
    dx, dy, dz = float(dv[0]), float(dv[1]), float(dv[2])
    s += f" | Δ dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}"
    xy_mag = (dx * dx + dy * dy) ** 0.5
    z_mag = abs(dz)
    s += f" |Δxy|={xy_mag:.3f} |Δz|={z_mag:.3f}"
    if sdim >= 6:
        drx, dry, drz = float(dv[3]), float(dv[4]), float(dv[5])
        rxy_mag = (drx * drx + dry * dry) ** 0.5
        rz_mag = abs(drz)
        s += f" |Δrxy|={rxy_mag:.2f} |Δrz|={rz_mag:.2f}"
    else:
        drz = 0.0
        rxy_mag = rz_mag = 0.0
    # Dominant-axis tag among {+z, -z, xy, +rz, -rz, rxy, none}.
    axes = {
        "+z" if dz > 0 else "-z": z_mag,
        "xy": xy_mag,
        "+rz" if (sdim >= 6 and drz > 0) else "-rz": rz_mag if sdim >= 6 else 0.0,
        "rxy": rxy_mag,
    }
    dominant, mag = max(axes.items(), key=lambda kv: kv[1])
    if mag < 1e-3:
        dominant = "none"
    s += f" dom={dominant}"
    return s


def _ee_states_from_ep(ep: dict) -> Optional[np.ndarray]:
    """Build a per-step EE pose array [n_steps, 6] = [x,y,z, rx,ry,rz].

    Position is ep["ee_pos"]; the rotation columns are the cumulative sum of the
    per-step rotational deltas in ep["ee_delta"][:, 3:6] (a monotone, readable
    proxy — exact world rpy is not needed, only the per-frame Δ tags matter for
    the boundary cue). Falls back to position-only when delta is missing.
    """
    ee_pos = ep.get("ee_pos")
    if ee_pos is None:
        return None
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    ee_delta = ep.get("ee_delta")
    if ee_delta is None or np.asarray(ee_delta).shape[-1] < 6:
        return ee_pos  # position-only (sdim==3)
    drot = np.asarray(ee_delta, dtype=np.float64)[:, 3:6]
    rot = np.cumsum(drot, axis=0)
    n = min(len(ee_pos), len(rot))
    return np.concatenate([ee_pos[:n], rot[:n]], axis=1)


# ===========================================================================
# VLM video labeling — INSIGHT primary path.
# Ported from densely_label_dataset.py :: label_episode_from_video() and
# assign_plan_to_frames() (coarse pass + EE captions). Adapted to take an
# in-memory keyframe list + a generic vlm.call(messages) client instead of a
# LeRobot dataset + raw OpenAI client.
# ===========================================================================
def _label_episode_from_video(
    vlm,
    keyframes: List[np.ndarray],
    keyframe_orig_idx: List[int],
    extra_keyframes: Optional[List[List[np.ndarray]]],
    extra_view_names: List[str],
    episode_length: int,
    task_description: str,
    plan: Optional[List[str]],
    known_primitives: List[str],
    states: Optional[np.ndarray],
    use_ee_captions: bool,
    model: Optional[str],
) -> List[dict]:
    """Send keyframes + the labeling prompt to the VLM; return frame-indexed
    segments [{"start_frame","end_frame","primitive_label"}, ...].

    When ``plan`` is provided we use the ASSIGN_PLAN_TO_FRAMES prompt (fixed
    primitive order); otherwise the open-vocabulary LABEL_EPISODE_FROM_VIDEO
    prompt.
    """
    if plan:
        plan_str = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(plan))
        prompt = ASSIGN_PLAN_TO_FRAMES_PROMPT.format(
            task_description=task_description,
            plan_str=plan_str,
            episode_length=episode_length,
            frame_subsample=1,  # we pass real keyframes; captions carry the orig idx
            n_plan=len(plan),
        )
        if use_ee_captions and states is not None:
            prompt += ASSIGN_PLAN_EE_NOTE
    else:
        known_str = "\n".join(f"  - {p}" for p in known_primitives)
        prompt = LABEL_EPISODE_FROM_VIDEO_PROMPT.format(
            fps_sub=10,
            frame_subsample=1,
            task_description=task_description,
            known_str=known_str,
            episode_length=episode_length,
        )

    cam_labels = ["primary"] + list(extra_view_names)
    content: list = [{"type": "text", "text": prompt}]
    for k, orig_fi in enumerate(keyframe_orig_idx):
        ee_str = _ee_caption(states, orig_fi) if (use_ee_captions and states is not None) else ""
        # primary view
        caption = f"Frame {orig_fi} ({cam_labels[0]})"
        if ee_str:
            caption += ee_str
        content.append({"type": "text", "text": caption + ":"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_encode_frame_b64(keyframes[k])}"}})
        # extra views
        if extra_keyframes:
            for vi, vname in enumerate(extra_view_names):
                content.append({"type": "text", "text": f"Frame {orig_fi} ({vname}):"})
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{_encode_frame_b64(extra_keyframes[vi][k])}"}})

    resp = vlm.call([{"role": "user", "content": content}], model=model)
    try:
        result = _extract_json(resp)
    except Exception as e:  # noqa: BLE001
        logger.warning("Video-labeling JSON parse failed: %s. Falling back to even split.", e)
        result = {}

    segments = result.get("segments") or []

    # If a fixed plan was given, force count/labels; else accept VLM labels.
    if plan:
        if len(segments) != len(plan):
            step = episode_length / max(len(plan), 1)
            segments = [{"start_frame": int(i * step),
                         "end_frame": int((i + 1) * step),
                         "primitive_label": plan[i]} for i in range(len(plan))]
        else:
            for i, seg in enumerate(segments):
                seg["primitive_label"] = plan[i]

    if not segments:
        # Last-resort even split over the known primitives (or 4 chunks).
        labels = list(plan) if plan else list(known_primitives[:4]) or ["segment"]
        step = episode_length / len(labels)
        segments = [{"start_frame": int(i * step),
                     "end_frame": int((i + 1) * step),
                     "primitive_label": labels[i]} for i in range(len(labels))]

    # Enforce contiguity + full coverage in FRAME units.
    segments[0]["start_frame"] = 0
    segments[-1]["end_frame"] = episode_length
    for i in range(1, len(segments)):
        segments[i]["start_frame"] = segments[i - 1]["end_frame"]
    # Clamp + drop degenerate (zero/negative-length) segments.
    cleaned = []
    for seg in segments:
        sf = int(max(0, min(episode_length, seg["start_frame"])))
        efr = int(max(0, min(episode_length, seg["end_frame"])))
        if efr > sf:
            cleaned.append({"start_frame": sf, "end_frame": efr,
                            "primitive_label": str(seg.get("primitive_label", "segment"))})
    if not cleaned:
        cleaned = [{"start_frame": 0, "end_frame": episode_length, "primitive_label": "segment"}]
    cleaned[0]["start_frame"] = 0
    cleaned[-1]["end_frame"] = episode_length
    for i in range(1, len(cleaned)):
        cleaned[i]["start_frame"] = cleaned[i - 1]["end_frame"]
    return cleaned


def _refine_boundaries_state(
    segments: List[dict],
    states: Optional[np.ndarray],
    episode_length: int,
    refine_window: int,
) -> List[dict]:
    """Pass 2: snap each coarse boundary to a state-trajectory changepoint
    within a local window. Ported from assign_plan_to_frames() pass-2 (the
    state-changepoint refinement branch). FRAME units in/out.
    """
    if states is None or len(segments) < 2:
        return segments
    for i in range(len(segments) - 1):
        coarse_b = segments[i]["end_frame"]
        prev_start = segments[i]["start_frame"]
        next_end = segments[i + 1]["end_frame"]
        lo = max(prev_start + 1, coarse_b - refine_window)
        hi = min(next_end - 1, coarse_b + refine_window)
        if hi - lo < 4:
            continue
        splits = find_action_changepoints(
            states, lo, hi, n_splits=1,
            window=3, normalize_features=True, backtrack="none",
        )
        if splits:
            refined = int(min(max(splits[0], prev_start + 1), next_end - 1))
            segments[i]["end_frame"] = refined
            segments[i + 1]["start_frame"] = refined
    return segments


# ===========================================================================
# Public API.
# ===========================================================================
def _frames_to_steps(segments_frame: List[dict], frame_to_step: np.ndarray, n_steps: int) -> List[dict]:
    """Map frame-indexed segments -> step-indexed segments via ep["frame_to_step"]."""
    f2s = np.asarray(frame_to_step).reshape(-1)
    T = len(f2s)

    def map_idx(fi: int) -> int:
        fi = int(min(max(fi, 0), T - 1)) if T > 0 else 0
        if T == 0:
            return int(min(max(fi, 0), n_steps))
        return int(f2s[fi])

    out: List[dict] = []
    for seg in segments_frame:
        ss = map_idx(seg["start_frame"])
        # end_frame is exclusive in frame space; map the last included frame then +1,
        # but clamp to n_steps. For the final segment force end == n_steps.
        ef = seg["end_frame"]
        es = map_idx(ef - 1) + 1 if ef > 0 else map_idx(ef)
        es = int(min(max(es, ss), n_steps))
        out.append({"start_step": ss, "end_step": es, "label": str(seg.get("primitive_label", "segment"))})

    # Enforce contiguity + full coverage in STEP units.
    out = [s for s in out if s["end_step"] > s["start_step"]] or [{"start_step": 0, "end_step": n_steps, "label": "segment"}]
    out[0]["start_step"] = 0
    out[-1]["end_step"] = n_steps
    for i in range(1, len(out)):
        out[i]["start_step"] = out[i - 1]["end_step"]
    # Final pass: drop any segment that collapsed to zero length after stitching.
    out = [s for s in out if s["end_step"] > s["start_step"]]
    if not out:
        out = [{"start_step": 0, "end_step": n_steps, "label": "segment"}]
    out[0]["start_step"] = 0
    out[-1]["end_step"] = n_steps
    return out


def segment_episode(ep: dict, vlm, cfg: SegConfig) -> List[dict]:
    """Segment one episode into ordered primitive segments (step units).

    Returns ``[{"start_step": int, "end_step": int, "label": str}, ...]``
    covering the whole episode (contiguous, non-overlapping,
    start==0, end==ep["n_steps"]).

    Method via ``cfg.segment_method``:
      * "video"        — VLM keyframe labeling (INSIGHT primary path) + optional
                         EE-caption boundary refinement.
      * "gripper"      — segment_episode_by_gripper (grasp tasks).
      * "action_change"— find_action_changepoints on the EE trajectory
                         (gripperless tasks; cfg.has_gripper=False).
    """
    n_steps = int(ep["n_steps"])
    task = ep.get("task", "")
    method = cfg.segment_method

    # Auto-route gripperless tasks away from gripper segmentation.
    if not cfg.has_gripper and not ep.get("has_gripper", True) and method == "gripper":
        logger.info("has_gripper=False -> forcing action_change method.")
        method = "action_change"

    if method == "gripper":
        gr = np.asarray(ep["gripper"]).reshape(-1)
        ee_delta = np.asarray(ep["ee_delta"])
        triples = segment_episode_by_gripper(
            gr, ee_delta,
            min_segment_length=cfg.min_segment_length,
            gripper_motion_threshold=cfg.gripper_motion_threshold,
            gripper_close_threshold=cfg.gripper_close_threshold,
        )
        segs = [{"start_step": int(s), "end_step": int(e), "label": t} for s, e, t in triples]

    elif method == "action_change":
        # Changepoints on the EE pose trajectory (peaks align with visual arrivals).
        states = _ee_states_from_ep(ep)
        feat = states if states is not None else np.asarray(ep["ee_delta"])
        splits = find_action_changepoints(
            feat, 0, n_steps, n_splits=cfg.changepoint_n_splits,
            window=cfg.changepoint_window,
            normalize_features=cfg.changepoint_normalize,
            backtrack=cfg.changepoint_backtrack,
        )
        bounds = [0] + sorted(set(int(s) for s in splits if 0 < s < n_steps)) + [n_steps]
        segs = [{"start_step": bounds[i], "end_step": bounds[i + 1], "label": f"segment_{i}"}
                for i in range(len(bounds) - 1)]

    else:  # "video" — INSIGHT primary path
        view = cfg.primary_view if cfg.primary_view in ep["frames"] else _DEFAULT_PRIMARY_VIEW
        frames = np.asarray(ep["frames"][view])
        T = len(frames)
        kf_idx = _sample_keyframe_indices(T, cfg.num_keyframes)
        keyframes = [frames[i] for i in kf_idx]
        # extra views, sampled at the same indices when present
        extra_names = [v for v in cfg.extra_views if v in ep["frames"]]
        extra_keyframes = [[np.asarray(ep["frames"][v])[i] for i in kf_idx] for v in extra_names] if extra_names else None

        states = _ee_states_from_ep(ep) if cfg.use_ee_captions else None
        plan = list(cfg.plan) if cfg.plan else None

        seg_frame = _label_episode_from_video(
            vlm,
            keyframes=keyframes,
            keyframe_orig_idx=kf_idx,
            extra_keyframes=extra_keyframes,
            extra_view_names=extra_names,
            episode_length=T,
            task_description=task,
            plan=plan,
            known_primitives=list(cfg.known_primitives),
            states=states,
            use_ee_captions=cfg.use_ee_captions,
            model=cfg.model,
        )
        if cfg.refine_boundaries:
            # Refine using the per-frame state trajectory (resampled to frames).
            frame_states = _frame_aligned_states(ep, T) if cfg.use_ee_captions else None
            seg_frame = _refine_boundaries_state(seg_frame, frame_states, T, cfg.refine_window)
        return _frames_to_steps(seg_frame, ep["frame_to_step"], n_steps)

    # gripper / action_change already produced step-indexed segments; normalize.
    segs = [s for s in segs if s["end_step"] > s["start_step"]] or [{"start_step": 0, "end_step": n_steps, "label": "segment"}]
    segs[0]["start_step"] = 0
    segs[-1]["end_step"] = n_steps
    for i in range(1, len(segs)):
        segs[i]["start_step"] = segs[i - 1]["end_step"]
    return segs


def _frame_aligned_states(ep: dict, T: int) -> Optional[np.ndarray]:
    """Project the per-STEP EE pose onto the per-FRAME timeline via frame_to_step.

    The boundary-refine pass operates in frame units, so we need an EE pose
    array indexed by video frame. We gather step poses at each frame's step.
    """
    states = _ee_states_from_ep(ep)
    if states is None:
        return None
    f2s = np.asarray(ep["frame_to_step"]).reshape(-1)
    if len(f2s) == 0:
        return states
    idx = np.clip(f2s[:T].astype(int), 0, len(states) - 1)
    return states[idx]


# ===========================================================================
# Annotated-video rendering.
# Ported from densely_label_dataset.py :: annotate_frame() (PIL text overlay),
# re-homed onto cv2 for mp4 writing per the task spec.
# ===========================================================================
def annotate_frame(img: np.ndarray, primitive_label: str, frame_idx: int,
                   total_frames: int, task_description: Optional[str] = None) -> np.ndarray:
    """Add a label/frame-counter/task banner to a frame (RGB uint8 in/out).

    Ported from densely_label_dataset.py :: annotate_frame() (PIL).
    """
    from PIL import Image, ImageDraw, ImageFont  # lazy

    pil_img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    img_width, img_height = pil_img.size
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:  # noqa: BLE001
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    label = (primitive_label or "").strip("\"'")
    padding = 6
    top_height = 30
    bottom_height = 28 if task_description else 0
    new_img = Image.new("RGB", (img_width, img_height + top_height + bottom_height), color=(30, 30, 40))
    new_img.paste(pil_img, (0, top_height))
    draw = ImageDraw.Draw(new_img)

    draw.text((padding, padding), label, fill=(255, 200, 0), font=font_large)
    frame_text = f"{frame_idx}/{total_frames}"
    bbox = draw.textbbox((0, 0), frame_text, font=font_small)
    text_width = bbox[2] - bbox[0]
    draw.text((img_width - text_width - padding, padding + 5), frame_text, fill=(200, 200, 200), font=font_small)

    if task_description:
        task_text = f"Task: {task_description}"
        max_width = img_width - 2 * padding
        line_height = 12
        words = task_text.split()
        lines: List[str] = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_bbox = draw.textbbox((0, 0), test_line, font=font_small)
            if test_bbox[2] - test_bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        for i, line in enumerate(lines[:2]):
            draw.text((padding, img_height + top_height + 2 + i * line_height), line, fill=(150, 200, 255), font=font_small)

    return np.array(new_img)


def write_annotated_video(ep: dict, segments: List[dict], out_path: str,
                          view: Optional[str] = None, fps: int = 10) -> str:
    """Render an mp4 with per-frame primitive-label overlay (cv2 writer).

    Segments are step-indexed; each video frame's step (via frame_to_step) picks
    the active segment label. Returns ``out_path``.
    """
    import cv2  # lazy

    view = view or (_DEFAULT_PRIMARY_VIEW if _DEFAULT_PRIMARY_VIEW in ep["frames"] else next(iter(ep["frames"])))
    frames = np.asarray(ep["frames"][view])
    T = len(frames)
    f2s = np.asarray(ep["frame_to_step"]).reshape(-1)
    task = ep.get("task", "")

    def label_for_step(step: int) -> str:
        for seg in segments:
            if seg["start_step"] <= step < seg["end_step"]:
                return seg["label"]
        return segments[-1]["label"] if segments else "segment"

    writer = None
    try:
        for fi in range(T):
            step = int(f2s[fi]) if fi < len(f2s) else (segments[-1]["end_step"] - 1 if segments else 0)
            annotated = annotate_frame(frames[fi], label_for_step(step), fi, T, task)
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()
    return out_path
