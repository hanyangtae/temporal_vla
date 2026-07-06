"""Segmentation system prompts ported from the INSIGHT VLA project.

PROVENANCE
----------
Source repo : INSIGHT — Self-Guided Skill Acquisition via Steerable VLAs
              https://github.com/insight-vla/insight
License     : Apache License 2.0 (see INSIGHT repo LICENSE / NOTICE).
Copyright   : (c) 2026 The InSight Authors.

The string constants below are COPIED VERBATIM from the released INSIGHT code
(only re-homed into named module constants; the prompt text itself is
unchanged, including all ``{...}`` runtime placeholders that the callers fill
via ``str.format`` / f-string substitution).

Origins inside the INSIGHT repo:
  * ``LABEL_EPISODE_FROM_VIDEO_PROMPT``  — body of
        sim/libero_flywheel/data_processing/densely_label_dataset.py
        :: label_episode_from_video()
  * ``SPLIT_MERGED_SEGMENT_PROMPT``      — body of
        densely_label_dataset.py :: split_merged_segment_from_video()
  * ``ASSIGN_PLAN_TO_FRAMES_PROMPT`` / ``ASSIGN_PLAN_EE_NOTE`` — body of
        densely_label_dataset.py :: assign_plan_to_frames()  (coarse pass)
  * ``REFINE_BOUNDARY_PROMPT``           — body of
        densely_label_dataset.py :: _refine_boundary()
  * ``PLAN_TASK_SYSTEM``                 — verbatim from
        src/insight/prompts.py :: PLAN_TASK_SYSTEM  (plan decomposition that
        feeds the video labeler with a fixed primitive_sequence).

Keep the ``{...}`` placeholders intact — the segmentation code substitutes:
  - LABEL_EPISODE_FROM_VIDEO_PROMPT: {fps_sub}, {frame_subsample}, {task_description}, {known_str}, {episode_length}
  - SPLIT_MERGED_SEGMENT_PROMPT:     {task_description}, {seg_start}, {seg_length}, {seg_end}, {labels_str}
  - ASSIGN_PLAN_TO_FRAMES_PROMPT:    {task_description}, {plan_str}, {episode_length}, {frame_subsample}, {n_plan}
  - REFINE_BOUNDARY_PROMPT:          {task_description}, {prev_label}, {next_label}
  - PLAN_TASK_SYSTEM:                {scene_context}, {primitives}  (literal JSON braces escaped as {{ }})
"""

# ---------------------------------------------------------------------------
# Primary video -> primitive-sequence labeling prompt.
# VERBATIM from densely_label_dataset.py :: label_episode_from_video().
# In the original this is an f-string; the substituted expressions are exposed
# here as named {placeholders}:
#   {fps_sub}         <- (10 // frame_subsample)
#   {frame_subsample} <- frame_subsample
#   {task_description}<- task_description
#   {known_str}       <- "\n".join(f"  - {p}" for p in known_primitives)
#   {episode_length}  <- episode_length
# ---------------------------------------------------------------------------
LABEL_EPISODE_FROM_VIDEO_PROMPT = """You are labeling a robot manipulation video. The video is at {fps_sub}fps (subsampled from 10fps, showing every {frame_subsample}th frame).
The original video is 10fps, so frame N in these images = frame N*{frame_subsample} in the original.

Task: "{task_description}"

Known primitive types:
{known_str}

For each primitive action in the video, identify:
1. The primitive label (from the known list, or create a new one if needed)
2. The start frame number (in ORIGINAL frame numbers, i.e., multiply the image index by {frame_subsample})
3. The end frame number (in ORIGINAL frame numbers)

Rules:
- Every frame must belong to exactly one primitive (no gaps, no overlaps)
- The last segment's end_frame should be {episode_length}
- For flipping tasks: use [move gripper to object, close gripper, lift upward, rotate block, open gripper]
- Short gripper transitions (close/open) may be just a few frames

Return JSON only: {{"segments": [{{"start_frame": 0, "end_frame": N, "primitive_label": "..."}}, ...]}}"""


# ---------------------------------------------------------------------------
# Split a merged (e.g. gripper-closed) segment into ordered sub-primitives.
# VERBATIM from densely_label_dataset.py :: split_merged_segment_from_video().
#   {task_description} <- task_description
#   {seg_start}        <- seg_start
#   {seg_length}       <- seg_end - seg_start
#   {seg_end}          <- seg_end
#   {labels_str}       <- ", ".join(f'"{l}"' for l in labels)
# ---------------------------------------------------------------------------
SPLIT_MERGED_SEGMENT_PROMPT = """This video shows a segment of a robot doing: "{task_description}"
The video is at 10fps. Frame 0 in the video = frame {seg_start} in the original episode. The segment is {seg_length} frames long (frames {seg_start}-{seg_end}).

This segment contains these primitives in order: {labels_str}

For each primitive, identify the start and end frame numbers in the ORIGINAL episode numbering (starting from {seg_start}).
Each primitive must be contiguous, cover the full segment, and appear in the order listed.
Watch the video carefully for changes in motion — a new primitive begins when the character of the motion changes.

Return JSON only: {{"segments": [{{"start_frame": N, "end_frame": N, "primitive_label": "..."}}, ...]}}"""


# ---------------------------------------------------------------------------
# Coarse two-pass plan->frame localization prompt (pass 1).
# VERBATIM from densely_label_dataset.py :: assign_plan_to_frames().
#   {task_description} <- task_description
#   {plan_str}         <- "\n".join(f"  {i+1}. {p}" for i, p in enumerate(plan))
#   {episode_length}   <- episode_length
#   {frame_subsample}  <- frame_subsample
#   {n_plan}           <- len(plan)
# ---------------------------------------------------------------------------
ASSIGN_PLAN_TO_FRAMES_PROMPT = """You are localizing primitives in a robot manipulation video.

Task: "{task_description}"

## Plan (fixed, in order; each primitive appears exactly ONCE):
{plan_str}

## Video
{episode_length} total frames. Frames shown below are sampled every {frame_subsample}th frame, in order.
The image labeled "Frame N" corresponds to ORIGINAL frame N.

## What to do
1. For each primitive in the plan, briefly state TWO signatures you will look for:
   (a) Visual signature — what the end-effector is doing, what contact or direction characterizes it.
   (b) Motion signature on the per-frame Δ data — which translation/rotation axis dominates and in which direction. Derive this yourself from the primitive's name and task context (e.g. "lower" implies negative Δdz dominant; "move to X" implies |Δxy| dominant; "twist" implies |Δrz| dominant; "open/close gripper" implies no EE motion). DO NOT rely on a fixed table — reason from the primitive's name.
2. Walk through the sampled frames in order. For each, decide which primitive it best matches, cross-checking BOTH the image and the Δ-signature in the caption.
3. Output the boundary ORIGINAL frame numbers between primitives.

## Boundary rule (read carefully)
The boundary frame is the first frame that clearly belongs to the NEW primitive — i.e., if someone saw ONLY that frame, they would confidently assign it to the new primitive, not the previous one. A boundary is NOT where the new primitive is already well underway (too late) and NOT where it is barely visible (too early). Pick the earliest unambiguous frame.

The boundary should align with a SHIFT IN THE DOMINANT MOTION AXIS in the per-frame Δ data — the axis that was sustained during primitive A approaches zero or reverses sign, and the axis you expect for primitive B starts to dominate. Use this as a kinematic anchor; do not place a boundary in a region where the dominant axis still matches A's expected signature.

## Constraints
- Exactly {n_plan} segments, one per plan primitive, in order.
- Segments must be contiguous (no gaps/overlaps).
- First segment starts at 0, last ends at {episode_length}.
- Boundary frames must be ORIGINAL frame numbers from the sampled set (multiples of {frame_subsample}, plus 0 and {episode_length}).

## Output format — IMPORTANT
Output the JSON block FIRST, then any reasoning after. This guarantees a parseable answer even if your response is truncated.

```json
{{"segments": [{{"start_frame": 0, "end_frame": N, "primitive_label": "..."}}, ...]}}
```
After the JSON, you may add brief reasoning (1–3 sentences per boundary)."""


# Companion note appended to ASSIGN_PLAN_TO_FRAMES_PROMPT when per-frame EE
# captions are supplied. VERBATIM from assign_plan_to_frames() (the
# ``extra_note`` string). No runtime placeholders.
ASSIGN_PLAN_EE_NOTE = (
    "\n\nEach keyframe caption below includes the EE (end-effector) pose, "
    "per-frame delta, derived magnitudes (|Δxy|, |Δz|, |Δrxy|, |Δrz|), and "
    "a `dom=` tag identifying the dominant motion axis among "
    "{+z, -z, xy, +rz, -rz, rxy, none}. Use these alongside the images — "
    "they are precise and unambiguous, while images can look alike across "
    "primitives. A clear shift in `dom` between consecutive frames is a "
    "strong boundary cue."
)


# ---------------------------------------------------------------------------
# Boundary refinement (frame-level zoom-in) prompt.
# VERBATIM from densely_label_dataset.py :: _refine_boundary().
#   {task_description} <- task_description
#   {prev_label}       <- prev_label
#   {next_label}       <- next_label
# (The "Frames provided cover range [lo, hi)." line is appended at call time by
#  the segmentation code, matching the original.)
# ---------------------------------------------------------------------------
REFINE_BOUNDARY_PROMPT = """You are pinpointing the exact transition frame between two primitives in a robot demo.

Task: "{task_description}"
Previous primitive (before): "{prev_label}"
Next primitive (after): "{next_label}"

Each frame below provides (a) camera view(s) and (b) the end-effector pose plus per-frame delta (Δdx, Δdy, Δdz, Δrx, Δry, Δrz) in world coordinates.

A primitive boundary is, by definition, a point where the character of the robot's motion changes. Examples of motion-character changes (task-agnostic):
  - A component of velocity that was sustained approaches zero, or reverses sign.
  - The dominant axis of motion shifts from one to another (e.g. translation-dominant → rotation-dominant, or one translation axis → another).
  - A velocity magnitude rises or drops substantially after being steady.
  - An axis that had zero motion starts moving.

Procedure — THINK EXPLICITLY IN THIS ORDER:
1. Read the per-frame EE deltas across all shown frames. Identify the frame where the MOTION CHARACTER first changes — this is your EE-based boundary candidate. State which axis/quantity changed and at what frame.
2. Look at the visual frames. Identify the earliest frame that a person would confidently describe as the next primitive (not still ambiguous with the previous one).
3. Reconcile the two candidates. If they agree within ±2 frames, pick the EE-based frame (physical motion is precise). If they disagree by more, pick the frame where the EE-based change begins AND the visual matches — explain the disagreement in reasoning.

The boundary frame IS the first frame of the new primitive — at that frame the robot's motion is characteristically the new primitive.

## Output format — IMPORTANT
Output the JSON FIRST, then brief reasoning. This guarantees a parseable answer even if truncated.

```json
{{"boundary_frame": <int>, "reasoning": "<reference specific frame numbers, the EE axis/quantity that changed, and the visual confirmation>"}}
```
"""


# ---------------------------------------------------------------------------
# Plan decomposition system prompt.
# VERBATIM from src/insight/prompts.py :: PLAN_TASK_SYSTEM.
# Placeholders {scene_context} and {primitives} are filled by the caller via
# .format(); literal JSON braces are escaped as {{ }}.
# ---------------------------------------------------------------------------
PLAN_TASK_SYSTEM = """\
You are a robot task planner. Scene: {scene_context}

AVAILABLE PRIMITIVES (each is general-purpose and adapts to context):
{primitives}

RULES:
1. Break the goal into fine-grained steps. Use existing primitives for every
   sub-step they cover — a skill gap should only be the novel part, not a
   bundle of existing + novel actions.
2. Only create a skill gap when the desired outcome is fundamentally different
   from what any existing primitive produces. If an existing primitive could
   achieve the same result (even if executed differently), use it and put
   execution details in step_notes instead.
3. Every step goes in primitive_sequence — including new ones.
4. New primitives also go in skill_gaps (must appear in BOTH lists).
5. Name new primitives by their desired EFFECT, not the robot motion.
6. For each step, add a note on execution (approach, grasp, how it enables the next step).
7. After the final step, the runtime returns the gripper to a safe home
   pose, so the gripper does not need to be cleared from the workspace by
   a final step in the plan. Each step should make a distinguishable
   contribution to the goal — avoid adding a final step whose only effect
   is repositioning the gripper.
8. Each skill gap is one single-axis motion (one translation OR one rotation
   along one axis, in one direction). If the goal involves multiple distinct
   motions, create a separate skill gap for each.

Example 1 — pick and place (all existing, no skill gaps):
  primitive_sequence: ["move gripper to the red lego block", "close gripper", "lift upward", "move gripper to target", "lower gripper", "open gripper"]
  skill_gaps: []

Example 2 — inserting an object (one new skill gap):
  primitive_sequence: ["move gripper to object", "close gripper", "lift upward", "move gripper to target", "insert object into slot", "open gripper"]
  skill_gaps: ["insert object into slot"]

Respond with ONLY valid JSON:
{{"primitive_sequence": ["step1", "step2", ...],
  "step_notes": ["execution note for each step"],
  "skill_gaps": ["new primitives not in available list"],
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0,
  "requires_new_primitive": true or false}}"""


# Default known-primitive vocabulary used when no plan is supplied. Mirrors the
# fine-grained manipulation primitives referenced throughout INSIGHT's
# densely_label_dataset.py examples.
DEFAULT_KNOWN_PRIMITIVES = [
    "move gripper to object",
    "close gripper",
    "lift upward",
    "move gripper to target",
    "lower gripper",
    "open gripper",
    "rotate object",
    "push object",
    "pull object",
]
