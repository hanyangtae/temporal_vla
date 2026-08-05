"""Frame-aligned robot-state recording for GR00T robocasa collection.

WHY
---
SAFE rollouts store one activation/state record per *policy inference* (= per
``get_action`` in ``src/collect/policy_clients.py``). For an N1.6 robocasa episode
that executes a 16-step action chunk per inference, that is e.g. **16 records**
for a ~256-env-step / 128-video-frame episode. That granularity is right for
steering/conceptor analysis (one hidden-state per inference), but too coarse for
INSIGHT-style segmentation, whose EE-caption + boundary refinement want a robot
state trajectory aligned with the *video frames*.

The env steps are unrolled (and frames rendered) inside the GR00T upstream
``VideoRecordingWrapper`` (``Isaac-GR00T/gr00t/eval/sim/wrapper/
video_recording_wrapper.py``). Its ``step()`` runs once per **env step** and
writes a frame when ``step_count % steps_per_render == 0``. This module mirrors
that exact cadence so the captured state list is **1:1 with the video frames**.

WHAT THIS GIVES YOU
-------------------
A per-frame proprio trajectory (eef_pos / eef_quat / gripper / base ...) of the
same length as the rendered video (e.g. 128 points), instead of the 16 inference
points. Segmentation can then build faithful per-frame EE captions and snap
boundaries precisely; ``rollout_adapter.load_episode`` would prefer these
``frame_states`` and set ``frame_to_step`` to identity.

INTEGRATION (collection side — runs in the robocasa Docker)
-----------------------------------------------------------
Option A (recommended, NO submodule edit) — monkey-patch the upstream wrapper:

    from state_recording_wrapper import enable_on_video_wrapper, get_frame_states
    enable_on_video_wrapper()            # call ONCE before building the env
    ...                                   # run the episode as usual
    frame_states = get_frame_states(env)  # list[dict], one per video frame
    payload["frame_states"] = frame_states  # save into the rollout pkl

Option B — insert ``StateRecordingWrapper`` next to ``VideoRecordingWrapper`` in
the wrapper stack (one edit in ``rollout_policy.py`` between the
VideoRecordingWrapper and MultiStepWrapper lines):

    env = VideoRecordingWrapper(env, ...)
    env = StateRecordingWrapper(env, steps_per_render=cfg.video.steps_per_render)
    env = MultiStepWrapper(env, ...)

CAVEAT
------
This only affects FUTURE collection — existing rollouts keep their 16 points and
cannot be upsampled. And it improves EE-caption fidelity + boundary precision
only; the VLM *video* labeling already runs at full frame resolution, so the
primitive labels themselves barely change. Re-collect only if you need fine
boundaries / faithful captions.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

try:  # real gymnasium in production / the robocasa Docker
    import gymnasium as gym
    _WrapperBase = gym.Wrapper
except Exception:  # minimal shim so this module imports & tests anywhere
    class _WrapperBase:  # type: ignore
        def __init__(self, env):
            self.env = env

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

        def __getattr__(self, name):
            return getattr(self.env, name)


StateExtractor = Callable[[dict], dict]


def default_proprio_extractor(obs: dict) -> dict:
    """Pull the proprio/state entries out of an observation dict.

    Robust to native (robomimic ``robot0_eef_pos`` ...) vs unified
    (``observation.state.eef_pos_rel`` ...) key naming: it keeps every key that
    is NOT an image/video frame and NOT a language annotation. Inspect the
    captured keys on the first real run and narrow this if desired.
    """
    out: dict[str, Any] = {}
    for k, v in obs.items():
        kl = str(k).lower()
        if "video" in kl or "image" in kl or "rgb" in kl:
            continue
        if kl.startswith("annotation.") or kl.startswith("language."):
            continue
        try:
            arr = np.asarray(v)
            # skip anything image-shaped that slipped through
            if arr.ndim >= 3:
                continue
            out[k] = arr.copy()
        except Exception:
            out[k] = v
    return out


class StateRecordingWrapper(_WrapperBase):
    """Record robot proprio state at the same cadence the video is rendered.

    Mirrors ``VideoRecordingWrapper`` exactly: ``reset`` sets ``step_count = 1``;
    ``step`` increments first, then captures when
    ``step_count % steps_per_render == 0``. Placed at the same level in the
    wrapper stack (below MultiStepWrapper), this yields one state per video
    frame, in order.

    Alignment: buffer index ``i`` is 1:1 with video frame ``i``. (It is captured
    from the obs after env step ``frame_steps[i] - 1`` — step_count starts at 1
    and increments after the env step — but you only need the index 1:1 mapping
    to attach state to frames.)
    """

    def __init__(
        self,
        env,
        steps_per_render: int = 2,
        state_extractor: Optional[StateExtractor] = None,
    ):
        super().__init__(env)
        self.steps_per_render = int(steps_per_render)
        self.state_extractor = state_extractor or default_proprio_extractor
        self.step_count = 0
        self._episode_states: list[dict] = []
        self._frame_steps: list[int] = []

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._episode_states = []
        self._frame_steps = []
        self.step_count = 1  # parity with VideoRecordingWrapper.reset
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if (self.step_count % self.steps_per_render) == 0:
            obs = result[0]
            self._episode_states.append(self.state_extractor(obs))
            self._frame_steps.append(self.step_count)
        return result

    # --- retrieval (call at end of episode, before the next reset) ---
    def get_episode_states(self) -> list[dict]:
        return list(self._episode_states)

    def get_frame_steps(self) -> list[int]:
        return list(self._frame_steps)


# ---------------------------------------------------------------------------
# Option A: monkey-patch the upstream VideoRecordingWrapper (no submodule edit)
# ---------------------------------------------------------------------------
def enable_on_video_wrapper(state_extractor: Optional[StateExtractor] = None) -> None:
    """Augment the upstream ``VideoRecordingWrapper`` to also buffer per-frame
    proprio, captured from the *same* obs and at the *same* cadence as each
    written frame (so alignment is exact, on the same wrapper instance).

    Call once, before the env is built, from collection code that runs inside
    the robocasa Docker (where ``gr00t`` is importable). Idempotent.
    """
    from gr00t.eval.sim.wrapper.video_recording_wrapper import (  # lazy
        VideoRecordingWrapper,
    )

    if getattr(VideoRecordingWrapper, "_frame_state_patched", False):
        return
    extractor = state_extractor or default_proprio_extractor
    orig_reset = VideoRecordingWrapper.reset
    orig_step = VideoRecordingWrapper.step

    def reset(self, **kwargs):
        self._frame_states = []  # noqa: SLF001
        return orig_reset(self, **kwargs)

    def step(self, action):
        result = orig_step(self, action)
        # orig_step already incremented step_count and wrote the frame iff this
        # condition holds — mirror it so we buffer state for exactly that frame.
        if self.file_path is not None and (self.step_count % self.steps_per_render) == 0:
            if not hasattr(self, "_frame_states"):
                self._frame_states = []
            self._frame_states.append(extractor(result[0]))
        return result

    VideoRecordingWrapper.reset = reset
    VideoRecordingWrapper.step = step
    VideoRecordingWrapper._frame_state_patched = True


def get_frame_states(env) -> list[dict]:
    """Walk the wrapper chain and return buffered per-frame states.

    Works for both the monkey-patch path (``_frame_states`` on the
    VideoRecordingWrapper) and the explicit ``StateRecordingWrapper``.
    """
    node = env
    seen = 0
    while node is not None and seen < 64:
        if isinstance(node, StateRecordingWrapper):
            return node.get_episode_states()
        fs = getattr(node, "_frame_states", None)
        if fs is not None:
            return list(fs)
        node = getattr(node, "env", None)
        seen += 1
    return []


# ---------------------------------------------------------------------------
# Self-test (runnable without robocasa): verifies frame/state cadence alignment
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import gymnasium as gym  # available in lerobot_safe

    class FakeProprioEnv(gym.Env):
        """Emits an obs dict with a video frame + proprio + annotation, like the
        robocasa env seen by VideoRecordingWrapper."""

        metadata: dict = {}

        def __init__(self, n=256):
            super().__init__()
            self.n = n
            self.t = 0
            self.observation_space = gym.spaces.Dict({})
            self.action_space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)

        def _obs(self):
            return {
                "video.cam0": np.zeros((4, 4, 3), np.uint8),
                "observation.state.eef_pos_rel": np.array(
                    [self.t * 0.01, 0.0, 0.5], np.float32),
                "observation.state.gripper_qpos": np.array([0.02, -0.02], np.float32),
                "annotation.human.task_description": "pick the thing",
            }

        def reset(self, **kw):
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            self.t += 1
            return self._obs(), 0.0, False, self.t >= self.n, {"success": False}

    spr = 2
    n_steps = 256
    env = StateRecordingWrapper(FakeProprioEnv(n=n_steps), steps_per_render=spr)
    env.reset()
    for _ in range(n_steps):
        env.step(env.action_space.sample())

    states = env.get_episode_states()
    fsteps = env.get_frame_steps()
    expected_frames = n_steps // spr
    expected_steps = list(range(spr, n_steps + 1, spr))  # mirror Video wrapper

    assert len(states) == expected_frames, (len(states), expected_frames)
    assert fsteps == expected_steps, (fsteps[:5], expected_steps[:5])
    # captured proprio keys are the state ones, not video/annotation
    assert set(states[0].keys()) == {
        "observation.state.eef_pos_rel",
        "observation.state.gripper_qpos",
    }, states[0].keys()
    # Alignment invariant: buffer index i is captured from the obs AFTER env step
    # (fsteps[i]-1) — because step_count starts at 1 and increments after the env
    # step (mirroring VideoRecordingWrapper). The fake env sets x = env_t * 0.01.
    def x(i):
        return float(states[i]["observation.state.eef_pos_rel"][0])
    assert abs(x(0) - (fsteps[0] - 1) * 0.01) < 1e-6, x(0)          # first frame
    assert abs((x(11) - x(10)) - spr * 0.01) < 1e-6, (x(10), x(11))  # frames spr apart
    assert all(x(i + 1) > x(i) for i in range(len(states) - 1))       # monotone in time

    print(f"OK: {n_steps} env steps, spr={spr} -> {len(states)} frame-aligned "
          f"states (frames at env-steps {[s-1 for s in fsteps[:3]]}.."
          f"{[s-1 for s in fsteps[-2:]]}); keys={sorted(states[0].keys())}")
    print("Frame i <-> state i alignment exact; default extractor dropped "
          "video/annotation keys.")


if __name__ == "__main__":
    _selftest()
