"""Env-step 해상도 phase/성공 GT 수집 (record 단위 라벨과 병행).

배경(docs/steering/18 §논의 2026-07-10): 기존 `feature_phases`는 get_action(=n_action_steps
env-step)당 1회만 sim 상태를 평가한다 — 이벤트 시점이 ±n_action_steps로 양자화되고,
labeler debounce(grasp_hold)가 의도(env-step)와 달리 record 스케일로 동작한다.
이 모듈은 **별도 라벨러 인스턴스**를 매 env-step 후 호출해 env-step GT를 추가 저장한다.
기존 record 단위 라벨·gated 스위칭 경로는 건드리지 않는다(activation 정합·하위호환 보존).

정합(alignment) 규약:
- `env_step_phases[k]` = k번의 env.step 실행 후 상태 s_k 의 라벨 (k=0 = reset 직후 s_0).
  길이 = 실행된 env-step 수 + 1.
- feature record r 가 본 상태 = s_{n_action_steps*r} → `env_step_phases[n_action_steps*r]`
  가 `feature_phases[r]` 에 대응. 단 debounce 스케일이 달라(2 env-step vs 2 record) 이벤트
  경계 부근에서 소수 불일치가 정상적으로 존재 → QA 용 `env_step_record_mismatch` 카운트 저장.
- `env_step_success[k]` = s_{k+1} (k번째 step 직후) 의 `_check_success()` (길이 = step 수).
"""

from __future__ import annotations

import gymnasium as gym

from src.collect.robocasa.event_labeler import find_robocasa_env, make_robocasa_event_labeler


class StepPhaseProbeWrapper(gym.Wrapper):
    """raw env 바로 위에 끼워 매 env.step 직후 EnvStepGT 콜백을 호출.

    MultiStepWrapper 가 chunk 를 내부 루프로 풀기 때문에, 그 아래층인 이 wrapper 만이
    개별 env-step 을 관측할 수 있다.
    """

    def __init__(self, env):
        super().__init__(env)
        self._gt = None

    def set_gt(self, gt) -> None:
        self._gt = gt

    def step(self, action):
        out = self.env.step(action)
        if self._gt is not None:
            self._gt.on_step()
        return out


def find_probe_wrapper(env) -> StepPhaseProbeWrapper | None:
    e = env
    while e is not None:
        if isinstance(e, StepPhaseProbeWrapper):
            return e
        e = getattr(e, "env", None)
    return None


class EnvStepGT:
    """env-step 해상도 GT 기록기 (자체 라벨러 인스턴스 — record 라벨러와 상태 분리)."""

    def __init__(self, env, env_name: str, proximity_phases: bool):
        self._kenv = find_robocasa_env(env)  # gymnasium unwrapped 는 kitchen 에 못 닿음
        self._labeler = make_robocasa_event_labeler(
            env, env_name, proximity_phases=proximity_phases
        )
        self.phases: list[str] = []
        self.success: list[bool] = []

    def start(self) -> None:
        """reset 직후(s_0) 호출."""
        self._labeler.reset()
        self.phases = [self._labeler.step()]  # s_0
        self.success = []

    def on_step(self) -> None:
        """매 env.step 직후(s_k) — StepPhaseProbeWrapper 가 호출."""
        self.phases.append(self._labeler.step())
        try:
            self.success.append(bool(self._kenv._check_success()))
        except Exception:
            self.success.append(False)

    def export(self, feature_phases: list[str] | None, n_action_steps: int) -> dict:
        out = {
            "env_step_phases": list(self.phases),
            "env_step_success": list(self.success),
            "env_step_event_steps": dict(getattr(self._labeler, "event_steps", {}) or {}),
            "env_step_grasp_steps": list(getattr(self._labeler, "grasp_steps", []) or []),
            "env_step_drop_steps": list(getattr(self._labeler, "drop_steps", []) or []),
            "env_step_wrong_grasp_steps": list(
                getattr(self._labeler, "wrong_grasp_steps", []) or []
            ),
            "env_step_n_action_steps": int(n_action_steps),
        }
        # fixture 태스크(drawer/fridge) 진단용 관절 열림도 궤적 (없는 라벨러는 생략)
        door_tl = getattr(self._labeler, "door_timeline", None)
        if door_tl:
            out["env_step_door_state"] = [float(v) for v in door_tl]
        # 성공을 지배하는 값(전체 도어 중 목표에서 가장 먼 것) — near-miss 사후 집계용.
        worst_tl = getattr(self._labeler, "door_worst_timeline", None)
        if worst_tl:
            out["env_step_door_worst"] = [float(v) for v in worst_tl]
        if feature_phases is not None:
            mism = sum(
                1
                for r, p in enumerate(feature_phases)
                if r * n_action_steps < len(self.phases)
                and self.phases[r * n_action_steps] != p
            )
            out["env_step_record_mismatch"] = int(mism)
        return out
