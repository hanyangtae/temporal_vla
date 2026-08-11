"""velocity 합성(RL²-VLA식 policy mixing) hook — **인터페이스 스텁, 미구현**.

구현 세션을 위한 배선 설계 기록 (2026-08-10, 검토 세션). 배경·재현 결과는
Notion "RL-2" 페이지(3b363918d42a80be88b6ead611406ccd) — velocity 합성은 재현 확정,
게이팅은 검출기-플랫폼 정합에 민감. 우리 이식의 요지: **합성은 그대로, "언제"(게이트)만
우리 online phase readout 으로 교체**하는 비교가 인자 하나 차이로 성립한다.

── velocity 는 어디서 태어나나 ────────────────────────────────────────────────
`lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py`
`get_action()` (≈346행)의 K=4 denoise 루프:

    for t in range(num_steps):
        model_output = self.model(...)            # DiT — 기존 캡처·steering hook 지점
        pred = self.action_decoder(model_output)  # ★ hidden → velocity 해독 (MLP)
        pred_velocity = pred[:, -self.action_horizon:]
        actions = actions + dt * pred_velocity    # 적분 (denoise step)

기존 인프라는 전부 `self.model` 안(residual/pre-velocity hidden)까지만 만진다.
velocity 자체는 아무도 안 본다 — **개입 지점은 `head.action_decoder` 의 forward hook**.
출력 `pred` 를 받아 마지막 `action_horizon` 토큰 슬라이스에만 합성을 적용하면
upstream(서브모듈) 수정 없이 끝난다. RL² 원본처럼 샘플링 루프를 고치는 방식은
서브모듈 수정 최소화 원칙과 충돌하므로 쓰지 않는다.

── hook 계약 (steering_hooks 와 폴리모픽 호환 — registry 순회가 요구) ─────────
- `reset_step_counter()`: serve 가 요청 시작마다 호출 (denoise call index _k 초기화).
- fail-loud: _k 가 K 를 넘는 발화는 RuntimeError (무음 오적용 방지).
- gated on/off: `set_alpha(None)` 이면 출력 텐서를 **그대로 반환** (clone 금지) —
  off≡identity 가 구조적으로 성립해야 함 (SetpointSteering 규약 동일).
- steering_hooks 에 넣지 않는 이유: 그 파일은 "단일 모델 latent 상수 연산" 전용.
  여기는 두 번째 모델(QAM) forward 가 매 call 발생 — 로드·VRAM·seed 관리 동반.

── serve 배선 (lerobot.py) ──────────────────────────────────────────────────
- `_register_compose_if_requested(loaded_policy, args)` 를
  `_register_steering_if_requested` 옆에 나란히 추가.
- CLI(안): `--compose-qam-ckpt <path> --compose-alpha <float>`
  ★ alpha 는 사후선택 없이 고정값을 primary 로 (재현 실측: alpha ±0.05 에 ±5pp 요동).
- `/health` steering 지문(`_steering_spec`)에 op="qam_compose"·alpha·qam ckpt sha 노출
  → 수집기 `_resolve_arm` 이 armsig 를 자동 계산 (ARM_PARAM_KEYS 그대로 들어맞음).
- 게이트 실험: `/steering_phase` gated 스위칭 + `_phase_readouts`(online phase) 재사용.

── 이식 결정 대기 (구현 전 필요) ─────────────────────────────────────────────
- RoboCasa 용 QAM 체크포인트 없음 → 재학습 필요 (학습 데이터 구성 미정).
- verifier(CoVer) 재학습 여부 · 평가 task set.
"""
from __future__ import annotations

from typing import Any


class VelocityCompose:
    """`head.action_decoder` forward hook — v' = v + α·(v_QAM − v).

    미구현 스텁. 위 모듈 docstring 의 계약을 따를 것.
    """

    def __init__(self, groot_model: Any, qam_model: Any, alpha: float | None):
        raise NotImplementedError(
            "velocity 합성은 인터페이스만 설계됨 — 모듈 docstring 의 배선 계획 참조"
        )
