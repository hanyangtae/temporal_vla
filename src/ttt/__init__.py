"""
TTT (Test-Time Training) 기반 실패 루프 탈출 모듈

성공 데이터로만 학습된 VLA 모델이 실패 시 같은 trajectory를 반복하는 문제를,
외부 TTT 모듈을 통해 VLA 백본 추가학습 없이 해결한다.

2단계 학습:
    Phase 1 — Progress Predictor 학습:
        TTTModule + ProgressHead를 expert trajectory에서 학습.
        label: y_t = t/T (normalized timestep). TTT inner loop이
        temporal context를 파라미터에 축적하고, ProgressHead가 진행률 예측.

    Phase 2 — VLA Projector 학습 (Δv 기반):
        TTTModule + ProgressHead를 freeze하고 VLAProjector만 학습.
        Δv = v(s_{t+1}) - v(s_t) 를 이용해 action 수정 방향을 학습.
        Loss: -Δv × log P(a_expert | s_t, θ_t)
        → Δv > 0 (진전) 구간의 expert action을 강화,
          Δv < 0 (퇴보) 구간은 억제.

추론 시:
    - TTT inner loop이 매 timestep SSL로 자체 업데이트
    - ProgressHead로 실패 감지 (단조증가 이탈)
    - VLAProjector가 VLA의 action 출력을 수정하여 실패 루프 탈출

주입 방식 (VLAProjector modes):
    1. hidden_add: h_final = h_LLM + α·Proj(h_TTT)
    2. logit_shift: L_final = L_VLA + α·Proj(h_TTT)  (이산화 출력 모델)
    3. film: γ(h_TTT)⊙x + β(h_TTT)  (diffusion action expert)
    4. token: input token으로 VLA에 주입  (가장 단순)

참고 논문:
    - "Learning to (Learn at Test Time)" (Sun et al., 2024)
    - "VITA" (Ziakas & Russo, ICLR 2026)
"""

from src.ttt.ttt_module import TTTModule
from src.ttt.progress_head import ProgressHead
from src.ttt.vla_projector import VLAProjector
from src.ttt.predictor import ProgressPredictor, TTTVLAAdapter

__all__ = [
    "TTTModule",
    "ProgressHead",
    "VLAProjector",
    "ProgressPredictor",
    "TTTVLAAdapter",
]
