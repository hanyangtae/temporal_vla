"""Stage 2 launch-time patch — wrap the loaded GR00T model with TTT.

Upstream ``gr00t.experiment.experiment.run`` builds the model via
``Gr00tN1d6Pipeline._create_model`` (which calls ``AutoModel.from_pretrained``).
We monkey-patch that method to wrap the returned model with
``Gr00tN1d6WithTTT`` whenever a Phase 1 predictor path is configured. Upstream
code is otherwise untouched — same trainer, same hyperparameters, same
checkpointing.

Usage (from ``scripts/train/launch_finetune_ttt.py``)::

    from src.ttt.integrations.launch_patch import setup_ttt_pipeline_patch
    setup_ttt_pipeline_patch(predictor_path="...", update_in_train=False)
    # ... then call gr00t.experiment.experiment.run(config) as usual
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_GR00T_PATH = "/temporal_vla/src/policies/Isaac-GR00T"
if _GR00T_PATH not in sys.path:
    sys.path.insert(0, _GR00T_PATH)


# 모듈 레벨 state — 패치된 _create_model 이 참조.
_TTT_CONFIG: dict = {
    "predictor_path": None,
    "predictor_input_dim": 2048,
    "predictor_proj_dim": 2048,
    "predictor_inner_model": "linear",
    "predictor_eta_base": 0.1,
    "update_in_train": False,
    "patched": False,
}


def setup_ttt_pipeline_patch(
    predictor_path: Optional[str],
    *,
    predictor_input_dim: int = 2048,
    predictor_proj_dim: int = 2048,
    predictor_inner_model: str = "linear",
    predictor_eta_base: float = 0.1,
    update_in_train: bool = False,
) -> None:
    """``Gr00tN1d6Pipeline._create_model`` 을 TTT wrap 버전으로 monkey-patch.

    ``predictor_path`` 가 None / 빈 문자열이면 wrap 하지 않음 (=baseline GR00T finetune).
    같은 함수를 두 번 호출해도 안전 (한 번만 패치).
    """
    from gr00t.model.gr00t_n1d6.setup import Gr00tN1d6Pipeline
    from src.ttt.integrations.groot_wrapper import attach_ttt_to_groot

    _TTT_CONFIG.update({
        "predictor_path": predictor_path or None,
        "predictor_input_dim": predictor_input_dim,
        "predictor_proj_dim": predictor_proj_dim,
        "predictor_inner_model": predictor_inner_model,
        "predictor_eta_base": predictor_eta_base,
        "update_in_train": update_in_train,
    })

    if _TTT_CONFIG["patched"]:
        logger.info("[ttt_patch] already patched; updated config only")
        return

    _orig_create_model = Gr00tN1d6Pipeline._create_model

    def _create_model_with_ttt(self):
        base_model = _orig_create_model(self)
        ppath = _TTT_CONFIG["predictor_path"]
        if not ppath:
            logger.info("[ttt_patch] no predictor_path → returning base GR00T (TTT disabled)")
            return base_model
        logger.info(
            "[ttt_patch] wrapping base GR00T with TTT predictor "
            "(predictor=%s, update_in_train=%s)",
            ppath, _TTT_CONFIG["update_in_train"],
        )
        return attach_ttt_to_groot(
            base_model,
            predictor_state_path=ppath,
            predictor_input_dim=_TTT_CONFIG["predictor_input_dim"],
            predictor_proj_dim=_TTT_CONFIG["predictor_proj_dim"],
            predictor_inner_model=_TTT_CONFIG["predictor_inner_model"],
            predictor_eta_base=_TTT_CONFIG["predictor_eta_base"],
            ttt_update_in_train=_TTT_CONFIG["update_in_train"],
        )

    Gr00tN1d6Pipeline._create_model = _create_model_with_ttt
    _TTT_CONFIG["patched"] = True
    logger.info("[ttt_patch] Gr00tN1d6Pipeline._create_model patched")
