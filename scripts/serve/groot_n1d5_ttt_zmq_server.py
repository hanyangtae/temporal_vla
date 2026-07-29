"""GR00T N1.5 (+TTT) ZMQ inference server — drop-in replacement for
``Isaac-GR00T-N1.5/scripts/inference_service.py --server``.

기존 N1.5 inference_service 의 서버 분기와 **동일 ZMQ 프로토콜** (msgpack +
``RobotInferenceServer`` endpoint ``get_action`` / ``get_modality_config``) 을
유지하므로 client (``scripts/eval/groot_n15_robocasa_zmq_eval.py``) 는 그대로
사용 가능.

핵심 추가:
  - ``attach_ttt_to_n1d5`` 로 TTT 모듈 부착 (predictor_path 비우면 vanilla 서빙).
  - ``Gr00tPolicy._get_action_from_normalized_input`` 의 ``torch.inference_mode``
    → ``torch.no_grad`` 로 monkey-patch — TTT inner-loop ``autograd.grad`` 차단
    방지 (inference_mode 안에서는 enable_grad 도 무력화돼 inference tensor 가
    되어 backward 불가).

실행:
  python scripts/serve/groot_n1d5_ttt_zmq_server.py \\
      --model-path /temporal_vla/outputs/groot_ttt_n1d5_target15_ep100 \\
      --predictor-path /temporal_vla/outputs/train/phase1_groot_robocasa/<...>/phase1_final.pt \\
      --predictor-inner-model linear_preLN \\
      --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \\
      --embodiment-tag new_embodiment \\
      --port 5555

vanilla 서빙 (TTT 없이 GPU 2 의 inference_service 와 동등):
  python scripts/serve/groot_n1d5_ttt_zmq_server.py \\
      --model-path /temporal_vla/outputs/train/groot_n1_5/260513094637-subset100/checkpoint-5895 \\
      --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \\
      --embodiment-tag new_embodiment \\
      --port 5555
  (predictor-path 비워두면 TTT attach 생략.)
"""

from __future__ import annotations

import argparse
import importlib
import sys

import torch

# ──────────────────────────────────────────────────────────────────
# Path setup (N1.5 codebase + configs/policies + repo root)
# ──────────────────────────────────────────────────────────────────
for _p in (
    "/temporal_vla/src/policies/Isaac-GR00T-N1.5",
    "/temporal_vla/configs/policies",
    "/temporal_vla",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# pytorch3d stub — N1.5 state_action transform import 우회 (실제 호출 X).
import types as _types  # noqa: E402
if "pytorch3d" not in sys.modules:
    _pt = _types.ModuleType("pytorch3d")
    _pt.transforms = _types.ModuleType("pytorch3d.transforms")
    for _name in (
        "matrix_to_quaternion", "quaternion_to_matrix",
        "matrix_to_euler_angles", "euler_angles_to_matrix",
        "matrix_to_rotation_6d", "rotation_6d_to_matrix",
        "quaternion_apply", "axis_angle_to_quaternion",
    ):
        setattr(_pt.transforms, _name, lambda *a, **k: None)
    sys.modules["pytorch3d"] = _pt
    sys.modules["pytorch3d.transforms"] = _pt.transforms


# ──────────────────────────────────────────────────────────────────
# Monkey-patch Gr00tPolicy: inference_mode → no_grad (TTT inner update 위해)
# ──────────────────────────────────────────────────────────────────
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

_COMPUTE_DTYPE = torch.bfloat16


def _ttt_aware_get_action_from_normalized_input(self, normalized_input):
    """Gr00tPolicy._get_action_from_normalized_input 의 TTT-aware 대체.

    원본은 ``torch.inference_mode()`` 안에서 ``model.get_action`` 호출 →
    inference tensor 가 만들어져 wrapper 의 ``autograd.grad`` (TTT inner update)
    가 RuntimeError. 여기서는 ``torch.no_grad()`` 로 약화 — wrapper 의
    ``_enable_inner_grad`` 가 transient ``enable_grad`` 로 inner update 통과.
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=_COMPUTE_DTYPE):
        model_pred = self.model.get_action(normalized_input)
    return model_pred["action_pred"].float()


Gr00tPolicy._get_action_from_normalized_input = _ttt_aware_get_action_from_normalized_input


# ──────────────────────────────────────────────────────────────────
# Server entry
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.5 (+TTT) ZMQ inference server",
    )
    parser.add_argument("--model-path", required=True,
                        help="N1.5 ckpt dir (finetune 결과 또는 pretrain).")
    parser.add_argument("--data-config", default=(
        "robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig"),
                        help="<module>:<ClassName> data config.")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=4)

    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--api-token", type=str, default=None)

    # TTT options (predictor-path 비우면 vanilla 서빙).
    parser.add_argument("--predictor-path", default="",
                        help="Phase 1 학습 결과 ckpt. 비우면 TTT 비활성.")
    parser.add_argument("--predictor-input-dim", type=int, default=2048)
    parser.add_argument("--predictor-proj-dim", type=int, default=1536)
    parser.add_argument("--predictor-inner-model", default="linear",
                        choices=["linear", "linear_preLN", "mlp"])
    parser.add_argument("--predictor-eta-base", type=float, default=0.1)
    parser.add_argument("--ttt-update-at-inference", action="store_true", default=True,
                        help="추론 시 매 timestep inner-loop SSL update (default True).")
    parser.add_argument("--no-ttt-update", action="store_true", default=False,
                        help="ttt_update_at_inference 강제 비활성.")

    args = parser.parse_args()

    # data_config 모듈:Class → modality_config + transforms.
    mod_name, cls_name = args.data_config.split(":", 1)
    data_config_cls = getattr(importlib.import_module(mod_name), cls_name)()
    modality_config = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()

    print(f"[zmq-n1d5] loading {args.model_path}")
    print(f"           embodiment={args.embodiment_tag} denoising_steps={args.denoising_steps}")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # TTT attach (predictor_path 있을 때만).
    if args.predictor_path:
        from src.ttt.integrations.groot_wrapper_n1d5 import attach_ttt_to_n1d5
        ttt_update = (not args.no_ttt_update) and args.ttt_update_at_inference
        print(f"[ttt] attach (inner={args.predictor_inner_model}, "
              f"update_at_inference={ttt_update})")
        attach_ttt_to_n1d5(
            policy.model,
            predictor_input_dim=args.predictor_input_dim,
            predictor_proj_dim=args.predictor_proj_dim,
            predictor_inner_model=args.predictor_inner_model,
            predictor_eta_base=args.predictor_eta_base,
            predictor_state_path=args.predictor_path,
            ttt_update_in_train=False,
            ttt_update_at_inference=ttt_update,
        )
    else:
        print("[zmq-n1d5] no predictor-path → vanilla 서빙 (no TTT)")

    # ZMQ 서버 시작 — N1.5 inference_service.py 와 동일 프로토콜.
    from gr00t.eval.robot import RobotInferenceServer
    server = RobotInferenceServer(policy, host=args.host, port=args.port, api_token=args.api_token)
    print(f"[zmq-n1d5] server listening on tcp://{args.host}:{args.port}")
    server.run()


if __name__ == "__main__":
    main()
