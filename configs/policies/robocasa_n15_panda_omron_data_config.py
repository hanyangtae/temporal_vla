"""Isaac-GR00T N1.5 BaseDataConfig for RoboCasa PandaOmron datasets.

Originally targets the per-task LeRobot v2.1 pretrain datasets downloaded by
``scripts/utils/download_robocasa_pretrain_human.sh`` to:

    data/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot

The same config also works with the merged dataset at
``data/datasets/robocasa_10tasks_lerobot_v21`` since they share the same
``meta/modality.json`` schema (state/action/video/language keys identical).
It also works with the target atomic-seen 15-task datasets under:

    data/robocasa/v1.0/target/atomic/<Task>/<date>/lerobot

Used via N1.5's ``new_embodiment`` route (N1.5's EmbodimentTag enum has no
``ROBOCASA_PANDA_OMRON`` entry — that exists only in N1.6). N1.6 pretrained
head benefits are NOT carried over here; this is for N1.5 sanity / debugging.

Usage:

    PYTHONPATH=$REPO_ROOT/src/policies/Isaac-GR00T-N1.5:$REPO_ROOT/configs/policies \\
    python $REPO_ROOT/src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py \\
        --dataset-path \\
            $REPO_ROOT/data/robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot \\
            $REPO_ROOT/data/robocasa/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot \\
            ... \\
        --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \\
        --embodiment_tag new_embodiment \\
        --num-gpus 1 --batch-size 16 --max-steps 500 \\
        --output-dir /tmp/n15_sanity

Origin reference: ``src/utils/common/robocasa_n15_data_config.py`` holds the
companion ``RobocasaKitchenPnPDataConfig`` for the HDF5-converted route
(not used in this project's main flow).
"""

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class RobocasaPandaOmron10TaskDataConfig(BaseDataConfig):
    """Data config for RoboCasa PandaOmron atomic datasets.

    State/action/video/language key order matches ``meta/modality.json`` of the
    NVIDIA-publish pretrain and target datasets (start-index order). The class
    name is historical; it is also used for the target atomic-seen 15-task split.
    """

    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.base_motion",
        "action.control_mode",
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
