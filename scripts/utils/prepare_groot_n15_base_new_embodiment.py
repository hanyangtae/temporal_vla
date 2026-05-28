#!/usr/bin/env python3
"""Create a local GR00T N1.5 base checkpoint with PandaOmron metadata.

The upstream N1.5 base snapshot has metadata for pretrained embodiments such as
``gr1`` but not for ``new_embodiment``.  This helper keeps the base weights as
symlinks and writes ``experiment_cfg/metadata.json`` from the RoboCasa
PandaOmron target-15 LeRobot datasets, so N1.5 inference can run before
fine-tuning as a sanity baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "configs/policies"))
sys.path.insert(0, str(REPO_ROOT / "src/policies/Isaac-GR00T-N1.5"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.path_setup import DATA_ROOT  # noqa: E402
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402


TARGET_15_TASKS = [
    ("CloseFridge", "20250816"),
    ("CloseToasterOvenDoor", "20250818"),
    ("CoffeeSetupMug", "20250813"),
    ("OpenCabinet", "20250813"),
    ("OpenDrawer", "20250816"),
    ("PickPlaceCounterToCabinet", "20250811"),
    ("PickPlaceCounterToStove", "20250818"),
    ("PickPlaceDrawerToCounter", "20250820"),
    ("PickPlaceSinkToCounter", "20250813"),
    ("PickPlaceToasterToCounter", "20250817"),
    ("SlideDishwasherRack", "20250820"),
    ("TurnOffStove", "20250812"),
    ("TurnOnElectricKettle", "20250817"),
    ("TurnOnMicrowave", "20250813"),
    ("TurnOnSinkFaucet", "20250812"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=Path,
        default=DATA_ROOT
        / "huggingface/hub/models--nvidia--GR00T-N1.5-3B"
        / "snapshots/869830fc749c35f34771aa5209f923ac57e4564e",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DATA_ROOT / "robocasa/v1.0/target/atomic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/checkpoints/groot_n15_base_pandaomron_new_embodiment",
    )
    parser.add_argument(
        "--data-config",
        default="robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig",
    )
    parser.add_argument("--video-backend", default="torchcodec")
    return parser.parse_args()


def symlink_base_files(base_model: Path, output: Path) -> None:
    required = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    output.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = base_model / name
        dst = output / name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)


def build_metadata(args: argparse.Namespace) -> dict:
    data_config = load_data_config(args.data_config)
    modality_configs = data_config.modality_config()
    datasets = []
    for task, date in TARGET_15_TASKS:
        dataset_path = args.target_root / task / date / "lerobot"
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        datasets.append(
            LeRobotSingleDataset(
                dataset_path=str(dataset_path),
                modality_configs=modality_configs,
                transforms=None,
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                video_backend=args.video_backend,
            )
        )
    mixture = LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in datasets],
        mode="train",
    )
    return {
        EmbodimentTag.NEW_EMBODIMENT.value: mixture.merged_metadata[
            EmbodimentTag.NEW_EMBODIMENT.value
        ].model_dump(mode="json")
    }


def main() -> None:
    args = parse_args()
    symlink_base_files(args.base_model, args.output)
    metadata = build_metadata(args)
    exp_cfg_dir = args.output / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = exp_cfg_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4))
    print(f"wrote {metadata_path}")
    print(f"checkpoint {args.output}")


if __name__ == "__main__":
    main()
