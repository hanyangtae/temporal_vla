"""Run Isaac-GR00T-N1.5 fine-tuning with per-dataset episode subset.

Wraps ``src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py`` and applies a
monkey-patch to ``gr00t.data.dataset.LeRobotSingleDataset._get_trajectories`` so
that each ``LeRobotSingleDataset`` only loads the first ``MAX_EPISODES_PER_DATASET``
episodes from its ``episodes.jsonl``.

Why: N1.5 ``LeRobotSingleDataset`` has no native subset arg, and the upstream
``gr00t_finetune.py`` instantiates it directly. Modifying the submodule is
undesirable, so we patch in this wrapper before invoking ``runpy``.

Env var (required to take effect):
  MAX_EPISODES_PER_DATASET=<int>   # 0 or unset → no subset (full dataset)

CLI: pass the exact args you would pass to ``gr00t_finetune.py``. They are
forwarded as ``sys.argv``.

Example:
  MAX_EPISODES_PER_DATASET=100 python scripts/train/gr00t_n15_finetune_subset.py \\
      --dataset-path ... --data-config ... --output-dir ... \\
      --batch-size 64 --max-steps 10000 --save-steps 5000
"""

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py"


def _install_episode_subset_patch() -> None:
    max_eps = int(os.environ.get("MAX_EPISODES_PER_DATASET", "0"))
    if max_eps <= 0:
        return

    from gr00t.data.dataset import LeRobotSingleDataset

    original = LeRobotSingleDataset._get_trajectories

    def patched(self):
        traj_ids, traj_lens = original(self)
        sliced_ids = traj_ids[:max_eps]
        sliced_lens = traj_lens[:max_eps]
        print(
            f"[episode-subset] {self._dataset_name}: "
            f"using {len(sliced_ids)}/{len(traj_ids)} episodes "
            f"(MAX_EPISODES_PER_DATASET={max_eps})"
        )
        return sliced_ids, sliced_lens

    LeRobotSingleDataset._get_trajectories = patched


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"upstream finetune script missing: {TARGET}")

    _install_episode_subset_patch()

    # forward CLI: drop our wrapper from argv[0], keep the rest as-is
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
