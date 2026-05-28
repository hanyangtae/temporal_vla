"""Stage 2 dataset-side patch — episode-prefix Eagle z_seq injection.

GR00T 의 ``ShardedSingleStepDataset`` 은 sample 마다 frame t 단위 obs+action 만
주는데, TTT episode-prefix replay 모드 는 각 sample 에 추가로 ``z_seq[0..t]``
(Eagle pre-LLM cache 슬라이스) 가 필요. 이 모듈이 두 가지를 monkey-patch:

1. ``DatasetFactory.build`` 가 호출하는 ``ShardedSingleStepDataset`` 을
   ``ShardedSingleStepDatasetWithTTT`` 로 교체. 후자는 dataset_path 에서 task 이름을
   추론해 Eagle cache 를 lazy-load 하고, 매 datapoint 에 ``ttt_z_seq`` /
   ``ttt_valid_mask`` 를 추가.

2. ``Gr00tN1d6DataCollator.__call__`` 을 변형해 가변 길이 ``ttt_z_seq`` 들을
   batch 내 max_T 로 dynamic padding.

Usage (from ``scripts/train/launch_finetune_ttt.py``)::

    from src.ttt.integrations.dataset_patch import setup_ttt_dataset_patch
    setup_ttt_dataset_patch(eagle_cache_root="/cache/datasets/robocasa_eagle_pre_llm")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_GR00T_PATH = "/temporal_vla/src/policies/Isaac-GR00T"
if _GR00T_PATH not in sys.path:
    sys.path.insert(0, _GR00T_PATH)


# 모듈 레벨 config — patched 함수들이 참조.
_TTT_DATA_CONFIG: dict = {
    "eagle_cache_root": None,
    "patched": False,
}


# ──────────────────────────────────────────────────────────────────────────
# Helper: dataset_path → task name + Eagle cache path
# ──────────────────────────────────────────────────────────────────────────


def _task_name_from_dataset_path(dataset_path: str) -> str:
    """``/cache/datasets/robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot``
    → ``"OpenDrawer"``.

    Convention: dataset_path 의 부모 3 level 가 ``<atomic_root>/<TaskName>/<date>/lerobot``.
    """
    p = Path(dataset_path).resolve()
    # path/.../<Task>/<date>/lerobot
    return p.parents[1].name


def _eagle_cache_path_for(task_name: str, cache_root: str | Path) -> Path:
    return Path(cache_root) / task_name / "embeddings.pt"


# ──────────────────────────────────────────────────────────────────────────
# Subclass: dataset that adds ttt_z_seq / ttt_valid_mask per datapoint
# ──────────────────────────────────────────────────────────────────────────


def _make_dataset_subclass():
    """ShardedSingleStepDataset 을 import 한 뒤 subclass 생성. import 가 패치 호출
    시점에 일어나야 해서 함수로 감쌈."""
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    class ShardedSingleStepDatasetWithTTT(ShardedSingleStepDataset):
        """Add ``ttt_z_seq`` (= Eagle pre-LLM cache for frames 0..t) and
        ``ttt_valid_mask`` to each datapoint.

        Eagle cache 는 task-local abs_idx → tensor[2048] dict 형태. 우리는 episode
        boundary 정보로 abs_idx 를 재구성 (ep_from + frame_in_episode).
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._task_name = _task_name_from_dataset_path(str(self.dataset_path))
            self._eagle_cache_path = _eagle_cache_path_for(
                self._task_name, _TTT_DATA_CONFIG["eagle_cache_root"]
            )
            self._eagle_cache: Optional[dict[int, torch.Tensor]] = None  # lazy
            self._ep_from: Optional[list[int]] = None
            logger.info(
                "[ttt_dataset] %s ← eagle_cache=%s",
                self._task_name, self._eagle_cache_path,
            )

        def _ensure_cache_loaded(self):
            if self._eagle_cache is not None:
                return
            if not self._eagle_cache_path.exists():
                raise FileNotFoundError(
                    f"Eagle cache missing for {self._task_name}: {self._eagle_cache_path}. "
                    f"Run scripts/extract/extract_eagle_pre_llm_robocasa.py first."
                )
            self._eagle_cache = torch.load(
                str(self._eagle_cache_path), map_location="cpu", weights_only=True,
            )
            # episode boundary — episode_lengths cumsum
            ep_lengths = self.episode_loader.episode_lengths
            self._ep_from = [0] + list(np.cumsum(ep_lengths)[:-1])
            logger.info(
                "[ttt_dataset] %s cache loaded: %d frames, %d episodes",
                self._task_name, len(self._eagle_cache), len(ep_lengths),
            )

        def _z_seq_for(self, ep_idx: int, step_index: int) -> torch.Tensor:
            """Return tensor [t+1, 2048] for Eagle z_seq from frame 0 to step_index."""
            self._ensure_cache_loaded()
            ep_from = self._ep_from[ep_idx]
            n = step_index + 1
            # stack frames 0..step_index
            return torch.stack([self._eagle_cache[ep_from + t] for t in range(n)], dim=0)

        def get_shard(self, idx: int) -> list:
            """Same as parent but injects ``ttt_z_seq`` / ``ttt_valid_mask`` per datapoint."""
            episodes = self.sharded_episodes[idx]
            datapoints = []
            for ep_idx, step_indices in episodes:
                episode_data = self.episode_loader[ep_idx]
                for step_index in step_indices:
                    dp = self.get_datapoint(episode_data, step_index)
                    t = int(step_index)
                    dp["ttt_z_seq"] = self._z_seq_for(ep_idx, t)               # [t+1, 2048] fp32
                    dp["ttt_valid_mask"] = torch.ones(t + 1, dtype=torch.bool) # 모두 valid
                    datapoints.append(dp)
            return datapoints

    return ShardedSingleStepDatasetWithTTT


# ──────────────────────────────────────────────────────────────────────────
# Patched collator — dynamic-pad ttt_z_seq + stack ttt_valid_mask
# ──────────────────────────────────────────────────────────────────────────


def _wrap_collator(original_cls):
    """원래 ``Gr00tN1d6DataCollator`` 를 wrap — ttt_z_seq / ttt_valid_mask 만 별도 처리,
    나머지는 super().__call__ 위임. 단 super 가 "all values stackable" 을 가정하므로,
    가변 길이 키는 미리 pop 후 super 호출하고 결과에 padding 처리한 tensor 를 다시 주입.
    """

    class Gr00tN1d6DataCollatorWithTTT(original_cls):
        def __call__(self, features):
            # 1) gather + pop ttt_z_seq / ttt_valid_mask 먼저 (super 가 stack 못함)
            z_seqs = []
            v_masks = []
            for f in features:
                z_seqs.append(f.pop("ttt_z_seq", None))
                v_masks.append(f.pop("ttt_valid_mask", None))

            # 2) super 가 나머지 batch
            out = original_cls.__call__(self, features)

            # 3) dynamic-pad ttt_z_seq
            if z_seqs[0] is not None:
                max_T = max(z.shape[0] for z in z_seqs)
                D = z_seqs[0].shape[1]
                B = len(z_seqs)
                padded_z = torch.zeros(B, max_T, D, dtype=z_seqs[0].dtype)
                mask = torch.zeros(B, max_T, dtype=torch.bool)
                for i, (z, m) in enumerate(zip(z_seqs, v_masks)):
                    T_i = z.shape[0]
                    padded_z[i, :T_i] = z
                    mask[i, :T_i] = m if m is not None else True

                # 4) inputs dict 에 주입 (forward 에서 inputs.pop 으로 받음)
                out["inputs"]["ttt_z_seq"] = padded_z
                out["inputs"]["ttt_valid_mask"] = mask

            return out

    return Gr00tN1d6DataCollatorWithTTT


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def setup_ttt_dataset_patch(eagle_cache_root: str | Path) -> None:
    """Monkey-patch GR00T dataset factory + Gr00tN1d6 collator to inject Eagle
    z_seq per sample (episode-prefix mode). Idempotent."""
    from gr00t.data.dataset import factory as factory_module
    from gr00t.model.gr00t_n1d6 import processing_gr00t_n1d6 as proc_module

    _TTT_DATA_CONFIG["eagle_cache_root"] = str(Path(eagle_cache_root).resolve())

    if _TTT_DATA_CONFIG["patched"]:
        logger.info("[ttt_dataset_patch] already patched; config updated")
        return

    # 1) DatasetFactory 가 import 하는 ShardedSingleStepDataset 을 subclass 로 교체
    new_dataset_cls = _make_dataset_subclass()
    factory_module.ShardedSingleStepDataset = new_dataset_cls

    # 2) Gr00tN1d6 Processor 의 collator class 교체
    new_collator_cls = _wrap_collator(proc_module.Gr00tN1d6DataCollator)
    proc_module.Gr00tN1d6DataCollator = new_collator_cls
    # Gr00tN1d6Processor.data_collator_class 도 같이 교체 (class-level attr)
    proc_module.Gr00tN1d6Processor.data_collator_class = new_collator_cls

    _TTT_DATA_CONFIG["patched"] = True
    logger.info(
        "[ttt_dataset_patch] patched: ShardedSingleStepDataset → %s, "
        "Gr00tN1d6DataCollator → %s (eagle_cache_root=%s)",
        new_dataset_cls.__name__, new_collator_cls.__name__,
        _TTT_DATA_CONFIG["eagle_cache_root"],
    )
