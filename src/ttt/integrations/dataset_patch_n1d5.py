"""Dataset patch for **GR00T N1.5 + TTT** Phase 2 finetune.

N1.5 의 ``LeRobotSingleDataset`` 을 subclass 로 wrap 하여 매 datapoint 에
``ttt_z_seq`` / ``ttt_valid_mask`` 를 부착. wrapper (``attach_ttt_to_n1d5``) 가
forward 시 사용.

N1.6 의 ``dataset_patch.py`` 와 mirror 패턴 — 단 N1.5 는
``Gr00tN1d6DataCollator`` 같은 별도 collator 가 아니라 transformers default
collator 사용 가능성이 있어 wrap 방식 다름.

이 모듈은 N1.5 codebase 의 ``gr00t.data.dataset`` 를 import 하므로 PYTHONPATH 가
``Isaac-GR00T-N1.5`` 우선해야 함 (호출자가 sys.path 관리).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# N1.5 codebase 경로
_N1d5_ROOT = "/temporal_vla/src/policies/Isaac-GR00T-N1.5"
if _N1d5_ROOT not in sys.path:
    sys.path.insert(0, _N1d5_ROOT)

from gr00t.data.dataset import LeRobotSingleDataset  # N1.5

from src.datasets.robocasa_v21_reader import RoboCasaV21Reader  # noqa: E402


def _task_name_from_dataset_path(dataset_path: str | Path) -> str:
    """``.../atomic/<Task>/<date>/lerobot`` → ``<Task>``."""
    p = Path(dataset_path).resolve()
    # lerobot/<date>/<Task>/atomic — reverse traverse
    parts = list(p.parts)
    if "lerobot" in parts:
        i = parts.index("lerobot")
        # parts[i-2] should be Task name (parts: ..., Task, date, lerobot)
        if i >= 2:
            return parts[i - 2]
    raise ValueError(f"Cannot infer task name from path: {dataset_path}")


class LeRobotSingleDatasetWithTTT(LeRobotSingleDataset):
    """N1.5 dataset + Eagle pre-LLM cache slice 부착.

    추가 fields per sample::

        ttt_z_seq:      Tensor [t+1, D]  — frames 0..t of this episode
        ttt_valid_mask: Tensor [t+1] bool — all True (padding 은 collator 가)
    """

    def __init__(
        self,
        *args,
        eagle_cache_root: str | Path,
        max_episode_index: int | None = None,
        **kwargs,
    ):
        """``max_episode_index``: keep samples whose episode_index < N (None=keep all)."""
        super().__init__(*args, **kwargs)
        self._eagle_cache_root = Path(eagle_cache_root)
        self._task_name = _task_name_from_dataset_path(self.dataset_path)
        self._eagle_cache: dict[int, torch.Tensor] | None = None

        # abs_frame_idx → (ep_idx, frame_in_ep). reader 통해 episode boundary 가져옴.
        reader = RoboCasaV21Reader(self.dataset_path)
        self._all_eps = reader.lerobot_meta_episodes()
        self._abs_to_ep: dict[int, tuple[int, int]] = {}
        for ep_idx, ep in self._all_eps.items():
            a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            for t_abs in range(a, b):
                self._abs_to_ep[t_abs] = (ep_idx, t_abs - a)

        # Episode-index filter: rebuild all_steps from kept episodes only.
        # all_steps[i] = (traj_id, base_idx). traj_id 가 곧 episode_index 이므로
        # 이 값으로 직접 필터. `all_steps` 는 base class property 라 직접 assign 불가 —
        # 내부 `_all_steps` 를 갱신.
        if max_episode_index is not None:
            n_before = len(self._all_steps)
            self._all_steps = [
                (traj_id, base_idx)
                for (traj_id, base_idx) in self._all_steps
                if int(traj_id) < int(max_episode_index)
            ]
            n_eps_kept = len({t for (t, _) in self._all_steps})
            print(
                f"[ttt_dataset_n1d5] {self._task_name} episode filter "
                f"(ep_idx < {max_episode_index}): "
                f"{n_before} → {len(self._all_steps)} steps "
                f"({n_eps_kept} episodes)"
            )

    def _ensure_cache(self):
        if self._eagle_cache is None:
            cache_path = self._eagle_cache_root / self._task_name / "embeddings.pt"
            if not cache_path.exists():
                raise FileNotFoundError(f"Eagle cache not found: {cache_path}")
            self._eagle_cache = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            print(f"[ttt_dataset_n1d5] {self._task_name} cache loaded: {len(self._eagle_cache)} frames")

    def _z_seq_for(self, abs_idx: int) -> torch.Tensor:
        self._ensure_cache()
        if abs_idx not in self._abs_to_ep:
            raise KeyError(f"abs_idx={abs_idx} not in dataset episodes")
        ep_idx, _t_in_ep = self._abs_to_ep[abs_idx]
        ep = self._all_eps[ep_idx]
        ep_from = int(ep["dataset_from_index"])
        # frames 0..t_in_ep (inclusive). abs idx = ep_from .. abs_idx
        z_list = []
        for i in range(ep_from, abs_idx + 1):
            if i not in self._eagle_cache:
                # missing cache (cap 200 ep 이후 frame 등) → skip
                continue
            z_list.append(self._eagle_cache[i])
        if not z_list:
            # fallback: single frame zero (shouldn't happen if cache covers ep)
            return torch.zeros(1, 2048, dtype=torch.float32)
        return torch.stack(z_list).float()

    def __getitem__(self, idx):
        # all_steps[idx] = (traj_id, base_idx). traj_id = episode_index.
        # abs_idx = ep dataset_from_index + base_idx — eagle cache 의 키.
        # (filter 안 했을 땐 abs_idx == idx 가 우연히 성립하지만, filter 후엔 어긋남.)
        traj_id, base_index = self.all_steps[idx]
        ep_from = int(self._all_eps[int(traj_id)]["dataset_from_index"])
        abs_idx = ep_from + int(base_index)

        sample = super().__getitem__(idx)
        z_seq = self._z_seq_for(abs_idx)                               # [t+1, D]
        sample["ttt_z_seq"] = z_seq
        sample["ttt_valid_mask"] = torch.ones(z_seq.shape[0], dtype=torch.bool)
        return sample


# ────────────────────────────────────────────────────────────────────
# Collator — variable-length ttt_z_seq 를 dynamic-pad
# ────────────────────────────────────────────────────────────────────
def make_ttt_collator(base_collator):
    """Wrap base collator to handle ttt_z_seq / ttt_valid_mask dynamic padding."""
    def _collated(features):
        # 1) extract + pop
        z_seqs = []
        valid_masks = []
        for f in features:
            z_seqs.append(f.pop("ttt_z_seq", None))
            valid_masks.append(f.pop("ttt_valid_mask", None))

        # 2) base collator on remaining fields
        batch = base_collator(features)

        # 2.5) float64 → float32 cast (TrainingArguments bf16=True 의 amp 가 float64 는 자동
        # cast 안 함 → state_encoder 에서 'expected BFloat16 but found Double' 에러).
        # state, action 등 N1.5 dataset 이 numpy.float64 로 만드는 항목 일괄 처리.
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                batch[k] = v.to(torch.float32)

        # 3) dynamic-pad ttt_z_seq
        if all(z is not None for z in z_seqs):
            max_T = max(z.shape[0] for z in z_seqs)
            B = len(z_seqs)
            D = z_seqs[0].shape[1]
            padded_z = torch.zeros(B, max_T, D, dtype=z_seqs[0].dtype)
            padded_mask = torch.zeros(B, max_T, dtype=torch.bool)
            for i, (z, m) in enumerate(zip(z_seqs, valid_masks)):
                t = z.shape[0]
                padded_z[i, :t] = z
                padded_mask[i, :t] = m
            batch["ttt_z_seq"] = padded_z
            batch["ttt_valid_mask"] = padded_mask
        return batch
    return _collated
