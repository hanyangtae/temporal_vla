"""Extract Eagle pre-LLM embeddings from RoboCasa LeRobot v2.1 datasets.

Stage 0 of TTT × GR00T finetune pipeline. Produces a per-task cache that
Stage 1 (`scripts/train/phase1_groot_robocasa.py`) consumes.

For every frame of every episode we run the frozen Eagle backbone with
``output_hidden_states=True`` and capture ``hidden_states[0]`` — the fused
patch + text embedding *before* the LLM transformer layers. This is the
representation TTT will see during Stage 2; pre-computing it lets Stage 1
train without GR00T in the loop.

Data layout assumed (LeRobot v2.1 — what GR00T expects)
-------------------------------------------------------
::

    {root}/meta/info.json            ← chunks_size, data_path, video_path
    {root}/meta/episodes.jsonl       ← per-episode length + tasks
    {root}/meta/tasks.jsonl          ← task_index → task_text
    {root}/data/chunk-{cc}/episode_{ee}.parquet
    {root}/videos/chunk-{cc}/{video_key}/episode_{ee}.mp4

We bypass ``lerobot.datasets.LeRobotDataset`` because the lerobot submodule
on this repo is v3.0-only and rejects v2.1. GR00T uses its own loader for
finetune; for one-time embedding extraction we read parquet + mp4 directly.

Pipeline per frame
------------------
1. Build a single-batch nested observation matching `Gr00tPolicy` format.
2. Run `Gr00tPolicy.processor` + collator (same path as inference) to get
   Eagle inputs (``input_ids``, ``attention_mask``, ``pixel_values``).
3. Call ``backbone.model(**vl_input, output_hidden_states=True)`` directly
   and grab ``hidden_states[0]``  — shape ``[1, T_vl, 2048]``.
4. Masked-mean-pool over ``T_vl`` (using ``attention_mask``) → ``[2048]``.

Modality config
---------------
We import ``configs/policies/groot_robocasa_panda_omron_config.py`` (same
file `launch_finetune.py` consumes via ``--modality_config_path``) so the
``ROBOCASA_PANDA_OMRON`` embodiment is registered with the LeRobot-native
key convention (``robot0_agentview_left/right``, ``robot0_eye_in_hand``,
``annotation.human.task_description``). Stage 0 thus sees exactly what
Stage 2 finetune will see.

Output structure
----------------
``data/robocasa_eagle_pre_llm/{Task}/embeddings.pt`` — single ``torch.save``
of ``dict[int, torch.FloatTensor[2048]]`` keyed by ``abs_frame_idx``
(cumulative ``ep_from + frame_in_episode``). Compatible with the
cache-loading branch of `Phase1EpisodicDataset._load_or_compute_embeddings`.

Run inside the groot container::

    docker compose run --rm groot \\
        python /temporal_vla/scripts/extract/extract_eagle_pre_llm_robocasa.py \\
            --tasks OpenDrawer
            # omit --tasks for all 10 atomic tasks
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

# Isaac-GR00T submodule (gr00t package) — but NOT lerobot v3.0 (intentionally
# absent here, since v3.0 lerobot rejects our v2.1 datasets).
sys.path.insert(0, "/temporal_vla/src/policies/Isaac-GR00T")


REPO_ROOT = Path("/temporal_vla")
DEFAULT_BASE = REPO_ROOT / "data/robocasa/v1.0/pretrain/atomic"
DEFAULT_SAVE = REPO_ROOT / "data/robocasa_eagle_pre_llm"
DEFAULT_MODEL_PATH = REPO_ROOT / "outputs/checkpoints/GR00T-N1.6-3B"
DEFAULT_MODALITY_CONFIG = REPO_ROOT / "configs/policies/groot_robocasa_panda_omron_config.py"

# RoboCasa LeRobot parquet 컬럼은 'observation.images.<modality_key>' 형태.
LEROBOT_IMAGE_PREFIX = "observation.images."


# ──────────────────────────────────────────────────────────────────────────
# Modality config registration
# ──────────────────────────────────────────────────────────────────────────


def register_modality_config(modality_config_path: Path) -> None:
    """Same as launch_finetune.py — importing the file triggers registration."""
    path = modality_config_path
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config not found: {path}")
    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"[modality] registered from {path}")


# ──────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────


def discover_lerobot_roots(base: Path, tasks: list[str] | None) -> list[tuple[str, Path]]:
    if not base.is_dir():
        raise FileNotFoundError(f"Atomic base not found: {base}")
    out: list[tuple[str, Path]] = []
    if tasks:
        for task in tasks:
            matches = sorted((base / task).glob("*/lerobot"))
            if not matches:
                raise FileNotFoundError(f"No lerobot/ under {base / task}")
            for m in matches:
                out.append((task, m))
    else:
        for m in sorted(base.glob("*/*/lerobot")):
            task = m.parents[1].name
            out.append((task, m))
    return out


# ──────────────────────────────────────────────────────────────────────────
# RoboCasa v2.1 reader (no lerobot dependency)
# ──────────────────────────────────────────────────────────────────────────


class RoboCasaV21Reader:
    """Read a single LeRobot v2.1 dataset root (parquet + per-camera mp4) without
    the ``lerobot`` python library. Yields per-episode iterators.
    """

    def __init__(self, lerobot_root: Path):
        self.root = Path(lerobot_root)
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path_tpl = self.info["data_path"]
        self.video_path_tpl = self.info["video_path"]

        # episodes.jsonl — episode_index, length, tasks
        self.episodes: list[dict] = []
        with open(self.root / "meta" / "episodes.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))
        self.episodes.sort(key=lambda e: int(e["episode_index"]))

        # tasks.jsonl — task_index → task_text
        self.task_text_by_index: dict[int, str] = {}
        with open(self.root / "meta" / "tasks.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.task_text_by_index[int(d["task_index"])] = d["task"]

        # 누적 frame 인덱스 — Phase1EpisodicDataset 의 dataset_from_index 와
        # 일치시켜야 캐시 lookup 이 동작.
        self._ep_from: dict[int, int] = {}
        cum = 0
        for ep in self.episodes:
            self._ep_from[int(ep["episode_index"])] = cum
            cum += int(ep["length"])
        self.total_frames = cum
        self.total_episodes = len(self.episodes)

    # ── episode helpers ────────────────────────────────────────────────

    def episode_paths(self, ep_idx: int) -> tuple[Path, dict[str, Path]]:
        """Return (parquet_path, {video_key: mp4_path}) for one episode."""
        chunk = ep_idx // self.chunks_size
        parquet = self.root / self.data_path_tpl.format(
            episode_chunk=chunk, episode_index=ep_idx,
        )
        video_dir = self.root / "videos" / f"chunk-{chunk:03d}"
        video_paths = {}
        if video_dir.is_dir():
            for sub in video_dir.iterdir():
                if not sub.is_dir():
                    continue
                p = sub / f"episode_{ep_idx:06d}.mp4"
                if p.exists():
                    video_paths[sub.name] = p  # name = "observation.images.<cam>"
        return parquet, video_paths

    def episode_from_index(self, ep_idx: int) -> int:
        return self._ep_from[int(ep_idx)]

    def episode_task_text(self, ep_idx: int) -> str:
        ep = next(e for e in self.episodes if int(e["episode_index"]) == ep_idx)
        tasks = ep.get("tasks") or []
        if tasks:
            return tasks[0]
        # fallback: task_index 컬럼에서 첫 frame 의 task_index → 텍스트
        parquet, _ = self.episode_paths(ep_idx)
        table = pq.read_table(str(parquet), columns=["task_index"])
        ti = int(table.column("task_index")[0].as_py())
        return self.task_text_by_index.get(ti, "")

    # ── frame iterator ────────────────────────────────────────────────

    def iter_episode_frames(
        self, ep_idx: int,
    ) -> Iterator[tuple[int, int, str, dict[str, np.ndarray]]]:
        """Yield ``(abs_frame_idx, frame_in_episode, task_text, images_uint8_hwc)``
        for every frame of one episode. ``images_uint8_hwc`` is a dict keyed
        by the LeRobot column name (e.g. ``observation.images.robot0_agentview_left``).
        """
        parquet, video_paths = self.episode_paths(ep_idx)
        if not parquet.exists():
            raise FileNotFoundError(f"Missing parquet: {parquet}")
        if not video_paths:
            raise FileNotFoundError(f"Missing videos for episode {ep_idx} under {self.root}")

        # task text — episode 내에서 동일 가정
        task_text = self.episode_task_text(ep_idx)

        # parquet 에서 length 만 확인
        table = pq.read_table(str(parquet), columns=["frame_index"])
        n_frames = table.num_rows

        ep_from = self._ep_from[ep_idx]

        # 모든 카메라의 모든 frame 을 메모리에 사전 디코드 (한 episode = ~200 frame).
        cam_frames: dict[str, list[np.ndarray]] = {}
        for cam_key, mp4_path in video_paths.items():
            frames = self._decode_video_frames(mp4_path)
            if len(frames) < n_frames:
                raise RuntimeError(
                    f"Decoded {len(frames)} frames from {mp4_path}, expected ≥ {n_frames}"
                )
            cam_frames[cam_key] = frames

        for t in range(n_frames):
            images = {cam: cam_frames[cam][t] for cam in cam_frames}
            yield ep_from + t, t, task_text, images

    @staticmethod
    def _decode_video_frames(mp4_path: Path) -> list[np.ndarray]:
        """Decode an mp4 to a list of HxWx3 uint8 numpy arrays. Uses pyav."""
        container = av.open(str(mp4_path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            frames: list[np.ndarray] = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                if arr.shape[:2] != (256, 256):
                    arr = np.array(Image.fromarray(arr).resize((256, 256)))
                frames.append(arr)
            return frames
        finally:
            container.close()


# ──────────────────────────────────────────────────────────────────────────
# Frame → nested obs (Gr00tPolicy format)
# ──────────────────────────────────────────────────────────────────────────


def build_nested_obs(
    images_by_lerobot_col: dict[str, np.ndarray],
    task_text: str,
    video_keys: list[str],
    state_keys: list[str],
    state_dims: dict[str, int],
    language_key: str,
) -> dict:
    """Construct a Gr00tPolicy-format observation from one frame.

    ``video_keys`` come from ``policy.modality_configs["video"].modality_keys``;
    LeRobot column name is ``observation.images.<modality_key>``. Eagle ignores
    state, but the processor pipeline still wants every state key declared by
    the modality config — fill with zeros of the correct dim. Shapes (B=1, T=1, ...).
    """
    nested: dict = {"video": {}, "state": {}, "language": {}}

    for mkey in video_keys:
        col = LEROBOT_IMAGE_PREFIX + mkey
        if col not in images_by_lerobot_col:
            raise KeyError(
                f"Frame missing column {col!r}; available: {list(images_by_lerobot_col)}"
            )
        nested["video"][mkey] = images_by_lerobot_col[col][None, None, ...]

    for k in state_keys:
        d = state_dims.get(k, 1)
        nested["state"][k] = np.zeros((1, 1, d), dtype=np.float32)

    nested["language"][language_key] = [[task_text]]
    return nested


# ──────────────────────────────────────────────────────────────────────────
# Eagle forward
# ──────────────────────────────────────────────────────────────────────────


def _process_nested_obs(policy, nested_obs: dict):
    """Single-frame nested obs (B=1, T=1, ...) → processed message list (len=1)."""
    from gr00t.data.types import MessageType

    unbatched = policy._unbatch_observation(nested_obs)
    out = []
    for obs in unbatched:
        vla_step = policy._to_vla_step_data(obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
        out.append(policy.processor(messages))
    return out


def _eagle_pre_llm_only(eagle_model, pixel_values, input_ids):
    """Compute hidden_states[0] WITHOUT running LLM transformer layers.

    Eagle.forward() (modeling_eagle3_vl.py:213) normally does:
       1. text embed (get_input_embeddings)
       2. vision encoder + mlp1 projector (extract_feature)
       3. <image> token positions ← vit_embeds
       4. language_model(inputs_embeds=...) — full LLM forward (28 Qwen3 layers)
       5. return hidden_states from LLM

    We only want hidden_states[0] = post-step-3 sequence. Steps 1-3 only.
    Skips step 4 entirely → ~5-10× faster on Qwen3-1.7B backbone.
    """
    input_embeds = eagle_model.language_model.get_input_embeddings()(input_ids)
    vit_embeds = eagle_model.extract_feature(pixel_values, image_flags=None)

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)
    flat_ids = input_ids.reshape(B * N)
    selected = flat_ids == eagle_model.image_token_index
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
    except Exception:
        n_tok = int(selected.sum().item())
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_tok]
    return input_embeds.reshape(B, N, C)


# ThreadPool 으로 batch 안 frame 별 ``policy.processor`` 호출을 병렬 실행
# (huggingface tokenizer + NumPy/Pillow image 전처리는 GIL 해제 코드).
_PROCESSOR_THREAD_POOL = None


def _get_processor_pool(num_workers: int):
    global _PROCESSOR_THREAD_POOL
    if _PROCESSOR_THREAD_POOL is None or _PROCESSOR_THREAD_POOL._max_workers != num_workers:
        import concurrent.futures
        if _PROCESSOR_THREAD_POOL is not None:
            _PROCESSOR_THREAD_POOL.shutdown(wait=False)
        _PROCESSOR_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    return _PROCESSOR_THREAD_POOL


def eagle_pre_llm_forward_batched(
    policy, nested_obs_list: list[dict], device: str, cpu_workers: int = 4,
) -> torch.Tensor:
    """Eagle backbone (vision encoder + embed + image-token merge) over N frames in one
    GPU step. **LLM transformer layer skip** + **CPU prep parallel via ThreadPool**.

    Each ``nested_obs_list[i]`` is a single-frame nested obs (B=1, T=1, ...). N frame 의
    ``policy.processor`` 호출은 ``_get_processor_pool(cpu_workers)`` thread 들에서 동시
    실행되어 CPU 측 가속. Returns ``[N, hidden_dim]`` fp32 cpu tensor.
    """
    import tree as dm_tree
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    if cpu_workers > 1 and len(nested_obs_list) > 1:
        pool = _get_processor_pool(cpu_workers)
        # 각 nested → processed messages (list[dict]).
        futures = [pool.submit(_process_nested_obs, policy, n) for n in nested_obs_list]
        processed: list = []
        for f in futures:
            processed.extend(f.result())
    else:
        processed = []
        for nested in nested_obs_list:
            processed.extend(_process_nested_obs(policy, nested))

    collated = policy.collate_fn(processed)
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    batch = collated["inputs"]

    backbone = policy.model.backbone
    target_device = next(backbone.model.parameters()).device
    target_dtype = torch.bfloat16

    def to_device(x):
        if not torch.is_tensor(x):
            return x
        if torch.is_floating_point(x):
            return x.to(device=target_device, dtype=target_dtype)
        return x.to(device=target_device)

    keys = ["input_ids", "attention_mask", "pixel_values"]
    vl_input = {k: dm_tree.map_structure(to_device, batch[k]) for k in keys}

    backbone.set_frozen_modules_to_eval_mode()
    with torch.inference_mode():
        pre_llm = _eagle_pre_llm_only(
            backbone.model,
            pixel_values=vl_input["pixel_values"],
            input_ids=vl_input["input_ids"],
        )                                                    # [N, T_vl, D]
    attn_mask = (vl_input["attention_mask"] == 1).to(pre_llm.dtype)

    mask = attn_mask.unsqueeze(-1)
    z = (pre_llm * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [N, D]
    return z.float().cpu()


# ──────────────────────────────────────────────────────────────────────────
# Per-task extraction
# ──────────────────────────────────────────────────────────────────────────


def extract_one_task(
    task_name: str,
    lerobot_root: Path,
    save_path: Path,
    policy,
    video_keys: list[str],
    state_keys: list[str],
    state_dims: dict[str, int],
    language_key: str,
    device: str,
    override: bool,
    batch_size: int = 32,
    cpu_workers: int = 4,
    max_episodes: int | None = None,
) -> None:
    """Per-task extraction.

    Buffer 가 ``batch_size`` 채워질 때마다 Eagle (vision encoder + embed merge,
    LLM transformer layer skip) 를 1번 forward. CPU 측 frame-별 ``policy.processor``
    호출은 ``cpu_workers`` 개 thread 로 병렬화 (GIL 해제 가능한 부분에서만 효과).
    """
    out_path = save_path / task_name / "embeddings.pt"
    if out_path.exists() and not override:
        print(f"[skip] {task_name}: {out_path} exists (use --override to redo)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = RoboCasaV21Reader(lerobot_root)
    eps_to_use = reader.episodes
    if max_episodes is not None and max_episodes > 0:
        eps_to_use = list(reader.episodes)[:max_episodes]
    print(
        f"[task={task_name}] episodes={len(eps_to_use)}/{reader.total_episodes} "
        f"frames={reader.total_frames} batch_size={batch_size} cpu_workers={cpu_workers}"
        f"{' (capped)' if max_episodes else ''}"
    )

    embeddings: dict[int, torch.Tensor] = {}
    buf_nested: list[dict] = []
    buf_abs_idx: list[int] = []

    def flush():
        if not buf_nested:
            return
        zs = eagle_pre_llm_forward_batched(
            policy, buf_nested, device, cpu_workers=cpu_workers,
        )
        for i, ai in enumerate(buf_abs_idx):
            embeddings[int(ai)] = zs[i]
        buf_nested.clear()
        buf_abs_idx.clear()

    pbar = tqdm(eps_to_use, desc=task_name)
    for ep in pbar:
        ep_idx = int(ep["episode_index"])
        for abs_idx, _t, task_text, images in reader.iter_episode_frames(ep_idx):
            nested = build_nested_obs(
                images_by_lerobot_col=images,
                task_text=task_text,
                video_keys=video_keys,
                state_keys=state_keys,
                state_dims=state_dims,
                language_key=language_key,
            )
            buf_nested.append(nested)
            buf_abs_idx.append(abs_idx)
            if len(buf_nested) >= batch_size:
                flush()
        pbar.set_postfix(ep=ep_idx, ep_len=int(ep["length"]), cached=len(embeddings))
    flush()  # 남은 frame 처리

    print(f"[save] {task_name}: {len(embeddings)} frames → {out_path}")
    torch.save(embeddings, out_path)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def load_policy(model_path: str, embodiment_tag_name: str, device: str):
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    embodiment_tag = EmbodimentTag[embodiment_tag_name]
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
        strict=False,
    )
    policy.processor.eval()

    # Saved processor_config.json 은 pretrain 컨벤션 (res256_image_*) 으로 modality 가
    # baked-in 되어 있다. `register_modality_config` 는 MODALITY_CONFIGS 모듈 글로벌만
    # 갱신하므로 Gr00tPolicy 가 saved JSON 으로 빌드한 processor.modality_configs 에는
    # 반영되지 않는다. 여기서 명시적으로 override → LeRobot v2.1 native key 컨벤션 사용.
    emb_value = embodiment_tag.value
    if emb_value in MODALITY_CONFIGS:
        new_cfg = MODALITY_CONFIGS[emb_value]
        policy.processor.modality_configs[emb_value] = new_cfg
        policy.modality_configs = new_cfg
        policy.language_key = new_cfg["language"].modality_keys[0]
        print(
            f"[modality] override processor[{emb_value}] → "
            f"video_keys={list(new_cfg['video'].modality_keys)}, "
            f"language_key={policy.language_key!r}"
        )
    else:
        print(
            f"[modality] WARN: {emb_value} not in MODALITY_CONFIGS — "
            f"using saved processor_config.json 그대로"
        )
    return policy, embodiment_tag


def collect_state_meta(policy, embodiment_tag) -> tuple[list[str], dict[str, int]]:
    state_keys = list(policy.modality_configs["state"].modality_keys)
    state_dims: dict[str, int] = {}
    try:
        norm = policy.processor.state_action_processor.norm_params[embodiment_tag.value]["state"]
        for k in state_keys:
            d = norm[k]["dim"]
            state_dims[k] = int(d) if hasattr(d, "__int__") else int(d.item())
    except Exception as e:
        print(f"[warn] could not read state dims from norm_params: {e}; defaulting to 1 each")
        for k in state_keys:
            state_dims[k] = 1
    return state_keys, state_dims


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Eagle pre-LLM embeddings from RoboCasa LeRobot v2.1 data."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--save_path", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--modality_config_path", type=Path, default=DEFAULT_MODALITY_CONFIG)
    parser.add_argument("--embodiment_tag", type=str, default="ROBOCASA_PANDA_OMRON")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--override", action="store_true")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Frame batch for Eagle forward. 32~64 안전 (80GB A100 기준).",
    )
    parser.add_argument(
        "--cpu_workers", type=int, default=4,
        help="ThreadPoolExecutor workers for parallel per-frame `policy.processor` calls. "
             "GIL 한계로 ~4 cores 정도가 효과. 1 이면 sequential.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Cap 추출 episode 수 per task (앞에서부터 N개). None 이면 전체.",
    )
    args = parser.parse_args()

    register_modality_config(args.modality_config_path)

    pairs = discover_lerobot_roots(args.data_root, args.tasks)
    print(f"[discover] {len(pairs)} datasets:")
    for t, p in pairs:
        print(f"  - {t}: {p}")

    print(f"[load] policy from {args.model_path} ({args.embodiment_tag})")
    policy, embodiment_tag = load_policy(args.model_path, args.embodiment_tag, args.device)

    video_keys = list(policy.modality_configs["video"].modality_keys)
    state_keys, state_dims = collect_state_meta(policy, embodiment_tag)
    language_key = policy.language_key
    print(f"[meta] video_keys={video_keys}")
    print(f"[meta] state_keys={state_keys}")
    print(f"[meta] state_dims={state_dims}")
    print(f"[meta] language_key={language_key!r}")

    args.save_path.mkdir(parents=True, exist_ok=True)
    for task_name, lerobot_root in pairs:
        extract_one_task(
            task_name=task_name,
            lerobot_root=lerobot_root,
            save_path=args.save_path,
            policy=policy,
            video_keys=video_keys,
            state_keys=state_keys,
            state_dims=state_dims,
            language_key=language_key,
            device=args.device,
            override=args.override,
            batch_size=args.batch_size,
            cpu_workers=args.cpu_workers,
            max_episodes=args.max_episodes,
        )

    print("[done]")


if __name__ == "__main__":
    main()
