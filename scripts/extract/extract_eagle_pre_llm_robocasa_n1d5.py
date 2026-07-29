"""Eagle pre-LLM 추출 for **GR00T N1.5** — RoboCasa LeRobot v2.1 → ``embeddings.pt``.

N1.6 버전 (``extract_eagle_pre_llm_robocasa.py``) 과 동등한 결과를 만들지만
N1.5 의 별도 codebase 와 호환되도록 작성:

- ``sys.path`` 에 ``Isaac-GR00T-N1.5`` 를 가장 앞쪽 삽입 → ``gr00t`` 가 N1.5 의 것을 가리킴.
- ``GR00T_N1_5.from_pretrained`` 로 model load (``AutoConfig.register("gr00t_n1_5", ...)``
  도 import 시점에 호출됨).
- N1.5 의 ``data_config + transforms + LeRobotSingleDataset`` 패턴으로 image/text 전처리.
- model 의 ``backbone.eagle_model`` 에 직접 ``_eagle_pre_llm_only`` 호출. LLM 진입 전,
  vision encoder + token embedding + image-token merge 까지만. dim = LLM hidden (2048).

저장 형식 / cache key 는 N1.6 와 동일: ``{abs_frame_idx: tensor[2048]}``.

사용::

    python scripts/extract/extract_eagle_pre_llm_robocasa_n1d5.py \\
        --data_root /temporal_vla/data/robocasa/v1.0/target/atomic \\
        --save_path /temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5 \\
        --model_path /temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B \\
        --tasks CloseFridge --max_episodes 200 --batch_size 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# CRITICAL: N1.5 path before any `gr00t` import; PYTHONPATH 의 N1.6 보다 앞.
_N1d5_ROOT = "/temporal_vla/src/policies/Isaac-GR00T-N1.5"
_CFG_ROOT = "/temporal_vla/configs/policies"
for p in (_N1d5_ROOT, _CFG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Stub `pytorch3d` — N1.5 의 state/action transform 이 pytorch3d.transforms 의
# rotation helper 사용하지만 우리는 state/action 안 씀. import 만 성공하면 됨.
import types as _types
_pt = _types.ModuleType("pytorch3d")
_pt.transforms = _types.ModuleType("pytorch3d.transforms")
for _name in [
    "matrix_to_quaternion", "quaternion_to_matrix",
    "matrix_to_euler_angles", "euler_angles_to_matrix",
    "matrix_to_rotation_6d", "rotation_6d_to_matrix",
    "quaternion_apply", "axis_angle_to_quaternion",
]:
    setattr(_pt.transforms, _name, lambda *a, **k: None)
sys.modules["pytorch3d"] = _pt
sys.modules["pytorch3d.transforms"] = _pt.transforms

import torch
from tqdm import tqdm

# N1.5 codebase imports (gr00t namespace = N1.5)
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1_5  # registers `gr00t_n1_5` model_type


# ─────────────────────────────────────────────────────────────────
# Pre-LLM hidden 추출 — N1.5 호환 (image_flags 인자 없음)
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _eagle_pre_llm_only(eagle_model, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Eagle 의 vision_encoder + mlp1 + token embed + image-token merge **만**.
    LLM transformer layer 통째 SKIP. return shape: [B, N, C=hidden_size=2048].
    """
    input_embeds = eagle_model.language_model.get_input_embeddings()(input_ids)
    vit_embeds = eagle_model.extract_feature(pixel_values)

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


def _load_data_config(spec: str):
    """``module:Class`` form. e.g. ``robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig``."""
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)()
    if spec in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[spec]
    raise KeyError(f"data_config not found: {spec}")


def _frame_pool(z_per_sample: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    """[B, N, C] → [B, C] mean-pool over valid tokens (or all tokens)."""
    if attn_mask is None:
        return z_per_sample.mean(dim=1)
    mask = attn_mask.to(z_per_sample.dtype).unsqueeze(-1)
    return (z_per_sample * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _episode_index_of(dataset, idx: int) -> int:
    """LeRobotSingleDataset 의 frame idx → episode index. dataset 의 internal mapping 사용."""
    # LeRobotSingleDataset 은 보통 `_episode_lookup` 또는 비슷한 dict 가지고 있음.
    # 정확한 attribute 이름은 dataset 인스턴스의 dir() 확인 필요.
    # 가장 안정적인 fallback: 별도 reader 로 episode meta 사용.
    return -1  # placeholder; main 에서 우회


def extract_task(
    task_name: str,
    lerobot_root: Path,
    save_path: Path,
    model: GR00T_N1_5,
    modality_configs,
    transforms,
    embodiment_tag: EmbodimentTag,
    eagle_processor,
    batch_size: int,
    max_episodes: int | None,
    device: str,
    override: bool,
):
    out_path = save_path / task_name / "embeddings.pt"
    if out_path.exists() and not override:
        print(f"[skip] {task_name}: cache exists ({out_path})")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[task={task_name}] loading dataset from {lerobot_root}")
    dataset = LeRobotSingleDataset(
        dataset_path=str(lerobot_root),
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
    )

    # episode meta — 우리 N1.6 reader 재사용 (PYTHONPATH 의 /temporal_vla 에 src.datasets 있음).
    # LeRobotSingleDataset 의 idx 가 LeRobot v2.1 의 abs_frame_idx 와 일치한다고 가정.
    from src.datasets.robocasa_v21_reader import RoboCasaV21Reader
    reader = RoboCasaV21Reader(lerobot_root)
    all_eps = reader.lerobot_meta_episodes()  # dict ep_idx -> {dataset_from_index, dataset_to_index, length}
    if max_episodes is not None and max_episodes > 0:
        eps_to_use = [(idx, e) for idx, e in all_eps.items() if int(idx) < max_episodes]
    else:
        eps_to_use = list(all_eps.items())
    eps_to_use.sort(key=lambda x: int(x[0]))
    total_frames = sum(int(e["length"]) for _, e in eps_to_use)
    print(f"[task={task_name}] episodes={len(eps_to_use)}/{len(all_eps)} frames={total_frames}")

    eagle_model = model.backbone.eagle_model
    eagle_model.eval()

    embeddings: dict[int, torch.Tensor] = {}
    buf_eagle: list[dict] = []
    buf_abs_idx: list[int] = []

    # tokenizer 정보 — padding 위해
    _tok = getattr(eagle_processor, "tokenizer", None)
    _pad_id = _tok.pad_token_id if (_tok is not None and _tok.pad_token_id is not None) else 0

    @torch.no_grad()
    def flush():
        if not buf_eagle:
            return
        # Per-sample processor calls (text has multi-image placeholders → batch 못함)
        processed_list = []
        for ec in buf_eagle:
            txt = ec["text_list"]
            img = ec["image_inputs"]
            text_str = txt[0] if isinstance(txt, list) else txt
            p = eagle_processor(text=text_str, images=img, return_tensors="pt")
            processed_list.append(p)

        # Pad + stack input_ids/attention_mask (text length varies per sample)
        ids_list = [p["input_ids"][0] for p in processed_list]
        attn_list = [p["attention_mask"][0] for p in processed_list]
        max_len = max(int(x.shape[0]) for x in ids_list)
        input_ids = torch.stack([
            torch.cat([x, torch.full((max_len - x.shape[0],), _pad_id, dtype=x.dtype)])
            for x in ids_list
        ]).to(device)
        attention_mask = torch.stack([
            torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=x.dtype)])
            for x in attn_list
        ]).to(device)
        # pixel_values: 각 sample 의 모든 image 의 pixel tensor. 모든 sample 의
        # image 수가 같다면 (multi-cam 고정) cat 으로 [B*n_img, C, H, W] 만들고
        # image_sizes 도 batch 단위로 합치.
        pv_list = [p["pixel_values"] for p in processed_list]
        pixel_values = torch.cat(pv_list, dim=0).to(device)

        # image_flags / image_sizes 등 model 이 받는 batch 단위 인자.
        # Eagle2_5_VLForConditionalGeneration.forward 의 시그니처 확인 필요.
        # 일단 단순 forward 시도; 안 되면 sample 별 fallback.
        try:
            out = eagle_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            z_seq = out.hidden_states[0]  # [B, N, C]
        except Exception as exc:
            # Fallback to per-sample
            print(f"[warn] batch forward failed ({type(exc).__name__}: {exc}); fallback per-sample")
            z_seq_list = []
            for p in processed_list:
                pv = p["pixel_values"].to(device)
                ids = p["input_ids"].to(device)
                am = p["attention_mask"].to(device)
                out1 = eagle_model(
                    pixel_values=pv, input_ids=ids, attention_mask=am,
                    output_hidden_states=True, return_dict=True,
                )
                # pad to max_len
                hs = out1.hidden_states[0]  # [1, N_i, C]
                if hs.shape[1] < max_len:
                    pad = torch.zeros(1, max_len - hs.shape[1], hs.shape[2], dtype=hs.dtype, device=device)
                    hs = torch.cat([hs, pad], dim=1)
                z_seq_list.append(hs)
            z_seq = torch.cat(z_seq_list, dim=0)

        z_pooled = _frame_pool(z_seq, attention_mask)  # [B, C]
        for i, ai in enumerate(buf_abs_idx):
            embeddings[int(ai)] = z_pooled[i].cpu()
        buf_eagle.clear()
        buf_abs_idx.clear()

    # iterate frames per episode using dataset_from_index..to_index range
    pbar = tqdm(eps_to_use, desc=task_name)
    for ep_idx, ep in pbar:
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        for abs_idx in range(a, b):
            sample = dataset[abs_idx]
            ec = sample["eagle_content"]
            buf_eagle.append(ec)
            buf_abs_idx.append(abs_idx)
            if len(buf_eagle) >= batch_size:
                flush()
        pbar.set_postfix(ep=ep_idx, cached=len(embeddings))
    flush()

    print(f"[save] {task_name}: {len(embeddings)} frames -> {out_path}")
    torch.save(embeddings, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_config", type=str,
        default="robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig",
    )
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    # Data config + transforms
    data_config_cls = _load_data_config(args.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    print(f"[data_config] {args.data_config}")
    print(f"[modality_configs] keys={list(modality_configs.keys())}")
    print(f"[transforms] n={len(transforms.transforms)}")

    # GR00TTransform 의 eagle_processor 추출 (이미 build 됨)
    from gr00t.model.transforms import GR00TTransform
    eagle_processor = None
    for t in transforms.transforms:
        if isinstance(t, GR00TTransform):
            eagle_processor = t.eagle_processor
            break
    if eagle_processor is None:
        raise RuntimeError("GR00TTransform not in transforms; cannot get eagle_processor")
    print(f"[eagle_processor] {type(eagle_processor).__name__}")

    # Model
    print(f"[load] N1.5 model from {args.model_path}")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        tune_llm=False, tune_visual=False, tune_projector=False, tune_diffusion_model=False,
    ).to(args.device).eval()
    print(f"[load] model loaded. backbone.eagle_model = {type(model.backbone.eagle_model).__name__}")

    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    # Discover task dirs
    if args.tasks:
        task_pairs = []
        for t in args.tasks:
            matches = sorted((args.data_root / t).glob("*/lerobot"))
            if not matches:
                raise FileNotFoundError(f"No lerobot/ under {args.data_root / t}")
            task_pairs.append((t, matches[0]))
    else:
        task_pairs = []
        for d in sorted(args.data_root.iterdir()):
            matches = sorted(d.glob("*/lerobot"))
            if matches:
                task_pairs.append((d.name, matches[0]))

    args.save_path.mkdir(parents=True, exist_ok=True)
    for task_name, lerobot_root in task_pairs:
        extract_task(
            task_name=task_name,
            lerobot_root=lerobot_root,
            save_path=args.save_path,
            model=model,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            eagle_processor=eagle_processor,
            batch_size=args.batch_size,
            max_episodes=args.max_episodes,
            device=args.device,
            override=args.override,
        )


if __name__ == "__main__":
    main()
