#!/usr/bin/env python
"""Runtime helpers for repo-local LeRobot GR00T N1.5 wrappers.

These helpers are process-local compatibility glue. They intentionally avoid
editing the ``lerobot`` submodule.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def avoid_path_shadowing(path: Path) -> None:
    """Remove one directory from ``sys.path`` if it shadows installed packages."""
    resolved_path = path.resolve()
    cleaned = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)
            continue
        try:
            if Path(entry).resolve() == resolved_path:
                continue
        except OSError:
            pass
        cleaned.append(entry)
    sys.path[:] = cleaned


def default_hf_cache_root() -> Path:
    """Return the repo cache doc's Hugging Face cache location."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)

    cache_root = os.environ.get("VLA_CACHE_ROOT")
    if cache_root:
        root = Path(cache_root)
        direct_hf = root / "huggingface"
        if direct_hf.exists():
            return direct_hf
        return root / "datasets" / "huggingface"
    return Path.home() / ".cache" / "temporal_vla" / "datasets" / "huggingface"


def setup_repo_local_hf_cache(root: Path | None = None) -> None:
    """Use the repo cache layout for Hugging Face/Xet downloads."""
    root = root or default_hf_cache_root()
    hub_cache = root / "hub"
    xet_cache = root / "xet"
    xdg_cache = root.parent / ".cache"

    for path in (root, hub_cache, xet_cache, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)


def setup_libero_config(assets_dir: Path, datasets_dir: Path) -> None:
    """Avoid LIBERO's first-import interactive dataset prompt."""
    import libero

    libero_root = Path(libero.__file__).resolve().parent / "libero"
    config_dir = Path.home() / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {libero_root}",
                f"bddl_files: {libero_root / 'bddl_files'}",
                f"init_states: {libero_root / 'init_files'}",
                f"datasets: {datasets_dir}",
                f"assets: {assets_dir}",
                "",
            ]
        )
    )


def _truthy_env(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _cuda_arch_tokens(value: str) -> set[str]:
    tokens = value.replace(",", ";").replace(" ", ";").split(";")
    normalized = set()
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        token = token.replace("sm_", "").replace("compute_", "").replace(".", "")
        normalized.add(token)
    return normalized


def _should_force_eager_attention() -> bool:
    override = _truthy_env("TEMPORAL_VLA_GROOT_EAGER_ATTENTION")
    if override is not None:
        return override

    archs = os.environ.get("FLASH_ATTN_CUDA_ARCHS")
    if not archs:
        return False

    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    current = f"{major}{minor}"
    return current not in _cuda_arch_tokens(archs)


def _set_attention_impl(config, impl: str) -> None:
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = impl
    if hasattr(config, "vision_config"):
        _set_attention_impl(config.vision_config, impl)
    if hasattr(config, "text_config"):
        _set_attention_impl(config.text_config, impl)


def _unwrap_single_string_list_repr(value):
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
        return parsed[0]
    return value


def _conversation_with_raw_language(conversation):
    changed = False
    converted = []
    for message in conversation:
        if not isinstance(message, dict):
            converted.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        new_content = []
        message_changed = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                raw_text = _unwrap_single_string_list_repr(item.get("text"))
                if raw_text != item.get("text"):
                    item = dict(item)
                    item["text"] = raw_text
                    message_changed = True
            new_content.append(item)
        if message_changed:
            message = dict(message)
            message["content"] = new_content
            changed = True
        converted.append(message)
    return converted if changed else conversation


def patch_groot_runtime() -> None:
    """Apply process-local GR00T N1.5 compatibility patches."""
    from torch.distributions import Beta as TorchBeta

    import lerobot.policies.groot.action_head.flow_matching_action_head as flow_matching
    import lerobot.policies.groot.groot_n1 as groot_n1
    import lerobot.policies.groot.processor_groot as processor_groot

    # torch 2.10 validates distribution args on the meta device during HF model
    # init. That trips Tensor.item() before checkpoint weights are loaded.
    flow_matching.Beta = lambda a, b: TorchBeta(a, b, validate_args=False)

    # transformers 5 expects this attribute on PreTrainedModel subclasses.
    if not hasattr(groot_n1.GR00TN15, "all_tied_weights_keys"):
        groot_n1.GR00TN15.all_tied_weights_keys = {}

    if not getattr(groot_n1.EagleBackbone, "_temporal_vla_eager_attention_patch", False):
        original_eagle_init = groot_n1.EagleBackbone.__init__

        def eagle_init_with_attention_fallback(self, *args, **kwargs):
            original_eagle_init(self, *args, **kwargs)
            if _should_force_eager_attention() and hasattr(self, "eagle_model"):
                _set_attention_impl(self.eagle_model.config, "eager")
                if hasattr(self.eagle_model, "vision_model"):
                    _set_attention_impl(self.eagle_model.vision_model.config, "eager")
                print(
                    "[GROOT] Forcing Eagle attention to eager because "
                    "FLASH_ATTN_CUDA_ARCHS excludes this GPU."
                )

        groot_n1.EagleBackbone.__init__ = eagle_init_with_attention_fallback
        groot_n1.EagleBackbone._temporal_vla_eager_attention_patch = True

    original_build = processor_groot._build_eagle_processor

    def build_eagle_processor_with_tensor_images(*args, **kwargs):
        proc = original_build(*args, **kwargs)
        image_processor_cls = type(proc.image_processor)
        if not getattr(image_processor_cls, "_temporal_vla_return_tensors_patch", False):
            original_call = image_processor_cls.__call__

            def call_with_return_tensors(self, *call_args, **call_kwargs):
                call_kwargs.setdefault("return_tensors", "pt")
                return original_call(self, *call_args, **call_kwargs)

            image_processor_cls.__call__ = call_with_return_tensors
            image_processor_cls._temporal_vla_return_tensors_patch = True
        if (
            _truthy_env("TEMPORAL_VLA_GROOT_RAW_LANGUAGE")
            and not getattr(proc, "_temporal_vla_raw_language_patch", False)
        ):
            original_template = proc.apply_chat_template

            def apply_chat_template_with_raw_language(conversation, *call_args, **call_kwargs):
                return original_template(
                    _conversation_with_raw_language(conversation),
                    *call_args,
                    **call_kwargs,
                )

            proc.apply_chat_template = apply_chat_template_with_raw_language
            proc._temporal_vla_raw_language_patch = True
        return proc

    processor_groot._build_eagle_processor = build_eagle_processor_with_tensor_images
