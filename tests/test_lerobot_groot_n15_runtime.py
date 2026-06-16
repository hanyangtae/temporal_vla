from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "utils"))

from runtime import (  # noqa: E402
    _conversation_with_raw_language,
    _should_force_eager_attention,
    _unwrap_single_string_list_repr,
    default_hf_cache_root,
)


class TestDefaultHfCacheRoot(unittest.TestCase):
    def test_eval_runtime_shadow_module_is_removed(self):
        self.assertFalse(
            (REPO_ROOT / "scripts" / "eval" / "lerobot_groot_n15_runtime.py").exists()
        )
        self.assertFalse(
            (REPO_ROOT / "scripts" / "utils" / "lerobot_groot_n15_runtime.py").exists()
        )

    def test_honors_hf_home_when_set(self):
        with patch.dict(os.environ, {"HF_HOME": "/tmp/custom_hf"}, clear=True):
            self.assertEqual(default_hf_cache_root(), Path("/tmp/custom_hf"))

    def test_prefers_existing_direct_huggingface_cache_under_vla_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "huggingface").mkdir()
            with patch.dict(os.environ, {"VLA_CACHE_ROOT": str(root)}, clear=True):
                self.assertEqual(default_hf_cache_root(), root / "huggingface")

    def test_falls_back_to_datasets_huggingface_under_vla_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"VLA_CACHE_ROOT": str(root)}, clear=True):
                self.assertEqual(default_hf_cache_root(), root / "datasets" / "huggingface")


class TestFlashAttentionArchFallback(unittest.TestCase):
    def test_forces_eager_when_flash_attn_arch_excludes_current_gpu(self):
        with patch.dict(os.environ, {"FLASH_ATTN_CUDA_ARCHS": "120"}, clear=True):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
                    self.assertTrue(_should_force_eager_attention())

    def test_keeps_flash_attention_when_arch_matches_current_gpu(self):
        with patch.dict(os.environ, {"FLASH_ATTN_CUDA_ARCHS": "120"}, clear=True):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
                    self.assertFalse(_should_force_eager_attention())

    def test_env_override_forces_eager_attention(self):
        with patch.dict(os.environ, {"TEMPORAL_VLA_GROOT_EAGER_ATTENTION": "1"}, clear=True):
            self.assertTrue(_should_force_eager_attention())


class TestRawLanguagePatch(unittest.TestCase):
    def test_unwraps_single_string_list_repr(self):
        self.assertEqual(
            _unwrap_single_string_list_repr("['Close the lid.']"),
            "Close the lid.",
        )

    def test_leaves_plain_text_unchanged(self):
        self.assertEqual(
            _unwrap_single_string_list_repr("Close the lid."),
            "Close the lid.",
        )

    def test_conversation_text_is_copied_with_raw_language(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": object()},
                    {"type": "text", "text": "['Close the lid.']"},
                ],
            }
        ]

        converted = _conversation_with_raw_language(conversation)

        self.assertEqual(converted[0]["content"][1]["text"], "Close the lid.")
        self.assertEqual(conversation[0]["content"][1]["text"], "['Close the lid.']")


if __name__ == "__main__":
    unittest.main()
