"""Thin Gemini VLM client for INSIGHT-style primitive segmentation.

PROVENANCE
----------
Adapted from the INSIGHT VLA project (https://github.com/insight-vla/insight,
Apache-2.0): retry/backoff structure, OpenAI-style message schema, and JSON
parsing mirror ``src/insight/vlm_client.py``. Re-homed onto the google-genai
SDK and trimmed to what the segmentation pilot needs (text + inline images).

The public surface is a single ``call(messages, model=None) -> str`` method
where ``messages`` is OpenAI-style:

    [{"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
        ...
    ]}, ...]

``GeminiVLMClient`` converts that schema to a google-genai ``generate_content``
call (text parts + inline image parts decoded from the base64 data URLs).
``MockVLMClient`` returns a deterministic, schema-valid JSON string so the
whole pipeline can be exercised without an API key or network.

NOTE on google-genai: it is imported LAZILY (inside ``__init__`` / ``call``),
never at module import time, so this module imports cleanly even when
``google-genai`` is not installed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default Gemini model. Overridable per-client and per-call.
# NOTE: the INSIGHT paper used "gemini-3-flash" (google/gemini-3-flash-preview
# via Vertex). gemini-3-flash is not broadly available on the public
# generativelanguage API yet, so we default to the closest stable public model.
DEFAULT_MODEL = "gemini-2.5-flash"

# API-key file fallback (checked after the GEMINI_API_KEY env var).
_API_KEY_FILE = Path.home() / ".config" / "temporal_vla" / "gemini_api_key"

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w./+-]+);base64,(?P<data>.*)$", re.DOTALL)


# ---------------------------------------------------------------------------
# API-key resolution
# ---------------------------------------------------------------------------
def resolve_api_key() -> Optional[str]:
    """Resolve the Gemini API key.

    Order:
      1. env ``GEMINI_API_KEY``
      2. file ``~/.config/temporal_vla/gemini_api_key`` (whitespace stripped)

    Returns the key string, or ``None`` if neither source yields one.
    """
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        if _API_KEY_FILE.is_file():
            file_key = _API_KEY_FILE.read_text(encoding="utf-8").strip()
            if file_key:
                return file_key
    except OSError as e:  # pragma: no cover - filesystem edge case
        logger.warning("Could not read API key file %s: %s", _API_KEY_FILE, e)
    return None


def _parse_data_url(url: str) -> tuple[str, bytes]:
    """Split a ``data:<mime>;base64,<payload>`` URL into (mime, raw_bytes)."""
    m = _DATA_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"Unsupported image_url (expected data URL): {url[:64]!r}...")
    mime = m.group("mime")
    raw = base64.b64decode(m.group("data"))
    return mime, raw


# ---------------------------------------------------------------------------
# Real Gemini client (google-genai)
# ---------------------------------------------------------------------------
class GeminiVLMClient:
    """Minimal google-genai wrapper exposing an OpenAI-style ``call``.

    google-genai is imported lazily so importing this module never requires the
    SDK to be installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        retries: int = 4,
        backoff: float = 2.0,
        temperature: float = 0.1,
    ):
        self.model = model
        self.retries = retries
        self.backoff = backoff
        self.temperature = temperature

        key = api_key or resolve_api_key()
        if not key:
            raise ValueError(
                "No Gemini API key. Set GEMINI_API_KEY or write "
                f"{_API_KEY_FILE}, or use MockVLMClient / get_vlm_client(use_mock=True)."
            )
        # Lazy import — keeps the module importable without google-genai present.
        from google import genai  # noqa: PLC0415

        self._genai = genai
        self._client = genai.Client(api_key=key)

    def _to_genai_contents(self, messages: list) -> list:
        """Convert OpenAI-style messages -> google-genai content parts.

        We flatten all roles into a single ordered list of parts (text +
        inline image), which is what generate_content accepts for a single-turn
        multimodal request. System messages are prefixed as plain text.
        """
        from google.genai import types  # noqa: PLC0415 - lazy

        parts: list = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    prefix = "[system] " if role == "system" else ""
                    parts.append(types.Part.from_text(text=prefix + content))
                continue
            # content is a list of typed blocks
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        prefix = "[system] " if role == "system" else ""
                        parts.append(types.Part.from_text(text=prefix + text))
                elif btype == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    mime, raw = _parse_data_url(url)
                    parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                else:
                    logger.warning("Skipping unknown content block type %r", btype)
        return parts

    def call(self, messages: list, model: Optional[str] = None) -> str:
        """Run a single multimodal generation. Returns the response text.

        Retries with exponential backoff on transient errors / empty responses,
        mirroring INSIGHT's vlm_client.chat retry behaviour.
        """
        from google.genai import types  # noqa: PLC0415 - lazy

        use_model = model or self.model
        parts = self._to_genai_contents(messages)
        config = types.GenerateContentConfig(temperature=self.temperature)

        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                resp = self._client.models.generate_content(
                    model=use_model,
                    contents=parts,
                    config=config,
                )
            except Exception as e:  # network / quota / transient
                last_err = e
                logger.warning(
                    "[VLM] generate_content error (attempt %d/%d): %s",
                    attempt + 1, self.retries + 1, e,
                )
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise
            elapsed = time.time() - t0
            text = getattr(resp, "text", None)
            if text:
                logger.info("[VLM] %s response in %.1fs (%d chars)",
                            use_model, elapsed, len(text))
                return text
            # Empty / filtered response — treat as transient, retry.
            logger.warning(
                "[VLM] empty response in %.1fs (attempt %d/%d)",
                elapsed, attempt + 1, self.retries + 1,
            )
            if attempt < self.retries:
                time.sleep(self.backoff * (attempt + 1))
                continue
        raise ValueError(f"VLM returned no text after retries (last_err={last_err})")


# ---------------------------------------------------------------------------
# Mock client (no API key / dry-run plumbing)
# ---------------------------------------------------------------------------
class MockVLMClient:
    """Deterministic stand-in for GeminiVLMClient.

    Inspects the prompt text minimally to return a JSON string matching the
    schema the segmentation code expects:
      * a ``boundary_frame`` object when the prompt is a boundary-refine call;
      * otherwise a ``segments`` object (primitive_sequence localization).

    The output is fully deterministic (no randomness, no network) so it is safe
    for unit tests and dry-runs.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    @staticmethod
    def _gather_text(messages: list) -> str:
        chunks: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                chunks.append(content)
            else:
                for block in content:
                    if block.get("type") == "text":
                        chunks.append(block.get("text", ""))
        return "\n".join(chunks)

    @staticmethod
    def _extract_int(prompt: str, pattern: str, default: int) -> int:
        m = re.search(pattern, prompt)
        if m:
            try:
                return int(m.group(1))
            except (TypeError, ValueError):
                return default
        return default

    def call(self, messages: list, model: Optional[str] = None) -> str:  # noqa: ARG002
        prompt = self._gather_text(messages)

        # Boundary-refine call -> return a boundary_frame in the stated range.
        if "boundary_frame" in prompt:
            # Prompt appends: "Frames provided cover range [lo, hi)."
            m = re.search(r"range \[(\d+),\s*(\d+)\)", prompt)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                mid = (lo + hi) // 2
            else:
                mid = 0
            return json.dumps(
                {"boundary_frame": mid,
                 "reasoning": "mock: midpoint of provided window"}
            )

        # Plan/video labeling call -> return ordered `segments` covering the
        # episode. Determine the episode length and the primitive labels from
        # the prompt so the mock output is internally consistent.
        episode_length = self._extract_int(
            prompt, r"end_frame should be (\d+)", 0
        ) or self._extract_int(prompt, r"last ends at (\d+)", 0) \
            or self._extract_int(prompt, r"(\d+) total frames", 0) \
            or self._extract_int(prompt, r"frames \{?(\d+)", 40) or 40

        # Pull plan labels if a numbered/dashed plan list is present.
        labels = re.findall(r"^\s*\d+\.\s*(.+)$", prompt, re.MULTILINE)
        if not labels:
            labels = re.findall(r"^\s*-\s*(.+)$", prompt, re.MULTILINE)
        labels = [l.strip().strip('"') for l in labels if l.strip()]
        if not labels:
            labels = ["move gripper to object", "close gripper",
                      "lift upward", "open gripper"]
        # Cap to a sane number of segments.
        labels = labels[:6] if len(labels) > 6 else labels

        n = len(labels)
        step = max(1, episode_length // n)
        segments = []
        start = 0
        for i, lab in enumerate(labels):
            end = episode_length if i == n - 1 else min(episode_length, start + step)
            segments.append(
                {"start_frame": start, "end_frame": end, "primitive_label": lab}
            )
            start = end
        # Guarantee full coverage.
        segments[0]["start_frame"] = 0
        segments[-1]["end_frame"] = episode_length
        return json.dumps({"segments": segments,
                           "primitive_sequence": labels})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_vlm_client(use_mock: Optional[bool] = None, model: Optional[str] = None):
    """Return a VLM client.

    * ``use_mock=True``              -> always MockVLMClient.
    * ``use_mock`` falsy + a key     -> GeminiVLMClient.
    * ``use_mock`` falsy + NO key    -> MockVLMClient (with a clear warning).

    ``model`` overrides the default model for the real client.
    """
    if use_mock is True:
        print("[insight_seg] Using MockVLMClient (use_mock=True): no API calls.")
        return MockVLMClient(model=model or DEFAULT_MODEL)

    key = resolve_api_key()
    if not key:
        print(
            "[insight_seg] WARNING: no Gemini API key resolvable "
            "(set GEMINI_API_KEY or write ~/.config/temporal_vla/gemini_api_key). "
            "Falling back to MockVLMClient — outputs are placeholder JSON, NOT real."
        )
        return MockVLMClient(model=model or DEFAULT_MODEL)

    return GeminiVLMClient(api_key=key, model=model or DEFAULT_MODEL)
