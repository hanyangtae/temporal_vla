"""Image helpers shared by model serving paths."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def decode_b64_pil(b64_str: str) -> Image.Image:
    """Decode a base64 PNG/JPEG payload into an RGB PIL ``Image``."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def decode_b64_image(b64_str: str) -> np.ndarray:
    """Decode a base64 PNG/JPEG payload into an RGB ``uint8`` numpy array."""
    return np.array(decode_b64_pil(b64_str))
