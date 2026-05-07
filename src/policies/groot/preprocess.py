"""GR00T 입력 이미지 전처리.

학습 (adapter) 과 서빙 (`scripts/serve/groot.py`) 양쪽이 동일한 변환을 써야
모델이 본 분포가 일관되게 유지됨. 따라서 단일 source of truth 로 분리.

GR00T native (`GrootRoboCasaEnv.process_img`) 와 동일한 로직:
정사각형 패딩 → cv2.resize INTER_AREA → (256, 256, 3) uint8.
"""

from __future__ import annotations

import base64
import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# GR00T N1 표준 입력 해상도. modality key prefix `res256_` 도 이 값에 대응.
FINAL_IMAGE_RESOLUTION: Tuple[int, int] = (256, 256)


def process_img(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 → 256x256x3 uint8 (square pad + resize).

    GR00T 학습 시 사용된 `GrootRoboCasaEnv.process_img` 과 동일 동작.
    H≠W 면 max(H,W) 정사각형으로 zero-pad 후 INTER_AREA 로 리사이즈.
    이미 256x256 이고 square 면 deep copy 만 해서 반환.
    """
    h, w, _ = img.shape
    if h != w:
        dim = max(h, w)
        y_offset = (dim - h) // 2
        x_offset = (dim - w) // 2
        img = np.pad(img, ((y_offset, y_offset), (x_offset, x_offset), (0, 0)))
    if img.shape[:2] != FINAL_IMAGE_RESOLUTION:
        img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
    return np.copy(img)


def decode_b64_image(b64_str: str) -> np.ndarray:
    """base64 PNG → HxWx3 uint8 numpy. (서빙 layer 전용)"""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))
