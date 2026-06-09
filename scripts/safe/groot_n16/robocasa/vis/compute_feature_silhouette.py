#!/usr/bin/env python3
"""Compatibility entrypoint for ``seen18.py silhouette``."""

from __future__ import annotations

import sys
from pathlib import Path

VIS_ROOT = Path(__file__).resolve().parent
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))

from analyses.silhouette import main  # noqa: E402


if __name__ == "__main__":
    main()
