"""pytest 루트 설정 — 프로젝트 루트를 sys.path 에 추가.

`from src.conceptor... import ...` 같은 절대 import 가 어느 디렉토리에서
pytest 를 실행하든 동작하도록 보장한다.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
