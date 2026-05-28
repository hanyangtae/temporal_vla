"""
UP-VLA 평가 스크립트 (Calvin 환경 측).

Python 3.8 호환. UP-VLA 코드를 임포트하지 않음.
calvin 컨테이너에서 실행:

# 통합 테스트 (debug dataset)
docker compose run --rm calvin python /temporal_vla/scripts/calvin_eval_upvla.py \
  --dataset-path /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset \
  --num-sequences 1 \
  --video-dir /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset/video \
  --num-videos 1

# 실제 평가 (full dataset)
docker compose run --rm calvin python /temporal_vla/scripts/calvin_eval_upvla.py \
  --dataset-path /cache/datasets/calvin/task_ABC_D \
  --num-sequences 1000 \
  --video-dir /cache/datasets/calvin/task_ABC_D/video
  --num-videos 10

upvla 서버(serve_upvla.py)가 먼저 실행 중이어야 한다.
"""

import argparse
import base64
import contextlib
import io
import json
import logging
import os

# Calvin-native imports (calvin 컨테이너에 설치됨)
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pyhash
import requests

sys.path.insert(0, "/temporal_vla/src/benchmarks/calvin/calvin_env")
from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── 인라인 유틸: policy_evaluation/utils.py에서 필요한 부분 ─────────────────

_hasher = pyhash.fnv1_32()


@contextlib.contextmanager
def _temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _get_env_state_for_initial_condition(
    initial_condition: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """초기 조건 dict → (robot_obs, scene_obs) numpy 배열."""
    # Calvin 로봇의 기본 홈 자세 (policy_evaluation/utils.py에서 복사).
    # 15개 값: [EE xyz(3)] [EE euler rpy(3)] [gripper_width(1)] [joint_angles(7)] [gripper_action(1)]
    # Calvin 평가 프로토콜의 고정 상수 — 변경하면 타 논문과 결과를 비교할 수 없다.
    robot_obs = np.array(
        [
            0.02586889,  # EE x
            -0.2313129,  # EE y
            0.5712808,  # EE z
            3.09045411,  # EE roll
            -0.02908596,  # EE pitch
            1.50013585,  # EE yaw
            0.07999963,  # gripper width
            -1.21779124,  # joint 0
            1.03987629,  # joint 1
            2.11978254,  # joint 2
            -2.34205014,  # joint 3
            -0.87015899,  # joint 4
            1.64119093,  # joint 5
            0.55344928,  # joint 6
            1.0,  # gripper action (1=open)
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = _hasher(str(initial_condition.values()))
    with _temp_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def _count_success(results: List[int]) -> List[float]:
    """결과 리스트 → 1~5 chain 성공률 리스트."""
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        step_success.append(n_success / len(results))
    return step_success


# ─── 인라인 유틸: policy_evaluation/multistep_sequences.py ───────────────────

# 태스크 카테고리 번호 (같은 숫자 = 같은 카테고리).
# 5-subtask 시퀀스를 구성할 때 동일 카테고리 태스크가 두 번 나오면 _check_sequence에서 거부된다.
# 예: rotate_red/blue/pink_block_* 은 모두 1번 → 시퀀스 내 회전 태스크는 1개만 허용.
# 이 제약으로 다양한 물체/동작을 고루 포함한 시퀀스만 평가에 사용된다.
_TASK_CATEGORIES = {
    "rotate_red_block_right": 1,
    "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1,
    "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1,
    "rotate_pink_block_left": 1,
    "push_red_block_right": 1,
    "push_red_block_left": 1,
    "push_blue_block_right": 1,
    "push_blue_block_left": 1,
    "push_pink_block_right": 1,
    "push_pink_block_left": 1,
    "move_slider_left": 2,
    "move_slider_right": 2,
    "open_drawer": 3,
    "close_drawer": 3,
    "lift_red_block_table": 4,
    "lift_red_block_slider": 5,
    "lift_red_block_drawer": 6,
    "lift_blue_block_table": 4,
    "lift_blue_block_slider": 5,
    "lift_blue_block_drawer": 6,
    "lift_pink_block_table": 4,
    "lift_pink_block_slider": 5,
    "lift_pink_block_drawer": 6,
    "place_in_slider": 7,
    "place_in_drawer": 7,
    "turn_on_lightbulb": 8,
    "turn_off_lightbulb": 8,
    "turn_on_led": 8,
    "turn_off_led": 8,
    "push_into_drawer": 9,
    "stack_block": 10,
    "unstack_block": 11,
}

_TASKS = {
    "rotate_red_block_right": [
        {
            "condition": {"red_block": "table", "grasped": 0},
            "effect": {"red_block": "table"},
        }
    ],
    "rotate_red_block_left": [
        {
            "condition": {"red_block": "table", "grasped": 0},
            "effect": {"red_block": "table"},
        }
    ],
    "rotate_blue_block_right": [
        {
            "condition": {"blue_block": "table", "grasped": 0},
            "effect": {"blue_block": "table"},
        }
    ],
    "rotate_blue_block_left": [
        {
            "condition": {"blue_block": "table", "grasped": 0},
            "effect": {"blue_block": "table"},
        }
    ],
    "rotate_pink_block_right": [
        {
            "condition": {"pink_block": "table", "grasped": 0},
            "effect": {"pink_block": "table"},
        }
    ],
    "rotate_pink_block_left": [
        {
            "condition": {"pink_block": "table", "grasped": 0},
            "effect": {"pink_block": "table"},
        }
    ],
    "push_red_block_right": [
        {
            "condition": {"red_block": "table", "grasped": 0},
            "effect": {"red_block": "table"},
        }
    ],
    "push_red_block_left": [
        {
            "condition": {"red_block": "table", "grasped": 0},
            "effect": {"red_block": "table"},
        }
    ],
    "push_blue_block_right": [
        {
            "condition": {"blue_block": "table", "grasped": 0},
            "effect": {"blue_block": "table"},
        }
    ],
    "push_blue_block_left": [
        {
            "condition": {"blue_block": "table", "grasped": 0},
            "effect": {"blue_block": "table"},
        }
    ],
    "push_pink_block_right": [
        {
            "condition": {"pink_block": "table", "grasped": 0},
            "effect": {"pink_block": "table"},
        }
    ],
    "push_pink_block_left": [
        {
            "condition": {"pink_block": "table", "grasped": 0},
            "effect": {"pink_block": "table"},
        }
    ],
    "move_slider_left": [
        {"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}
    ],
    "move_slider_right": [
        {"condition": {"slider": "left", "grasped": 0}, "effect": {"slider": "right"}}
    ],
    "open_drawer": [
        {"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}
    ],
    "close_drawer": [
        {"condition": {"drawer": "open", "grasped": 0}, "effect": {"drawer": "closed"}}
    ],
    "lift_red_block_table": [
        {
            "condition": {"red_block": "table", "grasped": 0},
            "effect": {"red_block": "grasped", "grasped": 1},
        }
    ],
    "lift_red_block_slider": [
        {
            "condition": {"red_block": "slider_left", "slider": "right", "grasped": 0},
            "effect": {"red_block": "grasped", "grasped": 1},
        },
        {
            "condition": {"red_block": "slider_right", "slider": "left", "grasped": 0},
            "effect": {"red_block": "grasped", "grasped": 1},
        },
    ],
    "lift_red_block_drawer": [
        {
            "condition": {"red_block": "drawer", "drawer": "open", "grasped": 0},
            "effect": {"red_block": "grasped", "grasped": 1},
        }
    ],
    "lift_blue_block_table": [
        {
            "condition": {"blue_block": "table", "grasped": 0},
            "effect": {"blue_block": "grasped", "grasped": 1},
        }
    ],
    "lift_blue_block_slider": [
        {
            "condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0},
            "effect": {"blue_block": "grasped", "grasped": 1},
        },
        {
            "condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0},
            "effect": {"blue_block": "grasped", "grasped": 1},
        },
    ],
    "lift_blue_block_drawer": [
        {
            "condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0},
            "effect": {"blue_block": "grasped", "grasped": 1},
        }
    ],
    "lift_pink_block_table": [
        {
            "condition": {"pink_block": "table", "grasped": 0},
            "effect": {"pink_block": "grasped", "grasped": 1},
        }
    ],
    "lift_pink_block_slider": [
        {
            "condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0},
            "effect": {"pink_block": "grasped", "grasped": 1},
        },
        {
            "condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0},
            "effect": {"pink_block": "grasped", "grasped": 1},
        },
    ],
    "lift_pink_block_drawer": [
        {
            "condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0},
            "effect": {"pink_block": "grasped", "grasped": 1},
        }
    ],
    "place_in_slider": [
        {
            "condition": {"red_block": "grasped", "slider": "right", "grasped": 1},
            "effect": {"red_block": "slider_right", "grasped": 0},
        },
        {
            "condition": {"red_block": "grasped", "slider": "left", "grasped": 1},
            "effect": {"red_block": "slider_left", "grasped": 0},
        },
        {
            "condition": {"blue_block": "grasped", "slider": "right", "grasped": 1},
            "effect": {"blue_block": "slider_right", "grasped": 0},
        },
        {
            "condition": {"blue_block": "grasped", "slider": "left", "grasped": 1},
            "effect": {"blue_block": "slider_left", "grasped": 0},
        },
        {
            "condition": {"pink_block": "grasped", "slider": "right", "grasped": 1},
            "effect": {"pink_block": "slider_right", "grasped": 0},
        },
        {
            "condition": {"pink_block": "grasped", "slider": "left", "grasped": 1},
            "effect": {"pink_block": "slider_left", "grasped": 0},
        },
    ],
    "place_in_drawer": [
        {
            "condition": {"red_block": "grasped", "drawer": "open", "grasped": 1},
            "effect": {"red_block": "drawer", "grasped": 0},
        },
        {
            "condition": {"blue_block": "grasped", "drawer": "open", "grasped": 1},
            "effect": {"blue_block": "drawer", "grasped": 0},
        },
        {
            "condition": {"pink_block": "grasped", "drawer": "open", "grasped": 1},
            "effect": {"pink_block": "drawer", "grasped": 0},
        },
    ],
    "stack_block": [
        {
            "condition": {"red_block": "grasped", "blue_block": "table", "grasped": 1},
            "effect": {
                "red_block": "stacked_top",
                "blue_block": "stacked_bottom",
                "grasped": 0,
            },
        },
        {
            "condition": {"red_block": "grasped", "pink_block": "table", "grasped": 1},
            "effect": {
                "red_block": "stacked_top",
                "pink_block": "stacked_bottom",
                "grasped": 0,
            },
        },
        {
            "condition": {"blue_block": "grasped", "red_block": "table", "grasped": 1},
            "effect": {
                "blue_block": "stacked_top",
                "red_block": "stacked_bottom",
                "grasped": 0,
            },
        },
        {
            "condition": {"blue_block": "grasped", "pink_block": "table", "grasped": 1},
            "effect": {
                "blue_block": "stacked_top",
                "pink_block": "stacked_bottom",
                "grasped": 0,
            },
        },
        {
            "condition": {"pink_block": "grasped", "red_block": "table", "grasped": 1},
            "effect": {
                "pink_block": "stacked_top",
                "red_block": "stacked_bottom",
                "grasped": 0,
            },
        },
        {
            "condition": {"pink_block": "grasped", "blue_block": "table", "grasped": 1},
            "effect": {
                "pink_block": "stacked_top",
                "blue_block": "stacked_bottom",
                "grasped": 0,
            },
        },
    ],
    "unstack_block": [
        {
            "condition": {
                "red_block": "stacked_top",
                "blue_block": "stacked_bottom",
                "grasped": 0,
            },
            "effect": {"red_block": "table", "blue_block": "table"},
        },
        {
            "condition": {
                "red_block": "stacked_top",
                "pink_block": "stacked_bottom",
                "grasped": 0,
            },
            "effect": {"red_block": "table", "pink_block": "table"},
        },
        {
            "condition": {
                "blue_block": "stacked_top",
                "red_block": "stacked_bottom",
                "grasped": 0,
            },
            "effect": {"blue_block": "table", "red_block": "table"},
        },
        {
            "condition": {
                "blue_block": "stacked_top",
                "pink_block": "stacked_bottom",
                "grasped": 0,
            },
            "effect": {"blue_block": "table", "pink_block": "table"},
        },
        {
            "condition": {
                "pink_block": "stacked_top",
                "red_block": "stacked_bottom",
                "grasped": 0,
            },
            "effect": {"pink_block": "table", "red_block": "table"},
        },
        {
            "condition": {
                "pink_block": "stacked_top",
                "blue_block": "stacked_bottom",
                "grasped": 0,
            },
            "effect": {"pink_block": "table", "blue_block": "table"},
        },
    ],
    "turn_on_lightbulb": [
        {"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}
    ],
    "turn_off_lightbulb": [
        {"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}
    ],
    "turn_on_led": [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
    "turn_off_led": [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
    "push_into_drawer": [
        {
            "condition": {
                "red_block": "table",
                "blue_block": ["slider_right", "slider_left"],
                "pink_block": ["slider_right", "slider_left"],
                "drawer": "open",
                "grasped": 0,
            },
            "effect": {"red_block": "drawer", "grasped": 0},
        },
        {
            "condition": {
                "blue_block": "table",
                "red_block": ["slider_right", "slider_left"],
                "pink_block": ["slider_right", "slider_left"],
                "drawer": "open",
                "grasped": 0,
            },
            "effect": {"blue_block": "drawer", "grasped": 0},
        },
        {
            "condition": {
                "pink_block": "table",
                "blue_block": ["slider_right", "slider_left"],
                "red_block": ["slider_right", "slider_left"],
                "drawer": "open",
                "grasped": 0,
            },
            "effect": {"pink_block": "drawer", "grasped": 0},
        },
    ],
}


def _check_condition(state: Dict, condition: Dict) -> bool:
    for k, v in condition.items():
        if isinstance(v, (str, int)):
            if state[k] != v:
                return False
        elif isinstance(v, list):
            if state[k] not in v:
                return False
    return True


def _valid_task(state: Dict, task: List[Dict]) -> List[Dict]:
    next_states = []
    for t in task:
        if _check_condition(state, t["condition"]):
            next_state = deepcopy(state)
            next_state.update(t["effect"])
            next_states.append(next_state)
    return next_states


def _check_sequence(state: Dict, seq: List[str]) -> bool:
    """주어진 초기 상태에서 seq의 5개 태스크가 순서대로 실행 가능한지 검증.

    두 가지 조건을 모두 만족해야 True:
      1. 각 태스크의 선행 조건(condition)이 이전 태스크 완료 후 상태와 일치 (물리적 실행 가능성)
      2. 시퀀스 내 모든 태스크의 카테고리가 서로 다름 (_TASK_CATEGORIES 중복 금지)
    """
    for task_name in seq:
        states = _valid_task(state, _TASKS[task_name])
        if len(states) != 1:
            return False
        state = states[0]
    categories = [_TASK_CATEGORIES[name] for name in seq]
    # 카테고리 중복 금지: 5개 카테고리가 모두 달라야 다양한 태스크 시퀀스가 됨
    return len(categories) == len(set(categories))


def _get_sequences_for_state(args: Tuple) -> List:
    state, num_sequences, i = args
    np.random.seed(i)
    seq_len = 5
    results = []
    while len(results) < num_sequences:
        seq = np.random.choice(list(_TASKS.keys()), size=seq_len, replace=False)
        if _check_sequence(state, seq):
            results.append(seq)
    return results


def _get_sequences(num_sequences: int = 1000) -> List[Tuple]:
    """Calvin 평가용 (initial_state, eval_sequence) 쌍 생성."""
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    # 물리적으로 유효한 초기 상태만 필터링:
    #   - 블록이 테이블에 1~2개만 있어야 함 (0개면 모든 블록이 슬라이더 → 비현실적)
    #   - slider_right, slider_left에 각각 1개 이하 (슬라이더 한 칸에 블록 2개는 불가)
    f = (
        lambda l: l.count("table") in [1, 2]
        and l.count("slider_right") < 2
        and l.count("slider_left") < 2
    )
    value_combinations = filter(f, product(*possible_conditions.values()))
    initial_states = [
        dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations
    ]

    num_per_state = [
        len(chunk)
        for chunk in np.array_split(range(num_sequences), len(initial_states))
    ]

    logger.info("Generating evaluation sequences...")
    with _temp_seed(0):
        with ProcessPoolExecutor() as executor:
            results_nested = list(
                executor.map(
                    _get_sequences_for_state,
                    zip(initial_states, num_per_state, range(len(initial_states))),
                )
            )
        results = [
            (state, tuple(seq.tolist()))
            for state, seqs in zip(initial_states, results_nested)
            for seq in seqs
        ]
        np.random.shuffle(results)
    logger.info("Done generating %d sequences.", len(results))
    return results


# ─── HTTP 클라이언트 (인라인, PIL만 의존) ────────────────────────────────────


def _encode_image_png(img: np.ndarray) -> str:
    """HxWx3 uint8 numpy → base64 PNG (무손실)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_b64png(b64_str: str) -> np.ndarray:
    """base64 PNG 문자열 → HxWx3 uint8 numpy."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


def _draw_border(
    frames: List[np.ndarray], success: bool, border: int = 3, repeat: int = 5
) -> List[np.ndarray]:
    """원본 draw_outcome()과 동일: 마지막 프레임에만 테두리를 그리고 repeat번 반복해서 붙인다.
    성공=초록(0,255,0), 실패=빨강(255,0,0).
    """
    color = (0, 255, 0) if success else (255, 0, 0)
    last = frames[-1].copy()
    last[:border, :] = color  # top
    last[-border:, :] = color  # bottom
    last[:, :border] = color  # left
    last[:, -border:] = color  # right
    return frames + [last] * repeat


def _draw_instruction(frames: List[np.ndarray], text: str) -> List[np.ndarray]:
    """원본 add_text() (utils.py L160)와 동일하게 instruction 텍스트를 프레임에 오버레이한다.
    cv2.putText 두 번: 검정 두꺼운 외곽선 → 흰 텍스트 (stroke 효과).
    위치: 좌하단 (1, height-10), 폰트 크기: 너비 기준 비례.
    """
    import cv2

    result = []
    for f in frames:
        img = f.copy()
        h, w = img.shape[:2]
        coord = (1, int(h - 10))
        font_scale = (0.7 / 500) * w
        cv2.putText(
            img,
            text,
            coord,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            coord,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        result.append(img)
    return result


def _save_video(frames: List[np.ndarray], path: Path, fps: int = 20):
    """프레임 리스트 → MP4 저장. imageio 없으면 GIF로 fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        imageio.mimwrite(str(path), frames, fps=fps)
    except ImportError:
        gif_path = path.with_suffix(".gif")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        logger.warning("imageio not available, saved as GIF: %s", gif_path)


def _predict(
    server_url: str,
    image_static: np.ndarray,
    instruction: str,
    image_gripper: Optional[np.ndarray] = None,
    return_images: bool = False,
    timeout: float = 60.0,
) -> Tuple:
    """POST /act → (actions [act_step, 7], recons_img or None, gen_img or None)."""
    payload = {
        "image_static": _encode_image_png(image_static),
        "instruction": instruction,
        "return_images": return_images,
    }
    if image_gripper is not None:
        payload["image_gripper"] = _encode_image_png(image_gripper)

    r = requests.post(f"{server_url}/act", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    actions = np.array(data["action"], dtype=np.float32)  # [act_step, 7]
    if return_images:
        return (
            actions,
            _decode_b64png(data["recons_image"]),
            _decode_b64png(data["gen_image"]),
        )
    return actions, None, None


def _wait_for_server(server_url: str, max_wait: float = 180.0, poll: float = 5.0):
    logger.info("Waiting for UP-VLA server at %s ...", server_url)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                info = r.json()
                logger.info("Server ready. act_step=%s", info.get("act_step"))
                return info
        except requests.ConnectionError:
            pass
        time.sleep(poll)
    raise TimeoutError(f"Server not ready after {max_wait}s")


# ─── Calvin env 셋업 ─────────────────────────────────────────────────────────


def _setup_egl(cuda_id: int = 0):
    if "EGL_VISIBLE_DEVICES" in os.environ:
        return
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    logger.info("EGL device: %d (CUDA %d)", egl_id, cuda_id)


def _get_raw_image(obs: Dict, key: str) -> np.ndarray:
    """Calvin raw obs에서 uint8 HxWx3 이미지 추출.

    Calvin env는 rgb_obs를 [-1, 1] 범위의 CHW tensor로 반환한다.
    원본(calvin_evaluate_upvla.py L233-237)과 동일하게 (img+1)/2*255 변환을 적용한다.
    """
    img = obs["rgb_obs"][key]
    if hasattr(img, "numpy"):  # tensor인 경우
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # CHW → HWC
    if img.dtype != np.uint8:
        # [-1, 1] → [0, 255] (원본과 동일한 변환)
        img = ((np.clip(img, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return img


def _step_env(env, action_7d: np.ndarray):
    """7D 연속 action → Calvin env.step() 실행.

    Calvin robot.py는 gripper_action in (-1, 1)을 강제 assert한다.
    7D array 형태는 유지하되 gripper(마지막 요소)만 이산화한다.
    """
    action = action_7d.copy()
    action[-1] = 1.0 if action[-1] > 0 else -1.0
    return env.step(action)  # (obs, reward, done, info)


# ─── 평가 루프 ───────────────────────────────────────────────────────────────


def _rollout(
    env,
    task_oracle,
    subtask: str,
    instruction: str,
    server_url: str,
    act_step: int,
    ep_len: int,
    record: bool = False,
    video_dir: Optional[Path] = None,
    seq_idx: int = 0,
    subtask_idx: int = 0,
) -> Tuple[bool, List[np.ndarray]]:
    """단일 subtask 롤아웃. (성공 여부, 프레임 리스트) 반환."""
    obs = env.get_obs()
    start_info = env.get_info()
    action_buffer = None
    frames = []
    prev_recons = None  # 이전 act_step의 VQ 재구성 이미지

    for step in range(ep_len):
        if step % act_step == 0:
            img_static = _get_raw_image(obs, "rgb_static")
            action_buffer, recons, gen = _predict(
                server_url, img_static, instruction, return_images=record
            )

            # PNG 저장: [이전 재구성 | 예측 이미지 | 현재 재구성] 나란히
            if record and video_dir is not None and prev_recons is not None:
                combined = np.concatenate([prev_recons, gen, recons], axis=1)
                png_dir = video_dir / "input_predict_truth"
                png_dir.mkdir(parents=True, exist_ok=True)
                png_path = png_dir / "seq{:04d}_sub{}_step{:03d}.png".format(
                    seq_idx, subtask_idx, step
                )
                Image.fromarray(combined).save(str(png_path))

            if record:
                prev_recons = recons

        obs, _, _, current_info = _step_env(env, action_buffer[step % act_step])

        if record:
            frames.append(_get_raw_image(obs, "rgb_static"))

        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            return True, frames

    return False, frames


def _evaluate_sequence(
    env,
    task_oracle,
    initial_state: Dict,
    eval_sequence: List[str],
    val_annotations: Dict,
    server_url: str,
    act_step: int,
    ep_len: int,
    record: bool = False,
    video_dir: Optional[Path] = None,
    seq_idx: int = 0,
) -> int:
    """5-subtask 시퀀스 평가. 연속 성공 수 반환."""
    robot_obs, scene_obs = _get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    all_frames = []
    for subtask_idx, subtask in enumerate(eval_sequence):
        instruction = val_annotations[subtask][0]
        success, frames = _rollout(
            env,
            task_oracle,
            subtask,
            instruction,
            server_url,
            act_step,
            ep_len,
            record=record,
            video_dir=video_dir,
            seq_idx=seq_idx,
            subtask_idx=subtask_idx,
        )
        if record and frames:
            frames = _draw_instruction(frames, instruction)
            frames = _draw_border(frames, success)
            all_frames.extend(frames)
        if success:
            success_counter += 1
        else:
            break

    # MP4 저장: 기록 대상이고 완전 성공(5/5)이 아닌 경우
    if record and video_dir is not None and all_frames and success_counter < 4:
        mp4_path = video_dir / "seq{:04d}_result{}.mp4".format(seq_idx, success_counter)
        _save_video(all_frames, mp4_path)

    return success_counter


def evaluate(
    dataset_path: Path,
    upvla_conf: Path,
    server_url: str,
    num_sequences: int,
    ep_len: int,
    output_path: Optional[Path],
    num_videos: int = 0,
    video_dir: Optional[Path] = None,
):
    # 서버 대기
    server_info = _wait_for_server(server_url)
    act_step = server_info.get("act_step", 10)

    # task oracle (calvin_env.envs.tasks.Tasks) — Calvin-native 클래스
    tasks_cfg = OmegaConf.load(
        upvla_conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(tasks_cfg)

    # 언어 어노테이션 (task_name → [annotation_strings])
    ann_cfg = OmegaConf.load(upvla_conf / "annotations/new_playtable_validation.yaml")
    val_annotations = OmegaConf.to_container(ann_cfg, resolve=True)

    # Calvin env
    # get_env()는 .hydra/config.yaml이 있는 split 디렉토리를 기대함
    # task_ABC_D/ 또는 calvin_debug_dataset/ 모두 하위에 validation/ 이 있는 구조
    env_path = dataset_path / "validation"
    if not env_path.exists():
        raise FileNotFoundError(
            f"{env_path} 가 없습니다. --dataset-path 는 split 루트 "
            f"(e.g. task_ABC_D/ 또는 calvin_debug_dataset/) 여야 합니다."
        )
    _setup_egl()
    env = get_env(str(env_path), show_gui=False)

    # 평가 시퀀스 생성
    sequences = _get_sequences(num_sequences)

    results = []
    pbar = tqdm(sequences, desc="Evaluating")
    for seq_idx, (initial_state, eval_sequence) in enumerate(pbar):
        record = num_videos > 0 and seq_idx < num_videos
        result = _evaluate_sequence(
            env,
            task_oracle,
            initial_state,
            list(eval_sequence),
            val_annotations,
            server_url,
            act_step,
            ep_len,
            record=record,
            video_dir=video_dir,
            seq_idx=seq_idx,
        )
        results.append(result)

        success_rates = _count_success(results)
        avg = sum(success_rates) / len(success_rates) * 5
        desc = " | ".join(f"{i+1}/5:{v*100:.1f}%" for i, v in enumerate(success_rates))
        pbar.set_description(f"{desc} | Avg:{avg:.2f}")

    # 결과 출력
    success_rates = _count_success(results)
    avg_len = np.mean(results)
    print("\n=== UP-VLA Calvin Evaluation Results ===")
    print(f"Sequences evaluated: {len(results)}")
    print(f"Average sequence length: {avg_len:.3f}")
    for i, sr in enumerate(success_rates):
        print(f"  {i+1}/5 success rate: {sr * 100:.1f}%")

    # 저장
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "avg_seq_len": float(avg_len),
            "chain_sr": {i + 1: sr for i, sr in enumerate(success_rates)},
            "num_sequences": len(results),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return results


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="UP-VLA Calvin 평가")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Calvin 데이터셋 경로 (e.g. /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset)",
    )
    parser.add_argument(
        "--upvla-conf",
        type=str,
        default="/temporal_vla/src/policies/UP-VLA/policy_conf",
        help="UP-VLA policy_conf 디렉토리 경로",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8300",
        help="serve_upvla.py 서버 URL",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="평가 시퀀스 수",
    )
    parser.add_argument(
        "--ep-len",
        type=int,
        default=360,
        help="subtask당 최대 스텝 수",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (.json)",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=0,
        help="비디오로 기록할 시퀀스 수 (0=비활성화). 성공 4회 미만인 경우만 MP4 저장.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="비디오/PNG 저장 디렉토리 (--num-videos > 0일 때 필수)",
    )
    args = parser.parse_args()

    if args.num_videos > 0 and args.video_dir is None:
        parser.error("--num-videos > 0 이면 --video-dir 를 지정해야 합니다.")

    evaluate(
        dataset_path=Path(args.dataset_path),
        upvla_conf=Path(args.upvla_conf),
        server_url=args.server_url,
        num_sequences=args.num_sequences,
        ep_len=args.ep_len,
        output_path=Path(args.output) if args.output else None,
        num_videos=args.num_videos,
        video_dir=Path(args.video_dir) if args.video_dir else None,
    )


if __name__ == "__main__":
    main()
