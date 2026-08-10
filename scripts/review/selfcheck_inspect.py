#!/usr/bin/env python3
"""inspect_rollout.py 자체 검증 — 음성 대조 (fault injection).

"검증 도구의 통과를 어떻게 믿나"에 대한 답: 일부러 망가뜨린 pkl 을 도구가
잡아내는지 본다. 오염 6종을 심고 전부 검출되면 도구를 신뢰할 근거가 생긴다.

    P=/home/dongkyu/miniconda3/envs/lerobot_safe/bin/python
    $P scripts/review/selfcheck_inspect.py [원본.pkl]

원본 미지정 시 phase_event_6p 의 아무 pkl 을 쓴다. 원본은 수정하지 않는다
(tmp 사본에만 오염 주입).
"""
from __future__ import annotations

import copy
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TREE = REPO / "outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts"

# (이름, 기대 검출 문구, 오염 함수)
def _flip_success(d):
    d["episode_success"] = 1 - d["episode_success"]

def _truncate_hidden(d):
    d["hidden_states"] = d["hidden_states"][:-5]

def _inject_nan(d):
    h = d["hidden_states"][3].clone().float()
    h[0, 0, :10] = float("nan")
    d["hidden_states"][3] = h.half()

def _wrong_seed(d):
    d["scenario_seed"] = 999999

def _wrong_instruction(d):
    d["canonical_instruction"] = "Pick the WRONG object."

def _zero_record(d):
    d["hidden_states"][5] = torch.zeros_like(d["hidden_states"][5])

FAULTS = [
    ("성공 라벨 뒤집기", "성공 라벨 불일치", _flip_success),
    ("hidden_states 절단", "축 길이 불일치", _truncate_hidden),
    ("NaN 주입", "NaN", _inject_nan),
    ("seed 불일치", "scene seed 불일치", _wrong_seed),
    ("instruction 불일치", "canonical_instruction != task_description", _wrong_instruction),
    ("record 통째 0", "분산 0", _zero_record),
]


def main() -> int:
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        src = next(DEFAULT_TREE.rglob("*succ1.pkl"), None)
    if src is None or not src.exists():
        print("원본 pkl 을 찾지 못함 — 인자로 지정할 것", file=sys.stderr)
        return 1
    print(f"원본: {src}")
    d0 = pickle.load(src.open("rb"))
    ep, succ = d0["episode_idx"], d0["episode_success"]

    with tempfile.TemporaryDirectory(prefix="selfcheck_") as tmp:
        cell = Path(tmp) / src.parent.parent.name / src.parent.name
        cell.mkdir(parents=True)
        names = []
        for i, (label, _, fn) in enumerate(FAULTS):
            d = copy.deepcopy(d0)
            fn(d)
            # 파일명은 원본 라벨 유지 — 라벨 오염이 "파일명 vs pkl"로 드러나게.
            name = f"task0--ep{ep}--succ{succ}--fault{i}.pkl"
            pickle.dump(d, (cell / name).open("wb"))
            names.append(name)
        clean = f"task0--ep{ep}--succ{succ}--clean.pkl"
        pickle.dump(d0, (cell / clean).open("wb"))

        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/review/inspect_rollout.py"), str(tmp)],
            capture_output=True, text=True)
        out = r.stdout

        print(f"\n{'오염':24s} {'기대 문구':32s} 검출")
        ok = 0
        for i, (label, expect, _) in enumerate(FAULTS):
            block = out.split(f"fault{i}.pkl")[1].split("──")[0] if f"fault{i}.pkl" in out else ""
            hit = expect in block
            ok += hit
            print(f"{label:24s} {expect:32s} {'✓' if hit else '✗ 못 잡음'}")
        clean_block = out.split("clean.pkl")[1].split("──")[0] if "clean.pkl" in out else ""
        clean_ok = "✓ 문제 없음" in clean_block
        print(f"{'(무오염 대조)':24s} {'문제 없음':32s} {'✓' if clean_ok else '✗ 오탐'}")

        print(f"\n판정: {ok}/{len(FAULTS)} 검출, 무오염 {'통과' if clean_ok else '오탐'}"
              f" → 도구 {'신뢰 가능' if ok == len(FAULTS) and clean_ok else '⚠ 신뢰 불가 — 수정 필요'}")
        return 0 if ok == len(FAULTS) and clean_ok else 2


if __name__ == "__main__":
    sys.exit(main())
