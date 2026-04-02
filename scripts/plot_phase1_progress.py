#!/usr/bin/env python3
"""
eval_phase1_predictor.py --save_json 로 생성된 *.progress.json 파일을 읽어
에피소드별 progress curve를 시각화합니다.

사용 예:
  # 특정 파일 하나
  python scripts/plot_phase1_progress.py --input_json eval_results/phase1_step50k/episodes/val/ep_000042.progress.json

  # 디렉토리 전체 (split별 서브디렉토리 포함)
  python scripts/plot_phase1_progress.py --input_json eval_results/phase1_step50k/episodes/val

  # 여러 에피소드를 grid PNG 하나로
  python scripts/plot_phase1_progress.py --input_json eval_results/phase1_step50k/episodes/val --grid

  # split 필터링
  python scripts/plot_phase1_progress.py --input_json eval_results/phase1_step50k/episodes --split val
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_json", required=True,
        help="*.progress.json 파일 경로 또는 해당 파일들이 있는 디렉토리",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="결과 PNG를 저장할 디렉토리 (기본: input_json과 같은 위치)",
    )
    parser.add_argument(
        "--split", default=None,
        help="특정 split만 처리 (train / val / unseen). 디렉토리 입력 시 적용.",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="모든 에피소드를 하나의 grid PNG로 저장 (기본: 에피소드별 개별 PNG)",
    )
    parser.add_argument(
        "--n_cols", type=int, default=4,
        help="grid 모드에서 열 수 (기본: 4)",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="처리할 최대 에피소드 수 (기본: 전체)",
    )
    parser.add_argument(
        "--sort_by", choices=["pearson_r", "mse", "mae", "mono_rate", "ep_idx"],
        default="ep_idx",
        help="grid 모드에서 에피소드 정렬 기준 (기본: ep_idx)",
    )
    parser.add_argument(
        "--sort_desc", action="store_true",
        help="내림차순 정렬 (기본: 오름차순)",
    )
    return parser.parse_args()


def should_include(json_path: Path, split_filter: str | None) -> bool:
    if split_filter is None:
        return True
    # 파일 내용의 split 필드로 확인
    try:
        payload = json.loads(json_path.read_text())
        return payload.get("split", "") == split_filter
    except Exception:
        return True


def load_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    records = payload["records"]
    preds   = np.array([r["progress_prediction"] for r in records])
    targets = np.array([r["gt_progress"]          for r in records])
    timesteps = np.array([r["timestep"]           for r in records])
    return {
        "episode_idx": payload["episode_idx"],
        "split":       payload.get("split", ""),
        "ep_len":      payload["ep_len"],
        "metrics":     payload["metrics"],
        "preds":       preds,
        "targets":     targets,
        "timesteps":   timesteps,
    }


def _plot_axes(ax: plt.Axes, data: dict, show_legend: bool = True):
    """axes에 단일 에피소드 progress curve를 그립니다."""
    m = data["metrics"]
    ax.plot(data["timesteps"], data["targets"], "k--", linewidth=1.4, label="GT (t/T)")
    ax.plot(data["timesteps"], data["preds"],   color="#1F77B4", linewidth=1.4, label="Predicted")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Progress", fontsize=8)
    ax.set_title(
        f"[{data['split']}] ep {data['episode_idx']}\n"
        f"r={m['pearson_r']:.3f}  MSE={m['mse']:.4f}  mono={m['mono_rate']:.3f}",
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7)


def plot_individual(data: dict, output_dir: Path):
    """에피소드 하나를 단독 PNG로 저장."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_axes(ax, data, show_legend=True)
    fig.tight_layout()
    save_path = output_dir / f"ep_{data['episode_idx']:06d}.png"
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_grid(all_data: list[dict], output_path: Path, n_cols: int = 4):
    """여러 에피소드를 grid PNG 하나로 저장."""
    n = len(all_data)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)

    for i, data in enumerate(all_data):
        row, col = divmod(i, n_cols)
        _plot_axes(axes[row][col], data, show_legend=(i == 0))

    # 남은 axes 숨기기
    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    splits = sorted({d["split"] for d in all_data})
    fig.suptitle(f"Progress curves — {', '.join(splits)}  ({n} episodes)", y=1.01, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {output_path}")


def main():
    args = parse_args()
    input_path = Path(args.input_json)

    # JSON 파일 수집
    if input_path.is_dir():
        json_paths = sorted(input_path.rglob("*.progress.json"))
        json_paths = [p for p in json_paths if should_include(p, args.split)]
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")
        json_paths = [input_path]

    if not json_paths:
        raise FileNotFoundError("처리할 *.progress.json 파일이 없습니다.")

    if args.max_episodes is not None:
        json_paths = json_paths[:args.max_episodes]

    print(f"처리할 에피소드: {len(json_paths)}개")

    # 데이터 로드
    all_data = [load_json(p) for p in json_paths]

    # 정렬
    if args.sort_by == "ep_idx":
        all_data.sort(key=lambda d: d["episode_idx"], reverse=args.sort_desc)
    else:
        all_data.sort(key=lambda d: d["metrics"][args.sort_by], reverse=args.sort_desc)

    # 출력 디렉토리
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif input_path.is_dir():
        out_dir = input_path
    else:
        out_dir = input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid:
        # 모든 에피소드를 grid PNG 하나로
        splits = sorted({d["split"] for d in all_data})
        tag = "_".join(splits) if splits else "progress"
        grid_path = out_dir / f"grid_{tag}_{len(all_data)}ep.png"
        plot_grid(all_data, grid_path, n_cols=args.n_cols)
    else:
        # 에피소드별 개별 PNG
        for data in all_data:
            save_path = plot_individual(data, out_dir)
            print(f"Saved: {save_path}")

    print("완료.")


if __name__ == "__main__":
    main()
