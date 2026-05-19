#!/usr/bin/env python3
"""Analyze per-rollout SAFE score timing against functional CP bands."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", default="/home/dongkyu/pdk_ws/SAFE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--cp-data-path", "--cal-data-path", dest="cp_data_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, action="append", default=None)
    parser.add_argument("--functional-cp-seed", type=int, default=0)
    parser.add_argument("--max-plots", type=int, default=32)
    return parser.parse_args()


def _manifest_by_new_path(manifest_path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_path = Path(row["new_path"])
            new_path = str((raw_path if raw_path.is_absolute() else Path.cwd() / raw_path).resolve())
            rows[new_path] = row
    return rows


def _to_float(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _score_rollouts(cfg, model, data_path: str, device: str, batch_size: int):
    from failure_prob.data import load_rollouts
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.utils.torch import move_to_device

    cfg.dataset.data_path = data_path
    rollouts = load_rollouts(cfg)
    dataset = RolloutDataset(cfg, rollouts, device="cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_scores, all_masks = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            all_scores.append(model(batch).squeeze(-1).detach().cpu())
            all_masks.append(batch["valid_masks"].detach().cpu())

    score_tensor = torch.cat(all_scores, dim=0).numpy()
    mask_tensor = torch.cat(all_masks, dim=0).numpy()
    lengths = mask_tensor.sum(axis=1).astype(int)
    scores_by_rollout = [
        score_tensor[i, : lengths[i]].astype(np.float64)
        for i in range(len(rollouts))
    ]
    return rollouts, scores_by_rollout


def _first_cross(score: np.ndarray, band: np.ndarray, end: int | None = None) -> int | None:
    n = len(score) if end is None else min(end, len(score))
    if n <= 0:
        return None
    limit = min(n, len(band))
    mask = score[:limit] >= band[:limit]
    if not np.any(mask):
        return None
    return int(np.argmax(mask))


def _copyable_video_path(source_path: str) -> str:
    return str(Path(source_path).with_suffix(".mp4"))


def _pkl_paths(data_path: str) -> list[Path]:
    return sorted(Path(data_path).rglob("*.pkl"))


def _parse_split_filename(path: Path) -> tuple[int, int, int]:
    stem = path.stem
    pieces = stem.split("--")
    task_id = int(pieces[0].replace("task", ""))
    episode_idx = int(pieces[1].replace("ep", ""))
    success = int(pieces[2].replace("succ", ""))
    return task_id, episode_idx, success


def _save_score_plot(path: Path, score: np.ndarray, task_min_step: int, bands: dict[float, np.ndarray], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    x = np.arange(len(score))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, score, label="SAFE score", color="#1f77b4", linewidth=2)
    for alpha, band in sorted(bands.items()):
        ax.plot(x[: min(len(x), len(band))], band[: len(x)], linestyle="--", linewidth=1.5, label=f"functional CP alpha={alpha:g}")
    ax.axvline(task_min_step, color="#444", linestyle=":", linewidth=1.5, label="task_min_step")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("frame")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_contact_sheet(
    path: Path,
    rows: list[dict[str, object]],
    max_items: int = 12,
) -> None:
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    selected = rows[:max_items]
    if not selected:
        return

    tile_w, tile_h = 220, 170
    label_h = 56
    cols = 3
    rows_n = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * tile_w, rows_n * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, item in enumerate(selected):
        video_path = str(item["mp4_path"])
        frame_idx = int(item.get("functional_cp_alpha0p2_final_first_cross_t") or item["length"] - 1)
        cap = cv2.VideoCapture(video_path)
        ok = False
        frame = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame).resize((tile_w, tile_h))
        else:
            image = Image.new("RGB", (tile_w, tile_h), "#dddddd")
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + label_h)
        canvas.paste(image, (x, y))
        label = (
            f"task{item['task_id']} ep{item['new_episode_idx']} src{item['source_episode_idx']}\n"
            f"len {item['length']} min {item['task_min_step']} det {item.get('functional_cp_alpha0p2_final_first_cross_t')}\n"
            f"{Path(video_path).name}"
        )
        draw.text((x + 4, y + tile_h + 4), label, fill="black", font=font)

    canvas.save(path)


def _read_video_frame(video_path: str, frame_idx: int, size: tuple[int, int]):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    ok = False
    frame = None
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
        ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).resize(size)
    return Image.new("RGB", size, "#dddddd")


def _save_sequence_sheet(path: Path, rows: list[dict[str, object]], max_items: int = 12) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    selected = rows[:max_items]
    if not selected:
        return

    frame_w, frame_h = 160, 124
    label_h = 54
    col_names = ["start", "task_min", "a0.2 det", "end"]
    canvas = Image.new("RGB", (len(col_names) * frame_w, len(selected) * (frame_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for row_idx, item in enumerate(selected):
        length = int(item["length"])
        det = item.get("functional_cp_alpha0p2_final_first_cross_t")
        det_idx = int(det) if det != "" else length - 1
        frame_indices = [
            0,
            min(int(item["task_min_step"]), length - 1),
            min(det_idx, length - 1),
            length - 1,
        ]
        y = row_idx * (frame_h + label_h)
        for col_idx, (name, frame_idx) in enumerate(zip(col_names, frame_indices)):
            x = col_idx * frame_w
            image = _read_video_frame(str(item["mp4_path"]), frame_idx, (frame_w, frame_h))
            canvas.paste(image, (x, y))
            draw.text((x + 3, y + 3), f"{name}: {frame_idx}", fill="yellow", font=font)

        label = (
            f"task{item['task_id']} ep{item['new_episode_idx']} src{item['source_episode_idx']} "
            f"min={item['task_min_step']} det={det} "
            f"maxEarly={float(item['score_max_to_task_min']):.3f} maxEnd={float(item['score_max_to_end']):.3f}"
        )
        draw.text((4, y + frame_h + 4), label, fill="black", font=font)

    canvas.save(path)


def main() -> None:
    args = parse_args()
    safe_root = Path(args.safe_root)
    sys.path.insert(0, str(safe_root))

    from failure_prob.conf import process_cfg
    from failure_prob.model import get_model
    from failure_prob.utils.metrics import eval_functional_conformal

    alphas = args.alpha or [0.2, 0.5]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.dataset.load_to_cuda = False
    cfg.model.batch_size = args.batch_size
    cfg = process_cfg(cfg)

    cfg.dataset.data_path = args.data_path
    from failure_prob.data import load_rollouts

    probe_rollouts = load_rollouts(cfg)
    input_dim = probe_rollouts[0].hidden_states.shape[-1]
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = get_model(cfg, input_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    test_rollouts, test_scores = _score_rollouts(cfg, model, args.data_path, device, args.batch_size)
    cal_rollouts, cal_scores = _score_rollouts(cfg, model, args.cp_data_path, device, args.batch_size)
    test_pkl_paths = _pkl_paths(args.data_path)
    if len(test_pkl_paths) != len(test_rollouts):
        raise RuntimeError(f"Expected one PKL per rollout, got {len(test_pkl_paths)} paths and {len(test_rollouts)} rollouts")

    np.random.seed(args.functional_cp_seed)
    functional_df, bands_by_alpha = eval_functional_conformal(
        {"val_seen": cal_rollouts, "val_unseen": test_rollouts},
        {"val_seen": list(cal_scores), "val_unseen": list(test_scores)},
        "model",
        alphas=alphas,
        calib_split_names=["val_seen"],
        test_split_names=["val_unseen"],
    )
    functional_df.to_csv(output_dir / "functional_cp_selected.tsv", sep="\t", index=False)

    manifest_rows = _manifest_by_new_path(Path(args.manifest))
    rows: list[dict[str, object]] = []
    plot_dir = output_dir / "score_plots"
    plot_dir.mkdir(exist_ok=True)

    bands = {float(alpha): np.asarray(band).reshape(-1) for alpha, band in bands_by_alpha.items()}
    for split_pkl_path, rollout, score in zip(test_pkl_paths, test_rollouts, test_scores):
        length = len(score)
        task_min_step = int(rollout.task_min_step or length)
        split_task_id, split_episode_idx, split_success = _parse_split_filename(split_pkl_path)
        manifest_row = manifest_rows.get(str(split_pkl_path.resolve()), {})
        source_path = manifest_row.get("source_path", str(split_pkl_path))
        mp4_path = _copyable_video_path(source_path)
        row: dict[str, object] = {
            "task_id": int(split_task_id),
            "task_description": rollout.task_description,
            "new_episode_idx": int(split_episode_idx),
            "source_episode_idx": int(manifest_row.get("source_episode_idx", rollout.episode_idx)),
            "episode_success": int(split_success),
            "label": int(1 - split_success),
            "length": int(length),
            "task_min_step": int(task_min_step),
            "task_min_step_ratio": float(task_min_step / length),
            "score_start": _to_float(score[0]),
            "score_at_task_min": _to_float(score[min(max(task_min_step - 1, 0), length - 1)]),
            "score_max_to_task_min": _to_float(score[:task_min_step].max()),
            "score_max_to_end": _to_float(score.max()),
            "score_argmax": int(score.argmax()),
            "split_path": str(split_pkl_path),
            "source_path": source_path,
            "mp4_path": mp4_path,
            "mp4_exists": int(Path(mp4_path).exists()),
        }
        for alpha, band in bands.items():
            prefix = f"functional_cp_alpha{str(alpha).replace('.', 'p')}"
            early_cross = _first_cross(score, band, task_min_step)
            final_cross = _first_cross(score, band, None)
            row[f"{prefix}_early_first_cross_t"] = "" if early_cross is None else early_cross
            row[f"{prefix}_early_first_cross_ratio"] = "" if early_cross is None else float(early_cross / task_min_step)
            row[f"{prefix}_final_first_cross_t"] = "" if final_cross is None else final_cross
            row[f"{prefix}_final_first_cross_ratio"] = "" if final_cross is None else float(final_cross / length)

        rows.append(row)

    failure_rows = [row for row in rows if row["label"] == 1]
    failure_rows.sort(key=lambda row: (
        row.get("functional_cp_alpha0p2_final_first_cross_ratio") == "",
        row.get("functional_cp_alpha0p2_final_first_cross_ratio") if row.get("functional_cp_alpha0p2_final_first_cross_ratio") != "" else 2.0,
    ))

    for row in failure_rows[: args.max_plots]:
        pkl_name = f"task{row['task_id']}--ep{int(row['new_episode_idx']):03d}--succ0"
        idx = next(i for i, path in enumerate(test_pkl_paths) if str(path) == row["split_path"])
        _save_score_plot(
            plot_dir / f"{pkl_name}.png",
            test_scores[idx],
            int(row["task_min_step"]),
            bands,
            f"task{row['task_id']} ep{row['new_episode_idx']} failure",
        )

    tsv_path = output_dir / "rollout_timing.tsv"
    fieldnames = sorted({key for row in rows for key in row})
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    failure_summary_path = output_dir / "failure_timing_summary.json"
    summary = {
        "n_test": len(rows),
        "n_failure": len(failure_rows),
        "n_success": len(rows) - len(failure_rows),
        "functional_cp": functional_df.to_dict("records"),
        "failure_alpha0p2_early_detected": sum(row.get("functional_cp_alpha0p2_early_first_cross_t") != "" for row in failure_rows),
        "failure_alpha0p2_final_detected": sum(row.get("functional_cp_alpha0p2_final_first_cross_t") != "" for row in failure_rows),
        "failure_alpha0p5_early_detected": sum(row.get("functional_cp_alpha0p5_early_first_cross_t") != "" for row in failure_rows),
        "failure_alpha0p5_final_detected": sum(row.get("functional_cp_alpha0p5_final_first_cross_t") != "" for row in failure_rows),
    }
    failure_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _save_contact_sheet(output_dir / "failure_detection_contact_sheet.png", failure_rows, max_items=12)
    earliest = sorted(
        failure_rows,
        key=lambda row: row.get("functional_cp_alpha0p2_final_first_cross_ratio")
        if row.get("functional_cp_alpha0p2_final_first_cross_ratio") != ""
        else 2.0,
    )[:6]
    latest = sorted(
        failure_rows,
        key=lambda row: row.get("functional_cp_alpha0p2_final_first_cross_ratio")
        if row.get("functional_cp_alpha0p2_final_first_cross_ratio") != ""
        else -1.0,
        reverse=True,
    )[:6]
    _save_sequence_sheet(output_dir / "failure_sequence_contact_sheet.png", earliest + latest, max_items=12)

    print(f"Wrote {tsv_path}")
    print(f"Wrote {failure_summary_path}")
    print(f"Wrote {plot_dir}")
    print(f"Wrote {output_dir / 'failure_detection_contact_sheet.png'}")
    print(f"Wrote {output_dir / 'failure_sequence_contact_sheet.png'}")


if __name__ == "__main__":
    main()
