#!/usr/bin/env python
"""Event-SAE (arXiv 2025, 저자 코드) 재현용 입력 변환기.

우리 판독 실험과 **완전히 같은 930 에피소드**(episode_keys.json)를 저자 파이프라인의
입력 포맷(trajectory_records.jsonl + videos/episode_%05d_*.mp4)으로 변환한다.
이후 단계는 저자 스크립트를 무수정 호출한다:
    extract_keyframes.py (AWE dp, pos_only, err 0.05)
  → extract_keyframe_media.py (frame offsets -4,-2,0,2,4)
  → build_event_features.py (SigLIP base, 5프레임 평균)
  → cluster_events.py (agglomerative cosine 0.18, coverage 0.5)

정렬 규약 (저자 코드의 가정 = step 인덱스 == 비디오 프레임 인덱스):
    우리 영상은 20fps, 1프레임 = 환경 2스텝. 상태는 환경 스텝 해상도(NPZ).
    → records 를 **프레임 해상도**로 쓰고, 프레임 f 의 상태 = env step min(2f, T-1).
    이렇게 하면 waypoint_index 가 프레임·records 양쪽을 같은 눈금으로 가리킨다.

적응 사항 (논문에 명시할 것):
    - eef_pos = observation.state.eef_pos_rel (robocasa serialize; task 내 일관 좌표)
    - gripper_action 이 저장에 없어 gripper_qpos[0] (손가락 관절값) 을 대신 사용.
      AWE waypoint 는 pos_only 라 이 값은 waypoint 선택에 안 쓰이고, descriptor 의
      state 성분(z-score)에만 들어간다.

출력 레이아웃: <out-root>/<slug>/{trajectory_records.jsonl, videos/episode_%05d_ep.mp4,
             episodes_map.json}  (episode_num ↔ (scene, noise))
episode_num = scene*100 + noise (slug 안에서 결정적) — 부분 재실행에도 번호가 안 변한다.
상태 NPZ 가 아직 없는 에피소드는 건너뛰고 빠진 목록을 보고한다.
"""
from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np

SLUGS_ALL = (
    "CoffeeSetupMug", "DishwasherRack_out", "OpenDrawer_left", "OpenDrawer_right",
    "OvenRack_out", "PPCC_apple", "PPCC_bread", "PPCC_candle", "PPCC_jug",
    "PPCC_marshmallow",
)
SLUG2INSTR = {
    "OpenDrawer_left": "Open the left drawer.",
    "OpenDrawer_right": "Open the right drawer.",
    "DishwasherRack_out": "Fully slide the top dishwasher rack out.",
    "OvenRack_out": "Fully slide the oven rack out.",
    "CoffeeSetupMug": "Pick the mug from the counter and place it under the coffee machine dispenser.",
}
for _o in ("apple", "bread", "candle", "jug", "marshmallow"):
    SLUG2INSTR[f"PPCC_{_o}"] = f"Pick the {_o} from the counter and place it in the cabinet."


def probe_frames(video: Path) -> int:
    """저자 extract_media 와 같은 imageio 리더로 프레임 수를 센다 (ffprobe 불필요)."""
    import imageio
    reader = imageio.get_reader(str(video))
    try:
        return int(reader.count_frames())
    finally:
        reader.close()


def index_videos(vid_root: Path) -> dict[tuple[str, int, int], Path]:
    """vids930 트리(…/<Task>/<obj>/sN/nM/base/video.mp4)를 (slug, scene, noise)로 색인."""
    idx: dict[tuple[str, int, int], Path] = {}
    two_level = {"PPCC", "OpenDrawer", "OvenRack", "DishwasherRack"}
    for mp4 in vid_root.rglob("video.mp4"):
        parts = mp4.parts   # …/<Task>[/<variant>]/s<N>/n<M>/base/video.mp4
        try:
            no = int(parts[-3][1:]); sc = int(parts[-4][1:])
        except (ValueError, IndexError):
            continue
        slug = f"{parts[-6]}_{parts[-5]}" if parts[-6] in two_level else parts[-5]
        if slug in SLUG2INSTR:
            idx[(slug, sc, no)] = mp4
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", type=Path, required=True, help="episode_keys.json (930판)")
    ap.add_argument("--states-dir", type=Path, required=True, help="kai_states NPZ 디렉토리")
    ap.add_argument("--vids-root", type=Path, required=True, help="vids930 트리")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--slugs", nargs="*", default=list(SLUGS_ALL))
    args = ap.parse_args()

    keys = sorted(tuple(k) for k in json.loads(args.keys.read_text()))
    vids = index_videos(args.vids_root)
    task_ids = {s: i for i, s in enumerate(SLUGS_ALL)}

    missing = {"state": [], "video": []}
    per_slug_count: dict[str, int] = {}
    writers: dict[str, list[str]] = {}
    slug_maps: dict[str, dict] = {}
    total = 0

    for slug, sc, no in keys:
        if slug not in args.slugs:
            continue
        npz_path = args.states_dir / f"{slug}__s{sc}__n{no}.npz"
        vid = vids.get((slug, sc, no))
        if not npz_path.is_file():
            missing["state"].append(f"{slug}/s{sc}/n{no}")
            continue
        if vid is None:
            missing["video"].append(f"{slug}/s{sc}/n{no}")
            continue
        d = np.load(npz_path)
        eef, grip = d["eef_pos"], d["gripper_qpos"]
        succ = bool(int(d["success"]) == 1)
        n_frames = probe_frames(vid)
        T = len(eef)

        ep_num = sc * 100 + no          # slug 안에서 결정적 (재실행 안정)
        run_dir = args.out_root / slug
        (run_dir / "videos").mkdir(parents=True, exist_ok=True)
        ep_idx = per_slug_count.get(slug, 0)
        link = run_dir / "videos" / f"episode_{ep_num:05d}_ep.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(vid.resolve())

        rows = writers.setdefault(slug, [])
        for f in range(n_frames):
            t = min(2 * f, T - 1)
            rows.append(json.dumps({
                "episode_num": ep_num,
                "task_id": task_ids[slug],
                "task_episode_idx": ep_idx,
                "task_description": SLUG2INSTR[slug],
                "step_in_episode": f,
                "eef_pos": [float(x) for x in eef[t]],
                "gripper_action": float(grip[t][0]),
                "gripper_qpos": [float(x) for x in grip[t]],
                "done": succ if f == n_frames - 1 else False,
            }))
        slug_maps.setdefault(slug, {})[str(ep_num)] = {
            "slug": slug, "scene": sc, "noise": no, "n_frames": n_frames,
            "n_env_steps": T, "success": succ}
        per_slug_count[slug] = ep_idx + 1
        total += 1

    for slug, rows in writers.items():
        run_dir = args.out_root / slug
        (run_dir / "trajectory_records.jsonl").write_text("\n".join(rows) + "\n")
        (run_dir / "episodes_map.json").write_text(
            json.dumps(slug_maps[slug], ensure_ascii=False, indent=1))
        print(f"[{slug:<22}] {per_slug_count[slug]:3d} ep  → {run_dir}")

    print(f"\n총 {total} ep 변환. 누락 — state {len(missing['state'])}개, "
          f"video {len(missing['video'])}개")
    if missing["state"]:
        print("  state 누락 예:", missing["state"][:5])
    (args.out_root / "missing.json").write_text(json.dumps(missing, indent=1))


if __name__ == "__main__":
    main()
