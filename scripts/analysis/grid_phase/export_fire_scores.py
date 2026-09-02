#!/usr/bin/env python
"""선택 episode 의 detector 발화 궤적 export (docs/steering/43 후속 B-1).

`failure_detector_sim.py` 가 학습·저장한 pertask detector 체크포인트를 **재사용**해
(재학습 없음) full test 시퀀스 위의 causal score_t, CP 밴드 δ_t, 첫 발화 t_fire 를
뽑아 JSON 으로 내보낸다. 오버레이 영상 렌더러(`render_fire_overlay.py`)의 입력.

## 규약
- detector·표준화·CP 밴드는 전부 `.pt` 에서 읽는다 — 여기서 학습/보정하지 않는다.
- test split 판만 대상 (sim_detail.json 의 `scene_split`). 절제는 학습·보정에만
  걸렸으므로 **test 시퀀스는 항상 full** — 여기서도 자르지 않는다.
- 같은 episode 를 여러 절제 mode 로 스코어링해 비교할 수 있게 한다.
- 산출 JSON 에 절대경로를 쓰지 않는다 (docs/04 §8). 영상은 grid 루트 기준 상대경로.

## episode 선택
`--select "<slug>:<fail|succ>:<n|all>,..."` (기본값 = 43 후속 지정 목록).
같은 (task, ep_id) 가 여러 번 선택되면 한 번만 export 한다. 정렬은 ep_id 오름차순
(결정적).

## 사용
    ~/anaconda3/bin/python scripts/analysis/grid_phase/export_fire_scores.py \
        --shard-dir ~/datasets/.../analysis/grid_phase/segA \
        --sim-root outputs/analysis/grid_phase/detector_trunc \
        --grid-root ~/datasets/.../groot/n15/grid \
        --out outputs/analysis/grid_phase/fire_videos/fire_scores.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_detector_sim import (  # noqa: E402
    ShardSpec, apply_std, build_detector, fire_step, load_shard_episodes, score_seq,
)

MODES = ("none", "rollout", "phase-gt")
DEFAULT_SELECT = ("OpenDrawer_left:fail:all,OpenDrawer_left:succ:2,"
                  "PPCC_bread:fail:2,PPCC_bread:succ:1,"
                  "PPCC_marshmallow:fail:1,OvenRack_out:fail:1")
# 수집 인자 (scripts/safe/groot_n15/robocasa/collect/collect_grid.sh:255-256).
# 프레임 정렬 근거는 render_fire_overlay.py 상단 주석 참조.
COLLECT_N_ACTION_STEPS = 5
COLLECT_STEPS_PER_RENDER = 2
COLLECT_VIDEO_FPS = 20


def parse_select(spec: str) -> list[tuple[str, str, str]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3 or parts[1] not in ("fail", "succ"):
            raise SystemExit(f"--select 항목 형식 오류: {item} (<slug>:<fail|succ>:<n|all>)")
        out.append((parts[0], parts[1], parts[2]))
    return out


def load_ckpt(sim_root: Path, mode: str, arm: str, model: str, slug: str) -> dict:
    p = sim_root / mode / f"detector_{arm}_{model}_{slug}.pt"
    if not p.exists():
        raise SystemExit(f"detector 체크포인트 없음: {p}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck["truncate"]["mode"] != mode:
        raise SystemExit(f"{p}: truncate.mode={ck['truncate']['mode']} != {mode}")
    net = build_detector(model, ck["input_dim"], ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    ck["net"] = net
    return ck


def band_delta(ck: dict, slug: str, alpha: float) -> np.ndarray:
    key = f"{alpha:.2f}"
    bands = ck["cp_bands"].get(slug)
    if not bands or key not in bands:
        raise SystemExit(f"{slug}: α={key} CP 밴드가 체크포인트에 없다 "
                         f"(있는 것: {sorted(bands or [])})")
    return np.asarray(bands[key]["delta"], dtype=np.float64)


def delta_series(delta: np.ndarray, T: int) -> np.ndarray:
    """t>밴드길이 는 마지막 밴드 유지 (failure_detector_sim.fire_step 과 동일 규약)."""
    idx = np.minimum(np.arange(T), len(delta) - 1)
    return delta[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--grid-root", default=None,
                    help="영상 루트 (있으면 video 경로 존재 검증)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--arm", default="pertask")
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--select", default=DEFAULT_SELECT)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--denoise", type=int, default=-1)
    ap.add_argument("--seg", default="all")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    shard_dir = Path(args.shard_dir)
    sim_root = Path(args.sim_root)
    sims = {m: json.loads((sim_root / m / "sim_detail.json").read_text(encoding="utf-8"))
            for m in modes}
    base = sims[modes[0]]
    for m in modes[1:]:
        if sims[m]["scene_split"] != base["scene_split"]:
            raise SystemExit(f"scene_split 이 mode 간 다르다 (none vs {m})")

    sel = parse_select(args.select)
    want: dict[str, list[tuple[str, str]]] = {}
    for slug, kind, n in sel:
        want.setdefault(slug, []).append((kind, n))

    episodes_out: list[dict] = []
    for slug in sorted(want):
        p = shard_dir / f"{slug}.npz"
        eps, spec = load_shard_episodes(p, args.layer, args.denoise, args.seg)
        meta = ShardSpec(p, args.layer, args.denoise, args.seg).meta
        plan_ids = meta.get("plan_id") or []
        if len(plan_ids) != 1:
            raise SystemExit(f"{slug}: plan_id 가 1개가 아니다 {plan_ids} — 영상 경로 모호")
        plan_id, machine = str(plan_ids[0]), str(meta["machine"])
        grid_instr = str(meta["instruction"])           # 예: "OpenDrawer/left"

        test_scenes = set(int(s) for s in base["scene_split"][slug]["test"])
        test_eps = sorted((e for e in eps if e.scene in test_scenes),
                          key=lambda e: e.ep_id)
        chosen: list = []
        for kind, n in want[slug]:
            pool = [e for e in test_eps if (e.succ == 0 if kind == "fail" else e.succ == 1)]
            take = pool if n == "all" else pool[: int(n)]
            if not take:
                raise SystemExit(f"{slug}: test split 에 {kind} 판이 없다")
            for e in take:
                if all(e.ep_id != c.ep_id for c in chosen):
                    chosen.append(e)
        chosen.sort(key=lambda e: e.ep_id)

        nets = {m: load_ckpt(sim_root, m, args.arm, args.model, slug) for m in modes}
        for e in chosen:
            rec = {
                "task": slug, "instruction": grid_instr,
                "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
                "succ": e.succ, "T": e.T,
                "phase_code": [int(c) for c in e.phase],
                "phase_names": {str(k): v for k, v in spec.phase_names.items()},
                "video": f"{plan_id}/{machine}/{grid_instr}/s{e.scene}/n{e.noise}"
                         f"/base/video.mp4",
                "modes": {},
            }
            for m in modes:
                ck = nets[m]
                X = apply_std(e, np.asarray(ck["std_mean"]), np.asarray(ck["std_std"]))
                sc = score_seq(ck["net"], X)
                dl = band_delta(ck, slug, args.alpha)
                ds = delta_series(dl, e.T)
                ft = fire_step(sc, dl)
                rec["modes"][m] = {
                    "scores": [round(float(v), 5) for v in sc],
                    "band": [round(float(v), 5) for v in ds],
                    "t_fire": None if ft is None else int(ft),
                    "band_L": int(len(dl)),
                    "W": ck["truncate"]["rollout_W"].get(slug),
                    "phase_caps": ck["truncate"]["phase_caps"].get(slug),
                }
            if args.grid_root:
                vp = Path(args.grid_root).expanduser() / rec["video"]
                rec["video_exists"] = vp.exists()
                if not vp.exists():
                    print(f"  [warn] 영상 없음: {rec['video']}", flush=True)
            episodes_out.append(rec)
            fires = {m: rec["modes"][m]["t_fire"] for m in modes}
            print(f"[{slug}] ep{e.ep_id} s{e.scene}n{e.noise} "
                  f"{'succ' if e.succ else 'FAIL'} T={e.T} t_fire={fires}", flush=True)

    payload = {
        "config": {
            "arm": args.arm, "model": args.model, "alpha": args.alpha,
            "modes": modes, "layer": args.layer, "denoise": args.denoise,
            "seg": args.seg, "select": args.select,
            "shards": [f"{s}.npz" for s in sorted(want)],
            "collect": {"n_action_steps": COLLECT_N_ACTION_STEPS,
                        "steps_per_render": COLLECT_STEPS_PER_RENDER,
                        "video_fps": COLLECT_VIDEO_FPS,
                        "source": "collect_grid.sh:255-256"},
            "note": "test 시퀀스는 full (절제는 학습·보정에만 적용됨)",
        },
        "scene_split": {s: base["scene_split"][s] for s in sorted(want)},
        "episodes": episodes_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\n[export] {len(episodes_out)} episode × {len(modes)} mode → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
