#!/usr/bin/env python3
"""exp4-1: fit-풀(수집 pkl) 실패 episode 의 사이드카 상당 json + mp4 를 스테이징으로 추출.

fit30 manifest(label=0)의 pkl 은 승준 노드에만 있고 개당 ~1GB (hidden_states 포함).
주석 팩·t0 manifest 에 필요한 것은 메타(env_step_phases·이벤트·seed 류)와 mp4 뿐이므로,
**승준 노드에서 실행**해 pkl 에서 스칼라/소형 필드만 json 으로 덜어내고 mp4 를 복사한다.
회수는 rsync 로 json+mp4 만 (pkl 이동 없음).

사용 (승준 노드, ~/anaconda3/bin/python — torch 필요):
  python extract_fitpool_sidecars.py \
    --manifest <task_PPCC_fit.tsv> --manifest <task_OpenDrawer_fit.tsv> \
    --out-root <staging>/fitpool_sidecars [--label 0]
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

# 대형 배열 필드 — json 에서 제외 (나머지는 소형 검사 통과 시 유지)
HEAVY_KEYS = {"hidden_states", "actions", "action_vectors"}
MAX_LIST_LEN = 2000


def _jsonable(v, depth: int = 0):
    """스칼라/소형 list/dict 만 통과. 텐서·ndarray·대형 배열은 None (drop 마커)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if depth >= 4:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_LIST_LEN:
            return None
        out = [_jsonable(x, depth + 1) for x in v]
        return None if any(x is None and y is not None for x, y in zip(out, v)) else out
    if isinstance(v, dict):
        out = {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
        return {k: x for k, x in out.items() if x is not None or v.get(k) is None}
    return None  # tensor/ndarray 등


def extract_one(pkl_path: Path, out_dir: Path) -> bool:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    meta = {}
    for k, v in d.items():
        if k in HEAVY_KEYS:
            continue
        jv = _jsonable(v)
        if jv is not None or v is None:
            meta[k] = jv
    meta["pool"] = "fit"
    meta["source_pkl"] = str(pkl_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pkl_path.stem
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False))
    mp4 = pkl_path.with_suffix(".mp4")
    if mp4.exists():
        shutil.copy2(mp4, out_dir / f"{stem}.mp4")
    else:
        print(f"[warn] mp4 없음: {mp4}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--label", type=int, default=0, help="추출 대상 label (0=실패)")
    args = ap.parse_args()

    rows = []
    for m in args.manifest:
        for line in m.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pkl, label = Path(parts[0]).expanduser(), int(parts[1])
            if label == args.label:
                rows.append(pkl)
    print(f"[extract] 대상 {len(rows)}개 (label={args.label})")

    ok = miss = 0
    for pkl in rows:
        if not pkl.exists():
            print(f"[miss] pkl 없음: {pkl}")
            miss += 1
            continue
        cell = pkl.parent.name
        ok += extract_one(pkl, args.out_root / cell)
    print(f"[done] ok={ok} miss={miss} / {len(rows)} → {args.out_root}")


if __name__ == "__main__":
    main()
