#!/usr/bin/env python
"""shard NPZ 의 `phase_code` 를 AE cluster 라벨로 갈아끼운 **사본** 생성기.

phase 정의를 GT 이벤트 라벨에서 task 별 K=8 activation cluster 로 바꿔 끼우는 어댑터다.
`ae_cluster.py --dump-labels` 가 만든 `labels_<slug>_k<K>.npz` 의 `cluster` 를 shard 의
`phase_code` 자리에 넣고, `meta_json.phase_codebook` 을 `{"c0":0, ..., "c7":7}` 로 바꾼다.
나머지 배열(X·ep_id·scene·noise·rec_idx·succ·ep_len ...)은 **그대로** 복사한다 —
downstream (failure_detector_sim / phase_sep_matrix / intrinsic_phase) 은 shard 인터페이스만
알면 되므로, 그 스크립트들을 건드리지 않고 phase 정의만 교체할 수 있다.

원본은 절대 수정하지 않는다 (읽기 전용). 출력은 tmp → rename (부분 파일 방지).

정렬 대조 (fail-loud)
    labels npz 의 `ep_id`/`rec_idx` 가 shard 의 그것과 **원소 단위로** 같아야 한다.
    ae_cluster 는 shard 행 순서를 보존하므로 같아야 정상이고, 다르면 다른 shard 의
    라벨을 덮어쓰는 사고이므로 즉시 중단한다. 행 수도 대조한다.

메모리
    X 는 shard 당 수 GB (fp16 [n_rec, 7, 4, 4, 1536]) 라 통째로 올리지 않는다. npz 는
    zip 이므로 교체하지 않는 멤버는 zip 스트림을 **청크 복사** 한다 (압축 방식 보존).

사용 예
    ~/anaconda3/bin/python scripts/analysis/grid_phase/rewrite_shard_clusters.py \
        --shard-dir  ~/workspace/.../analysis/grid_phase/segA \
        --labels-dir ~/workspace/.../analysis/grid_phase/ae_raw \
        --out-dir    ~/workspace/.../analysis/grid_phase/segA_ck8

실행 환경: 승준 노드 `~/anaconda3/bin/python` (numpy 만 쓴다. torch·scipy 불필요).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

LABELS_RE = re.compile(r"^labels_(?P<slug>.+)_k(?P<k>\d+)\.npz$")
CHUNK = 8 << 20


def discover_labels(labels_dir: Path, k: int | None,
                    only: set[str] | None) -> dict[str, Path]:
    """labels_<slug>_k<K>.npz 목록 → {slug: path}. 같은 slug 에 k 가 여럿이면 fail-loud."""
    found: dict[str, list[tuple[int, Path]]] = {}
    for p in sorted(labels_dir.glob("labels_*_k*.npz")):
        m = LABELS_RE.match(p.name)
        if not m:
            continue
        kk = int(m.group("k"))
        if k is not None and kk != k:
            continue
        found.setdefault(m.group("slug"), []).append((kk, p))
    if not found:
        raise SystemExit(f"labels NPZ 없음: {labels_dir} (k={k})")
    out = {}
    for slug, items in found.items():
        if only and slug not in only:
            continue
        if len(items) > 1:
            raise SystemExit(
                f"{slug}: k 가 여럿이다 {[i[0] for i in items]} — --k 로 하나를 지정하라")
        out[slug] = items[0][1]
    if only:
        missing = only - set(out)
        if missing:
            raise SystemExit(f"labels 없음: {sorted(missing)}")
    if not out:
        raise SystemExit("선택된 slug 가 없다")
    return out


def read_align_fields(shard: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """shard 에서 정렬 대조에 필요한 소용량 배열만 읽는다 (X 는 건드리지 않는다).

    NpzFile 은 접근한 멤버만 압축 해제하므로 여기서 X 는 로드되지 않는다.
    """
    with np.load(shard, allow_pickle=False) as f:
        for need in ("ep_id", "rec_idx", "phase_code"):
            if need not in f.files:
                raise SystemExit(f"{shard.name}: `{need}` 배열 없음")
        ep = np.asarray(f["ep_id"])
        rec = np.asarray(f["rec_idx"])
        n = len(np.asarray(f["phase_code"]))
    return ep, rec, n


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def rewrite_one(shard: Path, labels: Path, out: Path, k: int, dry: bool) -> dict:
    lab_f = np.load(labels, allow_pickle=False)
    try:
        for need in ("cluster", "ep_id", "rec_idx"):
            if need not in lab_f.files:
                raise SystemExit(f"{labels.name}: `{need}` 배열 없음")
        cluster = np.asarray(lab_f["cluster"])
        l_ep = np.asarray(lab_f["ep_id"])
        l_rec = np.asarray(lab_f["rec_idx"])
    finally:
        lab_f.close()

    s_ep, s_rec, n_shard = read_align_fields(shard)

    # ---- 정렬 대조 (fail-loud) ----
    if len(cluster) != n_shard:
        raise SystemExit(f"{shard.name}: labels {len(cluster)} 행 != shard {n_shard} 행")
    for name, a, b in (("ep_id", l_ep, s_ep), ("rec_idx", l_rec, s_rec)):
        if len(a) != len(b):
            raise SystemExit(f"{shard.name}: labels {name} 길이 {len(a)} != shard {len(b)}")
        if not np.array_equal(np.asarray(a).astype(np.int64),
                              np.asarray(b).astype(np.int64)):
            bad = int(np.flatnonzero(np.asarray(a).astype(np.int64)
                                     != np.asarray(b).astype(np.int64))[0])
            raise SystemExit(
                f"{shard.name}: labels 와 shard 의 {name} 순서가 다르다 "
                f"(첫 불일치 행 {bad}: labels={a[bad]} shard={b[bad]}) "
                "— 다른 shard 의 라벨이거나 재정렬된 라벨이다")

    lo, hi = int(cluster.min()), int(cluster.max())
    if lo < 0 or hi >= k:
        raise SystemExit(f"{shard.name}: cluster 범위 {lo}~{hi} 가 k={k} 와 맞지 않는다")

    new_code = np.asarray(cluster, dtype=np.int16)
    codebook = {f"c{i}": i for i in range(k)}

    if dry:
        return {"slug": shard.stem, "n_rec": n_shard, "k": k,
                "n_clusters_used": int(len(np.unique(new_code))), "written": False}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(shard, "r") as zin, zipfile.ZipFile(tmp, "w") as zout:
        names = zin.namelist()
        if "meta_json.npy" not in names:
            raise SystemExit(f"{shard.name}: meta_json.npy 없음")
        if "phase_code.npy" not in names:
            raise SystemExit(f"{shard.name}: phase_code.npy 없음")
        meta = dict(json.loads(str(np.lib.format.read_array(
            io.BytesIO(zin.read("meta_json.npy")), allow_pickle=False))))
        gt = meta.get("phase_codebook")
        meta["phase_codebook"] = codebook
        meta["phase_source"] = {
            "kind": "ae_cluster",
            "k": int(k),
            "labels_file": labels.name,
            "shard_file": shard.name,
            "note": "phase_code = ae_cluster.py 의 instruction 별 KMeans cluster id "
                    "(GT 이벤트 phase 아님). 나머지 배열은 원본 그대로.",
        }
        if gt:
            meta["phase_source"]["gt_phase_codebook"] = gt

        replace = {
            "phase_code.npy": npy_bytes(new_code),
            "meta_json.npy": npy_bytes(np.array(json.dumps(meta, ensure_ascii=False))),
        }
        for info in zin.infolist():
            if info.filename in replace:
                zout.writestr(zipfile.ZipInfo(info.filename, info.date_time),
                              replace[info.filename],
                              compress_type=info.compress_type)
                continue
            zi = zipfile.ZipInfo(info.filename, info.date_time)
            zi.compress_type = info.compress_type
            with zin.open(info, "r") as src, zout.open(zi, "w") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    tmp.replace(out)
    return {"slug": shard.stem, "n_rec": n_shard, "k": k,
            "n_clusters_used": int(len(np.unique(new_code))), "written": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard-dir", required=True, type=Path,
                    help="원본 shard NPZ 디렉토리 (예: segA). 수정하지 않는다")
    ap.add_argument("--labels-dir", required=True, type=Path,
                    help="ae_cluster.py --dump-labels 출력 디렉토리")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="사본 출력 디렉토리 (예: segA_ck8)")
    ap.add_argument("--k", type=int, default=None,
                    help="labels 파일의 k (labels_*_k<K>.npz). 미지정이면 파일명에서 유도")
    ap.add_argument("--shards", default=None, help="쉼표구분 slug 부분집합")
    ap.add_argument("--dry-run", action="store_true", help="대조만 하고 쓰지 않는다")
    args = ap.parse_args(argv)

    shard_dir = args.shard_dir.expanduser()
    labels_dir = args.labels_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    if out_dir.resolve() == shard_dir.resolve():
        raise SystemExit("--out-dir 가 --shard-dir 와 같다 (원본 덮어쓰기 금지)")

    only = set(args.shards.split(",")) if args.shards else None
    lab_map = discover_labels(labels_dir, args.k, only)

    rows = []
    for slug, lpath in sorted(lab_map.items()):
        shard = shard_dir / f"{slug}.npz"
        if not shard.is_file():
            raise SystemExit(f"shard 없음: {shard.name} (labels {lpath.name} 에 대응)")
        k = args.k if args.k is not None else int(LABELS_RE.match(lpath.name).group("k"))
        info = rewrite_one(shard, lpath, out_dir / f"{slug}.npz", k, args.dry_run)
        rows.append(info)
        print(f"[rewrite] {slug:<26} n_rec={info['n_rec']:>7} k={k} "
              f"used={info['n_clusters_used']} "
              f"{'(dry)' if not info['written'] else ''}", flush=True)

    print(f"[done] {len(rows)} shard "
          f"{'대조만' if args.dry_run else f'→ {out_dir.name}/'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
