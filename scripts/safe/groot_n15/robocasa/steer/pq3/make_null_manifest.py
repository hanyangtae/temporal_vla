#!/usr/bin/env python3
"""pq3 null(위약) fit manifest 생성 — 라벨 permutation, 사전고정 seed (계획서 v9 arm 표).

fit manifest(pkl\tlabel\tscene)의 label 열만 고정 seed 로 permute 한다 (클래스 수 보존).
위약 결정은 전부 manifest 생성 단계에서 (fit 은 무수정 경로) — pq2 관례 유지.
사용: python3 make_null_manifest.py --fit-manifest <in.tsv> --out <out.tsv> [--perm-seed 1]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm-seed", type=int, default=1,
                    help="라벨 permutation seed (사전고정 — 계획서 v9: seed 1)")
    args = ap.parse_args()

    src = Path(args.fit_manifest)
    text = src.read_text()
    rows, comments = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise SystemExit(f"{src}: 비정상 행 {line[:60]}")
        rows.append(parts)

    labels = [int(p[1]) for p in rows]
    perm = np.random.default_rng(args.perm_seed).permutation(len(rows))
    new_labels = [labels[i] for i in perm]
    assert sorted(new_labels) == sorted(labels), "permutation 은 클래스 수 보존"
    if new_labels == labels:
        raise SystemExit("permutation 이 항등 — perm-seed 재고 필요")

    out_lines = [
        f"# pq3 NULL(위약) fit manifest — 라벨 permutation seed={args.perm_seed}",
        f"# source={src} sha={hashlib.sha256(text.encode()).hexdigest()[:16]}",
        *comments,
    ]
    for parts, lb in zip(rows, new_labels):
        out_lines.append("\t".join([parts[0], str(lb), *parts[2:]]))
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    n1 = sum(new_labels)
    print(f"[null] {len(rows)} rows (succ {n1}/fail {len(rows)-n1}) "
          f"flipped={sum(a != b for a, b in zip(labels, new_labels))} -> {args.out}")


if __name__ == "__main__":
    main()
