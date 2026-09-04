#!/usr/bin/env python3
"""shard NPZ 안 `meta_json` 의 키 필드를 새 키로 패치한다 (파일명과 일치시키기).

**실사고(2026-09-05)**: `rename_swap_keys.py` 로 oven·washer 키를 맞교환할 때 **파일명만
바꾸고 NPZ 내부 `meta_json.instruction` 은 그대로 뒀다.** 그 결과 파일명은 신 키인데 내부
meta 는 구 키를 가리키는 **모순 상태**가 됐다 — 파일명으로 찾으면 맞고 meta 로 찾으면
반대쪽 물리 대상에 걸린다. 에러가 안 나고 조용히 틀린 산출물이 나오는 최악의 경로다
(fail detector 세션이 재학습 중 발견, 그쪽 산출물 오염은 없었다).

**메모리·I/O 주의**: shard 는 2~7GB 라 `np.load` → `np.savez` 로 다시 쓰면 X 를 통째로
메모리에 올린다. npz 는 ZIP_STORED zip 이므로, 여기서는 **zip 멤버를 스트림 복사**하고
`meta_json.npy` 만 교체한다 — 메모리는 상수, I/O 는 파일 크기만큼.

사용:
    python3 patch_shard_meta_keys.py --swap OvenRack/out-left=OvenRack/out-right \
        --swap DishwasherRack/out-left=DishwasherRack/out-right \
        --dir <segA_scene> --dir <segA> [--apply]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

META_MEMBER = "meta_json.npy"
# instruction 문자열이 들어가는 필드들 (있으면 패치, 없으면 무시)
STR_FIELDS = ("instruction", "grid_instruction")


def slug_of(s: str) -> str:
    out = s
    for ch in ("/", " ", "\\", "\t"):
        out = out.replace(ch, "_")
    return out


def read_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if "meta_json" not in z:
            raise SystemExit(f"{path.name}: meta_json 없음")
        return json.loads(str(z["meta_json"]))


def patch_npz_meta(path: Path, new_meta: dict, apply: bool) -> None:
    """meta_json 멤버만 교체하고 나머지는 zip 스트림 복사 (메모리 상수)."""
    if not apply:
        return
    buf = io.BytesIO()
    np.lib.format.write_array(
        buf, np.array(json.dumps(new_meta, ensure_ascii=False)), allow_pickle=False)
    meta_bytes = buf.getvalue()

    tmp = path.with_suffix(".npz.metatmp")
    with zipfile.ZipFile(path, "r") as src, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as dst:
        for item in src.infolist():
            if item.filename == META_MEMBER:
                continue
            # ⚠ `force_zip64` 없이 스트리밍 쓰기를 하면 2GB 넘는 멤버에서
            # "File size unexpectedly exceeded ZIP64 limit" 로 죽는다. shard 의 X 는
            # 판당 수 GB 라 항상 여기에 걸린다 (실사고 2026-09-05).
            with src.open(item) as fsrc, \
                    dst.open(item.filename, "w", force_zip64=True) as fdst:
                shutil.copyfileobj(fsrc, fdst, length=8 << 20)
        dst.writestr(META_MEMBER, meta_bytes)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swap", action="append", required=True, metavar="A=B",
                    help="instruction 맞교환 (슬래시 표기, 예 OvenRack/out-left=OvenRack/out-right)")
    ap.add_argument("--dir", action="append", required=True, type=Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    mapping: dict[str, str] = {}
    for spec in a.swap:
        x, y = spec.split("=", 1)
        mapping[x.strip()], mapping[y.strip()] = y.strip(), x.strip()

    mode = "APPLY" if a.apply else "DRY-RUN"
    print(f"[meta] {mode} — 매핑 {mapping}")
    n_patch = n_ok = n_bad = 0
    for d in a.dir:
        for p in sorted(d.glob("*.npz")):
            meta = read_meta(p)
            cur = str(meta.get("instruction", ""))
            file_slug = p.stem.split("__s")[0]
            want = mapping.get(cur, cur)          # 패치 후 기대 instruction
            if cur in mapping:
                # 맞교환 대상 — 패치 후 값이 파일명과 맞아야 한다
                if slug_of(want) != file_slug:
                    print(f"  ✗ {p.name}: 패치해도 파일명과 불일치 "
                          f"(meta {cur} → {want}, 파일 {file_slug})")
                    n_bad += 1
                    continue
                for f in STR_FIELDS:
                    if f in meta:
                        meta[f] = mapping.get(str(meta[f]), meta[f])
                meta.setdefault("key_renamed_from", cur)
                print(f"  {p.name}: instruction {cur} → {want}")
                patch_npz_meta(p, meta, a.apply)
                n_patch += 1
            else:
                if slug_of(cur) != file_slug:
                    print(f"  ✗ {p.name}: meta {cur} 가 파일명 {file_slug} 과 불일치 "
                          "(맞교환 대상도 아님 — 확인 필요)")
                    n_bad += 1
                else:
                    n_ok += 1
    print(f"[meta] 패치 {n_patch} · 이미 일치 {n_ok} · 문제 {n_bad}")
    if n_bad:
        return 13
    if a.apply:
        # 사후 검증 — 전 파일에서 파일명 slug == meta instruction slug
        bad = []
        for d in a.dir:
            for p in sorted(d.glob("*.npz")):
                m = read_meta(p)
                if slug_of(str(m.get("instruction", ""))) != p.stem.split("__s")[0]:
                    bad.append(p.name)
        print(f"[meta] 사후 검증: 불일치 {len(bad)}" + (f" {bad[:5]}" if bad else " — OK"))
        return 13 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
