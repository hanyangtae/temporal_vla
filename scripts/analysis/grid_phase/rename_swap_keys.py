#!/usr/bin/env python3
"""키 **맞교환** — shard·병합본·라벨 파일명 + 번들 안 키를 한 번에 바꾼다.

oven·washer 의 left↔right 의미 개정(2026-09-05 rebase)처럼 **두 키가 서로 이름을
바꾸는** 경우를 다룬다. pkl·활성화 내용은 불변이고 이름만 바뀌므로 재추출·재학습은
필요 없다 — 다만 순진하게 치환하면 조용히 망가지는 자리가 셋 있다:

1. **맞교환은 치환이 아니다.** `A→B` 를 먼저 하면 기존 B 가 덮여서 두 키가 같은 값이
   된다. 파일도 payload 키도 **임시 이름 경유**(A→tmp, B→A, tmp→B)로 처리한다.
2. **encoder·scaler 는 건드리지 않는다.** `mu`·`scalar_std`·`enc.*`·`arch`·`k`·`latent`
   는 전 shard 공용이라 키와 무관하다. 바꾸는 건 `centers.<slug>` · `slugs` 배열 ·
   `provenance` 안의 shard 목록뿐이다.
3. **번들과 라벨 파일은 같이 바꿔야 한다.** 소비 측(연산자 fit)이 "번들 centers 로 파생한
   라벨 == labels 파일" 을 검증하므로, 한쪽만 바꾸면 그 자리에서 죽는다.

검증: 맞교환 뒤 `centers.A` 가 교환 **전** `centers.B` 와 배열 동일한지 대조한다.
덮어쓰기 사고가 났으면 두 키 값이 같아져 여기서 바로 드러난다.

사용:
    python3 rename_swap_keys.py --swap OvenRack_out-left=OvenRack_out-right \
        --swap DishwasherRack_out-left=DishwasherRack_out-right \
        --bundle <ae>/ae_bundle_k8.npz --labels-dir <ae> --k 8 \
        --shard-dir <segA_scene> --merged-dir <segA> [--apply]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

CENTER_PREFIX = "centers."


def parse_swaps(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--swap {s!r} 형식은 A=B")
        a, b = s.split("=", 1)
        out.append((a.strip(), b.strip()))
    return out


def swap_files(paths: dict[str, Path], apply: bool) -> None:
    """{현재경로: 목표경로} 를 임시 이름 경유로 맞교환."""
    tmps = {}
    for src in paths:
        tmp = Path(str(src) + ".swaptmp")
        print(f"  {src.name} → (tmp)")
        if apply:
            src.rename(tmp)
        tmps[src] = tmp
    for src, dst in paths.items():
        print(f"  (tmp) → {dst.name}")
        if apply:
            tmps[src].rename(dst)


def rename_bundle(bundle: Path, swaps: list[tuple[str, str]], apply: bool) -> None:
    with np.load(bundle, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}

    mapping: dict[str, str] = {}
    for a, b in swaps:
        mapping[a], mapping[b] = b, a

    before = {}
    for a, b in swaps:
        for s in (a, b):
            key = CENTER_PREFIX + s
            if key not in payload:
                raise SystemExit(f"{bundle.name}: {key} 없음 — 맞교환 대상이 아니다")
            before[s] = payload[key].copy()

    # centers 맞교환 (사전에 모두 복사해 두고 한 번에 배치 — 덮어쓰기 불가능)
    for s, arr in before.items():
        payload[CENTER_PREFIX + mapping[s]] = arr

    # slugs 배열
    if "slugs" in payload:
        payload["slugs"] = np.asarray(
            [mapping.get(str(v), str(v)) for v in payload["slugs"]], dtype=np.str_)

    # provenance 안의 shard 목록 (있으면)
    if "provenance" in payload:
        prov = json.loads(str(payload["provenance"]))
        for field in ("shards", "shard_names", "slugs"):
            if isinstance(prov.get(field), list):
                prov[field] = [mapping.get(str(v), str(v)) for v in prov[field]]
        prov.setdefault("key_swaps", []).append(
            {"swaps": [list(s) for s in swaps], "note": "left<->right 의미 개정"})
        payload["provenance"] = np.array(json.dumps(prov, ensure_ascii=False))

    # 검증 — 맞교환 후 centers.A 는 교환 전 centers.B 와 같아야 한다
    ok = True
    for a, b in swaps:
        if not np.array_equal(payload[CENTER_PREFIX + a], before[b]):
            print(f"  ✗ centers.{a} != (교환 전) centers.{b}")
            ok = False
        if np.array_equal(payload[CENTER_PREFIX + a], payload[CENTER_PREFIX + b]):
            print(f"  ✗ centers.{a} == centers.{b} — 덮어쓰기 사고")
            ok = False
    print(f"  번들 맞교환 검증: {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("번들 맞교환 검증 실패 — 쓰지 않고 중단")

    if apply:
        tmp = bundle.with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **payload)
        tmp.replace(bundle)
        print(f"  → {bundle.name} 갱신")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swap", action="append", required=True, metavar="A=B")
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--shard-dir", type=Path, default=None,
                    help="scene shard 디렉토리 (`<slug>__s<i>.npz`)")
    ap.add_argument("--merged-dir", type=Path, default=None,
                    help="병합본 디렉토리 (`<slug>.npz`)")
    ap.add_argument("--apply", action="store_true", help="없으면 dry-run")
    a = ap.parse_args()
    swaps = parse_swaps(a.swap)
    mapping = {}
    for x, y in swaps:
        mapping[x], mapping[y] = y, x

    mode = "APPLY" if a.apply else "DRY-RUN"
    print(f"[swap] {mode} — {[f'{x}<->{y}' for x, y in swaps]}")

    if a.merged_dir:
        print("[swap] 병합본:")
        paths = {}
        for s, t in mapping.items():
            p = a.merged_dir / f"{s}.npz"
            if p.exists():
                paths[p] = a.merged_dir / f"{t}.npz"
        swap_files(paths, a.apply)

    if a.shard_dir:
        print("[swap] scene shard:")
        paths = {}
        for p in sorted(a.shard_dir.glob("*.npz")):
            stem = p.stem
            if "__s" not in stem:
                continue
            slug, tail = stem.split("__s", 1)
            if slug in mapping:
                paths[p] = a.shard_dir / f"{mapping[slug]}__s{tail}.npz"
        swap_files(paths, a.apply)

    if a.labels_dir:
        print("[swap] 라벨:")
        paths = {}
        for s, t in mapping.items():
            p = a.labels_dir / f"labels_{s}_k{a.k}.npz"
            if p.exists():
                paths[p] = a.labels_dir / f"labels_{t}_k{a.k}.npz"
        swap_files(paths, a.apply)

    if a.bundle:
        print("[swap] 번들:")
        if not a.apply:
            # dry-run 이어도 검증은 실제로 돌린다 (쓰지만 않는다)
            rename_bundle(a.bundle, swaps, apply=False)
        else:
            shutil.copy2(a.bundle, a.bundle.with_suffix(".npz.prerebase"))
            print(f"  백업 → {a.bundle.name}.prerebase")
            rename_bundle(a.bundle, swaps, apply=True)

    print(f"[swap] {mode} 완료" + ("" if a.apply else " — --apply 로 실행할 것"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
