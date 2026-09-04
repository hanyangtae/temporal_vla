#!/usr/bin/env python3
"""scene 단위 shard(`segA_scene/<slug>__s<i>.npz`) → instruction shard(`segA/<slug>.npz`).

수집 완료 단위가 (instruction, scene) 이라 scene 별로 먼저 추출하는데, AE·KMeans 는
**instruction 단위** 규약이라 그 instruction 의 scene 이 다 모이면 하나로 합쳐야 한다.
pkl 을 다시 읽지 않고 shard 끼리 이어 붙이므로 재추출(판당 ~500MB 재독) 이 필요 없다.

**결과는 직접 추출한 instruction shard 와 배열 단위로 같아야 한다.** 그러려면 두 가지를
맞춰야 한다:

1. **행 순서** — 직접 추출본의 `ep_id` 는 `Episode.key()`
   `(plan_id, machine, scene, noise, jitter)` 오름차순 인덱스다. scene shard 안의 순서는
   이미 `(plan, machine, noise, jitter)` 정렬이므로, **scene 오름차순으로 이어 붙이면**
   전체 순서가 재현된다 — 단 plan_id·machine 이 scene 간에 같아야 성립한다(instruction 당
   머신 하나 규약). 다르면 fail-loud.
2. **phase_code** — 코드북은 shard 마다 **그 shard 에 나타난 라벨로만** 만들어진다
   (`build_shard`). 그래서 같은 정수가 scene 마다 다른 phase 를 가리킬 수 있다. 합칠 때
   코드북을 union 하고 **각 shard 의 code 를 새 코드북으로 재매핑**해야 한다. 이걸 빼먹으면
   phase 라벨이 조용히 섞인다(에러 없이 결과만 틀린다).

사용:
    python3 merge_scene_shards.py --scene-dir <out>/segA_scene --out-dir <out>/segA \
        [--slug OpenDrawer_left] [--require-scenes 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SCENE_RE = re.compile(r"^(?P<slug>.+)__s(?P<scene>\d+)$")
# record 축 열 — 순서대로 이어 붙이기만 하면 되는 것들 (phase_code·ep_id 는 별도 처리)
CONCAT_COLS = ["X", "vl", "has_vl", "scene", "noise", "jitter", "rec_idx", "succ",
               "ep_len", "jitter_reset_idx", "base_lat", "base_back"]


def group_shards(scene_dir: Path, only: str | None) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(scene_dir.glob("*.npz")):
        m = SCENE_RE.match(p.stem)
        if not m:
            continue
        slug = m.group("slug")
        if only and slug != only:
            continue
        groups[slug].append((int(m.group("scene")), p))
    for slug in groups:
        groups[slug].sort(key=lambda t: t[0])          # scene 오름차순 = 전체 순서 재현
    return dict(groups)


def merge_one(slug: str, parts: list[tuple[int, Path]], out_dir: Path,
              dry_run: bool) -> dict:
    scenes = [s for s, _ in parts]
    if len(set(scenes)) != len(scenes):
        raise SystemExit(f"{slug}: scene 중복 {scenes}")

    loaded = []
    for sc, p in parts:
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(str(d["meta_json"]))
        got = sorted(set(int(v) for v in d["scene"]))
        if got != [sc]:
            raise SystemExit(f"{p.name}: scene 열 {got} 이 파일명 s{sc} 와 다르다")
        loaded.append((sc, p, d, meta))

    # plan_id·machine 이 scene 간 같아야 (plan, machine, scene, noise, jitter) 순서가 성립
    plans = {json.dumps(m.get("plan_id"), sort_keys=True) for _, _, _, m in loaded}
    machines = {json.dumps(m.get("machine"), sort_keys=True) for _, _, _, m in loaded}
    if len(plans) != 1 or len(machines) != 1:
        raise SystemExit(
            f"{slug}: scene 간 plan_id/machine 이 다르다 (plan={plans}, machine={machines})"
            " — 직접 추출본과 행 순서가 달라지므로 병합하지 않는다")

    # phase 코드북 union + 재매핑
    union: dict[str, int] = {}
    for _, _, _, m in loaded:
        for lab in m.get("phase_codebook", {}):
            union.setdefault(lab, len(union))
    phase_parts = []
    for _, p, d, m in loaded:
        cb = m.get("phase_codebook", {})
        inv = {int(v): k for k, v in cb.items()}
        codes = d["phase_code"].astype(np.int32)
        uniq = set(int(c) for c in np.unique(codes))
        missing = uniq - set(inv)
        if missing:
            raise SystemExit(f"{p.name}: 코드북에 없는 phase_code {sorted(missing)}")
        lut = np.full(max(inv) + 1 if inv else 1, -1, dtype=np.int16)
        for code, lab in inv.items():
            lut[code] = union[lab]
        phase_parts.append(lut[codes])

    arrays: dict[str, np.ndarray] = {}
    for col in CONCAT_COLS:
        have = [d for _, _, d, _ in loaded if col in d]
        if not have:
            continue
        if len(have) != len(loaded):
            raise SystemExit(f"{slug}: 열 {col} 이 일부 shard 에만 있다 — 추출기 버전 불일치")
        arrays[col] = np.concatenate([d[col] for _, _, d, _ in loaded], axis=0)
    arrays["phase_code"] = np.concatenate(phase_parts).astype(np.int16)

    # ep_id 재번호 — shard 마다 0부터 다시 시작하므로 offset 누적
    ep_parts, off = [], 0
    for _, _, d, _ in loaded:
        ep = d["ep_id"].astype(np.int64)
        ep_parts.append((ep + off).astype(np.int32))
        off += int(ep.max()) + 1
    arrays["ep_id"] = np.concatenate(ep_parts)

    base_meta = dict(loaded[0][3])
    base_meta["phase_codebook"] = union
    base_meta["n_episodes"] = int(off)
    base_meta["sigs"] = [s for _, _, _, m in loaded for s in m.get("sigs", [])]
    base_meta["merged_from"] = [f"{slug}__s{sc}" for sc, _, _, _ in loaded]
    base_meta["merged_scenes"] = scenes
    arrays["meta_json"] = np.array(json.dumps(base_meta, ensure_ascii=False))

    info = {"slug": slug, "scenes": scenes, "n_episodes": int(off),
            "n_records": int(arrays["ep_id"].shape[0]),
            "n_phase_labels": len(union)}
    if dry_run:
        print(f"[merge] (dry-run) {slug}: {info}")
        return info

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug}.npz"
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(out)
    size = out.stat().st_size / 1e9
    print(f"[merge] {slug}: scene {scenes} → {out.name} "
          f"판 {info['n_episodes']} · record {info['n_records']} · "
          f"phase {info['n_phase_labels']}종 · {size:.2f}GB")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--slug", default=None, help="이 slug 만 병합 (기본: 전부)")
    ap.add_argument("--require-scenes", type=int, default=0,
                    help="scene 이 이 개수 미만이면 건너뛴다 (완주 전 오병합 방지)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    groups = group_shards(a.scene_dir, a.slug)
    if not groups:
        raise SystemExit(f"{a.scene_dir}: `<slug>__s<i>.npz` 가 없다")
    done = 0
    for slug, parts in sorted(groups.items()):
        if a.require_scenes and len(parts) < a.require_scenes:
            print(f"[merge] skip {slug} — scene {len(parts)}/{a.require_scenes} "
                  f"(있는 것: {[s for s, _ in parts]})")
            continue
        merge_one(slug, parts, a.out_dir, a.dry_run)
        done += 1
    print(f"[merge] 완료 {done}/{len(groups)} slug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
