#!/usr/bin/env python3
"""아카이브 셀의 좌표+지문을 TSV 로 동결한다 (meta.json 만 읽는다 — pkl 미접근).

용도: 아카이브 **rebase/rename 전후 대조**의 기준표. 예정된 oven·washer
left↔right 맞교환처럼 "경로·이름만 바뀌고 pkl 은 불변" 이라는 주장을 검증하려면,
바뀌기 **전** 시점의 (좌표, 파일 지문) 표가 있어야 한다. 인덱스는 재생성되며 덮어써지고
부분 인덱스는 15분 주기로 갱신되므로, 별도 동결본이 필요하다.

지문은 `pkl_sha256`(파일 자체의 해시)을 정본으로 쓴다 — `sig` 는 산출 규약이 바뀌면
정의가 흔들릴 수 있지만 sha256 은 "pkl 불변" 을 직접 판정한다(실측상 sig 는 sha256 의
앞 16자다). 대조 키는 `(grid_instruction, scene_idx, jitter_idx, noise_idx, pkl_sha256)`
5-튜플 — 이러면 pkl 불변·좌표 보존·교환 1:1 이 한 번에 판정되고, 어느 셀이 어디로
잘못 갔는지도 드러난다.

사용:
    python3 snapshot_archive_fingerprints.py --grid-root <grid>/<plan_id> \
        --out prerebase_<날짜>.tsv [--filter OvenRack --filter DishwasherRack]

대조(나중에):
    python3 snapshot_archive_fingerprints.py --compare pre.tsv post.tsv \
        --swap "OvenRack/out-left=OvenRack/out-right" \
        --swap "DishwasherRack/out-left=DishwasherRack/out-right"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLS = ["grid_instruction", "scene_idx", "jitter_idx", "noise_idx",
        "pkl_sha256", "sig", "success", "plan_id", "machine", "rel_dir"]

# `--emit-index` 로 낼 추출기 호환 인덱스의 열. build_grid_index.py 산출과 같은 이름을
# 쓰되, extract_grid_matrix.read_index 가 실제로 읽는 것만 담는다(그 함수는 헤더로
# 열을 찾으므로 여분 열이 없어도 된다). 수집 측 부분 인덱스 생성이 막혔을 때
# **아카이브 자체를 진실의 출처로** 삼아 추출을 진행하기 위한 우회로다.
INDEX_COLS = ["plan_id", "machine", "grid_instruction", "scene_idx", "jitter_idx",
              "noise_idx", "jitter_reset_idx", "base_lat", "base_back",
              "armsig", "has_pkl", "sig", "pkl_sha256", "success", "rel_path"]


def scan(grid_root: Path, filters: list[str],
         scenes: set[int] | None = None) -> list[dict]:
    rows = []
    for meta in sorted(grid_root.rglob("meta.json")):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:                       # 손상 파일도 조용히 넘기지 않는다
            raise SystemExit(f"{meta}: meta.json 파싱 실패 — {e}")
        instr = str(d.get("grid_instruction", ""))
        if filters and not any(f in instr for f in filters):
            continue
        if scenes is not None and int(d.get("scene_idx", -1)) not in scenes:
            continue
        row = {c: d.get(c, "") for c in COLS if c != "rel_dir"}
        rel_dir = str(meta.parent.relative_to(grid_root))
        row["rel_dir"] = rel_dir
        row["_has_pkl"] = 1 if (meta.parent / "rollout.pkl").is_file() else 0
        row["_armsig"] = d.get("armsig", "base")
        row["_jitter_reset_idx"] = d.get("jitter_reset_idx", "")
        row["_base_lat"] = d.get("base_lat", 0.0)
        row["_base_back"] = d.get("base_back", 0.0)
        rows.append(row)
    return rows


def write_index(rows: list[dict], out: Path, plan_id: str) -> None:
    """추출기 호환 인덱스 TSV. `rel_path` 는 `<plan_id>/<셀 상대경로>` 규약."""
    lines = ["\t".join(INDEX_COLS)]
    for r in rows:
        vals = {
            "plan_id": r.get("plan_id", plan_id),
            "machine": r.get("machine", ""),
            "grid_instruction": r.get("grid_instruction", ""),
            "scene_idx": r.get("scene_idx", ""),
            "jitter_idx": r.get("jitter_idx", ""),
            "noise_idx": r.get("noise_idx", ""),
            "jitter_reset_idx": r.get("_jitter_reset_idx", ""),
            "base_lat": r.get("_base_lat", 0.0),
            "base_back": r.get("_base_back", 0.0),
            "armsig": r.get("_armsig", "base"),
            "has_pkl": r.get("_has_pkl", 1),
            "sig": r.get("sig", ""),
            "pkl_sha256": r.get("pkl_sha256", ""),
            "success": r.get("success", ""),
            "rel_path": f"{plan_id}/{r['rel_dir']}",
        }
        lines.append("\t".join(str(vals[c]) for c in INDEX_COLS))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(rows: list[dict], out: Path) -> None:
    lines = ["\t".join(COLS)]
    lines += ["\t".join(str(r.get(c, "")) for c in COLS) for r in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tsv(p: Path) -> list[dict]:
    lines = p.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def compare(pre: Path, post: Path, swaps: list[str]) -> int:
    """pre 에 교환 매핑을 적용해 post 와 5-튜플 집합 대조. 반환 = 종료코드."""
    mapping: dict[str, str] = {}
    for s in swaps:
        a, b = s.split("=", 1)
        mapping[a] = b
        mapping[b] = a                                # 맞교환이므로 양방향

    def key(r, apply_map: bool):
        instr = r["grid_instruction"]
        if apply_map:
            instr = mapping.get(instr, instr)
        return (instr, r["scene_idx"], r["jitter_idx"], r["noise_idx"],
                r["pkl_sha256"])

    pre_rows, post_rows = read_tsv(pre), read_tsv(post)
    pre_keys = {key(r, True) for r in pre_rows}
    post_keys = {key(r, False) for r in post_rows}
    only_pre = pre_keys - post_keys
    only_post = post_keys - pre_keys

    print(f"[compare] pre {len(pre_rows)}행 / post {len(post_rows)}행 "
          f"(교환 적용 {sorted(mapping)})")
    if not only_pre and not only_post and len(pre_rows) == len(post_rows):
        print("[compare] OK — 좌표·pkl_sha256 전부 1:1 일치 (pkl 불변·교환 정확)")
        return 0
    print(f"[compare] FAIL — pre에만 {len(only_pre)} / post에만 {len(only_post)}")
    for k in sorted(only_pre)[:10]:
        print(f"   pre-only : {k}")
    for k in sorted(only_post)[:10]:
        print(f"   post-only: {k}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", type=Path,
                    help="<grid>/<plan_id> — 이 아래 meta.json 을 전부 읽는다")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--filter", action="append", default=[],
                    help="instruction 부분문자열 필터 (여러 번 가능). 없으면 전부")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("PRE", "POST"))
    ap.add_argument("--swap", action="append", default=[], metavar="A=B",
                    help="맞교환 매핑 (compare 전용)")
    ap.add_argument("--scenes", default=None,
                    help="scene_idx 필터, 쉼표 목록 (예: 0,1). 없으면 전부")
    ap.add_argument("--emit-index", action="store_true",
                    help="지문표 대신 **추출기 호환 인덱스**를 쓴다 (수집 측 인덱스 "
                         "생성이 막혔을 때 아카이브를 출처로 삼는 우회로)")
    a = ap.parse_args()

    if a.compare:
        return compare(a.compare[0], a.compare[1], a.swap)

    if not a.grid_root or not a.out:
        ap.error("--grid-root 와 --out 이 필요하다 (또는 --compare)")
    if not a.grid_root.is_dir():
        raise SystemExit(f"없음: {a.grid_root}")
    scenes = ({int(s) for s in a.scenes.split(",") if s.strip() != ""}
              if a.scenes else None)
    rows = scan(a.grid_root, a.filter, scenes)
    if not rows:
        raise SystemExit(f"{a.grid_root}: 조건에 맞는 meta.json 이 없다")
    n_instr = len({r["grid_instruction"] for r in rows})
    n_hash = len({r["pkl_sha256"] for r in rows})
    if a.emit_index:
        write_index(rows, a.out, a.grid_root.name)
        n_nopkl = sum(1 for r in rows if not r["_has_pkl"])
        print(f"[index] {len(rows)}행 / instruction {n_instr} / pkl 없음 {n_nopkl} "
              f"→ {a.out}")
        if n_nopkl:
            print(f"[index] ⚠ pkl 없는 셀 {n_nopkl}건은 has_pkl=0 — 추출기가 건너뛴다",
                  file=sys.stderr)
        return 0
    write_tsv(rows, a.out)
    print(f"[snapshot] {len(rows)}셀 / instruction {n_instr} / "
          f"고유 pkl_sha256 {n_hash} → {a.out}")
    if n_hash != len(rows):
        print(f"[snapshot] ⚠ 지문 중복 {len(rows) - n_hash}건 — 같은 pkl 이 여러 좌표에 "
              "있다는 뜻이라 대조 전에 원인 확인 필요", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
