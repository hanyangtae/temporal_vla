#!/usr/bin/env python3
"""같은 좌표를 여러 번 돌린 결과(A,B,C,...)의 success·traj.csv bit 재현 대조.

인자 = 결과 루트 디렉토리들(각각 아래 어딘가에 meta.json + traj.csv 가 정확히 하나).
출력: 각 run 의 success/행수, 기준(첫 run) 대비 traj.csv 동일 여부·최대 절대차·첫 불일치 행.

``--check-base`` 를 주면 **base 재계산 일치** 항목이 더해진다 (v6 pull 키 = 오븐·식기세척기·
서랍처럼 base 오프셋이 있는 셀). meta.json 에 기록된 ``init_robot_base_pos`` 가 대조 대상
run 들에서 모두 같아야 한다 — 오프셋을 다시 계산하는 경로(수집/재실행/eval)가 같은 base 를
내는지 보는 항목이다. 기록이 아예 없으면(=오프셋 없는 키) 정보로만 남기고 실패로 치지 않는다.

exit 0 = 전부 bit 동일(+base 일치), 1 = 불일치 있음, 2 = 산출물 결손.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE_KEY = "init_robot_base_pos"
BASE_TOL = 1e-9   # 같은 계산식의 재현이므로 사실상 bit 동일을 기대한다


def load(root: Path):
    metas = list(root.rglob("meta.json"))
    if len(metas) != 1:
        return None, f"meta.json {len(metas)}개"
    m = json.loads(metas[0].read_text())
    succ = m.get("success", m.get("episode_success"))
    csvp = metas[0].parent / "traj.csv"
    if not csvp.exists():
        return None, "traj.csv 없음"
    with csvp.open() as f:
        rd = csv.reader(f); next(rd, None)  # header
        rows = [list(map(float, r)) for r in rd if r]
    return {"success": succ, "rows": rows, "raw": csvp.read_bytes(),
            "machine": m.get("machine"), "base_pos": m.get(BASE_KEY),
            "base_lat": m.get("base_lat"), "base_back": m.get("base_back")}, None


def check_base(runs: dict, want_tags: list[str] | None) -> bool:
    """base 재계산 일치 항목. 반환 False = 불일치(게이트 실패)."""
    tags = [t for t in (want_tags or list(runs)) if t in runs]
    missing_tags = [t for t in (want_tags or []) if t not in runs]
    if missing_tags:
        print(f"BASE: 대조 대상 run 없음 {missing_tags} — 건너뜀")
    if len(tags) < 2:
        print("BASE: 대조 가능한 run 이 2개 미만 — 건너뜀")
        return True
    have = [(t, runs[t]["base_pos"]) for t in tags if runs[t]["base_pos"] is not None]
    for t in tags:
        r = runs[t]
        print(f"BASE {t}: {BASE_KEY}={r['base_pos']} lat={r['base_lat']} back={r['base_back']}")
    if not have:
        print(f"BASE: 어느 run 에도 meta.{BASE_KEY} 기록 없음 "
              "— base 오프셋 없는 키이거나 기록 누락 (실패로 치지 않음)")
        return True
    if len(have) != len(tags):
        no = [t for t in tags if runs[t]["base_pos"] is None]
        print(f"BASE: 일부 run 에만 기록 있음 (없는 run={no}) — 기록 배선 확인 필요")
        return False
    ref_tag, ref = have[0]
    ok = True
    for t, v in have[1:]:
        try:
            same = (len(v) == len(ref)
                    and all(abs(float(a) - float(b)) <= BASE_TOL for a, b in zip(ref, v)))
        except (TypeError, ValueError):
            same = (v == ref)
        if not same:
            ok = False
            print(f"BASE {ref_tag} vs {t}: 불일치 {ref} != {v}")
    print("BASE_VERDICT:", "PASS(base 재계산 일치)" if ok else "FAIL(base 불일치)")
    return ok


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="결과 루트 디렉토리들")
    ap.add_argument("--check-base", action="store_true",
                    help=f"meta.{BASE_KEY} 재계산 일치도 대조 (v6 pull 키)")
    ap.add_argument("--base-runs", default="",
                    help="base 대조에 쓸 run 태그(디렉토리 이름) 쉼표 목록. 기본=전부")
    args = ap.parse_args(argv)

    runs = {}
    bad = False
    for a in args.roots:
        r, err = load(a)
        tag = a.name
        if err:
            print(f"{tag}: 결손 — {err}"); bad = True; continue
        runs[tag] = r
        print(f"{tag}: success={r['success']} n_rows={len(r['rows'])} machine={r['machine']}")
    if bad or len(runs) < 2:
        return 2
    base_tag = next(iter(runs)); base = runs[base_tag]
    all_same = True
    for tag, r in runs.items():
        if tag == base_tag:
            continue
        same = r["raw"] == base["raw"]
        if same:
            print(f"{base_tag} vs {tag}: traj.csv BIT 동일, success {base['success']}=={r['success']}")
            continue
        all_same = False
        n = min(len(r["rows"]), len(base["rows"]))
        first = next((i for i in range(n) if r["rows"][i] != base["rows"][i]), None)
        maxd = max((abs(x - y) for i in range(n) for x, y in zip(base["rows"][i], r["rows"][i])), default=0.0)
        print(f"{base_tag} vs {tag}: 불일치 — success {base['success']} vs {r['success']}, "
              f"rows {len(base['rows'])} vs {len(r['rows'])}, 첫 불일치 행={first}, maxdiff={maxd:.3e}")
    base_ok = True
    if args.check_base:
        want = [t.strip() for t in args.base_runs.split(",") if t.strip()] or None
        base_ok = check_base(runs, want)
    print("VERDICT:", "PASS(bit 동일)" if all_same else "FAIL(불일치)",
          "" if not args.check_base else ("+ BASE PASS" if base_ok else "+ BASE FAIL"))
    return 0 if (all_same and base_ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
