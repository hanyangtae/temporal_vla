#!/usr/bin/env python3
"""같은 좌표를 여러 번 돌린 결과(A,B,C,...)의 success·traj.csv bit 재현 대조.

인자 = 결과 루트 디렉토리들(각각 아래 어딘가에 meta.json + traj.csv 가 정확히 하나).
출력: 각 run 의 success/행수, 기준(첫 run) 대비 traj.csv 동일 여부·최대 절대차·첫 불일치 행.
exit 0 = 전부 bit 동일, 1 = 불일치 있음, 2 = 산출물 결손.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


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
        rows = [list(map(float, r)) for r in csv.reader(f)][1:] if True else []
    return {"success": succ, "rows": rows, "raw": csvp.read_bytes(), "machine": m.get("machine")}, None


def main(argv):
    runs = {}
    bad = False
    for a in argv:
        r, err = load(Path(a))
        tag = Path(a).name
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
    print("VERDICT:", "PASS(bit 동일)" if all_same else "FAIL(불일치)")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
