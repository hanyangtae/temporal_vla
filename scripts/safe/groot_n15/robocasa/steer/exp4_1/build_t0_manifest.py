#!/usr/bin/env python3
"""exp4-1: t0 주석 manifest 생성(init) / 검증·동결(build).

init: 구제 대상 풀(사이드카 json 트리들)을 스캔해 사용자 주석용 annotation_t0.tsv 템플릿을
만든다. ITT 분모가 이 파일에서 확정된다 — 이후 행 추가·삭제 금지(주석만 기입).
  - eval-풀: exp3 e30 ho_base 실패 사이드카 (env_step GT 인라인, 재실행 불필요)
  - fit-풀: extract_fitpool_sidecars.py 가 승준 pkl 에서 추출한 사이드카
build: 사용자가 t0_env_step 을 기입한 tsv 를 검증하고 t0_record = ceil(t0/nas) 로 변환해
t0_manifest.tsv 를 동결한다 (**ceil 변환은 이 스크립트가 유일 지점** — floor 는 t0 보다
최대 nas-1 env-step 먼저 개입하는 look-ahead 라 금지, docs/steering/24a §2.2).

사용:
  python build_t0_manifest.py init --pool eval:<ho_base raw_rollouts 루트> \
      --pool fit:<fitpool_sidecars 루트> --out annotation_t0.tsv
  python build_t0_manifest.py build --annot annotation_t0.tsv --out t0_manifest.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "cell", "pool", "episode_idx", "scenario_seed", "inference_seed", "machine",
    "feasible", "feas_method", "q_max", "sidecar_path", "ann_mp4",
    "t0_env_step", "t0_record", "note",
]

# 수집/평가 머신 귀속 (exp3 실측: arms.tsv machine 열 + docs/steering/19 fit 수집 규정)
MACHINES = {
    ("pq3_ppcc_bread", "eval"): "srv48",
    ("pq3_ppcc_beer", "eval"): "srv50",
    ("pq3_drawer_left", "eval"): "srv48",
    ("pq3_drawer_right", "eval"): "srv48",
    # fit 수집은 로컬(kanu) GPU 0/1/2 (+일부 .50 가능 — A0 sentinel 에서 검증)
    ("pq3_ppcc_bread", "fit"): "kanu",
    ("pq3_ppcc_beer", "fit"): "kanu",
    ("pq3_drawer_left", "fit"): "kanu",
    ("pq3_drawer_right", "fit"): "kanu",
    # mixer 는 신규 cell — 수집·eval 모두 srv50 귀속 (2026-07-22, kanu GPU 소진으로 전환)
    ("exp41_mixer", "fit"): "srv50",
    ("exp41_mixer", "eval"): "srv50",
}

# scene feasibility (24 공유문서 §5 + 본 세션 사용자 결정: ppcc 는 필터 생략)
FEAS_DEFAULT = {
    "pq3_ppcc_bread": ("1", "not_applied"),
    "pq3_ppcc_beer": ("1", "not_applied"),
    "pq3_drawer_left": ("1", "pending_scan"),
    "pq3_drawer_right": ("1", "pending_scan"),
    # mixer 는 수집 자체가 feasible seed 만 사용 (스캔 95/100, BLOCKED 5 제외 — 07-22)
    "exp41_mixer": ("1", "joint_sweep"),
}


def scan_pool(pool: str, root: Path) -> list[dict]:
    rows = []
    for j in sorted(root.rglob("*--succ0.json")):
        if j.name.endswith(".envstep.json"):
            continue
        d = json.loads(j.read_text())
        cell = j.parent.name
        ann = j.with_name(j.stem + "--phase.mp4")
        feas, method = FEAS_DEFAULT.get(cell, ("1", "unknown"))
        rows.append({
            "cell": cell, "pool": pool,
            "episode_idx": d.get("episode_idx"),
            "scenario_seed": d.get("scenario_seed"),
            "inference_seed": d.get("inference_seed"),
            "machine": MACHINES.get((cell, pool), "unknown"),
            "feasible": feas, "feas_method": method, "q_max": "",
            "sidecar_path": str(j), "ann_mp4": str(ann) if ann.exists() else "",
            "t0_env_step": "", "t0_record": "", "note": "",
        })
    # 폴더 탐색기와 동일한 숫자 정렬 (경로 사전식 ep1,ep11,..,ep2 방지 — 사용자 주석 편의)
    rows.sort(key=lambda r: (r["cell"], int(r["episode_idx"])))
    return rows


def cmd_init(args) -> None:
    cells = set(args.cells.split(",")) if args.cells else None
    rows = []
    for spec in args.pool:
        pool, root = spec.split(":", 1)
        assert pool in ("eval", "fit"), f"pool 은 eval|fit: {spec}"
        got = scan_pool(pool, Path(root))
        if cells is not None:
            got = [r for r in got if r["cell"] in cells]
        print(f"[init] {pool}:{root} → {len(got)}판")
        rows.extend(got)
    # (cell, pool, episode) 유일성 — 분모 중복 방지
    keys = [(r["cell"], r["pool"], r["episode_idx"]) for r in rows]
    assert len(keys) == len(set(keys)), "중복 episode 발견 — 입력 트리 확인"
    with open(args.out, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    per = {}
    for r in rows:
        per.setdefault((r["cell"], r["pool"]), 0)
        per[(r["cell"], r["pool"])] += 1
    for (c, p), n in sorted(per.items()):
        print(f"  {c} [{p}]: {n}")
    print(f"[done] {len(rows)}판 → {args.out} (이후 행 추가·삭제 금지, t0_env_step 만 기입)")


def cmd_build(args) -> None:
    lines = Path(args.annot).read_text().splitlines()
    header = lines[0].split("\t")
    assert header == COLUMNS, f"컬럼 불일치: {header}"
    out_rows, n_annot = [], 0
    seen = set()
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(COLUMNS, ln.split("\t")))
        key = (r["cell"], r["pool"], r["episode_idx"])
        assert key not in seen, f"중복 행: {key}"
        seen.add(key)
        t0s = r["t0_env_step"].strip()
        if t0s == "":
            r["t0_record"] = "NA"  # 미주석 → ITT 분모 유지·비구제 계상 (24a §1)
        else:
            t0 = int(t0s)
            side = json.loads(Path(r["sidecar_path"]).read_text())
            nas = int(side.get("n_action_steps") or 5)
            n_env = len(side.get("env_step_phases") or []) - 1
            if n_env <= 0:
                n_env = int(side.get("max_episode_steps") or 720)
            assert 0 <= t0 <= n_env, f"{key}: t0={t0} 범위 밖 [0,{n_env}]"
            r["t0_record"] = str(math.ceil(t0 / nas))
        n_annot += t0s != ""
        out_rows.append(r)
    with open(args.out, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in out_rows:
            f.write("\t".join(r[c] for c in COLUMNS) + "\n")
    sha = hashlib.sha256(Path(args.annot).read_bytes()).hexdigest()[:12]
    freeze = {
        "annot_sha256": sha,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(out_rows), "n_annotated": n_annot,
    }
    Path(str(args.out) + ".freeze.json").write_text(json.dumps(freeze, indent=1))
    print(f"[done] {len(out_rows)}판 (주석 {n_annot}) → {args.out} / freeze sha={sha}")
    print("[주의] 주석 동결 — 결과를 본 후의 재주석은 별도 라운드로 표기 (24a §2.2)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--pool", action="append", required=True,
                        help="'eval:<루트>' 또는 'fit:<루트>' (반복 가능)")
    p_init.add_argument("--cells", default="",
                        help="포함할 cell 콤마 목록 (기본 전부; exp4 는 pizza_cutter 제외용)")
    p_init.add_argument("--out", type=Path, required=True)
    p_init.set_defaults(fn=cmd_init)
    p_build = sub.add_parser("build")
    p_build.add_argument("--annot", type=Path, required=True)
    p_build.add_argument("--out", type=Path, required=True)
    p_build.set_defaults(fn=cmd_build)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
