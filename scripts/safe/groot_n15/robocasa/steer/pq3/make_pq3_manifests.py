#!/usr/bin/env python3
"""pq3 seed 레이아웃 manifest 생성기 (계획서 dynamic-riding-aurora v9 §Seed 레이아웃).

pq2 make_fit_manifests.py 는 디스크의 전 episode 를 층화 split 하므로 pq3 의
scene-diverse seed 구획(fit S1-S15+backfill / eval 30 예약·동결 / sweep=fit 재사용)을
강제할 수 없다 (Codex Gate1 R1#1). 이 도구가 pq3 의 seed 단일 출처다.

단계 (순서대로):
  plan   : seed 소스 tsv(cell 블록, seed 오름차순)에서 cell 별 수집 계획
           S1..S{n} × (env_seed_i, noise_seed_i = COLLECT_NOISE_BASE + i*NOISE_STRIDE)
           → <out>/<cell>/collect_plan.tsv. 수집·β sweep·(seen eval 의 env_seed)이
           전부 이 쌍을 공유한다 (arm 간 paired — noise 시리즈 사전 고정).
  freeze : p0 게이트 통과 후 — 수집 결과(succ 스템)를 스캔해
           fit_manifest.tsv(pkl\tlabel\tseed, fit --manifest 계약) +
           sweep_manifest.tsv(fit 재사용: ep_idx, env_seed, noise_seed, base_label) +
           eval_manifest.tsv(30행: episode_idx, env_seed, noise_seed, split=seen|unseen,
           noise = EVAL_NOISE_BASE + ep*NOISE_STRIDE — seen 15 = fit 실사용 seed 첫 15
           (새 noise), unseen 15 = fit 전체와 교집합 0 인 다음 fresh 15) +
           eval_reserved.json(+sha) **동결** — 재실행 시 내용이 다르면 abort.
  pool   : task-pooled fit manifest (drawer 2 cell / ppcc 3 cell 합본 — fit 은 task 단위).
  check  : fit manifest × eval_reserved 교차 검증 — unseen seed 가 fit 에 등장하면
           exit 5 (fit30 확장·러너 preflight·Gate A 회귀 테스트가 호출).

하드 규칙: eval 예약 침범 = 즉시 abort. 동결 후 freeze 재실행은 내용 동일할 때만 통과.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# noise seed 시리즈 (사전 고정 — episode index 별 상이, arm 간 공유).
# 클라이언트(http_feature_collect)는 episode 시작값에서 추론 call 마다 +1 하므로
# stride 는 episode 내 call 수(≤ 720/5 = 144)보다 충분히 크게.
# cell 별 오프셋(CELL_NOISE_SPAN): 같은 noise stream 이 cell 간 반복되지 않도록
# (Gate 2 중간#1 — task-pool pair 독립성). collect/eval 대역은 서로 disjoint.
COLLECT_NOISE_BASE = 500000
EVAL_NOISE_BASE = 3000000
CELL_NOISE_SPAN = 100000
NOISE_STRIDE = 1000

# csv 스템 = 수집 삼중항의 마커 (pkl 은 승준 직송 후 로컬 삭제될 수 있음 — csv 로 판정 유지)
STEM_RE = re.compile(r"^task(?P<ci>\d+)--ep(?P<ep>\d+)--succ(?P<succ>[01])\.(pkl|json|csv)$")


def read_seed_source(tsv_path: Path, tsv_cell_index: int) -> list[dict]:
    """selected_instruction_seeds.tsv 에서 cell_index 블록의 행(순서 보존) 반환."""
    rows = []
    header = None
    for line in tsv_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if header is None:
            header = parts
            continue
        row = dict(zip(header, parts))
        if int(row["cell_index"]) == tsv_cell_index:
            rows.append(row)
    if not rows:
        raise SystemExit(f"seed 소스에 cell_index={tsv_cell_index} 행 없음: {tsv_path}")
    return rows


def collect_noise_seed(cell_index: int, i: int) -> int:
    return COLLECT_NOISE_BASE + cell_index * CELL_NOISE_SPAN + i * NOISE_STRIDE


def eval_noise_seed(cell_index: int, ep_idx: int) -> int:
    return EVAL_NOISE_BASE + cell_index * CELL_NOISE_SPAN + ep_idx * NOISE_STRIDE


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def cmd_plan(args) -> None:
    rows = read_seed_source(Path(args.seeds_tsv), args.tsv_cell_index)
    seeds = [int(r["scenario_seed"]) for r in rows]
    if len(seeds) != len(set(seeds)):
        dup = sorted({s for s in seeds if seeds.count(s) > 1})
        raise SystemExit(f"seed 소스 중복 scenario_seed: {dup[:5]} (cell 블록 오염)")
    instrs = {r["canonical_instruction"] for r in rows}
    if len(instrs) != 1:
        raise SystemExit(f"seed 소스 canonical_instruction 불일치: {sorted(instrs)[:3]}")
    n = min(args.n_plan, len(rows))
    if n < args.n_plan:
        print(f"[warn] seed 소스 {len(rows)}개 < 계획 {args.n_plan} — {n}개로 절단")
    out_dir = Path(args.out_dir) / args.cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# pq3 collect plan — S index 순서(수집·backfill 은 이 순서 준수)",
        f"# cell_id={args.cell_id} tsv_cell_index={args.tsv_cell_index} "
        f"task={rows[0]['task']} env={rows[0]['env_name']}",
        f"# canonical_instruction={rows[0]['canonical_instruction']}",
        f"# noise: base={COLLECT_NOISE_BASE}+cell*{CELL_NOISE_SPAN} stride={NOISE_STRIDE}",
        "s_idx\tenv_seed\tnoise_seed",
    ]
    for i in range(n):
        lines.append(
            f"{i}\t{rows[i]['scenario_seed']}\t{collect_noise_seed(args.tsv_cell_index, i)}"
        )
    _write_atomic(out_dir / "collect_plan.tsv", "\n".join(lines) + "\n")
    print(f"[plan] {args.cell_id}: {n} seeds -> {out_dir / 'collect_plan.tsv'}")


def _read_plan(out_dir: Path) -> list[dict]:
    plan_path = out_dir / "collect_plan.tsv"
    if not plan_path.exists():
        raise SystemExit(f"collect_plan.tsv 없음 (plan 단계 선행): {plan_path}")
    rows = []
    for line in plan_path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("s_idx"):
            continue
        s_idx, env_seed, noise_seed = line.split("\t")
        rows.append({"s_idx": int(s_idx), "env_seed": int(env_seed), "noise_seed": int(noise_seed)})
    return rows


def scan_collected(collected_dir: Path) -> dict[int, dict]:
    """수집 디렉토리의 succ 스템 스캔 → {ep_idx: {succ, stem}} (ep_idx = plan s_idx).

    같은 ep 에 succ 값이 **둘 이상**이면 abort (재수집 잔재의 무음 선택 방지 —
    Gate2 R2 중간#3). 동일 stem 의 pkl/csv/json 공존만 중복 표현으로 허용
    (pkl 은 승준 직송으로 로컬에 없을 수 있음 — csv 마커가 판정 유지).
    """
    rank = {".pkl": 0, ".csv": 1, ".json": 2}
    out: dict[int, dict] = {}
    succs: dict[int, set] = {}
    for p in sorted(collected_dir.glob("task*--ep*--succ*")):
        m = STEM_RE.match(p.name)
        if not m:
            continue
        ep = int(m.group("ep"))
        succs.setdefault(ep, set()).add(int(m.group("succ")))
        if ep not in out or rank[p.suffix] < rank[out[ep]["path"].suffix]:
            out[ep] = {"succ": int(m.group("succ")), "path": p, "stem": p.stem}
    conflicts = {ep: sorted(v) for ep, v in succs.items() if len(v) > 1}
    if conflicts:
        raise SystemExit(
            f"수집 ep 에 상충 succ 스템 공존: {conflicts} — 재수집 잔재 정리 후 재실행"
        )
    return out


def cmd_freeze(args) -> None:
    out_dir = Path(args.out_dir) / args.cell_id
    plan = _read_plan(out_dir)
    plan_by_idx = {r["s_idx"]: r for r in plan}
    collected = scan_collected(Path(args.collected_dir))
    unknown = sorted(set(collected) - set(plan_by_idx))
    if unknown:
        raise SystemExit(f"수집 ep 가 collect_plan 밖: {unknown} (seed 구획 위반)")
    fit_eps = sorted(collected)
    if len(fit_eps) < args.fit_expect:
        raise SystemExit(
            f"fit 수집 {len(fit_eps)}판 < 기대 {args.fit_expect} — p0 게이트/backfill 미완"
        )
    # 수집 순서 강제: S0..S(n-1) prefix 여야 함 (중간 건너뜀 = seed-순서 선발 위반)
    if fit_eps != list(range(len(fit_eps))):
        raise SystemExit(
            f"수집 ep 가 plan prefix 가 아님: {fit_eps} (S 순서 수집·backfill 위반)"
        )
    fit_used_seeds = [plan_by_idx[e]["env_seed"] for e in fit_eps]

    # ── 모든 산출물을 메모리에서 먼저 구성 (동결 비교 후에만 기록 — Gate 2 높음#7) ──
    # fit manifest (fit --manifest 계약: pkl\tlabel\tscene) — scene 열에 env_seed 기록.
    # --pkl-prefix: fit 이 승준에서 돌므로 pkl 경로 루트를 승준 기준으로 재작성 가능.
    pkl_prefix = Path(args.pkl_prefix) if args.pkl_prefix else None
    fit_lines = [f"# pq3 fit manifest cell={args.cell_id} (label=env 원판정 stem succ)"]
    for e in fit_eps:
        rec = collected[e]
        pkl_path = (pkl_prefix / f"{rec['stem']}.pkl") if pkl_prefix else rec["path"].with_suffix(".pkl")
        fit_lines.append(f"{pkl_path}\t{rec['succ']}\t{plan_by_idx[e]['env_seed']}")
    fit_text = "\n".join(fit_lines) + "\n"

    sweep_lines = [
        "# pq3 beta-sweep manifest — fit 재사용 (COAST Stage3 faithful), 참조=base_label",
        "ep_idx\tenv_seed\tnoise_seed\tbase_label",
    ]
    for e in fit_eps:
        r = plan_by_idx[e]
        sweep_lines.append(f"{e}\t{r['env_seed']}\t{r['noise_seed']}\t{collected[e]['succ']}")
    sweep_text = "\n".join(sweep_lines) + "\n"

    # eval 30: seen 15 = fit 실사용 seed 첫 15 (새 noise 시리즈), unseen 15 = 교집합 0 fresh
    seen_seeds = fit_used_seeds[: args.eval_seen]
    fit_used_set = set(fit_used_seeds)
    src_rows = read_seed_source(Path(args.seeds_tsv), args.tsv_cell_index)
    fresh = [int(r["scenario_seed"]) for r in src_rows if int(r["scenario_seed"]) not in fit_used_set]
    if len(fresh) < args.eval_unseen:
        raise SystemExit(
            f"unseen fresh seed 부족: {len(fresh)} < {args.eval_unseen} (seed 소스 확장 필요)"
        )
    unseen_seeds = fresh[: args.eval_unseen]
    assert not (set(unseen_seeds) & fit_used_set), "unseen ∩ fit ≠ ∅ (구현 버그)"

    eval_lines = [
        f"# pq3 eval manifest cell={args.cell_id} — 예약·동결 (침범 시 abort)",
        f"# noise: base={EVAL_NOISE_BASE}+cell*{CELL_NOISE_SPAN} stride={NOISE_STRIDE} (arm 간 공유)",
        "episode_idx\tenv_seed\tnoise_seed\tsplit",
    ]
    for ep, seed in enumerate(seen_seeds + unseen_seeds):
        split = "seen" if ep < len(seen_seeds) else "unseen"
        eval_lines.append(f"{ep}\t{seed}\t{eval_noise_seed(args.tsv_cell_index, ep)}\t{split}")
    eval_text = "\n".join(eval_lines) + "\n"

    reserved = {
        "cell_id": args.cell_id,
        "tsv_cell_index": args.tsv_cell_index,
        "fit_used_seeds": fit_used_seeds,
        "seen_seeds": seen_seeds,
        "unseen_seeds": unseen_seeds,
        "eval_noise_base": EVAL_NOISE_BASE,
        "cell_noise_span": CELL_NOISE_SPAN,
        "noise_stride": NOISE_STRIDE,
        "eval_manifest_sha": _sha(eval_text),
        "fit_manifest_sha": _sha(fit_text),
        "sweep_manifest_sha": _sha(sweep_text),
    }
    reserved_text = json.dumps(reserved, indent=2, ensure_ascii=False)

    reserved_path = out_dir / "eval_reserved.json"
    built = {
        "eval_reserved.json": reserved_text,
        "fit_manifest.tsv": fit_text,
        "sweep_manifest.tsv": sweep_text,
        "eval_manifest.tsv": eval_text,
    }
    if reserved_path.exists():
        # 동결 상태: 어떤 산출물도 내용이 달라지면 기록 없이 abort (부분 덮어쓰기 금지)
        diffs = [
            name for name, text in built.items()
            if (out_dir / name).exists() and (out_dir / name).read_text().strip() != text.strip()
        ]
        if diffs:
            raise SystemExit(
                f"[동결 위반] {diffs} 내용 불일치 — eval 예약은 freeze 1회 후 불변 "
                "(fit30 확장이 침범했는지 확인하라; 아무 파일도 덮어쓰지 않음)"
            )
        print(f"[freeze] 기존 동결과 동일 (idempotent): {out_dir}")
    for name, text in built.items():
        _write_atomic(out_dir / name, text)
    print(
        f"[freeze] {args.cell_id}: fit={len(fit_eps)}ep seen={len(seen_seeds)} "
        f"unseen={len(unseen_seeds)} sha={reserved['eval_manifest_sha']} -> {out_dir}"
    )


def cmd_pool(args) -> None:
    out_path = Path(args.task_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# pq3 task-pooled fit manifest cells={args.cells}"]
    for cell in args.cells.split(","):
        cell_manifest = Path(args.out_dir) / cell.strip() / "fit_manifest.tsv"
        if not cell_manifest.exists():
            raise SystemExit(f"cell fit manifest 없음 (freeze 선행): {cell_manifest}")
        for line in cell_manifest.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                lines.append(line)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[pool] {args.cells} -> {out_path} ({len(lines) - 1} eps)")


def cmd_check(args) -> None:
    reserved = json.loads(Path(args.eval_reserved).read_text())
    unseen = set(int(s) for s in reserved["unseen_seeds"])
    violations, malformed = [], []
    for ln, line in enumerate(Path(args.fit_manifest).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3 or not parts[2].strip().lstrip("-").isdigit():
            # fail closed: scene(seed) 열 없는 행은 침범 검사를 우회하므로 오류
            malformed.append((ln, line[:60]))
            continue
        seed = int(parts[2])
        if seed in unseen:
            violations.append((parts[0], seed))
    if malformed:
        print(f"[check] malformed manifest 행(scene 열 필수): {malformed[:3]}", file=sys.stderr)
        sys.exit(5)
    if violations:
        print(
            f"[EVAL-SEED 침범] fit manifest 가 eval unseen 예약 seed 를 사용: "
            f"{violations[:5]}{' ...' if len(violations) > 5 else ''}",
            file=sys.stderr,
        )
        sys.exit(5)
    print(f"[check] OK — fit({args.fit_manifest}) ∩ eval-unseen = ∅")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("plan", help="수집 계획 (S index × env/noise seed)")
    p.add_argument("--seeds-tsv", required=True)
    p.add_argument("--cell-id", required=True, help="pq3 cell id (예: pq3_drawer_left)")
    p.add_argument("--tsv-cell-index", type=int, required=True, help="seed 소스 tsv 의 cell_index")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-plan", type=int, default=60, help="계획 seed 수 (S1..S60)")
    p.set_defaults(fn=cmd_plan)

    f = sub.add_parser("freeze", help="fit 확정 + eval 30 예약·동결")
    f.add_argument("--seeds-tsv", required=True)
    f.add_argument("--cell-id", required=True)
    f.add_argument("--tsv-cell-index", type=int, required=True)
    f.add_argument("--collected-dir", required=True, help="fit 수집 결과 디렉토리 (succ 스템)")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--fit-expect", type=int, default=15)
    f.add_argument("--eval-seen", type=int, default=15)
    f.add_argument("--eval-unseen", type=int, default=15)
    f.add_argument("--pkl-prefix", default=None,
                   help="fit manifest 의 pkl 경로 루트 재작성 (승준 절대경로 — pkl 직송 후 "
                        "로컬 부재 대응). 미지정 시 로컬 collected-dir 경로")
    f.set_defaults(fn=cmd_freeze)

    o = sub.add_parser("pool", help="task-pooled fit manifest 합본")
    o.add_argument("--out-dir", required=True)
    o.add_argument("--cells", required=True, help="콤마 목록 (예: pq3_drawer_left,pq3_drawer_right)")
    o.add_argument("--task-manifest", required=True, help="출력 경로")
    o.set_defaults(fn=cmd_pool)

    c = sub.add_parser("check", help="fit manifest × eval 예약 교차 검증 (침범 시 exit 5)")
    c.add_argument("--fit-manifest", required=True)
    c.add_argument("--eval-reserved", required=True)
    c.set_defaults(fn=cmd_check)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
