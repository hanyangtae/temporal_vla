#!/usr/bin/env python3
"""exp2(구 pq2) P3 held-out 본실험 집계 v2.

설계 근거: docs/collab/2026-07-10-steering-redesign-gate1.md (D1~D4) +
~/.claude/plans/sleepy-giggling-newell.md 집계 절.

- primary 채점: apple = corrected 0.10 (live, 파일명 succ = fork 술어), bread = 원판정.
- secondary: apple 은 ever@0.07 병기 (sidecar dual_scoring.tsv), discordant_rate 는 진단 전용.
- Δ = arm − base (같은 cell, n=60), 위약 위치 = Δ_arm − Δ_null 도 병기.
- base: 신규 cell 은 p3/<cell>/ho_base, 재사용 cell 은 구 steer_eval/<cell>/ho_base
  (ppcs_apple 은 corrected_summary.json 의 corrected010 으로 override — 원판정과 병기).
- machine: work_queue_p3/ledger.tsv 조인. (dir, ep) 중복은 첫 행 유지·경고.
- 권고 기준(비구속, D4): primary contrast = gated_per_scene_fit15 vs base, δ=+7.5%p,
  replication = instruction 별 ≥3/4 scene ΔSR>0 ∧ pooled Δ>0.

사용: python scripts/safe/groot_n15/robocasa/steer/exp2/aggregate_v2.py [--out-dir ...]
srv50(구 w2/worker2, .50) arm 은 dual_scoring.tsv + ARM_SPEC.json 만 로컬 트리로 pull 해 두면 집계에 포함된다.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
G15 = os.path.join(REPO, "outputs/eval/robocasa/groot_n15")

APPLE_PREFIX = "ppcs_"  # corrected 채점 대상 (PickPlaceCounterToStove)
NULL_TAG = "ho_null_per_scene_fit15"
EXPECT_N = 60

TAG_RE = re.compile(
    r"^ho_(?P<diag>diag_(?:kstack_)?(?:L15_)?)?(?P<strategy>base|null|pos|perm|gated)"
    r"(?:_(?P<scope>per_scene|cross_scene|grand))?(?:_fit(?P<fitn>\d+))?(?P<suffix>_L15)?$"
)


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        return {"strategy": tag, "scope": "", "fitn": "", "diag": ""}
    d = m.groupdict()
    diag = ""
    if d["diag"] or d["suffix"]:
        parts = []
        if d["diag"] and "kstack" in d["diag"]:
            parts.append("kstack")
        if (d["diag"] and "L15" in d["diag"]) or d["suffix"]:
            parts.append("L15")
        diag = "+".join(parts) or "diag"
    return {
        "strategy": d["strategy"],
        "scope": d["scope"] or "",
        "fitn": d["fitn"] or "",
        "diag": diag,
    }


def read_sidecar(path):
    """dual_scoring.tsv → ep 기준 dedup 된 (succ, ever007, ever010) dict."""
    eps = {}
    dups = 0
    with open(path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            m = re.search(r"--ep(\d+)--succ([01])", row["pkl"])
            if not m:
                continue
            ep = int(m.group(1))
            if ep in eps:
                dups += 1
                continue

            def _b(v):
                return None if v in (None, "", "None") else v in ("True", "1", "1.0", "true")

            eps[ep] = {
                "succ": m.group(2) == "1",
                "ever007": _b(row.get("ever007")),
                "ever010": _b(row.get("ever010")),
            }
    return eps, dups


def count_from_stems(rollout_dir):
    """sidecar 부재 시 파일명 stem(pkl/zst/csv 마커 통합) 기준 성공 수."""
    eps = {}
    for p in glob.glob(os.path.join(rollout_dir, "*", "*", "task*--ep*--succ*")):
        m = re.search(r"--ep(\d+)--succ([01])", os.path.basename(p))
        if m:
            eps.setdefault(int(m.group(1)), m.group(2) == "1")
    return eps


def arm_counts(arm_dir, is_apple):
    """arm 디렉토리 → dict(n, primary, secondary007, discordant, mismatch)."""
    side = os.path.join(arm_dir, "dual_scoring.tsv")
    if os.path.isfile(side) and os.path.getsize(side) > 0:
        eps, dups = read_sidecar(side)
        n = len(eps)
        primary = sum(1 for v in eps.values() if v["succ"])
        e007 = [v["ever007"] for v in eps.values()]
        has007 = all(x is not None for x in e007) and n > 0
        secondary = sum(1 for x in e007 if x) if has007 else None
        disc = None
        mismatch = 0
        if is_apple and has007:
            disc = sum(1 for v in eps.values() if v["ever010"] and not v["ever007"])
            mismatch = sum(
                1 for v in eps.values() if v["ever010"] is not None and v["ever010"] != v["succ"]
            )
        return {"n": n, "primary": primary, "secondary007": secondary,
                "discordant": disc, "dups": dups, "mismatch": mismatch, "src": "sidecar"}
    rr = os.path.join(arm_dir, "raw_rollouts")
    if os.path.isdir(rr):
        eps = count_from_stems(rr)
        if eps:
            return {"n": len(eps), "primary": sum(1 for s in eps.values() if s),
                    "secondary007": None, "discordant": None, "dups": 0,
                    "mismatch": 0, "src": "stems"}
    return None


def load_ledger(path):
    machines = {}
    if not os.path.isfile(path):
        return machines
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            spec = dict(kv.split("=", 1) for kv in parts[2].split() if "=" in kv)
            if "CELL" in spec and "TAG" in spec:
                machines[(spec["CELL"], spec["TAG"])] = parts[1]
    return machines


def base_for_cell(cell, p3_root, legacy_root, rejudge):
    """(primary, secondary007, n, 출처) — 신규 cell 은 p3, 재사용 cell 은 구 ho_base."""
    is_apple = cell.startswith(APPLE_PREFIX)
    c = arm_counts(os.path.join(p3_root, cell, "ho_base"), is_apple)
    if c and c["n"] >= EXPECT_N:
        return c["primary"], c["secondary007"], c["n"], "p3"
    key = f"{cell}|ho_base"
    if key in rejudge:
        r = rejudge[key]
        return r["corrected010"], r["ever007"], r["n"], "legacy+rejudge"
    legacy = os.path.join(legacy_root, cell, "ho_base")
    if os.path.isdir(legacy):
        eps = count_from_stems(os.path.join(legacy, "raw_rollouts"))
        if not eps:  # 구 트리는 raw_rollouts 없이 바로 task 디렉토리인 경우도 있음
            eps = count_from_stems(legacy)
        if eps:
            return sum(1 for s in eps.values() if s), None, len(eps), "legacy"
    return None, None, 0, "missing"


def fmt_delta(d):
    return "" if d is None else f"{d:+d}"


def se_diff(s1, n1, s2, n2):
    if not n1 or not n2:
        return None
    p1, p2 = s1 / n1, s2 / n2
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3-root", default=os.path.join(G15, "steer_eval_exp2/p3"))
    ap.add_argument("--legacy-root", default=os.path.join(G15, "steer_eval"))
    ap.add_argument("--rejudge-json",
                    default=os.path.join(G15, "rejudge_matrix_apple/corrected_summary.json"))
    ap.add_argument("--ledger",
                    default=os.path.join(G15, "work_queue_p3/ledger.tsv"))
    ap.add_argument("--out-dir", default=os.path.join(G15, "steer_eval_exp2/aggregate_v2"))
    ap.add_argument("--expected-arms", type=int, default=None,
                    help="기대 arm 총수 (미달 시 경고, 기대개수 대조)")
    args = ap.parse_args()

    rejudge = {}
    if os.path.isfile(args.rejudge_json):
        rejudge = json.load(open(args.rejudge_json))
    machines = load_ledger(args.ledger)

    cells = sorted(
        d for d in os.listdir(args.p3_root)
        if os.path.isdir(os.path.join(args.p3_root, d)) and not d.startswith("_")
    )
    rows = []
    warnings = []
    bases = {}
    for cell in cells:
        is_apple = cell.startswith(APPLE_PREFIX)
        b_pri, b_sec, b_n, b_src = base_for_cell(cell, args.p3_root, args.legacy_root, rejudge)
        bases[cell] = {"primary": b_pri, "secondary007": b_sec, "n": b_n, "src": b_src}
        if b_pri is None:
            warnings.append(f"[base-missing] {cell}")
        null_c = arm_counts(os.path.join(args.p3_root, cell, NULL_TAG), is_apple)
        null_delta = (null_c["primary"] - b_pri) if (null_c and b_pri is not None) else None

        for tag_dir in sorted(glob.glob(os.path.join(args.p3_root, cell, "ho_*"))):
            tag = os.path.basename(tag_dir)
            c = arm_counts(tag_dir, is_apple)
            if c is None:
                continue
            if c["n"] < EXPECT_N:
                warnings.append(f"[incomplete] {cell}/{tag} n={c['n']}")
            if c["dups"]:
                warnings.append(f"[dup-ep] {cell}/{tag} dups={c['dups']}")
            if c["mismatch"]:
                warnings.append(f"[succ!=ever010] {cell}/{tag} mismatch={c['mismatch']}")
            meta = {}
            spec_p = os.path.join(tag_dir, "ARM_SPEC.json")
            if os.path.isfile(spec_p):
                try:
                    meta = json.load(open(spec_p))
                except (json.JSONDecodeError, OSError) as e:
                    warnings.append(f"[arm-spec-bad] {cell}/{tag}: {e}")
            t = parse_tag(tag)
            delta = c["primary"] - b_pri if (b_pri is not None and tag != "ho_base") else None
            vs_null = (delta - null_delta) if (delta is not None and null_delta is not None
                                               and tag not in ("ho_base", NULL_TAG)) else None
            se = se_diff(c["primary"], c["n"], b_pri, b_n) if b_pri is not None else None
            rows.append({
                "cell": cell, "tag": tag, "strategy": t["strategy"], "scope": t["scope"],
                "fitn": t["fitn"], "diag": t["diag"],
                "n": c["n"], "succ_primary": c["primary"],
                "sr_primary": round(c["primary"] / c["n"], 4) if c["n"] else None,
                "succ_ever007": c["secondary007"],
                "discordant": c["discordant"],
                "delta_vs_base": delta,
                "delta_vs_null": vs_null,
                "se_diff_pp": round(se * 100, 1) if se else None,
                "machine": machines.get((cell, tag), meta.get("machine", "")),
                "alpha": meta.get("alpha", ""), "beta": meta.get("beta", ""),
                "layers": meta.get("layers", ""), "count_src": c["src"],
            })

    if args.expected_arms is not None and len(rows) != args.expected_arms:
        warnings.append(f"[count] rows={len(rows)} != expected={args.expected_arms}")

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_p = os.path.join(args.out_dir, "arms.tsv")
    cols = list(rows[0].keys()) if rows else []
    with open(tsv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    # cell × arm 매트릭스 (bread/apple 분리)
    md = ["# exp2 P3 집계 (aggregate_v2)", ""]
    for group, label in ((lambda c: not c.startswith(APPLE_PREFIX), "bread (원판정 채점)"),
                         (lambda c: c.startswith(APPLE_PREFIX), "apple (corrected 0.10 채점)")):
        gcells = [c for c in cells if group(c)]
        if not gcells:
            continue
        md.append(f"## {label}")
        md.append("")
        base_line = " · ".join(
            f"{c} base={bases[c]['primary']}/{bases[c]['n']}({bases[c]['src']})" for c in gcells
        )
        md.append(f"base: {base_line}")
        md.append("")
        tags = sorted({r["tag"] for r in rows if r["cell"] in gcells and r["tag"] != "ho_base"})
        md.append("| arm | " + " | ".join(gcells) + " |")
        md.append("|---|" + "---|" * len(gcells))
        for tag in tags:
            line = [tag.replace("ho_", "")]
            for c in gcells:
                r = next((x for x in rows if x["cell"] == c and x["tag"] == tag), None)
                if r is None:
                    line.append("")
                elif r["n"] < EXPECT_N:
                    line.append(f"({r['succ_primary']}/{r['n']} 진행중)")
                else:
                    cellv = f"{r['succ_primary']} ({fmt_delta(r['delta_vs_base'])})"
                    if r["succ_ever007"] is not None and r["succ_ever007"] != r["succ_primary"]:
                        cellv += f" [0.07:{r['succ_ever007']}]"
                    line.append(cellv)
            md.append("| " + " | ".join(line) + " |")
        md.append("")

    # 권고 기준(비구속) — gated_per_scene_fit15 vs base, δ=+7.5%p
    md.append("## 권고 기준 판정 (비구속, Gate1 D4)")
    md.append("")
    md.append("primary contrast = gated_per_scene_fit15 vs base, δ=+7.5%p (n=60 → +4.5판)")
    md.append("")
    prim = [r for r in rows if r["tag"] == "ho_gated_per_scene_fit15" and r["n"] >= EXPECT_N]
    pooled_d, pooled_n = 0, 0
    for r in prim:
        b = bases[r["cell"]]
        if b["primary"] is None:
            continue
        pooled_d += r["succ_primary"] - b["primary"]
        pooled_n += 1
        md.append(f"- {r['cell']}: Δ={fmt_delta(r['delta_vs_base'])} "
                  f"(위약대비 {fmt_delta(r['delta_vs_null'])}, SE≈{r['se_diff_pp']}%p)")
    if pooled_n:
        md.append(f"- pooled Δ = {pooled_d:+d}판 / {pooled_n} cell "
                  f"(δ 충족엔 cell 당 평균 +4.5판 필요)")
    md.append("")
    if warnings:
        md.append("## 경고")
        md.extend(f"- {w}" for w in warnings)

    md_p = os.path.join(args.out_dir, "matrix.md")
    with open(md_p, "w") as f:
        f.write("\n".join(md) + "\n")
    json.dump({"bases": bases, "rows": rows, "warnings": warnings},
              open(os.path.join(args.out_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[aggregate_v2] arms={len(rows)} warnings={len(warnings)} -> {args.out_dir}")
    for w in warnings[:20]:
        print("  " + w)


if __name__ == "__main__":
    sys.exit(main())
