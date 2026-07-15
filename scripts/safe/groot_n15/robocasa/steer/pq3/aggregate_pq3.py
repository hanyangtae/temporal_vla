#!/usr/bin/env python3
"""pq3 eval 집계·판정 (계획서 v9 §F) — EXPECT_N=30, seen/unseen 분해, Holm 6 primary.

판정 규칙은 pq3_decision.py (동결 모듈 — sha256 을 summary 에 기록) 단일 출처.
입력은 캡처-OFF 사이드카 json 스템 (task*--ep*--succ{0|1}.json) + ARM_SPEC + manifest.

출력 (--out):
  arms.tsv        cell×arm 승수 (pooled/seen/unseen) + machine
  hypotheses.json 6 primary p/p_adj/reject + null 관문 + 비재현 CI + FWER sim
  summary.json    메타(결정 모듈 sha, expect, host 균형 표)
  matrix.md       사람용 표

주의: 판정은 pooled 단독. seen/unseen 은 방향 읽기용 병기. 보고 전 confound-audit 필수.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pq3_decision as D  # noqa: E402
from make_pq3_manifests import STEM_RE  # noqa: E402

EXPECT_N = 30
ARM_TAGS = ["ho_base", "ho_coast_cross_scene", "ho_gated_cross_scene", "ho_null_cross_scene"]


def read_arm(eval_root: Path, cell: str, task: str, arm: str) -> tuple[dict[int, dict], list[str]]:
    """arm 사이드카 로드 → ({ep: {succ, env_seed, noise_seed, run_tag}}, problems).

    filename 만 믿지 않고 json 본문을 읽어 재라벨링·중복을 탐지 (Gate 2 치명#4):
    같은 ep 파일 2개(succ0/succ1 공존 포함) = 오염, 본문 seed 는 manifest 와 대조.
    """
    d = eval_root / cell / arm / "raw_rollouts" / task / cell
    out: dict[int, dict] = {}
    problems: list[str] = []
    for p in sorted(d.glob("task*--ep*--succ*.json")):
        m = STEM_RE.match(p.name)
        if not m:
            continue
        ep = int(m.group("ep"))
        if ep in out:
            problems.append(f"{cell}/{arm}: ep{ep} 파일 중복 ({p.name})")
            continue
        try:
            body = json.loads(p.read_text())
        except Exception as exc:
            problems.append(f"{cell}/{arm}: {p.name} json 파싱 실패 ({exc})")
            continue
        if int(body.get("episode_success", -1)) != int(m.group("succ")):
            problems.append(f"{cell}/{arm}: {p.name} filename succ != 본문 episode_success")
        out[ep] = {
            "succ": int(m.group("succ")),
            "env_seed": body.get("scenario_seed"),
            "noise_seed": body.get("inference_seed"),
            "run_tag": body.get("run_tag"),
        }
    return out, problems


def read_eval_manifest(manifest_dir: Path, cell: str) -> dict[int, dict]:
    path = manifest_dir / cell / "eval_manifest.tsv"
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("episode_idx"):
            continue
        ep, env_seed, noise, split = line.split("\t")
        rows[int(ep)] = {"env_seed": int(env_seed), "noise_seed": int(noise), "split": split}
    return rows


def read_arm_spec(eval_root: Path, cell: str, arm: str) -> tuple[dict, str]:
    p = eval_root / cell / arm / "ARM_SPEC.json"
    if not p.exists():
        return {}, ""
    txt = p.read_text()
    return json.loads(txt), txt


def recompute_spec_sha(raw_text: str) -> str | None:
    """ARM_SPEC 원문에서 core 원시 문자열을 추출해 spec_sha 를 재계산 (위조 탐지).

    러너가 `..,"core":<CORE_SPEC>}\n` 형태로 기록 — core 부분 그대로 sha256[:12].
    """
    marker = '"core":'
    idx = raw_text.find(marker)
    if idx < 0:
        return None
    raw_core = raw_text[idx + len(marker):].rstrip()
    if not raw_core.endswith("}"):
        return None
    raw_core = raw_core[:-1]  # 바깥 닫는 중괄호 제거 → core 자체 json
    return hashlib.sha256(raw_core.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True, help="steer_eval_pq3/e1")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--arm-config", required=True, help="build_pq3_queue 와 동일 json (cells)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-n", type=int, default=EXPECT_N)
    ap.add_argument("--fwer-sims", type=int, default=10000)
    ap.add_argument("--gated-gate-report", required=True,
                    help="pq3_gated_gate.py --enforce 산출 json — gated N/A 를 여기서 "
                         "자동 파생 (수동 override 금지, Gate2 R3 높음#3)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="미완주 arm 허용 (중간 점검용 — 판정 산출은 완주 시에만)")
    args = ap.parse_args()
    gate_report = json.loads(Path(args.gated_gate_report).read_text())
    bad_enforce = [t for t, v in gate_report.items() if not v.get("enforce")]
    if bad_enforce:
        raise SystemExit(f"gate report 가 enforce 산출이 아님: {bad_enforce}")
    gated_na = tuple(sorted({
        ("drawer" if "Drawer" in task else "ppcc")
        for task, v in gate_report.items() if not v.get("pass")
    }))

    eval_root = Path(args.eval_root)
    manifest_dir = Path(args.manifest_dir)
    cells: dict[str, str] = json.loads(Path(args.arm_config).read_text())["cells"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_sha = hashlib.sha256(
        (Path(__file__).parent / "pq3_decision.py").read_bytes()
    ).hexdigest()[:16]

    # ── 수집: cell×arm×ep outcome + split + 무결성 검증 (Gate 2 치명#4) ──────
    outcomes: dict[tuple[str, str], dict[int, int]] = {}
    splits: dict[str, dict[int, dict]] = {}
    machines: dict[tuple[str, str], str] = {}
    incomplete, integrity = [], []
    for cell, task in sorted(cells.items()):
        splits[cell] = read_eval_manifest(manifest_dir, cell)
        task_key = "drawer" if "Drawer" in task else "ppcc"
        for arm in ARM_TAGS:
            if arm == "ho_gated_cross_scene" and task_key in gated_na:
                continue  # gated 성립 게이트 미통과 — H2/H3 N/A
            rows, problems = read_arm(eval_root, cell, task, arm)
            integrity.extend(problems)
            spec, spec_raw = read_arm_spec(eval_root, cell, arm)
            machines[(cell, arm)] = spec.get("machine", "?")
            # fail-closed: ARM_SPEC 부재/spec_sha 부재/재계산 불일치 전부 위반 (R2 높음#4)
            if not spec or not spec.get("spec_sha"):
                integrity.append(f"{cell}/{arm}: ARM_SPEC 부재 또는 spec_sha 없음")
                expect_tag = None
            else:
                recomputed = recompute_spec_sha(spec_raw)
                if recomputed != spec["spec_sha"]:
                    integrity.append(
                        f"{cell}/{arm}: spec_sha 재계산 불일치 ({recomputed} != {spec['spec_sha']})"
                    )
                expect_tag = f"{arm}:{spec['spec_sha']}"
            for ep, row in rows.items():
                man = splits[cell].get(ep)
                if man is None:
                    integrity.append(f"{cell}/{arm}: ep{ep} 가 eval manifest 에 없음")
                    continue
                # 필수 필드 부재 = 위반 (None 허용은 대조 우회 경로 — R2 높음#4)
                for field in ("env_seed", "noise_seed", "run_tag"):
                    if row[field] is None:
                        integrity.append(f"{cell}/{arm}: ep{ep} 사이드카 {field} 부재")
                if row["env_seed"] is not None and int(row["env_seed"]) != man["env_seed"]:
                    integrity.append(f"{cell}/{arm}: ep{ep} env_seed {row['env_seed']} != manifest {man['env_seed']}")
                if row["noise_seed"] is not None and int(row["noise_seed"]) != man["noise_seed"]:
                    integrity.append(f"{cell}/{arm}: ep{ep} noise_seed {row['noise_seed']} != manifest {man['noise_seed']}")
                if expect_tag and row["run_tag"] is not None and row["run_tag"] != expect_tag:
                    integrity.append(f"{cell}/{arm}: ep{ep} run_tag {row['run_tag']} != ARM_SPEC {expect_tag}")
            got = {ep: row["succ"] for ep, row in rows.items()}
            expect_eps = set(splits[cell])
            if set(got) != expect_eps or len(got) != args.expect_n:
                incomplete.append((cell, arm, len(got)))
            outcomes[(cell, arm)] = got
    if integrity:
        for line in integrity[:20]:
            print(f"[integrity] {line}", file=sys.stderr)
        raise SystemExit(f"사이드카 무결성 위반 {len(integrity)}건 — 판정 불가 (오염 조사 필요)")
    if incomplete and not args.allow_incomplete:
        for cell, arm, n in incomplete:
            print(f"[incomplete] {cell}/{arm}: {n}/{args.expect_n}", file=sys.stderr)
        raise SystemExit(f"미완주 {len(incomplete)} arm — 판정 불가 (--allow-incomplete 로 중간 점검만 가능)")

    # ── arms.tsv ────────────────────────────────────────────────────────────
    lines = ["cell\tarm\twins\twins_seen\twins_unseen\tn\tmachine"]
    for (cell, arm), got in sorted(outcomes.items()):
        seen = sum(v for e, v in got.items() if splits[cell].get(e, {}).get("split") == "seen")
        unseen = sum(v for e, v in got.items() if splits[cell].get(e, {}).get("split") == "unseen")
        lines.append(f"{cell}\t{arm}\t{sum(got.values())}\t{seen}\t{unseen}\t{len(got)}\t{machines[(cell, arm)]}")
    (out_dir / "arms.tsv").write_text("\n".join(lines) + "\n")

    # ── task pool + 가설 검정 (동결 모듈) ───────────────────────────────────
    tasks: dict[str, list[str]] = {}
    for cell, task in cells.items():
        key = "drawer" if "Drawer" in task else "ppcc"
        tasks.setdefault(key, []).append(cell)

    def pool(task_key: str, arm: str) -> dict[str, int]:
        """(cell, ep) → outcome, 키 'cell:ep' 로 arm 간 pairing."""
        out = {}
        for cell in tasks[task_key]:
            for e, v in outcomes[(cell, arm)].items():
                out[f"{cell}:{e}"] = v
        return out

    pvals, detail = {}, {}
    episode_triples = {}
    for tk in sorted(tasks):
        base = pool(tk, D.BASE_ARM)
        perm = pool(tk, "ho_coast_cross_scene")
        gated = pool(tk, "ho_gated_cross_scene") if tk not in gated_na else None
        null = pool(tk, D.NULL_ARM)
        keys = sorted(base)
        episode_triples[tk] = [
            (base[k], perm[k], gated[k] if gated is not None else 0) for k in keys
        ]
        n_expected = args.expect_n * len(tasks[tk])
        if len(keys) != n_expected:
            raise SystemExit(f"task {tk} pooled n={len(keys)} != 기대 {n_expected}")
        arm_of = {"ho_base": base, "ho_coast_cross_scene": perm, "ho_gated_cross_scene": gated}
        for h, (a, b_arm) in D.HYPOTHESES.items():
            if tk in gated_na and h in ("H2", "H3"):
                continue  # gated 성립 게이트 미통과 task — H2/H3 N/A (사전 등록 규칙)
            av, bv = arm_of[a], arm_of[b_arm]
            b = sum(1 for k in keys if av[k] == 1 and bv[k] == 0)
            c = sum(1 for k in keys if av[k] == 0 and bv[k] == 1)
            p = D.exact_mcnemar_one_sided(b, c)
            n = len(keys)
            pair_diffs = [av[k] - bv[k] for k in keys]
            def _split_delta(split):
                ks = [k for k in keys
                      if splits[k.split(":")[0]][int(k.split(":")[1])]["split"] == split]
                return (sum(av[k] for k in ks) - sum(bv[k] for k in ks), len(ks))
            detail[f"{h}:{tk}"] = {
                "A": a, "B": b_arm, "n": n, "b": b, "c": c, "p": p,
                "delta_games": sum(av.values()) - sum(bv.values()),
                "delta_sr": (sum(av.values()) - sum(bv.values())) / n,
                # 판정용 CI = bootstrap (동결 방법), 정규 근사는 참고 병기
                "ci95_upper_bootstrap": D.paired_delta_ci_upper_bootstrap(pair_diffs),
                "ci95_upper_normal_ref": D.paired_delta_ci_upper_one_sided(b, c, n),
                "seen_delta": _split_delta("seen"), "unseen_delta": _split_delta("unseen"),
            }
            pvals[f"{h}:{tk}"] = p
        null_delta = sum(null.values()) - sum(base.values())
        detail[f"null_gate:{tk}"] = {
            "null_minus_base_games": null_delta,
            "margin": D.NULL_GATE_MARGIN_GAMES,
            "h1_interpretable": abs(null_delta) <= D.NULL_GATE_MARGIN_GAMES,
        }

    holm_res = D.holm(pvals)
    for k, v in holm_res.items():
        detail[k].update(v)

    # 최종 상태 (null 관문으로 H1 mask + 비재현 CI 규칙 — 동결 모듈이 산출, 수동 해석 배제)
    status = D.final_status(
        holm_results={k: detail[k] for k in pvals},
        null_gates={tk: detail[f"null_gate:{tk}"]["h1_interpretable"] for tk in tasks},
        ci_uppers={f"H1:{tk}": detail[f"H1:{tk}"]["ci95_upper_bootstrap"] for tk in tasks},
        gated_na_tasks=gated_na,
    )

    fwer = D.fwer_sim(episode_triples, n_sim=args.fwer_sims, gated_na_tasks=gated_na)

    # ── host 검사 (Gate 2 중간#2): cell 블록 설계 — 한 cell 의 전 arm 이 같은 host
    # class 여야 paired contrast 에서 host 가 상쇄된다. 위반은 경고 목록으로 기록.
    host_table: dict[str, dict[str, int]] = {}
    for (cell, arm), m in machines.items():
        cls = m.split("-")[0]
        host_table.setdefault(arm, {}).setdefault(cls, 0)
        host_table[arm][cls] += 1
    block_violations = []
    for cell in cells:
        # 실제 존재하는 arm 만 조회 (gated N/A task 의 KeyError 방지 — R2 높음#5)
        classes = {m.split("-")[0] for (c, _a), m in machines.items() if c == cell}
        if len(classes) > 1:
            block_violations.append({"cell": cell, "classes": sorted(classes)})

    hyp_out = {"pvals": detail, "holm_alpha": D.ALPHA, "fwer_sim": fwer,
               "fwer_sims": args.fwer_sims, "decision_sha": decision_sha,
               "final_status": status}
    (out_dir / "hypotheses.json").write_text(json.dumps(hyp_out, indent=2, ensure_ascii=False))

    summary = {"cells": cells, "expect_n": args.expect_n, "decision_sha": decision_sha,
               "incomplete": incomplete, "host_table": host_table,
               "host_block_violations": block_violations, "fwer_sim": fwer,
               "final_status": status["overall"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = ["# pq3 집계 (판정=pooled McNemar+Holm 단독, seen/unseen 은 방향 읽기용)", ""]
    md.append("| 가설:task | A−B(판) | ΔSR | b/c | p | p_adj | 기각 | seen Δ | unseen Δ |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(pvals):
        d = detail[k]
        md.append(
            f"| {k} | {d['delta_games']:+d} | {d['delta_sr']:+.3f} | {d['b']}/{d['c']} "
            f"| {d['p']:.4f} | {d['p_adj']:.4f} | {'✔' if d['reject'] else '—'} "
            f"| {d['seen_delta'][0]:+d}/{d['seen_delta'][1]} | {d['unseen_delta'][0]:+d}/{d['unseen_delta'][1]} |"
        )
    md.append("")
    for tk in sorted(tasks):
        g = detail[f"null_gate:{tk}"]
        md.append(f"- null 관문[{tk}]: Δ={g['null_minus_base_games']:+d}판 "
                  f"(margin ±{g['margin']}) → H1 해석 {'유효' if g['h1_interpretable'] else '무효'}")
    md.append(f"- global-null FWER sim: {fwer:.4f} (목표 ≤ {D.ALPHA})")
    md.append(f"- **최종 상태: {status['overall']}** (동결 decision 모듈 산출)")
    if block_violations:
        md.append(f"- ⚠ host cell-블록 위반: {block_violations}")
    md.append(f"- decision module sha: {decision_sha}")
    (out_dir / "matrix.md").write_text("\n".join(md) + "\n")
    print(f"[aggregate-pq3] -> {out_dir} (status={status['overall']}, "
          f"fwer_sim={fwer:.4f}, decision_sha={decision_sha})")


if __name__ == "__main__":
    main()
