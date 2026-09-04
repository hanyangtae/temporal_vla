#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 LOKO rescue eval 집계 (stdlib only, py3.10).

입력
  1) results tree: <root>/<arm>/<slug>_s<scene>_j<jitter>/<slug>/<runner_arm>/per_episode.tsv
     (+ 같은 디렉터리의 raw_rollouts/**/task*--ep<EP>--succ*.json 사이드카)
     --results-root 는 여러 번 줄 수 있음 (원격 트리를 로컬로 rsync 한 경로 포함).
  2) eval cell 표 (구 키): --cells (기본 .claude/jobs/43998117/tmp/cells_eval.tsv)
  3) 구→신 키 매핑: --key-map
  4) detector 요약 (신 키): --detector
  5) 연산자 metadata (신 키): --op-root / --op-jfair / --op-plain

출력 (--out-dir): cells.tsv, arms.tsv, episodes.tsv, summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import OrderedDict, defaultdict

# ---------------------------------------------------------------- 기본 경로
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

DEF_RESULTS = os.path.join(
    REPO, "outputs/eval/robocasa/groot_n15/og_v6_expand")
DEF_CELLS = "/home/dongkyu/.claude/jobs/43998117/tmp/cells_eval.tsv"
DEF_DETECTOR = os.path.join(
    REPO, "outputs/analysis/grid_phase/detector_v6/cells_summary.tsv")
DEF_OP_ROOT = os.path.join(REPO, "outputs/steer/online_pipe_v4_pilot")
DEF_OUT = os.path.join(REPO, "outputs/analysis/v6_expand_agg")

# archive rebase 로 좌/우가 뒤집힌 키 (구 → 신)
DEF_KEY_MAP = ("OvenRack/out-left=OvenRack/out-right,"
               "OvenRack/out-right=OvenRack/out-left,"
               "DishwasherRack/out-left=DishwasherRack/out-right,"
               "DishwasherRack/out-right=DishwasherRack/out-left")

# arm 정렬 우선순위 (디렉터리명에서 'ps_' 제거한 짧은 이름)
ARM_ORDER = ["base", "setm_gt_b06", "setm_gt_b07", "setm_gt_b08",
             "setm_gt_b09", "setm_gt_b10", "setm_gtplain_b08", "reseed"]

NA = "NA"


# ---------------------------------------------------------------- 유틸
def read_tsv(path, comment="#"):
    """헤더 있는 tsv → list[dict]. '#' 로 시작하는 줄은 주석."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if comment and line.startswith(comment):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(dict(zip(header, parts[:len(header)])))
    return rows


def to_int(s, default=None):
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return default


def to_float(s, default=None):
    try:
        v = float(str(s).strip())
    except (TypeError, ValueError):
        return default
    return v


def fmt(v, nd=4):
    if v is None:
        return NA
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") or "0"
    return str(v)


def kn(k, n):
    """'k/n (0.xx)' 표기."""
    if not n:
        return f"{k}/0 (--)"
    return f"{k}/{n} ({k / n:.2f})"


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def arm_short(dirname):
    return dirname[3:] if dirname.startswith("ps_") else dirname


def arm_sort_key(short):
    return (ARM_ORDER.index(short) if short in ARM_ORDER else 99, short)


# ---------------------------------------------------------------- 키 매핑
class KeyMap:
    def __init__(self, spec):
        self.instr = {}
        for item in (spec or "").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"--key-map 형식 오류: {item!r} (old=new)")
            old, new = item.split("=", 1)
            self.instr[old.strip()] = new.strip()
        self.slug = {o.replace("/", "_"): n.replace("/", "_")
                     for o, n in self.instr.items()}

    def new_instr(self, old):
        return self.instr.get(old, old)

    def new_slug(self, old):
        return self.slug.get(old, old)


# ---------------------------------------------------------------- 입력 로딩
def load_cells(path, kmap):
    """(slug_old, scene, jitter) → cell 메타 + 기대 에피소드 수."""
    rows = read_tsv(path)
    if not rows:
        raise SystemExit(f"cell 표를 읽지 못함: {path}")
    cells = OrderedDict()
    for r in rows:
        slug_old = r.get("slug", "")
        scene = to_int(r.get("scene_idx"))
        jit = to_int(r.get("jitter_idx"))
        if not slug_old or scene is None or jit is None:
            continue
        key = (slug_old, scene, jit)
        c = cells.get(key)
        if c is None:
            c = {
                "slug_old": slug_old,
                "instr_old": r.get("grid_instruction", ""),
                "slug_new": kmap.new_slug(slug_old),
                "instr_new": kmap.new_instr(r.get("grid_instruction", "")),
                "scene": scene,
                "jitter": jit,
                "machine": r.get("machine", ""),
                "jitter_reset_idx": r.get("jitter_reset_idx", ""),
                "n_eval_eps": 0,
                "env_seeds": set(),
            }
            cells[key] = c
        c["n_eval_eps"] += 1
        if r.get("env_seed"):
            c["env_seeds"].add(r["env_seed"])
    return cells


def load_detector(path):
    """(instruction_new, scene, jitter) → detector 필드."""
    out = {}
    for r in read_tsv(path):
        instr = r.get("instruction", "")
        scene = to_int(r.get("scene"))
        jit = to_int(r.get("jitter"))
        if not instr or scene is None or jit is None:
            continue
        out[(instr, scene, jit)] = {
            "registered": to_int(r.get("registered")),
            "td10_holdout": to_float(r.get("td10_holdout")),
            "fire_p50": to_float(r.get("fire_p50")),
            "fpr_target_succ": to_float(r.get("fpr_target_succ")),
            "n_succ_calib": to_int(r.get("n_succ_calib")),
        }
    return out


def _op_cell_dir(op_root, variant, slug_new, scene, jitter):
    return os.path.join(op_root, variant, slug_new,
                        f"s{scene}", f"j{jitter}")


def load_op_meta(op_root, variant, slug_new, scene, jitter):
    """<variant>/<slug>/s<i>/j<r>/<phase>/dit_L12/metadata.json 전부 읽기."""
    base = _op_cell_dir(op_root, variant, slug_new, scene, jitter)
    metas = []
    if not os.path.isdir(base):
        return metas
    for phase in sorted(os.listdir(base)):
        pdir = os.path.join(base, phase)
        if not os.path.isdir(pdir):
            continue
        for layer in sorted(os.listdir(pdir)):
            mpath = os.path.join(pdir, layer, "metadata.json")
            if os.path.exists(mpath):
                try:
                    with open(mpath, "r", encoding="utf-8") as fh:
                        d = json.load(fh)
                except (OSError, ValueError):
                    continue
                d["_phase"] = phase
                metas.append(d)
    return metas


def _first_key(d, names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


COS_KEYS = ("cos_jfair_vs_pool", "cos_vs_pool", "cos_jfair", "cosine")
MINFAIL_KEYS = ("min_fail_ep_in_mixed", "min_fail_ep", "min_fail_eps_in_mixed")


def op_fields(op_root, jfair_var, plain_var, slug_new, scene, jitter):
    mj = load_op_meta(op_root, jfair_var, slug_new, scene, jitter)
    mp = load_op_meta(op_root, plain_var, slug_new, scene, jitter)
    cos = [to_float(_first_key(d, COS_KEYS)) for d in mj]
    cos = [c for c in cos if c is not None]
    minf = [to_int(_first_key(d, MINFAIL_KEYS)) for d in mj]
    minf = [m for m in minf if m is not None]
    return {
        "n_phase_jfair": len(mj),
        "n_phase_plain": len(mp),
        "phases_jfair": ",".join(d["_phase"] for d in mj) or NA,
        "cos_median": med(cos),
        "cos_min": min(cos) if cos else None,
        "min_fail_ep_in_mixed": min(minf) if minf else None,
    }


# ---------------------------------------------------------------- 사이드카
SIDECAR_FIELDS = [
    "sc_fire_count", "sc_applied_count", "sc_fallback_mode",
    "sc_gate_fallback_n", "sc_gate_fallback_top", "sc_gate_skipped_n",
    "sc_llr_fallback_n", "sc_cand_log_n", "sc_rerun_ms_n", "sc_rerun_ms_med",
]


def _count_nonnull(seq):
    if not isinstance(seq, list):
        return None
    return sum(1 for x in seq if x is not None and x is not False and x != "")


def read_sidecar(run_dir, ep):
    """raw_rollouts 아래 task*--ep<EP>--succ*.json 을 찾아 요약 필드 추출."""
    raw = os.path.join(run_dir, "raw_rollouts")
    if not os.path.isdir(raw):
        return None
    target = f"--ep{ep}--"
    hit = None
    for dirpath, _dirnames, filenames in os.walk(raw):
        for fn in filenames:
            if fn.endswith(".json") and target in fn:
                hit = os.path.join(dirpath, fn)
                break
        if hit:
            break
    if hit is None:
        return None
    try:
        with open(hit, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None

    gf = d.get("gate_fallback")
    gf_vals = [x for x in gf if x] if isinstance(gf, list) else []
    top = NA
    if gf_vals:
        cnt = defaultdict(int)
        for v in gf_vals:
            cnt[str(v)] += 1
        top = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    rerun = d.get("rerun_ms")
    rerun_vals = [x for x in rerun if isinstance(x, (int, float))] \
        if isinstance(rerun, list) else []
    return {
        "sc_fire_count": d.get("perstep_fire_count"),
        "sc_applied_count": _first_key(
            d, ("perstep_applied_count", "applied_count")),
        "sc_fallback_mode": _first_key(
            d, ("perstep_fallback_mode", "perstep_fallback")),
        "sc_gate_fallback_n": len(gf_vals) if isinstance(gf, list) else None,
        "sc_gate_fallback_top": top,
        "sc_gate_skipped_n": _count_nonnull(
            _first_key(d, ("perstep_gate_skipped", "gate_skipped"))),
        "sc_llr_fallback_n": _count_nonnull(
            _first_key(d, ("llr_fallback", "perstep_llr_fallback"))),
        "sc_cand_log_n": _count_nonnull(d.get("cand_logs")),
        "sc_rerun_ms_n": len(rerun_vals) if isinstance(rerun, list) else None,
        "sc_rerun_ms_med": med(rerun_vals),
    }


# ---------------------------------------------------------------- 결과 스캔
EP_COLS = ["ep", "scene_idx", "env_seed", "inference_seed", "success", "steps",
           "n_inferences", "trigger_step", "phase_at_trigger", "n_gated",
           "gated_mode", "arm", "slug", "beta", "op", "noise_idx",
           "collection_success", "cell_key", "jitter_reset_idx", "jitter_idx"]


def parse_cell_dirname(name):
    """'<slug>_s<scene>_j<jitter>' → (slug, scene, jitter) 또는 None."""
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        return None
    slug, sp, jp = parts
    if not (sp.startswith("s") and jp.startswith("j")):
        return None
    scene, jit = to_int(sp[1:]), to_int(jp[1:])
    if scene is None or jit is None:
        return None
    return slug, scene, jit


def scan_results(roots, limit_arms, with_sidecar=True):
    """→ {(arm_short, slug_old, scene, jitter): [episode dict, ...]}"""
    out = defaultdict(list)
    seen = set()
    dupes = 0
    for root in roots:
        if not os.path.isdir(root):
            print(f"[warn] results-root 없음: {root}", file=sys.stderr)
            continue
        for arm_dir in sorted(os.listdir(root)):
            adir = os.path.join(root, arm_dir)
            if not os.path.isdir(adir) or arm_dir == "logs":
                continue
            short = arm_short(arm_dir)
            if limit_arms and short not in limit_arms:
                continue
            for cell_dir in sorted(os.listdir(adir)):
                parsed = parse_cell_dirname(cell_dir)
                if parsed is None:
                    continue
                slug, scene, jit = parsed
                sdir = os.path.join(adir, cell_dir, slug)
                if not os.path.isdir(sdir):
                    continue
                for runner in sorted(os.listdir(sdir)):
                    rdir = os.path.join(sdir, runner)
                    tsv = os.path.join(rdir, "per_episode.tsv")
                    if not os.path.exists(tsv):
                        continue
                    for r in read_tsv(tsv, comment=None):
                        ep = r.get("ep", "")
                        sig = (short, slug, scene, jit, ep,
                               r.get("env_seed", ""),
                               r.get("inference_seed", ""))
                        if sig in seen:
                            dupes += 1
                            continue
                        seen.add(sig)
                        rec = {c: r.get(c, "") for c in EP_COLS}
                        rec["runner_arm"] = runner
                        rec["arm"] = short
                        rec["src_root"] = root
                        rec["_run_dir"] = rdir
                        rec["success_i"] = to_int(r.get("success"), 0) or 0
                        if with_sidecar:
                            sc = read_sidecar(rdir, ep)
                            for f in SIDECAR_FIELDS:
                                rec[f] = (sc or {}).get(f)
                        out[(short, slug, scene, jit)].append(rec)
    if dupes:
        print(f"[info] 중복 에피소드 {dupes}건 제거 (다중 results-root)",
              file=sys.stderr)
    return out


# ---------------------------------------------------------------- 집계
def regime_of(det):
    if not det:
        return NA
    p50, fpr = det.get("fire_p50"), det.get("fpr_target_succ")
    if p50 is None and fpr is None:
        return NA
    if (p50 is not None and p50 <= 10) or (fpr is not None and fpr >= 0.75):
        return "초기조건형"
    return "실행표류형"


def build(args):
    kmap = KeyMap(args.key_map)
    cells = load_cells(args.cells, kmap)
    detector = load_detector(args.detector)
    limit = set(a.strip() for a in args.limit_arms.split(",") if a.strip()) \
        if args.limit_arms else None
    res = scan_results(args.results_root, limit,
                       with_sidecar=not args.no_sidecar)

    arms = sorted({k[0] for k in res}, key=arm_sort_key)

    # 결과에는 있는데 cell 표에 없는 셀 (기대치 미상)
    orphans = sorted({(k[1], k[2], k[3]) for k in res if
                      (k[1], k[2], k[3]) not in cells})

    rows = []
    for key, c in cells.items():
        slug_old, scene, jit = key
        det = detector.get((c["instr_new"], scene, jit))
        opf = op_fields(args.op_root, args.op_jfair, args.op_plain,
                        c["slug_new"], scene, jit)
        row = dict(c)
        row.pop("env_seeds", None)
        row["regime"] = regime_of(det)
        row["small_n"] = 1 if c["n_eval_eps"] <= args.small_n else 0
        row["det_registered"] = det.get("registered") if det else None
        row["td10_holdout"] = det.get("td10_holdout") if det else None
        row["fire_p50"] = det.get("fire_p50") if det else None
        row["fpr_target_succ"] = det.get("fpr_target_succ") if det else None
        row["n_succ_calib"] = det.get("n_succ_calib") if det else None
        row.update(opf)
        row["_arms"] = {}
        for a in arms:
            eps = res.get((a, slug_old, scene, jit), [])
            n = len(eps)
            resc = sum(e["success_i"] for e in eps)
            complete = 1 if (n and n == c["n_eval_eps"]) else 0
            row["_arms"][a] = {"n": n, "rescued": resc, "complete": complete,
                               "eps": eps}
        rows.append(row)

    if args.strict:
        bad = [r for r in rows
               if any(v["n"] and not v["complete"] for v in r["_arms"].values())]
        if bad:
            for r in bad:
                inc = [f"{a}({v['n']}/{r['n_eval_eps']})"
                       for a, v in r["_arms"].items() if v["n"] and not v["complete"]]
                print(f"[strict] 미완료 {r['instr_new']} s{r['scene']} "
                      f"j{r['jitter']}: {', '.join(inc)}", file=sys.stderr)
            raise SystemExit("--strict: 미완료 셀 존재")

    return rows, arms, orphans, cells


# ---------------------------------------------------------------- 출력
def write_cells(path, rows, arms):
    hdr = ["instruction_new", "old_instruction", "slug_new", "slug_old",
           "scene", "jitter", "machine", "jitter_reset_idx",
           "n_eval_eps", "small_n"]
    for a in arms:
        hdr += [f"n_{a}", f"rescued_{a}", f"complete_{a}"]
    hdr += ["regime", "det_registered", "td10_holdout", "fire_p50",
            "fpr_target_succ", "n_succ_calib",
            "n_phase_jfair", "n_phase_plain", "phases_jfair",
            "cos_median", "cos_min", "min_fail_ep_in_mixed"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            vals = [r["instr_new"], r["instr_old"], r["slug_new"],
                    r["slug_old"], r["scene"], r["jitter"], r["machine"],
                    r["jitter_reset_idx"], r["n_eval_eps"], r["small_n"]]
            for a in arms:
                v = r["_arms"][a]
                vals += [v["n"], v["rescued"], v["complete"]]
            vals += [r["regime"], r["det_registered"], r["td10_holdout"],
                     r["fire_p50"], r["fpr_target_succ"], r["n_succ_calib"],
                     r["n_phase_jfair"], r["n_phase_plain"], r["phases_jfair"],
                     r["cos_median"], r["cos_min"], r["min_fail_ep_in_mixed"]]
            fh.write("\t".join(fmt(v) for v in vals) + "\n")


def group_stats(rows, arms, ref_arm="reseed"):
    """(group_type, group_value, arm) → 통계. 완료 셀만."""
    out = OrderedDict()

    def groups_of(r):
        g = [("all", "all"), ("instruction", r["instr_new"])]
        if r["regime"] != NA:
            g.append(("regime", r["regime"]))
        return g

    for r in rows:
        ref = r["_arms"].get(ref_arm, {"complete": 0, "eps": []})
        for a in arms:
            v = r["_arms"][a]
            if not v["complete"]:
                continue
            for gt, gv in groups_of(r):
                k = (gt, gv, a)
                s = out.setdefault(k, {"n_cells": 0, "n_eps": 0,
                                       "n_rescued": 0, "n_paired": 0,
                                       "only_arm": 0, "only_ref": 0,
                                       "both": 0, "neither": 0})
                s["n_cells"] += 1
                s["n_eps"] += v["n"]
                s["n_rescued"] += v["rescued"]
                if a != ref_arm and ref["complete"]:
                    rmap = {e["ep"]: e["success_i"] for e in ref["eps"]}
                    for e in v["eps"]:
                        if e["ep"] not in rmap:
                            continue
                        s["n_paired"] += 1
                        am, rf = e["success_i"], rmap[e["ep"]]
                        if am and not rf:
                            s["only_arm"] += 1
                        elif rf and not am:
                            s["only_ref"] += 1
                        elif am and rf:
                            s["both"] += 1
                        else:
                            s["neither"] += 1
    return out


def write_arms(path, stats, arms):
    hdr = ["group_type", "group_value", "arm", "n_cells", "n_eps",
           "n_rescued", "rescue_rate", "n_paired",
           "rescued_by_arm_not_reseed", "rescued_by_reseed_not_arm",
           "both", "neither"]
    order = {"all": 0, "instruction": 1, "regime": 2}
    keys = sorted(stats, key=lambda k: (order.get(k[0], 9), k[1],
                                        arm_sort_key(k[2])))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for k in keys:
            gt, gv, a = k
            s = stats[k]
            rate = s["n_rescued"] / s["n_eps"] if s["n_eps"] else None
            fh.write("\t".join(fmt(v) for v in [
                gt, gv, a, s["n_cells"], s["n_eps"], s["n_rescued"],
                None if rate is None else round(rate, 4), s["n_paired"],
                s["only_arm"], s["only_ref"], s["both"], s["neither"]]) + "\n")


def write_episodes(path, rows, arms):
    hdr = ["instruction_new", "old_instruction", "scene", "jitter", "machine",
           "arm", "runner_arm", "ep", "env_seed", "inference_seed", "success",
           "steps", "n_inferences", "trigger_step", "phase_at_trigger",
           "n_gated", "gated_mode", "beta", "op", "noise_idx",
           "collection_success", "cell_complete"] + SIDECAR_FIELDS
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            for a in arms:
                v = r["_arms"][a]
                for e in v["eps"]:
                    vals = [r["instr_new"], r["instr_old"], r["scene"],
                            r["jitter"], r["machine"], a, e["runner_arm"],
                            e["ep"], e["env_seed"], e["inference_seed"],
                            e["success"], e["steps"], e["n_inferences"],
                            e["trigger_step"], e["phase_at_trigger"],
                            e["n_gated"], e["gated_mode"], e["beta"], e["op"],
                            e["noise_idx"], e["collection_success"],
                            v["complete"]]
                    vals += [e.get(f) for f in SIDECAR_FIELDS]
                    fh.write("\t".join(fmt(v) for v in vals) + "\n")


def write_summary(path, rows, arms, stats, orphans, args):
    L = []
    A = L.append
    A("# v6 LOKO 구제 eval 집계")
    A("")
    A(f"- results-root: {', '.join(args.results_root)}")
    A(f"- cell 표: {args.cells}")
    A(f"- 키 매핑 적용(구→신), 표는 모두 신 키")
    A("")

    A("## arm 별 구제율 (완료 셀만)")
    A("")
    A("| arm | 셀 | 구제 | vs reseed (arm만/reseed만/둘다) |")
    A("|---|---|---|---|")
    for a in arms:
        s = stats.get(("all", "all", a))
        if not s:
            A(f"| {a} | 0 | -- | -- |")
            continue
        pair = (f"{s['only_arm']}/{s['only_ref']}/{s['both']}"
                f" (쌍 {s['n_paired']})") if a != "reseed" else "--"
        A(f"| {a} | {s['n_cells']} | {kn(s['n_rescued'], s['n_eps'])} | {pair} |")
    A("")

    A("## instruction 별")
    A("")
    A("| instruction | " + " | ".join(arms) + " |")
    A("|" + "---|" * (len(arms) + 1))
    instrs = sorted({k[1] for k in stats if k[0] == "instruction"})
    for gv in instrs:
        cells_ = []
        for a in arms:
            s = stats.get(("instruction", gv, a))
            cells_.append(kn(s["n_rescued"], s["n_eps"]) if s else "--")
        A(f"| {gv} | " + " | ".join(cells_) + " |")
    A("")

    A("## regime 별")
    A("")
    A("| regime | " + " | ".join(arms) + " |")
    A("|" + "---|" * (len(arms) + 1))
    for gv in sorted({k[1] for k in stats if k[0] == "regime"}):
        cells_ = []
        for a in arms:
            s = stats.get(("regime", gv, a))
            cells_.append(kn(s["n_rescued"], s["n_eps"]) if s else "--")
        A(f"| {gv} | " + " | ".join(cells_) + " |")
    A("")

    A("## (instruction, scene) 셀 단위")
    A("")
    hdr = ["instruction", "s", "j", "n_ep"] + arms + \
          ["regime", "fire_p50", "fpr", "td10_ho", "cos_med", "n_ph_jf"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["instr_new"], x["scene"], x["jitter"])):
        if not any(r["_arms"][a]["n"] for a in arms):
            continue
        cells_ = []
        for a in arms:
            v = r["_arms"][a]
            if not v["n"]:
                cells_.append("--")
            else:
                mark = "" if v["complete"] else "*"
                cells_.append(f"{v['rescued']}/{v['n']}{mark}")
        A("| " + " | ".join([
            r["instr_new"], str(r["scene"]), str(r["jitter"]),
            str(r["n_eval_eps"])] + cells_ + [
            r["regime"], fmt(r["fire_p50"], 1), fmt(r["fpr_target_succ"], 2),
            fmt(r["td10_holdout"], 2), fmt(r["cos_median"], 3),
            fmt(r["n_phase_jfair"])]) + " |")
    A("")
    A("(`*` = 미완료 셀, 비율 집계 제외)")
    A("")

    # 주의
    A("## 주의")
    A("")
    inc = []
    for r in rows:
        bad = [f"{a} {v['n']}/{r['n_eval_eps']}"
               for a, v in r["_arms"].items() if v["n"] and not v["complete"]]
        if bad:
            inc.append(f"{r['instr_new']} s{r['scene']} j{r['jitter']}: "
                       + ", ".join(bad))
    A(f"- 미완료 셀 {len(inc)}건" + ("" if not inc else ":"))
    for x in inc:
        A(f"  - {x}")
    small = [f"{r['instr_new']} s{r['scene']} j{r['jitter']} (n={r['n_eval_eps']})"
             for r in rows if r["small_n"] and any(r["_arms"][a]["n"] for a in arms)]
    A(f"- small_n (n<={args.small_n}) 셀 {len(small)}건"
      + ("" if not small else ": " + ", ".join(small)))
    init = [f"{r['instr_new']} s{r['scene']} j{r['jitter']}"
            for r in rows if r["regime"] == "초기조건형"
            and any(r["_arms"][a]["n"] for a in arms)]
    A(f"- 초기조건형 셀 {len(init)}건" + ("" if not init else ": " + ", ".join(init)))
    zero = [f"{r['instr_new']} s{r['scene']} j{r['jitter']}"
            for r in rows if r["n_phase_jfair"] == 0
            and any(r["_arms"][a]["n"] for a in arms)]
    A(f"- jfair phase 0 셀 {len(zero)}건" + ("" if not zero else ": " + ", ".join(zero)))
    nodet = [f"{r['instr_new']} s{r['scene']} j{r['jitter']}"
             for r in rows if r["regime"] == NA
             and any(r["_arms"][a]["n"] for a in arms)]
    A(f"- detector 미조인 셀 {len(nodet)}건" + ("" if not nodet else ": " + ", ".join(nodet)))
    if orphans:
        A(f"- cell 표에 없는 결과 셀 {len(orphans)}건: "
          + ", ".join(f"{s} s{sc} j{j}" for s, sc, j in orphans))
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description="v6 expand LOKO 구제 eval 집계")
    p.add_argument("--results-root", action="append", default=None,
                   help="결과 트리 루트 (여러 번 지정 가능)")
    p.add_argument("--cells", default=DEF_CELLS, help="eval cell 표 (구 키)")
    p.add_argument("--detector", default=DEF_DETECTOR)
    p.add_argument("--op-root", default=DEF_OP_ROOT)
    p.add_argument("--op-jfair", default="instr_setm_v6_gt")
    p.add_argument("--op-plain", default="instr_setm_v6_gt_plain")
    p.add_argument("--key-map", default=DEF_KEY_MAP,
                   help="구→신 instruction 매핑 'old=new,...'")
    p.add_argument("--out-dir", default=DEF_OUT)
    p.add_argument("--limit-arms", default=None,
                   help="집계할 arm 짧은 이름 콤마 목록 (예: setm_gt_b08,reseed)")
    p.add_argument("--small-n", type=int, default=6)
    p.add_argument("--strict", action="store_true",
                   help="미완료 셀이 있으면 에러")
    p.add_argument("--no-sidecar", action="store_true",
                   help="raw_rollouts 사이드카 JSON 읽지 않음 (빠름)")
    args = p.parse_args(argv)
    if not args.results_root:
        args.results_root = [DEF_RESULTS]
    args.results_root = [os.path.abspath(r) for r in args.results_root]

    rows, arms, orphans, _cells = build(args)
    os.makedirs(args.out_dir, exist_ok=True)
    stats = group_stats(rows, arms)

    write_cells(os.path.join(args.out_dir, "cells.tsv"), rows, arms)
    write_arms(os.path.join(args.out_dir, "arms.tsv"), stats, arms)
    write_episodes(os.path.join(args.out_dir, "episodes.tsv"), rows, arms)
    write_summary(os.path.join(args.out_dir, "summary.md"),
                  rows, arms, stats, orphans, args)

    # 콘솔 요약
    n_cells_tot = len(rows)
    touched = [r for r in rows if any(r["_arms"][a]["n"] for a in arms)]
    print(f"cells(표) {n_cells_tot} · 결과 있는 셀 {len(touched)} · arm {len(arms)}")
    for a in arms:
        s = stats.get(("all", "all", a))
        n_any = sum(1 for r in rows if r["_arms"][a]["n"])
        n_comp = sum(1 for r in rows if r["_arms"][a]["complete"])
        if s:
            print(f"  {a:<18} 셀 {n_comp}/{n_any} 완료 · "
                  f"구제 {kn(s['n_rescued'], s['n_eps'])}")
        else:
            print(f"  {a:<18} 셀 {n_comp}/{n_any} 완료 · 구제 --")
    print(f"out: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
