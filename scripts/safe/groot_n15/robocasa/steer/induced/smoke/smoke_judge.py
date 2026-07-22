"""exp4-2 smoke S1–S5 공용 판정기 (PASS/FAIL + exit code).

서브커맨드:
- csv-bitwise A.csv B.csv [--expect-diff]      : action csv 비교 (기본 bitwise 일치 기대;
  불일치 시 수치 파싱해 max|Δ| 보고. --expect-diff 는 반대로 "달라야 PASS" — 실효 증명)
- fields A.{pkl|json} B.{pkl|json} --fields k1,k2,...  : 판정 필드 json-equal 비교
- perturb-audit EP.{pkl|json} --nas 5          : perturb_* 키 산술 무결성 (모드별)
- status-audit STATUS.json --expect-start T0 --expect-len W [--expect-total N]
  : /patch_status fired_records 대조 (VL 은 --expect-total 로 요청당 1-fire 검증)

pkl 입력은 torch unpickle 이 필요하므로 lerobot 컨테이너에서 실행할 것.
stdout 마지막 줄 = "PASS ..." | "FAIL ...", exit 0/1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fail(msg: str) -> int:
    print(f"FAIL {msg}")
    return 1


def _ok(msg: str) -> int:
    print(f"PASS {msg}")
    return 0


def _load_fields(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".pkl":
        import pickle

        import torch  # noqa: F401 — unpickle (컨테이너 전용)

        with open(path, "rb") as f:
            return pickle.load(f)
    raise SystemExit(f"unsupported input: {path}")


def cmd_csv_bitwise(args: argparse.Namespace) -> int:
    a = Path(args.a).read_bytes()
    b = Path(args.b).read_bytes()
    same = a == b
    if args.expect_diff:
        if same:
            return _fail(f"csv-bitwise: {args.a} == {args.b} (실효 증명 실패 — 섭동 무효)")
        return _ok(f"csv-bitwise: 다름 확인 (실효) {args.a} vs {args.b}")
    if same:
        return _ok(f"csv-bitwise: identical {args.a} vs {args.b}")
    # 수치 파싱해 max|Δ| 보고 (bf16→fp16 왕복 등 "≈" 강등 판정 근거)
    import csv

    def rows(p):
        with open(p) as f:
            return [
                [float(x) for x in row]
                for row in csv.reader(f)
                if row and not any(c.isalpha() for c in "".join(row))
            ]

    ra, rb = rows(Path(args.a)), rows(Path(args.b))
    if len(ra) != len(rb):
        return _fail(f"csv-bitwise: 행수 불일치 {len(ra)} != {len(rb)}")
    max_d = 0.0
    for xa, xb in zip(ra, rb):
        if len(xa) != len(xb):
            return _fail("csv-bitwise: 열수 불일치")
        for va, vb in zip(xa, xb):
            max_d = max(max_d, abs(va - vb))
    return _fail(f"csv-bitwise: NOT identical, rows={len(ra)} max|delta|={max_d:.3e}")


def cmd_fields(args: argparse.Namespace) -> int:
    da, db = _load_fields(Path(args.a)), _load_fields(Path(args.b))
    keys = [k.strip() for k in args.fields.split(",") if k.strip()]
    bad = []
    for k in keys:
        va, vb = da.get(k), db.get(k)
        if json.dumps(va, sort_keys=True, default=str) != json.dumps(
            vb, sort_keys=True, default=str
        ):
            bad.append((k, va, vb))
    if bad:
        for k, va, vb in bad[:5]:
            print(f"  field {k}: {va!r} != {vb!r}")
        return _fail(f"fields: {len(bad)}/{len(keys)} 불일치 ({args.a} vs {args.b})")
    return _ok(f"fields: {len(keys)}개 일치")


def cmd_perturb_audit(args: argparse.Namespace) -> int:
    d = _load_fields(Path(args.ep))
    nas = int(args.nas)
    spec = d.get("perturb_spec")
    if spec is None:
        return _fail("perturb-audit: perturb_spec 없음")
    mode = spec.get("mode")
    win = d.get("perturb_record_window")
    applied = d.get("perturb_applied_env_steps") or []
    offset = int(d.get("perturb_env_step_offset") or 0)
    sham = bool(spec.get("sham"))
    errs = []
    if "spec_sha12" not in spec:
        errs.append("spec_sha12 없음")
    if mode == "C1_camera":
        if not d.get("perturb_persistent"):
            errs.append("C1 인데 perturb_persistent 아님")
        cams = d.get("perturb_cameras") or {}
        if not cams:
            errs.append("perturb_cameras 없음")
        for cam, ba in cams.items():
            moved = ba["pos_before"] != ba["pos_after"]
            if sham and moved:
                errs.append(f"{cam}: sham 인데 pos 변경")
            if not sham and not moved and spec.get("scale", 0) > 0:
                errs.append(f"{cam}: 실섭동인데 pos 불변")
    elif mode == "G1_gripper_init":
        if sham:
            if offset != 0 or not d.get("sham_skipped_sim_write"):
                errs.append(f"G1-sham: offset={offset} skip플래그={d.get('sham_skipped_sim_write')}")
        else:
            if offset <= 0 or offset % nas != 0:
                errs.append(f"G1: offset={offset} (nas 배수 아님)")
            ach = d.get("perturb_achieved_delta")
            tgt = spec.get("delta_xyz_m")
            if ach is None:
                errs.append("achieved_delta 없음")
            else:
                import math

                err = math.dist(ach, tgt)
                # OSC 추종 오차 허용: 목표 δ의 50% 이내 도달이면 실효로 판정
                if err > max(0.5 * math.dist([0, 0, 0], tgt), 0.02):
                    errs.append(f"achieved {ach} vs 목표 {tgt} (오차 {err:.3f}m)")
    elif mode in ("P1_displace", "P2_force"):
        t = spec.get("trigger_record")
        if not sham:
            if not applied:
                errs.append("applied_env_steps 비어 있음")
            elif applied[0] != t * nas + offset:
                errs.append(f"applied[0]={applied[0]} != trigger*nas={t * nas + offset}")
        if mode == "P2_force":
            dur = spec.get("duration_records")
            if win != [t, t + dur - 1]:
                errs.append(f"window={win} != [{t},{t + dur - 1}]")
            if not sham and len(applied) != dur * nas:
                errs.append(f"applied {len(applied)} != dur*nas {dur * nas}")
        elif win != [t, t]:
            errs.append(f"window={win} != [{t},{t}]")
    else:
        errs.append(f"unknown mode {mode}")
    if errs:
        for e in errs[:6]:
            print(f"  {e}")
        return _fail(f"perturb-audit[{mode} sham={sham}]: {len(errs)}건")
    return _ok(f"perturb-audit[{mode} sham={sham}]: 산술 무결")


def cmd_status_audit(args: argparse.Namespace) -> int:
    d = json.loads(Path(args.status).read_text())
    hooks = d.get("hooks") or {}
    errs = []
    for layer, h in hooks.items():
        fired = h.get("fired_records") or []
        if args.expect_total is not None and h.get("fired_total") != args.expect_total:
            errs.append(f"L{layer}: fired_total={h.get('fired_total')} != {args.expect_total}")
        if args.expect_start is not None and args.expect_len is not None:
            want = list(range(args.expect_start, args.expect_start + args.expect_len))
            if fired != want:
                errs.append(f"L{layer}: fired_records={fired} != {want}")
        if args.expect_full:
            # 요청당 1-fire 전창 발화 (VL 스모크): fired == [0..record_idx] 무결.
            n = int(h.get("record_idx", -1)) + 1
            if h.get("fired_total") != n or fired != list(range(n)):
                errs.append(
                    f"L{layer}: 전창 발화 아님 fired_total={h.get('fired_total')} "
                    f"records={n} fired={fired[:5]}..."
                )
        if h.get("exhausted_at"):
            errs.append(f"L{layer}: exhausted_at={h['exhausted_at']} (창 내 고갈)")
    if not hooks:
        errs.append("hooks 비어 있음")
    if errs:
        for e in errs[:6]:
            print(f"  {e}")
        return _fail(f"status-audit: {len(errs)}건")
    return _ok(f"status-audit: fired 창 일치 ({list(hooks.keys())})")


def cmd_fit_audit(args: argparse.Namespace) -> int:
    """S4: fit_inputs.json 의 절단 산술 검증 — fit_records == pkl 원 record 수 − start.

    pkl 로드에 torch 필요 (lerobot 컨테이너 실행).
    """
    fi = json.loads(Path(args.fit_inputs).read_text())
    eps = fi.get("episodes") or []
    if not eps:
        return _fail("fit-audit: episodes 비어 있음")
    errs = []
    for ep in eps:
        if int(ep.get("fit_start_record", -1)) != args.expect_start:
            errs.append(f"{ep['pkl']}: fit_start_record={ep.get('fit_start_record')} "
                        f"!= {args.expect_start}")
            continue
        d = _load_fields(Path(ep["pkl"]))
        n_orig = len(d["hidden_states"])
        want = n_orig - args.expect_start
        if int(ep.get("fit_records", -1)) != want:
            errs.append(f"{ep['pkl']}: fit_records={ep.get('fit_records')} != "
                        f"{n_orig}-{args.expect_start}={want}")
    if errs:
        for e in errs[:5]:
            print(f"  {e}")
        return _fail(f"fit-audit: {len(errs)}/{len(eps)}건 불일치")
    return _ok(f"fit-audit: {len(eps)} episodes 절단 산술 무결 (start={args.expect_start})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("csv-bitwise")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--expect-diff", action="store_true")
    p = sub.add_parser("fields")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--fields", required=True)
    p = sub.add_parser("perturb-audit")
    p.add_argument("ep")
    p.add_argument("--nas", type=int, default=5)
    p = sub.add_parser("status-audit")
    p.add_argument("status")
    p.add_argument("--expect-start", type=int, default=None)
    p.add_argument("--expect-len", type=int, default=None)
    p.add_argument("--expect-total", type=int, default=None)
    p.add_argument("--expect-full", action="store_true")
    p = sub.add_parser("fit-audit")
    p.add_argument("fit_inputs")
    p.add_argument("--expect-start", type=int, required=True)
    args = ap.parse_args()
    return {
        "csv-bitwise": cmd_csv_bitwise,
        "fields": cmd_fields,
        "perturb-audit": cmd_perturb_audit,
        "status-audit": cmd_status_audit,
        "fit-audit": cmd_fit_audit,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
