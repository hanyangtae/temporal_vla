"""exp4-2 P0 집계 — config 별 유도 실패율 표 + 40–70% 게이트 flag + sham/무결성 감사.

  python aggregate_p0.py --p0-dir <.../exp42_induced/p0> [--nas 5] [--out p0_failure_rates.tsv]

읽기: <p0>/grid/<config>/raw_rollouts/**/task*--ep*--succ*.json (+ baseline).
출력: tsv(모드/config/n/fail/rate/gate) + stdout 요약 + sham·산술 경고.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

GATE_LO, GATE_HI = 0.40, 0.70


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p0-dir", required=True)
    ap.add_argument("--nas", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p0 = Path(args.p0_dir)

    base_fail = base_n = 0
    base_succ_by_ep: dict[int, int] = {}
    for p in (p0 / "baseline").glob("raw_rollouts/*/*/task*--ep*--succ*.json"):
        d = json.loads(p.read_text())
        base_n += 1
        base_fail += 1 - int(d.get("episode_success", 0))
        base_succ_by_ep[int(d.get("episode_idx", -1))] = int(d.get("episode_success", 0))

    rows = []
    warns = []
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for p in sorted((p0 / "grid").glob("*/raw_rollouts/*/*/task*--ep*--succ*.json")):
        config = p.relative_to(p0 / "grid").parts[0]
        d = json.loads(p.read_text())
        d["_file"] = str(p)
        by_cfg[config].append(d)

    for config, ds in sorted(by_cfg.items()):
        real = [d for d in ds if not (d.get("perturb_spec") or {}).get("sham")]
        sham = [d for d in ds if (d.get("perturb_spec") or {}).get("sham")]
        for d in ds:
            spec = d.get("perturb_spec") or {}
            if not spec:
                warns.append(f"{d['_file']}: perturb_spec 없음")
                continue
            if d.get("run_tag") and spec.get("tag") and d["run_tag"] != spec["tag"]:
                warns.append(f"{d['_file']}: run_tag {d['run_tag']} != spec.tag {spec['tag']}")
            mode = spec.get("mode")
            if mode == "P2_force" and not spec.get("sham") and d.get("perturb_released") is False:
                # 에피소드가 해제 전에 끝남 — 물리적 문제는 없으나 dose 상이 기록
                warns.append(f"{d['_file']}: P2 미해제 종료 (dose 짧음)")
            if mode == "G1_gripper_init" and not spec.get("sham"):
                off = int(d.get("perturb_env_step_offset") or 0)
                if off <= 0 or off % args.nas:
                    warns.append(f"{d['_file']}: G1 offset={off} 이상")
        n = len(real)
        fail = sum(1 - int(d.get("episode_success", 0)) for d in real)
        rate = fail / n if n else float("nan")
        gate = "PASS" if n and GATE_LO <= rate <= GATE_HI else "-"
        mode = (real or ds)[0].get("perturb_spec", {}).get("mode", "?") if ds else "?"
        rows.append((mode, config, n, fail, f"{rate:.2f}" if n else "nan", gate))
        for d in sham:
            # sham 은 같은 ep 의 baseline 성공 여부를 정확히 재현해야 함 (S1 bitwise 의 지속 감시판)
            ep = int(d.get("episode_idx", -1))
            if ep in base_succ_by_ep and int(d.get("episode_success", -1)) != base_succ_by_ep[ep]:
                warns.append(f"{d['_file']}: sham 판정 {d.get('episode_success')} != "
                             f"baseline ep{ep} {base_succ_by_ep[ep]} (배선 드리프트 의심)")

    hdr = ("mode", "config", "n", "fail", "rate", "gate40-70")
    lines = ["\t".join(hdr)] + ["\t".join(str(x) for x in r) for r in rows]
    table = "\n".join(lines)
    print(f"[baseline] n={base_n} fail={base_fail} (실패율 {base_fail / base_n:.2f})"
          if base_n else "[baseline] 없음")
    print(table)
    if warns:
        print(f"\n[경고 {len(warns)}건]")
        for w in warns[:20]:
            print(f"  {w}")
    out = Path(args.out) if args.out else p0 / "p0_failure_rates.tsv"
    out.write_text(table + "\n")
    print(f"\nwrote {out}")
    p1_files = [d["_file"] for ds in by_cfg.values() for d in ds
                if (d.get("perturb_spec") or {}).get("mode") == "P1_displace"]
    if p1_files:
        print(f"[P1 mp4 육안 목록] {len(p1_files)}판 — teleport 폭주(물체 소실/관통) 확인 권장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
