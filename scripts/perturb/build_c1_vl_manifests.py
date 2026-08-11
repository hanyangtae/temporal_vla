"""exp5-2 C1 VL conceptor fit 용 manifest 빌더 (clean vs perturbed 대조).

fit_phase_conceptor_n15.py 의 manifest 계약(pkl<TAB>label<TAB>scene)에 맞춰
  label 1 = clean (baseline_cap, succ1 만)   → C_success 슬롯 = C_clean
  label 0 = perturbed (capture/<cfg>, 전 라벨) → C_failure 슬롯 = C_perturbed
을 주입한다. C_steer = C_clean ∧ ¬C_perturbed 가 되어 "섭동 성분 억제" 연산자.

split: fit = 짝수 episode, held-out = 홀수 episode (locked 유지).
record_start: perturbed 는 rs_all.tsv(C1 은 전부 0), clean 은 0 을 채워 fail-loud 회피.
placebo: episode 라벨 순열 (클래스 크기 보존, seed 고정).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _ep(p: Path) -> int:
    return int(re.search(r"--ep(\d+)--", p.name).group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--perturb-dir", required=True, help="<cfg>/raw_rollouts 하위 pkl 루트")
    ap.add_argument("--record-start", required=True, help="perturbed rs_all.tsv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--placebo-seed", type=int, default=1)
    args = ap.parse_args()

    clean = sorted(Path(args.clean_dir).glob("task*--ep*--succ1.pkl"))
    pert = sorted(Path(args.perturb_dir).glob("**/task*--ep*--succ*.pkl"))
    if not clean or not pert:
        raise SystemExit(f"pkl 없음: clean={len(clean)} pert={len(pert)}")

    rs = {}
    for line in Path(args.record_start).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, b = line.split("\t")[:2]
        rs[str(Path(a).resolve())] = int(b)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = {"fit": [], "held": []}
    for p in clean:
        rows["fit" if _ep(p) % 2 == 0 else "held"].append((p, 1))
    for p in pert:
        rows["fit" if _ep(p) % 2 == 0 else "held"].append((p, 0))

    rs_lines = []
    for split, rr in rows.items():
        with (out / f"{split}_manifest.tsv").open("w") as f:
            for p, lab in rr:
                f.write(f"{p}\t{lab}\t0\n")   # scene 열은 미사용(eval-reserved 검사 비활성)
        for p, _ in rr:
            key = str(p.resolve())
            if key not in rs and "capture" in key:
                raise SystemExit(f"rs_all.tsv 미등재 perturbed pkl: {p}")
            rs_lines.append(f"{key}\t{rs.get(key, 0)}")
    (out / "record_start.tsv").write_text("\n".join(rs_lines) + "\n")

    # 위약: fit split episode 라벨 순열 (클래스 크기 보존)
    fit_rows = rows["fit"]
    labs = np.asarray([l for _, l in fit_rows])
    rng = np.random.default_rng(args.placebo_seed)
    perm = rng.permutation(labs)
    with (out / "fit_manifest_placebo.tsv").open("w") as f:
        for (p, _), lab in zip(fit_rows, perm):
            f.write(f"{p}\t{int(lab)}\t0\n")

    n = {k: (sum(1 for _, l in v if l == 1), sum(1 for _, l in v if l == 0)) for k, v in rows.items()}
    print(f"fit clean={n['fit'][0]} pert={n['fit'][1]} | held clean={n['held'][0]} pert={n['held'][1]}")
    print(f"placebo 라벨 일치율 = {float((perm == labs).mean()):.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
