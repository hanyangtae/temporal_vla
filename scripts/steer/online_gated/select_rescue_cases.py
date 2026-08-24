#!/usr/bin/env python3
"""index_rollouts_v4.tsv → 구제 케이스 선정 (기준 ①~④) + 파일럿 1케이스/instruction 추출.

기준 (사용자 확정, 2026-08-24):
  ① scene 단위 성공 > 5              — 정책이 아예 못 하는 배포지는 제외
  ② 대상 k 의 성공이 1~4판          — 성공이 하나라도 있어야 구제 가능성이 증명됨
  ③ 연산자 fit = 대상 k 를 뺀 나머지 k 전부 + 대상 k 의 실패 rollout
  ④ 대상 k 를 뺀 나머지 k 에도 실패가 1판 이상
     (없으면 연산자가 "대상 k = 실패, 나머지 k = 성공" 즉 k 좌표 자체를 학습한다)

출력:
  --out-cases  전체 케이스 tsv (instruction, scene, k, 대상판, fit succ/fail ...)
  --out-pilot  instruction 당 1케이스 × 대상 1판 (파일럿). 케이스 선택 = fit 균형
               min(succ,fail) 최대 → 동률이면 대상 k 성공 많은 쪽(구제 가능성 근거 강함).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

TRUE = ("1", "True", "true")


def ksort(k: str):
    return (1, 0) if k == "base" else (0, int(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out-cases", type=Path, default=None)
    ap.add_argument("--out-pilot", type=Path, default=None)
    ap.add_argument("--fit-manifest-dir", type=Path, default=None,
                    help="케이스별 fit episode 매니페스트 출력 (fit_cond_guidance --episode-manifest 계약)")
    ap.add_argument("--pilot-only", action="store_true",
                    help="--fit-manifest-dir 를 파일럿 케이스(instruction 당 1개)에만 쓴다")
    args = ap.parse_args()

    with args.index.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    cell: dict[tuple, list[dict]] = {}
    for r in rows:
        cell.setdefault((r["grid_instruction"], int(r["scene_idx"]),
                         r["jitter_reset_idx"]), []).append(r)
    ks_by: dict[tuple, set] = {}
    for (ins, sc, k) in cell:
        ks_by.setdefault((ins, sc), set()).add(k)

    cases = []
    for (ins, sc), ks in sorted(ks_by.items()):
        eps = [r for k in ks for r in cell[(ins, sc, k)]]
        ts = sum(1 for r in eps if r["success"] in TRUE)
        tf = len(eps) - ts
        if ts <= 5:                                            # ①
            continue
        for k in sorted(ks, key=ksort):
            kr = cell[(ins, sc, k)]
            ks_succ = sum(1 for r in kr if r["success"] in TRUE)
            ks_fail = len(kr) - ks_succ
            if not (1 <= ks_succ <= 4):                        # ②
                continue
            if tf - ks_fail < 1:                               # ④
                continue
            fit_rows = [r for kk2 in ks if kk2 != k for r in cell[(ins, sc, kk2)]]
            fit_rows += [r for r in kr if r["success"] not in TRUE]   # ③ 대상 k 의 실패도 fit 에
            cases.append({
                "fit_rows": fit_rows,
                "instruction": ins, "scene": sc, "target_k": k,
                "target_succ": ks_succ, "target_fail": ks_fail,
                "scene_succ": ts, "scene_fail": tf,
                "fit_succ": ts - ks_succ, "fit_fail": tf,       # ③
                "other_fail": tf - ks_fail,
                "target_rows": sorted((r for r in kr if r["success"] not in TRUE),
                                      key=lambda r: int(r["noise_idx"])),
            })

    if args.out_cases:
        args.out_cases.parent.mkdir(parents=True, exist_ok=True)
        with args.out_cases.open("w", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["instruction", "scene", "target_k", "target_succ", "target_fail",
                        "scene_succ", "scene_fail", "fit_succ", "fit_fail", "other_fail"])
            for c in cases:
                w.writerow([c[x] for x in ("instruction", "scene", "target_k",
                                           "target_succ", "target_fail", "scene_succ",
                                           "scene_fail", "fit_succ", "fit_fail", "other_fail")])
        print(f"[cases] {len(cases)} 케이스 / 대상 {sum(c['target_fail'] for c in cases)}판 → {args.out_cases}")

    def write_fit_manifest(c, out_dir: Path) -> Path:
        slug = c["instruction"].replace("/", "_")
        out = out_dir / f"{slug}_s{c['scene']}_k{c['target_k']}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            fh.write("pkl_path\tlabel\tbase_scene\tcell_si\tnoise_idx\n")
            for r in sorted(c["fit_rows"], key=lambda x: (int(x["cell_si"]), int(x["noise_idx"]))):
                lab = 1 if r["success"] in TRUE else 0
                fh.write(f"{r['rel_path'].rstrip('/')}/rollout.pkl\t{lab}\t{c['scene']}"
                         f"\t{r['cell_si']}\t{r['noise_idx']}\n")
        return out

    # 파일럿: instruction 당 1케이스, 그 케이스의 대상 1판(가장 작은 noise_idx)
    best: dict[str, dict] = {}
    for c in cases:
        key = c["instruction"]
        score = (min(c["fit_succ"], c["fit_fail"]), c["target_succ"])
        if key not in best or score > best[key]["_score"]:
            best[key] = {**c, "_score": score}
    if args.out_pilot:
        args.out_pilot.parent.mkdir(parents=True, exist_ok=True)
        with args.out_pilot.open("w", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["instruction", "scene", "target_k", "noise_idx", "cell_si",
                        "env_seed", "inference_seed", "machine", "rel_path",
                        "fit_succ", "fit_fail", "target_k_succ_fail"])
            for ins in sorted(best):
                c = best[ins]
                r = c["target_rows"][0]
                w.writerow([ins, c["scene"], c["target_k"], r["noise_idx"], r["cell_si"],
                            r["env_seed"], r["inference_seed"], r["machine"], r["rel_path"],
                            c["fit_succ"], c["fit_fail"],
                            f"{c['target_succ']}/{c['target_fail']}"])
                print(f"[pilot] {ins:22s} s{c['scene']} k{c['target_k']:5s} n{r['noise_idx']} "
                      f"(대상k {c['target_succ']}/{c['target_fail']}, fit {c['fit_succ']}/{c['fit_fail']}, "
                      f"machine={r['machine']})")
        print(f"[pilot] {len(best)} instruction × 1판 → {args.out_pilot}")

    if args.fit_manifest_dir:
        targets = list(best.values()) if args.pilot_only else cases
        for c in targets:
            out = write_fit_manifest(c, args.fit_manifest_dir)
            print(f"[fit] {c['instruction']:22s} s{c['scene']} k{c['target_k']:5s} "
                  f"→ {len(c['fit_rows'])}판 (succ {c['fit_succ']} / fail {c['fit_fail']}) {out.name}")
        print(f"[fit] 매니페스트 {len(targets)}개 → {args.fit_manifest_dir}")


if __name__ == "__main__":
    main()
