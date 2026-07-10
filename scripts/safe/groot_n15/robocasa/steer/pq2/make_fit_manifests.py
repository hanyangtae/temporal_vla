"""fit 표본 manifest 생성기 (재설계 라운드 pq2 — docs/steering/17 + Gate1 합의 구현).

scene 모드: 한 scene 의 fit 재료(ep0-59 pkl)를 라벨 층화·고정 seed 로 30/30 disjoint split
  (fit-half / select-half) → fit-half 에서 per_scene fit{15,30} manifest 추출.
  fit↔P2 선택 episode 중첩(in-sample rescue 아티팩트) 차단이 목적. select-half 는 P2 전용.
pool 모드: 여러 scene 의 split.json 을 모아 cross_scene(동일 instruction 4 scene) /
  grand(8 scene) pool manifest — 총량 고정(fitN = 총 N판), scene 하한 + 클래스 하한(pool 전역).
위약(null): --null-seed 지정 시 동일 episode 집합에서 scene 층 내 라벨 permutation
  (클래스 수 보존) manifest 를 추가 생성 — record 는 fit 단계에서 episode 라벨 상속.

라벨 규칙 (2026-07-10 사용자 확정 D1):
  기본 = pkl 파일명 succ 플래그 (bread 등 재채점 없는 scene).
  --rejudge-tsv (apple) = corrected(ever@0.1) 라벨로 override, 0.07/0.10 판정이 갈리는 판
  (discordant)은 성공도 실패도 아닌 것으로 보고 fit 표본에서 제외.

split 규칙 (Codex Gate1 합의):
  층화 50/50, fit-half 클래스 하한(fit15≥3, fit30≥6) 충족 위해 select-half 에서 최대
  ASYM_MAX(2)판 비대칭 이동 허용(기록). 그래도 미충족이면 해당 fitN 은 N/A 로 기록만
  (arm 생성 단계가 스킵) — split 을 넘는 backfill 금지.

산출 manifest tsv: `pkl_path(repo상대)\tlabel\tscene` + '#' 주석(seed·구성). fit 스크립트
`--manifest` 가 소비. 모든 난수는 고정 seed 파생 문자열 → 재현 가능.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]

ASYM_MAX = 2  # fit-half 클래스 하한 충족용 비대칭 이동 상한 (Gate1 합의: 1~2판)
MIN_CLASS = {15: 3, 30: 6}  # fitN → fit 표본 내 클래스별 최소 episode (사용자 확정)


def scene_episodes(scene_dir: Path, rejudge_tsv: Path | None):
    """[(ep:int, pkl:Path, label:int)] — discordant 제외 적용 후. 라벨 출처도 반환."""
    eps = {}
    for p in sorted(scene_dir.glob("task*--ep*--succ*.pkl")):
        m = re.match(r"task\d+--ep(\d+)--succ([01])\.pkl", p.name)
        if m:
            eps[int(m.group(1))] = {"pkl": p, "label": int(m.group(2))}
    if not eps:
        raise SystemExit(f"pkl 없음: {scene_dir}")
    excluded, source = [], "filename"
    if rejudge_tsv:
        source = f"rejudge:{rejudge_tsv.name}"
        rows = [l.split("\t") for l in rejudge_tsv.read_text().splitlines()]
        hdr = rows[0]
        i_dir, i_ep = hdr.index("dir"), hdr.index("ep")
        i_07, i_10 = hdr.index("ever@0.07"), hdr.index("ever@0.1")
        suffix = str(scene_dir).rstrip("/").split("/")[-1]
        seen = set()
        for r in rows[1:]:
            if len(r) <= max(i_07, i_10) or not r[i_dir].rstrip("/").endswith(suffix):
                continue
            ep = int(r[i_ep].lstrip("ep"))
            if ep not in eps or ep in seen:  # (dir,ep) 중복 행은 첫 행 채택
                continue
            seen.add(ep)
            lab07, lab10 = int(r[i_07]), int(r[i_10])
            if lab07 != lab10:  # 0.07~0.10 걸친 애매 판 → fit 제외 (D1)
                excluded.append(ep)
                del eps[ep]
            else:
                eps[ep]["label"] = lab10
        missing = [e for e in eps if e not in seen and e not in excluded]
        if missing:
            raise SystemExit(f"재채점 미커버 ep {len(missing)}개 (예: {sorted(missing)[:5]}) — "
                             f"재채점 완주 후 재실행 필요")
    return eps, excluded, source


def stratified_split(eps: dict, seed: str):
    """라벨 층화 50/50 split + fit-half 하한(fit30 기준 6) 비대칭 보정. → (fit_half, select_half, moved)"""
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for ep, d in eps.items():
        by_label[d["label"]].append(ep)
    fit_half, select_half = [], []
    fit_gets_remainder = True  # 홀수 클래스 나머지를 fit/select 에 교대로 배정 (half 크기 균형)
    for lab, group in sorted(by_label.items()):
        group = sorted(group)
        rng.shuffle(group)
        k = len(group) // 2
        if len(group) % 2 and fit_gets_remainder:
            k += 1
        if len(group) % 2:
            fit_gets_remainder = not fit_gets_remainder
        fit_half += group[:k]
        select_half += group[k:]
    moved = []
    need = MIN_CLASS[30]  # 가장 큰 하한 기준으로 보정 (fit15 하한은 자동 충족)
    for lab in (0, 1):
        have = [e for e in fit_half if eps[e]["label"] == lab]
        pool = [e for e in select_half if eps[e]["label"] == lab]
        while len(have) < need and pool and len(moved) < ASYM_MAX:
            e = pool.pop(0)
            select_half.remove(e)
            fit_half.append(e)
            have.append(e)
            moved.append(e)
    return sorted(fit_half), sorted(select_half), moved


def sample_stratified(cands: list, labels: dict, n: int, min_class: int, rng,
                      allow_short: bool = False) -> list | None:
    """후보 episode 에서 총 n 개 층화 추출 (클래스 비율 유지, 클래스별 ≥min_class). 불가 시 None.

    allow_short: 후보가 n 미만이면 전체 사용 (fit30 = fit-half 전체 — Gate1 합의:
    discordant 제외 등으로 half 가 28~29 판이어도 명목 fit30 으로 취급, 실제 n 은 기록).
    """
    by = {0: sorted(e for e in cands if labels[e] == 0), 1: sorted(e for e in cands if labels[e] == 1)}
    if len(by[0]) < min_class or len(by[1]) < min_class:
        return None
    if len(cands) < n:
        if not allow_short:
            return None
        n = len(cands)
    n0 = max(min_class, min(len(by[0]), round(n * len(by[0]) / len(cands))))
    n0 = min(n0, n - min_class)  # 상대 클래스 하한 보장
    n1 = n - n0
    if n1 > len(by[1]):
        n1 = len(by[1]); n0 = n - n1
    rng.shuffle(by[0]); rng.shuffle(by[1])
    return sorted(by[0][:n0] + by[1][:n1])


def write_manifest(path: Path, rows: list, comments: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {c}" for c in comments]
    for pkl, label, scene in rows:
        rel = Path(pkl).resolve()
        rel = rel.relative_to(REPO) if str(rel).startswith(str(REPO)) else rel
        lines.append(f"{rel}\t{label}\t{scene}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def null_permute(rows: list, seed: str) -> list:
    """scene 층 내 라벨 permutation (클래스 수 보존). rows=(pkl,label,scene)."""
    rng = random.Random(seed)
    out = []
    for scene in sorted({r[2] for r in rows}):
        grp = [r for r in rows if r[2] == scene]
        labels = [r[1] for r in grp]
        rng.shuffle(labels)
        out += [(p, l, s) for (p, _, s), l in zip(grp, labels)]
    return sorted(out)


def cmd_scene(args):
    scene_dir = Path(args.scene_dir)
    scene = scene_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "outputs/eval/robocasa/groot_n15/pq2_manifests" / scene
    eps, excluded, source = scene_episodes(scene_dir, Path(args.rejudge_tsv) if args.rejudge_tsv else None)
    fit_half, select_half, moved = stratified_split(eps, f"{args.seed}:{scene}")
    labels = {e: eps[e]["label"] for e in eps}
    n_f = {h: sum(1 for e in h_eps if labels[e] == 0) for h, h_eps in
           [("fit", fit_half), ("select", select_half)]}
    split = {"scene": scene, "seed": args.seed, "labels_source": source,
             "excluded_discordant": excluded, "asym_moved": moved,
             "labels": {str(e): labels[e] for e in sorted(eps)},  # 단일 출처 (corrected 반영)
             "fit_half": fit_half, "select_half": select_half,
             "class_counts": {"fit_half": {"fail": n_f["fit"], "succ": len(fit_half) - n_f["fit"]},
                              "select_half": {"fail": n_f["select"], "succ": len(select_half) - n_f["select"]}},
             "scene_dir": str(scene_dir.resolve().relative_to(REPO) if str(scene_dir.resolve()).startswith(str(REPO)) else scene_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False))
    print(f"[{scene}] 사용 {len(eps)}ep (제외 {len(excluded)}) → fit {len(fit_half)} / select {len(select_half)}"
          f" (비대칭 이동 {moved}) 클래스 {split['class_counts']}")

    for n in (15, 30):
        rng = random.Random(f"{args.seed}:{scene}:fit{n}")
        cands = sample_stratified(fit_half, labels, n, MIN_CLASS[n], rng, allow_short=(n == 30))
        if cands is None:
            nf = sum(1 for e in fit_half if labels[e] == 0)
            print(f"  [N/A] per_scene_fit{n}: fit-half {len(fit_half)}ep "
                  f"(fail {nf}/succ {len(fit_half) - nf}) 클래스 하한 {MIN_CLASS[n]} 미충족")
            continue
        rows = [(eps[e]["pkl"], labels[e], scene) for e in cands]
        comments = [f"scope=per_scene fitN={n} n_actual={len(cands)} scene={scene} seed={args.seed}",
                    f"labels={source} excluded_discordant={excluded}",
                    f"episodes={cands}"]
        write_manifest(out_dir / f"per_scene_fit{n}.tsv", rows, comments)
        if args.null_seed is not None:
            nrows = null_permute(rows, f"null{args.null_seed}:{scene}:fit{n}")
            write_manifest(out_dir / f"per_scene_fit{n}_null{args.null_seed}.tsv", nrows,
                           comments + [f"null_permutation seed={args.null_seed} (scene 층 내, 클래스 수 보존)"])


def cmd_pool(args):
    """cross_scene/grand pool: 각 scene split.json 의 fit-half 에서 총 N 층화 추출.

    라벨은 split.json["labels"] 가 단일 출처 (corrected·discordant 제외 반영) —
    filename 라벨 재계산 금지. pkl 경로만 scene_dir glob 으로 해석.
    """
    if args.fit_n not in MIN_CLASS:
        raise SystemExit(f"fit_n 은 {sorted(MIN_CLASS)} 중 하나")
    n, need = args.fit_n, MIN_CLASS[args.fit_n]
    scenes, pool, labels_all = {}, [], {}
    for d in args.scene_manifest_dirs:
        s = json.loads((Path(d) / "split.json").read_text())
        scene_dir = REPO / s["scene_dir"]
        eps, _, _ = scene_episodes(scene_dir, None)  # pkl 경로 해석 전용
        scenes[s["scene"]] = {"split": s, "eps": eps}
        for e in s["fit_half"]:
            pool.append((s["scene"], e))
            labels_all[(s["scene"], e)] = int(s["labels"][str(e)])
    rng = random.Random(f"{args.seed}:{args.scope}:fit{n}")
    # scene 하한 우선 배정 → 잔여는 전 scene fit-half 합동 무작위 (클래스 하한은 pool 전역 — 문서17)
    chosen = []
    for name, sc in sorted(scenes.items()):
        fh = sorted(sc["split"]["fit_half"])
        chosen += [(name, e) for e in rng.sample(fh, min(args.scene_min, len(fh)))]
    rest = [c for c in pool if c not in chosen]
    rng.shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    if len(chosen) < n:
        raise SystemExit(f"pool 표본 부족: {len(chosen)} < {n}")
    chosen = chosen[:n]
    # 클래스 하한 보정: 부족 클래스를 잔여 pool 에서 교체. 교체로 빠지는 쪽은 scene 하한을
    # 깨지 않는 scene 에서만 뽑는다.
    def scene_count(name):
        return sum(1 for c in chosen if c[0] == name)

    for lack in (0, 1):
        cands = [c for c in pool if c not in chosen and labels_all[c] == lack]
        rng.shuffle(cands)
        while sum(1 for c in chosen if labels_all[c] == lack) < need and cands:
            surplus = [c for c in chosen
                       if labels_all[c] != lack and scene_count(c[0]) > args.scene_min]
            if not surplus:
                break
            chosen.remove(surplus[-1])
            chosen.append(cands.pop())
    counts = {lab: sum(1 for c in chosen if labels_all[c] == lab) for lab in (0, 1)}
    if counts[0] < need or counts[1] < need:
        raise SystemExit(f"pool 클래스 하한 미충족: fail={counts[0]} succ={counts[1]} < {need}")
    rows = [(scenes[name]["eps"][e]["pkl"], labels_all[(name, e)], name) for name, e in sorted(chosen)]
    comments = [f"scope={args.scope} fitN={n} seed={args.seed} scene_min={args.scene_min}",
                f"scenes={sorted(scenes)} class_counts={counts}",
                f"chosen={sorted(chosen)}"]
    out = Path(args.out)
    write_manifest(out, rows, comments)
    if args.null_seed is not None:
        write_manifest(out.with_name(out.stem + f"_null{args.null_seed}.tsv"),
                       null_permute(rows, f"null{args.null_seed}:{args.scope}:fit{n}"),
                       comments + [f"null_permutation seed={args.null_seed}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scene", help="scene split + per_scene manifest")
    s.add_argument("--scene-dir", required=True)
    s.add_argument("--rejudge-tsv", default=None, help="apple corrected 라벨 tsv (ever@0.1 override)")
    s.add_argument("--seed", default="pq2-20260710")
    s.add_argument("--null-seed", type=int, default=None)
    s.add_argument("--out-dir", default=None)
    s.set_defaults(fn=cmd_scene)
    p = sub.add_parser("pool", help="cross_scene/grand pool manifest")
    p.add_argument("--scene-manifest-dirs", nargs="+", required=True, help="scene 모드 out-dir 들")
    p.add_argument("--scope", choices=["cross_scene", "grand"], required=True)
    p.add_argument("--fit-n", type=int, required=True)
    p.add_argument("--scene-min", type=int, required=True, help="scene당 최소 표본 (cross=3, grand=1)")
    p.add_argument("--seed", default="pq2-20260710")
    p.add_argument("--null-seed", type=int, default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pool)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
