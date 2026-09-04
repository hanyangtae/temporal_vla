"""LOKO fit 증분 러너 — 중추 셀표(v6_loko_cells.tsv)를 읽어 (키, scene)별로 fit 을 돌린다.

셀표 행 = 선정 셀의 실패 판이라 `(grid_instruction, scene_idx, jitter_idx)` distinct 가
fit 대상이다. 이 스크립트는 그 조합을 (키, scene) 로 묶어 지터 목록을 만들고, shard 가
올라온 키만 골라 원격(승준)에서 `loko_fit.py` 를 실행한다.

- 기본은 **--only-gt**: GT phase 라벨은 AE 번들과 무관하므로, 완료 instruction 의 shard
  로 만든 gt 산출물은 임시가 아니라 최종이다. ck8 은 정식 번들(`ae_k8/`) 이후 `--ck8` 로.
- 완주 판정은 로그의 `V5_LOKO_DONE <slug> s<scene>` 센티넬(셀 수 대조).
- 원격 CPU cap 8(승준 규약). 코드는 ssh stdin 으로 넣으므로 원격 repo 를 건드리지 않는다.

usage:
  python run_loko_fit.py --cells configs/collect/n15_grid_v6_scene_jitter/v6_loko_cells.tsv
                         [--only-slugs PPCC_bread,OpenDrawer_left] [--ck8] [--dry-run]
"""
import argparse
import collections
import csv
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
FIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loko_fit.py")
SSH = ["ssh", "-p", "11112", "kimseungjun@166.104.146.37"]
SEG_INSTR = "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/segA"
# scene 단위 shard 는 별도 디렉토리다 — instruction shard 폴더에 섞으면 ae_cluster 가
# scene shard 를 별개 instruction 으로 잡아 KMeans 단위가 조용히 바뀐다(action phase).
SEG_SCENE = "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/segA_scene"
SENTINEL = "V5_LOKO_DONE"        # loko_fit.py 가 (키, scene) 완주 시 찍는 문자열
AUDIT_INSTR = ("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/"
               "audit_cells.tsv")
AUDIT_SCENE = ("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/"
               "audit_cells_scene.tsv")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="중추 셀표 TSV")
    ap.add_argument("--only-slugs", default=None, help="쉼표 목록으로 한정")
    ap.add_argument("--ck8", action="store_true",
                    help="ck8 도 산출(정식 AE 번들 이후에만). 기본은 gt 전용")
    ap.add_argument("--bundle", default=None, help="AE 번들 경로 override(임시본 사용 시)")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--coord", default="j")
    ap.add_argument("--audit", default=None,
                    help="셀 감사표 경로. 기본은 scene·instruction 감사표를 둘 다 읽어 합침")
    ap.add_argument("--no-audit", action="store_true", help="교차대조 생략(권장하지 않음)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def cross_check(rows, audit_text, cells):
    """중추 셀표(index 유래) vs action phase 감사표(shard 유래) 대조.

    두 표는 **출처가 다르다** — 하나는 수집 인덱스, 하나는 추출된 activation shard.
    어긋나면 "라벨이 다른 세계" 계열 사고(v4r 반전 전례)이므로 fit 전에 멈춘다.
    """
    ad = {}
    for r in csv.DictReader(audit_text.splitlines(), delimiter="\t"):
        ad[(r["slug"], int(r["scene"]), int(r["jitter"]))] = (
            int(r["eps"]), int(r["expected"]), int(r["succ_eps"]))
    if not ad:
        return ["감사표가 비었거나 형식이 다르다 — 대조 불가"]

    # 셀표에서 (slug, scene) 별 지터 구성 재구성
    tgt_fail = collections.Counter()
    pool_succ = {}
    for r in rows:
        key = (r["slug"], int(r["scene_idx"]), int(r["jitter_idx"]))
        tgt_fail[key] += 1
        if "pool_succ" in r:
            pool_succ[key] = int(r["pool_succ"])

    bad = []
    for (slug, scene), js in sorted(cells.items()):
        scene_js = [k for k in ad if k[0] == slug and k[1] == scene]
        if not scene_js:
            continue                      # shard 미도착 — 대기로 처리됨
        for j in sorted(js):
            k = (slug, scene, j)
            if k not in ad:
                bad.append(f"{slug} s{scene} j{j}: 감사표에 없음")
                continue
            eps, exp, succ = ad[k]
            if eps != exp:
                bad.append(f"{slug} s{scene} j{j}: shard eps {eps} != 기대 {exp}")
            n_fail_shard = eps - succ
            if tgt_fail[k] != n_fail_shard:
                bad.append(f"{slug} s{scene} j{j}: 대상 실패 셀표 {tgt_fail[k]} != "
                           f"shard {n_fail_shard}")
            if k in pool_succ:
                ps = sum(ad[q][2] for q in scene_js if q[2] != j)
                if pool_succ[k] != ps:
                    bad.append(f"{slug} s{scene} j{j}: pool 성공 셀표 {pool_succ[k]} != "
                               f"shard {ps}")
    return bad


def main():
    args = parse_args()
    rows = list(csv.DictReader(open(args.cells), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{args.cells}: 행 0")
    need = ("grid_instruction", "slug", "scene_idx", "jitter_idx")
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise SystemExit(f"{args.cells}: 열 없음 {miss} (있는 열 {sorted(rows[0])})")

    cells = collections.defaultdict(set)
    for r in rows:
        cells[(r["slug"], int(r["scene_idx"]))].add(int(r["jitter_idx"]))
    only = set(args.only_slugs.split(",")) if args.only_slugs else None

    # shard 가 올라온 것만. scene 단위(`<slug>__s<i>.npz`, segA_scene/) 를 우선 쓰고,
    # instruction 단위(`<slug>.npz`, segA/) 는 그 instruction 이 다 모인 뒤의 병합본이다.
    ls = subprocess.run(SSH + [f"ls {SEG_SCENE}/*.npz {SEG_INSTR}/*.npz 2>/dev/null"],
                        capture_output=True, text=True).stdout.split()
    have_scene, have_instr = {}, set()
    for path in ls:
        base = os.path.basename(path)[:-4]
        if "__s" in base:
            sl, sc = base.rsplit("__s", 1)
            if sc.isdigit():
                have_scene[(sl, int(sc))] = path
        else:
            have_instr.add(base)
    print(f"[shard] scene 단위 {len(have_scene)}개 {sorted(have_scene)} | "
          f"instruction 단위 {len(have_instr)}개 {sorted(have_instr)}")

    todo, skip = [], []
    for (slug, scene), js in sorted(cells.items()):
        if only and slug not in only:
            continue
        if (slug, scene) in have_scene:
            todo.append((slug, scene, sorted(js), os.path.dirname(have_scene[(slug, scene)])))
        elif slug in have_instr:
            todo.append((slug, scene, sorted(js), SEG_INSTR))
        else:
            skip.append((slug, scene, sorted(js)))
    for slug, scene, js in skip:
        print(f"[대기] {slug} s{scene} j{js} — shard 미도착")
    if not todo:
        print("[run] 돌릴 셀 없음")
        return 0

    if not args.no_audit and not args.dry_run:
        paths = [args.audit] if args.audit else [AUDIT_SCENE, AUDIT_INSTR]
        txt = ""
        for ap_ in paths:
            t = subprocess.run(SSH + [f"cat {ap_} 2>/dev/null"],
                               capture_output=True, text=True).stdout
            if not t.strip():
                continue
            txt = t if not txt else txt + "\n".join(t.splitlines()[1:]) + "\n"
        if not txt.strip():
            print(f"[audit] 감사표 없음({args.audit}) — 대조 생략하고 진행")
        else:
            done_keys = {(s_, c_) for s_, c_, _, _ in todo}
            bad = cross_check(rows, txt, {k: v for k, v in cells.items()
                                          if k in done_keys})
            if bad:
                print("[audit] ★불일치 — fit 중단 (셀표=index 유래, 감사표=shard 유래)")
                for b in bad[:20]:
                    print("   ", b)
                return 2
            print("[audit] 셀표 x shard 감사표 대조 통과")

    fail = 0
    for slug, scene, js, segdir in todo:
        # scene shard 는 파일명이 <slug>__s<i>.npz 라 fit 에 --shard 로 실경로를 준다
        shard_arg = (f" --shard {segdir}/{slug}__s{scene}.npz"
                     if segdir.endswith("segA_scene") else f" --seg-dir {segdir}")
        cmd = (f"export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; "
               f"~/anaconda3/bin/python - {slug} {scene} --tag {args.tag} "
               f"--coord {args.coord} --jitters {','.join(map(str, js))}{shard_arg}")
        if not args.ck8:
            cmd += " --only-gt"
        elif args.bundle:
            cmd += f" --bundle {args.bundle}"
        print(f"\n=== {slug} s{scene} j{js} ({'gt+ck8' if args.ck8 else 'gt'}) "
              f"[{os.path.basename(segdir)}] ===")
        if args.dry_run:
            print("  [dry-run]", cmd)
            continue
        with open(FIT) as f:
            p = subprocess.run(SSH + [cmd], stdin=f, capture_output=True, text=True)
        out = p.stdout + p.stderr
        print(out.rstrip())
        if f"{SENTINEL} {slug} s{scene}" not in out:
            print(f"  [fail] 센티넬 없음 — rc={p.returncode}")
            fail += 1
    print(f"\n[run] 대상 {len(todo)} · 실패 {fail} · 대기 {len(skip)}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
