"""LOKO 연산자 성립 가능성 감사 — activation 없이 index TSV 만으로 사전 판정.

`loko_fit.py` 의 pool 계약(타 k 전판 + 대상 k 실패판, 대상 k 성공 제외)에서
**성공 클래스가 pool 에 몇 판 남는지**는 index 만으로 알 수 있다. 성공 pool 이 비거나
1 ep 이면 그 (instr, scene, k) 는 shard 를 뽑기 전에 이미 fit 불가다 — 추출·fit 을
돌리기 전에 eval 분모와 "미등록 예상 셀"을 확정하는 용도.

판정 기준 (fit 스크립트의 게이트에서 역산)
⚠ 좌표 열: v6 는 `jitter_idx`(plan jitters 순서), `jitter_reset_idx` 는 출처 열이다
  (oven/washer 는 전부 0) — 자동 선택은 jitter_idx→jitter 순이고 없으면 fail-loud.

  - pool 성공 0 ep         → 불가: 성공 가우시안·mean-diff 자체가 없음
  - pool 성공 1 ep         → 불가: 공분산 추정 불가 + ep 단위 CV 게이트(테스트 분할에
                             succ ≥2 ep 필요)가 원리적으로 못 돈다
  - pool 성공 2~4 ep       → 위험: 게이트 CV 표본 부족(5-seed 중 다수 시드가 스킵될 것)
  - pool 성공 ≥5 ep        → 가능
record 수 기준(setM ≥20/클래스·LLR ≥60/클래스)은 shard 가 있어야 알 수 있어 여기서
판정하지 않는다 — 이 표는 **필요조건**만 본다.

usage:
  python v5_loko_feasibility.py [--index TSV] [--cells "instr:scene,instr:scene,..."]
                                [--tsv OUT.tsv]
"""
import argparse
import collections
import csv
import math

DEF_INDEX = "configs/collect/n15_grid_v6/index_rollouts_v6.tsv"
AXIS_CANDIDATES = ("jitter_idx", "jitter")     # jitter_reset_idx 는 v6 에서 출처 열
# 중추('전체 파이프라인') v6 eval 대상 초안 (n10 재수집 후 확정 예정)
DEF_CELLS = ("drawer-left:0,drawer-right:1,oven-left:1,oven-right:2,"
             "washer-right:1,ppcc-bread:0,ppcc-candle:1,ppcc-jug:1")


def verdict(n_pool_succ, n_pairs, p_min, min_succ, min_pairs, p_cap) -> str:
    """등록이 **원리적으로** 가능한지 — 표본 구성만으로 갈리는 필요조건."""
    if n_pool_succ == 0:
        return "불가(pool 성공 0)"
    if n_pool_succ == 1:
        return "불가(pool 성공 1ep — 공분산 추정 불가)"
    if n_pairs < min_pairs:
        return f"불가(혼합 지터 쌍 {n_pairs} < {min_pairs})"
    if p_min > p_cap:
        return f"불가(순열 p 하한 {p_min:.3f} > {p_cap})"
    if n_pool_succ < min_succ:
        return f"위험(pool 성공 {n_pool_succ} < {min_succ})"
    return "가능"


def perm_p_floor(mixed) -> float:
    """지터 안 라벨 순열검정이 **달성할 수 있는 최소 p**.

    혼합 지터 j 마다 라벨 배치는 C(n_j, s_j) 가지이고 완벽 분리는 그중 1 가지다.
    지터끼리 독립이므로 완벽 분리 확률 = Π 1/C(n_j, s_j). 이 값이 0.05 를 넘으면
    **활성화가 아무리 잘 갈라도 게이트를 통과할 수 없다** — 수집·fit 전에 index 만으로
    판정되는 구조적 검정력 상한이다.
    """
    tot = 1
    for n_s, n_f in mixed:
        tot *= math.comb(n_s + n_f, n_s)
    return 1.0 / tot if tot else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEF_INDEX)
    ap.add_argument("--cells", default=DEF_CELLS)
    ap.add_argument("--tsv", default=None, help="지정 시 결과를 TSV 로도 저장")
    ap.add_argument("--axis-col", default=None,
                    help="좌표 열 명시. 기본 자동(jitter_idx→jitter). v6 에서 "
                         "jitter_reset_idx 는 출처 열이라 자동 선택하지 않는다")
    ap.add_argument("--min-pool-succ", type=int, default=9,
                    help="pool 성공 ep 하한 (중추 detector α=0.1 기준 9)")
    ap.add_argument("--min-pairs", type=int, default=6, help="혼합 지터 쌍 수 하한(게이트와 동일)")
    ap.add_argument("--p-cap", type=float, default=0.05, help="순열 p 상한(게이트와 동일)")
    ap.add_argument("--key-col", default=None,
                    help="instruction 키 열 명시. 기본 자동(grid_instruction→instruction_key)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.index), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{args.index}: 행 0")
    cols = rows[0].keys()
    axis = args.axis_col or next((c for c in AXIS_CANDIDATES if c in cols), None)
    if axis is None:
        raise SystemExit(f"{args.index}: 좌표 열 없음 (열 {sorted(cols)}). v6 는 jitter_idx — "
                         "jitter_reset_idx 는 출처 열이라 자동 선택하지 않는다(--axis-col 로 명시)")
    keyc = args.key_col or next((c for c in ("grid_instruction", "instruction_key") if c in cols), None)
    if keyc is None:
        raise SystemExit(f"{args.index}: instruction 키 열 없음 (열 {sorted(cols)})")
    print(f"[cols] 키={keyc} 좌표={axis}")
    cells = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1]))
             for c in args.cells.split(",") if c.strip()]

    out = []
    for ins, s in cells:
        sub = [r for r in rows if r[keyc] == ins and int(r["scene_idx"]) == s]
        if not sub:
            raise SystemExit(f"index 에 {ins} s{s} 행이 없다 — cells 인자 확인")
        for k in sorted({r[axis] for r in sub}, key=int):
            tgt_f = [r for r in sub if r[axis] == k and r["success"] != "1"]
            if not tgt_f:
                continue          # 대상 k 는 "실패가 있는 k" 뿐
            pool_s = [r for r in sub if r[axis] != k and r["success"] == "1"]
            pool_f = [r for r in sub if r["success"] != "1"]
            excl = [r for r in sub if r[axis] == k and r["success"] == "1"]
            mixed = []
            for kk in {r[axis] for r in sub}:
                ns = sum(1 for r in sub if r[axis] == kk and r["success"] == "1" and kk != k)
                nf = sum(1 for r in sub if r[axis] == kk and r["success"] != "1")
                if ns and nf:
                    mixed.append((ns, nf))
            n_pairs = sum(ns * nf for ns, nf in mixed)
            p_min = perm_p_floor(mixed)
            out.append((ins, s, int(k), len(pool_s), len(pool_f), len(tgt_f), len(excl),
                        len(mixed), n_pairs, p_min,
                        verdict(len(pool_s), n_pairs, p_min,
                                args.min_pool_succ, args.min_pairs, args.p_cap)))

    w = max(len(r[0]) for r in out)
    print(f"{'instruction':<{w}} {'s':>2} {'j':>3} {'poolS':>6}{'poolF':>6}{'tgtF':>5}"
          f"{'exclS':>6}{'혼합j':>6}{'쌍':>5}{'p하한':>8}  판정")
    print("-" * (w + 62))
    agg = collections.Counter()
    for ins, s, k, ps, pf, tf, ex, nm, npair, pmin, v in out:
        agg[v.split("(")[0]] += 1
        print(f"{ins:<{w}} {s:>2} {k:>3} {ps:>6}{pf:>6}{tf:>5}{ex:>6}{nm:>6}{npair:>5}"
              f"{pmin:>8.4f}  {v}")
    print("-" * (w + 62))
    print(f"대상 지터-셀 {len(out)} | " + " ".join(f"{k} {v}" for k, v in sorted(agg.items())))
    print(f"eval 판수(대상 지터 실패 합) {sum(r[5] for r in out)} | "
          f"그중 '불가' 셀 판수 {sum(r[5] for r in out if r[10].startswith('불가'))}")

    by_cell = collections.defaultdict(list)
    for r in out:
        by_cell[(r[0], r[1])].append(r)
    rank = []
    for (ins_, s_), rs in by_cell.items():
        ok = [r for r in rs if r[10] == "가능"]
        rank.append((len(ok), sum(r[5] for r in ok), ins_, s_, len(rs),
                     sum(r[5] for r in rs), sorted(r[9] for r in rs)[len(rs) // 2]))
    rank.sort(reverse=True)
    print("\n[후보 순위] 등록 가능 지터 수 → 그 지터들의 eval 판수")
    print(f"{'instruction':<{w}} {'s':>2} {'가능j':>6}{'가능판':>7}{'전체j':>6}{'전체판':>7}"
          f"{'p하한중앙':>10}")
    for nok, nokf, ins_, s_, nall, nallf, pmed in rank:
        print(f"{ins_:<{w}} {s_:>2} {nok:>6}{nokf:>7}{nall:>6}{nallf:>7}{pmed:>10.4f}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("instruction_key\tscene\tjitter\tn_ep_pool_succ\tn_ep_pool_fail"
                    "\tn_ep_tgt_fail\tn_ep_excluded_succ_tgt\tn_mixed_jitter\tn_pairs"
                    "\tperm_p_floor\tverdict\n")
            for r in out:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"[tsv] {args.tsv}")


if __name__ == "__main__":
    main()
