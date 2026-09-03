"""v5 LOKO 연산자 성립 가능성 감사 — activation 없이 index_v5 만으로 사전 판정.

`v5_fit_loko.py` 의 pool 계약(타 k 전판 + 대상 k 실패판, 대상 k 성공 제외)에서
**성공 클래스가 pool 에 몇 판 남는지**는 index 만으로 알 수 있다. 성공 pool 이 비거나
1 ep 이면 그 (instr, scene, k) 는 shard 를 뽑기 전에 이미 fit 불가다 — 추출·fit 을
돌리기 전에 eval 분모와 "미등록 예상 셀"을 확정하는 용도.

판정 기준 (fit 스크립트의 게이트에서 역산)
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

DEF_INDEX = "configs/collect/n15_grid_v5_scenario/index_rollouts_v5.tsv"
# 중추('전체 파이프라인') 확정 eval 대상 = instruction 당 balance scene 1개
DEF_CELLS = ("PPCC/bread:1,OpenDrawer/left:3,PPCC/marshmallow:3,PPCC/candle:3,"
             "PPCC/jug:4,DishwasherRack/out:4,OvenRack/out:4")


def verdict(n_pool_succ: int) -> str:
    if n_pool_succ == 0:
        return "불가(pool 성공 0)"
    if n_pool_succ == 1:
        return "불가(pool 성공 1ep — 공분산·게이트 불성립)"
    if n_pool_succ < 5:
        return "위험(pool 성공 <5ep — 게이트 CV 표본 부족)"
    return "가능"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEF_INDEX)
    ap.add_argument("--cells", default=DEF_CELLS)
    ap.add_argument("--tsv", default=None, help="지정 시 결과를 TSV 로도 저장")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.index), delimiter="\t"))
    cells = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1]))
             for c in args.cells.split(",") if c.strip()]

    out = []
    for ins, s in cells:
        sub = [r for r in rows if r["grid_instruction"] == ins and int(r["scene_idx"]) == s]
        if not sub:
            raise SystemExit(f"index 에 {ins} s{s} 행이 없다 — cells 인자 확인")
        for k in sorted({r["jitter_reset_idx"] for r in sub}, key=int):
            tgt_f = [r for r in sub if r["jitter_reset_idx"] == k and r["success"] != "1"]
            if not tgt_f:
                continue          # 대상 k 는 "실패가 있는 k" 뿐
            pool_s = [r for r in sub if r["jitter_reset_idx"] != k and r["success"] == "1"]
            pool_f = [r for r in sub if r["success"] != "1"]
            excl = [r for r in sub if r["jitter_reset_idx"] == k and r["success"] == "1"]
            out.append((ins, s, int(k), len(pool_s), len(pool_f), len(tgt_f), len(excl),
                        verdict(len(pool_s))))

    w = max(len(r[0]) for r in out)
    print(f"{'instruction':<{w}} {'s':>2} {'k':>3} {'poolS':>6}{'poolF':>6}{'tgtF':>5}"
          f"{'exclS':>6}  판정")
    print("-" * (w + 40))
    agg = collections.Counter()
    for ins, s, k, ps, pf, tf, ex, v in out:
        agg[v.split("(")[0]] += 1
        print(f"{ins:<{w}} {s:>2} {k:>3} {ps:>6}{pf:>6}{tf:>5}{ex:>6}  {v}")
    print("-" * (w + 40))
    print(f"대상 k-셀 {len(out)} | " + " ".join(f"{k} {v}" for k, v in sorted(agg.items())))
    print(f"eval 판수(대상 k 실패 합) {sum(r[5] for r in out)} | "
          f"그중 '불가' 셀 판수 {sum(r[5] for r in out if r[7].startswith('불가'))}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("grid_instruction\tscene\tk\tn_ep_pool_succ\tn_ep_pool_fail"
                    "\tn_ep_tgt_fail\tn_ep_excluded_succ_tgt\tverdict\n")
            for r in out:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"[tsv] {args.tsv}")


if __name__ == "__main__":
    main()
