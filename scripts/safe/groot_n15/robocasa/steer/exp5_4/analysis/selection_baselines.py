#!/usr/bin/env python3
"""exp5-4 Step 1 — 자명 특징 baseline 대조 (Phase D).

학습 축(활성 mean-diff) 과 **완전히 같은 top-1 절차·같은 split** 으로 아래 점수들을 재고,
학습 축이 이들을 이기는지 본다.

  배포가능(deployable): act_norm_L*, a0_pos_norm, a0_full_norm, chunk_speed_mean,
                        chunk_tv(chunk 내부 total variation), chunk_jerk
  음성대조            : seed_only (모든 scene 에서 같은 순위 → 무효여야 정상)
  privileged oracle   : oracle_handle_cos (손잡이 좌표 + 초기 eef — 배포 불가)

설계 2종: in-fold K=8 (전체 후보) / prospective K=4 (gate 와 동일한 seed-out fold).
검정 (★2026-07-28 정정 — Codex Gate2 리뷰)
  · baseline 은 라벨과 무관한 **고정 선택자** → Poisson-binomial DP exact 유효(pb_valid=true).
  · **학습 축(LOSO)** 은 방향이 다른 scene 의 라벨에 의존해 scene 별 적중이 독립이 아니므로
    PB 는 부적합 → 라벨셔플 재fit / seed column 공통순열 순열검정을 primary 로 쓴다.
  · chunk 통계량은 csv(=replan 별 첫 action 1행) 가 아니라 **pkl actions[0] (16-step chunk)** 에서 계산.

주의: 초기 Gaussian noise 재구성 baseline 은 **불가** — 근거는 --help 및 출력 note 참조.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sel_common import (load, to_matrix, loso_select, score_select,  # noqa: E402
                         delta_stat, pb_pvalue, seed_permutations)
from _sel_baselines import (activation_norm_scores, seed_only_scores,  # noqa: E402
                            pkl_pass, load_handle_tsv, DEPLOYABLE)

NOISE_NOTE = (
    "초기 Gaussian noise 재구성 baseline 미구현(불가). 근거: scripts/serve/lerobot.py "
    "_apply_inference_seed() 가 torch.manual_seed + cuda.manual_seed_all 로 시드하고 "
    "노이즈는 action head 의 flow-matching 루프에서 **GPU 생성기**로 뽑힌다. "
    "eval 은 step 마다 seed = base + ep*max_steps + step 로 재시드하므로 "
    "(lerobot_http_eval.py:203) 텐서 shape/dtype/호출 횟수를 모델 forward 없이 알 수 없고, "
    "CPU 생성기 randn 은 실제 draw 와 다른 스트림이다. 시드 정수만으로 만드는 스칼라는 "
    "seed_only 음성대조와 동치.")


def seedperm_p(select_fn, Y, perms, obs_delta):
    null = np.array([delta_stat(select_fn(Y[:, list(pi)]))["delta"] for pi in perms])
    return float((np.sum(null >= obs_delta) + 1) / (len(perms) + 1)), null


def eval_scores(name, sc, Y, folds, ascending, perms):
    rows, deltas = [], []
    for fname, test_mask in folds:
        r = score_select(sc, Y, eval_seeds=test_mask, ascending=ascending)
        st = delta_stat(r)
        v = r["valid"]
        p, _ = pb_pvalue(r["base"][v].astype(float), int(r["top1"][v].sum()))
        psp, _n = seedperm_p(lambda Yp: score_select(sc, Yp, eval_seeds=test_mask,
                                                     ascending=ascending),
                             Y, perms, st["delta"])
        rows.append(dict(fold=fname, **st, p_exact_pb=float(p), p_seedperm=psp))
        deltas.append(st["delta"])
    return dict(name=name, ascending=ascending, delta_pooled=float(np.mean(deltas)),
                folds=rows, n_scene_min=int(min(r["n_scene"] for r in rows)))


def main():
    ap = argparse.ArgumentParser(epilog=NOISE_NOTE)
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cells", default="pq3_drawer_right,pq3_ppcc_beer,exp41_mixer")
    ap.add_argument("--layers", default="0", help="학습축·act_norm 을 볼 layer")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--with-oracle", action="store_true",
                    help="손잡이 기하 oracle (drawer 전용, scene 당 pkl 1개 로드)")
    ap.add_argument("--handle-tsv", default="/home/kimseungjun/exp53_analysis/handle_all.tsv")
    ap.add_argument("--n-perm-label", type=int, default=2000)
    ap.add_argument("--pkl-cache-dir", default="/home/kimseungjun/exp54_results/pkl_cache")
    ap.add_argument("--n-perm-seed", type=int, default=40320,
                    help="seed column 공통 순열 (8!=40320 이면 전수)")
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--out", default="/home/kimseungjun/exp54_results/selection_baselines.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    want_layers = [int(x) for x in a.layers.split(",")]
    out = dict(config=vars(a), noise_baseline_note=NOISE_NOTE, cells={})

    for cell in a.cells.split(","):
        eps, layers = load(Path(a.npz_dir), cell)
        if not eps:
            print(f"[skip] {cell}: npz 없음")
            continue
        _A0, Y, scenes, seeds, E = to_matrix(eps, 0, W=a.window)
        S, J = Y.shape
        half = J // 2
        infold = [("infold", np.ones(J, bool))]
        prosp = [("fold1", np.arange(J) >= half), ("fold2", np.arange(J) < half)]
        # fold 이름은 test seed 집합 기준: fold1 = 뒤 4 seed 가 후보(train=앞 4)
        perms, exhaustive = seed_permutations(J, a.n_perm_seed, rng)
        print(f"\n{'='*96}\n{cell}: {S} scene × {J} draw · SR {Y.mean():.3f}"
              f" · seed 순열 {len(perms)}개({'전수' if exhaustive else '부분표본'})")
        cell_out = dict(n_scene=S, n_seed=J, sr_all=float(Y.mean()),
                        infold={}, prospective={})

        scores, meta = {}, {}
        geom = (load_handle_tsv(a.handle_tsv)
                if (a.with_oracle and Path(a.handle_tsv).expanduser().exists()) else None)
        act, oracle, diag = pkl_pass(cell, E, scenes, geom=geom,
                                     cache=f"{a.pkl_cache_dir}/{cell}_chunk.npz")
        print(f"  pkl chunk: {diag['n_ok']}/{diag['n_total']}판 (결측 {diag['n_miss']})"
              f" · 출처 {diag['source']}")
        cell_out["pkl_note"] = {k: v for k, v in diag.items() if k != "cached"}
        scores.update(act)
        scores["seed_only"] = seed_only_scores(seeds, S)
        for L in want_layers:
            li = layers.index(L)
            A, *_ = to_matrix(eps, li, W=a.window)
            scores[f"act_norm_L{L}"] = activation_norm_scores(A)
            meta[f"learned_L{L}"] = A

        if geom is not None and np.isfinite(oracle).any():
            scores["oracle_handle_cos"] = oracle
            mc = diag.get("mean_oracle_cos")
            print("  oracle(privileged): 평균 cos(chunk 첫 step, 손잡이방향) "
                  + (f"{mc:.3f}" if mc is not None else "n/a"))

        for label, folds in (("infold", infold), ("prospective", prosp)):
            rows = []
            # 학습 축
            for L in want_layers:
                A = meta[f"learned_L{L}"]
                dd, fr = [], []
                for fname, test_mask in folds:
                    fit_mask = np.ones(J, bool) if label == "infold" else ~test_mask
                    r = loso_select(A, Y, fit_seeds=fit_mask, eval_seeds=test_mask)
                    st = delta_stat(r)
                    v = r["valid"]
                    p, _ = pb_pvalue(r["base"][v].astype(float), int(r["top1"][v].sum()))
                    sel = (lambda Yp, fm=fit_mask, te=test_mask:
                           loso_select(A, Yp, fit_seeds=fm, eval_seeds=te))
                    psp, _n = seedperm_p(sel, Y, perms, st["delta"])
                    nullL = np.array([delta_stat(sel(np.stack([rng.permutation(r0)
                                                               for r0 in Y])))["delta"]
                                      for _ in range(a.n_perm_label)])
                    plab = float((np.sum(nullL >= st["delta"]) + 1) / (a.n_perm_label + 1))
                    fr.append(dict(fold=fname, **st, p_labelshuffle=plab, p_seedperm=psp,
                                   p_exact_pb_reference_only=float(p), pb_valid=False))
                    dd.append(st["delta"])
                rows.append(dict(name=f"학습축(mean-diff) L{L}", ascending=True,
                                 delta_pooled=float(np.mean(dd)), folds=fr,
                                 kind="learned"))
            for bname, sc in sorted(scores.items()):
                for asc in (True, False):
                    e = eval_scores(bname, sc, Y, folds, asc, perms)
                    e["kind"] = ("negative_control" if bname == "seed_only" else
                                 "oracle" if bname.startswith("oracle") else
                                 "deployable" if (bname in DEPLOYABLE
                                                  or bname.startswith("act_norm")) else "other")
                    rows.append(e)
            cell_out[label] = rows

            print(f"\n  === {label} ({'K=8 전체후보' if label=='infold' else 'K=4 seed-out'}) ===")
            print(f"  {'점수':28} {'방향':10} {'구분':16} {'Δ̂':>8} {'top1':>7} "
                  f"{'base':>7} {'p_exact(fold)':>22}")
            for e in sorted(rows, key=lambda r: -r["delta_pooled"]):
                d = e["folds"]
                print(f"  {e['name']:28} {'낮은쪽' if e['ascending'] else '높은쪽':10} "
                      f"{e.get('kind','learned'):16} {e['delta_pooled']:+8.3f} "
                      f"{np.mean([x['sr_top1'] for x in d]):7.3f} "
                      f"{np.mean([x['sr_base'] for x in d]):7.3f} "
                      + " ".join(f"{('lab%.3f' % x['p_labelshuffle']) if 'p_labelshuffle' in x else ('PB%.3f' % x['p_exact_pb'])}"
                                 f"/sp{x['p_seedperm']:.3f}" for x in d))

            best_l = max((e for e in rows if e.get("kind") == "learned"),
                         key=lambda e: e["delta_pooled"], default=None)
            best_d = max((e for e in rows if e.get("kind") == "deployable"
                          and e.get("n_scene_min", S) == S),
                         key=lambda e: e["delta_pooled"], default=None)
            if best_l and best_d:
                win = best_l["delta_pooled"] > best_d["delta_pooled"]
                cell_out[f"{label}_verdict"] = dict(
                    learned=best_l["name"], learned_delta=best_l["delta_pooled"],
                    best_deployable=f"{best_d['name']}({'낮은쪽' if best_d['ascending'] else '높은쪽'})",
                    best_deployable_delta=best_d["delta_pooled"], learned_wins=bool(win))
                print(f"  ★ 학습축 {best_l['delta_pooled']:+.3f} vs 최강 배포가능 baseline "
                      f"{best_d['name']} {best_d['delta_pooled']:+.3f} → "
                      f"{'학습축 우세' if win else 'activation 고유 기여 근거 없음'}")
        out["cells"][cell] = cell_out

    print(f"\n[note] {NOISE_NOTE}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
