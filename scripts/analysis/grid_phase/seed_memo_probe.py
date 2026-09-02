#!/usr/bin/env python
"""seed(scene) 암기 진단 — "초기 succ/fail 분리는 실패 잦은 seed 를 외운 것 아닌가".

grid_phase 분리도(`phase_sep_matrix.py`)가 내는 phase 조건부 AUROC 는 scene 을 fold 축으로
LOSO 를 돌지만, **최종 AUROC 는 scene 을 가로질러 pooling** 한다. 어떤 scene 이 구조적으로
실패율이 높고 activation 이 그 scene 을 그냥 알아본다면, episode 내용에 succ/fail 신호가
0 이어도 pooled AUROC 는 올라간다. 이 스크립트는 그 가설을 네 숫자로 가른다.

한 cell (`--shard/--layer/--seg/--phase`) 당:
  1. **activation LOSO AUROC** — 엔진(`phase_sep_matrix.eval_cell`)과 같은 계산. 대조 기준.
  2. **scene-SR 베이스라인 AUROC** — episode 점수 = "자기를 뺀 같은 scene 의 성공률"
     (leave-one-episode-out). activation 을 전혀 안 쓰고 scene 정체성만으로 얻는 AUROC =
     "이미 아는 seed 반응" 검출기의 상한. 1 번이 이 값 근처면 scene 암기로 설명된다.
  3. **within-scene(층화) AUROC** — 1 번의 LOSO 점수를 쓰되 **같은 scene 안의 succ/fail 쌍만**
     비교. Σ_scene(일치 쌍)/Σ_scene(쌍 수). scene 정체성은 쌍 안에서 상수라 scene 암기만으로는
     0.5 를 못 넘는다. + scene **내부** 라벨 순열 null (같은 층화 통계) → z, p.
  4. **fixed t=0 within-scene AUROC** — phase 진입 **첫 record** 단독 점수의 층화 AUROC
     (조기성: phase 시작 순간에 이미 갈리는가).

`--all-noise-check <segA dir>`: 별도 모드. 수집 격자의 noise_idx 가 성공률에 주효과를 갖는지
(즉 noise 축이 사실상 또 하나의 scene 축인지) 를 exp5-4 식 column 순열로 검정한다.
관측 통계 = noise_idx 별 풀링 SR 의 분산, null = 각 (instruction, scene) 블록 안에서 noise
라벨 순열.

판정 코어(auroc/within_dir/loso/perm_null)와 shard 로더·equal-budget cell 구성은 새로 쓰지
않고 `phase_sep_matrix.py` / `g2_residual_read.py` 에서 파일경로 import 로 그대로 가져온다.
`loso_scores()` 만 `g2_residual_read.loso()` 의 점수 루프를 그대로 옮겨 놓았다 — 원본은 AUROC
스칼라만 돌려주는데 층화 통계에는 row 별 점수가 필요하기 때문. (자기검증에서 이 점수를
pooled AUROC 로 되돌리면 원본 `loso()` 와 일치하는지 assert 한다.)

사용 예
    python seed_memo_probe.py --shard .../OvenRack_out.npz --layer 2 --seg future \
        --phase reach-to-rack --n-perm 500 --out-dir outputs/analysis/grid_phase/seed_probe
    python seed_memo_probe.py --all-noise-check .../segA --n-perm 1000
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_F = Path(__file__).resolve()
REPO = _F.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_mod(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"원본 모듈이 없다: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ENGINE = _load_mod(_F.parent / "phase_sep_matrix.py", "_phase_sep_matrix")
Shard = _ENGINE.Shard
phase_labels = _ENGINE.phase_labels
build_phase_cells = _ENGINE.build_phase_cells
auroc = _ENGINE.auroc
within_dir = _ENGINE.within_dir
loso = _ENGINE.loso


# ───────────────────────────────────────────── LOSO 점수 (원본 loso 의 점수 루프)
def loso_scores(V, y, sc):
    """`g2_residual_read.loso()` 의 점수 계산부를 그대로 — row 별 점수 [n] + 평가 mask.

    원본은 AUROC 만 반환한다. 층화(within-scene) 통계에는 row 점수가 필요해서 점수 배열과
    "혼재 scene ∧ 점수 유효" mask 를 같이 돌려준다. 계산 자체는 원본과 동일
    (within-scene 방향 + held-out scene 내부 중심화, 혼재 scene 만 평가).
    """
    sco = np.full(len(y), np.nan)
    for s in np.unique(sc):
        te = sc == s
        tr = ~te
        w = within_dir(V[tr], y[tr], sc[tr])
        if w is None:
            continue
        n = np.linalg.norm(w)
        if n == 0:
            continue
        Vte = V[te] - V[te].mean(0, keepdims=True)
        sco[te] = Vte @ (w / n)
    keep = np.zeros(len(y), bool)
    for s in np.unique(sc):
        m = sc == s
        if (y[m] == 1).sum() > 0 and (y[m] == 0).sum() > 0:
            keep |= m
    return sco, keep & ~np.isnan(sco)


def stratified_auroc(s, y, sc):
    """같은 scene 안의 (succ, fail) 쌍만 비교하는 층화 AUROC.

    Σ_scene [ #(s_succ > s_fail) + 0.5·#(tie) ] / Σ_scene [ n_succ·n_fail ].
    scene 정체성이 쌍 안에서 상수이므로 "scene 을 알아보는" 성분은 이 통계를 못 올린다.
    """
    s = np.asarray(s, np.float64)
    y = np.asarray(y)
    sc = np.asarray(sc)
    num = den = 0.0
    per_scene = []
    for sid in np.unique(sc):
        m = sc == sid
        a = s[m & (y == 1)]
        b = s[m & (y == 0)]
        if len(a) == 0 or len(b) == 0:
            continue
        d = a[:, None] - b[None, :]
        win = float((d > 0).sum()) + 0.5 * float((d == 0).sum())
        npair = float(len(a) * len(b))
        num += win
        den += npair
        per_scene.append({"scene": int(sid), "n_succ": int(len(a)), "n_fail": int(len(b)),
                          "auroc": win / npair})
    if den == 0:
        return float("nan"), 0, per_scene
    return num / den, int(den), per_scene


def scene_sr_baseline(y, sc):
    """episode 점수 = 자기를 뺀 같은 scene 의 성공률(leave-one-episode-out).

    activation 을 전혀 안 쓴다. y=1(성공) 을 높게 주므로 AUROC>0.5 방향이 "scene 정체성이
    라벨을 예측한다" 를 뜻한다. 이 값이 activation AUROC 의 사실상 상한 역할.
    """
    y = np.asarray(y, np.float64)
    sc = np.asarray(sc)
    out = np.full(len(y), np.nan)
    for sid in np.unique(sc):
        idx = np.where(sc == sid)[0]
        if len(idx) < 2:
            continue
        tot = y[idx].sum()
        out[idx] = (tot - y[idx]) / (len(idx) - 1)
    return out


# ─────────────────────────────────────────────────────────────── cell 진단 본체
def probe_cell(V, cell, n_perm: int, seed: int) -> dict:
    y, sc = cell.y, cell.sc

    # (1) activation LOSO — 엔진과 동일
    M = cell.pool(V)
    a_ref, n_eval = loso(M, y, sc)
    s_act, keep = loso_scores(M, y, sc)
    a_pool = auroc(s_act[keep], y[keep])
    if not (np.isnan(a_ref) and np.isnan(a_pool)):
        assert abs(float(a_ref) - float(a_pool)) < 1e-9, (a_ref, a_pool)  # 원본과 일치 확인

    # (3) within-scene 층화
    a_ws, n_pair, per_scene = stratified_auroc(s_act, y, sc)

    # (4) fixed t=0
    M0 = cell.pool(V, t=0)
    a0_ref, _ = loso(M0, y, sc)
    s0, keep0 = loso_scores(M0, y, sc)
    a0_ws, n_pair0, _ = stratified_auroc(s0, y, sc)

    # (2) scene-SR 베이스라인 — activation 평가 대상(혼재 scene) 과 전체 둘 다
    sr = scene_sr_baseline(y, sc)
    ok = ~np.isnan(sr)
    a_sr_all = auroc(sr[ok], y[ok])
    a_sr_eval = auroc(sr[keep], y[keep]) if keep.any() else float("nan")

    # 순열 null (scene 내부 라벨 순열) — pooled·within-scene·t0 을 한 루프에서
    rng = np.random.default_rng(seed)
    nul_pool, nul_ws, nul_ws0 = [], [], []
    for _ in range(max(int(n_perm), 0)):
        yp = y.copy()
        for sid in np.unique(sc):
            idx = np.where(sc == sid)[0]
            yp[idx] = rng.permutation(y[idx])
        sp, kp = loso_scores(M, yp, sc)
        v = auroc(sp[kp], yp[kp]) if kp.any() else np.nan
        if not np.isnan(v):
            nul_pool.append(float(v))
        w, _, _ = stratified_auroc(sp, yp, sc)
        if not np.isnan(w):
            nul_ws.append(float(w))
        sp0, _ = loso_scores(M0, yp, sc)
        w0, _, _ = stratified_auroc(sp0, yp, sc)
        if not np.isnan(w0):
            nul_ws0.append(float(w0))

    def _zp(obs, nul):
        nul = np.asarray(nul, np.float64)
        if np.isnan(obs) or len(nul) == 0:
            return {"null_mean": None, "null_sd": None, "null_z": None, "p": None,
                    "n_perm": int(len(nul))}
        sd = float(nul.std())
        return {"null_mean": float(nul.mean()), "null_sd": sd,
                "null_z": float((obs - nul.mean()) / sd) if sd > 0 else None,
                "p": float((nul >= obs).mean()), "n_perm": int(len(nul))}

    nz = int((y == 1).sum())
    return {
        "n_ep": int(len(y)), "n_succ": nz, "n_fail": int(len(y) - nz),
        "n_scene": int(len(np.unique(sc))),
        "n_mixed_scene": int(cell.n_mixed_scenes), "budget_B": int(cell.budget),
        "activation_loso": {"auroc": None if np.isnan(a_ref) else float(a_ref),
                            "n_eval": int(n_eval), **_zp(a_ref, nul_pool)},
        "scene_sr_baseline": {"auroc_all": None if np.isnan(a_sr_all) else float(a_sr_all),
                              "auroc_eval_subset": (None if np.isnan(a_sr_eval)
                                                    else float(a_sr_eval)),
                              "note": "activation 미사용 — scene 정체성만의 판별력(상한)"},
        "within_scene": {"auroc": None if np.isnan(a_ws) else float(a_ws),
                         "n_pair": n_pair, **_zp(a_ws, nul_ws),
                         "per_scene": per_scene},
        "fixed_t0_within_scene": {"auroc": None if np.isnan(a0_ws) else float(a0_ws),
                                  "n_pair": n_pair0,
                                  "pooled_auroc": None if np.isnan(a0_ref) else float(a0_ref),
                                  **_zp(a0_ws, nul_ws0)},
        "length_auroc": (lambda v: None if np.isnan(v) else float(v))(
            auroc(cell.counts.astype(np.float64), y)),
    }


# ───────────────────────────────────────────────────────── noise 주효과 (독립 모드)
def load_ep_table(path: Path) -> dict:
    """X 를 건드리지 않고 라벨만 (npz 는 key 별 lazy 로딩) → episode 단위 표."""
    z = np.load(path, allow_pickle=False)
    ep = np.asarray(z["ep_id"]).astype(np.int64).ravel()
    sc = np.asarray(z["scene"]).astype(np.int64).ravel()
    no = np.asarray(z["noise"]).astype(np.int64).ravel()
    su = np.asarray(z["succ"]).astype(np.int64).ravel()
    meta = z["meta_json"]
    meta = json.loads(meta.item() if hasattr(meta, "item") else str(meta))
    _, first = np.unique(ep, return_index=True)
    first = np.sort(first)
    return {"instruction": str(meta.get("instruction", path.stem)), "slug": path.stem,
            "ep": ep[first], "scene": sc[first], "noise": no[first], "succ": su[first]}


def noise_main_effect(tables: list[dict], n_perm: int, seed: int) -> dict:
    instr = np.concatenate([np.full(len(t["ep"]), i) for i, t in enumerate(tables)])
    scene = np.concatenate([t["scene"] for t in tables])
    noise = np.concatenate([t["noise"] for t in tables])
    succ = np.concatenate([t["succ"] for t in tables]).astype(np.float64)
    noise_vals = np.unique(noise)

    def stat(y):
        sr = np.array([y[noise == nv].mean() if (noise == nv).any() else np.nan
                       for nv in noise_vals], np.float64)
        return float(np.nanvar(sr)), float(np.nanmax(sr) - np.nanmin(sr)), sr

    var_obs, rng_obs, sr_obs = stat(succ)

    # null: (instruction, scene) 블록 안에서 noise 라벨 순열 == succ 를 블록 내 재배치
    blocks = []
    for i in np.unique(instr):
        for s in np.unique(scene[instr == i]):
            blocks.append(np.where((instr == i) & (scene == s))[0])
    rng = np.random.default_rng(seed)
    nv, nr = [], []
    for _ in range(int(n_perm)):
        yp = succ.copy()
        for b in blocks:
            yp[b] = rng.permutation(succ[b])
        v, r, _ = stat(yp)
        nv.append(v)
        nr.append(r)
    nv = np.asarray(nv)
    nr = np.asarray(nr)

    per_instr = []
    for i, t in enumerate(tables):
        row = {"instruction": t["instruction"], "slug": t["slug"], "n_ep": int(len(t["ep"])),
               "sr_overall": float(t["succ"].mean()),
               "sr_by_noise": {int(v): (float(t["succ"][t["noise"] == v].mean())
                                        if (t["noise"] == v).any() else None)
                               for v in noise_vals},
               "n_by_noise": {int(v): int((t["noise"] == v).sum()) for v in noise_vals}}
        per_instr.append(row)

    return {
        "noise_values": [int(v) for v in noise_vals],
        "pooled_sr_by_noise": {int(v): (None if np.isnan(x) else float(x))
                               for v, x in zip(noise_vals, sr_obs)},
        "n_by_noise": {int(v): int((noise == v).sum()) for v in noise_vals},
        "stat_var_obs": var_obs, "stat_range_obs": rng_obs,
        "null_var_mean": float(nv.mean()), "null_var_sd": float(nv.std()),
        "p_var": float((nv >= var_obs).mean()), "p_range": float((nr >= rng_obs).mean()),
        "n_perm": int(n_perm), "n_block": len(blocks), "n_ep_total": int(len(succ)),
        "per_instruction": per_instr,
        "null_note": "(instruction, scene) 블록 내 noise 라벨 순열 — exp5-4 column 순열 방식",
    }


# ──────────────────────────────────────────────────────────────────────── 출력
def _f(v, nd=3):
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def print_cell(res: dict) -> None:
    c = res["cell"]
    r = res["result"]
    print(f"\n=== {c['instruction']} | L{c['layer']} | seg={c['seg']} | phase={c['phase']} "
          f"| denoise={c['denoise']} ===")
    print(f"  n_ep={r['n_ep']} (succ {r['n_succ']} / fail {r['n_fail']}), "
          f"scene {r['n_scene']} (혼재 {r['n_mixed_scene']}), budget_B={r['budget_B']}, "
          f"length_auroc={_f(r['length_auroc'])}")
    a = r["activation_loso"]
    b = r["scene_sr_baseline"]
    w = r["within_scene"]
    t0 = r["fixed_t0_within_scene"]
    print(f"  {'(1) activation LOSO AUROC':34s} {_f(a['auroc'])}  z={_f(a['null_z'],2)} "
          f"p={_f(a['p'],3)} n_eval={a['n_eval']}")
    print(f"  {'(2) scene-SR baseline AUROC':34s} {_f(b['auroc_eval_subset'])}  "
          f"(전체 {_f(b['auroc_all'])})  ← activation 미사용 상한")
    print(f"  {'(3) within-scene AUROC':34s} {_f(w['auroc'])}  z={_f(w['null_z'],2)} "
          f"p={_f(w['p'],3)} n_pair={w['n_pair']}")
    print(f"  {'(4) fixed t=0 within-scene AUROC':34s} {_f(t0['auroc'])}  z={_f(t0['null_z'],2)} "
          f"p={_f(t0['p'],3)} (pooled {_f(t0['pooled_auroc'])})")


def print_noise(res: dict) -> None:
    nvals = res["noise_values"]
    print("\n=== noise_idx 주효과 검정 (episode 단위) ===")
    print(f"  episode {res['n_ep_total']}, 블록(instruction×scene) {res['n_block']}, "
          f"n_perm={res['n_perm']}")
    hdr = "  " + f"{'instruction':26s}" + "".join(f"{('n' + str(v)):>7s}" for v in nvals) + \
          f"{'all':>7s}{'n':>6s}"
    print(hdr)
    for row in res["per_instruction"]:
        cells = "".join(_f(row["sr_by_noise"][v], 2).rjust(7) for v in nvals)
        print(f"  {row['instruction'][:26]:26s}{cells}{_f(row['sr_overall'],2):>7s}"
              f"{row['n_ep']:>6d}")
    pooled = "".join(_f(res["pooled_sr_by_noise"][v], 2).rjust(7) for v in nvals)
    print(f"  {'POOLED':26s}{pooled}")
    print(f"  var(pooled SR) obs={res['stat_var_obs']:.5f} "
          f"null={res['null_var_mean']:.5f}±{res['null_var_sd']:.5f} "
          f"p={res['p_var']:.3f} | range obs={res['stat_range_obs']:.3f} "
          f"p={res['p_range']:.3f}")


# ──────────────────────────────────────────────────────────────────────── main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", help="cell 모드: NPZ 경로")
    ap.add_argument("--layer", help="물리 capture layer 번호 (예: 2)")
    ap.add_argument("--seg", help="state|future|action|all")
    ap.add_argument("--phase", help="GT phase 이름 (shard phase_codebook)")
    ap.add_argument("--denoise", default="mean", choices=["mean", "0", "1", "2", "3"])
    ap.add_argument("--min-budget", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-noise-check", help="독립 모드: segA shards 디렉토리")
    ap.add_argument("--out-dir", default="outputs/analysis/grid_phase/seed_probe")
    ap.add_argument("--tag", default=None, help="출력 파일 이름 접미사")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_noise_check:
        d = Path(args.all_noise_check)
        paths = sorted(d.glob("*.npz")) if d.is_dir() else [d]
        if not paths:
            raise FileNotFoundError(f"shard 없음: {d}")
        tables = []
        for p in paths:
            t = load_ep_table(p)
            tables.append(t)
            print(f"[loaded] {p.name}: ep={len(t['ep'])} scene={len(np.unique(t['scene']))} "
                  f"noise={len(np.unique(t['noise']))} SR={t['succ'].mean():.3f}", flush=True)
        res = noise_main_effect(tables, args.n_perm, args.seed)
        res["shards"] = [p.name for p in paths]
        print_noise(res)
        fp = out_dir / f"noise_check{('_' + args.tag) if args.tag else ''}.json"
        fp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("[written]", fp, flush=True)
        return 0

    for req in ("shard", "layer", "seg", "phase"):
        if getattr(args, req) is None:
            ap.error("cell 모드에는 --shard/--layer/--seg/--phase 가 모두 필요하다 "
                     "(또는 --all-noise-check)")

    sh = Shard(Path(args.shard))
    if sh.tier != "A":
        raise ValueError(f"{sh.slug}: Tier A shard 만 지원 (tier={sh.tier})")
    code, def_name, names = phase_labels(sh, "gt", args.seed, {})
    keep = np.ones(sh.n_rec, bool)
    cells, skips = build_phase_cells(sh, code, names, keep, args.min_budget)
    hit = [c for c in cells if c.phase_name == args.phase]
    if not hit:
        avail = [c.phase_name for c in cells]
        raise SystemExit(f"phase '{args.phase}' 없음. 유효 cell: {avail} / skip: "
                         f"{[(s['phase'], s['skip_reason']) for s in skips]}")
    cell = hit[0]
    (lname, li), = sh.layer_indices(str(args.layer))
    (sname, si), = sh.seg_indices(args.seg)
    V = sh.slice_A(li, si, args.denoise)
    del sh.X                                  # 3~5GB fp16 즉시 해제 (공유 노드)
    res = {
        "cell": {"shard": Path(args.shard).name, "slug": sh.slug, "instruction": sh.instruction,
                 "layer": lname, "seg": sname, "denoise": args.denoise, "phase": cell.phase_name,
                 "phase_code": int(cell.phase_code), "phase_def": def_name,
                 "min_budget": args.min_budget, "n_perm": args.n_perm, "seed": args.seed},
        "result": probe_cell(V, cell, args.n_perm, args.seed),
    }
    print_cell(res)
    tag = args.tag or f"{sh.slug}_L{lname}_{sname}_{cell.phase_name}"
    fp = out_dir / f"cell_{tag}.json"
    fp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[written]", fp, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
