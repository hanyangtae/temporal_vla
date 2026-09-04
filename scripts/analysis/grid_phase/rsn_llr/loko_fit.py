"""LOKO fit — scene-local 계약: (instruction 키, scene, 대상 지터)당 연산자 1개.

pool = 대상 scene의 타 지터 전판(succ+fail) + 대상 지터 실패판 (대상 지터 성공판 제외
= success-blind). 대상 = 그 scene에서 실패 판이 있는 지터 전부(자동 산출) —
leave-one-jitter-out 을 지터마다 반복.

⚠ **좌표 축 이름**: v6 부터 좌표는 `jitter_idx`(=j, plan `jitters` 순서)이고
`jitter_reset_idx` 는 **출처 열**(연속 reset 횟수)이다 — oven/washer 키는 전부 0 이라
그걸 좌표로 읽으면 5개 지터가 한 값으로 뭉개진다. 이 스크립트는 shard 에서 좌표 열을
`jitter_idx` → `jitter` 순으로만 찾고, 없으면 fail-loud (`--axis-col` 로 명시 가능).
v5 이하(k 층)는 `--axis-col jitter --coord k --tag v5`.

데이터 = segA shard(slug별 NPZ) + ae_bundle_*_k8.npz(encoder+centers).
ck8 라벨은 번들 centers 최근접으로 파생(assign 식 = 온라인 cluster_phase와 동일) —
labels npz가 있으면 파생 라벨과 전행 일치를 검증한다(불일치 = 번들≠라벨 세계, fail-loud).

usage (승준, ~/anaconda3/bin/python — torch 필요, scipy 불요):
  python v5_fit_loko.py <slug> <scene:int> [--seg-dir D] [--bundle F] [--labels F] [--out ROOT]

산출 (ROOT 기본 = ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/):
  A. instr_setm_<tag>_gt/<slug>/s<i>/<coord><r>/<phase>/dit_L12/conceptors.npz (+ metadata.json)
     — 기본형: v = normalize(μ_s − μ_f), s = μ_s·v (pool 성공이 같은 scene이라
       v4sb/v4r 하이브리드 setpoint 불요 — scene 도메인 offset 소거됨)
  B. instr_setm_<tag>_ck8/<slug>/s<i>/<coord><r>/c<j>/dit_L12/conceptors.npz — phase = ck8
  C. rsn_llr_reg_<tag>[_all]/<slug>/s<i>/<coord><r>.npz — LLR 번들, 계약 = llr_scorer.py
     docstring(단일 출처). entry "s<i>__c<j>", 앵커 = pool record cluster 평균,
     게이트 = **k-matched** ep-LOO AUROC(아래), ood_lo = 대상 k 실패 후반절반
     max(logN) 5pct (scorer 스케일 — 2π 상수 포함 필수).

★ 게이트가 k-matched 인 이유 (2026-09-03 실측으로 설계 변경)
  이 pool 은 성공이 **타 k 에서만** 오므로 succ/fail 라벨이 지터 k 와 강하게 상관된다.
  index 만으로 잰 "지터 정체성 단독" ep-AUROC 가 v5 대상 26 셀에서 **중앙값 0.835**
  (PPCC/marshmallow s3 = 0.99~1.00, PPCC/jug s4 = 0.95~0.98) — 즉 pool 전체 AUROC 로
  등록하면 **활성화에서 outcome 이 아니라 k 를 읽는 판별기가 그대로 통과한다.**
  그런데 이 채점기의 용도는 *같은 episode·같은 시점*에서 갈라진 best-of-N 후보의 순위
  매기기라 후보들의 k 는 전부 동일 — k 를 읽는 성분은 후보 순위에 원리적으로 기여할 수
  없다(길이가 후보를 순위 매길 수 없는 것과 같은 범주). 따라서 등록 기준은
  **succ·fail 이 공존하는 k 안의 (실패, 성공) 쌍만 센 concordance**(k-계층화 AUROC)이고,
  pool 전체·길이·k단독 기준선은 aux 로 병기만 한다. k 마다 AUROC 를 따로 내는 방식은
  한 클래스가 1 ep 인 k(marsh s3 k4 = 1승 4패)에서 정의되지 않아 검정력이 0이 됐다
  (합성 양성 대조가 기각됨) — 쌍 단위 합산으로 교정하고 쌍 수 하한을 둔다.
  채점은 ep 단위 leave-one-out(자기 episode 제외 fit) — 셀당 성공 6~15 ep 규모라
  5-seed 60% 분할보다 표본 효율이 높다.
"""
import argparse
import hashlib
import json
import math
import os

import numpy as np

DEF_ROOT = "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/"
CONST16 = 0.5 * 16 * math.log(2 * math.pi)
LAYER = 12       # DiT block residual layer (capture index로 역산)
DENOISE = 3      # 마지막 denoise call
MIN_REC_SETM = 20
MIN_REC_LLR = 60
GATE_SEEDS = 5
GATE_MED = 0.70
GATE_EACH = 0.65     # (per-k 참고 표기용, 등록 판정엔 미사용)
GATE_MIN_PAIRS = 6   # k-계층화 쌍 수 하한


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("scene", type=int)
    ap.add_argument("--seg-dir", default=None, help="기본 ROOT/segA")
    ap.add_argument("--bundle", default=None, help="기본 ROOT/ae_k8/ae_bundle_k8.npz")
    ap.add_argument("--labels", default=None,
                    help="기본 ROOT/ae_k8/labels_<slug>_k8.npz (있으면 대조, 없으면 생략)")
    ap.add_argument("--setm-form", choices=("jfair", "plain", "both"), default="both",
                    help="setM 방향 산출. jfair(기본)=지터 안 대조 평균+대상 지터 앵커 "
                         "하이브리드 setpoint / plain=pool 전체 mean-diff(지터 혼입 미보정, "
                         "비교용) / both(기본)=둘 다 산출, plain 은 루트 뒤에 _plain. "
                         "지터가 물리 오프셋인 키(oven·washer)에서 특히 중요")
    ap.add_argument("--jitters", default=None,
                    help="대상 지터를 쉼표로 한정 (예 0,2,3). 기본은 실패가 있는 지터 전부. "
                         "중추 셀 tsv(v6_loko_cells.tsv)로 선정된 j 만 돌릴 때 쓴다")
    ap.add_argument("--tag", default="v6", help="산출 루트 태그 (instr_setm_<tag>_*, rsn_llr_reg_<tag>)")
    ap.add_argument("--coord", default="j", help="좌표 폴더·registry 열 이름 (v6=j, v5=k)")
    ap.add_argument("--axis-col", default=None,
                    help="shard 의 좌표 열 이름 명시. 기본 자동(jitter_idx→jitter). "
                         "jitter_reset_idx 는 v6 에서 출처 열이라 자동 선택하지 않는다")
    ap.add_argument("--out", default=None, help="기본 ROOT")
    ap.add_argument("--only-gt", action="store_true",
                    help="setm_gt 만 산출 (AE 번들 불요 — 번들 지연 시 선행용; "
                         "번들 도착 후 전체 재실행하면 gt 는 동일 산출로 덮임)")
    # 아래 3개는 합성 스모크 전용 완화 옵션 — 본 fit 은 기본값 그대로 쓸 것
    ap.add_argument("--min-rec-llr", type=int, default=MIN_REC_LLR)
    ap.add_argument("--gate-med", type=float, default=GATE_MED)
    ap.add_argument("--gate-each", type=float, default=GATE_EACH)
    ap.add_argument("--phase-mode", choices=("ck8", "all"), default="ck8",
                    help="채점 entry 단위. ck8=cluster 별(기본) / all=phase 조건화 없이 "
                         "scene 전체 1 entry. cluster 조건화가 outcome 대조를 흡수하는지 "
                         "실측 비교용 arm (합성 대조에서 흡수 확인: 전체 0.93 → 클러스터별 "
                         "0.50~0.63). 산출 루트가 갈린다(rsn_llr_reg_<tag>_all).")
    ap.add_argument("--n-perm", type=int, default=2000, help="게이트 순열검정 반복")
    ap.add_argument("--gate-p", type=float, default=0.05, help="순열 p 상한")
    ap.add_argument("--gate-min-pairs", type=int, default=GATE_MIN_PAIRS,
                    help="k-계층화 쌍 수 하한 (기본 6 — 4쌍이면 우연 통과 확률 0.2)")
    return ap.parse_args()


def fmt_summary(row):
    c_, ns_, nf_, mk, ma, ml, bk, dg, npair, pp, _p = row
    return (f"{c_:<4} rec s/f {ns_}/{nf_}  k계층 {mk:.2f}({npair}쌍, p={pp:.3f})  "
            f"pool {ma:.2f} len {ml:.2f} k단독 {bk:.2f}  진단(tgt) {dg:.2f}")


def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gfit(Xc):
    mu = Xc.mean(0)
    Craw = np.cov(Xc.T)
    C = Craw + 1e-3 * np.trace(Craw) / Xc.shape[1] * np.eye(Xc.shape[1])
    return mu, np.linalg.inv(C), float(np.linalg.slogdet(C)[1]), Craw


def gll(Xc, p):
    D = Xc - p[0]
    return -0.5 * ((D @ p[1]) * D).sum(1) - 0.5 * p[2] - CONST16


def main():
    args = parse_args()
    root = os.path.expanduser(args.out or DEF_ROOT)
    seg_dir = os.path.expanduser(args.seg_dir or os.path.join(root, "segA"))
    bundle_p = os.path.expanduser(args.bundle or os.path.join(root, "ae_k8/ae_bundle_k8.npz"))
    labels_p = os.path.expanduser(args.labels or os.path.join(
        root, f"ae_k8/labels_{args.slug}_k8.npz"))
    SLUG, S = args.slug, args.scene

    # ── shard ──
    d = np.load(os.path.join(seg_dir, f"{SLUG}.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    if meta.get("jitter_axis") is False:
        raise SystemExit(f"{SLUG}: meta_json.jitter_axis=false — 지터 축 없는 shard")
    if args.axis_col:
        axis_col = args.axis_col
    else:
        cand = [c for c in ("jitter_idx", "jitter") if c in d.files]
        if not cand:
            raise SystemExit(
                f"{SLUG}: shard 에 좌표 열이 없다 (찾은 열 {sorted(d.files)}). "
                "v6 좌표는 jitter_idx — jitter_reset_idx 는 출처 열이라 자동 선택하지 "
                "않는다(oven/washer 는 전부 0). 필요하면 --axis-col 로 명시할 것")
        axis_col = cand[0]
    if axis_col == "jitter_reset_idx":
        print(f"[warn] 좌표 열로 jitter_reset_idx 를 쓴다 — v6 에서는 출처 열이며 "
              f"pull 계열은 전부 0 이라 지터가 뭉갤 수 있다. 의도한 것인지 확인할 것")
    cap = [int(x) for x in meta["capture_layers"]]
    seg = meta["segment_names"].index("all")
    X = d["X"][:, cap.index(LAYER), DENOISE, seg, :].astype(np.float32)
    pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]
    jit, eplen = d[axis_col], d["ep_len"]
    print(f"[axis] 좌표 열 = {axis_col} (고유값 {sorted(set(jit.tolist()))[:10]})")
    CB = {v: k for k, v in meta["phase_codebook"].items()}

    # ── AE 번들 → encode + ck8 라벨 파생 ──
    Z = ck = b = None
    ae_sig, ae_dir, partial_sig = "(only-gt: 번들 미사용)", "", ""
    if args.only_gt:
        print("[mode] --only-gt: setm_gt 만 산출 (ck8·LLR 생략)")
    if not args.only_gt:
        b = np.load(bundle_p, allow_pickle=False)
        arch = json.loads(str(b["arch"]))
        assert arch["input_dim"] == 1536 and arch["latent"] == 16, arch
        muA = b["mu"].astype(np.float32)
        sdA = float(b["scalar_std"])
        centers = np.asarray(b[f"centers.{SLUG}"], np.float64)
        import torch
        Ws = [(torch.from_numpy(np.asarray(b[f"enc.{k}.weight"], np.float32)),
               torch.from_numpy(np.asarray(b[f"enc.{k}.bias"], np.float32)))
              for k in ("net.0", "net.2", "head")]
        zs = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                t = torch.from_numpy((X[i:i + 8192] - muA) / sdA)
                t = torch.nn.functional.gelu(t @ Ws[0][0].T + Ws[0][1])   # exact erf GELU
                t = torch.nn.functional.gelu(t @ Ws[1][0].T + Ws[1][1])
                zs.append((t @ Ws[2][0].T + Ws[2][1]).numpy())
        Z = np.concatenate(zs).astype(np.float64)
        ck = np.concatenate([np.argmin(((Z[i:i + 8192, None, :] - centers[None]) ** 2).sum(-1), 1)
                             for i in range(0, len(Z), 8192)]).astype(np.int64)
        if os.path.exists(labels_p):
            lab = np.load(labels_p, allow_pickle=True)["cluster"]
            if len(lab) != len(ck):
                raise SystemExit(f"{SLUG}: labels 행수 {len(lab)} != shard {len(ck)}")
            mism = int((np.asarray(lab).astype(np.int64) != ck).sum())
            if mism:
                raise SystemExit(f"{SLUG}: 파생 ck8 라벨 불일치 {mism}행 — 번들과 labels 가 다른 세계")
            print(f"[labels] {SLUG}: 파생 라벨 == labels npz 전행 일치 ({len(ck)})")
        else:
            print(f"[labels] {SLUG}: labels npz 없음 → 번들 파생 라벨만 사용 ({labels_p})")
        ae_sig = hashlib.sha256(open(bundle_p, "rb").read()).hexdigest()[:16]
        ae_dir = os.path.basename(os.path.dirname(bundle_p))
        ps_path = os.path.join(os.path.dirname(bundle_p), "PARTIAL_SHARDS.txt")
        partial_sig = (hashlib.sha256(open(ps_path, "rb").read()).hexdigest()[:16]
                       if os.path.exists(ps_path) else "")
        if partial_sig:
            print(f"[ae] 임시 번들({ae_dir}) 사용 — 정식 번들 나오면 ck8 재fit 필요 "
                  f"(sig {ae_sig}, shards {partial_sig})")

    # ── scene 인덱스 ──
    m_sc = sc == S
    if not m_sc.any():
        raise SystemExit(f"{SLUG}: scene s{S} record 0")
    eps_s = np.unique(ep[m_sc])
    em = {}
    for e in eps_s:
        i0 = np.where(ep == e)[0]
        em[e] = (int(su[i0[0]]), int(jit[i0[0]]), float(eplen[i0[0]]), i0)
    ks_all = sorted(set(v[1] for v in em.values()))
    ks_fail = sorted(set(v[1] for v in em.values() if v[0] == 0))
    if args.jitters:
        want = [int(x) for x in args.jitters.split(",") if x.strip() != ""]
        miss = [j for j in want if j not in ks_fail]
        if miss:
            raise SystemExit(
                f"{SLUG} s{S}: --jitters 로 지정한 {miss} 는 실패 판이 없다"
                f" (실패 있는 지터 {ks_fail}) — 셀표와 shard 가 어긋난다")
        ks_fail = want
    if len(ks_all) < 2:
        raise SystemExit(
            f"{SLUG} s{S}: 좌표 열 {axis_col} 의 고유값이 {ks_all} 하나뿐 — LOKO 불가. "
            "v6 에서 jitter_reset_idx 를 좌표로 읽으면(oven/washer 전부 0) 이렇게 된다")
    n_s = sum(1 for v in em.values() if v[0] == 1)
    print(f"[scene] {SLUG} s{S}: ep {len(eps_s)} (succ {n_s} / fail {len(eps_s) - n_s}), "
          f"{args.coord} = {ks_all}, 실패 있는 {args.coord} = {ks_fail}")
    registry = []   # (k, n_reg, entries, pool succ/fail ep, tgt fail ep, excl succ ep)

    for k_tgt in ks_fail:
        allow = m_sc & ((jit != k_tgt) | (su == 0))
        ep_pool_s = [e for e in eps_s if em[e][0] == 1 and em[e][1] != k_tgt]
        ep_pool_f = [e for e in eps_s if em[e][0] == 0]
        ep_tgt_f = [e for e in eps_s if em[e][0] == 0 and em[e][1] == k_tgt]
        n_excl = sum(1 for e in eps_s if em[e][0] == 1 and em[e][1] == k_tgt)
        pool_note = {"n_ep_pool_succ": len(ep_pool_s), "n_ep_pool_fail": len(ep_pool_f),
                     "n_ep_tgt_fail": len(ep_tgt_f), "n_ep_excluded_succ_tgt": n_excl,
                     "pool": "scene-local LOKO: 타 k 전판 + 대상 k 실패판 (대상 k 성공 제외)"}

        # ── A/B. setM (gt / ck8) ──
        tags = [("gt", pc, lambda c: CB[int(c)])]
        if not args.only_gt:
            tags.append(("ck8", ck, lambda c: f"c{int(c)}"))
        forms = ["jfair", "plain"] if args.setm_form == "both" else [args.setm_form]
        for tag, la, name_of in tags:
            n_ph = n_skip = 0
            for code in sorted(set(la[allow].tolist())):
                ms = np.where(allow & (la == code) & (su == 1))[0]
                mf = np.where(allow & (la == code) & (su == 0))[0]
                if len(ms) < MIN_REC_SETM or len(mf) < MIN_REC_SETM:
                    continue
                mu_s = X[ms].mean(0).astype(np.float64)
                mu_f = X[mf].mean(0).astype(np.float64)
                mtf = np.where(allow & (la == code) & (su == 0) & (jit == k_tgt))[0]
                # ── 방향 v ──
                # ★ jfair(기본): **같은 지터 안의** 클래스 평균차를 지터마다 구해 평균한다.
                # pool 성공은 타 지터에서만 오므로 pool 전체 mean-diff 는 지터 정체성을
                # 그대로 먹는다 — v6 의 oven·washer 는 지터가 로봇 base 오프셋(5~15cm)
                # 이라, 그 방향으로 밀면 "성공 쪽"이 아니라 "로봇을 다른 위치로" 미는 셈이
                # 된다(물리 거울 판정과 같은 함정). 지터 안 대조는 그 성분이 상쇄된다.
                # setpoint 는 대상 지터 실패 평균 좌표 + 지터 안 평균 이동량(하이브리드) —
                # 절대 좌표가 지터마다 다르므로 pool 성공 좌표를 그대로 쓰면 안 된다.
                per_j, n_mix = [], 0
                for jj in ks_all:
                    mjs = np.where(allow & (la == code) & (su == 1) & (jit == jj))[0]
                    mjf = np.where(allow & (la == code) & (su == 0) & (jit == jj))[0]
                    if len(mjs) >= MIN_REC_SETM and len(mjf) >= MIN_REC_SETM:
                        per_j.append(X[mjs].mean(0).astype(np.float64)
                                     - X[mjf].mean(0).astype(np.float64))
                        n_mix += 1
                wrote = False
                for form in forms:
                    if form == "jfair":
                        if n_mix < 2 or len(mtf) < MIN_REC_SETM:
                            continue
                        delta = np.mean(per_j, 0)
                        v = delta / (np.linalg.norm(delta) + 1e-12)
                        s_val = (float(X[mtf].mean(0).astype(np.float64) @ v)
                                 + float(np.mean([d_ @ v for d_ in per_j])))
                        note = ("v=normalize(mean_j(mu_s,j − mu_f,j)) — 지터 안 대조 평균; "
                                "s=mu_f,tgt·v + mean_j(delta_j·v) (대상 지터 앵커 하이브리드)")
                    else:
                        delta = mu_s - mu_f
                        v = delta / (np.linalg.norm(delta) + 1e-12)
                        s_val = float(mu_s @ v)
                        note = ("v=normalize(mu_s−mu_f) (pool 전체 — 지터 혼입 미보정); "
                                "s=mu_s·v")
                    # 두 형태를 함께 낼 때만 부수 형태를 별도 루트로 분리한다
                    sfx = "" if form == forms[0] else f"_{form}"
                    d_out = os.path.join(root, f"instr_setm_{args.tag}_{tag}{sfx}", SLUG,
                                         f"s{S}", f"{args.coord}{k_tgt}",
                                         name_of(code), "dit_L12")
                    os.makedirs(d_out, exist_ok=True)
                    np.savez_compressed(os.path.join(d_out, "conceptors.npz"),
                                        alpha0_v_steer=v.astype(np.float32),
                                        alpha0_s=np.float32(s_val))
                    md = {"op": "setpoint",
                          "variant": f"instr_setm_{args.tag}_{tag}{sfx}",
                          "phase": name_of(code), "target_scene": S,
                          "target_jitter": k_tgt, "axis_col": axis_col,
                          "setm_form": form, "n_mixed_jitter": n_mix,
                          "n_rec_s": int(len(ms)), "n_rec_f": int(len(mf)),
                          "n_rec_tgt_fail": int(len(mtf)),
                          "cos_jfair_vs_pool": float(
                              (mu_s - mu_f) @ v / (np.linalg.norm(mu_s - mu_f) + 1e-12)),
                          **pool_note, "phase_label_source": tag, "layer": LAYER,
                          "note": note}
                    if tag == "ck8":
                        # AE 번들 의존 — 임시 번들(ae_k8_partial)로 만든 산출물은 정식
                        # 번들이 나오면 **재fit 대상**이다. 절대경로는 규약상 금지라
                        # 디렉토리 basename + 내용 지문으로 식별자를 남긴다(docs/04).
                        md.update({"ae_bundle_dir": ae_dir, "ae_bundle_sig": ae_sig,
                                   "ae_partial_shards_sig": partial_sig,
                                   "ae_is_partial": bool(partial_sig)})
                    with open(os.path.join(d_out, "metadata.json"), "w") as f:
                        json.dump(md, f, ensure_ascii=False, indent=1)
                    wrote = True
                if wrote:
                    n_ph += 1
                else:
                    n_skip += 1
            print(f"[setm_{tag}] {SLUG} s{S} {args.coord}{k_tgt}: {n_ph} phase 산출"
                  f"{f' (혼합 지터<2 로 {n_skip} phase 건너뜀)' if n_skip else ''} → {outroot}")

        # ── C. LLR 번들 ──
        if args.only_gt:
            continue
        arts, summary = {}, []
        groups = ([(f"c{c}", ck == c) for c in range(centers.shape[0])]
                  if args.phase_mode == "ck8"
                  else [("call", np.ones(len(Z), bool))])
        for gname, gmask in groups:
            m_c = allow & gmask
            if m_c.sum() < 10:
                continue
            # ★ 앵커는 **k-local**: 그 cluster 안에서 k 별 평균을 각각 빼서 fit 한다.
            # pool 전체 평균 하나로 중심화하면 가우시안이 k 오프셋을 학습해, 소수 클래스
            # episode(예: 실패 k 안의 유일한 성공)를 **체계적으로 역순** 매긴다 — 합성
            # 양성 대조에서 k계층 concordance 0.00 으로 실측됨(2026-09-03).
            # serve 는 succ_mean.<entry> 하나를 빼므로, 번들에는 **대상 k 의 앵커**를 싣는다
            # (대상 k 는 실패 record 로만 앵커가 잡힌다 — success-blind 계약과 일치).
            anchors_k = {}
            for kk in ks_all:
                mk_ = m_c & (jit == kk)
                if mk_.sum() >= 5:
                    anchors_k[kk] = Z[mk_].mean(0)
            if k_tgt not in anchors_k:
                summary.append((gname, 0, 0, float("nan"), float("nan"), float("nan"),
                                float("nan"), float("nan"), 0, float("nan"), False))
                continue
            anchor = anchors_k[k_tgt]

            def zc(idx):
                """record 인덱스 → k-local 중심화 latent (해당 k 앵커 감산)."""
                out = Z[idx].copy()
                for kk, a_ in anchors_k.items():
                    sel = jit[idx] == kk
                    if sel.any():
                        out[sel] -= a_
                return out
            ms = np.where(m_c & (su == 1) & np.isin(jit, list(anchors_k)))[0]
            mf = np.where(m_c & (su == 0) & np.isin(jit, list(anchors_k)))[0]
            if len(ms) < args.min_rec_llr or len(mf) < args.min_rec_llr:
                summary.append((gname, len(ms), len(mf), float("nan"), float("nan"),
                                float("nan"), float("nan"), float("nan"), 0,
                                float("nan"), False))
                continue
            # ── 게이트: ep 단위 leave-one-out LLR 을 **k 안에서** 채점 ──
            # ⚠ pool 은 성공이 타 k 에서만 오므로 succ/fail 이 k 와 상관된다. index 만으로
            # 재는 "k 정체성 단독" AUROC 가 이 셀들에서 중앙값 0.835(marsh 0.99~1.00)라,
            # pool 전체 AUROC 로 등록하면 **k 판독기가 통과한다**. 채점기는 같은 episode
            # 같은 시점의 후보들(=같은 k)을 순위 매기는 물건이라 k 를 읽는 성분은 무용 —
            # 그래서 등록 기준은 **succ·fail 이 공존하는 k 안에서만 잰 AUROC 의 중앙값**.
            # pool 전체·길이·k단독은 aux 로 병기만 한다(길이 규칙과 같은 취지).
            o_eps = [e for e in eps_s
                     if (em[e][0] == 0 or em[e][1] != k_tgt) and em[e][1] in anchors_k
                     and ((ep == e) & gmask & allow).sum() >= 4]
            rec_of = {e: np.where((ep == e) & gmask & allow)[0] for e in o_eps}
            n_s_ep = sum(1 for e in o_eps if em[e][0] == 1)
            n_f_ep = sum(1 for e in o_eps if em[e][0] == 0)
            loo = {}                       # ep -> mean LLR (자기 제외 fit)
            if n_s_ep >= 3 and n_f_ep >= 3:
                for e in o_eps:
                    tr = [x for x in o_eps if x != e]
                    # ⚠ 앵커(k-중심화)도 **held-out 을 빼고** 다시 잡아야 한다. 전체 record
                    # 로 잡은 앵커를 쓰면 held-out 이 자기 k 중심에 기여해 클래스와 상관된
                    # 누수가 생긴다 — 순수 노이즈 합성에서 concordance 0.73(AE-LLR)·
                    # 0.97(raw mean-diff)로 실측됨(2026-09-03). 반드시 LOO 안에서 계산.
                    anc_tr = {}
                    for kk in ks_all:
                        idxs = [rec_of[x] for x in tr if em[x][1] == kk]
                        if idxs:
                            cat = np.concatenate(idxs)
                            if len(cat) >= 5:
                                anc_tr[kk] = Z[cat].mean(0)
                    if em[e][1] not in anc_tr:
                        continue

                    def zc_tr(idx, _a=anc_tr):
                        out = Z[idx].copy()
                        for kk_, a_ in _a.items():
                            sel = jit[idx] == kk_
                            if sel.any():
                                out[sel] -= a_
                        return out
                    Xs_t = [zc_tr(rec_of[x]) for x in tr
                            if em[x][0] == 1 and em[x][1] in anc_tr]
                    Xf_t = [zc_tr(rec_of[x]) for x in tr
                            if em[x][0] == 0 and em[x][1] in anc_tr]
                    if len(Xs_t) < 2 or len(Xf_t) < 2:
                        continue
                    Xs_t, Xf_t = np.concatenate(Xs_t), np.concatenate(Xf_t)
                    if len(Xs_t) < 40 or len(Xf_t) < 40:
                        continue
                    gs_, gf_ = gfit(Xs_t), gfit(Xf_t)
                    Zb = zc_tr(rec_of[e])
                    loo[e] = float((gll(Zb, gf_) - gll(Zb, gs_)).mean())
            # 등록 기준 = k-계층화 AUROC: **같은 k 안의 (실패, 성공) 쌍만** 세어 일치율을
            # 낸다(= 조건부 concordance). k 마다 따로 AUROC 를 내면 한 클래스가 1 ep 인
            # k(marsh k4 = 1승4패)에서 추정이 정의되지 않아 검정력이 0이 된다 — 쌍 단위로
            # 합치면 그 4 쌍도 증거로 쓰인다. 쌍 수(n_pairs)를 함께 남겨 얇은 근거를 표시.
            by_k = {}
            for e, v_ in loo.items():
                by_k.setdefault(em[e][1], {0: [], 1: []})[em[e][0]].append((v_, em[e][2]))
            conc, n_pairs, ks_used = 0.0, 0, 0
            per_k = []
            for vv in by_k.values():
                if not vv[0] or not vv[1]:
                    continue
                hit = 0.0
                for vf, _ in vv[0]:
                    for vs, _ in vv[1]:
                        hit += 1.0 if vf > vs else (0.5 if vf == vs else 0.0)
                np_k = len(vv[0]) * len(vv[1])
                conc += hit
                n_pairs += np_k
                ks_used += 1
                per_k.append(hit / np_k)
            med_k = conc / n_pairs if n_pairs else float("nan")
            pooled = {0: [], 1: []}
            for e, v_ in loo.items():
                pooled[em[e][0]].append((v_, em[e][2]))
            med_a = auroc(np.array([x[0] for x in pooled[0]]),
                          np.array([x[0] for x in pooled[1]])) if loo else float("nan")
            med_l = auroc(np.array([x[1] for x in pooled[0]]),
                          np.array([x[1] for x in pooled[1]])) if loo else float("nan")
            # k 단독 기준선 (라벨 구성만으로 얻어지는 AUROC — 위약 하한)
            frate = {}
            for e in o_eps:
                frate.setdefault(em[e][1], []).append(em[e][0] == 0)
            frate = {kk: sum(v) / len(v) for kk, v in frate.items()}
            base_k = auroc(np.array([frate[em[e][1]] for e in o_eps if em[e][0] == 0]),
                           np.array([frate[em[e][1]] for e in o_eps if em[e][0] == 1]))
            # ── 순열검정(k 안 라벨 셔플) ──
            # 쌍 수는 24여도 실제 표본은 episode 10개 수준이라 쌍 단위 임계(0.70)는
            # 느슨하다 — 순수 노이즈 합성에서 5 셀 중 3 셀이 0.71~0.75 로 오등록됐다.
            # 그래서 절대 임계에 더해, **각 k 안에서 succ/fail 라벨만 섞은** 귀무분포
            # 대비 상위 5% 를 요구한다(라벨 구성·k 구조·표본 수를 그대로 보존하는 정확
            # 순열). 점수는 LOO 로 이미 고정돼 있어 재적합 없이 셔플만으로 계산한다.
            p_perm = float("nan")
            if n_pairs >= args.gate_min_pairs and not np.isnan(med_k):
                items = [(em[e][1], em[e][0], v_) for e, v_ in loo.items()]
                ks_list = sorted({it[0] for it in items})
                grouped = {kk: [(lab, v) for k2, lab, v in items if k2 == kk]
                           for kk in ks_list}
                rng_p = np.random.default_rng(0)
                ge = 0
                for _ in range(args.n_perm):
                    hit_p = n_p = 0.0
                    for kk, lst in grouped.items():
                        labs = np.array([x[0] for x in lst])
                        vals = np.array([x[1] for x in lst])
                        labs = rng_p.permutation(labs)
                        vf_ = vals[labs == 0]
                        vs_ = vals[labs == 1]
                        if not len(vf_) or not len(vs_):
                            continue
                        diff = vf_[:, None] - vs_[None, :]
                        hit_p += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
                        n_p += diff.size
                    if n_p and hit_p / n_p >= med_k - 1e-12:
                        ge += 1
                p_perm = (ge + 1) / (args.n_perm + 1)
            # 등록 = 절대 임계 + 쌍 수 하한 + 순열 p ≤ 0.05
            passed = (n_pairs >= args.gate_min_pairs and not np.isnan(med_k)
                      and med_k >= args.gate_med and p_perm <= args.gate_p)
            # 전체 fit + 대상 진단 (대상 k 실패 ep vs pool 성공 ep)
            gs, gf = gfit(zc(ms)), gfit(zc(mf))

            def ep_llr(e):
                m = np.where((ep == e) & gmask & allow)[0]
                if not len(m):
                    return None
                Zb = zc(m)
                return float((gll(Zb, gf) - gll(Zb, gs)).mean())
            dvals_f = [v for v in (ep_llr(e) for e in ep_tgt_f) if v is not None]
            dvals_s = [v for v in (ep_llr(e) for e in ep_pool_s) if v is not None]
            diag = auroc(np.array(dvals_f), np.array(dvals_s))
            summary.append((gname, len(ms), len(mf), med_k, med_a, med_l, base_k,
                            diag, n_pairs, p_perm, passed))
            if not passed:
                continue
            # ood_lo: 대상 k 실패 후반절반 (전 record — cluster 무관, v4 규약 유지)
            mx = []
            for e in ep_tgt_f:
                i0 = em[e][3]
                for i in i0[len(i0) // 2:]:
                    z = Z[i] - anchors_k[k_tgt]
                    mx.append(max(float(gll(z[None], gs)[0]), float(gll(z[None], gf)[0])))
            key = f"s{S}__{gname}"
            arts[f"succ_mean.{key}"] = anchor.astype(np.float32)
            arts[f"mu_s.{key}"] = gs[0].astype(np.float32)
            arts[f"cov_s.{key}"] = gs[3].astype(np.float32)
            arts[f"mu_f.{key}"] = gf[0].astype(np.float32)
            arts[f"cov_f.{key}"] = gf[3].astype(np.float32)
            arts[f"ood_lo.{key}"] = np.array(float(np.percentile(mx, 5)) if mx else -1e9,
                                             np.float32)
            # [등록기준 k내 AUROC, pool 전체, 길이, k단독 기준선, 채점된 k 수]
            # [k계층 AUROC, pool 전체, 길이, k단독 기준선, 쌍 수, 기여 k 수, 순열 p]
            arts[f"aux_gate.{key}"] = np.array(
                [med_k, med_a, med_l, base_k, n_pairs, ks_used, p_perm], np.float32)
            arts[f"aux_diag_tgt.{key}"] = np.array(diag, np.float32)

        reg = sorted(set(k.split(".", 1)[1] for k in arts if k.startswith("mu_s.")))
        registry.append((k_tgt, len(reg), ";".join(reg) or "-", len(ep_pool_s),
                         len(ep_pool_f), len(ep_tgt_f), n_excl))
        if not reg:
            # 등록 0건 = 게이트 전면 탈락 → 번들 미생성 (오케스트레이터가 파일 부재로
            # identity/reseed 처리 — v4r oven 관례). 판수·게이트 수치는 아래 로그로 남김.
            print(f"[llr] {SLUG} s{S} {args.coord}{k_tgt}: 등록 0 — 번들 미생성")
            for row in summary:
                print("   " + fmt_summary(row) + "  -")
            continue
        arts["scaler_mu"] = muA.astype(np.float32)
        arts["scaler_std"] = np.array(sdA, np.float32)
        for a, bk in [("enc.0", "net.0"), ("enc.2", "net.2"), ("enc.4", "head")]:
            arts[f"{a}.weight"] = np.asarray(b[f"enc.{bk}.weight"], np.float32)
            arts[f"{a}.bias"] = np.asarray(b[f"enc.{bk}.bias"], np.float32)
        arts["registered"] = np.array(reg)
        arts["meta"] = json.dumps({
            "task": SLUG, "ae_ref": f"ae_bundle_k8 sig={ae_sig}",
            "phase_mode": args.phase_mode,
            "target_scene": S, "target_jitter": k_tgt, "axis_col": axis_col,
            "scenes": [f"s{S}"], "phases": sorted(set(r.split("__")[1] for r in reg)),
            **pool_note,
            "anchor": "★k-local: cluster×k 평균으로 중심화해 fit; 번들 succ_mean 은 "
                      "대상 k 앵커(대상 k 실패 record 평균) — serve 가 빼는 값과 동일",
            "gate": "★k-계층화: succ·fail 공존 k 안의 (실패,성공) 쌍만 센 ep-LOO "
                    f"concordance >={args.gate_med}, 쌍 수 >={args.gate_min_pairs}, "
                    f"k 안 라벨 순열 p<={args.gate_p}({args.n_perm}회). pool 전체·길이·"
                    "k단독 기준선은 aux 병기(aux_gate=[k내, pool, len, k단독, n_k]) — pool "
                    "전체로 등록하면 k 정체성 판독기가 통과한다(index 실측 k단독 중앙값 0.835). "
                    "대상 진단은 aux_diag_tgt(등록 미사용)",
            "ood": "대상 k 실패 후반절반 max(logN) 5pct (scorer 스케일, 2π 상수 포함)"})
        reg_root = (f"rsn_llr_reg_{args.tag}" if args.phase_mode == "ck8"
                    else f"rsn_llr_reg_{args.tag}_all")
        out_d = os.path.join(root, reg_root, SLUG, f"s{S}")
        os.makedirs(out_d, exist_ok=True)
        np.savez_compressed(os.path.join(out_d, f"{args.coord}{k_tgt}.npz"), **arts)
        print(f"[llr] {SLUG} s{S} {args.coord}{k_tgt}: 등록 {len(reg)} → "
              f"{reg_root}/{SLUG}/s{S}/{args.coord}{k_tgt}.npz")
        for row in summary:
            print("   " + fmt_summary(row) + ("  REG" if row[-1] else "  -"))

    # 등록표 (중추 rsn_llr 분모 고정용 — 한 invocation = 한 (slug, scene) 전체라 멱등 rewrite)
    if not args.only_gt:
        reg_d = os.path.join(root, f"rsn_llr_reg_{args.tag}" if args.phase_mode == "ck8"
                             else f"rsn_llr_reg_{args.tag}_all", SLUG, f"s{S}")
        os.makedirs(reg_d, exist_ok=True)
        with open(os.path.join(reg_d, "registry.tsv"), "w") as f:
            f.write(f"slug\tscene\t{args.coord}\tn_registered\tentries\t"
                "n_ep_pool_succ\tn_ep_pool_fail"
                    "\tn_ep_tgt_fail\tn_ep_excluded_succ_tgt\n")
            for row in registry:
                f.write(f"{SLUG}\t{S}\t" + "\t".join(str(x) for x in row) + "\n")
    print(f"V5_LOKO_DONE {SLUG} s{S}" + (" (only-gt)" if args.only_gt else ""))


if __name__ == "__main__":
    main()
