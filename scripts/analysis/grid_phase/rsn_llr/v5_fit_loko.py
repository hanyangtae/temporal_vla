"""v5 LOKO fit — scene-local 계약: (instruction, scene, 대상 k)당 연산자 1개.

pool = 대상 scene의 타 k 전판(succ+fail) + 대상 k 실패판 (대상 k 성공판 제외 = success-blind).
대상 k = 그 scene에서 실패 판이 있는 k 전부(자동 산출) — leave-one-k-out을 k마다 반복.

데이터 = segA_v5 shard(slug별 NPZ, `jitter` 열 실좌표) + ae_bundle_v5_k8.npz(encoder+centers).
ck8 라벨은 번들 centers 최근접으로 파생(assign 식 = 온라인 cluster_phase와 동일) —
labels npz가 있으면 파생 라벨과 전행 일치를 검증한다(불일치 = 번들≠라벨 세계, fail-loud).

usage (승준, ~/anaconda3/bin/python — torch 필요, scipy 불요):
  python v5_fit_loko.py <slug> <scene:int> [--seg-dir D] [--bundle F] [--labels F] [--out ROOT]

산출 (ROOT 기본 = ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v5/):
  A. instr_setm_v5_gt/<slug>/s<i>/k<r>/<phase>/dit_L12/conceptors.npz (+ metadata.json)
     — 기본형: v = normalize(μ_s − μ_f), s = μ_s·v (pool 성공이 같은 scene이라
       v4sb/v4r 하이브리드 setpoint 불요 — scene 도메인 offset 소거됨)
  B. instr_setm_v5_ck8/<slug>/s<i>/k<r>/c<j>/dit_L12/conceptors.npz — phase = ck8 cluster
  C. rsn_llr_reg_v5/<slug>/s<i>/k<r>.npz — LLR 번들, 계약 = src/failure_online/llr_scorer.py
     docstring(단일 출처). entry "s<i>__c<j>", 앵커 = pool record cluster 평균,
     게이트 = pool ep CV 절대 기준(5-seed 중앙값 ≥0.70 + 과반 ≥0.65; 길이 AUROC 병기),
     ood_lo = 대상 k 실패 후반절반 max(logN) 5pct (scorer 스케일 — 2π 상수 포함 필수).

주의: 게이트의 succ/fail episode 구분은 k와 상관된다(성공은 타 k에서만 옴) — 순수
k-정체성 판독 위험은 aux_gate_k(양 클래스가 공존하는 타 k 내부 AUROC 중앙값)로 병기만
하고 등록 기준으론 쓰지 않는다(표본 희박). 최종 심판은 eval(rsn_llr vs reseed).
"""
import argparse
import hashlib
import json
import math
import os

import numpy as np

DEF_ROOT = "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v5/"
CONST16 = 0.5 * 16 * math.log(2 * math.pi)
LAYER = 12       # DiT block residual layer (capture index로 역산)
DENOISE = 3      # 마지막 denoise call
MIN_REC_SETM = 20
MIN_REC_LLR = 60
GATE_SEEDS = 5
GATE_MED = 0.70
GATE_EACH = 0.65


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("scene", type=int)
    ap.add_argument("--seg-dir", default=None, help="기본 ROOT/segA")
    ap.add_argument("--bundle", default=None, help="기본 ROOT/ae_v5_k8/ae_bundle_v5_k8.npz")
    ap.add_argument("--labels", default=None,
                    help="기본 ROOT/ae_v5_k8/labels_<slug>_k8.npz (있으면 대조, 없으면 생략)")
    ap.add_argument("--out", default=None, help="기본 ROOT")
    # 아래 3개는 합성 스모크 전용 완화 옵션 — 본 fit 은 기본값 그대로 쓸 것
    ap.add_argument("--min-rec-llr", type=int, default=MIN_REC_LLR)
    ap.add_argument("--gate-med", type=float, default=GATE_MED)
    ap.add_argument("--gate-each", type=float, default=GATE_EACH)
    return ap.parse_args()


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
    bundle_p = os.path.expanduser(args.bundle or os.path.join(root, "ae_v5_k8/ae_bundle_v5_k8.npz"))
    labels_p = os.path.expanduser(args.labels or os.path.join(
        root, f"ae_v5_k8/labels_{args.slug}_k8.npz"))
    SLUG, S = args.slug, args.scene

    # ── shard ──
    d = np.load(os.path.join(seg_dir, f"{SLUG}.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    if not meta.get("jitter_axis"):
        raise SystemExit(f"{SLUG}: meta_json.jitter_axis 가 false — v5 3축 shard 가 아님")
    cap = [int(x) for x in meta["capture_layers"]]
    seg = meta["segment_names"].index("all")
    X = d["X"][:, cap.index(LAYER), DENOISE, seg, :].astype(np.float32)
    pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]
    jit, eplen = d["jitter"], d["ep_len"]
    CB = {v: k for k, v in meta["phase_codebook"].items()}

    # ── AE 번들 → encode + ck8 라벨 파생 ──
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
    n_s = sum(1 for v in em.values() if v[0] == 1)
    print(f"[scene] {SLUG} s{S}: ep {len(eps_s)} (succ {n_s} / fail {len(eps_s) - n_s}), "
          f"k = {ks_all}, 실패 있는 k = {ks_fail}")
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
        for tag, la, name_of in (("gt", pc, lambda c: CB[int(c)]),
                                 ("ck8", ck, lambda c: f"c{int(c)}")):
            outroot = os.path.join(root, f"instr_setm_v5_{tag}", SLUG, f"s{S}", f"k{k_tgt}")
            n_ph = 0
            for code in sorted(set(la[allow].tolist())):
                ms = np.where(allow & (la == code) & (su == 1))[0]
                mf = np.where(allow & (la == code) & (su == 0))[0]
                if len(ms) < MIN_REC_SETM or len(mf) < MIN_REC_SETM:
                    continue
                mu_s = X[ms].mean(0).astype(np.float64)
                mu_f = X[mf].mean(0).astype(np.float64)
                delta = mu_s - mu_f
                v = delta / (np.linalg.norm(delta) + 1e-12)
                s_val = float(mu_s @ v)
                d_out = os.path.join(outroot, name_of(code), "dit_L12")
                os.makedirs(d_out, exist_ok=True)
                np.savez_compressed(os.path.join(d_out, "conceptors.npz"),
                                    alpha0_v_steer=v.astype(np.float32),
                                    alpha0_s=np.float32(s_val))
                with open(os.path.join(d_out, "metadata.json"), "w") as f:
                    json.dump({"op": "setpoint", "variant": f"instr_setm_v5_{tag}",
                               "phase": name_of(code), "target_scene": S, "target_k": k_tgt,
                               "n_rec_s": int(len(ms)), "n_rec_f": int(len(mf)),
                               **pool_note, "phase_label_source": tag, "layer": LAYER,
                               "note": "v=normalize(mu_s−mu_f) (scene-local pool); "
                                       "s=mu_s·v (기본형 — 하이브리드 불요)"},
                              f, ensure_ascii=False, indent=1)
                n_ph += 1
            print(f"[setm_{tag}] {SLUG} s{S} k{k_tgt}: {n_ph} phase → {outroot}")

        # ── C. LLR 번들 ──
        arts, summary = {}, []
        for c in range(centers.shape[0]):
            m_c = allow & (ck == c)
            if m_c.sum() < 10:
                continue
            anchor = Z[m_c].mean(0)
            ms = np.where(m_c & (su == 1))[0]
            mf = np.where(m_c & (su == 0))[0]
            if len(ms) < args.min_rec_llr or len(mf) < args.min_rec_llr:
                summary.append((f"c{c}", len(ms), len(mf), float("nan"), float("nan"),
                                float("nan"), float("nan"), False))
                continue
            # 게이트: pool ep CV (5-seed) — ep 단위 mean LLR AUROC (양성 = 실패)
            o_eps = [e for e in eps_s
                     if (em[e][0] == 0 or em[e][1] != k_tgt)
                     and ((ep == e) & (ck == c) & allow).sum() >= 4]
            ga, gl_, gk = [], [], []
            for seed in range(GATE_SEEDS):
                rng = np.random.default_rng(seed)
                order = rng.permutation(o_eps)
                tr = set(order[: int(len(o_eps) * 0.6)].tolist())
                te = [e for e in o_eps if e not in tr]

                def cls_recs(u, eps_):
                    out = [Z[np.where((ep == e) & (ck == c) & allow)[0]] - anchor
                           for e in eps_ if em[e][0] == u]
                    out = [o for o in out if len(o)]
                    return np.concatenate(out) if out else np.zeros((0, 16))
                Xs_t, Xf_t = cls_recs(1, tr), cls_recs(0, tr)
                if len(Xs_t) < 40 or len(Xf_t) < 40:
                    continue
                gs_, gf_ = gfit(Xs_t), gfit(Xf_t)
                vals = {0: [], 1: []}
                by_k = {}
                for e in te:
                    m = np.where((ep == e) & (ck == c) & allow)[0]
                    Zb = Z[m] - anchor
                    v_ = float((gll(Zb, gf_) - gll(Zb, gs_)).mean())
                    vals[em[e][0]].append((v_, em[e][2]))
                    by_k.setdefault(em[e][1], {0: [], 1: []})[em[e][0]].append(v_)
                if len(vals[0]) >= 2 and len(vals[1]) >= 2:
                    ga.append(auroc(np.array([x[0] for x in vals[0]]),
                                    np.array([x[0] for x in vals[1]])))
                    gl_.append(auroc(np.array([x[1] for x in vals[0]]),
                                     np.array([x[1] for x in vals[1]])))
                pk = [auroc(np.array(v[0]), np.array(v[1])) for v in by_k.values()
                      if len(v[0]) >= 2 and len(v[1]) >= 2]
                if pk:
                    gk.append(float(np.nanmedian(pk)))
            ok = [a >= args.gate_each for a in ga if not np.isnan(a)]
            med_a = float(np.nanmedian(ga)) if ga else float("nan")
            med_l = float(np.nanmedian(gl_)) if gl_ else float("nan")
            med_k = float(np.nanmedian(gk)) if gk else float("nan")
            passed = bool(ok) and med_a >= args.gate_med and sum(ok) * 2 > len(ok)
            # 전체 fit + 대상 진단 (대상 k 실패 ep vs pool 성공 ep)
            gs, gf = gfit(Z[ms] - anchor), gfit(Z[mf] - anchor)

            def ep_llr(e):
                m = np.where((ep == e) & (ck == c) & allow)[0]
                if not len(m):
                    return None
                Zb = Z[m] - anchor
                return float((gll(Zb, gf) - gll(Zb, gs)).mean())
            dvals_f = [v for v in (ep_llr(e) for e in ep_tgt_f) if v is not None]
            dvals_s = [v for v in (ep_llr(e) for e in ep_pool_s) if v is not None]
            diag = auroc(np.array(dvals_f), np.array(dvals_s))
            summary.append((f"c{c}", len(ms), len(mf), med_a, med_l, med_k, diag, passed))
            if not passed:
                continue
            # ood_lo: 대상 k 실패 후반절반 (전 record — cluster 무관, v4 규약 유지)
            mx = []
            for e in ep_tgt_f:
                i0 = em[e][3]
                for i in i0[len(i0) // 2:]:
                    z = Z[i] - anchor
                    mx.append(max(float(gll(z[None], gs)[0]), float(gll(z[None], gf)[0])))
            key = f"s{S}__c{c}"
            arts[f"succ_mean.{key}"] = anchor.astype(np.float32)
            arts[f"mu_s.{key}"] = gs[0].astype(np.float32)
            arts[f"cov_s.{key}"] = gs[3].astype(np.float32)
            arts[f"mu_f.{key}"] = gf[0].astype(np.float32)
            arts[f"cov_f.{key}"] = gf[3].astype(np.float32)
            arts[f"ood_lo.{key}"] = np.array(float(np.percentile(mx, 5)) if mx else -1e9,
                                             np.float32)
            arts[f"aux_gate.{key}"] = np.array([med_a, med_l, med_k], np.float32)
            arts[f"aux_diag_tgt.{key}"] = np.array(diag, np.float32)

        reg = sorted(set(k.split(".", 1)[1] for k in arts if k.startswith("mu_s.")))
        registry.append((k_tgt, len(reg), ";".join(reg) or "-", len(ep_pool_s),
                         len(ep_pool_f), len(ep_tgt_f), n_excl))
        if not reg:
            # 등록 0건 = 게이트 전면 탈락 → 번들 미생성 (오케스트레이터가 파일 부재로
            # identity/reseed 처리 — v4r oven 관례). 판수·게이트 수치는 아래 로그로 남김.
            print(f"[llr] {SLUG} s{S} k{k_tgt}: 등록 0 — 번들 미생성")
            for c_, ns_, nf_, a, l, kk, dg, p in summary:
                print(f"   {c_:<3} rec s/f {ns_}/{nf_}  CV {a:.2f} len {l:.2f} k내부 {kk:.2f} "
                      f" 진단(tgt) {dg:.2f}  -")
            continue
        arts["scaler_mu"] = muA.astype(np.float32)
        arts["scaler_std"] = np.array(sdA, np.float32)
        for a, bk in [("enc.0", "net.0"), ("enc.2", "net.2"), ("enc.4", "head")]:
            arts[f"{a}.weight"] = np.asarray(b[f"enc.{bk}.weight"], np.float32)
            arts[f"{a}.bias"] = np.asarray(b[f"enc.{bk}.bias"], np.float32)
        arts["registered"] = np.array(reg)
        arts["meta"] = json.dumps({
            "task": SLUG, "ae_ref": f"ae_bundle_v5_k8 sig={ae_sig}",
            "target_scene": S, "target_k": k_tgt,
            "scenes": [f"s{S}"], "phases": sorted(set(r.split("__")[1] for r in reg)),
            **pool_note,
            "anchor": "pool record cluster 평균 (허용 record — 대상 k 성공 제외)",
            "gate": "pool ep CV 절대 기준(중앙값>=0.70, 과반>=0.65); 길이·타k내부 AUROC는 "
                    "aux 병기(aux_gate=[llr,len,k내부]); 대상 진단은 aux_diag_tgt(등록 미사용)",
            "ood": "대상 k 실패 후반절반 max(logN) 5pct (scorer 스케일, 2π 상수 포함)"})
        out_d = os.path.join(root, "rsn_llr_reg_v5", SLUG, f"s{S}")
        os.makedirs(out_d, exist_ok=True)
        np.savez_compressed(os.path.join(out_d, f"k{k_tgt}.npz"), **arts)
        print(f"[llr] {SLUG} s{S} k{k_tgt}: 등록 {len(reg)} → rsn_llr_reg_v5/{SLUG}/s{S}/k{k_tgt}.npz")
        for c_, ns_, nf_, a, l, kk, dg, p in summary:
            print(f"   {c_:<3} rec s/f {ns_}/{nf_}  CV {a:.2f} len {l:.2f} k내부 {kk:.2f} "
                  f" 진단(tgt) {dg:.2f}  {'REG' if p else '-'}")

    # 등록표 (중추 rsn_llr 분모 고정용 — 한 invocation = 한 (slug, scene) 전체라 멱등 rewrite)
    reg_d = os.path.join(root, "rsn_llr_reg_v5", SLUG, f"s{S}")
    os.makedirs(reg_d, exist_ok=True)
    with open(os.path.join(reg_d, "registry.tsv"), "w") as f:
        f.write("slug\tscene\tk\tn_registered\tentries\tn_ep_pool_succ\tn_ep_pool_fail"
                "\tn_ep_tgt_fail\tn_ep_excluded_succ_tgt\n")
        for row in registry:
            f.write(f"{SLUG}\t{S}\t" + "\t".join(str(x) for x in row) + "\n")
    print(f"V5_LOKO_DONE {SLUG} s{S}")


if __name__ == "__main__":
    main()
