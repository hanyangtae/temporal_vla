"""신규 라운드 fit (pool 규약: 타 4 scene 전체 + 대상 scene 실패만. 대상 성공 전면 금지).

usage: python - <slug> <target_scene:int> <ae.pt>

산출 (승준 analysis/grid_phase_v4/):
  A. instr_setm_v4sb_{gt,ck8}/<slug>/<phase>/dit_L12/conceptors.npz
     — v = normalize(mean_scene(μ_s−μ_f)) (타 scene 쌍대 mean-diff),
       s = μ_f,target·v + mean_scene((μ_s−μ_f)·v)  [하이브리드 setpoint]
  B. rsn_llr_reg_pool/<slug>.npz — LLR 번들(계약 동일), 앵커=scene별 허용-record phase 평균,
     게이트=타-scene CV(LLR>len 5-seed), 대상-scene 진단 AUROC는 aux(등록 미사용),
     ood_lo=대상 실패 후반절반 5퍼센타일(scorer 스케일).
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

SLUG, TGT, AE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
A = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/")
CONST16 = 0.5 * 16 * math.log(2 * math.pi)


def load(cache):
    d = np.load(A + cache + f"/{SLUG}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    cap = [int(x) for x in meta["capture_layers"]]
    seg = meta["segment_names"].index("all")
    X = d["X"][:, cap.index(12), 3, seg, :].astype(np.float32)
    return X, d["phase_code"], d["ep_id"], d["scene"], d["succ"], meta


def ep_index(ep, sc, su):
    ids = np.unique(ep)
    return ids, {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ids}


def allowed(em, e):
    s_, u = em[e]
    return s_ != TGT or u == 0


# ── A. setM (gt / ck8) ──────────────────────────────────────────────
for tag, cache in [("ck8", "segA_v4r_ck8")]:
    X, pc, ep, sc, su, meta = load(cache)
    ep_ids, em = ep_index(ep, sc, su)
    CB = {v: k for k, v in meta["phase_codebook"].items()}
    outroot = A + f"instr_setm_v4r_{tag}/{SLUG}"
    n_ph = 0
    for pcode in sorted(set(pc.tolist())):
        ph = CB[pcode]
        deltas, s_terms = [], []
        for s_ in sorted(set(v[0] for v in em.values())):
            if s_ == TGT:
                continue
            ms = np.where((sc == s_) & (pc == pcode) & (su == 1))[0]
            mf = np.where((sc == s_) & (pc == pcode) & (su == 0))[0]
            if len(ms) < 20 or len(mf) < 20:
                continue
            mu_s, mu_f = X[ms].mean(0), X[mf].mean(0)
            deltas.append(mu_s - mu_f)
        if len(deltas) < 2:
            continue
        delta = np.mean(deltas, 0)
        v = (delta / (np.linalg.norm(delta) + 1e-12)).astype(np.float64)
        mtf = np.where((sc == TGT) & (pc == pcode) & (su == 0))[0]
        if len(mtf) < 10:
            continue
        s_val = float(X[mtf].mean(0) @ v) + float(np.mean([d @ v for d in deltas]))
        d_out = os.path.join(outroot, ph, "dit_L12")
        os.makedirs(d_out, exist_ok=True)
        np.savez_compressed(os.path.join(d_out, "conceptors.npz"),
                            alpha0_v_steer=v.astype(np.float32),
                            alpha0_s=np.float32(s_val))
        with open(os.path.join(d_out, "metadata.json"), "w") as f:
            json.dump({"op": "setpoint", "variant": f"instr_setm_v4r_{tag}",
                       "phase": ph, "target_scene": TGT,
                       "pool": "other-scenes(all)+target-fail",
                       "n_scene_delta": len(deltas), "n_target_fail_rec": int(len(mtf)),
                       "phase_label_source": tag, "layer": 12,
                       "note": "v=scene-paired mean-diff avg; s=mu_f_tgt.v + mean(delta.v)"},
                      f, ensure_ascii=False, indent=1)
        n_ph += 1
    print(f"[setm_{tag}] {SLUG} tgt=s{TGT}: {n_ph} phase 산출 → instr_setm_v4sb_{tag}/{SLUG}")

# ── B. LLR 번들 (ck8) ────────────────────────────────────────────────
import torch

sys.path.insert(0, "/tmp/kai_lab/repo/scripts/analysis/grid_phase")
import ae_cluster as ac

X, pc, ep, sc, su, meta = load("segA_v4r_ck8")
ep_ids, em = ep_index(ep, sc, su)
CB = {v: k for k, v in meta["phase_codebook"].items()}
ck = torch.load(AE, map_location="cpu")
mK = ac.BaseAE(ac.Encoder(1536, 16), ac.Decoder(16, 1536))
mK.load_state_dict(ck["state_dict"])
mK.eval()
muK = np.asarray(ck["scaler"]["mu"], np.float32)
sdK = float(ck["scaler"]["scalar_std"])
with torch.no_grad():
    Z = np.concatenate([mK.latent(torch.from_numpy((X[i:i + 8192] - muK) / sdK)).numpy()
                        for i in range(0, len(X), 8192)]).astype(np.float64)
ae_sig = hashlib.sha256(open(AE, "rb").read()).hexdigest()[:16]


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


arts = {}
summary = []
for pcode in sorted(set(pc.tolist())):
    ph = CB[pcode]
    # scene별 앵커 = 허용 record 의 phase 평균 (대상 scene = 실패만)
    anchors = {}
    for s_ in sorted(set(v[0] for v in em.values())):
        if s_ == TGT:
            m = np.where((sc == s_) & (pc == pcode) & (su == 0))[0]
        else:
            m = np.where((sc == s_) & (pc == pcode))[0]
        if len(m) >= 10:
            anchors[s_] = Z[m].mean(0)
    if TGT not in anchors:
        continue
    def ZA(mask_scene, mask_extra):
        out = []
        for s_, anc in anchors.items():
            if not mask_scene(s_):
                continue
            m = mask_extra(s_)
            if len(m):
                out.append(Z[m] - anc)
        return np.concatenate(out) if out else np.zeros((0, 16))
    Zs = ZA(lambda s_: s_ != TGT,
            lambda s_: np.where((sc == s_) & (pc == pcode) & (su == 1))[0])
    Zf = ZA(lambda s_: True,
            lambda s_: np.where((sc == s_) & (pc == pcode) & (su == 0))[0])
    if len(Zs) < 60 or len(Zf) < 60:
        continue
    # 게이트: 타-scene episode CV (5-seed)
    o_eps = [e for e in ep_ids if em[e][0] != TGT and len(np.where((ep == e) & (pc == pcode))[0]) >= 4]
    ga, gl_ = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        order = rng.permutation(o_eps)
        tr = set(order[: int(len(o_eps) * 0.6)].tolist())
        te = [e for e in o_eps if e not in tr]
        def cls_recs(u, eps_):
            out = []
            for e in eps_:
                s_ = em[e][0]
                if em[e][1] != u or s_ not in anchors:
                    continue
                m = np.where((ep == e) & (pc == pcode))[0]
                out.append(Z[m] - anchors[s_])
            return np.concatenate(out) if out else np.zeros((0, 16))
        Xs_t, Xf_t = cls_recs(1, tr), cls_recs(0, tr)
        if len(Xs_t) < 40 or len(Xf_t) < 40:
            continue
        gs_, gf_ = gfit(Xs_t), gfit(Xf_t)
        by = {}
        for e in te:
            s_ = em[e][0]
            if s_ not in anchors:
                continue
            m = np.where((ep == e) & (pc == pcode))[0]
            Zb = Z[m] - anchors[s_]
            v_ = float((gll(Zb, gf_) - gll(Zb, gs_)).mean())
            by.setdefault(s_, {0: [], 1: []})[em[e][1]].append((v_, float((ep == e).sum())))
        pa, pl = [], []
        for s_, vv in by.items():
            if len(vv[0]) >= 2 and len(vv[1]) >= 2:
                pa.append(auroc(np.array([x[0] for x in vv[0]]), np.array([x[0] for x in vv[1]])))
                pl.append(auroc(np.array([x[1] for x in vv[0]]), np.array([x[1] for x in vv[1]])))
        if pa:
            ga.append(float(np.median(pa)))
            gl_.append(float(np.median(pl)))
    ok = [a >= 0.65 for a in ga if not np.isnan(a)]
    med_a = float(np.nanmedian(ga)) if ga else float("nan")
    med_l = float(np.nanmedian(gl_)) if gl_ else float("nan")
    passed = bool(ok) and med_a >= 0.70 and sum(ok) * 2 > len(ok)
    # 대상 scene 진단 (aux — 등록 미사용, 대상 성공 라벨 사용을 명시)
    gs, gf = gfit(Zs), gfit(Zf)
    diag = float("nan")
    mts = np.where((sc == TGT) & (pc == pcode) & (su == 1))[0]
    mtf = np.where((sc == TGT) & (pc == pcode) & (su == 0))[0]
    if len(mts) >= 4 and len(mtf) >= 4:
        diag = auroc(gll(Z[mtf] - anchors[TGT], gf) - gll(Z[mtf] - anchors[TGT], gs),
                     gll(Z[mts] - anchors[TGT], gf) - gll(Z[mts] - anchors[TGT], gs))
    summary.append((ph, len(Zs), len(Zf), med_a, med_l, diag, passed))
    if not passed:
        continue
    # ood_lo: 대상 실패 후반절반
    mx = []
    for e in ep_ids:
        if em[e][0] != TGT or em[e][1] == 1:
            continue
        i0 = np.where(ep == e)[0]
        for i in i0[len(i0) // 2:]:
            z = Z[i] - anchors[TGT]
            mx.append(max(float(gll(z[None], gs)[0]), float(gll(z[None], gf)[0])))
    key = f"s{TGT}__{ph}"
    arts[f"succ_mean.{key}"] = anchors[TGT].astype(np.float32)
    arts[f"mu_s.{key}"] = gs[0].astype(np.float32)
    arts[f"cov_s.{key}"] = gs[3].astype(np.float32)
    arts[f"mu_f.{key}"] = gf[0].astype(np.float32)
    arts[f"cov_f.{key}"] = gf[3].astype(np.float32)
    arts[f"ood_lo.{key}"] = np.array(float(np.percentile(mx, 5)) if mx else -1e9, np.float32)
    arts[f"aux_gate.{key}"] = np.array([med_a, med_l], np.float32)
    arts[f"aux_diag_tgt.{key}"] = np.array(diag, np.float32)

reg = sorted(set(k.split(".", 1)[1] for k in arts if k.startswith("mu_s.")))
sd_k = ck["state_dict"]
arts["scaler_mu"] = muK.astype(np.float32)
arts["scaler_std"] = np.array(sdK, np.float32)
for a, b in [("enc.0", "enc.net.0"), ("enc.2", "enc.net.2"), ("enc.4", "enc.head")]:
    arts[f"{a}.weight"] = sd_k[f"{b}.weight"].numpy()
    arts[f"{a}.bias"] = sd_k[f"{b}.bias"].numpy()
arts["registered"] = np.array(reg)
arts["meta"] = json.dumps({
    "task": SLUG, "ae_ref": f"ae16_930 sig={ae_sig}", "target_scene": TGT,
    "scenes": [f"s{TGT}"], "phases": sorted(set(r.split("__")[1] for r in reg)),
    "pool": "other-scenes(all)+target-fail; target-succ banned (fit·anchor·gate·ood 전부)",
    "anchor": "scene별 허용-record phase 평균 (대상 scene=실패 평균)",
    "gate": "scene-내 CV 절대 기준(중앙값>=0.70, 과반>=0.65) — 채점기 용도에서 길이는 후보 순위 불가라 비교 기준 아님(병기만); 대상-scene AUROC는 aux_diag_tgt(등록 미사용)",
    "ood": "대상 실패 후반절반 max(logN) 5pct (scorer 스케일, 2π 상수 포함)"})
out = A + "rsn_llr_reg_v4r/"
os.makedirs(out, exist_ok=True)
np.savez_compressed(out + SLUG + ".npz", **arts)
print(f"[llr] {SLUG} tgt=s{TGT}: 등록 {len(reg)} → rsn_llr_reg_pool/{SLUG}.npz")
for ph, ns, nf, a, l, dg, p in summary:
    print(f"   {ph:<5} pool s/f rec {ns}/{nf}  CV {a:.2f} vs len {l:.2f}  진단(tgt) {dg:.2f}  {'REG' if p else '-'}")
print(f"POOLFIT_DONE {SLUG}")
