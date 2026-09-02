"""rsN_llr 등록표 + 채점기 산출물 (v4, scene-전용 fit). usage: python - <slug> <ae.pt>

입력: segA_v4_ck8/<slug>.npz (X[rec,L,D,seg,1536], phase_code, scene, jitter, succ)
출력: analysis/grid_phase_v4/rsn_llr_reg/<slug>.npz + stdout 등록표.
게이트: scene×phase cell — n_s>=8 & n_f>=8 & [held-out LLR AUROC > 길이단독] 5-seed 중앙값+과반.
산출물(등록 cell): mu_s/cov_s/mu_f/cov_f(inv·logdet 포함), center(scene 성공-phase 평균),
B, ood_thr(train max-logN 5퍼센타일). AE-16 공간. docs/04: 절대경로 미기록.
"""
import hashlib
import json
import os
import sys

import numpy as np

SLUG, AE = sys.argv[1], sys.argv[2]
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/segA_v4_ck8/")
OUT = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/rsn_llr_reg/")
os.makedirs(OUT, exist_ok=True)
d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
cap = [int(x) for x in meta["capture_layers"]]
seg = meta["segment_names"].index("all")
X_raw = d["X"][:, cap.index(12), 3, seg, :].astype(np.float32)
pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]
CB = {v: k for k, v in meta["phase_codebook"].items()}

import torch
sys.path.insert(0, "/tmp/kai_lab/repo/scripts/analysis/grid_phase")
import ae_cluster as ac

ck = torch.load(AE, map_location="cpu")
mK = ac.BaseAE(ac.Encoder(1536, 16), ac.Decoder(16, 1536))
mK.load_state_dict(ck["state_dict"])
mK.eval()
muK = np.asarray(ck["scaler"]["mu"], np.float32)
sdK = float(ck["scaler"]["scalar_std"])
Z = []
with torch.no_grad():
    for i in range(0, len(X_raw), 8192):
        Z.append(mK.latent(torch.from_numpy((X_raw[i:i + 8192] - muK) / sdK)).numpy())
X = np.concatenate(Z).astype(np.float64)
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


def gll(Xc, mu, Ci, ld, _raw=None):
    D = Xc - mu
    return -0.5 * ((D @ Ci) * D).sum(1) - 0.5 * ld


ep_ids = np.unique(ep)
em = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids}
arts = {}
rows_out = []
for s_ in sorted(set(v[0] for v in em.values())):
    eps_sc = [e for e in ep_ids if em[e][0] == s_]
    for pcode in sorted(set(pc.tolist())):
        ph = CB[pcode]
        rows = {e: np.where((ep == e) & (pc == pcode))[0] for e in eps_sc}
        rows = {e: i for e, i in rows.items() if len(i) >= 4}
        ns = sum(em[e][1] for e in rows)
        nf = len(rows) - ns
        if ns < 8 or nf < 8:
            if ns + nf > 0 and min(ns, nf) >= 3:
                rows_out.append((s_, ph, ns, nf, "-", "-", "SKIP(n<8)"))
            continue
        elist = sorted(rows)
        # held-out 게이트 (5-seed, scene 내 episode split)
        gate_a, gate_l = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            order = rng.permutation(elist)
            tr = [e for e in order[: int(len(elist) * 0.6)]]
            te = [e for e in elist if e not in tr]
            if sum(em[e][1] for e in tr) < 3 or sum(1 - em[e][1] for e in tr) < 3:
                continue
            cen = np.concatenate([X[rows[e]] for e in tr if em[e][1] == 1]).mean(0)
            dw = sorted(len(rows[e]) for e in tr if em[e][1] == 1)
            B = max(3, dw[len(dw) // 4])
            try:
                gs = gfit(np.concatenate([X[rows[e]][:B] - cen for e in tr if em[e][1] == 1]))
                gf = gfit(np.concatenate([X[rows[e]][:B] - cen for e in tr if em[e][1] == 0]))
            except np.linalg.LinAlgError:
                continue
            va = {0: [], 1: []}
            vl = {0: [], 1: []}
            for e in te:
                if len(rows[e]) < B:
                    continue
                Zb = X[rows[e]][:B] - cen
                va[em[e][1]].append(float((gll(Zb, *gf) - gll(Zb, *gs)).mean()))
                vl[em[e][1]].append(float(len(rows[e])))
            gate_a.append(auroc(np.array(va[0]), np.array(va[1])))
            gate_l.append(auroc(np.array(vl[0]), np.array(vl[1])))
        ok = [a > l for a, l in zip(gate_a, gate_l) if not (np.isnan(a) or np.isnan(l))]
        med_a = float(np.nanmedian(gate_a)) if gate_a else float("nan")
        med_l = float(np.nanmedian(gate_l)) if gate_l else float("nan")
        passed = bool(ok) and med_a > med_l and sum(ok) * 2 > len(ok)
        rows_out.append((s_, ph, ns, nf, f"{med_a:.2f}", f"{med_l:.2f}",
                         "REGISTER" if passed else "fail-gate"))
        if not passed:
            continue
        # 최종 fit = scene 전체 (split 무관)
        cen = np.concatenate([X[rows[e]] for e in elist if em[e][1] == 1]).mean(0)
        dw = sorted(len(rows[e]) for e in elist if em[e][1] == 1)
        B = max(3, dw[len(dw) // 4])
        Xs = np.concatenate([X[rows[e]][:B] - cen for e in elist if em[e][1] == 1])
        Xf = np.concatenate([X[rows[e]][:B] - cen for e in elist if em[e][1] == 0])
        gs, gf = gfit(Xs), gfit(Xf)
        allB = np.concatenate([X[rows[e]][:B] - cen for e in elist])
        mx = np.maximum(gll(allB, *gs), gll(allB, *gf))
        key = f"s{s_}__{ph}"
        arts[f"succ_mean.{key}"] = cen.astype(np.float32)
        arts[f"mu_s.{key}"] = gs[0].astype(np.float32)
        arts[f"cov_s.{key}"] = gs[3].astype(np.float32)
        arts[f"mu_f.{key}"] = gf[0].astype(np.float32)
        arts[f"cov_f.{key}"] = gf[3].astype(np.float32)
        arts[f"ood_lo.{key}"] = np.array(float(np.percentile(mx, 5)), np.float32)
        arts[f"aux_B.{key}"] = np.array(B)
        arts[f"aux_auroc.{key}"] = np.array([med_a, med_l, ns, nf], np.float32)

reg = sorted(set(k.split(".", 1)[1] for k in arts if k.startswith("mu_s.")))
sd_k = {k: v for k, v in ck["state_dict"].items()}
arts["scaler_mu"] = muK.astype(np.float32)
arts["scaler_std"] = np.array(sdK, np.float32)
arts["enc.0.weight"] = sd_k["enc.net.0.weight"].numpy()
arts["enc.0.bias"] = sd_k["enc.net.0.bias"].numpy()
arts["enc.2.weight"] = sd_k["enc.net.2.weight"].numpy()
arts["enc.2.bias"] = sd_k["enc.net.2.bias"].numpy()
arts["enc.4.weight"] = sd_k["enc.head.weight"].numpy()
arts["enc.4.bias"] = sd_k["enc.head.bias"].numpy()
arts["registered"] = np.array(reg)
arts["meta"] = json.dumps({
    "task": SLUG, "ae_ref": f"ae16_930 sig={ae_sig}",
    "scenes": sorted(set(int(r.split("__")[0][1:]) for r in reg)),
    "phases": sorted(set(r.split("__")[1] for r in reg)),
    "space": "ae16(scene succ-phase centered)", "layer": 12,
    "denoise": "last", "segment": "all(token-mean)",
    "score": "LLR=logN(z;mu_f,cov_f)-logN(z;mu_s,cov_s); cov는 로드 시 +1e-3*tr/16*I; 후보 chunk record 평균 argmin",
    "ood": "max(logN_s,logN_f) < ood_lo -> 기각",
    "gate": "scene-전용 n>=8/8, held-out LLR>len 5-seed", "src_cache": "segA_v4_ck8"})
np.savez_compressed(OUT + SLUG + ".npz", **arts)
print(f"== {SLUG} 등록 {len(set(reg))} cell → rsn_llr_reg/{SLUG}.npz")
for r in rows_out:
    print(f"  s{r[0]:<3} {r[1]:<16} s/f={r[2]}/{r[3]:<4} LLR {r[4]:<5} len {r[5]:<5} {r[6]}")
print(f"REG_DONE {SLUG}")
