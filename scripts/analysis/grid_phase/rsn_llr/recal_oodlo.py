"""ood_lo 발화-분포 재보정: entry별 = 그 scene 실패 ep 후반절반 record의
max(logN_s,logN_f) 5퍼센타일 (scorer 스케일). 기존 fit-창 임계는 aux_oodlo_fit로 보존."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/tmp_segb"))
from llr_scorer import LLRScorer

A = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/")
for f in sorted(glob.glob(A + "rsn_llr_reg/*.npz")):
    slug = os.path.basename(f)[:-4]
    d = dict(np.load(f, allow_pickle=True).items())
    if "registered" not in d or len(d["registered"]) == 0:
        continue
    sc_obj = LLRScorer.from_bundle(f)
    seg_d = np.load(A + f"segA_v4_ck8/{slug}.npz", allow_pickle=True)
    meta = json.loads(str(seg_d["meta_json"]))
    cap = [int(x) for x in meta["capture_layers"]]
    seg = meta["segment_names"].index("all")
    X = seg_d["X"][:, cap.index(12), 3, seg, :].astype(np.float32)
    pc, ep, scn, su = seg_d["phase_code"], seg_d["ep_id"], seg_d["scene"], seg_d["succ"]
    for entry in [str(x) for x in d["registered"]]:
        s_, ph = entry.split("__")
        s_i = int(s_[1:])
        mx = []
        for e in np.unique(ep):
            i0 = np.where(ep == e)[0]
            if int(scn[i0[0]]) != s_i or int(su[i0[0]]) == 1:
                continue
            for i in i0[len(i0) // 2:]:
                r = sc_obj.score(X[i], s_i, ph)
                mx.append(max(r["log_s"], r["log_f"]))
        if not mx:
            continue
        d[f"aux_oodlo_fit.{entry}"] = d[f"ood_lo.{entry}"]
        new = float(np.percentile(mx, 5))
        d[f"ood_lo.{entry}"] = np.array(new, np.float32)
        rej = float(np.mean(np.array(mx) < new))
        print(f"{slug} {entry}: ood_lo {float(d[f'aux_oodlo_fit.{entry}']):.1f} → {new:.1f} "
              f"(발화대리 {len(mx)}rec, 재보정 후 기각률 {rej:.0%})")
    np.savez_compressed(f, **d)
print("RECAL_DONE")
