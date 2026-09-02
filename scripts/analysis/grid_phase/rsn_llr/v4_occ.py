"""등록 cluster의 실패-record 점유율 (발화 가능성 대리, stdout).

per (task, scene): 실패 episode record의 cluster 분포 — 등록 cluster 점유율
(전체 / 후반 절반 = 발화 구간 대리). segA_v4_ck8 phase_code = serve k8 번들 동일.
"""
import glob, json, os
import numpy as np

BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/segA_v4_ck8/")
REG = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/rsn_llr_reg/")
print(f"{'task':<20} {'scene':<6} {'등록cluster':<14} {'실패rec점유(전체)':<18} {'후반절반':<10} 실패ep 상위 cluster")
for f in sorted(glob.glob(REG + "*.npz")):
    slug = os.path.basename(f)[:-4]
    r = np.load(f, allow_pickle=True)
    if "registered" not in r or len(r["registered"]) == 0:
        continue
    d = np.load(BASE + slug + ".npz", allow_pickle=True)
    pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]
    meta = json.loads(str(d["meta_json"]))
    CB = {v: k for k, v in meta["phase_codebook"].items()}
    by_scene = {}
    for e in np.unique(ep):
        m = ep == e
        i0 = np.where(m)[0]
        if int(su[i0[0]]) == 1:
            continue
        by_scene.setdefault(int(sc[i0[0]]), []).append(i0)
    reg_by_scene = {}
    for entry in r["registered"]:
        s_, c_ = str(entry).split("__")
        reg_by_scene.setdefault(int(s_[1:]), []).append(c_)
    for s_, clist in sorted(reg_by_scene.items()):
        idxs = by_scene.get(s_, [])
        if not idxs:
            print(f"{slug:<20} s{s_:<5} {'+'.join(clist):<14} 실패 ep 없음")
            continue
        codes = set(k for k, v in meta["phase_codebook"].items() if k in clist)
        code_ids = [v for k, v in meta["phase_codebook"].items() if k in clist]
        occ_all, occ_late = [], []
        top = {}
        for i0 in idxs:
            pcs = pc[i0]
            occ_all.append(np.isin(pcs, code_ids).mean())
            late = pcs[len(pcs) // 2:]
            occ_late.append(np.isin(late, code_ids).mean())
            for c in late:
                top[CB[int(c)]] = top.get(CB[int(c)], 0) + 1
        tot = sum(top.values())
        tops = ", ".join(f"{k}:{v/tot:.0%}" for k, v in sorted(top.items(), key=lambda x: -x[1])[:4])
        print(f"{slug:<20} s{s_:<5} {'+'.join(clist):<14} "
              f"{np.mean(occ_all):.0%} (min {min(occ_all):.0%})    "
              f"{np.mean(occ_late):.0%}       {tops}")
print("OCC_DONE")
