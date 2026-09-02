"""OOD 전원기각 판독 분리 (stdout): fit-창 record를 serve LLRScorer에 그대로 통과.

fit-창이 비-OOD 통과 → (a) 진성 분포 격리 / fit-창조차 기각 → (b) 전처리 버그.
+ encode 일치 검증(torch vs scorer numpy). usage: python - <slug> <scene> <phase>
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/tmp_segb"))
from llr_scorer import LLRScorer

SLUG, SC, PH = sys.argv[1], int(sys.argv[2]), sys.argv[3]
A = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/")
d = np.load(A + f"segA_v4_ck8/{SLUG}.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
cap = [int(x) for x in meta["capture_layers"]]
seg = meta["segment_names"].index("all")
X = d["X"][:, cap.index(12), 3, seg, :].astype(np.float32)
pc, ep, sc_, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]
code = meta["phase_codebook"][PH]

scorer = LLRScorer.from_bundle(A + f"rsn_llr_reg/{SLUG}.npz")
reg = np.load(A + f"rsn_llr_reg/{SLUG}.npz", allow_pickle=True)
B = int(reg[f"aux_B.s{SC}__{PH}"])

# encode 일치 검증 (1 벡터)
import torch
sys.path.insert(0, "/tmp/kai_lab/repo/scripts/analysis/grid_phase")
import ae_cluster as ac
ck = torch.load(os.path.expanduser(
    "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/ae16_930.pt"), map_location="cpu")
mK = ac.BaseAE(ac.Encoder(1536, 16), ac.Decoder(16, 1536))
mK.load_state_dict(ck["state_dict"]); mK.eval()
v = X[0]
with torch.no_grad():
    z_t = mK.latent(torch.from_numpy(
        ((v - np.asarray(ck["scaler"]["mu"], np.float32)) / float(ck["scaler"]["scalar_std"]))[None])).numpy()[0]
z_s = scorer.encode(v)
print(f"# encode max|Δ| torch vs scorer = {np.abs(z_t - z_s).max():.2e}")

res = {0: [], 1: []}
ood = {0: 0, 1: 0}
tot = {0: 0, 1: 0}
for e in np.unique(ep):
    m = (ep == e)
    if int(sc_[np.where(m)[0][0]]) != SC:
        continue
    u = int(su[np.where(m)[0][0]])
    idx = np.where(m & (pc == code))[0][:B]
    for i in idx:
        r = scorer.score(X[i], SC, PH)
        tot[u] += 1
        ood[u] += r["ood_reject"]
        res[u].append(r["llr"])
for u, nm in [(1, "succ"), (0, "fail")]:
    if tot[u]:
        print(f"{nm}: fit-창 record {tot[u]}개, OOD기각 {ood[u]} ({ood[u]/tot[u]:.0%}), "
              f"LLR 중앙값 {np.median(res[u]):.1f}")
# 후반절반(발화 구간 대리, phase 무관 전 record)도 측정
late_ood, late_n = 0, 0
for e in np.unique(ep):
    m = (ep == e)
    i0 = np.where(m)[0]
    if int(sc_[i0[0]]) != SC or int(su[i0[0]]) == 1:
        continue
    for i in i0[len(i0) // 2:]:
        r = scorer.score(X[i], SC, PH)
        late_n += 1
        late_ood += r["ood_reject"]
print(f"실패 후반절반(발화 대리, phase 무관): {late_n}개 중 OOD기각 {late_ood} ({late_ood/late_n:.0%})")
print("DISCRIM_DONE")
