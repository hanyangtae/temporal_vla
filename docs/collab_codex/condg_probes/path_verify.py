"""phase 내 이동(경로)의 실재성·일관성 4종 검증 (stdout 전용).

(a) episode별 변위의 경로방향 정렬: cos(δ_e, 경로방향) — 1536D 랜덤 null ≈ 0±0.026
(b) held-out: 짝수 ep 로 bin 중심 fit → 홀수 ep 의 τ̂ 슬로프
(c) 순서 셔플 null: episode 내 record 순서를 섞고 경로길이 재계산 (실경로/셔플경로 비)
(d) scene 간 경로방향 일치: scene별 q1→q4 방향의 쌍별 cos 중앙값
"""
import json
import sys

import numpy as np

SHARD, PHNAME = sys.argv[1], sys.argv[2]
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NQ = 4
rng = np.random.default_rng(0)

d = np.load(SHARD, allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
cap = [int(x) for x in meta["capture_layers"]]
X = d["X"][:, cap.index(LAYER), 3, 3, :].astype(np.float64)
cb = {k: int(v) for k, v in meta["phase_codebook"].items()}
pc = cb[PHNAME]
succ = d["succ"].astype(bool)

# scene 중심화 (성공 해당-phase 평균 기준)
for s in set(d["scene"]):
    m = d["scene"] == s
    ref = X[m & succ & (d["phase_code"] == pc)]
    X[m] = X[m] - (ref.mean(0) if len(ref) >= 5 else X[m].mean(0))

eps = []
for e in set(d["ep_id"][(d["phase_code"] == pc) & succ]):
    idx = np.where((d["ep_id"] == e) & (d["phase_code"] == pc))[0]
    idx = idx[np.argsort(d["rec_idx"][idx])]
    if len(idx) >= NQ:
        eps.append(idx)


def qmeans(ep_list, order=None):
    rows = [[] for _ in range(NQ)]
    for idx in ep_list:
        seq = idx if order is None else rng.permutation(idx)
        for j, i in enumerate(seq):
            rows[min(NQ - 1, j * NQ // len(seq))].append(i)
    return np.stack([X[np.array(r)].mean(0) for r in rows])


M = qmeans(eps)
path_dir = M[-1] - M[0]
path_dir /= np.linalg.norm(path_dir)
path_len = sum(np.linalg.norm(M[i + 1] - M[i]) for i in range(NQ - 1))

# (a) episode별 변위 정렬
cs = []
for idx in eps:
    k = max(1, len(idx) // 4)
    delta = X[idx[-k:]].mean(0) - X[idx[:k]].mean(0)
    n = np.linalg.norm(delta)
    if n > 1e-9:
        cs.append(float(delta @ path_dir / n))
cs = np.array(cs)
print(f"(a) episode 변위·경로 cos: median {np.median(cs):+.2f}, >0 비율 "
      f"{np.mean(cs > 0):.2f} (n={len(cs)}, 랜덤 null ≈ 0±{1/np.sqrt(X.shape[1]):.3f})")

# (b) held-out 슬로프
fit_eps, ho_eps = eps[::2], eps[1::2]
Mf = qmeans(fit_eps)
slopes = []
for idx in ho_eps:
    tau = np.linalg.norm(X[idx][:, None, :] - Mf[None], axis=2).argmin(1) / (NQ - 1)
    pos = np.arange(len(idx)) / max(1, len(idx) - 1)
    slopes.append(float(np.polyfit(pos, tau, 1)[0]))
print(f"(b) held-out τ̂ 슬로프: median {np.median(slopes):+.2f} "
      f"(fit {len(fit_eps)}ep → held-out {len(ho_eps)}ep; 1=전진, 0=무구조)")

# (c) 순서 셔플 null
sh = [sum(np.linalg.norm(m[i + 1] - m[i]) for i in range(NQ - 1))
      for m in (qmeans(eps, order="shuffle") for _ in range(20))]
print(f"(c) 경로길이 실측 {path_len:.1f} vs 순서셔플 {np.mean(sh):.1f}±{np.std(sh):.1f} "
      f"(비 {path_len/np.mean(sh):.1f}배)")

# (d) scene 간 경로방향 일치
dirs = []
for s in set(d["scene"]):
    se = [idx for idx in eps if d["scene"][idx[0]] == s]
    if len(se) < 3:
        continue
    Ms = qmeans(se)
    v = Ms[-1] - Ms[0]
    n = np.linalg.norm(v)
    if n > 1e-9:
        dirs.append(v / n)
pair = [float(dirs[i] @ dirs[j]) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
print(f"(d) scene 간 경로방향 cos: median {np.median(pair):+.2f} "
      f"(scene {len(dirs)}개, 쌍 {len(pair)}개, 랜덤 null ≈ 0)")
