"""grid v2 전 격자 압축 캐시 추출 — task별 npz (L12·last-denoise·전토큰 mean).

출력: ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/<slug>.npz
  X (N,1536) fp16 · P (N,18) proprio+속도 · phase_code (N) · ep_id (N) ·
  scene (N) · noise (N) · succ (N) · meta_json (codebook, layer=12, denoise=last)
docs/04: 분석 산출물, 절대경로 미기록.
"""
import glob, json, os, pickle, sys
import numpy as np

GRID = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/grid/")
OUT = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
os.makedirs(OUT, exist_ok=True)
TASKS = {"OpenDrawer/left": "OpenDrawer_left", "OpenDrawer/right": "OpenDrawer_right",
         "DishwasherRack/out": "DishwasherRack_out", "OvenRack/out": "OvenRack_out",
         "PPCC/candle": "PPCC_candle", "PPCC/bread": "PPCC_bread",
         "PPCC/marshmallow": "PPCC_marshmallow", "PPCC/jug": "PPCC_jug",
         "PPCC/apple": "PPCC_apple", "CoffeeSetupMug": "CoffeeSetupMug"}

all_pkls = glob.glob(GRID + "**/base/rollout.pkl", recursive=True)
print(f"total pkl: {len(all_pkls)}", flush=True)

for tkey, slug in TASKS.items():
    out_f = OUT + slug + ".npz"
    if os.path.exists(out_f):
        print(f"[skip] {slug} 존재", flush=True)
        continue
    pkls = sorted(p for p in all_pkls if f"/{tkey}/s" in p)
    Xs, Ps, PH, EP, SC, NO, SU = [], [], [], [], [], [], []
    codebook = {}
    seen = set()
    n_err = 0
    for ep_i, p in enumerate(pkls):
        try:
            d = pickle.load(open(p, "rb"))
            cell = (int(d["scene_idx"]), int(d["noise_idx"]))
            if cell in seen:
                continue
            seen.add(cell)
            cap = [int(x) for x in d["capture_layers"]]
            li = cap.index(12)
            H = np.stack([np.asarray(h[li, 3], dtype=np.float32).mean(0)
                          for h in d["hidden_states"]]).astype(np.float16)
            st = [np.concatenate([np.asarray(s["observation.state.eef_pos_rel"]),
                                  np.asarray(s["observation.state.eef_quat_rel"]),
                                  np.asarray(s["observation.state.gripper_qpos"])])
                  for s in d["states"]]
            P0 = np.stack(st).astype(np.float32)
            P = np.hstack([P0, np.vstack([np.zeros((1, 9), np.float32), np.diff(P0, axis=0)])])
            for ph in d["feature_phases"]:
                if ph not in codebook:
                    codebook[ph] = len(codebook)
            pc = np.array([codebook[ph] for ph in d["feature_phases"]], np.int16)
            T = len(H)
            Xs.append(H); Ps.append(P); PH.append(pc)
            EP.append(np.full(T, len(seen) - 1, np.int32))
            SC.append(np.full(T, cell[0], np.int16)); NO.append(np.full(T, cell[1], np.int16))
            SU.append(np.full(T, int(d["episode_success"]), np.int8))
        except Exception as e:
            n_err += 1
            print(f"  err {p.split('/grid/')[-1]}: {e}", flush=True)
    if not Xs:
        print(f"[empty] {slug}", flush=True)
        continue
    np.savez_compressed(out_f, X=np.concatenate(Xs), P=np.concatenate(Ps),
                        phase_code=np.concatenate(PH), ep_id=np.concatenate(EP),
                        scene=np.concatenate(SC), noise=np.concatenate(NO),
                        succ=np.concatenate(SU),
                        meta_json=json.dumps({"phase_codebook": codebook, "layer": 12,
                                              "denoise": "last(idx3)", "token": "mean",
                                              "src": "grid_v2", "err": n_err}))
    n_ep = len(seen)
    print(f"[done] {slug}: ep={n_ep} rec={sum(len(x) for x in Xs)} err={n_err}", flush=True)
print("SEGB_EXTRACT_DONE", flush=True)
