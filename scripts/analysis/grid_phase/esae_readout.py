#!/usr/bin/env python
"""Event-SAE(저자 파이프라인) 이벤트 군집을 우리 phase 판독 프로토콜로 평가.

입력
    <esae-root>/<slug>/event_features.jsonl   저자 build_event_features 출력 (무수정)
    <esae-root>/episodes_map.json             episode_num ↔ (slug, scene, noise)
    <labels-dir>/labels_<slug>_k*.npz         GT phase_code (record 해상도)

프로토콜 (phase_readout.py 와 동일)
    scene 단위 train/test 분리 (test 2 scene, split seed 0/1/2 평균).
    descriptor(z-score)·군집·cluster→최빈 phase 매핑 전부 train scene 에서만 적합.
    "학습에 쓰지 않은 scene" 의 **record 단위** 정확도·macro-F1 로 평가.

Event-SAE 를 record 판독으로 확장하는 규칙 (원 방법은 keyframe 단위 이벤트만 산출):
    1) descriptor = [1.0·L2(vision) ‖ 0.5·L2(zscore(state)) ‖ 0.4·zscore(progress)]
       → 행 L2  (저자 build_task_vectors 와 같은 수식; 단 z-score 통계는 train 에서만
       추정해 test 에 적용 — scene-split 프로토콜상 불가피한 유일한 변경)
    2) AgglomerativeClustering(cosine, distance_threshold 0.18) 을 train waypoint 에 적합,
       test waypoint 는 train cluster 중심과의 cosine 최근접으로 배정.
    3) record 예측 = 그 에피소드에서 시간상 가장 가까운 waypoint 의 cluster 가 갖는
       최빈 phase.  (waypoint frame f → env step 2f → record floor(2f/5))

실행 (sklearn 필요)
    ~/miniconda3/envs/event-sae-dev/bin/python scripts/analysis/grid_phase/esae_readout.py \
        --esae-root <esae 디렉토리> --labels-dir <labels 디렉토리>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def macro_f1(y_true, y_pred, classes) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def l2_rows(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)


def build_vectors(vision, state, progress, tr_mask,
                  w_vision=1.0, w_state=0.5, w_progress=0.4):
    """저자 build_task_vectors 수식. z-score 통계만 train 에서 추정."""
    mu_s, sd_s = state[tr_mask].mean(0), state[tr_mask].std(0)
    sd_s = np.where(sd_s > 1e-12, sd_s, 1.0)
    mu_p, sd_p = progress[tr_mask].mean(), progress[tr_mask].std()
    sd_p = sd_p if sd_p > 1e-12 else 1.0
    v = l2_rows(vision) * w_vision
    s = l2_rows((state - mu_s) / sd_s) * w_state
    p = ((progress - mu_p) / sd_p)[:, None] * w_progress
    return l2_rows(np.concatenate([v, s, p], axis=1))


def eval_slug(slug: str, feats: list[dict], ep_map: dict, lab_path: Path,
              seed: int, n_test_scene: int = 2, threshold: float = 0.18):
    d = np.load(lab_path, allow_pickle=True)
    y_all = d["phase_code"].astype(np.int64)
    ep_all = d["ep_id"].astype(np.int64)
    rec_all = d["rec_idx"].astype(np.int64)
    scene_all = d["scene"].astype(np.int64)
    noise_all = (d["noise"].astype(np.int64) if "noise" in d.files
                 else np.zeros(len(y_all), np.int64))
    ok = y_all >= 0
    y_all, ep_all, rec_all = y_all[ok], ep_all[ok], rec_all[ok]
    scene_all, noise_all = scene_all[ok], noise_all[ok]

    # NPZ ep_id ↔ esae episode_num 대응: (scene, noise) 키로 잇는다
    sn2npz = {}
    for e in np.unique(ep_all):
        m = ep_all == e
        sn2npz[(int(scene_all[m][0]), int(noise_all[m][0]))] = int(e)

    # waypoint 표: (npz_ep, record, scene) + vision/state/progress
    rows = []
    for r in feats:
        info = ep_map[str(r["episode_num"])]
        key = (info["scene"], info["noise"])
        if key not in sn2npz:
            continue
        wp_rec = (2 * int(r["waypoint_step"])) // 5
        rows.append((sn2npz[key], wp_rec, info["scene"],
                     np.asarray(r["vision_embedding"], np.float32),
                     np.asarray(r["state_vector"], np.float32),
                     float(r["progress_percent"])))
    if not rows:
        return None
    wp_ep = np.array([r[0] for r in rows])
    wp_rec = np.array([r[1] for r in rows])
    wp_scene = np.array([r[2] for r in rows])
    vision = np.stack([r[3] for r in rows])
    state = np.stack([r[4] for r in rows])
    progress = np.array([r[5] for r in rows])

    # phase_readout.run_one 과 동일한 scene split
    scenes = np.unique(scene_all)
    rng = np.random.default_rng(seed)
    test_scenes = rng.choice(scenes, size=min(n_test_scene, max(len(scenes) - 1, 1)),
                             replace=False)
    te_rec = np.isin(scene_all, test_scenes)
    tr_rec = ~te_rec
    tr_wp = ~np.isin(wp_scene, test_scenes)
    if tr_wp.sum() < 5 or te_rec.sum() < 10:
        return None

    # waypoint 의 GT phase = 그 record 의 phase (train 매핑에만 사용)
    rec2y = {(int(e), int(r)): int(v) for e, r, v in zip(ep_all, rec_all, y_all)}
    ep_maxrec = {}
    for e in np.unique(ep_all):
        ep_maxrec[int(e)] = int(rec_all[ep_all == e].max())
    wp_y = np.array([rec2y.get((int(e), min(int(r), ep_maxrec.get(int(e), 0))), -1)
                     for e, r in zip(wp_ep, wp_rec)])

    X = build_vectors(vision, state, progress, tr_wp)

    from sklearn.cluster import AgglomerativeClustering
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                 metric="cosine", linkage="average")
    lab_tr = cl.fit_predict(X[tr_wp])
    n_cl = int(lab_tr.max()) + 1
    cent = np.stack([X[tr_wp][lab_tr == c].mean(0) for c in range(n_cl)])
    cent = l2_rows(cent)

    classes = np.unique(y_all)
    fallback = int(np.bincount(y_all[tr_rec]).argmax())
    m_ok = wp_y[tr_wp] >= 0
    cl2phase = np.full(n_cl, fallback)
    for c in range(n_cl):
        yy = wp_y[tr_wp][(lab_tr == c) & m_ok]
        if len(yy):
            cl2phase[c] = int(np.bincount(yy).argmax())

    # test: waypoint → 최근접 train 중심 → phase; record → 시간상 최근접 waypoint
    te_wp = ~tr_wp
    pred_rec = np.full(len(y_all), fallback)
    if te_wp.sum():
        wp_cl_te = np.argmax(X[te_wp] @ cent.T, axis=1)
        wp_phase_te = cl2phase[wp_cl_te]
        for e in np.unique(ep_all[te_rec]):
            sel = (wp_ep == e) & te_wp
            if not sel.sum():
                continue
            wr = wp_rec[sel]
            wp_ph = wp_phase_te[sel[te_wp]]
            m = (ep_all == e)
            nearest = np.argmin(np.abs(rec_all[m][:, None] - wr[None, :]), axis=1)
            pred_rec[m] = wp_ph[nearest]

    acc = float((pred_rec[te_rec] == y_all[te_rec]).mean())
    f1 = macro_f1(y_all[te_rec], pred_rec[te_rec], classes)
    return {"esae_acc": acc, "esae_f1": f1, "n_clusters_train": n_cl,
            "n_wp_train": int(tr_wp.sum()), "n_wp_test": int(te_wp.sum()),
            "n_test_rec": int(te_rec.sum()),
            "test_scenes": [int(s) for s in np.sort(test_scenes)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--esae-root", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test-scenes", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.18)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/analysis/grid_phase/phase_readout/esae.json"))
    args = ap.parse_args(argv)

    gmap_path = args.esae_root / "episodes_map.json"
    global_map = json.loads(gmap_path.read_text()) if gmap_path.is_file() else {}
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    result = {}
    for run_dir in sorted(args.esae_root.iterdir()):
        fpath = run_dir / "event_features.jsonl"
        if not fpath.is_file():
            continue
        slug = run_dir.name
        # slug별 map(episode_num=scene*100+noise, v2) 우선, 없으면 전역 map(v1)
        smap_path = run_dir / "episodes_map.json"
        ep_map = (json.loads(smap_path.read_text()) if smap_path.is_file()
                  else global_map)
        labs = sorted(args.labels_dir.glob(f"labels_{slug}_k*.npz"))
        if not labs:
            print(f"[skip] {slug}: labels 없음")
            continue
        feats = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]
        runs = [r for r in (eval_slug(slug, feats, ep_map, labs[0], s,
                                      args.test_scenes, args.threshold)
                            for s in seeds) if r]
        if not runs:
            continue
        agg = {k: float(np.mean([r[k] for r in runs]))
               for k in ("esae_acc", "esae_f1", "n_clusters_train")}
        agg["n_seed"] = len(runs)
        agg["runs"] = runs
        result[slug] = agg
        print(f"[{slug:<22}] acc {agg['esae_acc']:.3f}  f1 {agg['esae_f1']:.3f}  "
              f"clusters(train) {agg['n_clusters_train']:.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "per_instruction": result,
        "meta": {"script": "scripts/analysis/grid_phase/esae_readout.py",
                 "method": "Event-SAE 이벤트 군집 (AWE pos_only 0.05 + SigLIP + "
                           "agglomerative cosine 0.18, 저자 파이프라인) → record 판독",
                 "protocol": "phase_readout 과 동일 scene split; z-score·군집·매핑 "
                             "train 에서만 적합; test waypoint 는 centroid 최근접",
                 "seeds": seeds, "test_scenes": args.test_scenes,
                 "threshold": args.threshold}},
        ensure_ascii=False, indent=1))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
