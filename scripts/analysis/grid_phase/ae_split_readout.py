#!/usr/bin/env python
"""누수 제거판 판독: scaler·AE까지 train scene 만으로 적합해 unseen-scene 판독을 재측정.

배경 (Codex 리뷰 ⑤): 기존 파이프(ae_cluster.py)는 AE·표준화를 930판 전체로 한 번
학습한 latent 를 쓰고, phase_readout.py 가 그 위에서 scene 분할을 적용했다.
라벨은 안 들어가지만 표현 학습이 test scene 의 activation 을 본 셈이므로,
여기서는 **split(seed 0/1/2)마다** 다음을 train scene 만으로 다시 한다:

    표준화(standardize_fit) → AE(train_ae) → KMeans(k) → 군집→최빈 phase 매핑

test scene 은 (train 통계로 표준화 → train AE 인코딩 → train 중심 최근접 배정)만
거친다. split 선택은 phase_readout.run_one 과 bit-동일한 규칙
(default_rng(seed).choice, ok=phase_code>=0 필터 후) 을 쓴다.

원격(승준) 실행용 — shard 는 segA 트리, torch 는 anaconda python.
usage:
    python ae_split_readout.py --shard-dir <segA> --out <json> \
        [--shards A.npz,B.npz] [--seeds 0,1,2] [--k 8] [--latent 16] [--epochs 200]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--shards", default=None, help="쉼표 목록 (기본: 전부)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ks", default=None,
                    help="k sweep 목록(쉼표). 지정 시 --k 대신 각 k로 판독")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--test-scenes", type=int, default=2)
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    AC = load_module(here / "ae_cluster.py", "ae_cluster_mod")
    PR = load_module(here / "phase_readout.py", "phase_readout_mod")
    IP = load_module(here / "intrinsic_phase.py", "intrinsic_phase_mod")

    names = args.shards.split(",") if args.shards else None
    paths = IP.discover_shards(args.shard_dir.expanduser(), names)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {}
    if args.out.is_file():                     # 이어달리기 (shard 단위 재개)
        results = json.loads(args.out.read_text()).get("per_instruction", {})

    for p in paths:
        slug = p.stem
        if slug in results and len(results[slug].get("runs", [])) == len(seeds):
            print(f"[skip] {slug} (완료)", flush=True)
            continue
        t0 = time.time()
        s = IP.load_shard(p)
        feat = s["feat"]
        y = s["phase_code"].astype(np.int64)
        scene = s["scene"].astype(np.int64)
        ok = y >= 0
        F, Y, SC = feat[ok], y[ok], scene[ok]
        del feat, s
        print(f"[load] {slug}: rec {len(F)} ({time.time()-t0:.0f}s)", flush=True)

        runs = []
        for seed in seeds:
            scenes = np.unique(SC)
            rng = np.random.default_rng(seed)
            test_scenes = rng.choice(
                scenes, size=min(args.test_scenes, max(len(scenes) - 1, 1)),
                replace=False)
            te = np.isin(SC, test_scenes)
            tr = ~te

            # ---- 누수 제거 핵심: scaler·AE 를 train 에서만 적합 ----
            scaler = AC.standardize_fit(F[tr])
            Xtr = AC.standardize_apply(F[tr], scaler)
            model, summ = AC.train_ae(Xtr, args.latent, args.epochs,
                                      args.batch_size, 1000 + seed,
                                      patience=args.patience)
            Ztr = AC.encode(model, Xtr)
            del Xtr
            Zte = AC.encode(model, AC.standardize_apply(F[te], scaler))

            # ---- 판독 (phase_readout.run_one 과 동일 절차; k sweep 은 AE 재사용) ----
            classes = np.unique(Y)
            fallback = int(np.bincount(Y[tr]).argmax())
            ks = ([int(x) for x in args.ks.split(",")] if args.ks else [args.k])
            per_k = {}
            for kk in ks:
                C = PR.kmeans(Ztr.astype(np.float64), kk, seed=seed)
                m = PR.majority_map(PR.assign(Ztr.astype(np.float64), C), Y[tr],
                                    kk, fallback)
                pred = m[PR.assign(Zte.astype(np.float64), C)]
                per_k[str(kk)] = {
                    "cluster_acc": float((pred == Y[te]).mean()),
                    "cluster_f1": PR.macro_f1(Y[te], pred, classes)}
            base = per_k[str(args.k)] if str(args.k) in per_k else per_k[str(ks[0])]
            runs.append({"seed": seed, **base, "per_k": per_k,
                         "test_scenes": [int(x) for x in np.sort(test_scenes)],
                         "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "ae_epochs": int(summ.get("best_epoch", -1))})
            print(f"  [{slug} seed{seed}] " + " ".join(
                f"k{kk}:{per_k[str(kk)]['cluster_acc']:.3f}" for kk in ks), flush=True)
            del Ztr, Zte, model

        entry = {"runs": runs,
                 "cluster_acc": float(np.mean([r["cluster_acc"] for r in runs])),
                 "cluster_f1": float(np.mean([r["cluster_f1"] for r in runs]))}
        results[slug] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "per_instruction": results,
            "meta": {"script": "ae_split_readout.py",
                     "protocol": "scaler·AE·KMeans·매핑 전부 train scene 적합 "
                                 "(누수 제거판); split 규칙은 phase_readout 동일",
                     "k": args.k, "latent": args.latent, "epochs": args.epochs,
                     "seeds": seeds}}, ensure_ascii=False, indent=1))
        print(f"[{slug}] mean acc {entry['cluster_acc']:.3f} "
              f"f1 {entry['cluster_f1']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    accs = [v["cluster_acc"] for v in results.values()]
    if accs:
        print(f"\nSPLIT_AE_DONE n={len(accs)} 중앙값 acc {np.median(accs):.3f}")


if __name__ == "__main__":
    main()
