#!/usr/bin/env python
"""exp5/G1 — SAE 입력 빌더: fit30 rollout pkl → layer 별 per-token 행렬 + 행 메타.

핸드아웃 `docs/steering/30_sae_g1_port_handout.md` §4 Phase B 구현.

**실행 위치 = 승준**(pkl 이 거기 있음). 그래서 이 파일은 **numpy/torch/stdlib 만** 쓴다
(승준 base python3 에는 torch 가 없으므로 `~/anaconda3/bin/python`, scipy·sklearn 없음).

핵심 계약 (2026-07-27 원격 정찰 실측):
  - manifest tsv: 헤더 없음, `#` 주석, 탭 3열 = pkl경로 / label(1=succ) / scenario_seed.
    cell 필터는 pkl 경로의 **부모 디렉토리 이름** (fit_mean_diff.load_cell_rolls:403 관례).
  - pkl["hidden_states"] = list[record], record = torch.Tensor fp16 [L=7, K=4, T=49, D=1536].
  - pkl["capture_layers"] = [0,2,4,8,10,12,15] → 물리 layer 는 반드시 `cap.index(L)` 로 인덱싱
    (리스트 위치 != layer 번호. 하드코딩 금지 — 핸드아웃 §6-7).
  - K(denoise) 축은 **평균** — fit_mean_diff.py:420-421 `np.asarray(rec).mean(axis=1)` 관례와 동일.
  - T=49 = state[0:1] + future[1:33] + action[33:49] (fit_phase_conceptor SEGMENTS 규약).
  - pkl["feature_phases"] len == record 수, ep_meta.layout_id/style_id, 최상위 scenario_seed·
    episode_idx·episode_success·inference_seed.

동료 레포(robots-oh task_classification) 빌더(`make_phase_dataset.py:102-109`)를 **재사용하지
않는 이유**: 그 빌더는 토큰 축을 평균으로 없앤다. 우리는 per-token 보존이 필수다
(memory `feedback-no-rollout-pooling`: phase 는 timestep 구분이 load-bearing).
PCA(동료 `phase/data/pca.py`)도 쓰지 않는다 — 원본 1536-d 에 overcomplete SAE 를 얹는 게
이번 재설계의 전제(핸드아웃 §2.2-2). 표준화는 train-split feature-wise mean/std 만.

메모리 주의: `np.asarray(rec[li], dtype=np.float32)` — **layer 인덱싱을 float32 변환보다 먼저**.
record 전체를 float32 로 올리면 7배 낭비(4배 dtype × layer 수).

산출 (--out-dir):
  X_L{L}.npz    : X [N_rows, 1536] fp16          (원시값. 표준화는 학습 시 적용)
  stats_L{L}.npz: mean [1536] / std [1536] float32 (train split 행만으로 계산)
  meta.npz      : 행 단위 메타 (아래 META_FIELDS) + phase 코드북
  split.json    : episode → split 배정표 (재현성)

사용 예:
  ~/anaconda3/bin/python scripts/scene_sae/build_sae_inputs.py \
      --manifest ~/datasets/.../manifests_fit30/pq3_drawer_left/fit_manifest.tsv \
      --cell pq3_drawer_left --layers 0,2,8,10,12 \
      --out-dir ~/workspace/temporal_vla/outputs/scene_sae/pq3_drawer_left
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

# 토큰 세그먼트 (N1.5 DiT T=49) — fit_phase_conceptor_n15.SEGMENTS 규약과 동일.
SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))

META_FIELDS = ("episode_idx", "record_idx", "token_idx", "token_seg", "phase_code",
               "success", "scenario_seed", "layout_id", "style_id", "split")

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------- manifest
def read_manifest(manifest: Path, cell: str) -> list[dict]:
    """manifest tsv → [{pkl, label, scene}] (fit_mean_diff.load_cell_rolls:393-410 관례)."""
    rows = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if cell and p.parent.name != cell:
            continue
        rows.append({"pkl": p, "label": int(parts[1]),
                     "scene": parts[2] if len(parts) > 2 else ""})
    if not rows:
        raise SystemExit(f"manifest 에 cell={cell} 행 없음: {manifest}")
    missing = [str(r["pkl"]) for r in rows if not r["pkl"].exists()]
    if missing:
        raise SystemExit(f"pkl 누락 {len(missing)}개: {missing[:3]}")
    return rows


def seg_of_token(t: int) -> int:
    for i, (_n, lo, hi) in enumerate(SEGMENTS):
        if lo <= t < hi:
            return i
    raise ValueError(f"token {t} 세그먼트 밖 (T 계약 위반)")


def token_seg_vector(T: int) -> np.ndarray:
    return np.asarray([seg_of_token(t) for t in range(T)], dtype=np.int8)


# ------------------------------------------------------------------------ split
def assign_splits(eps: list[dict], seed: int) -> dict[int, int]:
    """episode 단위 split 배정 (record/token split 은 자기상관 누수 — 핸드아웃 §6-2).

    층화 = scene 그룹 (layout_id, style_id) — fit30 drawer_left 에서 두 필드는 완전 공선이라
    사실상 layout 5종. 그룹 안에서 succ/fail 을 번갈아 배열해(성공률 쏠림 방지) 앞에서부터
    train → val → test 로 자른다. **모든 layout 이 train 과 test 양쪽에 나타나야** scene probe
    (라벨=layout) 가 held-out 평가 가능하므로:
      n == 1  → train (그 layout 은 probe 평가 불가, 경고)
      n == 2  → train 1 / test 1   (실측 layout (2,2) 케이스 — val 없음)
      n >= 3  → test = max(1, round(.20n)), val = max(1, round(.133n)) if n>=5 else 0,
                나머지 train
    실측 drawer_left(6/2/7/8/7 ep)에서 정확히 train 20 / val 4 / test 6 이 된다.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = {}
    for e in eps:
        groups.setdefault((e["layout_id"], e["style_id"]), []).append(e)

    out: dict[int, int] = {}
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        g = groups[key]
        succ = sorted([e for e in g if e["success"] == 1], key=lambda e: e["episode_idx"])
        fail = sorted([e for e in g if e["success"] != 1], key=lambda e: e["episode_idx"])
        rng.shuffle(succ)
        rng.shuffle(fail)
        order = []
        for i in range(max(len(succ), len(fail))):      # succ/fail 번갈아 = 성공률 균형
            if i < len(succ):
                order.append(succ[i])
            if i < len(fail):
                order.append(fail[i])
        n = len(order)
        if n == 1:
            n_test = n_val = 0
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test = max(1, int(round(0.20 * n)))
            n_val = max(1, int(round(0.1333 * n))) if n >= 5 else 0
            n_val = min(n_val, max(0, n - n_test - 1))
        n_train = n - n_test - n_val
        for i, e in enumerate(order):
            out[e["episode_idx"]] = 0 if i < n_train else (1 if i < n_train + n_val else 2)
    return out


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="fit30 pkl → layer 별 per-token SAE 입력")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cell", required=True, help="pkl 부모 디렉토리 이름 (예: pq3_drawer_left)")
    ap.add_argument("--layers", default="0,2,8,10,12",
                    help="물리 layer 목록. capture_layers 와의 교집합만 사용")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="X 저장 dtype")
    ap.add_argument("--seed", type=int, default=424101, help="split 배정 seed (exp4-1 관례)")
    ap.add_argument("--limit-eps", type=int, default=0, help=">0 이면 앞 N episode 만 (smoke)")
    args = ap.parse_args()

    want_layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = np.float16 if args.dtype == "fp16" else np.float32

    rows = read_manifest(args.manifest.expanduser(), args.cell)
    if args.limit_eps:
        rows = rows[: args.limit_eps]
    print(f"[build] cell={args.cell} episodes={len(rows)} layers={want_layers}", flush=True)

    # ---- 1 pass: 로드하며 layer 별 X 누적 + 행 메타 수집
    X_acc: dict[int, list[np.ndarray]] = {L: [] for L in want_layers}
    meta_acc: dict[str, list[np.ndarray]] = {f: [] for f in META_FIELDS if f != "split"}
    phase_codebook: dict[str, int] = {}
    eps_info: list[dict] = []
    contract = None
    t_start = time.time()

    for i, m in enumerate(rows):
        with open(m["pkl"], "rb") as f:
            d = pickle.load(f)
        hs = d.get("hidden_states") or []
        if not hs:
            raise SystemExit(f"{m['pkl']}: hidden_states 비어 있음")
        cap = list(d.get("capture_layers") or [])
        if not cap:
            raise SystemExit(f"{m['pkl']}: capture_layers 없음 — layer 인덱싱 불가")
        miss = [L for L in want_layers if L not in cap]
        if miss:
            raise SystemExit(f"{m['pkl']}: capture_layers={cap} 에 요청 layer {miss} 없음")
        li = {L: cap.index(L) for L in want_layers}      # 물리 layer → 리스트 위치

        phases = list(d.get("feature_phases") or [])
        if len(phases) != len(hs):
            raise SystemExit(f"{m['pkl']}: feature_phases {len(phases)} != records {len(hs)}")

        shape0 = tuple(np.shape(hs[0]))
        here = (tuple(cap), shape0[1], shape0[2], shape0[3])   # (cap, K, T, D)
        if contract is None:
            contract = here
            print(f"[build] contract capture_layers={cap} K={here[1]} T={here[2]} D={here[3]}",
                  flush=True)
        elif here != contract:
            raise SystemExit(f"{m['pkl']}: pkl 계약 불일치 {here} != {contract}")
        K, T, D = contract[1], contract[2], contract[3]
        seg_vec = token_seg_vector(T)

        ep_meta = d.get("ep_meta") or {}
        ep = {
            "episode_idx": int(d.get("episode_idx", i)),
            "success": int(m["label"]),          # manifest 라벨 override (fit_phase_conceptor 관례)
            "episode_success": int(d.get("episode_success", -1)),
            "scenario_seed": int(d.get("scenario_seed", -1)),
            "inference_seed": int(d.get("inference_seed", -1)),
            "layout_id": int(ep_meta.get("layout_id", -1)),
            "style_id": int(ep_meta.get("style_id", -1)),
            "n_records": len(hs),
            "pkl": str(m["pkl"]),
        }
        eps_info.append(ep)

        for L in want_layers:
            idx = li[L]
            # ★ layer 인덱싱을 float32 변환 **전에** (record 전체 변환 시 메모리 7배)
            chunk = np.empty((len(hs) * T, D), dtype=save_dtype)
            for r, rec in enumerate(hs):
                a = np.asarray(rec[idx], dtype=np.float32)       # [K, T, D]
                x = a.mean(axis=0)                               # K(denoise) 평균 — fit_mean_diff:421
                chunk[r * T:(r + 1) * T] = x.astype(save_dtype)
            X_acc[L].append(chunk)

        n_rec = len(hs)
        rec_idx = np.repeat(np.arange(n_rec, dtype=np.int32), T)
        tok_idx = np.tile(np.arange(T, dtype=np.int16), n_rec)
        for ph in phases:
            phase_codebook.setdefault(str(ph), len(phase_codebook))
        ph_code = np.repeat(
            np.asarray([phase_codebook[str(p)] for p in phases], dtype=np.int16), T)
        meta_acc["episode_idx"].append(np.full(n_rec * T, ep["episode_idx"], np.int32))
        meta_acc["record_idx"].append(rec_idx)
        meta_acc["token_idx"].append(tok_idx)
        meta_acc["token_seg"].append(np.tile(seg_vec, n_rec))
        meta_acc["phase_code"].append(ph_code)
        meta_acc["success"].append(np.full(n_rec * T, ep["success"], np.int8))
        meta_acc["scenario_seed"].append(np.full(n_rec * T, ep["scenario_seed"], np.int64))
        meta_acc["layout_id"].append(np.full(n_rec * T, ep["layout_id"], np.int32))
        meta_acc["style_id"].append(np.full(n_rec * T, ep["style_id"], np.int32))

        print(f"[build] ({i+1}/{len(rows)}) ep{ep['episode_idx']} succ={ep['success']} "
              f"rec={n_rec} layout={ep['layout_id']} style={ep['style_id']} "
              f"seed={ep['scenario_seed']} ({time.time()-t_start:.0f}s)", flush=True)

    # ---- split (episode 단위)
    ep2split = assign_splits(eps_info, args.seed)
    meta = {f: np.concatenate(v) for f, v in meta_acc.items()}
    split_row = np.zeros(len(meta["episode_idx"]), dtype=np.int8)
    for e_idx, s in ep2split.items():
        split_row[meta["episode_idx"] == e_idx] = s
    meta["split"] = split_row

    split_json = {
        "seed": args.seed, "cell": args.cell,
        "ratio_target": {"train": 0.667, "val": 0.133, "test": 0.20},
        "episodes": [{**{k: v for k, v in e.items() if k != "pkl"},
                      "split": SPLIT_NAMES[ep2split[e["episode_idx"]]],
                      "pkl": e["pkl"]} for e in eps_info],
    }
    (out_dir / "split.json").write_text(json.dumps(split_json, indent=2, ensure_ascii=False))

    np.savez_compressed(
        out_dir / "meta.npz",
        phase_codebook=np.asarray(json.dumps(phase_codebook, ensure_ascii=False)),
        split_names=np.asarray(SPLIT_NAMES),
        segment_names=np.asarray([s[0] for s in SEGMENTS]),
        **meta)

    # ---- layer 별 X + train-split 표준화 통계
    tr = meta["split"] == 0
    for L in want_layers:
        X = np.concatenate(X_acc[L], axis=0)
        X_acc[L] = []                                   # 즉시 해제 (메모리)
        if len(X) != len(split_row):
            raise SystemExit(f"L{L}: 행수 {len(X)} != meta {len(split_row)}")
        np.savez(out_dir / f"X_L{L}.npz", X=X)
        Xtr = X[tr].astype(np.float32)                  # 표준화 통계는 **train 행만** (누수 방지)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)               # 상수 feature 보호
        np.savez(out_dir / f"stats_L{L}.npz", mean=mu.astype(np.float32),
                 std=sd.astype(np.float32), n_train_rows=np.int64(int(tr.sum())))
        print(f"[build] saved X_L{L} shape={X.shape} dtype={X.dtype} "
              f"train_rows={int(tr.sum())}", flush=True)
        del X, Xtr

    # ---- 요약
    n_rows = len(split_row)
    print("\n===== 요약 =====")
    print(f"episodes={len(eps_info)}  records={sum(e['n_records'] for e in eps_info)}  rows={n_rows}")
    for s, name in enumerate(SPLIT_NAMES):
        sel = split_row == s
        eps_s = [e for e in eps_info if ep2split[e["episode_idx"]] == s]
        print(f"  {name:5s}: ep={len(eps_s):3d} (succ {sum(e['success'] for e in eps_s)}) "
              f"rows={int(sel.sum()):8d}")
    print("  layout 분포 (ep 기준):")
    lay = {}
    for e in eps_info:
        k = (e["layout_id"], e["style_id"])
        lay.setdefault(k, [0, 0, 0])[ep2split[e["episode_idx"]]] += 1
    for k in sorted(lay, key=lambda x: (str(x[0]), str(x[1]))):
        print(f"    layout={k[0]} style={k[1]}: train/val/test = {lay[k]}")
    print(f"  success rows: succ={int((meta['success'] == 1).sum())} "
          f"fail={int((meta['success'] != 1).sum())}")
    print(f"  phases={len(phase_codebook)} {sorted(phase_codebook)}")
    print(f"  산출물: {len(want_layers)} × X_L*.npz + {len(want_layers)} × stats_L*.npz "
          f"+ meta.npz + split.json  → {out_dir}")
    print("  (완료 판정은 로그 문자열이 아니라 위 산출물 **개수**로 — 핸드아웃 §6-6)")


if __name__ == "__main__":
    sys.exit(main())
