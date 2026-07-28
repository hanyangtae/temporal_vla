#!/usr/bin/env python3
"""exp5-4 Phase A: probe 활성 → 후보 순위 확정·봉인 (rollout **전에** 실행).

이 스크립트가 만드는 selection_manifest.tsv + .sha256 가 없으면 probe_collect.sh
--stage rollout 은 진행하지 않는다 (Gate2 P1). 봉인 시점이 rollout 이전임을 파일 해시로
증명하는 것이 목적 — 계획서 §3-1 "사후 선택이다" 비판에 대한 유일한 방어다.

점수 = 방향 w 와 record 0 활성의 내적. 활성은 probe npz 의 [k, L, K, T, D] 에서
지정 layer 를 골라 K(denoise)·T(token) 평균 → [k, D] 로 축약한다 (exp5-3 ~/sm_npz 축약과
동일 규약). 점수는 **작을수록 성공 예측**(방향이 fail−succ 평균차라서) — 오름차순 rank 1
= top1.

방향 NPZ 계약 (--direction-npz, 옛 0~7e6 수집으로 LOSO fit 한 산출물):
  · scene 별 방향: 키 ``dir_<scene>`` [D] (해당 scene 을 뺀 LOSO 방향 — 권장)
  · 공통 방향   : 키 ``direction`` [D]
  scene 별 키가 있으면 우선 사용하고, 없으면 공통 방향으로 폴백한다. 둘 다 없으면 abort.
  (fit 산출물은 아직 없음 — 인터페이스만 고정해 둔 상태.)

동률 처리: 점수 동률이면 base_seed 오름차순 (사전 고정).
random_pick: scene 별 RNG(424101 + scene) 로 사전 추첨한 후보 1개 (위약 arm).

출력 TSV 열:
  scene cand_idx base_seed score rank is_top1 is_worst1 random_pick direction_sha code_sha
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

RNG_BASE = 424101


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _code_sha() -> str:
    """git HEAD (봉인 시점 코드 상태). git 이 없으면 'nogit'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[6]),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "nogit"


def _reduce(hidden: np.ndarray, layer_pos: int) -> np.ndarray:
    """[k, L, K, T, D] → [k, D] (지정 layer, denoise·token 평균)."""
    if hidden.ndim != 5:
        raise SystemExit(f"probe hidden 은 [k,L,K,T,D] 여야 함: {hidden.shape}")
    return hidden[:, layer_pos].mean(axis=(1, 2)).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True, type=Path, help="scene*.npz 디렉토리")
    ap.add_argument("--seed-manifest", required=True, type=Path)
    ap.add_argument(
        "--direction-npz",
        required=True,
        type=Path,
        help="옛 0~7e6 수집 LOSO fit 방향 (dir_<scene> 또는 direction 키)",
    )
    ap.add_argument("--out", required=True, type=Path, help="selection_manifest.tsv")
    ap.add_argument("--layer", type=int, default=0, help="사영에 쓸 capture layer (기본 L0)")
    args = ap.parse_args()

    rows = []
    with args.seed_manifest.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] in ("", "scene_idx") or parts[0].startswith("#"):
                continue
            rows.append(
                {
                    "scene_idx": int(parts[0]),
                    "scene": int(parts[1]),
                    "cand_idx": int(parts[2]),
                    "base_seed": int(parts[3]),
                }
            )
    if not rows:
        raise SystemExit("seed manifest 행 0")

    dnpz = np.load(args.direction_npz)
    direction_sha = _sha256_file(args.direction_npz)
    common_dir = np.asarray(dnpz["direction"], dtype=np.float64) if "direction" in dnpz.files else None
    code_sha = _code_sha()

    scenes = sorted({r["scene"] for r in rows})
    out_rows = []
    for scene in scenes:
        srows = sorted((r for r in rows if r["scene"] == scene), key=lambda r: r["cand_idx"])
        npz_path = args.probe_dir / f"scene{scene}.npz"
        if not npz_path.exists():
            raise SystemExit(f"probe npz 없음: {npz_path}")
        z = np.load(npz_path)
        seeds = [int(s) for s in z["seeds"]]
        want = [r["base_seed"] for r in srows]
        if sorted(seeds) != sorted(want):
            raise SystemExit(
                f"scene {scene}: probe npz seeds 가 manifest 와 불일치 {seeds} vs {want}"
            )
        layers = [int(x) for x in z["capture_layers"]]
        if args.layer not in layers:
            raise SystemExit(f"scene {scene}: layer {args.layer} 미캡처 (있는 층 {layers})")
        feats = _reduce(np.asarray(z["hidden"]), layers.index(args.layer))  # [k, D]

        key = f"dir_{scene}"
        if key in dnpz.files:
            w = np.asarray(dnpz[key], dtype=np.float64)
        elif common_dir is not None:
            w = common_dir
        else:
            raise SystemExit(
                f"방향 NPZ 에 '{key}' 도 'direction' 도 없음: {args.direction_npz}"
            )
        if w.shape[-1] != feats.shape[-1]:
            raise SystemExit(f"scene {scene}: 방향 차원 {w.shape} != 활성 {feats.shape}")
        w = w / (np.linalg.norm(w) or 1.0)

        # 점수: scene 내 상대 비교만 하므로 오프셋 보정 불필요 (계획서 §1 ①)
        scores = {int(s): float(feats[i] @ w) for i, s in enumerate(seeds)}
        # 오름차순(= 성공 예측 우선), 동률은 base_seed 오름차순 — 사전 고정 tie-break
        order = sorted(srows, key=lambda r: (scores[r["base_seed"]], r["base_seed"]))
        rank_of = {r["base_seed"]: i + 1 for i, r in enumerate(order)}
        top1 = order[0]["base_seed"]
        worst1 = order[-1]["base_seed"]
        rng = np.random.default_rng(RNG_BASE + scene)
        rand_pick = srows[int(rng.integers(len(srows)))]["base_seed"]

        for r in srows:
            b = r["base_seed"]
            out_rows.append(
                (
                    scene,
                    r["cand_idx"],
                    b,
                    f"{scores[b]:.9g}",
                    rank_of[b],
                    int(b == top1),
                    int(b == worst1),
                    int(b == rand_pick),
                    direction_sha[:16],
                    code_sha[:12],
                )
            )

    header = (
        "scene\tcand_idx\tbase_seed\tscore\trank\tis_top1\tis_worst1\t"
        "random_pick\tdirection_sha\tcode_sha"
    )
    text = "\n".join([header] + ["\t".join(str(v) for v in r) for r in out_rows]) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    sha_path = args.out.with_suffix(".sha256")
    sha_path.write_text(f"{sha}  {args.out.name}\n")
    meta = {
        "layer": args.layer,
        "direction_npz": str(args.direction_npz),
        "direction_sha256": direction_sha,
        "code_sha": code_sha,
        "seed_manifest_sha256": _sha256_file(args.seed_manifest),
        "probe_dir": str(args.probe_dir),
        "scenes": len(scenes),
        "rows": len(out_rows),
        "tie_break": "score asc, then base_seed asc",
        "random_pick_rng": f"default_rng({RNG_BASE} + scene)",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}: scenes={len(scenes)} rows={len(out_rows)} layer=L{args.layer}")
    print(f"sha256={sha}  (봉인: {sha_path.name})")


if __name__ == "__main__":
    main()
