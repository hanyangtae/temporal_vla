"""P1 후보 추출: refit 산출물에서 cell/tier 별 layer·α 후보 JSON 생성.

후보 규칙 (docs/steering/17 + Gate1):
  - single = global 그룹 dit layer 중 quota_steer(선택 α 기준) top-1
  - multi  = 현행 4-8-12 고정 (직전 라운드 유산 config)
  - α = 각 NPZ 의 selected_alpha (serve loader 가 meta 에서 자동 적용 — 5615da9)
  - selection_mode=closest(overlap 밴드 미도달) 는 경고 플래그 — P2 포함 여부 사전 결정용
  - gated 실행가능성: 5개 phase 그룹 fit 존재 여부 기록

usage: p1_candidates.py --fits-root <OUT_ROOT> --out <json> <stem>...
  stem 예: ppcc_bread/per_scene_fit15 (fits-root 상대)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASES = ["reach-to-object", "grasp", "transport", "place", "insert-settle"]
MULTI = [4, 8, 12]


def scan_stem(root: Path, stem: str) -> dict:
    base = root / stem
    summary_p = base / "fit_summary.json"
    if not summary_p.exists():
        return {"stem": stem, "status": "missing"}
    summary = json.loads(summary_p.read_text())
    if not summary:
        return {"stem": stem, "status": "empty"}
    layers = {}
    for tag_dir in sorted((base / "global").glob("dit_L*")):
        m = json.loads((tag_dir / "metadata.json").read_text())
        a = f"{m['selected_alpha']:g}"
        layers[m["steering_layer"]] = {
            "layer_tag": m["layer_tag"], "selected_alpha": m["selected_alpha"],
            "selection_mode": m["selection_mode"],
            "quota_steer_at_sel": m.get("quota_steer", {}).get(a),
            "n_eps": [m.get("n_success_eps"), m.get("n_failure_eps")],
        }
    if not layers:
        return {"stem": stem, "status": "no_global_dit"}
    scored = {L: v for L, v in layers.items() if v["quota_steer_at_sel"] is not None}
    top1 = max(scored, key=lambda L: scored[L]["quota_steer_at_sel"]) if scored else None
    phase_cover = {}
    for L, v in layers.items():
        tags = [ph for ph in PHASES if (base / ph / v["layer_tag"] / "conceptors.npz").exists()]
        phase_cover[L] = tags
    multi_ok = all(L in layers for L in MULTI)
    return {
        "stem": stem, "status": "ok",
        "candidates": {
            "single_quota_top1": None if top1 is None else {
                "layer": top1, **layers[top1], "phase_groups": phase_cover.get(top1, [])},
            "multi_4_8_12": {
                "layers": MULTI, "available": multi_ok,
                "per_layer": {str(L): layers.get(L) for L in MULTI},
                "phase_groups": {str(L): phase_cover.get(L, []) for L in MULTI}},
        },
        "warnings": sorted({f"L{L}:closest" for L, v in layers.items()
                            if v["selection_mode"] != "band"}),
        "all_layers_quota": {str(L): v["quota_steer_at_sel"] for L, v in sorted(layers.items())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits-root", default="outputs/eval/robocasa/groot_n15/pq2_analysis/conceptors")
    ap.add_argument("--out", required=True)
    ap.add_argument("stems", nargs="+")
    args = ap.parse_args()
    root = Path(args.fits_root)
    result = [scan_stem(root, s) for s in args.stems]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for r in result:
        c = r.get("candidates", {}).get("single_quota_top1")
        print(f"{r['stem']}: {r['status']}"
              + (f" top1=L{c['layer']}(q={c['quota_steer_at_sel']:.3f},α={c['selected_alpha']:g})"
                 f" warn={r['warnings']}" if c else ""))


if __name__ == "__main__":
    main()
