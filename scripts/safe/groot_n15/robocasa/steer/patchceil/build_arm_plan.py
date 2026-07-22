"""patchceil arm 플랜 생성 (PROTOCOL.md 사전등록 규칙의 기계 번역, 결정적).

행 = rollout 1개: cell, target_ep, arm, donor_ep, npz(컨테이너 경로), t0(start_record),
donor_start, patch_len, inference_seed(collector --inference-seed 값), tag.

- 대상: targets_fit.tsv 실패 전량. donor/placebo = passB_manifest round-robin (ep 오름차순).
- t0 = target first_grasp, donor_start = donor first_grasp (patchceil_meta 기반).
- arms: nopatch / donor / placebo / shuffle (+ anchor_a2 1판, anchor_a3_sham 2판 per cell).
- anchor_a2: target 1판을 donor 의 inference_seed 로 실행 + 전창 패치 → actions == donor.

출력: patchceil/<cell>/arm_plan.tsv. stdlib 전용.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15/patchceil"
CONT_GROOT = "/temporal_vla/outputs/eval/robocasa/groot_n15/patchceil"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
ARMS = ["nopatch", "donor", "placebo", "shuffle"]


def first_grasp(cell: str, ep: int) -> int:
    d = json.loads((GROOT / "patchceil_meta" / cell / f"ep{ep}.json").read_text())
    fg = next(i for i, p in enumerate(d["feature_phases"]) if p == "grasp")
    return fg


def main() -> None:
    for cell in CELLS:
        targets = [
            int(r["episode_idx"])
            for r in csv.DictReader(open(GROOT / cell / "targets_fit.tsv"), delimiter="\t", lineterminator="\n")
            if r["role"] == "target"
        ]
        donors, placebos, shams = [], [], []
        for r in csv.DictReader(open(GROOT / cell / "passB_manifest.tsv"), delimiter="\t"):
            ep = int(r["episode_idx"])
            if r["role"] == "donor":
                donors.append(ep)
            else:
                placebos.append(ep)
                if r["role"] == "placebo+sham":
                    shams.append(ep)
        donors.sort()
        placebos.sort()
        targets.sort()

        def npz(ep: int, shuf: bool = False) -> str:
            suf = "_shuf" if shuf else ""
            return f"{CONT_GROOT}/{cell}/donors/ep{ep}_L15{suf}.npz"

        rows = []
        for i, t in enumerate(targets):
            d_ep = donors[i % len(donors)]
            p_ep = placebos[i % len(placebos)]
            if p_ep == t:  # placebo 는 target 자신 금지 (자기 이식 = sham 이 되어버림)
                p_ep = placebos[(i + 1) % len(placebos)]
            t0 = first_grasp(cell, t)
            base = {
                "cell": cell, "target_ep": t, "inference_seed": t * 1000,
                "patch_len": -1, "t0": t0,
            }
            # 빈 필드 금지: bash read 가 연속 탭(IFS whitespace)을 하나로 합쳐 필드가
            # 밀린다 (2026-07-16 nopatch 77판 argparse 즉사 사고). placeholder "-".
            rows.append({**base, "arm": "nopatch", "donor_ep": "-", "npz": "-",
                         "donor_start": "-", "tag": f"{cell}-ep{t}-nopatch"})
            rows.append({**base, "arm": "donor", "donor_ep": d_ep, "npz": npz(d_ep),
                         "donor_start": first_grasp(cell, d_ep),
                         "tag": f"{cell}-ep{t}-donor{d_ep}"})
            rows.append({**base, "arm": "placebo", "donor_ep": p_ep, "npz": npz(p_ep),
                         "donor_start": first_grasp(cell, p_ep),
                         "tag": f"{cell}-ep{t}-placebo{p_ep}"})
            rows.append({**base, "arm": "shuffle", "donor_ep": d_ep,
                         "npz": npz(d_ep, shuf=True),
                         "donor_start": first_grasp(cell, d_ep),
                         "tag": f"{cell}-ep{t}-shuf{d_ep}"})
        # anchor A2 — cross-scene action-equivalence: **다른 cell 의 env** 에서 이 cell 의
        # donor seed + 전창 L15 patch → emitted actions == donor 저장 actions (정확 일치).
        # 같은 scene+같은 seed 는 patch 가 identity 라 검증력이 없다 (upstream 이 이미 donor).
        other = next(c for c in CELLS if c != cell)
        a2_d = donors[0]
        rows.append({
            "cell": other,  # env/scenario 는 상대 cell (upstream 을 강제로 다르게)
            "target_ep": a2_d,  # episode idx 는 donor 것 (스템 식별용)
            "arm": "anchor_a2", "donor_ep": a2_d,
            "npz": npz(a2_d), "t0": 0, "donor_start": 0, "patch_len": -1,
            "inference_seed": a2_d * 1000,  # donor 의 seed (ε 시퀀스 일치)
            "tag": f"{cell}-a2-env{other[-7:]}-d{a2_d}",
        })
        for s in shams:
            rows.append({
                "cell": cell, "target_ep": s, "arm": "anchor_a3_sham", "donor_ep": s,
                "npz": npz(s), "t0": 0, "donor_start": 0, "patch_len": -1,
                "inference_seed": s * 1000,
                "tag": f"{cell}-a3-ep{s}",
            })
        out = GROOT / cell / "arm_plan.tsv"
        cols = ["cell", "target_ep", "arm", "donor_ep", "npz", "t0", "donor_start",
                "patch_len", "inference_seed", "tag"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        n_by = {}
        for r in rows:
            n_by[r["arm"]] = n_by.get(r["arm"], 0) + 1
        print(f"{cell}: {len(rows)} rollouts {n_by} -> {out.name}")


if __name__ == "__main__":
    main()
