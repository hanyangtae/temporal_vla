"""patchceil primary 판정 (PROTOCOL §판정) — stdlib + exp3_decision(구 pq3_decision).exact_mcnemar import.

입력: rollouts/<arm>/raw_rollouts 결과(성공은 스템 succ0/1) + status_ep*.json (발화 창 검증).
출력: cell×arm 구제표, donor 별 분해, paired exact McNemar
  p_final = max(p_donor_vs_placebo, p_donor_vs_shuffle), cell 별 + 합산(stratified 병기).

무결성 규칙:
- patch arm 인데 status 의 armed tag 불일치·fired_total=0 → 해당 rollout 무효(discard 목록).
- nopatch 가 성공(결정론 위반) → 해당 target 판정 제외·보고.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15/patchceil"
TASK = "PickPlaceCounterToCabinet"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
MAIN_ARMS = ["nopatch", "donor", "placebo", "shuffle"]

sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/steer/exp3"))
from exp3_decision import exact_mcnemar_one_sided  # noqa: E402  (import 만 — 파일 수정 금지)


def rollout_succ(plan_cell: str, arm: str, env_cell: str, ep: int) -> int | None:
    rdir = GROOT / plan_cell / "rollouts" / arm / "raw_rollouts" / TASK / env_cell
    hits = sorted(list(rdir.glob(f"task5--ep{ep}--succ*.pkl")) + list(rdir.glob(f"task5--ep{ep}--succ*.json")))
    if not hits:
        return None
    return 1 if "--succ1" in hits[-1].name else 0


def status_ok(plan_cell: str, arm: str, ep: int, tag: str) -> tuple[bool, str]:
    p = GROOT / plan_cell / "rollouts" / arm / f"status_ep{ep}.json"
    if not p.exists():
        return False, "status 없음"
    try:
        s = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False, "status 파싱 실패"
    hook = s.get("hooks", {}).get("15", {})
    if arm == "nopatch":
        return (not hook.get("armed", True) or hook.get("fired_total", 0) == 0,
                "nopatch 인데 발화" if hook.get("fired_total", 0) else "")
    if hook.get("tag") != tag:
        return False, f"tag 불일치 {hook.get('tag')}!={tag}"
    if hook.get("fired_total", 0) <= 0:
        return False, "발화 0"
    fr = hook.get("fired_records") or []
    if fr and fr[0] != hook.get("start_record"):
        return False, f"발화 시작 {fr[0]} != t0 {hook.get('start_record')}"
    return True, ""


def main() -> int:
    pooled = {}
    for cell in CELLS:
        plan = [r for r in csv.DictReader(open(GROOT / cell / "arm_plan.tsv"), delimiter="\t")
                if r["arm"] in MAIN_ARMS]
        targets = sorted({int(r["target_ep"]) for r in plan})
        res: dict[tuple[int, str], int | None] = {}
        invalid: list[str] = []
        donor_of: dict[int, int] = {}
        for r in plan:
            ep, arm = int(r["target_ep"]), r["arm"]
            s = rollout_succ(cell, arm, r["cell"], ep)
            ok, why = status_ok(cell, arm, ep, r["tag"]) if s is not None else (True, "")
            if s is not None and not ok:
                invalid.append(f"{r['tag']}: {why}")
                s = None
            res[(ep, arm)] = s
            if arm == "donor":
                donor_of[ep] = int(r["donor_ep"])

        # nopatch 결정론 위반 target 제외
        excluded = [ep for ep in targets if res.get((ep, "nopatch")) == 1]
        valid_t = [ep for ep in targets if ep not in excluded]
        print(f"\n===== {cell} (targets {len(targets)}, 제외 {len(excluded)}: {excluded}) =====")
        if invalid:
            print(f"  무효 rollout {len(invalid)}: " + "; ".join(invalid[:5]) + (" …" if len(invalid) > 5 else ""))
        for arm in MAIN_ARMS:
            xs = [res.get((ep, arm)) for ep in valid_t]
            done = [x for x in xs if x is not None]
            print(f"  {arm:8s} 구제 {sum(done)}/{len(done)} (미실행 {len(xs) - len(done)})")

        # donor 별 분해
        by_d: dict[int, list[int]] = {}
        for ep in valid_t:
            s = res.get((ep, "donor"))
            if s is not None:
                by_d.setdefault(donor_of[ep], []).append(s)
        print("  donor별: " + ", ".join(f"ep{d}:{sum(v)}/{len(v)}" for d, v in sorted(by_d.items())))

        # paired McNemar
        for ctrl in ("placebo", "shuffle"):
            b = sum(1 for ep in valid_t
                    if res.get((ep, "donor")) == 1 and res.get((ep, ctrl)) == 0)
            c = sum(1 for ep in valid_t
                    if res.get((ep, "donor")) == 0 and res.get((ep, ctrl)) == 1)
            p = exact_mcnemar_one_sided(b, c)
            print(f"  donor vs {ctrl}: b={b} c={c} p={p:.4f}")
            k = ("pool", ctrl)
            pb, pc = pooled.get(k, (0, 0))
            pooled[k] = (pb + b, pc + c)

    print("\n===== pooled (cell-stratified discordant 합산) =====")
    ps = []
    for (_, ctrl), (b, c) in sorted(pooled.items()):
        p = exact_mcnemar_one_sided(b, c)
        ps.append(p)
        print(f"  donor vs {ctrl}: b={b} c={c} p={p:.4f}")
    if ps:
        print(f"  primary p_final = max = {max(ps):.4f} (α=0.05)")
        print("  → " + ("유의: exploratory 진행 가능" if max(ps) < 0.05 else
                        "비유의: exploratory 미실행 (PROTOCOL hierarchical gate)"))
    print("\nclaim 등급: intervention effect — specified donor-trajectory transplant, cell-conditional")
    return 0


if __name__ == "__main__":
    sys.exit(main())
