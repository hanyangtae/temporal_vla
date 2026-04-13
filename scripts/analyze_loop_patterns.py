"""GR00T 실패 에피소드 Loop 패턴 분석.

collect_groot_trajectories.py로 수집한 .npz trajectory에서
반복(루프) 패턴을 탐지하고 분류한다.

분석 항목:
1. Action cosine similarity — 연속된 action이 유사한 구간 탐지
2. State 변화량 정체 — state delta가 threshold 이하인 구간 탐지
3. Gripper 반복 — gripper open/close가 짧은 주기로 반복하는 패턴

사용법:
    python scripts/analyze_loop_patterns.py \
        --traj_dir outputs/trajectories/PnPCounterToCab \
        --output_dir outputs/analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ── 분석 함수 ──────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 cosine similarity. 둘 다 zero면 1.0 반환."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0  # zero action → 정체로 간주
    return float(np.dot(a, b) / (norm_a * norm_b))


def detect_action_repetition(
    actions: np.ndarray,
    cos_threshold: float = 0.98,
    min_run_length: int = 10,
) -> list[dict]:
    """연속 action cosine similarity가 threshold 이상인 구간을 탐지.

    Returns:
        list of {start, end, length, avg_cosine}
    """
    if len(actions) < 2:
        return []

    similarities = []
    for i in range(len(actions) - 1):
        similarities.append(cosine_similarity(actions[i], actions[i + 1]))
    similarities = np.array(similarities)

    # threshold 이상인 연속 구간 찾기
    above = similarities >= cos_threshold
    runs = []
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            length = i - start + 1
            if length >= min_run_length:
                runs.append({
                    "start": int(start),
                    "end": int(i),
                    "length": int(length),
                    "avg_cosine": float(np.mean(similarities[start:i])),
                })
            start = None
    if start is not None:
        length = len(above) - start + 1
        if length >= min_run_length:
            runs.append({
                "start": int(start),
                "end": int(len(above) - 1),
                "length": int(length),
                "avg_cosine": float(np.mean(similarities[start:])),
            })
    return runs


def detect_state_stagnation(
    states: np.ndarray,
    delta_threshold: float = 1e-3,
    min_run_length: int = 10,
) -> list[dict]:
    """연속 state 변화량이 threshold 이하인 구간을 탐지."""
    if len(states) < 2 or states.size == 0:
        return []

    deltas = np.linalg.norm(np.diff(states, axis=0), axis=-1)

    below = deltas <= delta_threshold
    runs = []
    start = None
    for i, v in enumerate(below):
        if v and start is None:
            start = i
        elif not v and start is not None:
            length = i - start + 1
            if length >= min_run_length:
                runs.append({
                    "start": int(start),
                    "end": int(i),
                    "length": int(length),
                    "avg_delta": float(np.mean(deltas[start:i])),
                })
            start = None
    if start is not None:
        length = len(below) - start + 1
        if length >= min_run_length:
            runs.append({
                "start": int(start),
                "end": int(len(below) - 1),
                "length": int(length),
                "avg_delta": float(np.mean(deltas[start:])),
            })
    return runs


def detect_gripper_oscillation(
    actions: np.ndarray,
    gripper_idx: int = -1,
    min_flips: int = 4,
    window_size: int = 20,
) -> list[dict]:
    """Gripper action의 부호가 짧은 구간에서 빈번히 바뀌는 패턴 탐지.

    gripper_idx: action 벡터에서 gripper에 해당하는 인덱스 (기본 -1 = 마지막).
    """
    if len(actions) < window_size:
        return []

    gripper = actions[:, gripper_idx]
    # 부호 변화 탐지
    sign_changes = np.diff(np.sign(gripper)) != 0

    oscillations = []
    for i in range(len(sign_changes) - window_size + 1):
        flips = int(np.sum(sign_changes[i:i + window_size]))
        if flips >= min_flips:
            oscillations.append({
                "start": int(i),
                "end": int(i + window_size),
                "n_flips": flips,
            })

    # 겹치는 구간 병합
    if not oscillations:
        return []

    merged = [oscillations[0]]
    for osc in oscillations[1:]:
        if osc["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], osc["end"])
            merged[-1]["n_flips"] = max(merged[-1]["n_flips"], osc["n_flips"])
        else:
            merged.append(osc)

    for m in merged:
        m["length"] = m["end"] - m["start"]

    return merged


def detect_action_norm_stagnation(
    actions: np.ndarray,
    norm_threshold: float = 0.01,
    min_run_length: int = 15,
) -> list[dict]:
    """Action norm이 매우 작은 구간 탐지 (모델이 아무것도 안 하는 상태)."""
    norms = np.linalg.norm(actions, axis=-1)
    below = norms <= norm_threshold

    runs = []
    start = None
    for i, v in enumerate(below):
        if v and start is None:
            start = i
        elif not v and start is not None:
            length = i - start
            if length >= min_run_length:
                runs.append({
                    "start": int(start),
                    "end": int(i - 1),
                    "length": int(length),
                    "avg_norm": float(np.mean(norms[start:i])),
                })
            start = None
    if start is not None:
        length = len(below) - start
        if length >= min_run_length:
            runs.append({
                "start": int(start),
                "end": int(len(below) - 1),
                "length": int(length),
                "avg_norm": float(np.mean(norms[start:])),
            })
    return runs


# ── 에피소드 분석 ──────────────────────────────────────────────


def analyze_episode(npz_path: Path) -> dict:
    """단일 에피소드 .npz 파일 분석."""
    data = np.load(npz_path, allow_pickle=True)
    actions = data["actions"]
    states = data["states"] if "states" in data and data["states"].size > 0 else None
    final_success = bool(data["final_success"])
    n_steps = int(data["n_steps"])

    result = {
        "file": str(npz_path.name),
        "success": final_success,
        "n_steps": n_steps,
        "action_shape": list(actions.shape),
    }

    # 1. Action repetition
    action_reps = detect_action_repetition(actions)
    result["action_repetition_runs"] = action_reps
    result["total_repetition_steps"] = sum(r["length"] for r in action_reps)

    # 2. State stagnation
    if states is not None and states.ndim == 2:
        state_stag = detect_state_stagnation(states)
        result["state_stagnation_runs"] = state_stag
        result["total_stagnation_steps"] = sum(r["length"] for r in state_stag)
    else:
        result["state_stagnation_runs"] = []
        result["total_stagnation_steps"] = 0

    # 3. Gripper oscillation
    if actions.shape[-1] >= 2:
        gripper_osc = detect_gripper_oscillation(actions)
        result["gripper_oscillation_runs"] = gripper_osc
        result["total_oscillation_steps"] = sum(r["length"] for r in gripper_osc)
    else:
        result["gripper_oscillation_runs"] = []
        result["total_oscillation_steps"] = 0

    # 4. Near-zero action (frozen)
    frozen_runs = detect_action_norm_stagnation(actions)
    result["frozen_runs"] = frozen_runs
    result["total_frozen_steps"] = sum(r["length"] for r in frozen_runs)

    # 루프 분류
    result["loop_type"] = classify_loop(result)

    return result


def classify_loop(ep_result: dict) -> str:
    """에피소드의 loop 유형을 분류."""
    if ep_result["success"]:
        return "success"

    n = ep_result["n_steps"]
    rep_ratio = ep_result["total_repetition_steps"] / max(n, 1)
    stag_ratio = ep_result["total_stagnation_steps"] / max(n, 1)
    osc_ratio = ep_result["total_oscillation_steps"] / max(n, 1)
    frozen_ratio = ep_result["total_frozen_steps"] / max(n, 1)

    if frozen_ratio > 0.3:
        return "frozen"  # 모델이 거의 아무것도 안 함
    if osc_ratio > 0.1:
        return "gripper_oscillation"  # gripper 반복
    if rep_ratio > 0.3 and stag_ratio > 0.2:
        return "stuck_loop"  # 같은 action 반복 + state 정체 = 전형적 실패 루프
    if rep_ratio > 0.3:
        return "action_repetition"  # action만 반복 (state는 변할 수 있음)
    if stag_ratio > 0.3:
        return "state_stagnation"  # state가 안 변하는데 action은 다양
    return "other_failure"  # 위 패턴에 해당하지 않는 실패


# ── 메인 ──────────────────────────────────────────────────────


def analyze_task(traj_dir: Path, output_dir: Path):
    """한 태스크의 전체 에피소드 분석."""
    npz_files = sorted(traj_dir.glob("ep_*.npz"))
    if not npz_files:
        print(f"No trajectory files found in {traj_dir}")
        return

    task_name = traj_dir.name
    print(f"\n{'='*60}")
    print(f"Task: {task_name} ({len(npz_files)} episodes)")
    print(f"{'='*60}")

    results = []
    for f in npz_files:
        r = analyze_episode(f)
        results.append(r)

    # 요약 통계
    n_total = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_fail = n_total - n_success

    loop_types = {}
    for r in results:
        lt = r["loop_type"]
        loop_types[lt] = loop_types.get(lt, 0) + 1

    summary = {
        "task": task_name,
        "n_episodes": n_total,
        "n_success": n_success,
        "n_fail": n_fail,
        "success_rate": n_success / max(n_total, 1),
        "loop_type_counts": loop_types,
        "episodes": results,
    }

    # 콘솔 출력
    print(f"  Success: {n_success}/{n_total} ({summary['success_rate']:.1%})")
    print(f"  Loop type breakdown (failures only):")
    for lt, cnt in sorted(loop_types.items()):
        if lt != "success":
            print(f"    {lt}: {cnt}")

    # 실패 에피소드 상세
    fail_eps = [r for r in results if not r["success"]]
    if fail_eps:
        print(f"\n  Failed episodes detail:")
        for r in fail_eps:
            print(
                f"    {r['file']}: type={r['loop_type']}, steps={r['n_steps']}, "
                f"rep={r['total_repetition_steps']}, stag={r['total_stagnation_steps']}, "
                f"osc={r['total_oscillation_steps']}, frozen={r['total_frozen_steps']}"
            )

    # JSON 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{task_name}_analysis.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved to {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze loop patterns in GR00T trajectories")
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="Trajectory directory (single task or parent of multiple tasks)")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    output_dir = Path(args.output_dir)

    # 단일 태스크 디렉토리인지 여러 태스크의 부모인지 판별
    npz_in_root = list(traj_dir.glob("ep_*.npz"))
    subdirs = [d for d in traj_dir.iterdir() if d.is_dir() and list(d.glob("ep_*.npz"))]

    all_summaries = []

    if npz_in_root:
        # 단일 태스크
        s = analyze_task(traj_dir, output_dir)
        if s:
            all_summaries.append(s)
    elif subdirs:
        # 여러 태스크
        for subdir in sorted(subdirs):
            s = analyze_task(subdir, output_dir)
            if s:
                all_summaries.append(s)
    else:
        print(f"No trajectory files found in {traj_dir}")
        return

    # 전체 요약
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY ({len(all_summaries)} tasks)")
        print(f"{'='*60}")

        total_ep = sum(s["n_episodes"] for s in all_summaries)
        total_success = sum(s["n_success"] for s in all_summaries)
        print(f"  Total episodes: {total_ep}")
        print(f"  Overall success rate: {total_success}/{total_ep} ({total_success/max(total_ep,1):.1%})")

        # 전체 loop type 합산
        all_types = {}
        for s in all_summaries:
            for lt, cnt in s["loop_type_counts"].items():
                all_types[lt] = all_types.get(lt, 0) + cnt

        print(f"\n  Loop type distribution (all failures):")
        for lt, cnt in sorted(all_types.items(), key=lambda x: -x[1]):
            if lt != "success":
                print(f"    {lt}: {cnt} ({cnt/max(total_ep-total_success,1):.1%} of failures)")

        # 전체 결과 JSON
        overall_file = output_dir / "overall_summary.json"
        with open(overall_file, "w") as f:
            json.dump({
                "n_tasks": len(all_summaries),
                "total_episodes": total_ep,
                "total_success": total_success,
                "overall_success_rate": total_success / max(total_ep, 1),
                "loop_type_distribution": all_types,
                "per_task": [{
                    "task": s["task"],
                    "success_rate": s["success_rate"],
                    "loop_types": s["loop_type_counts"],
                } for s in all_summaries],
            }, f, indent=2)
        print(f"\n  Overall summary saved to {overall_file}")


if __name__ == "__main__":
    main()
