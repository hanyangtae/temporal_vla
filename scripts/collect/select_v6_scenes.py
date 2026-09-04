"""grid v6 — instruction 키 12개 × scene 3개 선택표 생성기.

정본 계약: handoff `v6_contract.md` §1~§3 (층위 = instruction 키 / scene(주방) / jitter j / noise n).
이 스크립트는 **scene 층(주방 = (layout, style) 쌍 + pull 계열의 스폰 side)** 만 확정한다.
jitter/noise 층과 plan 생성은 별도(에이전트 A, `build_v6_plan.py`) 소관이며 여기 산출물
`configs/collect/n15_grid_v6_scene_jitter/scene_selection.json` 을 입력으로 쓴다.

입력: `outputs/analysis/seed_scan/fixture_groups/*_target10_*.tsv`
  (주방 목록 target 10 = `layout_and_style_ids=[[1,1],…,[10,10]]` 로 돌린 seed 스캔)
  열: seed layout_id style_id fixture_key fixture_group should_pull rack_level lang
      base_x base_y base_yaw err

핵심 판정 두 가지
-----------------
1) **side (로봇 시점 좌/우)** — 오븐·식기세척기는 문을 연 채 시작해 정면 앵커가 막히고
   로봇이 좌/우 약 0.45m 로 밀려난다 (seed 에 묶인 50/50). 같은 (task, layout, lang,
   fixture_group) 의 base 좌표를 **앵커 축(범위가 큰 축)** 으로 두 무리로 가르고
   (경계 = 그 축 범위의 중앙값 (min+max)/2), 두 무리 중심의 중점을 fixture 중심 근사로 삼아

       l   = (−sin yaw, cos yaw)          # 로봇 좌측 단위벡터
       lat = l · (base − 중점)             # >0 → left, ≤0 → right

   drawer/ppcc/coffee 는 스폰이 단봉(±15cm)이라 side 축이 없다 → 무리 하나, 중점 = 전체 평균,
   side=None. (서랍의 left/right 는 스폰이 아니라 **대상 서랍** = 문장 변형이다.)

2) **feasibility** — 정책 무관 관절 스윕. OpenDrawer 는 `drawer_scene_feasibility.py`,
   SlideOvenRack 은 `ovenrack_feasibility.py`, SlideDishwasherRack 은
   `dishwasher_rack_feasibility.py` (모두 메인 트리, robocasa 컨테이너 실행, GPU 불필요).
   PPCC·coffee 는 대응 스윕 프로브가 없어 미검사(`feasible: null`)로 둔다.
   결과는 `feasibility_cache.json` 에 seed 단위로 캐시 → 재실행 멱등·증분.

주방 우선순위 (§2 공통 축)
--------------------------
  오븐        : L4 → L9 → L7 → (없으면) L2 → 나머지는 seed 수 순
  그 외 단일키: L4 → L9 → L5 → 나머지는 seed 수 순
  PPCC        : 5물체 공통 layout 을 최우선(공통도 desc), 동률이면 위 우선순위 → seed 수 순.
                물체가 그 공통 layout 을 못 채우면 그 물체 고유 layout 으로 보충.

scene 당 seed 1개 = 그 (키, layout) 무리에서 **앵커 중심에 가장 가까운** seed
(= |spawn_lat − 무리 평균 lat| 최소; side 키는 해당 side 무리 안에서). feasibility 를
통과하지 못하면 다음으로 가까운 seed 로 내려간다.

사용 (호스트, 표준 라이브러리만 필요):
  python scripts/collect/select_v6_scenes.py            # 프로브 포함 (docker exec)
  python scripts/collect/select_v6_scenes.py --no-probe # 캐시만 쓰고 새 프로브 금지
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

# ── 상수 ──────────────────────────────────────────────────────────────────────

# 메인 트리(컨테이너 /temporal_vla 마운트 원본). 워크트리에서 실행해도 스캔·프로브는
# 메인 트리 것을 읽는다(읽기 전용).
MAIN_TREE = Path("/home/dongkyu/pkt_ws/temporal_vla")
CONTAINER_REPO = "/temporal_vla"

LAYOUT_STYLE_IDS = [[i, i] for i in range(1, 11)]  # target split 10주방
LAYOUT_STYLE_ENV = ",".join(f"{a}:{b}" for a, b in LAYOUT_STYLE_IDS)

# 오븐 "out" 문장 (top 은 제외 — §2: L4 는 "oven rack out", L2/7/9 는 bottom 문장 인정)
OVEN_LANGS = (
    "Fully slide the oven rack out.",
    "Fully slide the bottom oven rack out.",
)
WASHER_LANG = "Fully slide the top dishwasher rack out."
PPCC_OBJECTS = ("apple", "jug", "candle", "bread", "marshmallow")

# 키별 layout 우선순위 (공통 축)
PRIORITY_OVEN = (4, 9, 7, 2)
PRIORITY_DEFAULT = (4, 9, 5)

MACHINE_ASSIGNMENT = {
    "kanu": ["OvenRack/out-left", "OvenRack/out-right",
             "DishwasherRack/out-left", "DishwasherRack/out-right"],
    "srv48": ["OpenDrawer/left", "OpenDrawer/right", "CoffeeSetupMug"],
    "srv50": ["PPCC/apple", "PPCC/jug", "PPCC/candle", "PPCC/bread", "PPCC/marshmallow"],
}

# task → feasibility 프로브 스크립트 (메인 트리 상대경로). None = 프로브 없음.
FEAS_SCRIPT = {
    "OpenDrawer": "scripts/collect/drawer_scene_feasibility.py",
    "SlideOvenRack": "scripts/safe/groot_n15/robocasa/analyze/ovenrack_feasibility.py",
    "SlideDishwasherRack":
        "scripts/safe/groot_n15/robocasa/analyze/dishwasher_rack_feasibility.py",
    "PickPlaceCounterToCabinet": None,
    "CoffeeSetupMug": None,
}

# (키, layout) 당 feasibility 를 미리 물어볼 후보 수
PROBE_DEPTH = 6


# ── 스캔 TSV 로드 ─────────────────────────────────────────────────────────────

def load_scan(scan_dir: Path) -> dict[str, list[dict]]:
    """task → 행 리스트. 같은 task 의 여러 seed 범위 파일을 합친다(seed 중복 제거)."""
    tasks: dict[str, list[dict]] = {}
    for tsv in sorted(scan_dir.glob("*_target10*.tsv")):
        task = tsv.name.split("_target10")[0]
        with open(tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("err"):  # 스캔 자체가 실패한 seed 는 버린다
                    continue
                try:
                    rec = {
                        "seed": int(row["seed"]),
                        "layout": int(row["layout_id"]),
                        "style": int(row["style_id"]),
                        "fixture_key": row["fixture_key"],
                        "fixture_group": row["fixture_group"],
                        "should_pull": row["should_pull"],
                        "rack_level": row["rack_level"],
                        "lang": row["lang"],
                        "base_x": float(row["base_x"]),
                        "base_y": float(row["base_y"]),
                        "base_yaw": float(row["base_yaw"]),
                    }
                except (KeyError, ValueError):
                    continue
                tasks.setdefault(task, []).append(rec)
    for task, rows in tasks.items():
        seen: dict[int, dict] = {}
        for r in rows:
            seen.setdefault(r["seed"], r)
        tasks[task] = sorted(seen.values(), key=lambda r: r["seed"])
    return tasks


# ── side / spawn_lat 판정 ────────────────────────────────────────────────────

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def annotate_lat(rows: list[dict], bimodal: bool) -> None:
    """같은 (task, layout, lang, fixture_group) 무리에 spawn_lat / side 를 채운다.

    bimodal=True (오븐·식기세척기): 앵커 축 범위 중앙값으로 두 무리를 갈라 중심의 중점을
    fixture 중심 근사로 쓴다. False: 무리 하나, 중점 = 전체 평균, side=None.
    """
    if not rows:
        return
    xs = [r["base_x"] for r in rows]
    ys = [r["base_y"] for r in rows]
    yaw = _median([r["base_yaw"] for r in rows])
    lx, ly = -math.sin(yaw), math.cos(yaw)  # 로봇 좌측 단위벡터

    if bimodal and len(rows) >= 4:
        # 앵커 축 = 범위가 큰 축
        rng_x, rng_y = max(xs) - min(xs), max(ys) - min(ys)
        axis = "base_x" if rng_x >= rng_y else "base_y"
        vals = [r[axis] for r in rows]
        cut = 0.5 * (min(vals) + max(vals))  # 범위 중앙값
        lo = [r for r in rows if r[axis] <= cut]
        hi = [r for r in rows if r[axis] > cut]
        if lo and hi:
            cx = 0.5 * (sum(r["base_x"] for r in lo) / len(lo)
                        + sum(r["base_x"] for r in hi) / len(hi))
            cy = 0.5 * (sum(r["base_y"] for r in lo) / len(lo)
                        + sum(r["base_y"] for r in hi) / len(hi))
        else:  # 실제로는 단봉 → side 없음으로 강등
            bimodal = False
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    else:
        bimodal = False
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    for r in rows:
        lat = lx * (r["base_x"] - cx) + ly * (r["base_y"] - cy)
        r["spawn_lat"] = round(lat, 4)
        r["side"] = ("left" if lat > 0 else "right") if bimodal else None


def annotate_all(tasks: dict[str, list[dict]]) -> None:
    """task 전체를 (layout, lang, fixture_group) 무리로 나눠 lat/side 를 채운다."""
    bimodal_tasks = {"SlideOvenRack", "SlideDishwasherRack"}
    for task, rows in tasks.items():
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            groups.setdefault((r["layout"], r["lang"], r["fixture_group"]), []).append(r)
        for g in groups.values():
            annotate_lat(g, bimodal=task in bimodal_tasks)


# ── 키 정의 ──────────────────────────────────────────────────────────────────

def key_defs() -> list[dict]:
    """12개 instruction 키의 정의(필터 조건 포함). 순서 = 보고 표 순서."""
    defs: list[dict] = []
    for side in ("left", "right"):
        defs.append({
            "key": f"OvenRack/out-{side}", "task": "SlideOvenRack",
            "kind": "pull_side", "side": side, "langs": OVEN_LANGS,
            "priority": PRIORITY_OVEN,
        })
    for side in ("left", "right"):
        defs.append({
            "key": f"DishwasherRack/out-{side}", "task": "SlideDishwasherRack",
            "kind": "pull_side", "side": side, "langs": (WASHER_LANG,),
            "priority": PRIORITY_DEFAULT,
        })
    for side in ("left", "right"):
        defs.append({
            "key": f"OpenDrawer/{side}", "task": "OpenDrawer",
            "kind": "pull_drawer", "side": None,
            "langs": (f"Open the {side} drawer.",),
            "priority": PRIORITY_DEFAULT,
        })
    for obj in PPCC_OBJECTS:
        defs.append({
            "key": f"PPCC/{obj}", "task": "PickPlaceCounterToCabinet",
            "kind": "pickplace", "side": None,
            "langs": (f"Pick the {obj} from the counter and place it in the cabinet.",),
            "priority": PRIORITY_DEFAULT, "ppcc_object": obj,
        })
    defs.append({
        "key": "CoffeeSetupMug", "task": "CoffeeSetupMug", "kind": "coffee",
        "side": None,
        "langs": ("Pick the mug from the counter and place it under "
                  "the coffee machine dispenser.",),
        "priority": PRIORITY_DEFAULT,
    })
    return defs


def candidates_by_layout(tasks: dict[str, list[dict]], kd: dict) -> dict[int, list[dict]]:
    """키 조건을 만족하는 행을 layout 별로, '앵커 중심에 가까운' 순으로 정렬해 반환."""
    rows = [r for r in tasks.get(kd["task"], [])
            if r["lang"] in kd["langs"]
            and (kd["side"] is None or r.get("side") == kd["side"])]
    by_layout: dict[int, list[dict]] = {}
    for r in rows:
        by_layout.setdefault(r["layout"], []).append(r)
    for lay, group in by_layout.items():
        # 무리 평균 lat 에 가장 가까운 seed 가 가장 '전형적'인 스폰이다.
        mean_lat = sum(r["spawn_lat"] for r in group) / len(group)
        group.sort(key=lambda r: (abs(r["spawn_lat"] - mean_lat), r["seed"]))
    return by_layout


def rank_layouts(available: list[int], priority: tuple[int, ...],
                 counts: dict[int, int]) -> list[int]:
    """우선순위 layout 먼저, 나머지는 seed 수(→layout 번호) 순."""
    pri = [l for l in priority if l in available]
    rest = sorted((l for l in available if l not in pri),
                  key=lambda l: (-counts.get(l, 0), l))
    return pri + rest


def ppcc_shared_layouts(cands: dict[str, dict[int, list[dict]]]) -> list[int]:
    """5물체 공통도가 높은 layout 을 우선순위 규칙으로 정렬해 반환."""
    all_layouts = sorted({l for c in cands.values() for l in c})

    def rank(l: int) -> tuple:
        share = sum(1 for c in cands.values() if l in c)          # 공통도 (0~5)
        pri = PRIORITY_DEFAULT.index(l) if l in PRIORITY_DEFAULT else len(PRIORITY_DEFAULT)
        total = sum(len(c.get(l, [])) for c in cands.values())    # 총 seed 수
        return (-share, pri, -total, l)

    return sorted(all_layouts, key=rank)


# ── feasibility 프로브 (robocasa 컨테이너) ────────────────────────────────────

def run_probe(task: str, seeds: list[int], container: str, jobs: int) -> list[dict]:
    """docker exec 로 관절 스윕 프로브를 돌려 seed 별 결과 행을 돌려준다."""
    script = FEAS_SCRIPT.get(task)
    if not script or not seeds:
        return []
    out_json = f"/tmp/v6_feas_{task}.json"
    cmd = [
        "docker", "exec",
        "-e", "MUJOCO_GL=egl",
        "-e", f"ROBOCASA_LAYOUT_STYLE_IDS={LAYOUT_STYLE_ENV}",
        "-e", "OMP_NUM_THREADS=1",
        "-e", (f"PYTHONPATH={CONTAINER_REPO}/src/policies/Isaac-GR00T:"
               f"{CONTAINER_REPO}/src/benchmarks/robocasa:"
               f"{CONTAINER_REPO}/src/benchmarks/robosuite:{CONTAINER_REPO}"),
        container, "python", f"{CONTAINER_REPO}/{script}",
        "--task", task,
        "--seeds", ",".join(str(s) for s in seeds),
        "--jobs", str(jobs),
        "--out", out_json,
    ]
    print(f"[probe] {task}: {len(seeds)} seeds (jobs={jobs}) …", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[probe] {task} FAILED rc={proc.returncode}\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}", flush=True)
        return []
    cat = subprocess.run(["docker", "exec", container, "cat", out_json],
                         capture_output=True, text=True)
    if cat.returncode != 0:
        print(f"[probe] {task}: 결과 JSON 읽기 실패\n{cat.stderr[-500:]}", flush=True)
        return []
    try:
        return json.loads(cat.stdout)
    except json.JSONDecodeError as e:
        print(f"[probe] {task}: JSON 파싱 실패 {e}", flush=True)
        return []


def ensure_feasibility(tasks_needed: dict[str, set[int]], cache: dict,
                       container: str, jobs: int, do_probe: bool) -> None:
    """캐시에 없는 seed 만 프로브해서 cache[task][str(seed)] 를 채운다(멱등)."""
    for task, seeds in tasks_needed.items():
        if FEAS_SCRIPT.get(task) is None:
            continue
        have = cache.setdefault(task, {})
        missing = sorted(s for s in seeds if str(s) not in have)
        if not missing:
            continue
        if not do_probe:
            print(f"[probe] {task}: 미검사 {len(missing)} seed (--no-probe 라 생략)",
                  flush=True)
            continue
        for row in run_probe(task, missing, container, jobs):
            have[str(row["seed"])] = row


def feas_of(cache: dict, task: str, seed: int):
    """(feasible, 프로브 행) — 프로브 없는 task 나 미검사 seed 는 (None, None)."""
    row = cache.get(task, {}).get(str(seed))
    if row is None:
        return None, None
    return bool(row.get("feasible")), row


# ── 선택 ─────────────────────────────────────────────────────────────────────

def pick_scene(kd: dict, layout: int, cand: list[dict], cache: dict) -> dict | None:
    """한 layout 안에서 feasibility 를 통과하는 첫 후보를 scene 레코드로."""
    for r in cand:
        feasible, prow = feas_of(cache, kd["task"], r["seed"])
        if feasible is False:
            continue  # 관절 스윕이 막힌 seed → 다음 후보
        scene = {
            "env_seed": r["seed"],
            "layout": r["layout"],
            "style": r["style"],
            "side": kd["side"],
            "lang": r["lang"],
            "fixture_group": r["fixture_group"],
            "spawn_lat": r["spawn_lat"],
            "base_x": r["base_x"],
            "base_y": r["base_y"],
            "base_yaw": r["base_yaw"],
            "feasible": feasible,  # None = 프로브 없는 task(PPCC/coffee) 또는 미검사
        }
        if prow is not None:
            scene["feasibility"] = {
                "q_max_feasible": prow.get("q_max_feasible"),
                "q_start": prow.get("q_start"),
                "blocker_geom": prow.get("blocker_geom"),
                # 스캔 lang 과 프로브가 본 ep_lang 이 같아야 한다(주방 목록 일치 확인).
                "probe_lang": prow.get("ep_lang"),
                "lang_match": prow.get("ep_lang") == r["lang"],
            }
        return scene
    return None


def select(tasks: dict[str, list[dict]], cache: dict, container: str,
           jobs: int, do_probe: bool, n_scenes: int) -> dict:
    defs = key_defs()
    cands = {kd["key"]: candidates_by_layout(tasks, kd) for kd in defs}

    # PPCC 는 5물체 공통 layout 을 먼저 쓴다.
    ppcc_cands = {kd["key"]: cands[kd["key"]] for kd in defs if kd["kind"] == "pickplace"}
    shared = ppcc_shared_layouts(ppcc_cands) if ppcc_cands else []

    # 키별 layout 순서 확정
    order: dict[str, list[int]] = {}
    for kd in defs:
        c = cands[kd["key"]]
        counts = {l: len(v) for l, v in c.items()}
        if kd["kind"] == "pickplace":
            head = [l for l in shared if l in c]
            tail = rank_layouts([l for l in c if l not in head], kd["priority"], counts)
            order[kd["key"]] = head + tail
        else:
            order[kd["key"]] = rank_layouts(sorted(c), kd["priority"], counts)

    # feasibility: 각 (키, layout) 상위 PROBE_DEPTH 후보를 미리 조회
    needed: dict[str, set[int]] = {}
    for kd in defs:
        if FEAS_SCRIPT.get(kd["task"]) is None:
            continue
        for lay in order[kd["key"]][:n_scenes + 2]:
            for r in cands[kd["key"]][lay][:PROBE_DEPTH]:
                needed.setdefault(kd["task"], set()).add(r["seed"])
    ensure_feasibility(needed, cache, container, jobs, do_probe)

    keys_out: dict[str, dict] = {}
    for kd in defs:
        scenes: list[dict] = []
        for lay in order[kd["key"]]:
            if len(scenes) >= n_scenes:
                break
            s = pick_scene(kd, lay, cands[kd["key"]][lay], cache)
            if s is not None:
                scenes.append(s)
        entry = {
            "task_env": f"robocasa_panda_omron/{kd['task']}_PandaOmron_Env",
            "task": kd["task"],
            "kind": kd["kind"],
            "scenes": scenes,
        }
        if kd["side"] is not None:
            entry["spawn_side"] = kd["side"]
        if "ppcc_object" in kd:
            entry["object"] = kd["ppcc_object"]
        if len(scenes) < n_scenes:
            entry["shortfall"] = n_scenes - len(scenes)  # 스캔 부족 = "부족" 표시
        keys_out[kd["key"]] = entry

    return {
        "schema": "v6_scene_selection/1",
        "n_scenes_per_key": n_scenes,
        "env_kwargs": {"layout_and_style_ids": LAYOUT_STYLE_IDS},
        "machine_assignment": MACHINE_ASSIGNMENT,
        "ppcc_shared_layouts": shared[:n_scenes],
        "keys": keys_out,
    }


# ── 요약 표 ──────────────────────────────────────────────────────────────────

def summary_table(sel: dict) -> str:
    n = sel["n_scenes_per_key"]
    lines = [f"{'key':<24} {'s0':<22} {'s1':<22} {'s2':<22} 비고",
             "-" * 100]
    for key, e in sel["keys"].items():
        cells = []
        for s in e["scenes"]:
            side = s["side"] or "-"
            feas = {True: "OK", False: "X", None: "?"}[s["feasible"]]
            cells.append(f"L{s['layout']}/{side}/{s['env_seed']}/{feas}")
        while len(cells) < n:
            cells.append("-")
        note = "부족" if e.get("shortfall") else ""
        lines.append(f"{key:<24} " + " ".join(f"{c:<22}" for c in cells) + f" {note}")
    lines.append("")
    lines.append("셀 = layout/side/env_seed/feasible (OK=스윕통과, X=차단, ?=프로브 없음)")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir",
                    default=str(MAIN_TREE / "outputs/analysis/seed_scan/fixture_groups"))
    ap.add_argument("--out",
                    default=str(repo / "configs/collect/n15_grid_v6_scene_jitter"
                                       "/scene_selection.json"))
    ap.add_argument("--cache", default=None,
                    help="feasibility 캐시 JSON (기본: --out 과 같은 디렉토리)")
    ap.add_argument("--container", default="robocasa")
    ap.add_argument("--jobs", type=int, default=8, help="프로브 동시 프로세스 (≤8)")
    ap.add_argument("--scenes", type=int, default=3, help="키당 scene 수")
    ap.add_argument("--no-probe", action="store_true",
                    help="새 프로브를 돌리지 않고 캐시만 사용")
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    if not scan_dir.is_dir():
        sys.exit(f"스캔 디렉토리 없음: {scan_dir}")
    tasks = load_scan(scan_dir)
    if not tasks:
        sys.exit(f"스캔 TSV 없음: {scan_dir}")
    print("[scan] " + ", ".join(f"{t}={len(r)}" for t, r in sorted(tasks.items())),
          flush=True)
    annotate_all(tasks)

    out_path = Path(args.out)
    cache_path = Path(args.cache) if args.cache else out_path.parent / "feasibility_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    sel = select(tasks, cache, args.container, min(args.jobs, 8),
                 not args.no_probe, args.scenes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
    out_path.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[wrote] {out_path}\n[wrote] {cache_path}\n", flush=True)
    print(summary_table(sel), flush=True)


if __name__ == "__main__":
    main()
