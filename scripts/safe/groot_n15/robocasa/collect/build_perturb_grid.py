"""exp4-2/exp5-2 P0 캘리브레이션 grid 생성 — baseline 사이드카 앵커 → spec json + grid.tsv.

Phase A baseline(`--no-features` 사이드카)에서 성공 episode 와 **앵커 event**(record)를 읽어,
모드(C1/G1/P1/P2) × config × episode 행을 만든다. 모든 확률량은 spec_seed 로 결정
(spec_seed = sha256(mode|config|ep) 파생 — tsv 재생성이 곧 재현).

  # ppcc_bread (exp4-2 기존 — 인자 무변경 시 동작 동일)
  python build_perturb_grid.py --baseline-dir <.../p0/baseline/raw_rollouts/<TASK>/<CELL>> \
      --out-dir <.../p0> [--eps-per-config 4] [--n-cal 12]
  # exp5-2 신규 cell
  python build_perturb_grid.py --cell drawer_left --baseline-dir <...> --out-dir <...>
  python build_perturb_grid.py --cell mixer --baseline-dir <...> --out-dir <...>

출력: <out>/specs/<config>/ep{idx}.json, <out>/grid.tsv
grid.tsv 열: mode  config  ep_idx  inference_seed  spec(컨테이너 경로 아님 — 러너가 변환)  tag

앵커 event 어휘는 phase 라벨러(`scripts/safe/groot_n16/robocasa/collect/robocasa_event_labeler.py`)
가 사이드카 `event_steps` 로 내보내는 키에서 고른다 (cell 표 근거는 CELL_DEFAULTS 주석).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# config 격자 (24b 개정 메뉴): C1 scale × WAM σ / G1 σxyz / P1 δ / P2 F×dur
C1_SCALES = [0.5, 1.0, 2.0]
G1_SIGMAS = [0.05, 0.10, 0.15]
P1_MAGS = [0.03, 0.08, 0.15]
P2_GRID = [(5.0, 2), (15.0, 2), (40.0, 2), (5.0, 5), (15.0, 5), (40.0, 5)]

# ── cell 별 기본값 (exp5-2, docs/steering/29 §0 표) ────────────────────────────────
# anchor_event: 라벨러 event_steps 키 (P1/P2 trigger 앵커).
#   ppcc_bread  = "grasp:obj"   (robocasa_event_labeler.py:156 `_pnp_events`)
#   drawer_left = "near:handle" (동 파일 :582 DrawerPhaseLabeler, grasp-handle 첫 프레임)
#   mixer       = "near:head"   (동 파일 :879 EV_NEAR + StandMixerPhaseLabeler.EV_NEAR)
# target: perturbation.py 의 P1/P2 대상 (평문=자유물체 / "fixture:<attr>:<joint_key>").
#   drawer/mixer 는 조작 대상이 자유물체가 아니라 fixture 관절 → fixture 문법.
# p1_direction: 1-DoF fixture 관절 δ 의 부호(x 성분). Drawer 는 qpos<0 이 "열림"
#   (robocasa cabinets.py get_door_state sign −1) → [-1,0] = 살짝 열어 손잡이 pose 를 틀기.
#   Mixer head 는 관절 range 부호를 정적으로 확정 못 함 → GPU smoke 에서 확인 후 확정(TODO).
# p1_mags 단위: 자유물체=m(xy), fixture=raw 관절 단위(slide m / hinge rad).
#   ★ 아래 drawer/mixer 값은 **초기 추정치** — grid phase 실패율 40–70% 게이트로 재캘리브레이션.
CELL_DEFAULTS: dict[str, dict] = {
    "ppcc_bread": {
        "modes": "C1,G1,P1,P2",
        "anchor_event": "grasp:obj",
        "target": "obj",
        "p1_direction": None,          # seeded 랜덤 방향 (xy 평면) — exp4-2 기존 동작
        "p1_mags": P1_MAGS,
        "p2_grid": P2_GRID,
        # sham 대표 config — exp4-2 산출과 동일하게 고정 (p1_d008 / p2_f015d2)
        "p1_sham_rep": 0.08,
        "p2_sham_rep": (15.0, 2),
    },
    "drawer_left": {
        # 사용자 결정(07-27): drawer 는 **C1/P1 만** — P2 는 수평 wrench 대상이 마땅치 않아 제외
        # (docs/steering/29 §4 의 P5 damping 대체안은 미구현).
        "modes": "C1,P1",
        "anchor_event": "near:handle",
        "target": "fixture:drawer:slidejoint",
        "p1_direction": (-1.0, 0.0),
        "p1_mags": [0.02, 0.05, 0.10],  # m (서랍 전체 스트로크 ≈ size[1]*0.55)
        "p1_sham_rep": 0.05,
        "p2_grid": [],
        "p2_sham_rep": None,
    },
    "mixer": {
        "modes": "C1,P1,P2",
        "anchor_event": "near:head",
        "target": "fixture:stand_mixer:head",
        "p1_direction": (1.0, 0.0),     # TODO(GPU): head hinge 부호 실측 후 확정
        "p1_mags": [0.05, 0.10, 0.20],  # rad
        "p2_grid": [(5.0, 2), (15.0, 2), (40.0, 2), (15.0, 5), (40.0, 5)],
        "p1_sham_rep": 0.10,
        "p2_sham_rep": (15.0, 2),
    },
}


def spec_seed(mode: str, config: str, ep: int) -> int:
    h = hashlib.sha256(f"{mode}|{config}|{ep}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def _parse_p2_grid(text: str) -> list[tuple[float, int]]:
    """"5x2,15x2,40x5" → [(5.0,2),(15.0,2),(40.0,5)]"""
    out: list[tuple[float, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        mag, dur = part.lower().split("x")
        out.append((float(mag), int(dur)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", required=True,
                    help="Phase A 사이드카 디렉토리 (task*--ep*--succ*.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-cal", type=int, default=12, help="캘리브레이션 성공 ep 수")
    ap.add_argument("--eps-per-config", type=int, default=4,
                    help="config 당 1라운드 ep 수 (톱업은 값 키워 재생성 — 결정적 재현)")
    ap.add_argument("--cell", default="ppcc_bread", choices=sorted(CELL_DEFAULTS),
                    help="cell 기본값 묶음 (modes/anchor-event/target/P1·P2 격자)")
    ap.add_argument("--modes", default=None,
                    help="미지정 시 cell 기본값 (ppcc_bread=C1,G1,P1,P2 / drawer_left=C1,P1)")
    ap.add_argument("--anchor-event", default=None,
                    help="P1/P2 trigger 앵커 event 키 (미지정 시 cell 기본값)")
    ap.add_argument("--target", default=None,
                    help="P1/P2 대상 (미지정 시 cell 기본값). 평문=자유물체, "
                         "'fixture:<attr>:<joint_key>'=fixture 관절")
    ap.add_argument("--p1-mags", default=None, help="쉼표 구분 P1 δ 목록 (미지정 시 cell 기본값)")
    ap.add_argument("--c1-scales", default=None,
                    help="쉼표 구분 C1 scale 목록 (미지정 시 모듈 기본 C1_SCALES)")
    ap.add_argument("--p2-grid", default=None,
                    help="'FxDUR' 쉼표 목록 예 '5x2,15x2,40x5' (미지정 시 cell 기본값)")
    ap.add_argument("--sham-eps", type=int, default=1,
                    help="모드당 sham 행 ep 수 (P0 중 배선 드리프트 감시)")
    ap.add_argument("--seed-stride", type=int, default=1000,
                    help="신선 ep 확장 시 inference_seed = ep_idx * stride "
                         "(collect_perturb_grid.sh STRIDE=1000 / baseline worker 와 동일 공식)")
    ap.add_argument("--fresh-ep-start", type=int, default=None,
                    help="C1/G1 신선 ep 확장 시작 인덱스 (미지정 시 baseline 사이드카 최대 ep+1)")
    ap.add_argument("--fresh-ep-block", type=int, default=100,
                    help="config 당 신선 ep 예약 블록 크기 — 톱업(--eps-per-config 증가) 시 "
                         "기존 행이 그대로 유지되도록 config 별 고정 구간을 준다")
    args = ap.parse_args()

    cell = CELL_DEFAULTS[args.cell]
    anchor_event = args.anchor_event or cell["anchor_event"]
    target = args.target or cell["target"]
    p1_mags = _parse_floats(args.p1_mags) if args.p1_mags else list(cell["p1_mags"])
    global C1_SCALES
    if args.c1_scales:
        C1_SCALES = _parse_floats(args.c1_scales)
    p2_grid = _parse_p2_grid(args.p2_grid) if args.p2_grid else list(cell["p2_grid"])
    p1_direction = cell["p1_direction"]

    base = Path(args.baseline_dir)
    eps = []  # (ep_idx, inf_seed, anchor_record)
    for p in sorted(base.glob("task*--ep*--succ1.json")):
        d = json.loads(p.read_text())
        m = re.search(r"--ep(\d+)--", p.name)
        if m is None:
            continue
        ep_idx = int(m.group(1))
        anchor = (d.get("event_steps") or {}).get(anchor_event)
        eps.append((ep_idx, int(d.get("inference_seed") or 0), anchor))
    # baseline 이 실제로 소비한 ep 인덱스 범위 (성공/실패 사이드카 전부) — 신선 ep 확장의 시작점.
    base_eps = [int(m.group(1)) for p in base.glob("task*--ep*--succ*.json")
                if (m := re.search(r"--ep(\d+)--", p.name))]
    fresh_start = (args.fresh_ep_start if args.fresh_ep_start is not None
                   else (max(base_eps) + 1 if base_eps else 0))
    if len(eps) < args.n_cal:
        print(f"ABORT: 성공 baseline {len(eps)}개 < --n-cal {args.n_cal} — Phase A 부족",
              file=sys.stderr)
        return 2
    eps = eps[: args.n_cal]
    no_anchor = [e for e in eps if e[2] is None]
    if no_anchor:
        print(f"WARN: 앵커({anchor_event}) 없는 성공 ep {[e[0] for e in no_anchor]} "
              "— P1/P2 에서 제외")

    modes = [m.strip() for m in (args.modes or cell["modes"]).split(",") if m.strip()]
    if len(no_anchor) == len(eps) and ({"P1", "P2"} & set(modes)):
        print(f"ABORT: 성공 ep 전부에 앵커 event {anchor_event!r} 없음 — 라벨러/cell 불일치 "
              "(사이드카 event_steps 키 확인)", file=sys.stderr)
        return 3
    out = Path(args.out_dir)
    specs_dir = out / "specs"
    rows = []

    def add(mode: str, config: str, ep_idx: int, inf: int, spec: dict, sham: bool = False):
        tag = f"{config}{'_sham' if sham else ''}_ep{ep_idx}"
        spec = {**spec, "spec_seed": spec_seed(mode, config + ("_sham" if sham else ""), ep_idx),
                "sham": sham, "tag": tag}
        sp = specs_dir / config / f"ep{ep_idx}{'_sham' if sham else ''}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(spec, indent=1))
        rows.append((mode, config, ep_idx, inf, str(sp), tag))

    # 기본값(자유물체 "obj" / seeded 랜덤 방향)일 땐 키를 아예 안 넣어 exp4-2 spec json 을
    # 바이트 단위로 보존한다 (ppcc 회귀 방지).
    target_kw = {} if target == "obj" else {"target": target}
    direction_kw = {} if p1_direction is None else {"direction": list(p1_direction)}

    fresh_next = [fresh_start]

    def pick(config_i: int, config: str, anchored: bool):
        """config 별 ep 배분 — 라운드로빈 오프셋으로 config 간 ep 다양화.

        --eps-per-config 가 캘리브 성공 풀(len(eps))보다 크면 예전 코드는 같은
        (ep_idx, inference_seed) 를 반복 생성했고, 러너의 done_mark 가 config 디렉토리 안에서
        ep_idx 로만 판정하므로 2회차 이후가 통째로 skip 됐다 (mixer: 48 요청 → 10판 수집).
        모드별 처리:
          - anchored=False (C1/G1): trigger 앵커가 필요 없으므로 **신선 ep_idx 로 확장**.
            scenario_seed 는 cell 고정 seed(mixer 100000 / drawer 100001) 이고 collector 가
            n-episodes=1 로 돌아 `scenario_seed = --seed + 0` → ep_idx 를 늘려도 fixture scene
            은 동일, 바뀌는 것은 정책 rng(inference_seed) 와 출력 스템뿐이다.
            inference_seed 는 baseline 워커와 같은 공식 ep_idx * STRIDE(1000)
            (collect_perturb_grid.sh:52, 기존 grid 행 ep10→10000 과 일치).
          - anchored=True (P1/P2): 성공 ep 사이드카의 event record 에 종속이라 확장 불가 →
            풀 크기로 clamp + WARN.
        """
        n = len(eps)
        want = args.eps_per_config
        if want <= n:  # 기존 경로 (ppcc: 풀 55 ≥ 48) — 동작·산출 바이트 불변
            return [eps[(config_i + j) % n] for j in range(want)]
        picked = [eps[(config_i + j) % n] for j in range(n)]
        if anchored:
            print(f"WARN: {config} rows capped at {n} (anchored pool) — 요청 {want}")
            return picked
        # config 별 고정 블록 (fresh_start + ci*block + j) — 톱업으로 --eps-per-config 를 키워도
        # 이미 수집된 앞쪽 행의 (ep_idx, inference_seed) 가 그대로 유지된다.
        if want - n > args.fresh_ep_block:
            print(f"ABORT: {config} 신선 ep {want - n}개 > --fresh-ep-block "
                  f"{args.fresh_ep_block} (블록 충돌)", file=sys.stderr)
            raise SystemExit(4)
        base_ep = fresh_start + config_i * args.fresh_ep_block
        for j in range(want - n):
            ep_new = base_ep + j
            picked.append((ep_new, ep_new * args.seed_stride, None))
        fresh_next[0] = max(fresh_next[0], base_ep + (want - n))
        return picked

    ci = 0
    for mode in modes:
        if mode == "C1":
            for s in C1_SCALES:
                config = f"c1_s{int(s * 100):03d}"
                for ep_idx, inf, _ in pick(ci, config, anchored=False):
                    add("C1_camera", config, ep_idx, inf, {"mode": "C1_camera", "scale": s})
                ci += 1
        elif mode == "G1":
            for s in G1_SIGMAS:
                config = f"g1_x{int(s * 100):03d}"
                for ep_idx, inf, _ in pick(ci, config, anchored=False):
                    add("G1_gripper_init", config, ep_idx, inf,
                        {"mode": "G1_gripper_init", "sigma_xyz_m": s})
                ci += 1
        elif mode == "P1":
            for mag in p1_mags:
                config = f"p1_d{int(round(mag * 100)):03d}"
                for ep_idx, inf, anchor in pick(ci, config, anchored=True):
                    if anchor is None:
                        continue
                    add("P1_displace", config, ep_idx, inf,
                        {"mode": "P1_displace", "magnitude": mag,
                         **target_kw, **direction_kw,
                         "trigger_record": max(0, int(anchor) - 2)})
                ci += 1
        elif mode == "P2":
            for mag, dur in p2_grid:
                config = f"p2_f{int(mag):03d}d{dur}"
                for ep_idx, inf, anchor in pick(ci, config, anchored=True):
                    if anchor is None:
                        continue
                    add("P2_force", config, ep_idx, inf,
                        {"mode": "P2_force", "magnitude": mag, "duration_records": dur,
                         **target_kw,
                         "trigger_record": int(anchor) + 1})
                ci += 1
        else:
            print(f"ABORT: unknown mode {mode}", file=sys.stderr)
            return 2

    # sham 감시 행 (모드 대표 config 1개 × sham-eps). P1/P2 대표는 격자 중앙값 —
    # ppcc 기본 격자에서는 exp4-2 와 동일한 p1_d008 / p2_f015d2 가 선택된다.
    p1_rep = cell.get("p1_sham_rep") if args.p1_mags is None else None
    if p1_rep is None:
        p1_rep = p1_mags[len(p1_mags) // 2] if p1_mags else None
    p2_rep = cell.get("p2_sham_rep") if args.p2_grid is None else None
    if p2_rep is None:
        p2_rep = p2_grid[len(p2_grid) // 2] if p2_grid else None
    sham_reps = {"C1": ("C1_camera", "c1_s100", {"mode": "C1_camera", "scale": 1.0}),
                 "G1": ("G1_gripper_init", "g1_x010", {"mode": "G1_gripper_init", "sigma_xyz_m": 0.10})}
    if p1_rep is not None:
        sham_reps["P1"] = ("P1_displace", f"p1_d{int(round(p1_rep * 100)):03d}",
                           {"mode": "P1_displace", "magnitude": p1_rep,
                            **target_kw, **direction_kw})
    if p2_rep is not None:
        sham_reps["P2"] = ("P2_force", f"p2_f{int(p2_rep[0]):03d}d{p2_rep[1]}",
                           {"mode": "P2_force", "magnitude": p2_rep[0],
                            "duration_records": p2_rep[1], **target_kw})
    for m in modes:
        if m not in sham_reps:
            print(f"WARN: {m} sham 대표 config 없음 (격자 비어 있음) — sham 행 생략")
            continue
        mode_name, config, spec = sham_reps[m]
        for ep_idx, inf, anchor in eps[: args.sham_eps]:
            if m in ("P1", "P2"):
                if anchor is None:
                    continue
                spec = {**spec, "trigger_record": max(0, int(anchor) - 2) if m == "P1"
                        else int(anchor) + 1}
            add(mode_name, config, ep_idx, inf, spec, sham=True)

    out.mkdir(parents=True, exist_ok=True)
    grid = out / "grid.tsv"
    with grid.open("w") as f:
        f.write("mode\tconfig\tep_idx\tinference_seed\tspec\ttag\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    n_cfg = len({r[1] for r in rows})
    print(f"wrote {grid}: {len(rows)} rows, {n_cfg} configs, cell={args.cell}, "
          f"modes={modes}, anchor={anchor_event}, target={target}, "
          f"cal_eps={[e[0] for e in eps]}, fresh_eps={fresh_start}..{fresh_next[0] - 1}"
          if fresh_next[0] > fresh_start else f"cal_eps={[e[0] for e in eps]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
