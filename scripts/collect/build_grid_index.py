#!/usr/bin/env python3
"""좌표 저장소 인덱서 — docs/04 §3 레이아웃 → §4 인덱스 표.

구 인덱서는 `activations/<sig>/` 평면(= sig 가 식별자)을 전제했다. 2026-08-05 개정으로
식별은 **좌표**가 하고 `sig` 는 무결성 검증 열이 되었으므로(§3.1) 이 스크립트가 그 자리를
대신한다. 좌표 레이아웃은 두 가지이며 둘 다 읽는다 (§3.1.1, 2026-09-03 개정):

  3축(정본)  `<plan_id>/<machine>/<instruction>/s<i>/k<r>/n<j>/<arm>/`
  legacy 2축 `<plan_id>/<machine>/<instruction>/s<i>/n<j>/<arm>/`      (v1·v2 등 지터 없는 plan)

`s<i>` = base scene index(평탄 인코딩 금지), `k<r>` = `jitter_reset_idx`(채택 k 값 그대로,
연속 아님), `n<j>` = noise index. 표의 좌표 열도 같은 3열이고 legacy 행은
`jitter_reset_idx` 가 빈 칸이다. 평탄 `cell_si` 열은 없다 — 필요하면 `s*100+k` 로 파생한다.

산출:
  rollouts.tsv       행 = base 셀 (좌표 + 신원 + 캡처 밀도 5열, §4)
  arm_bindings.tsv   행 = arm 디렉토리(config.json 보유), §3.3 의 개입 파라미터

읽는 것은 `meta.json`·`config.json` 뿐이다 — **pkl 을 열지 않는다.** torch 없는 base
python3(승준 노드)에서 아카이브 루트를 그대로 돌 수 있어야 하기 때문이고, 캡처 밀도는
이미 §6 규약대로 meta 에 기록돼 있다.

단일 파일 자족: repo 없이 scp 한 장으로 원격에서 돌아간다. `--plan-json` 은 없어도 되고
(plan_id 는 디렉토리에서 발견) 주면 계획 대비 결손·스키마(version 등)를 보강한다.

사용:
    python3 scripts/collect/build_grid_index.py \
        --grid-root /home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid \
        --plan-json configs/collect/n15_grid_v1/collection_plan.json \
        --plan-json configs/collect/n15_grid_v1b/collection_plan.json \
        --out /home/kimseungjun/datasets/temporal_vla_store/groot/n15/index

종료 코드: machine 결측·중복 좌표 등 무결성 위반이 있으면 1 (`--strict` 면 결손도 포함).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

BASE_ARM = "base"
PLAN_NAME = "collection_plan.json"

_S_RE = re.compile(r"^s(\d+)$")
_K_RE = re.compile(r"^k(\d+)$")      # 지터 층. 이 패턴일 때만 k 층으로 본다 (§3.1.1)
_N_RE = re.compile(r"^n(\d+)$")

# docs/04 §4 — kind=activation 행에 필수. 없으면 4MB/판 과 565MB/판 이 같은 질의에 섞인다.
CAPTURE_COLS = ("capture_token_mode", "feature_kind", "feature_axes",
                "record_shape", "capture_layers")

# 좌표 복합키 (§4). `armsig="base"` 가 baseline 행.
COORD_COLS = ("plan_id", "machine", "grid_instruction",
              "scene_idx",           # base scene (평탄 si 아님)
              "jitter_reset_idx",    # 지터 축 k. legacy 2축 행은 빈 칸
              "noise_idx", "armsig")

ROLLOUT_COLS = COORD_COLS + (
    "kind",                 # activation | eval — pkl 유무가 아니라 arm 여부로도 갈린다
    "sig",                  # 무결성 검증 열 (식별자 아님, §3.1)
    "pkl_sha256",
    "model", "version", "ckpt",
    "task", "task_id", "instruction", "env_name",
    "env_seed", "inference_seed", "episode_idx",
    "success",
    "steps", "n_inferences", "n_action_steps", "chunk_len",
    "has_pkl",
) + CAPTURE_COLS + (
    "plan_name",
    "machine_source", "ckpt_source", "meta_source",
    "rel_path",             # grid-root 상대 — 절대경로 기록 금지 (§8)
)

# docs/04 §3.3 config.json 의 개입 파라미터 전량 + 좌표.
ARM_COLS = COORD_COLS + (
    "op", "beta", "bindings", "token_select", "denoise",
    "steer_from_record", "gated_phases",
    "rel_path",
)


# ─────────────────────────── 계획 (최소 파싱) ───────────────────────────
# src.collect.plan.CollectionPlan 과 같은 뜻이지만 import 하지 않는다 — 이 스크립트는
# repo 없이 원격에 한 장만 올려도 돌아야 한다. plan_id 는 저장된 JSON 의 필드를 그대로
# 쓰므로(plan.save 가 항상 기록한다) 해싱 로직을 복제하지 않는다.

class PlanInfo:
    def __init__(self, body: dict[str, Any], src: Path):
        self.src = src
        self.plan_id: str = body.get("plan_id") or ""
        self.name: str = body.get("name", "")
        self.model: str = body.get("model", "")
        self.version: str = body.get("version", "")
        self.ckpt: str = body.get("ckpt", "")
        self.capture_layers: list[int] = list(body.get("capture_layers") or [])
        self.instructions: dict[str, list[int]] = dict(body.get("instructions") or {})
        self.noise_seeds: list[int] = list(body.get("noise_seeds") or [])
        # 3축 plan 은 jitter[instr][scene_idx] = 채택 k 목록 (§3.1.1). 없으면 legacy 2축.
        jit = body.get("jitter")
        self.jitter: dict[str, list[list[int]]] | None = (
            {k: [list(ks) for ks in v] for k, v in jit.items()} if jit else None)
        # legacy 전용 필터. 3축 plan 은 jitter 가 곧 수집 대상 전부라 쓰지 않는다.
        extra = body.get("extra") or {}
        self.adopted_cells: set[str] | None = (
            set(extra["adopted_cells"]) if extra.get("adopted_cells") else None)
        if not self.plan_id:
            raise ValueError(
                f"{src}: plan_id 없음 — CollectionPlan.save 가 쓴 파일이 아니다. "
                "plan_id 는 계획 내용의 지문이라 여기서 추측할 수 없다 (docs/04 §5.1)")

    def cell_keys(self) -> Iterator[tuple[str, int, int | None, int]]:
        """기대 셀 = (instruction, scene_idx, jitter_reset_idx|None, noise_idx).

        3축 plan 이면 `jitter[instr][si]` 의 k 를 그대로 돈다 (채택 k 만이 수집 대상).
        legacy 2축이면 scene×noise 전조합이되, `extra.adopted_cells` 가 있으면 그 필터를
        유지한다 — 계획에 없던 셀을 결손으로 세지 않기 위함이다.
        """
        for instr, scenes in self.instructions.items():
            for si in range(len(scenes)):
                if self.jitter is not None:
                    ks: list[int | None] = list(self.jitter.get(instr, [])[si]) \
                        if si < len(self.jitter.get(instr, [])) else []
                else:
                    ks = [None]
                for k in ks:
                    for ni in range(len(self.noise_seeds)):
                        if (self.jitter is None and self.adopted_cells is not None
                                and f"{instr}|s{si}|n{ni}" not in self.adopted_cells):
                            continue
                        yield (instr, si, k, ni)

    @property
    def n_cells(self) -> int:
        return sum(1 for _ in self.cell_keys())


def load_plan(path: Path) -> PlanInfo:
    p = path / PLAN_NAME if path.is_dir() else path
    return PlanInfo(json.loads(p.read_text()), p)


# ─────────────────────────── 좌표 파싱 ───────────────────────────

def parse_coords(rel: Path) -> tuple[str, str, str, int, int | None, int, str] | None:
    r"""grid-root 상대 meta 경로 → (plan_id, machine, instruction, s, k|None, n, arm).

    형태(§3.1.1): ``<plan_id>/<machine>/<instruction...>/s<i>[/k<r>]/n<j>/<arm>/<file>``
    **뒤에서부터** 판별한다 — arm, n<j>, 그다음 조각이 ``^k\d+$`` 면 지터 층이고 그 앞이
    s<i>, 아니면 그 자리가 곧 s<i>(legacy 2축). instruction 은 ``PPCC/bread`` 처럼 '/' 를
    포함할 수 있어 machine 뒤부터 s<i> 앞까지 전부다 (verify_grid.parse_coords 와 같은 규칙).
    """
    parts = rel.parts[:-1]          # 파일명 제거
    if len(parts) < 6:              # plan machine instr s n arm (지터 있으면 7)
        return None
    plan_id, machine, arm = parts[0], parts[1], parts[-1]
    m_n = _N_RE.match(parts[-2])
    if not m_n:
        return None
    m_k = _K_RE.match(parts[-3])
    s_at = -4 if m_k else -3        # k 층 유무로 s<i> 위치가 한 칸 밀린다
    m_s = _S_RE.match(parts[s_at])
    if not m_s:
        return None
    instruction = "/".join(parts[2:s_at])
    if not instruction:
        return None
    return (plan_id, machine, instruction, int(m_s.group(1)),
            int(m_k.group(1)) if m_k else None, int(m_n.group(1)), arm)


# ─────────────────────────── 값 직렬화 ───────────────────────────

def cell(v: Any) -> str:
    """TSV 한 칸. 결측은 빈 칸으로 둔다 — 추측으로 채우지 않는다 (docs/04 §5).

    list/dict 는 compact JSON 그대로 넣는다. csv 모듈의 인용·이스케이프를 쓰지 않으므로
    ``bindings``·``record_shape`` 열을 그대로 ``json.loads`` 할 수 있다.
    """
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, dict)):
        v = json.dumps(v, ensure_ascii=False, separators=(",", ":"),
                       sort_keys=isinstance(v, dict))
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _cell_key(instr: str, si: int, k: int | None, ni: int) -> str:
    """셀 키 (§3.1.1). `GridCell.key` 와 같은 문자열이어야 한다 — legacy 는 k 층이 없다."""
    return f"{instr}|s{si}|k{k}|n{ni}" if k is not None else f"{instr}|s{si}|n{ni}"


def write_tsv(path: Path, header: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    """원자적 쓰기. 인덱스는 파생물이라 갱신은 허용되지만 부분 파일은 남기지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(cell(r.get(c)) for c in header) + "\n")
    tmp.replace(path)


# ─────────────────────────── 본체 ───────────────────────────

def build(grid_root: Path, plans: dict[str, PlanInfo]) -> tuple[list[dict], list[dict], list[str]]:
    rollouts: list[dict] = []
    arms: list[dict] = []
    problems: list[str] = []

    for mp in sorted(grid_root.rglob("meta.json")):
        rel = mp.relative_to(grid_root)
        coords = parse_coords(rel)
        if coords is None:
            problems.append(
                f"{rel}: 좌표 경로가 아님 (<plan_id>/<machine>/<instr>/s<i>[/k<r>]/n<j>/<arm>)")
            continue
        plan_id, machine, instruction, s_idx, k_idx, n_idx, arm = coords
        try:
            meta = json.loads(mp.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: meta.json 파싱 실패 — {exc}")
            continue
        plan = plans.get(plan_id)
        has_pkl = (mp.parent / "rollout.pkl").is_file()

        # 경로 좌표가 정본이다. meta 가 다르면 분류가 어긋난 것 — 채우지 말고 보고한다.
        for key, want in (("plan_id", plan_id), ("grid_instruction", instruction),
                          ("scene_idx", s_idx), ("noise_idx", n_idx), ("machine", machine)):
            got = meta.get(key)
            if got is not None and str(got) != str(want):
                problems.append(f"{rel}: meta.{key}={got!r} != 경로 {want!r}")
        # 지터 축은 "없음(legacy)" 도 값이라 위 루프(None 이면 통과)로 못 잡는다 — 따로 본다.
        got_k = meta.get("jitter_reset_idx")
        if k_idx is None and got_k is not None:
            problems.append(f"{rel}: meta.jitter_reset_idx={got_k!r} 인데 경로에 k 층 없음")
        elif k_idx is not None and got_k is None:
            problems.append(f"{rel}: 경로 k{k_idx} 인데 meta.jitter_reset_idx 없음 (§3.1.1)")
        elif k_idx is not None and str(got_k) != str(k_idx):
            problems.append(f"{rel}: meta.jitter_reset_idx={got_k!r} != 경로 {k_idx!r}")

        row = {
            "plan_id": plan_id, "machine": machine, "grid_instruction": instruction,
            "scene_idx": s_idx, "jitter_reset_idx": k_idx, "noise_idx": n_idx,
            "armsig": meta.get("armsig") or arm,
            "kind": "activation" if (arm == BASE_ARM and has_pkl) else "eval",
            "has_pkl": 1 if has_pkl else 0,
            "version": (plan.version if plan else None),
            "plan_name": (plan.name if plan else None),
            "ckpt": meta.get("ckpt") or (plan.ckpt if plan else None),
            "machine_source": "grid_coord",
            "ckpt_source": "meta" if meta.get("ckpt") else ("plan" if plan else None),
            "meta_source": "meta_json",
            "rel_path": rel.parent.as_posix(),
        }
        for col in ("sig", "pkl_sha256", "model", "task", "task_id", "instruction",
                    "env_name", "env_seed", "inference_seed", "episode_idx", "success",
                    "steps", "n_inferences", "n_action_steps", "chunk_len") + CAPTURE_COLS:
            row.setdefault(col, meta.get(col))
        if arm == BASE_ARM:
            rollouts.append(row)
        # arm 행은 rollouts.tsv 에 넣지 않는다 — 이 표의 행은 base 셀이다.
        # arm 의 신원·개입 파라미터는 arm_bindings.tsv 가 담는다 (§3.3).

    for cp in sorted(grid_root.rglob("config.json")):
        rel = cp.relative_to(grid_root)
        coords = parse_coords(rel)
        if coords is None:
            problems.append(f"{rel}: config.json 이 좌표 경로 밖")
            continue
        plan_id, machine, instruction, s_idx, k_idx, n_idx, arm = coords
        if arm == BASE_ARM:
            problems.append(f"{rel}: base 에 config.json — base 는 개입 없는 arm 이다 (§3.3)")
            continue
        try:
            cfg = json.loads(cp.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: config.json 파싱 실패 — {exc}")
            continue
        armsig = cfg.get("armsig")
        if armsig and not arm.startswith(armsig):
            problems.append(f"{rel}: 디렉토리 {arm!r} 이 config.armsig={armsig!r} 와 불일치")
        arms.append({
            "plan_id": plan_id, "machine": machine, "grid_instruction": instruction,
            "scene_idx": s_idx, "jitter_reset_idx": k_idx, "noise_idx": n_idx,
            "armsig": armsig or arm,
            "op": cfg.get("op"), "beta": cfg.get("beta"),
            "bindings": cfg.get("bindings"),
            "token_select": cfg.get("token_select"), "denoise": cfg.get("denoise"),
            "steer_from_record": cfg.get("steer_from_record"),
            "gated_phases": cfg.get("gated_phases"),
            "rel_path": rel.parent.as_posix(),
        })

    return rollouts, arms, problems


def report(grid_root: Path, plans: dict[str, PlanInfo], rollouts: list[dict],
           arms: list[dict], problems: list[str], list_limit: int) -> int:
    """검증 리포트 → stdout. 반환값 = 위반 건수."""
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    print(f"# build_grid_index  실행시각 {now}")
    print(f"# grid-root {grid_root}")
    print(f"# ⚠ 수집·전송이 진행 중이면 셀 수는 이 시각 기준 스냅샷이다.\n")

    by_plan: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        by_plan[r["plan_id"]].append(r)

    print("## 계획별 셀 수 (base)")
    for plan_id in sorted(set(by_plan) | set(plans)):
        plan = plans.get(plan_id)
        rows = by_plan.get(plan_id, [])
        planned = f"/{plan.n_cells}" if plan else "/? (계획 미지정)"
        name = plan.name if plan else "?"
        print(f"  {plan_id}  {name:<16s} {len(rows):>5d}{planned}")
        per_instr = Counter(r["grid_instruction"] for r in rows)
        expected = Counter()
        want_keys: set[tuple] = set()
        if plan:
            for instr, si, k, ni in plan.cell_keys():
                expected[instr] += 1
                want_keys.add((instr, si, k, ni))
        for instr in sorted(set(per_instr) | set(expected)):
            exp = expected.get(instr)
            mark = "✓" if exp is not None and per_instr.get(instr, 0) == exp else " "
            print(f"    {mark} {instr:<28s} {per_instr.get(instr, 0):>5d}"
                  f"{'/' + str(exp) if exp is not None else ''}")
        if plan:
            # 계획 대비 결손·초과. 셀 키는 3축 instr|s<i>|k<r>|n<j> (legacy 는 k 없음).
            have_keys = {(r["grid_instruction"], r["scene_idx"],
                          r["jitter_reset_idx"], r["noise_idx"]) for r in rows}
            missing, stray = sorted(want_keys - have_keys), sorted(have_keys - want_keys)
            print(f"    결손 {len(missing)} · 계획 밖 {len(stray)}")
            for k in missing[:list_limit]:
                print(f"        - {_cell_key(*k)}")
            if len(missing) > list_limit:
                print(f"        ... 외 {len(missing) - list_limit}")
            for k in stray[:list_limit]:
                print(f"        ! 계획 밖 {_cell_key(*k)}")

    # ── machine 결측 — 이번 재수집 라운드의 존재 이유 (§3.2) ──
    no_machine = [r for r in rollouts if not r.get("machine")]
    print(f"\n## machine 결측 {len(no_machine)}  (0 이어야 함 — §3.2)")
    for r in no_machine[:list_limit]:
        print(f"    ✗ {r['rel_path']}")
    if len(no_machine) > list_limit:
        print(f"    ... 외 {len(no_machine) - list_limit}")
    print(f"   machine 분포: {dict(Counter(r.get('machine') or '(없음)' for r in rollouts))}")

    # ── success 분포 ──
    print("\n## success 분포")
    for plan_id in sorted(by_plan):
        c = Counter(str(r.get("success")) if r.get("success") is not None else "(없음)"
                    for r in by_plan[plan_id])
        n1, n0 = c.get("1", 0), c.get("0", 0)
        sr = f"{n1 / (n1 + n0):.3f}" if (n1 + n0) else "n/a"
        print(f"  {plan_id}  succ={n1} fail={n0} 결측={c.get('(없음)', 0)}  SR={sr}")
        per = defaultdict(lambda: [0, 0])
        for r in by_plan[plan_id]:
            if r.get("success") in (0, 1, "0", "1"):
                per[r["grid_instruction"]][int(r["success"]) == 1] += 1
        for instr in sorted(per):
            f_, s_ = per[instr]
            print(f"      {instr:<28s} succ={s_:>4d} fail={f_:>4d} SR={s_/(s_+f_):.3f}")

    # ── 캡처 밀도 5열 결측 (§4) ──
    print("\n## 캡처 밀도 5열 결측 (kind=activation 행 대상)")
    act = [r for r in rollouts if r["kind"] == "activation"]
    for col in CAPTURE_COLS:
        miss = [r for r in act if r.get(col) in (None, "", [])]
        print(f"    {col:<20s} 결측 {len(miss):>4d} / {len(act)}")
        for r in miss[:3]:
            print(f"        ✗ {r['rel_path']}")
    shapes = Counter(cell(r.get("record_shape")) for r in act)
    print(f"    record_shape 분포: {dict(shapes)}")

    # ── 중복 좌표 ──
    exact = Counter((r["plan_id"], r["machine"], r["grid_instruction"], r["scene_idx"],
                     r["jitter_reset_idx"], r["noise_idx"]) for r in rollouts)
    dup_exact = [k for k, v in exact.items() if v > 1]
    cross: dict[tuple, set[str]] = defaultdict(set)
    for r in rollouts:
        cross[(r["plan_id"], r["grid_instruction"], r["scene_idx"],
               r["jitter_reset_idx"], r["noise_idx"])].add(r["machine"])
    dup_cross = [k for k, v in cross.items() if len(v) > 1]
    print(f"\n## 중복 좌표")
    print(f"    같은 (plan,machine,instr,s,k,n) 에 base 2개: {len(dup_exact)}")
    for k in dup_exact[:list_limit]:
        print(f"        ✗ {k}")
    print(f"    같은 좌표가 여러 machine 에 (§3.2 층화 위험): {len(dup_cross)}")
    for k in dup_cross[:list_limit]:
        print(f"        ! {k} -> {sorted(cross[k])}")
    dup_sig = [s for s, v in Counter(r.get("sig") for r in rollouts if r.get("sig")).items() if v > 1]
    print(f"    같은 sig 가 여러 셀에 (내용 중복)          : {len(dup_sig)}")

    print(f"\n## arm 디렉토리 {len(arms)}  (config.json 보유분. 현 라운드는 0 이 정상)")

    n_viol = len(problems) + len(no_machine) + len(dup_exact)
    if problems:
        print(f"\n## 산출물 문제 {len(problems)} 건")
        for p in problems[:50]:
            print(f"    ✗ {p}")
        if len(problems) > 50:
            print(f"    ... 외 {len(problems) - 50}")
    print(f"\n판정: 위반 {n_viol} 건 "
          f"(경로/meta 불일치 {len(problems)} + machine 결측 {len(no_machine)} + 좌표중복 {len(dup_exact)})")
    return n_viol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True, type=Path,
                    help="좌표 저장소 루트 (아래에 <plan_id>/<machine>/... 이 있다)")
    ap.add_argument("--plan-json", type=Path, action="append", default=[],
                    help="collection_plan.json (복수 지정 가능). 없어도 되지만 주면 "
                         "계획 대비 결손·version 등을 보강한다")
    ap.add_argument("--out", required=True, type=Path, help="인덱스 출력 디렉토리")
    ap.add_argument("--list-limit", type=int, default=20)
    ap.add_argument("--strict", action="store_true",
                    help="위반이 있으면 표를 쓰지 않고 종료")
    args = ap.parse_args()

    if not args.grid_root.is_dir():
        print(f"ERROR: --grid-root 없음: {args.grid_root}", file=sys.stderr)
        return 2

    plans: dict[str, PlanInfo] = {}
    for pj in args.plan_json:
        p = load_plan(pj)
        plans[p.plan_id] = p
    # grid-root 하위 plan_id 자동 발견 — 계획 파일이 없어도 인덱스는 만들어진다.
    for d in sorted(args.grid_root.iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")) and d.name not in plans:
            local = d / PLAN_NAME
            if local.is_file():
                p = load_plan(local)
                plans.setdefault(p.plan_id, p)

    rollouts, arms, problems = build(args.grid_root, plans)
    n_viol = report(args.grid_root, plans, rollouts, arms, problems, args.list_limit)

    if args.strict and n_viol:
        print("\n--strict: 위반이 있어 표를 쓰지 않았다.")
        return 1
    write_tsv(args.out / "rollouts.tsv", ROLLOUT_COLS, rollouts)
    write_tsv(args.out / "arm_bindings.tsv", ARM_COLS, arms)
    print(f"\n쓴 파일: {args.out / 'rollouts.tsv'} ({len(rollouts)} 행), "
          f"{args.out / 'arm_bindings.tsv'} ({len(arms)} 행)")
    return 1 if n_viol else 0


if __name__ == "__main__":
    raise SystemExit(main())
