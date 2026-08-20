#!/usr/bin/env python3
"""좌표 그리드 수집 검증기 — docs/04 §3 레이아웃 전수 검사 + 계획 대비 결손 리포트.

구 `verify_rollout_collection.py`(stem 레이아웃·n16 상수 하드코딩)의 좌표판 재작성.
체크리스트(검토 지도)와의 대응: #1 캡처 밀도 · #2 분류 정합 · #3 손상(sig) 을 여기서,
#5(bias)는 별도 분석, 결손은 `plan.missing()` 으로.

롤링 shipper 구조를 전제한다: 로컬 staging 엔 아직 안 보낸 셀만 남으므로,
결손 판정은 디렉토리 스캔 ∪ `--done-list`(shipped_cells.txt, 복수 가능) 합산으로 한다.
같은 도구가 아카이브 루트(승준)에서도 그대로 돈다 (torch 불필요 — --deep 만 예외).

사용:
    python scripts/collect/verify_grid.py --grid-root outputs/collect/grid_staging \
        --plan-json outputs/collect/n15_grid_v1/collection_plan.json \
        [--done-list outputs/collect/grid_staging/shipped_cells.txt] \
        [--deep 3] [--require-complete] [--list-missing 20]

종료 코드: 산출물 오류가 있으면 1. 결손은 수집 진행 중엔 정보로만 보고하고,
`--require-complete` 일 때만 오류로 친다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collect.plan import BASE_ARM, CollectionPlan  # noqa: E402

# 좌표 꼬리: .../<instruction...>/s<i>/n<j>/<arm>/meta.json
_COORD_RE = re.compile(r"^s(\d+)$"), re.compile(r"^n(\d+)$")

# base(수집, pkl 有) / arm-eval(pkl 無) 필수 파일 — docs/04 §3.1·§3.3
BASE_FILES = ("rollout.pkl", "traj.csv", "video.mp4", "meta.json")
ARM_FILES = ("traj.csv", "video.mp4", "meta.json", "config.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_coords(meta_path: Path, plan_id: str):
    """meta.json 경로 → (machine, instruction, scene_idx, noise_idx, arm). 실패 시 None.

    경로 형태: <...>/<plan_id>/<machine>/<instruction...>/s<i>/n<j>/<arm>/meta.json
    instruction 은 '/' 를 포함할 수 있어 plan_id·machine 뒤부터 s<i> 앞까지 전부다.
    """
    parts = meta_path.parts
    try:
        pi = len(parts) - 1 - parts[::-1].index(plan_id)
    except ValueError:
        return None
    tail = parts[pi + 1 : -1]  # machine ... arm
    if len(tail) < 4:
        return None
    machine, arm = tail[0], tail[-1]
    m_s, m_n = _COORD_RE[0].match(tail[-3]), _COORD_RE[1].match(tail[-2])
    if not (m_s and m_n):
        return None
    instruction = "/".join(tail[1:-3])
    return machine, instruction, int(m_s.group(1)), int(m_n.group(1)), arm


def check_cell_dir(dirpath: Path, meta: dict, coords, plan: CollectionPlan, cell,
                   errors: list[str]) -> None:
    machine, instruction, s_idx, n_idx, arm = coords
    loc = str(dirpath)

    # ── 파일 완결성 ──
    required = BASE_FILES if arm == BASE_ARM else ARM_FILES
    for name in required:
        f = dirpath / name
        if not f.is_file() or f.stat().st_size == 0:
            errors.append(f"{loc}: {name} 없음 또는 0바이트")

    # ── 분류 정합: 경로 좌표 == meta 기록 ──
    for key, want in (("plan_id", plan.plan_id), ("grid_instruction", instruction),
                      ("scene_idx", s_idx), ("noise_idx", n_idx)):
        got = meta.get(key)
        if got is not None and got != want:
            errors.append(f"{loc}: meta.{key}={got!r} != 경로 {want!r}")
    if meta.get("machine") is not None and meta["machine"] != machine:
        errors.append(f"{loc}: meta.machine={meta['machine']!r} != 경로 {machine!r}")
    if meta.get("ckpt") is not None and meta["ckpt"] != plan.ckpt:
        errors.append(f"{loc}: meta.ckpt={meta['ckpt']!r} != plan.ckpt={plan.ckpt!r}")

    # ── 재현 조건: seed 가 계획의 그 칸인가 ──
    if cell is None:
        errors.append(f"{loc}: 계획에 없는 좌표 (instruction={instruction!r} s{s_idx} n{n_idx})")
    else:
        if meta.get("env_seed") is not None and int(meta["env_seed"]) != cell.env_seed:
            errors.append(f"{loc}: env_seed={meta['env_seed']} != plan {cell.env_seed}")
        if (meta.get("inference_seed") is not None
                and int(meta["inference_seed"]) != cell.inference_seed):
            errors.append(
                f"{loc}: inference_seed={meta['inference_seed']} != plan {cell.inference_seed}")

    if meta.get("success") not in (0, 1, None):
        errors.append(f"{loc}: success={meta.get('success')!r} — 0/1 아님")

    # ── 손상: pkl sha 재계산 == meta 기록 (base 만 — arm eval 은 pkl 없음) ──
    pkl = dirpath / "rollout.pkl"
    if pkl.is_file():
        want = meta.get("pkl_sha256")
        if want:
            got = sha256_file(pkl)
            if got != want:
                errors.append(f"{loc}: pkl sha256 불일치 (기록 {want[:16]} != 실물 {got[:16]})")
            elif meta.get("sig") and meta["sig"] != got[:16]:
                errors.append(f"{loc}: sig={meta['sig']} != sha256[:16]={got[:16]}")
        else:
            errors.append(f"{loc}: meta 에 pkl_sha256 없음")

    # ── 캡처 밀도 5열 (base 만) — docs/04 §4·§6 ──
    if arm == BASE_ARM:
        for col in ("capture_token_mode", "feature_kind", "feature_axes",
                    "record_shape", "capture_layers"):
            if meta.get(col) is None:
                errors.append(f"{loc}: 캡처 밀도 열 {col} 없음")
        layers = meta.get("capture_layers")
        if layers is not None and list(layers) != list(plan.capture_layers):
            errors.append(f"{loc}: capture_layers={layers} != plan {plan.capture_layers}")
        shape = meta.get("record_shape")
        if shape:
            # n15 multilayer 표준 [L, K, T, D]. 판정은 ndim 이 아니라 축 크기(docs/04).
            if layers is not None and shape[0] != len(layers):
                errors.append(f"{loc}: record_shape[0]={shape[0]} != 층수 {len(layers)}")
            if len(shape) >= 2 and shape[1] != plan.denoise_k:
                errors.append(f"{loc}: record_shape[1]={shape[1]} != denoise_k {plan.denoise_k}")
            if len(shape) >= 3 and shape[2] <= 1:
                errors.append(f"{loc}: 토큰축 {shape[2]} — pooled 저장은 캡처 밀도 위반")

    # ── traj.csv 최소 형태 ──
    traj = dirpath / "traj.csv"
    if traj.is_file() and traj.stat().st_size > 0:
        with traj.open() as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            errors.append(f"{loc}: traj.csv 에 데이터 행 없음")


def deep_check(dirpath: Path, errors: list[str]) -> None:
    """--deep 표본: pkl 을 실제 로드해 NaN/Inf·record 수·dtype 확인 (torch 필요할 수 있음)."""
    import pickle
    try:
        import numpy as np
        with (dirpath / "rollout.pkl").open("rb") as f:
            payload = pickle.load(f)
    except ModuleNotFoundError as exc:
        errors.append(f"{dirpath}: deep 로드 실패 — {exc} (numpy/torch 있는 python 으로 재시도)")
        return
    hs = payload.get("hidden_states")
    if not hs:
        errors.append(f"{dirpath}: hidden_states 비어있음")
        return
    with (dirpath / "traj.csv").open() as f:
        n_rows = sum(1 for _ in f) - 1
    if len(hs) != n_rows:
        errors.append(f"{dirpath}: records {len(hs)} != traj.csv 행 {n_rows}")
    arr = np.asarray(hs[0], dtype=np.float32)
    if not np.isfinite(arr).all():
        errors.append(f"{dirpath}: hidden_states[0] 에 NaN/Inf")
    if list(arr.shape) != list(payload.get("record_shape", arr.shape)):
        # payload 쪽 record_shape 는 meta 와 같아야 함 — 다르면 기록 오류
        errors.append(f"{dirpath}: 실물 shape {list(arr.shape)} != record_shape 기록")


def caption_burnin_check(video_path: Path, errors: list[str]) -> None:
    """수집 영상 하단 caption burn-in tripwire (2026-08 kanu 219판 사고 재발 감지).

    upstream VideoRecordingWrapper(overlay_text=True)는 프레임 하단 장면 위에
    검은 사각형+흰 텍스트를 굽는다. 마지막 프레임의 하단 밴드에서 그 시그니처
    (저휘도 배경 비율 高 + 순백 텍스트 픽셀 존재)를 탐지한다. reader 가 없으면
    경고만 남긴다(무음 통과 금지).
    """
    try:
        import imageio.v2 as iio
    except ModuleNotFoundError as exc:
        errors.append(f"{video_path}: caption 검사 불가 — {exc} (imageio 있는 python 으로 재시도)")
        return
    try:
        reader = iio.get_reader(str(video_path))
        n = reader.count_frames()
        frame = reader.get_data(max(0, n - 1))
        reader.close()
    except Exception as exc:  # 손상 mp4 등
        errors.append(f"{video_path}: caption 검사 중 읽기 실패 — {exc}")
        return
    # 판정식: 하단 40% 에서 "검정 배경(≥0.55) + 순백 텍스트로 행이 거의 채워짐(≥0.85)"인
    # 연속 행 run ≥ 8 + run 내 순백 픽셀 > 1%. 실측 마진: kanu burn-in 30/30 탐지
    # (run≥9), 클린 3머신 30/30 통과 (run=0).
    band = frame[int(frame.shape[0] * 0.60):]
    gray = band.mean(axis=2)
    row_dark = (gray < 40).mean(axis=1)
    row_white = (gray > 235).mean(axis=1)
    cap_rows = (row_dark >= 0.55) & ((row_dark + row_white) >= 0.85)
    best, cur, s, best_s = 0, 0, 0, 0
    for i, d in enumerate(cap_rows):
        cur = cur + 1 if d else 0
        if cur == 1:
            s = i
        if cur > best:
            best, best_s = cur, s
    white_in_run = 0.0
    if best >= 8:
        seg = gray[best_s:best_s + best]
        white_in_run = float((seg > 235).mean())
    if best >= 8 and white_in_run > 0.01:
        errors.append(
            f"{video_path}: 하단 caption burn-in 의심 (연속 caption 행 {best}, "
            f"white {white_in_run:.3f}) — overlay_text 경로 부활 여부 확인 필요")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True, type=Path)
    ap.add_argument("--plan-json", required=True, type=Path)
    ap.add_argument("--done-list", type=Path, action="append", default=[],
                    help="shipped_cells.txt (복수 가능) — 결손 판정에 합산")
    ap.add_argument("--deep", type=int, default=0, help="pkl 실로드 표본 수 (torch 필요)")
    ap.add_argument("--check-captions", type=int, default=20,
                    help="video.mp4 하단 caption burn-in 검사 표본 수 (기본 20, -1=전수, 0=끔)")
    ap.add_argument("--require-complete", action="store_true",
                    help="결손 셀도 오류로 취급 (수집 완주 후 최종 검수용)")
    ap.add_argument("--list-missing", type=int, default=10, help="결손 셀 출력 상한")
    args = ap.parse_args()

    plan = CollectionPlan.load(args.plan_json)
    cell_by_key = {c.key: c for c in plan.cells()}
    cell_by_coord = {(c.instruction, c.scene_idx, c.noise_idx): c for c in plan.cells()}

    errors: list[str] = []
    base_machines: dict[str, set[str]] = defaultdict(set)   # cell.key -> machines (중복 검출)
    arm_mismatch = 0
    base_dirs: list[Path] = []
    n_arm = 0

    metas = sorted(args.grid_root.rglob("meta.json"))
    for mp in metas:
        coords = parse_coords(mp, plan.plan_id)
        if coords is None:
            errors.append(f"{mp}: plan_id={plan.plan_id} 좌표 경로가 아님")
            continue
        machine, instruction, s_idx, n_idx, arm = coords
        try:
            meta = json.loads(mp.read_text())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mp}: meta.json 파싱 실패 — {exc}")
            continue
        cell = cell_by_coord.get((instruction, s_idx, n_idx))
        check_cell_dir(mp.parent, meta, coords, plan, cell, errors)
        key = f"{instruction}|s{s_idx}|n{n_idx}"
        if arm == BASE_ARM:
            base_machines[key].add(machine)
            base_dirs.append(mp.parent)
        else:
            n_arm += 1
            # base·arm 같은 머신 규칙 (docs/04 §3.2) — base 가 로컬에 있으면 대조
            if key in base_machines and machine not in base_machines[key]:
                arm_mismatch += 1
                errors.append(f"{mp.parent}: arm 머신 {machine} 이 base {base_machines[key]} 와 다름")

    for key, machines in base_machines.items():
        if len(machines) > 1:
            errors.append(f"셀 {key}: base 가 머신 {sorted(machines)} 에 중복 수집됨")

    # ── 결손: 디렉토리 ∪ done-list ──
    collected = set(base_machines)
    for dl in args.done_list:
        if dl.is_file():
            for line in dl.read_text().splitlines():
                line = line.strip()
                if line:
                    collected.add(line)
        else:
            errors.append(f"--done-list {dl}: 파일 없음")
    unknown_done = collected - set(cell_by_key)
    for k in sorted(unknown_done):
        errors.append(f"done-list 의 셀 {k!r} 이 계획에 없음")
    missing = plan.missing(collected)

    if args.deep and base_dirs:
        for d in random.Random(0).sample(base_dirs, min(args.deep, len(base_dirs))):
            deep_check(d, errors)

    if args.check_captions and base_dirs:
        vids = [d / "video.mp4" for d in base_dirs if (d / "video.mp4").exists()]
        if args.check_captions > 0:
            vids = random.Random(1).sample(vids, min(args.check_captions, len(vids)))
        for v in vids:
            caption_burnin_check(v, errors)

    # ── 리포트 ──
    per_instr = defaultdict(lambda: [0, 0])  # instruction -> [expected, collected]
    for c in plan.cells():
        per_instr[c.instruction][0] += 1
        if c.key in collected:
            per_instr[c.instruction][1] += 1
    print(f"plan_id={plan.plan_id}  root={args.grid_root}")
    print(f"발견: base {len(base_dirs)} (로컬) + done-list {len(collected) - len(base_machines)}"
          f" / arm {n_arm}")
    for instr, (exp, got) in sorted(per_instr.items()):
        mark = "✓" if got == exp else " "
        print(f"  {mark} {instr:<40s} {got:>4d}/{exp}")
    print(f"결손 {len(missing)} / {plan.n_cells}")
    for c in missing[: args.list_missing]:
        print(f"    missing {c.key} (env_seed={c.env_seed}, inf_seed={c.inference_seed})")
    if len(missing) > args.list_missing:
        print(f"    ... 외 {len(missing) - args.list_missing}")

    if args.require_complete and missing:
        errors.append(f"결손 {len(missing)} 셀 (--require-complete)")
    if errors:
        print(f"\nERROR {len(errors)} 건:")
        for e in errors[:50]:
            print(f"  ✗ {e}")
        if len(errors) > 50:
            print(f"  ... 외 {len(errors) - 50}")
        return 1
    print("OK — 산출물 오류 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
