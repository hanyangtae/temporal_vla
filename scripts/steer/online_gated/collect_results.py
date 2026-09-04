#!/usr/bin/env python3
"""online-gated eval 사이드카 → per_episode.tsv (+ 요약).

캡처-OFF eval 의 판정 원천은 `raw_rollouts/<TASK>/<cell>/task*--ep*--succ{0,1}.json`
사이드카다 (`http_feature_collect.py --no-features`). 여기서 판정(success)과 개입 감사
필드(trigger_step / phase_at_trigger / phase_gated_flags / serve_steering)를 뽑아
한 arm 의 표를 만든다 — 로그 파싱 금지, 판정은 항상 산출물에서.

replay 모드(`--cells-tsv`)에서는 셀 좌표(scene_idx/noise_idx)와 **수집 당시 판정**
(collection_success)을 함께 붙인다. 같은 (env_seed, inference_seed) 를 재생한 것이라
  - 구제(rescue) = 수집 실패 셀 중 이번에 성공
  - 파손(break)  = 수집 성공 셀 중 이번에 실패
가 되고, fit 셀(FIT_SCENES × FIT_NOISES) 기준 사분면
(seen/unseen scene × seen/unseen noise) 별로 요약을 찍는다.

열:
  ep  scene_idx  env_seed  inference_seed  success  steps  n_inferences
  trigger_step  phase_at_trigger  n_gated  gated_mode  arm  slug  beta  op  serve_gpu  run_tag
  noise_idx  seen_scene  seen_noise  quadrant  collection_success
  cell_key  jitter_reset_idx  jitter_idx
(새 열은 **뒤에** 붙인다 — 기존 소비자의 열 위치를 깨지 않기 위함.)

셀 표(`replay_cells.py` 출력, v6 11열)의 열은 `scene_idx noise_idx env_seed
inference_seed collection_success task env_name instruction_text jitter_idx
jitter_reset_idx lang` 이다 — 3축 좌표 규약 docs/04 §3.1.1(v6 = s/j/n).
지터 좌표는 **v6 면 `jitter_idx`(j), legacy(v5 k 층) 면 `jitter_reset_idx`(k)** 이고,
표에서 v6 여부는 `jitter_idx` 열이 정수인지로 판정한다(legacy 는 `"NA"`).
한 scene 의 지터 셀들이 env_seed 를 공유하므로 셀 되찾기 키는 **`ep`**
(= 평탄 cell_key*EP_IDX_STRIDE + noise, `--ep-idx-stride`) 다. `cell_key` 열은 그 평탄
파생값(지터 scene*100+j / 2축 legacy scene)으로, 러너의 episode_idx·TRIGGER_TSV 조회
키와 같은 규약이다. 옛 9열 표(jitter_reset_idx 만)도 그대로 읽힌다.

사용:
    # replay
    python scripts/steer/online_gated/collect_results.py \
        --job-dir <out>/OvenRack_out/online --cells-tsv <logs>/cells_OvenRack_out.tsv \
        --fit-scenes 0,1,2,3,4 --fit-noises 0,1,2,3,4 \
        --arm online --slug OvenRack_out --expect 40
    # fresh (구 방식)
    python scripts/steer/online_gated/collect_results.py \
        --job-dir <out>/OvenRack_out/online --scenes-tsv <logs>/scenes_OvenRack_out.tsv \
        --arm online --slug OvenRack_out --expect 20
`--expect` 를 주면 행 수가 모자랄 때 exit 13 (러너의 미완 판정).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

V4_SCENE_STRIDE = 100   # 평탄 cell_key = scene*100 + 지터좌표 (판 번호 유일성 파생값)
STEM_RE = re.compile(r"task(?P<tid>\d+)--ep(?P<ep>\d+)--succ(?P<succ>[01])$")
COLUMNS = ("ep", "scene_idx", "env_seed", "inference_seed", "success", "steps",
           "n_inferences", "trigger_step", "phase_at_trigger", "n_gated", "gated_mode",
           "arm", "slug", "beta", "op", "serve_gpu", "run_tag",
           "noise_idx", "seen_scene", "seen_noise", "quadrant", "collection_success",
           "cell_key", "jitter_reset_idx", "jitter_idx")


def cell(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(int(v))
    return str(v).replace("\t", " ")


def load_scenes(path: Path | None) -> dict[int, int]:
    """env_seed → scene_idx (fresh 모드 scenes tsv)."""
    if path is None:
        return {}
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        p = ln.split("\t")
        out[int(p[1])] = int(p[0])
    return out


EMPTY = ("", "base", "NA", "None")


def load_cells(path: Path | None, ep_stride: int = 100) -> tuple[dict, bool]:
    """셀 표(replay_cells.py 출력) → (ep → 셀, 지터 축 유무).

    v6 열(11) = `scene_idx noise_idx env_seed inference_seed collection_success
    task env_name instruction_text jitter_idx jitter_reset_idx lang`.
    legacy 열(9) = `... instruction_text jitter_reset_idx` (v5 k 층).
    두 경우 모두 **9번째 열(p[8])이 지터 좌표**다 — v6 는 j, legacy 는 k. v6 표에서는
    p[9] 에 plan 유래 reset_idx 가 따로 실린다 (좌표 아님).

    한 scene 의 지터 셀들이 **같은 env_seed 를 공유**하므로 seed 쌍으로는 셀을 구분할 수
    없다 → 키는 러너와 동일한 `ep = 평탄 cell_key*stride + noise`
    (평탄 cell_key = 지터행 `scene*100+j` / 2축 legacy행 `scene`).
    지터 열이 없던 옛 8열 표도 legacy 로 그대로 읽힌다 (ep = scene*stride + noise).
    """
    if path is None:
        return {}, False
    out: dict = {}
    has_jitter = False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        p = ln.split("\t")
        if len(p) == 10:
            raise SystemExit(
                f"{path}: 10열 셀 표 = 구 v4(평탄 cell id + base_scene) 포맷이다 — "
                "replay_cells.py 로 다시 생성할 것 (docs/04 §3.1.1)")
        if len(p) > 11:
            raise SystemExit(f"{path}: {len(p)}열 셀 표 — 9(legacy)/11(v6) 열만 읽는다")
        si, ni, cs = int(p[0]), int(p[1]), p[4]
        raw_j = p[8].strip() if len(p) > 8 else ""          # v6 = jitter_idx, legacy = k
        raw_reset = p[9].strip() if len(p) > 9 else ""      # v6 전용 출처 열
        if raw_j in EMPTY:
            # v6 표인데 jitter_idx 가 비면 legacy 행(k) 이거나 2축 base 행이다.
            jit_coord = "base" if raw_reset in EMPTY else raw_reset
        else:
            jit_coord = raw_j
        if jit_coord == "base":
            cell_key = si
            reset = raw_reset or "base"
        else:
            has_jitter = True
            cell_key = si * V4_SCENE_STRIDE + int(jit_coord)
            reset = raw_reset or jit_coord     # legacy 표는 좌표 = reset 횟수
        out[cell_key * ep_stride + ni] = {
            "scene_idx": si, "noise_idx": ni,          # 사분면 태깅은 scene 기준
            "cell_key": cell_key,
            "jitter_idx": jit_coord if raw_j not in EMPTY else "NA",
            "jitter_reset_idx": reset,
            "collection_success": None if cs in ("", "NA") else int(cs),
        }
    return out, has_jitter


def parse_int_list(spec: str) -> set[int]:
    out: set[int] = set()
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def quadrant_report(rows: list[dict]) -> list[str]:
    """사분면별 SR + 구제율/파손율. 구제·파손은 수집 판정이 있는 판만 분모."""
    lines = ["[quadrant] quadrant                 n   SR     rescue(fail→succ)  break(succ→fail)"]
    order = ["seen_scene×seen_noise", "seen_scene×unseen_noise",
             "unseen_scene×seen_noise", "unseen_scene×unseen_noise"]
    groups = {q: [r for r in rows if r["quadrant"] == q] for q in order}
    groups["__all__"] = list(rows)
    for q in order + ["__all__"]:
        g = groups.get(q) or []
        if not g:
            continue
        n = len(g)
        sr = sum(r["success"] for r in g) / n
        cf = [r for r in g if r["collection_success"] == 0]
        cs = [r for r in g if r["collection_success"] == 1]
        n_res = sum(1 for r in cf if r["success"] == 1)
        n_brk = sum(1 for r in cs if r["success"] == 0)
        res = f"{n_res}/{len(cf)}" + (f" ({n_res/len(cf):.2f})" if cf else "")
        brk = f"{n_brk}/{len(cs)}" + (f" ({n_brk/len(cs):.2f})" if cs else "")
        lines.append(f"[quadrant] {q:24s} {n:3d} {sr:.3f}  {res:17s}  {brk}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", type=Path, required=True, help="<out>/<slug>/<arm>")
    ap.add_argument("--scenes-tsv", type=Path, default=None, help="fresh 모드 scene 표")
    ap.add_argument("--cells-tsv", type=Path, default=None,
                    help="replay 모드 셀 표 (replay_cells.py 출력)")
    ap.add_argument("--fit-scenes", default="0,1,2,3,4", help="연산자·detector fit scene")
    ap.add_argument("--fit-noises", default="0,1,2,3,4", help="연산자·detector fit noise")
    ap.add_argument("--arm", default="NA")
    ap.add_argument("--slug", default="NA")
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="기본 <job-dir>/per_episode.tsv")
    ap.add_argument("--ep-idx-stride", type=int, default=100,
                    help="러너 EP_IDX_STRIDE (v4 셀 표를 ep 로 되찾을 때 사용)")
    args = ap.parse_args()

    seed2scene = load_scenes(args.scenes_tsv)
    cells, _has_jitter = load_cells(args.cells_tsv, args.ep_idx_stride)
    fit_scenes = parse_int_list(args.fit_scenes)
    fit_noises = parse_int_list(args.fit_noises)
    rows = []
    for jp in sorted((args.job_dir / "raw_rollouts").rglob("task*--ep*--succ*.json")):
        m = STEM_RE.match(jp.stem)
        if m is None:
            continue
        sc = json.loads(jp.read_text(encoding="utf-8"))
        steer = sc.get("serve_steering") or {}
        flags = sc.get("phase_gated_flags") or []
        env_seed = sc.get("scenario_seed", sc.get("seed"))
        inf_seed = sc.get("inference_seed")
        # replay: ep(= 평탄 cell_key*stride + noise) 로 셀을 되찾아 좌표·수집판정을
        # 붙인다 — 지터 축에서는 (env_seed, inference_seed) 가 셀 간 공유라 키가 안 된다.
        cellrec = None
        if cells:
            cellrec = cells.get(int(m.group("ep")))
            if cellrec is None:
                raise SystemExit(
                    f"{jp}: ep={m.group('ep')} 가 셀 표에 없다 — 재생 좌표가 어긋났다 "
                    f"(--ep-idx-stride={args.ep_idx_stride} 가 러너와 같은지 확인)")
        scene_idx = (cellrec["scene_idx"] if cellrec is not None else
                     (seed2scene.get(int(env_seed)) if env_seed is not None else None))
        noise_idx = cellrec["noise_idx"] if cellrec is not None else None
        seen_scene = None if scene_idx is None else int(scene_idx in fit_scenes)
        seen_noise = None if noise_idx is None else int(noise_idx in fit_noises)
        quad = None
        if seen_scene is not None and seen_noise is not None:
            quad = (f"{'seen' if seen_scene else 'unseen'}_scene×"
                    f"{'seen' if seen_noise else 'unseen'}_noise")
        rows.append({
            "ep": int(m.group("ep")),
            "scene_idx": scene_idx,
            "env_seed": env_seed,
            "inference_seed": inf_seed,
            "noise_idx": noise_idx,
            "seen_scene": seen_scene,
            "seen_noise": seen_noise,
            "quadrant": quad,
            "collection_success": (cellrec["collection_success"]
                                   if cellrec is not None else None),
            # 지터 축 (docs/04 §3.1.1): 평탄 cell id 파생값 + 좌표 j(v6)/k(legacy).
            "cell_key": cellrec.get("cell_key") if cellrec is not None else None,
            "jitter_reset_idx": (cellrec.get("jitter_reset_idx")
                                 if cellrec is not None else None),
            "jitter_idx": (cellrec.get("jitter_idx") if cellrec is not None else None),
            # 스템의 succ 와 본문 판정이 어긋나면 사이드카가 손상된 것 → fail-loud
            "success": int(sc.get("episode_success", m.group("succ"))),
            "steps": sc.get("steps"),
            "n_inferences": sc.get("n_inferences"),
            "trigger_step": sc.get("trigger_step"),
            "phase_at_trigger": sc.get("phase_at_trigger"),
            "n_gated": sum(1 for f in flags if f),
            "gated_mode": sc.get("gated_steering_mode", "off"),
            "arm": args.arm,
            "slug": args.slug,
            "beta": steer.get("beta"),
            "op": steer.get("op"),
            "serve_gpu": sc.get("serve_gpu"),
            "run_tag": sc.get("run_tag"),
        })
        if int(sc.get("episode_success", -1)) not in (-1, int(m.group("succ"))):
            raise SystemExit(f"{jp}: 스템 succ 와 episode_success 불일치")
    rows.sort(key=lambda r: r["ep"])

    out_path = args.out or (args.job_dir / "per_episode.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = ["\t".join(COLUMNS)]
    body += ["\t".join(cell(r[c]) for c in COLUMNS) for r in rows]
    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")

    n = len(rows)
    s = sum(r["success"] for r in rows)
    fired = [r for r in rows if r["trigger_step"] not in (None, "NA")]
    sr = f"{s / n:.3f}" if n else "NA"
    print(f"[results] {args.slug}/{args.arm}: n={n} succ={s} SR={sr} "
          f"fired={len(fired)}/{n} → {out_path}", flush=True)
    if cells and rows:
        for ln in quadrant_report(rows):
            print(ln, flush=True)
    if args.expect and n < args.expect:
        print(f"[results] INCOMPLETE — {n} < expect {args.expect}", file=sys.stderr)
        return 13
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
