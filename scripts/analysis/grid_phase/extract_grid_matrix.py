#!/usr/bin/env python3
"""grid 좌표 트리 rollout.pkl → instruction 당 1개 shard NPZ (스트리밍 축약).

승준 원격 노드(`~/anaconda3/bin/python`, torch+numpy 만 있고 scipy/sklearn 없음)에서
`raw` 를 옮기지 않고 도는 추출기다. 판당 ~405MB 인 full-token pkl(record 마다
`[L=7,K=4,T=49,D=1536]`)을 **record 단위로 읽어 즉시 토큰축을 축약**하고 원본을 버린다 —
`[n,L,K,T,D]` 를 통째로 올리면 판 하나가 수 GB 라 노드가 죽는다.

산출 (다른 분석 스크립트와 공유하는 계약이라 열 이름·축 순서를 바꾸지 말 것):

  Tier A  `<out>/segA/<instr_slug>.npz`
      X   fp16 [n_rec, 7, 4, 4, 1536]   축 = (record, layer, denoise K, segment, dim)
          segment = (state, future, action, all) — 각 토큰 세그먼트의 토큰축 mean
  Tier B  `<out>/tokB/<instr_slug>__L<layers>.npz`
      X   fp16 [n_rec, n_layers, 49, 1536]  — K 축 mean pool, 토큰 49 보존

두 tier 모두 record 별 열(`ep_id/scene/noise/jitter/rec_idx/succ/phase_code/ep_len`)과
`vl`, `meta_json` 을 함께 담는다. `jitter` 는 지터 축(k = reset_idx) 좌표로 —
3축 좌표 `s<i>/k<r>/n<j>` 의 셋째 축, docs/04 §3.1.1 — **항상 존재하는 열**이다.
인덱스에 `jitter_reset_idx` 열이 없는 legacy(2축) 추출에서는 전부 `-1` 로 채운다
("지터 축 없음"). 지터 축 인덱스 안의 legacy(빈 값) 행은 99. `meta_json.jitter_axis` 가 이 열이 실좌표
(true)인지 -1 채움(false)인지 알려 준다. `meta_json` 에는 **절대경로를 쓰지 않는다** (docs/04 §8) —
출처는 `sigs`(meta.json 의 내용 지문)로만 남긴다.

사용:
    python3 scripts/analysis/grid_phase/extract_grid_matrix.py \
        --grid-root  .../groot/n15/grid \
        --index-tsv  .../groot/n15/index/rollouts.tsv \
        --out-dir    .../analysis/grid_phase \
        --instructions all --tier segA --workers 4
"""

from __future__ import annotations

import os

# numpy/torch import 전에 스레드 cap — 공유 20코어 노드에서 판당 BLAS 폭주 방지.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# repo root 를 __file__ 기준으로 — worktree/원격 checkout 어디서 돌려도 같게 잡힌다.
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ── 계약 상수 (fail-loud 검사 기준) ──────────────────────────────────────────
EXPECT_LAYERS = [0, 2, 4, 8, 10, 12, 15]
EXPECT_K = 4
EXPECT_T = 49
EXPECT_D = 1536
N_FUTURE = 32                      # state 1 + future 32 + action 16 = 49
SEGMENT_NAMES = ["state", "future", "action", "all"]
BASE_ARM = "base"
JITTER_BASE = 99          # index 의 jitter_reset_idx == "base" (지터 없는 원본 판)
JITTER_NONE = -1          # 지터 축 자체가 없는 인덱스(v1/v2) 의 채움값


def token_segments(T: int, action_horizon: int, n_future: int = N_FUTURE):
    """fit/fit_phase_conceptor.py `_token_segments` 와 같은 규약 (state/future/action)."""
    if 1 + n_future + action_horizon != T:
        raise ValueError(
            f"token segment 가정(1+{n_future}+{action_horizon}) != T={T}")
    return {"state": (0, 1), "future": (1, 1 + n_future),
            "action": (T - action_horizon, T)}


def instr_slug(instruction: str) -> str:
    """`PPCC/bread` 처럼 '/' 를 품은 instruction → 파일명 안전 slug."""
    out = instruction.strip()
    for ch in ("/", " ", "\\", "\t"):
        out = out.replace(ch, "_")
    return out


# ── 인덱스 ────────────────────────────────────────────────────────────────
class Episode:
    """index tsv 한 행 = base 셀 하나 (좌표 + 신원). pkl 은 아직 열지 않는다."""

    __slots__ = ("plan_id", "machine", "instruction", "scene", "noise",
                 "jitter", "success", "rel_path", "sig")

    def __init__(self, plan_id, machine, instruction, scene, noise,
                 success, rel_path, sig, jitter=JITTER_NONE):
        self.plan_id = plan_id
        self.machine = machine
        self.instruction = instruction
        self.scene = scene
        self.noise = noise
        # 지터 축 좌표 k (docs/04 §3.1.1). "base"→99, 지터 축 없는 인덱스→-1.
        self.jitter = jitter
        self.success = success
        self.rel_path = rel_path
        self.sig = sig

    def key(self):
        """정렬·중복 판정의 정본 키.

        **이 순서가 shard 의 `ep_id` 를 정한다** — ep_id = 이 키로 오름차순 정렬한
        판 목록의 0-based 인덱스. v1/v2(지터 축 없음)에서는 jitter 가 전부 -1 이라
        (plan, machine, scene, noise) 정렬과 완전히 같아 ep_id 가 바뀌지 않는다.
        """
        return (self.plan_id, self.machine, self.scene, self.noise, self.jitter)

    def cell(self):
        """머신을 뺀 좌표 — union 가능성(머신 간 셀 서로소) 판정용."""
        return (self.plan_id, self.scene, self.noise, self.jitter)


def _cells_disjoint(per: dict[str, list[Episode]]) -> bool:
    """머신별 셀 집합이 pairwise 서로소인가 (= 중복 좌표가 하나도 없는가)."""
    total = 0
    seen: set[tuple] = set()
    for eps in per.values():
        cells = {e.cell() for e in eps}
        if len(cells) != len(eps):
            return False          # 같은 머신 안에서 이미 중복 — union 대상 아님
        total += len(cells)
        seen |= cells
    return len(seen) == total


def select_machine(instruction: str, ep_list: list[Episode],
                   override: dict[str, str]) -> tuple[str, list[Episode], dict]:
    """instruction 하나의 판 목록 → (선택 머신, 그 머신의 판, 선택 기록).

    같은 좌표 `(instruction, s, n, k)` 가 두 머신에 있는 경우가 실제로 있다 (정본 수집 +
    다른 머신의 부산물). 머신을 섞으면 §3.2 층화 위험이라 원칙은 **instruction 당 머신
    하나**. 규칙 = 셀 수가 가장 많은(= 완결된) 머신. 동수면 자동 선택이 근거 없는 추측이
    되므로 fail-loud — `--machine <slug>=<machine>` 으로 사람이 정한다.

    예외 `union_disjoint`: 머신들의 **셀 집합 `(plan, scene, noise, jitter)` 이 서로소**
    면 union 한다. v4 지터 인덱스의 `PPCC/bread` 가 이 경우다 — 지터 100 판은 kanu,
    base 25 판은 worker1 이라 좌표가 하나도 겹치지 않는다. 겹치는 좌표가 없으면 "어느
    판이 정본인가" 라는 모호함 자체가 없으므로 버릴 이유가 없다 (버리면 base arm 전체가
    사라진다). 하나라도 겹치면 기존 규칙(다수 머신 채택 + drop 로그)으로 되돌아간다.
    """
    per: dict[str, list[Episode]] = {}
    for e in ep_list:
        per.setdefault(e.machine, []).append(e)
    counts = {m: len(v) for m, v in per.items()}

    slug = instr_slug(instruction)
    forced = override.get(slug) or override.get(instruction)
    if forced is not None:
        if forced not in per:
            raise ValueError(
                f"{instruction}: --machine 이 지정한 {forced!r} 이 인덱스에 없다 "
                f"(있는 머신: {sorted(per)})")
        chosen, how = forced, "override"
    elif len(per) == 1:
        chosen, how = next(iter(per)), "only"
    elif _cells_disjoint(per):
        kept_all = sorted(ep_list, key=lambda e: e.key())
        machines = sorted(per)
        info = {"machine": machines, "rule": "union_disjoint",
                "cells_per_machine": counts, "dropped_dup_cells": 0,
                "dropped_machines": []}
        return "+".join(machines), kept_all, info
    else:
        top = max(counts.values())
        tied = sorted(m for m, c in counts.items() if c == top)
        if len(tied) > 1:
            raise ValueError(
                f"{instruction}: 머신 {tied} 이 셀 수 {top} 로 동수 — 어느 쪽이 정본인지 "
                f"추측할 수 없다. --machine {slug}=<machine> 으로 지정할 것 "
                f"(분포: {counts})")
        chosen, how = tied[0], "max_cells"

    kept = per[chosen]
    info = {"machine": chosen, "rule": how, "cells_per_machine": counts,
            "dropped_dup_cells": sum(c for m, c in counts.items() if m != chosen),
            "dropped_machines": sorted(m for m in counts if m != chosen)}
    return chosen, kept, info


def _int_or_none(s: str):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _jitter_of(raw: str) -> int:
    """index 의 `jitter_reset_idx` 문자열 → 정수 좌표.

    `"base"`(지터 없는 원본 판) → 99, 그 외 정수 문자열 → 그 값. 빈 칸은 base 취급
    (지터 축 인덱스에서 값이 비면 base 재사용 행이다).
    """
    v = (raw or "").strip()
    if v == "" or v.lower() == BASE_ARM:
        return JITTER_BASE
    try:
        return int(v)
    except ValueError:
        raise ValueError(
            f"jitter_reset_idx={raw!r} — 정수 또는 {BASE_ARM!r} 이어야 한다") from None


def read_index(index_tsv: Path, grid_root: Path) -> tuple[list[Episode], bool]:
    """rollouts.tsv → (Episode 목록, 지터 축 유무).

    열은 `scripts/collect/build_grid_index.py` 의 ROLLOUT_COLS 를 헤더로 읽는다
    (하드코딩 인덱스 금지 — 열이 늘어나도 깨지지 않게). base arm + pkl 보유 행만 남긴다.
    `rel_path` 가 없는 옛 인덱스면 좌표로 경로를 조립한다.

    `jitter_reset_idx` 열이 있으면(v3/v4 지터 인덱스) 지터 축 k 를 **정식 좌표로**
    싣는다 — 같은 `(scene, noise)` 셀에 지터 변형이 여러 판 있으므로 이게 없으면
    중복 좌표로 오인된다. 열이 없으면(v1/v2) 전부 -1 이라 이전 동작과 완전히 같다.
    """
    lines = index_tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{index_tsv}: 비어 있음")
    header = lines[0].split("\t")
    need = {"plan_id", "machine", "grid_instruction", "scene_idx", "noise_idx"}
    missing = need - set(header)
    if missing:
        raise ValueError(
            f"{index_tsv}: 필수 열 없음 {sorted(missing)} — build_grid_index.py 로 "
            "만든 rollouts.tsv 인지 확인할 것 (헤더: {})".format(",".join(header[:8])))
    col = {c: i for i, c in enumerate(header)}
    has_jitter = "jitter_reset_idx" in col

    def get(parts, name, default=""):
        i = col.get(name)
        return parts[i] if (i is not None and i < len(parts)) else default

    eps: list[Episode] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        armsig = get(parts, "armsig", BASE_ARM)
        has_pkl = get(parts, "has_pkl", "1")
        if armsig != BASE_ARM:
            continue
        if has_pkl.strip() in ("0", "false", "False"):
            continue
        scene, noise = _int_or_none(get(parts, "scene_idx")), _int_or_none(get(parts, "noise_idx"))
        if scene is None or noise is None:
            continue
        instruction = get(parts, "grid_instruction").strip()
        plan_id = get(parts, "plan_id").strip()
        machine = get(parts, "machine").strip()
        jitter = _jitter_of(get(parts, "jitter_reset_idx")) if has_jitter \
            else JITTER_NONE
        rel = get(parts, "rel_path").strip()
        if not rel:
            # 3축 폴더층 s<i>/k<r>/n<j> (docs/04 §3.1.1). 지터 축이 없거나 legacy 행이면
            # k 층 없이 s<i>/n<j> (구 레이아웃).
            k_seg = "" if jitter in (JITTER_NONE, JITTER_BASE) else f"k{jitter}/"
            rel = (f"{plan_id}/{machine}/{instruction}/s{scene}/{k_seg}"
                   f"n{noise}/{BASE_ARM}")
        eps.append(Episode(plan_id, machine, instruction, scene, noise,
                           _int_or_none(get(parts, "success")), rel,
                           get(parts, "sig").strip(), jitter))
    if not eps:
        raise ValueError(f"{index_tsv}: base+pkl 행이 하나도 없다")
    _ = grid_root  # 경로 검증은 워커에서 (인덱싱 시점엔 파일 접근하지 않는다)
    return eps, has_jitter


# ── 판 하나 처리 (워커) ────────────────────────────────────────────────────
def _as_f32(x) -> np.ndarray:
    """torch fp16 텐서 / ndarray / list → float32 ndarray (numpy 로만 계산)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if hasattr(x, "detach"):           # torch.Tensor
        x = x.detach().cpu()
        if x.dtype.__str__().endswith("float16"):
            x = x.float()
        return x.numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def check_contract(d: dict, rel_path: str, index_success: int | None) -> None:
    layers = d.get("capture_layers")
    if list(layers or []) != EXPECT_LAYERS:
        raise ValueError(f"{rel_path}: capture_layers={layers} != {EXPECT_LAYERS}")
    hs = d.get("hidden_states")
    if not hs:
        raise ValueError(f"{rel_path}: hidden_states 없음/비어 있음")
    ph = d.get("feature_phases")
    if ph is None:
        raise ValueError(f"{rel_path}: feature_phases 없음 (GT phase 라벨 필수)")
    if len(ph) != len(hs):
        raise ValueError(
            f"{rel_path}: feature_phases {len(ph)} != hidden_states {len(hs)}")
    succ = d.get("episode_success")
    if succ is None:
        raise ValueError(f"{rel_path}: episode_success 없음")
    if index_success is not None and int(succ) != int(index_success):
        raise ValueError(
            f"{rel_path}: index success={index_success} != pkl episode_success={succ} "
            "— 인덱스와 pkl 이 다른 판을 가리킨다")


def process_episode(job: dict) -> dict:
    """pkl 한 장 → 축약 배열 dict. 예외는 호출부(--skip-bad)가 처리."""
    rel_path = job["rel_path"]
    pkl = Path(job["grid_root"]) / rel_path / "rollout.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"{rel_path}: rollout.pkl 없음")
    with open(pkl, "rb") as f:
        d = pickle.load(f)          # torch.load 아님 — pickle.dump 산출물 (artifacts.py)

    check_contract(d, rel_path, job["index_success"])
    hs = d["hidden_states"]
    phases = [str(p) for p in d["feature_phases"]]
    n_rec = len(hs)
    horizon = int(d.get("model_action_horizon") or 16)
    tier, layers_sel = job["tier"], job["layers_sel"]

    if tier == "segA":
        X = np.zeros((n_rec, len(EXPECT_LAYERS), EXPECT_K, len(SEGMENT_NAMES),
                      EXPECT_D), dtype=np.float16)
    else:
        X = np.zeros((n_rec, len(layers_sel), EXPECT_T, EXPECT_D), dtype=np.float16)

    segs = None
    for i in range(n_rec):
        a = _as_f32(hs[i])
        hs[i] = None                  # 원본 record 즉시 버린다 (피크 억제)
        if a.ndim != 4:
            raise ValueError(f"{rel_path}: record{i} shape {a.shape} — [L,K,T,D] 기대")
        L, K, T, D = a.shape
        if [L, K, T, D] != [len(EXPECT_LAYERS), EXPECT_K, EXPECT_T, EXPECT_D]:
            raise ValueError(
                f"{rel_path}: record{i} shape {a.shape} != "
                f"[{len(EXPECT_LAYERS)},{EXPECT_K},{EXPECT_T},{EXPECT_D}]")
        if segs is None:
            segs = token_segments(T, horizon)
        if tier == "segA":
            cols = [a[:, :, lo:hi, :].mean(axis=2) for lo, hi in
                    (segs["state"], segs["future"], segs["action"])]
            cols.append(a.mean(axis=2))                    # "all" = 49 토큰 전체
            X[i] = np.stack(cols, axis=2).astype(np.float16)
        else:
            X[i] = a[layers_sel].mean(axis=1).astype(np.float16)   # K 축 pool
        del a

    vl_raw = d.get("vl_hidden_states")
    vl = np.zeros((n_rec, 2048), dtype=np.float16)
    has_vl = bool(vl_raw)
    if has_vl:
        if len(vl_raw) != n_rec:
            raise ValueError(
                f"{rel_path}: vl_hidden_states {len(vl_raw)} != records {n_rec}")
        for i, v in enumerate(vl_raw):
            va = _as_f32(v)
            va = va.mean(axis=0) if va.ndim == 2 else va       # [T_vl,Dvl] → [Dvl]
            if va.shape[-1] != vl.shape[1]:
                raise ValueError(f"{rel_path}: vl dim {va.shape[-1]} != {vl.shape[1]}")
            vl[i] = va.astype(np.float16)

    return {
        "rel_path": rel_path,
        "X": X,
        "vl": vl,
        "has_vl": np.full(n_rec, has_vl, dtype=bool),
        "phases": phases,
        "n_rec": n_rec,
        "scene": job["scene"],
        "noise": job["noise"],
        "jitter": job.get("jitter", JITTER_NONE),
        "succ": int(d["episode_success"]),
        "sig": job["sig"] or str(d.get("sig") or ""),
        "plan_id": job["plan_id"],
        "machine": job["machine"],
    }


def _safe_process(job: dict) -> dict:
    try:
        return process_episode(job)
    except Exception as exc:  # noqa: BLE001 — 사유를 담아 상위로 (--skip-bad 판정)
        return {"error": f"{type(exc).__name__}: {exc}", "rel_path": job["rel_path"]}


def _init_worker() -> None:
    try:
        import torch  # noqa: PLC0415
        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001 — numpy 만 있는 판이면 그냥 넘어간다
        pass


# ── shard 조립 ────────────────────────────────────────────────────────────
def build_shard(results: list[dict], instruction: str, tier: str,
                layers_sel: list[int], layer_ids: list[int],
                extra_meta: dict | None = None,
                jitter_axis: bool = False) -> dict[str, Any]:
    """판별 결과 → NPZ 배열 묶음. record 축으로만 concat (rollout pooling 없음).

    `ep_id` 규칙: `results` 는 호출 전에 `Episode.key()`
    = `(plan_id, machine, scene, noise, jitter)` 오름차순으로 정렬돼 있고, ep_id 는 그
    목록의 0-based 인덱스다. 지터 축이 키에 들어가므로 v4 처럼 같은 `(scene, noise)` 에
    변형이 여러 판 있어도 판마다 ep_id 가 유일하고 순서가 결정적이다
    (`rewrite_shard_clusters` 가 ep_id/rec_idx 를 원소 단위로 대조한다).

    `jitter` 열은 에피소드 좌표를 record 축으로 브로드캐스트한 값이다 — 지터 축이 없는
    v1/v2 인덱스에서는 열을 그대로 두되 전부 -1 로 채우고 `meta_json.jitter_axis=false`
    로 표시한다 (다운스트림은 열을 무시해도 무방).
    """
    codebook: dict[str, int] = {}
    for lab in sorted({p for r in results for p in r["phases"]}):
        codebook[lab] = len(codebook)

    Xs, vls, has_vls = [], [], []
    ep_id, scene, noise, rec_idx, succ, phase_code, ep_len = [], [], [], [], [], [], []
    jitter: list[np.ndarray] = []
    sigs = []
    for e, r in enumerate(results):
        n = r["n_rec"]
        Xs.append(r["X"])
        vls.append(r["vl"])
        has_vls.append(r["has_vl"])
        ep_id.append(np.full(n, e, dtype=np.int32))
        scene.append(np.full(n, r["scene"], dtype=np.int16))
        noise.append(np.full(n, r["noise"], dtype=np.int16))
        jitter.append(np.full(n, r.get("jitter", JITTER_NONE), dtype=np.int16))
        rec_idx.append(np.arange(n, dtype=np.int16))
        succ.append(np.full(n, r["succ"], dtype=np.int8))
        phase_code.append(np.array([codebook[p] for p in r["phases"]], dtype=np.int16))
        ep_len.append(np.full(n, n, dtype=np.int16))
        sigs.append(r["sig"])

    machines = sorted({r["machine"] for r in results})
    meta = {
        "instruction": instruction,
        "plan_id": sorted({r["plan_id"] for r in results}),
        # instruction 당 머신 하나 (select_machine) — 리스트가 아니라 선택된 값.
        "machine": machines[0] if len(machines) == 1 else machines,
        "capture_layers": EXPECT_LAYERS,
        "segment_names": SEGMENT_NAMES,
        "phase_codebook": codebook,
        "n_episodes": len(results),
        "denoise_k": EXPECT_K,
        "dim": EXPECT_D,
        "tier": tier,
        # 지터 축 실좌표인가(true) / -1 채움인가(false) — jitter 열 해석의 단일 근거.
        "jitter_axis": bool(jitter_axis),
        "sigs": sigs,          # 출처는 sig 로만 — 절대경로 기록 금지 (docs/04 §8)
    }
    if tier == "tokB":
        meta["layers"] = layer_ids
        meta["n_tokens"] = EXPECT_T
        meta["denoise_pool"] = "mean"
    if extra_meta:
        meta.update(extra_meta)

    arrays = {
        "X": np.concatenate(Xs, axis=0),
        "vl": np.concatenate(vls, axis=0),
        "has_vl": np.concatenate(has_vls, axis=0),
        "ep_id": np.concatenate(ep_id),
        "scene": np.concatenate(scene),
        "noise": np.concatenate(noise),
        "jitter": np.concatenate(jitter),
        "rec_idx": np.concatenate(rec_idx),
        "succ": np.concatenate(succ),
        "phase_code": np.concatenate(phase_code),
        "ep_len": np.concatenate(ep_len),
        "meta_json": np.array(json.dumps(meta, ensure_ascii=False)),
    }
    _ = layers_sel
    return arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True, type=Path)
    ap.add_argument("--index-tsv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--instructions", default="all",
                    help="all | 콤마 목록 (예: 'OpenDrawer,PPCC/bread')")
    ap.add_argument("--tier", choices=("segA", "tokB"), default="segA")
    ap.add_argument("--layers", default="12",
                    help="tokB 용 capture_layers 값 콤마 목록 (예: 12 또는 2,12)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-eps", type=int, default=0,
                    help="smoke: instruction 당 N 판만 (0 = 전부)")
    ap.add_argument("--plan-ids", default="all",
                    help="all | 콤마 목록 (기본: 인덱스에 있는 plan 전부)")
    ap.add_argument("--machine", action="append", default=[],
                    metavar="INSTR_SLUG=MACHINE",
                    help="instruction 별 머신 강제 (중복 좌표 자동선택 override). "
                         "예: --machine PPCC_marshmallow=dongkyu-MS-7D43")
    ap.add_argument("--skip-bad", action="store_true",
                    help="계약 위반 판을 예외 대신 사유 기록 후 건너뛴다")
    args = ap.parse_args()

    layer_ids = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
    if args.tier == "tokB":
        bad = [l for l in layer_ids if l not in EXPECT_LAYERS]
        if bad:
            print(f"ERROR: --layers {bad} 는 capture_layers {EXPECT_LAYERS} 에 없다",
                  file=sys.stderr)
            return 2
        layers_sel = [EXPECT_LAYERS.index(l) for l in layer_ids]
    else:
        layers_sel = list(range(len(EXPECT_LAYERS)))

    eps, jitter_axis = read_index(args.index_tsv, args.grid_root)
    if jitter_axis:
        print("[extract] index 에 jitter_reset_idx 열 있음 — 지터 축 k 를 정식 좌표로 "
              "쓴다 (base=99)", flush=True)
    if args.plan_ids != "all":
        want = {p.strip() for p in args.plan_ids.split(",") if p.strip()}
        eps = [e for e in eps if e.plan_id in want]
    if args.instructions != "all":
        want_i = {i.strip() for i in args.instructions.split(",") if i.strip()}
        eps = [e for e in eps if e.instruction in want_i]
    if not eps:
        print("ERROR: 조건에 맞는 판이 없다 (--instructions/--plan-ids 확인)",
              file=sys.stderr)
        return 2

    override: dict[str, str] = {}
    for spec in args.machine:
        if "=" not in spec:
            print(f"ERROR: --machine {spec!r} 형식은 <instr_slug>=<machine>",
                  file=sys.stderr)
            return 2
        k, v = spec.split("=", 1)
        override[k.strip()] = v.strip()

    by_instr: dict[str, list[Episode]] = {}
    for e in eps:
        by_instr.setdefault(e.instruction, []).append(e)

    # 중복 좌표 대응 — instruction 당 머신 하나 (동수면 여기서 fail-loud).
    machine_pick: dict[str, dict] = {}
    for instruction in sorted(by_instr):
        _, kept, info = select_machine(instruction, by_instr[instruction], override)
        by_instr[instruction] = sorted(kept, key=lambda e: e.key())
        machine_pick[instruction] = info
        if info["dropped_dup_cells"]:
            print(f"[machine] {instruction}: {info['machine']} 선택 ({info['rule']}), "
                  f"중복 {info['dropped_dup_cells']} 셀 제외 "
                  f"{info['dropped_machines']} — 분포 {info['cells_per_machine']}",
                  flush=True)
        elif info["rule"] == "union_disjoint":
            print(f"[machine] {instruction}: {info['machine']} union "
                  f"(셀 서로소, 제외 0) — 분포 {info['cells_per_machine']}", flush=True)
    # 같은 머신 안에서도 좌표가 겹치면 인덱스가 깨진 것 — 조용히 중복 집계하지 않는다.
    for instruction, v in by_instr.items():
        seen: dict[tuple, str] = {}
        for e in v:
            # 지터 축 포함 (v1/v2 는 jitter 가 상수 -1 이라 이전 키와 동치).
            k = (e.plan_id, e.machine, e.scene, e.noise, e.jitter)
            if k in seen:
                raise ValueError(
                    f"{instruction}: 좌표 중복 {k} — {seen[k]} 와 {e.rel_path}")
            seen[k] = e.rel_path

    tier_dir = args.out_dir / args.tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "tier": args.tier, "n_instructions": len(by_instr),
        "layers": layer_ids if args.tier == "tokB" else EXPECT_LAYERS,
        "instructions": {},
    }
    t0 = time.time()
    print(f"[extract] tier={args.tier} instructions={len(by_instr)} "
          f"episodes={len(eps)} workers={args.workers}", flush=True)

    for instruction in sorted(by_instr):
        ep_list = by_instr[instruction]
        if args.limit_eps:
            ep_list = ep_list[:args.limit_eps]
        slug = instr_slug(instruction)
        pick = machine_pick[instruction]
        # success 결측 행 — pkl 의 episode_success 를 정본으로 쓰고 대조는 건너뛴다.
        n_succ_missing = sum(1 for e in ep_list if e.success is None)
        print(f"[extract] {instruction}: {len(ep_list)} eps (machine={pick['machine']}, "
              f"success결측 {n_succ_missing}) → {slug}", flush=True)

        jobs = [{
            "grid_root": str(args.grid_root), "rel_path": e.rel_path,
            "index_success": e.success, "scene": e.scene, "noise": e.noise,
            "jitter": e.jitter,
            "sig": e.sig, "plan_id": e.plan_id, "machine": e.machine,
            "tier": args.tier, "layers_sel": layers_sel,
        } for e in ep_list]

        results: list[dict] = []
        skipped: list[dict] = []
        ti = time.time()

        def _handle(res: dict) -> None:
            if "error" in res:
                msg = f"{res['rel_path']}: {res['error']}"
                if not args.skip_bad:
                    raise RuntimeError(msg)
                print(f"[skip] {msg}", file=sys.stderr, flush=True)
                skipped.append({"rel_path": res["rel_path"], "reason": res["error"]})
            else:
                results.append(res)
            done = len(results) + len(skipped)
            if done % 10 == 0 or done == len(jobs):
                print(f"[extract]   {instruction} {done}/{len(jobs)} "
                      f"({time.time() - ti:.0f}s)", flush=True)

        if args.workers > 1:
            ctx = mp.get_context("spawn")   # fork 시 부모의 큰 배열이 워커에 딸려온다
            with ctx.Pool(args.workers, initializer=_init_worker) as pool:
                for res in pool.imap_unordered(_safe_process, jobs, chunksize=1):
                    _handle(res)
            # Episode.key() 와 **같은 순서** — ep_id 가 이 정렬의 인덱스라 어긋나면 안 된다.
            results.sort(key=lambda r: (r["plan_id"], r["machine"], r["scene"],
                                        r["noise"], r.get("jitter", JITTER_NONE)))
        else:
            _init_worker()
            for job in jobs:
                _handle(_safe_process(job))

        if not results:
            print(f"[extract] {instruction}: 유효 판 0 — shard 쓰지 않음", flush=True)
            summary["instructions"][instruction] = {
                "n_eps": 0, "n_rec": 0, "n_succ": 0,
                "machine": pick["machine"], "machine_rule": pick["rule"],
                "cells_per_machine": pick["cells_per_machine"],
                "dropped_dup_cells": pick["dropped_dup_cells"],
                "dropped_machines": pick["dropped_machines"],
                "n_success_missing_in_index": n_succ_missing,
                "skipped": skipped}
            continue

        arrays = build_shard(results, instruction, args.tier, layers_sel, layer_ids,
                             extra_meta={"machine_rule": pick["rule"],
                                         "dropped_dup_cells": pick["dropped_dup_cells"]},
                             jitter_axis=jitter_axis)
        name = slug if args.tier == "segA" else \
            f"{slug}__L{'-'.join(str(l) for l in layer_ids)}"
        out_path = tier_dir / f"{name}.npz"
        tmp = out_path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        tmp.replace(out_path)

        n_rec = int(arrays["X"].shape[0])
        n_succ = int(sum(1 for r in results if r["succ"] == 1))
        summary["instructions"][instruction] = {
            "n_eps": len(results), "n_rec": n_rec, "n_succ": n_succ,
            "machine": pick["machine"], "machine_rule": pick["rule"],
            "cells_per_machine": pick["cells_per_machine"],
            "dropped_dup_cells": pick["dropped_dup_cells"],
            "dropped_machines": pick["dropped_machines"],
            "n_success_missing_in_index": n_succ_missing,
            "shard": out_path.name, "X_shape": list(arrays["X"].shape),
            "skipped": skipped,
        }
        print(f"[extract] {instruction}: 판 {len(results)} (succ {n_succ}) "
              f"record {n_rec} → {out_path.name} "
              f"{out_path.stat().st_size / 1e9:.2f}GB "
              f"({time.time() - ti:.0f}s)", flush=True)

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (args.out_dir / f"{args.tier}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    n_skip = sum(len(v["skipped"]) for v in summary["instructions"].values())
    print(f"[extract] 완료 {summary['elapsed_sec']}s — skip {n_skip} "
          f"({args.out_dir / (args.tier + '_summary.json')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
