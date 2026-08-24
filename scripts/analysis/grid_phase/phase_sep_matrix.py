#!/usr/bin/env python
"""grid phase separation — layer × token-segment 해상도의 phase별 succ/fail 분리도.

새 N1.5 grid 수집(9 task × scene 10 × noise 10)에서 **phase 조건부** succ/fail 분리도를
capture layer × denoise step × token segment 격자로 잰다. 질문은 "어느 layer/segment 에서,
어느 phase 에, succ/fail 이 갈리는가" 하나다.

## 분석 단위 = (episode, phase) 1행 — rollout 전체 pooling 금지

1. record 별 phase 라벨 부여 (`--phase-def`).
2. (task, phase) 별 **동일예산** B = 그 phase 에 도달한 episode 들의 phase 내 record 수의 최소값.
   B < `--min-budget` 이면 그 phase 는 skip(사유 기록).
3. episode 마다 그 phase 의 **첫 B 개 record** 만 mean-pool → 1행 [D].
   (`equal_budget_pool` 로직 출처: scripts/safe/groot_n15/robocasa/analyze/phase_separation.py:80.
    phase 내 길이차가 라벨을 대신 결정하는 것을 막는다 — 실패=timeout 이라 record 수 자체가
    라벨과 상관된다.)
4. LOSO(fold 축 = `scene`, held-out scene 내부 중심화) → AUROC, + scene **내부** 라벨 순열 null z.
5. 병기 열 `length_auroc` = "그 phase 의 record 수 자체"를 점수로 쓴 AUROC (길이 confound 자가감사).
   이 값이 본 AUROC 와 비슷하면 그 cell 은 길이로 설명된다.

판정 코어(`auroc`/`within_dir`/`loso`/`perm_null`)는 새로 쓰지 않고
`scripts/scene_sae/g2_residual_read.py` 에서 **파일경로 import** 로 그대로 가져온다
(scripts 는 패키지가 아니라 importlib 로 로드). 통계량 혼용 금지 — 이 파일에서 나오는
AUROC 는 전부 그 `loso()` 하나다.

## 입력 NPZ 계약
`<shards-dir>/<instr_slug>.npz`
  - Tier A: `X` fp16 [n_rec, n_layer, K=4, S=4, D] — (record, capture_layers 순, denoise, segment)
  - Tier B: `X` fp16 [n_rec, n_layer, 49, D] (token 해상도, `tokB/<slug>__L*.npz`)
  - `vl` fp16 [n_rec, 2048], `has_vl` bool
  - `ep_id` i32, `scene` i16, `noise` i16, `rec_idx` i16, `succ` i8, `phase_code` i16, `ep_len` i16
  - `meta_json`: {instruction, capture_layers|layers, segment_names, phase_codebook{label:code}, ...}

## 사용 예
    python phase_sep_matrix.py --shards runs/gridA --layers all --segs state,future,action,all \\
        --denoise mean --vl --phase-def gt --n-perm 100 \\
        --out out/gridA.json --tsv out/gridA.tsv
"""
from __future__ import annotations

import os

# numpy import 전에 cap (공유 노드 — CLAUDE.md CPU 예산).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_F = Path(__file__).resolve()
REPO = _F.parents[3]          # scripts/analysis/grid_phase/<this> → repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ───────────────────────────────────────────── 판정 코어 (g2_residual_read 재사용)
def _load_g2():
    """scripts 는 패키지가 아니므로 파일경로로 로드한다."""
    p = REPO / "scripts" / "scene_sae" / "g2_residual_read.py"
    if not p.exists():
        raise FileNotFoundError(f"판정 코어 원본이 없다: {p}")
    spec = importlib.util.spec_from_file_location("_g2_residual_read", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_G2 = _load_g2()
auroc = _G2.auroc                # auroc(s, y) -> float
within_dir = _G2.within_dir      # within_dir(V, y, sc) -> [D] | None
loso = _G2.loso                  # loso(V, y, sc, Vfolds=None) -> (auroc, n_eval)
perm_null = _G2.perm_null        # perm_null(V, y, sc, n_perm, rng, Vfolds=None) -> [n_perm]

MIXED_SCENE_MIN = 4              # 혼재 scene 이 이보다 적으면 exploratory 플래그


# ──────────────────────────────────────────────────────────────────── shard 로딩
class Shard:
    """한 instruction 의 NPZ 하나. X 는 fp16 그대로 두고 슬라이스 시점에 float32 캐스팅."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.slug = self.path.stem
        z = np.load(self.path, allow_pickle=False)
        self.X = z["X"]
        self.ep_id = np.asarray(z["ep_id"]).astype(np.int64).ravel()
        self.scene = np.asarray(z["scene"]).astype(np.int64).ravel()
        self.noise = np.asarray(z["noise"]).astype(np.int64).ravel()
        self.rec_idx = np.asarray(z["rec_idx"]).astype(np.int64).ravel()
        self.succ = np.asarray(z["succ"]).astype(np.int64).ravel()
        self.phase_code = np.asarray(z["phase_code"]).astype(np.int64).ravel()
        self.ep_len = np.asarray(z["ep_len"]).astype(np.int64).ravel() if "ep_len" in z else None
        self.vl = z["vl"] if "vl" in z.files else None
        self.has_vl = bool(np.asarray(z["has_vl"]).ravel()[0]) if "has_vl" in z.files else False
        meta = z["meta_json"]
        self.meta = json.loads(meta.item() if hasattr(meta, "item") else str(meta))
        self.instruction = str(self.meta.get("instruction", self.slug))

        if self.X.ndim == 5:
            self.tier = "A"
            self.n_layer, self.n_denoise, self.n_seg, self.dim = self.X.shape[1:]
            self.n_token = None
        elif self.X.ndim == 4:
            self.tier = "B"
            self.n_layer, self.n_token, self.dim = self.X.shape[1:]
            self.n_denoise = self.n_seg = None
        else:
            raise ValueError(f"{self.path}: X.ndim={self.X.ndim} (5=TierA, 4=TierB 만 지원)")

        # Tier B(tokB) 는 부분 layer 만 담아 meta["layers"] 가 실제 layer 축이고,
        # meta["capture_layers"] 는 원본 수집 규약(7층)이라 축 검증에 쓰면 안 된다.
        if self.tier == "B":
            cl = self.meta.get("layers") or self.meta.get("capture_layers")
        else:
            cl = self.meta.get("capture_layers") or self.meta.get("layers")
        if cl is None:
            raise ValueError(f"{self.path}: meta 에 capture_layers/layers 가 없다")
        self.capture_layers = [int(v) for v in cl]
        if len(self.capture_layers) != self.n_layer:
            raise ValueError(
                f"{self.path}: capture_layers {len(self.capture_layers)} != X layer축 {self.n_layer}")
        self.segment_names = [str(s) for s in (self.meta.get("segment_names") or [])]
        if self.tier == "A" and len(self.segment_names) != self.n_seg:
            self.segment_names = ["state", "future", "action", "all"][: self.n_seg]
        self.phase_codebook = {str(k): int(v) for k, v in
                               (self.meta.get("phase_codebook") or {}).items()}
        self.n_rec = int(self.X.shape[0])
        for name, arr in (("ep_id", self.ep_id), ("scene", self.scene), ("rec_idx", self.rec_idx),
                          ("succ", self.succ), ("phase_code", self.phase_code)):
            if len(arr) != self.n_rec:
                raise ValueError(f"{self.path}: {name} 길이 {len(arr)} != n_rec {self.n_rec}")

    # ── 축 슬라이스 ────────────────────────────────────────────────────────
    def layer_indices(self, spec: str) -> list[tuple[str, int]]:
        """'all' | 물리 layer 번호 콤마목록 → [(표시명, X 의 layer 축 index)]."""
        if spec.strip().lower() == "all":
            return [(str(L), i) for i, L in enumerate(self.capture_layers)]
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            L = int(tok)
            if L not in self.capture_layers:
                raise ValueError(f"{self.slug}: layer {L} 은 capture_layers {self.capture_layers} 에 없다")
            out.append((str(L), self.capture_layers.index(L)))
        return out

    def seg_indices(self, spec: str) -> list[tuple[str, int]]:
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in self.segment_names:
                raise ValueError(f"{self.slug}: segment '{tok}' 없음 (있는 것: {self.segment_names})")
            out.append((tok, self.segment_names.index(tok)))
        return out

    def slice_A(self, li: int, si: int, denoise: str) -> np.ndarray:
        """[n_rec, D] float32 — Tier A (layer, seg, denoise 지정)."""
        if denoise == "mean":
            v = self.X[:, li, :, si, :].astype(np.float32).mean(axis=1)
        else:
            v = self.X[:, li, int(denoise), si, :].astype(np.float32)
        return np.ascontiguousarray(v)

    def slice_B(self, li: int, ti: int) -> np.ndarray:
        """[n_rec, D] float32 — Tier B (layer, token 지정)."""
        return np.ascontiguousarray(self.X[:, li, ti, :].astype(np.float32))

    def slice_vl(self) -> np.ndarray:
        if self.vl is None:
            raise ValueError(f"{self.slug}: vl 배열이 없다")
        return np.ascontiguousarray(np.asarray(self.vl).astype(np.float32))


# ────────────────────────────────────────────────────────────────── phase 라벨
def phase_labels(sh: Shard, phase_def: str, seed: int,
                 labels_cache: dict[str, dict]) -> tuple[np.ndarray, str, dict[int, str]]:
    """record 별 phase 코드 [n_rec] + 정의 이름 + {code: 표시명}."""
    kind = phase_def.split(":", 1)[0]
    arg = phase_def.split(":", 1)[1] if ":" in phase_def else ""

    if kind == "gt":
        code = sh.phase_code.copy()
        names = {v: k for k, v in sh.phase_codebook.items()}
        return code, "gt", names

    if kind == "labels":
        key = str(Path(arg))
        if key not in labels_cache:
            z = np.load(arg, allow_pickle=False)
            labels_cache[key] = {k: z[k] for k in z.files}
        blob = labels_cache[key]
        arr = None
        for cand in (sh.slug, sh.instruction):
            if cand in blob:
                arr = np.asarray(blob[cand]).astype(np.int64).ravel()
                break
        if arr is None:
            raise KeyError(f"labels npz 에 '{sh.slug}' 키가 없다 (있는 키: {sorted(blob)[:10]}…)")
        if len(arr) != sh.n_rec:
            raise ValueError(f"labels['{sh.slug}'] 행수 {len(arr)} != shard n_rec {sh.n_rec}")
        dn = blob.get("def_name")
        dn = str(dn.item() if hasattr(dn, "item") else dn) if dn is not None else "labels"
        return arr, dn, {}

    if kind == "time-quantile":
        K = int(arg)
        if K < 1:
            raise ValueError("time-quantile K >= 1")
        code = np.zeros(sh.n_rec, np.int64)
        for ep in np.unique(sh.ep_id):
            m = np.where(sh.ep_id == ep)[0]
            order = m[np.argsort(sh.rec_idx[m], kind="mergesort")]
            n = len(order)
            code[order] = np.minimum((np.arange(n) * K) // max(n, 1), K - 1)
        return code, f"time-quantile:{K}", {i: f"q{i}" for i in range(K)}

    if kind == "shuffle":
        # GT 의 run-length 는 보존하고 episode 안에서 **구간 순서만** 셔플.
        rng = np.random.default_rng(seed)
        code = sh.phase_code.copy()
        for ep in np.unique(sh.ep_id):
            m = np.where(sh.ep_id == ep)[0]
            order = m[np.argsort(sh.rec_idx[m], kind="mergesort")]
            seq = sh.phase_code[order]
            runs, i = [], 0
            while i < len(seq):
                j = i
                while j + 1 < len(seq) and seq[j + 1] == seq[i]:
                    j += 1
                runs.append((int(seq[i]), j - i + 1))
                i = j + 1
            perm = rng.permutation(len(runs))
            new, pos = np.empty(len(seq), np.int64), 0
            for p in perm:
                lab, ln = runs[p]
                new[pos:pos + ln] = lab
                pos += ln
            code[order] = new
        names = {v: k for k, v in sh.phase_codebook.items()}
        return code, "shuffle", names

    raise ValueError(f"알 수 없는 --phase-def: {phase_def}")


# ──────────────────────────────────────────────────────── (episode, phase) 집계
class PhaseCell:
    """한 (instruction, phase) 의 episode 인덱스 구조 — 모든 layer/seg cell 이 공유."""

    def __init__(self, phase_code: int, phase_name: str, eps: list[int],
                 rows: list[np.ndarray], y: np.ndarray, sc: np.ndarray,
                 counts: np.ndarray, budget: int):
        self.phase_code = phase_code
        self.phase_name = phase_name
        self.eps = eps
        self.rows = rows              # episode 별 (첫 B 개) record row index [B]
        self.y = y
        self.sc = sc
        self.counts = counts          # phase 내 record 수 (길이 감사용, budget 자르기 전)
        self.budget = budget

    @property
    def n_mixed_scenes(self) -> int:
        n = 0
        for s in np.unique(self.sc):
            m = self.sc == s
            if (self.y[m] == 1).any() and (self.y[m] == 0).any():
                n += 1
        return n

    def pool(self, V: np.ndarray, t: int | None = None) -> np.ndarray:
        """[n_ep, D] — t=None 이면 첫 B 개 mean(equal-budget), t 면 그 t 번째 record 단독."""
        if t is None:
            return np.stack([V[r].mean(axis=0) for r in self.rows], axis=0)
        return np.stack([V[r[t]] for r in self.rows], axis=0)


def build_phase_cells(sh: Shard, code: np.ndarray, names: dict[int, str],
                      keep: np.ndarray, min_budget: int,
                      align: str = "first") -> tuple[list[PhaseCell], list[dict]]:
    """(task, phase) 별 동일예산 구조 생성. 반환 = (유효 cell, skip 기록).

    align="first" = phase 앞쪽 B개(조기성). align="end" = 마지막 B개 —
    phase 종료 직전(예: reach 의 grasp 시도 구간) 정렬. end 는 실패 판에서
    늦은 절대시각을 읽게 되므로 사후 판독 혼입 가능 — length_auroc 와 함께 읽을 것."""
    cells, skips = [], []
    for pc in sorted(int(v) for v in np.unique(code[keep])):
        pname = names.get(pc, str(pc))
        eps, rows, y, sc, counts = [], [], [], [], []
        for ep in np.unique(sh.ep_id[keep]):
            m = np.where(keep & (sh.ep_id == ep) & (code == pc))[0]
            if len(m) == 0:
                continue                                   # 그 phase 미도달 episode
            m = m[np.argsort(sh.rec_idx[m], kind="mergesort")]
            eps.append(int(ep))
            rows.append(m)
            y.append(int(sh.succ[m[0]]))
            sc.append(int(sh.scene[m[0]]))
            counts.append(len(m))
        if not eps:
            continue
        counts = np.asarray(counts, np.int64)
        B = int(counts.min())
        y = np.asarray(y, np.int64)
        sc = np.asarray(sc, np.int64)
        if B < min_budget:
            skips.append({"phase": pname, "phase_code": pc, "budget_B": B,
                          "n_succ": int((y == 1).sum()), "n_fail": int((y == 0).sum()),
                          "skip_reason": f"budget_B={B} < min_budget={min_budget}"})
            continue
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            skips.append({"phase": pname, "phase_code": pc, "budget_B": B,
                          "n_succ": int((y == 1).sum()), "n_fail": int((y == 0).sum()),
                          "skip_reason": "single-class (succ 또는 fail 이 0)"})
            continue
        rows = [r[-B:] for r in rows] if align == "end" else [r[:B] for r in rows]
        cell = PhaseCell(pc, pname, eps, rows, y, sc, counts, B)
        cell.align = align
        cells.append(cell)
    return cells, skips


# ─────────────────────────────────────────────────────────────────── cell 평가
def eval_cell(V: np.ndarray, pcell: PhaseCell, n_perm: int, rng, fixed_t: bool) -> dict:
    M = pcell.pool(V)
    a, n_eval = loso(M, pcell.y, pcell.sc)
    out = {
        "auroc": None if np.isnan(a) else float(a),
        "n_eval": int(n_eval),
        "null_z": None, "null_mean": None, "null_sd": None, "p": None,
    }
    if not np.isnan(a) and n_perm > 0:
        nul = perm_null(M, pcell.y, pcell.sc, n_perm, rng)
        if len(nul):
            sd = float(nul.std())
            out.update(null_mean=float(nul.mean()), null_sd=sd, n_perm=int(len(nul)),
                       null_z=float((a - nul.mean()) / sd) if sd > 0 else None,
                       p=float((nul >= a).mean()))
    if fixed_t:
        curve = []
        for t in range(pcell.budget):
            at, _ = loso(pcell.pool(V, t=t), pcell.y, pcell.sc)
            curve.append(None if np.isnan(at) else float(at))
        out["fixed_t_auroc"] = curve
    return out


def base_row(sh: Shard, pcell: PhaseCell, phase_def_name: str) -> dict:
    """cell 마다 반복되는 공통 열 (길이 감사·표본 규모)."""
    la = auroc(pcell.counts.astype(np.float64), pcell.y)   # 길이 자체의 판별력
    nmix = pcell.n_mixed_scenes
    return {
        "instruction": sh.instruction,
        "slug": sh.slug,
        "tier": sh.tier,
        "phase_def": phase_def_name,
        "phase": pcell.phase_name,
        "phase_code": pcell.phase_code,
        "budget_B": pcell.budget,
        "align": getattr(pcell, "align", "first"),
        "n_succ": int((pcell.y == 1).sum()),
        "n_fail": int((pcell.y == 0).sum()),
        "n_ep": int(len(pcell.y)),
        "n_mixed_scenes": nmix,
        "exploratory": bool(nmix < MIXED_SCENE_MIN),
        "length_auroc": None if np.isnan(la) else float(la),
        "skipped": False,
        "skip_reason": None,
    }


# ──────────────────────────────────────────────────────────────────────── main
def collect_shards(spec: str) -> list[Path]:
    out: list[Path] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        p = Path(tok)
        if p.is_dir():
            out.extend(sorted(p.glob("*.npz")))
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(tok)
    if not out:
        raise FileNotFoundError(f"shard 없음: {spec}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="dir 또는 npz 콤마목록")
    ap.add_argument("--layers", default="all", help="all | 물리 layer 번호 콤마목록")
    ap.add_argument("--segs", default="state,future,action,all", help="Tier A 전용")
    ap.add_argument("--denoise", default="mean", choices=["mean", "0", "1", "2", "3"])
    ap.add_argument("--vl", action="store_true", help="VL 2048 도 분석 축에 추가")
    ap.add_argument("--phase-def", default="gt",
                    help="gt | labels:<npz> | time-quantile:<K> | shuffle")
    ap.add_argument("--window", type=int, default=0, help="rec_idx < N 컷 (0=무제한)")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--min-budget", type=int, default=3)
    ap.add_argument("--align", choices=["first", "end"], default="first",
                    help="phase 내 equal-budget 정렬: first=앞쪽 B개(조기성), end=마지막 B개(전환 직전)")
    ap.add_argument("--fixed-t", action="store_true")
    ap.add_argument("--tokens", default="all", help="Tier B: all | token index 콤마목록")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 JSON")
    ap.add_argument("--tsv", default=None, help="결과 TSV (평평한 같은 내용)")
    args = ap.parse_args()

    paths = collect_shards(args.shards)
    labels_cache: dict[str, dict] = {}
    cells_out: list[dict] = []
    phase_def_name = args.phase_def

    for path in paths:
        sh = Shard(path)
        code, phase_def_name, names = phase_labels(sh, args.phase_def, args.seed, labels_cache)
        keep = np.ones(sh.n_rec, bool) if args.window <= 0 else (sh.rec_idx < args.window)
        if not keep.any():
            print(f"[skip] {sh.slug}: window={args.window} 로 남는 record 0", flush=True)
            continue
        pcells, skips = build_phase_cells(sh, code, names, keep, args.min_budget,
                                          align=args.align)
        for s in skips:
            cells_out.append({"instruction": sh.instruction, "slug": sh.slug, "tier": sh.tier,
                              "phase_def": phase_def_name, "layer": None, "seg": None,
                              "token": None, "denoise": args.denoise, "auroc": None,
                              "null_z": None, "length_auroc": None, "n_mixed_scenes": None,
                              "exploratory": None, "skipped": True, **s})
        if not pcells:
            print(f"[skip] {sh.slug}: 유효 phase cell 0", flush=True)
            continue

        # 분석 축 목록 = (표시 layer, 표시 seg/token, 슬라이스 함수)
        axes: list[tuple[str, str | None, int | None, callable]] = []
        for lname, li in sh.layer_indices(args.layers):
            if sh.tier == "A":
                for sname, si in sh.seg_indices(args.segs):
                    axes.append((lname, sname, None,
                                 (lambda li=li, si=si: sh.slice_A(li, si, args.denoise))))
            else:
                toks = (range(sh.n_token) if args.tokens.strip().lower() == "all"
                        else [int(t) for t in args.tokens.split(",") if t.strip()])
                for ti in toks:
                    axes.append((lname, None, int(ti), (lambda li=li, ti=ti: sh.slice_B(li, ti))))
        if args.vl and sh.has_vl and sh.vl is not None:
            axes.append(("VL", "vl", None, sh.slice_vl))

        for lname, sname, tok, getter in axes:
            V = getter()
            for pcell in pcells:
                rng = np.random.default_rng(args.seed)   # cell 마다 동일 seed → 결정론
                row = base_row(sh, pcell, phase_def_name)
                row.update(layer=lname, seg=sname, token=tok, denoise=args.denoise)
                row.update(eval_cell(V, pcell, args.n_perm, rng, args.fixed_t))
                cells_out.append(row)
            del V
        print(f"[done] {sh.slug}: phases={len(pcells)} axes={len(axes)} "
              f"cells={len(pcells) * len(axes)}", flush=True)

    payload = {
        "config": {
            "shards": [str(p) for p in paths], "layers": args.layers, "segs": args.segs,
            "denoise": args.denoise, "vl": bool(args.vl), "phase_def": args.phase_def,
            "phase_def_name": phase_def_name, "window": args.window, "n_perm": args.n_perm,
            "min_budget": args.min_budget, "fixed_t": bool(args.fixed_t), "seed": args.seed,
            "mixed_scene_min": MIXED_SCENE_MIN,
            "unit": "(episode, phase) 1행 / equal-budget 첫 B record mean / LOSO fold=scene",
        },
        "cells": cells_out,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print("[written]", args.out, flush=True)
    if args.tsv:
        write_tsv(cells_out, Path(args.tsv))
        print("[written]", args.tsv, flush=True)
    if not args.out and not args.tsv:
        print(json.dumps(payload, ensure_ascii=False)[:2000])

    ok = [c for c in cells_out if not c.get("skipped") and c.get("auroc") is not None]
    if ok:
        top = sorted(ok, key=lambda c: -(c["auroc"] or 0))[:10]
        print(f"\n{'instr':22s} {'L':>4s} {'seg/tok':>8s} {'phase':>10s} "
              f"{'AUROC':>6s} {'z':>6s} {'lenA':>6s} {'B':>3s}")
        for c in top:
            st = c["seg"] if c["seg"] is not None else f"t{c['token']}"
            print(f"{str(c['instruction'])[:22]:22s} {str(c['layer']):>4s} {str(st):>8s} "
                  f"{str(c['phase'])[:10]:>10s} {_f(c['auroc']):>6s} {_f(c['null_z']):>6s} "
                  f"{_f(c['length_auroc']):>6s} {c['budget_B']:>3d}")
    return 0


TSV_COLS = ["instruction", "slug", "tier", "layer", "seg", "token", "denoise", "phase_def",
            "phase", "phase_code", "auroc", "null_z", "p", "length_auroc", "budget_B",
            "n_succ", "n_fail", "n_ep", "n_eval", "n_mixed_scenes", "exploratory",
            "skipped", "skip_reason"]


def write_tsv(cells: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_COLS)]
    for c in cells:
        lines.append("\t".join("" if c.get(k) is None else str(c.get(k)) for k in TSV_COLS))
    path.write_text("\n".join(lines) + "\n")


def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    raise SystemExit(main())
