#!/usr/bin/env python
"""grid shard 위 SAFE식 failure detector + 온라인 발화 시뮬레이션 (rung0 관문).

질문 하나: **"실패를 온라인으로, 개입할 시간이 남은 시점에 읽을 수 있나"**.
분리도 지도(`phase_sep_matrix.py`)는 사후 판독이고, 여기서는 causal detector
(step t 까지만 보고 점수)를 학습해 test 판에서 **첫 발화 시점**을 잰다. 발화가
전부 종반에 몰리면(relpos≈1, 남은 step≈0) steering 은 걸 곳이 없다 — 그 판정이 목적.

## 파이프라인

1. shard NPZ(`extract_grid_matrix.py` Tier A) → record 를 `ep_id` 로 묶고 `rec_idx`
   순 정렬 → episode 당 [T, D] 시퀀스. 라벨 y=1 은 **failure**(SAFE 규약).
2. scene 단위 결정적 분할 (train 6 / calib 2 / test 2). scene 을 섞지 않는다 —
   같은 scene 의 판이 train/test 에 갈리면 scene 암기가 검출로 보인다
   ([[scene-matched-succfail-verdict]]).
3. detector 학습 (SAFE-LSTM / SAFE-MLP). 구현 출처 =
   `scripts/safe/groot_n16/robocasa/analyze/pathway_lstm_detector.py` +
   `scripts/safe/groot_n16/robocasa/vis/core/lstm.py` (아키텍처·손실·정규화 동일).
   원격 노드에 그 파일들이 없을 수 있어 **import 하지 않고 이식**했다.
4. functional CP 밴드 δ_t = μ_t + bw·σ_t (성공 궤적의 시변 밴드).
   μ_t/σ_t = train 성공 판, bw = **calib 성공 판**의 max-normalized 초과량의
   (1−α) 분위수. t 가 밴드 길이를 넘으면 마지막 밴드 유지(plateau).
   `--truncate-train` 을 켜면 밴드 출처(train/calib)가 절제되어 밴드 길이가 W 로
   줄지만, **plateau 규약이 그대로 적용**되어 full test 시퀀스의 t>W 구간은 마지막
   밴드값으로 판정된다 (별도 처리 불필요).
5. 발화 시뮬: test 판마다 `score_t > δ_t` 인 **첫 t**. 기록 = fired / t_fire /
   relpos = t_fire/(T−1) / 그 순간의 GT phase 이름 / 종결까지 남은 step.
6. 길이 confound 통제 (`--truncate-train {none,rollout,phase-gt}`) + 그 판정을 위한
   무feature **timer 기준선**(t ≥ W 발화) · `tpr_before_W` · `lead_vs_W` ·
   고정 판정시각 AUROC(`auroc_td*`). 절제는 학습·보정에만 걸고 test 는 항상 full.

## arm

- `pertask` — task 별 detector 9개 (task 내부 일반화).
- `mixed`   — 전 task 풀링 detector 1개. 표준화·가중치는 공유하되 **CP 밴드는
  per-task calib 로 보정**(RL²-VLA 방식) — task 별 점수 스케일 차이를 밴드가 흡수.
- `loto`    — leave-one-task-out. task 하나를 빼고 학습, 그 task 의 **전 episode** 를
  zero-shot test (밴드도 train task 풀링으로 잡아 held-out 판을 한 개도 안 쓴다).
  질문 = "학습에 없던 task 에서 작동하나, phase 절제가 전이를 살리나"
  (SAFE 논문 unseen 전이 주장 재검 — seen18 재현은 unseen 0.434 였다).
- `loko-cell` — **scene-local leave-one-k(지터)-out**. 셀 = (instruction, scene,
  jitter, noise) 중 (slug, scene s, jitter j) 단위로 detector 를 하나씩 만든다.
  학습 = 같은 scene 의 **다른 j 전판**(pool_other) + **대상 j 의 실패판**,
  대상 j 의 **성공판은 학습에서 제외**(success-blind — 그 판이 곧 위양성 측정 대상).
  CP 밴드는 calib split 을 따로 떼지 않고 pool_other 성공판 위에서 **episode LOO
  (또는 k-fold)** 로 잡는다 (셀 하나의 성공 판 수가 적어 calib 을 또 쪼갤 수 없다).
  대상 j 실패판은 학습에 들어간 **in-sample** 이므로 그 발화율은 상한이지 일반화가
  아니다 — 행마다 `in_train` 열로 명시하고, 일반화는 `pool_other` / j-층화 AUROC 로 읽는다.

## 입력 계약 (Tier A shard)
`<shard-dir>/<instr_slug>.npz`
  X fp16 [n_rec, n_layer, K, S, D] / ep_id i32 / scene i16 / noise i16 / rec_idx i16
  succ i8 / phase_code i16 / ep_len i16 / meta_json{capture_layers|layers,
  segment_names, phase_codebook}
feature 좌표 기본값 = layer 12 × 마지막 denoise(k=3) × segment "all"(49토큰 평균).
layer 는 meta 의 layer 리스트에서 **인덱스 역산**한다 (하드코딩 금지).

## 출력 (`--out`)
  sim_summary.tsv                     task × arm × model × α 행 (TPR/FPR/발화시점/phase 분포)
  sim_detail.json                     episode 별 발화 기록
  detector_<arm>_<model>_<task>.pt    state_dict + 표준화 + feature 좌표 + CP 밴드
산출물에 절대경로를 쓰지 않는다 (docs/04 §8) — 입력은 shard 파일명(basename)만 기록.

## 사용
    # 로컬 게이트 (합성 데이터로 파이프라인 검증)
    python scripts/analysis/grid_phase/failure_detector_sim.py --self-test

    # 원격 CPU 노드 (torch CPU + numpy 만)
    ~/anaconda3/bin/python scripts/analysis/grid_phase/failure_detector_sim.py \
        --shard-dir ~/workspace/.../analysis/grid_phase/segA \
        --out ~/workspace/.../analysis/grid_phase/detector_sim \
        --arm both --models lstm,mlp --threads 8

    # 길이 절제 ablation (학습·보정만 절제, test 는 full)
    ... failure_detector_sim.py --shard-dir <segA> --out <out>/trunc_rollout \
        --truncate-train rollout --arm both --models lstm,mlp --threads 8
    ... failure_detector_sim.py --shard-dir <segA> --out <out>/trunc_phasegt \
        --truncate-train phase-gt --arm both --models lstm,mlp --threads 8

    # task 전이 (leave-one-task-out) — 절제 모드 3종을 각각
    for M in none rollout phase-gt; do \
      ~/anaconda3/bin/python scripts/analysis/grid_phase/failure_detector_sim.py \
        --shard-dir <segA> --out <out>/loto_$M --arm loto --truncate-train $M \
        --models lstm --threads 8 --quiet; done

    # scene-local LOKO (셀별 detector; 셀 목록 TSV 필수)
    ... failure_detector_sim.py --shard-dir <segA> --out <out>/loko \
        --arm loko-cell --loko-cells-tsv <cells.tsv> --models lstm \
        --min-pool-fail 3 --min-calib-succ 9 --cp-folds 0 --threads 8
"""
from __future__ import annotations

import os

# BLAS/torch 스레드 cap — 공유 노드. numpy/torch import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import csv
import json
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3)
MIN_BAND_EPS = 3          # μ/σ, bw 추정에 필요한 최소 성공 판 수 (참조 구현과 동일)
EPS = 1e-8
TD_GRID = (5, 10, 15, 20, 30)   # 고정 판정시각 AUROC 격자 (causal, full test 시퀀스 위)
MIN_TRUNC_LEN = 2         # 절제 후 이보다 짧은 시퀀스는 버린다
JSTRAT_TD = 10            # loko-cell j-층화 AUROC 의 고정 판정시각 (TD_GRID 안의 값)
# 셀 TSV 의 지터 열 — jitter_idx 가 있으면 그것만 쓴다 (jitter_reset_idx 는 legacy).
CELL_JITTER_COLS = ("jitter_idx", "jitter_reset_idx")
# instruction 식별 열 (매칭 시도 순서). v6 TSV 는 slug+grid_instruction 을 담는다.
CELL_INSTR_COLS = ("slug", "grid_instruction", "instruction")
# 사람이 읽는 라벨로 쓸 열 (없으면 slug)
CELL_LABEL_COLS = ("grid_instruction", "instruction", "slug")


# =============================================================================
# detector (SAFE-LSTM / SAFE-MLP) — 참조 구현 이식 (import 의존 없이 standalone)
# =============================================================================

class LSTMDetector(nn.Module):
    """단층 LSTM + linear + sigmoid. causal: out[t] 는 x[:t+1] 에만 의존."""
    cumulative = False

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):                       # [B,T,D] → [B,T]
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


class MLPDetector(nn.Module):
    """SAFE-MLP: per-step MLP→scalar→sigmoid (step 독립). 검출 score 는 출력 누적평균."""
    cumulative = True

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):                       # [B,T,D] → [B,T]
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_detector(kind: str, input_dim: int, hidden: int) -> nn.Module:
    if kind == "lstm":
        return LSTMDetector(input_dim, hidden)
    if kind == "mlp":
        return MLPDetector(input_dim, hidden)
    raise ValueError(f"알 수 없는 detector: {kind}")


# =============================================================================
# shard I/O
# =============================================================================

class Episode:
    """판 하나. X 는 표준화 **전** 원본 [T,D] (표준화는 arm 별로 다시 건다)."""

    __slots__ = ("task", "ep_id", "scene", "noise", "succ", "X", "phase", "T", "jitter")

    def __init__(self, task, ep_id, scene, noise, succ, X, phase, jitter=-1):
        self.task, self.ep_id, self.scene, self.noise = task, ep_id, scene, noise
        self.succ = int(succ)
        self.X = X
        self.phase = phase
        self.T = int(X.shape[0])
        self.jitter = int(jitter)   # 지터 축 k (reset_idx); 2축 legacy shard 는 -1

    @property
    def y(self) -> int:
        """SAFE 규약: failure = positive."""
        return 1 - self.succ


class ShardSpec:
    """feature 좌표(layer/denoise/seg)를 meta 에서 역산한 결과 — 계약 위반은 fail-loud."""

    def __init__(self, path: Path, layer: int, denoise: int, seg: str):
        with np.load(path, allow_pickle=False) as f:
            meta_raw = f["meta_json"] if "meta_json" in f.files else None
            shape = f["X"].shape
        if meta_raw is None:
            raise ValueError(f"{path.name}: meta_json 없음 — Tier A shard 가 맞나")
        self.meta = json.loads(str(meta_raw))
        if len(shape) != 5:
            raise ValueError(
                f"{path.name}: X.ndim={len(shape)} — Tier A [n_rec,L,K,S,D] 만 지원 "
                "(tokB shard 는 이 도구의 입력이 아니다)")
        self.n_rec, self.n_layer, self.n_denoise, self.n_seg, self.dim = shape

        layers = self.meta.get("capture_layers") or self.meta.get("layers")
        if layers is None:
            raise ValueError(f"{path.name}: meta 에 capture_layers/layers 가 없다")
        self.layers = [int(v) for v in layers]
        if len(self.layers) != self.n_layer:
            raise ValueError(
                f"{path.name}: capture_layers {len(self.layers)} != X layer축 {self.n_layer}")
        if layer not in self.layers:
            raise ValueError(f"{path.name}: layer {layer} 없음 (있는 것: {self.layers})")
        self.layer_idx = self.layers.index(layer)

        seg_names = [str(s) for s in (self.meta.get("segment_names") or [])]
        if len(seg_names) != self.n_seg:
            seg_names = ["state", "future", "action", "all"][: self.n_seg]
        self.segment_names = seg_names
        if seg not in seg_names:
            raise ValueError(f"{path.name}: segment '{seg}' 없음 (있는 것: {seg_names})")
        self.seg_idx = seg_names.index(seg)

        d = self.n_denoise - 1 if denoise < 0 else denoise
        if not (0 <= d < self.n_denoise):
            raise ValueError(f"{path.name}: denoise {denoise} 범위 밖 (K={self.n_denoise})")
        self.denoise_idx = d

        self.phase_names = {int(v): str(k) for k, v in
                            (self.meta.get("phase_codebook") or {}).items()}
        self.instruction = str(self.meta.get("instruction", path.stem))


def load_shard_episodes(path: Path, layer: int, denoise: int, seg: str
                        ) -> tuple[list[Episode], ShardSpec]:
    """shard NPZ → episode 목록. rollout pooling 없음 — per-record 시퀀스를 유지한다."""
    spec = ShardSpec(path, layer, denoise, seg)
    task = path.stem
    with np.load(path, allow_pickle=False) as f:
        X = f["X"]
        feat = np.ascontiguousarray(
            X[:, spec.layer_idx, spec.denoise_idx, spec.seg_idx, :]).astype(np.float32)
        del X
        cols = {}
        for k in ("ep_id", "scene", "noise", "rec_idx", "succ", "phase_code", "ep_len"):
            if k not in f.files:
                raise ValueError(f"{path.name}: 필수 열 '{k}' 없음")
            cols[k] = np.asarray(f[k]).ravel().astype(np.int64)
        if "jitter" in f.files:
            cols["jitter"] = np.asarray(f["jitter"]).ravel().astype(np.int64)
    n = len(feat)
    for k, v in cols.items():
        if len(v) != n:
            raise ValueError(f"{path.name}: {k} 길이 {len(v)} != n_rec {n}")

    eps: list[Episode] = []
    for ep in np.unique(cols["ep_id"]):
        m = np.where(cols["ep_id"] == ep)[0]
        m = m[np.argsort(cols["rec_idx"][m], kind="mergesort")]
        su = np.unique(cols["succ"][m])
        if len(su) != 1:
            raise ValueError(f"{path.name}: ep{ep} 의 succ 이 record 마다 다르다 {su}")
        sc = np.unique(cols["scene"][m])
        if len(sc) != 1:
            raise ValueError(f"{path.name}: ep{ep} 의 scene 이 record 마다 다르다 {sc}")
        if "ep_len" in cols:
            el = int(np.unique(cols["ep_len"][m])[0])
            if el != len(m):
                raise ValueError(
                    f"{path.name}: ep{ep} ep_len={el} != record 수 {len(m)}")
        eps.append(Episode(task, int(ep), int(sc[0]), int(cols["noise"][m[0]]),
                           int(su[0]), feat[m], cols["phase_code"][m],
                           jitter=int(cols["jitter"][m[0]]) if "jitter" in cols else -1))
    if not eps:
        raise ValueError(f"{path.name}: episode 0")
    return eps, spec


def discover_shards(shard_dir: Path, only: list[str] | None) -> list[Path]:
    paths = sorted(p for p in shard_dir.glob("*.npz") if not p.name.endswith("_fit.npz"))
    if only:
        keep = set(only)
        paths = [p for p in paths if p.stem in keep]
        missing = keep - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"shard 없음: {sorted(missing)}")
    if not paths:
        raise SystemExit(f"shard NPZ 없음: {shard_dir}")
    return paths


# =============================================================================
# 셀 TSV (v6 열 계약) — exclude / loko 공용 리더
# =============================================================================
# 셀 키 = (instruction, scene_idx, jitter_idx, noise_idx). instruction 은 사람이 쓰는
# 표기("PPCC/bread")와 shard 파일 stem(slug, "PPCC_bread")이 갈리므로 둘 다로 매칭한다.


def _instr_variants(v) -> set[str]:
    """매칭 후보 문자열 집합 ('/'·공백 → '_' 변형 포함)."""
    s = str(v or "").strip()
    if not s:
        return set()
    return {s, s.replace("/", "_").replace(" ", "_")}


def shard_slug_index(paths: list[Path]) -> dict[str, str]:
    """slug(stem) → meta instruction. meta_json 만 읽는다 (X 는 안 푼다)."""
    idx: dict[str, str] = {}
    for p in paths:
        instr = ""
        try:
            with np.load(p, allow_pickle=False) as f:
                if "meta_json" in f.files:
                    instr = str(json.loads(str(f["meta_json"])).get("instruction", ""))
        except (OSError, ValueError, KeyError):
            instr = ""      # 깨진 shard 는 실제 로드(ShardSpec)에서 fail-loud 된다
        idx[p.stem] = instr
    return idx


def read_cell_tsv(path, slug_instr: dict[str, str], what: str = "--cells-tsv"
                  ) -> tuple[list[dict], dict[str, list[dict]]]:
    """셀 TSV → (행 목록, slug별 인덱스). 계약 위반은 전부 fail-loud.

    필수 열: scene_idx, noise_idx, 식별 열(instruction|slug|grid_instruction 중 하나
    이상), 지터 열 하나 이상. 그 밖의 열(machine·sig·rel_path·pool_* 등)은 무시한다.

    지터 열 규약: **`jitter_idx` 가 있으면 그것만 셀 키로 쓴다** (`jitter_reset_idx` 는
    무시). v6 TSV 는 두 열을 다 담고, oven/washer 는 jitter_reset_idx 가 전부 0 이라
    두 값이 정당하게 다르다 — 불일치는 정보성 로그 한 줄로만 남긴다.
    `jitter_idx` 가 없으면 legacy `jitter_reset_idx` 를 쓰되, 그 값이 전부 같으면
    지터 축이 없는 것이므로 fail-loud.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what}: 파일 없음 {p.name}")
    with open(p, encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols = list(rd.fieldnames or [])
        raw_rows = [dict(r) for r in rd]

    miss = [c for c in ("scene_idx", "noise_idx") if c not in cols]
    if miss:
        raise SystemExit(f"{what}: 필수 열 누락 {miss} (있는 열: {cols})")
    if not any(c in cols for c in CELL_INSTR_COLS):
        raise SystemExit(f"{what}: 식별 열이 필요하다 {CELL_INSTR_COLS} "
                         f"(있는 열: {cols})")
    jcol = next((c for c in CELL_JITTER_COLS if c in cols), None)
    if jcol is None:
        raise SystemExit(f"{what}: 지터 열 없음 — {CELL_JITTER_COLS[0]} (또는 legacy "
                         f"{CELL_JITTER_COLS[1]}) 이 필요하다 (있는 열: {cols})")
    if jcol == "jitter_idx" and "jitter_reset_idx" in cols:
        n_diff = sum(1 for r in raw_rows
                     if str(r.get("jitter_idx", "")).strip()
                     and str(r.get("jitter_reset_idx", "")).strip()
                     and str(r["jitter_idx"]).strip() != str(r["jitter_reset_idx"]).strip())
        if n_diff:
            print(f"[cells] jitter_idx≠jitter_reset_idx 행 {n_diff}/{len(raw_rows)} "
                  "— jitter_idx 채택", flush=True)
    if jcol == "jitter_reset_idx":
        vals = {str(r.get(jcol, "")).strip() for r in raw_rows}
        vals.discard("")
        if len(vals) <= 1:
            raise SystemExit(f"{what}: 지터 축 없음 (jitter_reset_idx 값 {sorted(vals)}) "
                             "— v6 jitter_idx 열 필요")

    # 매칭 키 → slug (충돌은 fail-loud)
    keymap: dict[str, str] = {}
    for slug, instr in sorted(slug_instr.items()):
        for k in _instr_variants(slug) | _instr_variants(instr):
            if k in keymap and keymap[k] != slug:
                raise SystemExit(f"{what}: instruction 키 '{k}' 가 shard "
                                 f"{keymap[k]} / {slug} 둘에 걸린다 (모호)")
            keymap[k] = slug

    rows: list[dict] = []
    by_slug: dict[str, list[dict]] = {}
    unmatched: list[str] = []
    for i, r in enumerate(raw_rows, start=2):
        cands: list[str] = []
        for c in CELL_INSTR_COLS:                 # slug → grid_instruction → instruction
            cands += list(_instr_variants(r.get(c)))
        slug = next((keymap[c] for c in cands if c in keymap), None)
        if slug is None:
            unmatched.append("line{}: {}".format(
                i, {c: r.get(c, "") for c in CELL_INSTR_COLS if c in cols}))
            continue
        jv = str(r.get(jcol, "")).strip()
        if jv == "":
            raise SystemExit(f"{what}: line{i} 지터 값({jcol})이 비어 있다")
        row = {"slug": slug,
               "instruction": str(next((r[c] for c in CELL_LABEL_COLS
                                        if r.get(c)), "") or slug),
               "scene": int(r["scene_idx"]), "jitter": int(jv),
               "noise": int(r["noise_idx"]), "raw": r}
        rows.append(row)
        by_slug.setdefault(slug, []).append(row)
    # 이 실행에 로드된 shard 에 없는 instruction 행은 **정상적으로 있을 수 있다** —
    # 셀 TSV 는 전 instruction 공용이고 러너는 준비된 instruction 만 골라 돌리기 때문
    # (2026-09-04: 증분 학습에서 이걸 fail-loud 로 막아 첫 셀이 무음 실패했다).
    # 따라서 미매칭은 건너뛰고 수만 로그로 남기되, **한 행도 안 맞으면** 잘못된 TSV·
    # shard 조합이므로 그때는 멈춘다.
    if unmatched:
        print(f"[cells] {what}: 로드된 shard 밖 행 {len(unmatched)}개 건너뜀 "
              f"(shard: {sorted(slug_instr)})", flush=True)
    if not rows:
        raise SystemExit(f"{what}: 유효 행 0 — 로드된 shard {sorted(slug_instr)} 와 "
                         f"겹치는 셀이 없다 (미매칭 {len(unmatched)}: {unmatched[:5]})")
    return rows, by_slug


# =============================================================================
# split (scene 단위 결정적 분할)
# =============================================================================

def split_scenes(task: str, scenes: list[int], n_train: int, n_calib: int,
                 n_test: int, seed: int) -> dict[str, list[int]]:
    """정렬 후 (seed, task) 로 결정적 셔플 → train/calib/test scene.

    scene 이 모자라면 test/calib 1개씩부터 채우고 나머지를 train 에 준다 (fail-loud
    최소값 = 3 scene). 남는 scene 은 train 으로 — 학습 표본을 우선한다.
    """
    uniq = sorted(set(int(s) for s in scenes))
    if len(uniq) < 3:
        raise ValueError(f"{task}: scene {len(uniq)}개 — train/calib/test 분할 불가(최소 3)")
    rng = np.random.default_rng([seed, zlib.crc32(task.encode("utf-8"))])
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    n = len(order)
    te = min(n_test, max(1, n - 2))
    ca = min(n_calib, max(1, n - te - 1))
    tr = n - te - ca
    if tr < 1:
        raise ValueError(f"{task}: scene {n}개로 train scene 을 못 만든다")
    return {"test": sorted(order[:te]), "calib": sorted(order[te:te + ca]),
            "train": sorted(order[te + ca:])}


def standardizer(eps: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    allf = np.concatenate([e.X for e in eps], axis=0)
    mu = allf.mean(axis=0)
    sd = allf.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_std(e: Episode, mu, sd) -> np.ndarray:
    return ((e.X - mu) / sd).astype(np.float32)


# =============================================================================
# 학습 데이터 길이 절제 (length confound ablation)
# =============================================================================
# 실패는 항상 timeout(길다) / 성공은 조기종료(짧다) → full 시퀀스 학습은 "길면 실패"를
# 학습할 수 있다([[seen18-rollout-length-confound]]). cap 정의는 레포 규약과 동일:
#   rollout  W    = TRAIN 성공 episode record 수의 ceil(μ+1σ)
#   phase-gt cap  = TRAIN 성공 episode 의 그 phase dwell(>0) 의 ceil(μ+1σ)
# (`scripts/fit/fit_setm.py::phase_dwell_caps`,
#  `scripts/fit/fit_phase_conceptor.py::compute_length_caps` 와 같은 식. np.std 는
#  모집단 표준편차(ddof=0) — 그쪽 구현과 동일하게 맞춘다.)
# 성공 dwell 이 없는 phase 는 fit 쪽 규약대로 **대조 불가 → skip**: 해당 record 를
# 버린다. phase_code<0 (unknown/terminal) 도 같은 규칙이 적용된다 — 성공 판에 그
# code 가 있으면 cap 이 생기고, 없으면 drop.
# 절제는 **TRAIN·CALIB 에만** 건다. TEST 는 항상 full (온라인 현실성).


def _ceil_mu_sigma(vals: list[int]) -> int:
    v = np.asarray([x for x in vals if x > 0], dtype=np.float64)
    return int(np.ceil(v.mean() + v.std()))


def rollout_cap(train_eps: list[Episode]) -> int | None:
    """W = TRAIN 성공 판 record 수의 ceil(μ+1σ). 성공 판이 없으면 None."""
    lens = [e.T for e in train_eps if e.succ == 1]
    return _ceil_mu_sigma(lens) if any(x > 0 for x in lens) else None


def phase_dwell_caps(train_eps: list[Episode]) -> dict[int, int]:
    """phase code → dwell cap. TRAIN 성공 판의 dwell(>0) 만 사용."""
    codes: set[int] = set()
    for e in train_eps:
        codes.update(int(c) for c in np.unique(e.phase))
    caps: dict[int, int] = {}
    for c in sorted(codes):
        dw = [int((e.phase == c).sum()) for e in train_eps if e.succ == 1]
        dw = [d for d in dw if d > 0]
        if dw:
            caps[c] = _ceil_mu_sigma(dw)
    return caps


def truncate_episode(e: Episode, mode: str, W: int | None,
                     caps: dict[int, int] | None) -> Episode | None:
    """절제된 사본. 절제 불가/너무 짧으면 None (호출부에서 skip 카운트)."""
    if mode == "none":
        return e
    if mode == "rollout":
        if W is None:
            return e
        idx = np.arange(min(e.T, int(W)))
    elif mode == "phase-gt":
        if not caps:
            return e
        keep: list[int] = []
        for c, cap in caps.items():
            sel = np.where(e.phase == c)[0][:cap]     # 시간순 앞쪽 우선
            keep.extend(int(i) for i in sel)
        idx = np.asarray(sorted(keep), dtype=np.int64)
    else:
        raise ValueError(f"알 수 없는 truncate 모드: {mode}")
    if len(idx) < MIN_TRUNC_LEN:
        return None
    return Episode(e.task, e.ep_id, e.scene, e.noise, e.succ,
                   np.ascontiguousarray(e.X[idx]), np.ascontiguousarray(e.phase[idx]),
                   jitter=e.jitter)


def apply_truncation(split: dict, mode: str, W: int | None,
                     caps: dict[int, int] | None) -> dict:
    """TRAIN·CALIB 만 절제하고 dropped 수를 센다. TEST 는 손대지 않는다."""
    dropped = {"train": 0, "calib": 0}
    for part in ("train", "calib"):
        kept = []
        for e in split[part]:
            te = truncate_episode(e, mode, W, caps)
            if te is None:
                dropped[part] += 1
            else:
                kept.append(te)
        split[part] = kept
    return dropped


# =============================================================================
# 학습 / 채점
# =============================================================================

def train_detector(kind: str, seqs: list[tuple[np.ndarray, int]], input_dim: int,
                   epochs: int, lr: float, hidden: int, lam: float, grad_clip: float,
                   batch_size: int, seed: int, device: str = "cpu",
                   verbose: bool = True) -> nn.Module:
    """per-step BCE(episode 라벨을 전 step 에 broadcast) + λ·L2(bias 제외) + grad clip.

    참조 구현은 bs=1 이라 판 수가 많으면 느리다 → 패딩 배치 + mask 로 pad step 제외
    (mask 평균이라 bs=1 과 손실 정의가 같다).
    """
    torch.manual_seed(seed)
    model = build_detector(kind, input_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(seqs)
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            batch = [seqs[i] for i in idx]
            L = max(len(b[0]) for b in batch)
            xb = torch.zeros(len(batch), L, input_dim)
            yb = torch.zeros(len(batch), L)
            mb = torch.zeros(len(batch), L)
            for i, (X, y) in enumerate(batch):
                t = len(X)
                xb[i, :t] = torch.from_numpy(X)
                yb[i, :t] = float(y)
                mb[i, :t] = 1.0
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            sc = model(xb).clamp(1e-6, 1 - 1e-6)
            bce = -(yb * torch.log(sc) + (1 - yb) * torch.log(1 - sc))
            loss = (bce * mb).sum() / mb.sum().clamp(min=1.0)
            if lam > 0:
                loss = loss + lam * sum((p ** 2).sum() for nm, p in model.named_parameters()
                                        if "bias" not in nm)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"      epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def score_seq(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """[T] per-step 검출 score. MLP 는 step 독립이라 출력 **누적평균**이 score."""
    xb = torch.from_numpy(np.ascontiguousarray(X)).float().unsqueeze(0).to(device)
    raw = model(xb).squeeze(0).cpu().numpy()
    if getattr(model, "cumulative", False):
        raw = np.cumsum(raw) / np.arange(1, len(raw) + 1)
    return raw.astype(np.float64)


def _pad_to(s: np.ndarray, L: int) -> np.ndarray:
    return s[:L] if len(s) >= L else np.concatenate([s, np.full(L - len(s), s[-1])])


def functional_cp_band(band_scores: list[np.ndarray], calib_scores: list[np.ndarray],
                       alpha: float) -> dict | None:
    """δ_t = μ_t + bw·σ_t. μ/σ = band_scores(성공), bw = calib 성공의 초과량 (1−α) 분위.

    궤적은 밴드 길이 L 로 forward-fill 패딩(성공 종료 후 plateau) — 짧은 성공 판이
    밴드를 끌어내려 종반에 헛발화하는 것을 막는다.
    """
    if len(band_scores) < MIN_BAND_EPS or len(calib_scores) < MIN_BAND_EPS:
        return None
    L = max(len(s) for s in band_scores)
    tr = np.stack([_pad_to(s, L) for s in band_scores])
    ca = np.stack([_pad_to(s, L) for s in calib_scores])
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0, ddof=1) + EPS
    bw = float(np.quantile(np.max((ca - mu) / sd, axis=1), 1.0 - alpha))
    return {"mu": mu, "sd": sd, "bw": bw, "delta": mu + bw * sd, "L": int(L)}


def loo_cp_band(succ_scores: list[np.ndarray], alpha: float, folds: int = 0
                ) -> dict | None:
    """calib 분리 없이 **성공 판 LOO(또는 k-fold)** 로 잡는 functional CP 밴드.

    `functional_cp_band` 는 μ/σ 출처와 bw 출처(calib)를 다른 판 집합으로 나눈다.
    셀 하나(=한 scene 의 다른 지터 판들) 위에서는 성공 판이 10판 안팎이라 또 쪼개면
    양쪽 다 무너진다 → 같은 집합을 fold 로 돌려 쓴다: 각 fold 에서 held-out 판을 뺀
    나머지로 μ_t/σ_t 를 잡고, held-out 판의 초과량 max_t((s−μ)/σ) 를 모아 (1−α) 분위를
    bw 로 쓴다 (split-CP 대신 cross-conformal). 최종 μ/σ 는 전체 성공 판으로 다시 잡고
    δ = μ + bw·σ. 패딩·ddof·EPS 규약은 `functional_cp_band` 와 동일.
    """
    n = len(succ_scores)
    if n < MIN_BAND_EPS or n < 2:
        return None
    L = max(len(s) for s in succ_scores)
    M = np.stack([_pad_to(s, L) for s in succ_scores])
    k = n if folds is None or int(folds) <= 0 else min(int(folds), n)
    assign = np.arange(n) % k                    # 결정적 분할 (셔플 없음)
    exceed: list[float] = []
    for f in range(k):
        ho = np.where(assign == f)[0]
        tr = np.where(assign != f)[0]
        if len(tr) < 2:                          # ddof=1 σ 가 정의되지 않는다
            continue
        mu_f = M[tr].mean(axis=0)
        sd_f = M[tr].std(axis=0, ddof=1) + EPS
        for i in ho:
            exceed.append(float(np.max((M[i] - mu_f) / sd_f)))
    if not exceed:
        return None
    bw = float(np.quantile(np.asarray(exceed, dtype=np.float64), 1.0 - alpha))
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1) + EPS
    return {"mu": mu, "sd": sd, "bw": bw, "delta": mu + bw * sd, "L": int(L),
            "n_cal": int(n), "folds": int(k)}


def fire_step(score: np.ndarray, delta: np.ndarray) -> int | None:
    """첫 발화 t. t 가 밴드 길이를 넘으면 **마지막 밴드 유지**."""
    idx = np.minimum(np.arange(len(score)), len(delta) - 1)
    cross = np.where(score > delta[idx])[0]
    return int(cross[0]) if len(cross) else None


# =============================================================================
# 지표
# =============================================================================

def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """rank AUROC (tie 평균). y=1 이 positive(=failure)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    s = np.zeros(cnt.size)
    np.add.at(s, inv, ranks)
    ranks = (s / cnt)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _med(v: list[float]) -> float | None:
    return round(float(np.median(v)), 4) if v else None


def td_auroc(pairs: list[tuple[float, int]]) -> tuple[float | None, int]:
    """(score, y) 목록 → AUROC + 표본 수. 단일 클래스면 None."""
    if not pairs:
        return None, 0
    sc = np.array([p[0] for p in pairs], dtype=np.float64)
    yy = np.array([p[1] for p in pairs], dtype=np.int64)
    a = auroc(sc, yy)
    return (None if a is None else round(a, 4)), len(pairs)


def td_pairs(test_scores: list[tuple[Episode, np.ndarray]], t_d: int
             ) -> list[tuple[float, int]]:
    """t_d 시점의 causal score. **t_d 에 이미 끝난 판(T ≤ t_d)은 제외** — 종료 자체가
    성공 신호라 포함하면 길이 confound 를 AUROC 에 되돌려 넣는 꼴이다."""
    return [(float(sc[t_d]), e.y) for e, sc in test_scores if e.T > t_d]


def td_metrics(test_scores: list[tuple[Episode, np.ndarray]]) -> dict:
    out: dict = {}
    for t_d in TD_GRID:
        a, n = td_auroc(td_pairs(test_scores, t_d))
        out[f"auroc_td{t_d}"] = a
        out[f"n_td{t_d}"] = n
    return out


def timer_records(test_eps: list[Episode], task: str, W: int | None,
                  phase_names: dict) -> list[dict]:
    """무feature 기준선: t ≥ W 이면 발화. W 이후로 판이 이어지는가만 본다."""
    recs = []
    for e in test_eps:
        ft = int(W) if (W is not None and e.T > int(W)) else None
        recs.append({
            "task": task, "arm": "-", "model": "timer", "alpha": None,
            "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
            "succ": e.succ, "y": e.y, "T": e.T, "W": None if W is None else int(W),
            "fired": ft is not None, "t_fire": ft,
            "relpos": None if ft is None else round(ft / max(e.T - 1, 1), 4),
            "steps_before_end": None if ft is None else int(e.T - 1 - ft),
            "fire_phase": None if ft is None else
                          phase_names.get(int(e.phase[ft]), str(int(e.phase[ft]))),
            "max_score": None,
        })
    return recs


def fire_percentiles(records: list[dict]) -> dict:
    """발화 record 의 첫 발화시각 분포 (median 계열과 별개로 4분위를 남긴다)."""
    v = [float(r["t_fire"]) for r in records if r.get("fired") and r.get("t_fire") is not None]
    if not v:
        return {"t_fire_p25": None, "t_fire_p50": None, "t_fire_p75": None, "n_fired": 0}
    q = np.percentile(np.asarray(v, dtype=np.float64), [25, 50, 75])
    return {"t_fire_p25": round(float(q[0]), 2), "t_fire_p50": round(float(q[1]), 2),
            "t_fire_p75": round(float(q[2]), 2), "n_fired": len(v)}


def summarize(records: list[dict], phase_top: int = 4) -> dict:
    """episode 발화 기록 → TSV 한 행 분량의 지표."""
    fail = [r for r in records if r["y"] == 1]
    succ = [r for r in records if r["y"] == 0]
    f_fired = [r for r in fail if r["fired"]]
    s_fired = [r for r in succ if r["fired"]]
    dist: dict[str, int] = {}
    for r in f_fired:
        k = str(r["fire_phase"])
        dist[k] = dist.get(k, 0) + 1
    top = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:phase_top]
    la = auroc(np.array([r["T"] for r in records], dtype=np.float64),
               np.array([r["y"] for r in records]))
    # timer(=길이만 보는 기준선) 대비 조기성. W 는 판마다 task 별 값이 실려 있다.
    early = [r for r in f_fired if r.get("W") is not None and r["t_fire"] < r["W"]]
    lead = [float(r["W"] - r["t_fire"]) for r in f_fired if r.get("W") is not None]
    ws = sorted({r["W"] for r in records if r.get("W") is not None})
    # 시퀀스 수준(=판 끝까지 본) 분리도. causal 하지 않으므로 조기 검출 근거가 아니라
    # "길이만 학습했는지" 대조용 (length_auroc 와 나란히 읽는다).
    ms = [r["max_score"] for r in records]
    msa = (auroc(np.array(ms, dtype=np.float64), np.array([r["y"] for r in records]))
           if all(v is not None for v in ms) and records else None)
    return {
        "n_test_ep": len(records),
        "n_fail": len(fail), "n_succ": len(succ),
        "tpr": round(len(f_fired) / len(fail), 4) if fail else None,
        "fpr": round(len(s_fired) / len(succ), 4) if succ else None,
        "tpr_before_W": round(len(early) / len(fail), 4) if fail and ws else None,
        "lead_vs_W": _med(lead),
        "W": ws[0] if len(ws) == 1 else ("|".join(map(str, ws)) if ws else None),
        "median_relpos_fail": _med([r["relpos"] for r in f_fired]),
        "median_steps_before_end_fail": _med([r["steps_before_end"] for r in f_fired]),
        "median_relpos_fp": _med([r["relpos"] for r in s_fired]),
        "fire_phase_dist": "|".join(f"{k}:{v}" for k, v in top),
        "length_auroc": None if la is None else round(la, 4),
        "maxscore_auroc": None if msa is None else round(msa, 4),
    }


# =============================================================================
# arm 실행
# =============================================================================

def run_arm(arm: str, kind: str, tasks: dict, splits: dict, args,
            out_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """arm × model 하나 학습 + 발화 시뮬. 반환 = (요약행, 상세기록, 체크포인트 payload들)."""
    alphas = args.alphas
    rows: list[dict] = []
    detail: list[dict] = []
    ckpts: dict[str, dict] = {}
    td_pool: dict[int, list[tuple[float, int]]] = {t_d: [] for t_d in TD_GRID}

    groups = [("__all__", sorted(tasks))] if arm == "mixed" else \
             [(t, [t]) for t in sorted(tasks)]

    for gname, gtasks in groups:
        tr_eps = [e for t in gtasks for e in splits[t]["train"]]
        if not tr_eps:
            print(f"  [skip] {arm}/{kind}/{gname}: train 판 0", flush=True)
            continue
        mu, sd = standardizer(tr_eps)
        seqs = [(apply_std(e, mu, sd), e.y) for e in tr_eps]
        input_dim = seqs[0][0].shape[1]
        n_fail_tr = sum(1 for _, y in seqs if y == 1)
        print(f"  [train] arm={arm} model={kind} group={gname} "
              f"n_train={len(seqs)} (fail {n_fail_tr}) dim={input_dim}", flush=True)
        if n_fail_tr == 0 or n_fail_tr == len(seqs):
            print(f"  [skip] {arm}/{kind}/{gname}: train 이 단일 클래스", flush=True)
            for t in gtasks:
                rows.append({"task": t, "arm": arm, "model": kind, "alpha": None,
                             "skip_reason": "train single-class"})
            continue
        model = train_detector(kind, seqs, input_dim, args.epochs, args.lr, args.hidden,
                               args.lambda_reg, args.grad_clip, args.batch_size,
                               args.seed, verbose=not args.quiet)

        bands_ck: dict[str, dict] = {}
        for t in gtasks:
            sp = splits[t]
            band_src = [e for e in sp["train"] if e.succ == 1] if args.band_mu == "train" \
                else [e for e in sp["calib"] if e.succ == 1]
            calib_src = [e for e in sp["calib"] if e.succ == 1]
            band_scores = [score_seq(model, apply_std(e, mu, sd)) for e in band_src]
            calib_scores = [score_seq(model, apply_std(e, mu, sd)) for e in calib_src]
            test_eps = sp["test"]
            if not test_eps:
                rows.append({"task": t, "arm": arm, "model": kind, "alpha": None,
                             "skip_reason": "test 판 0"})
                continue
            test_scores = [(e, score_seq(model, apply_std(e, mu, sd))) for e in test_eps]
            tdm = td_metrics(test_scores)          # α 무관 (밴드 안 씀) — 행마다 동일값
            for t_d in TD_GRID:
                td_pool[t_d].extend(td_pairs(test_scores, t_d))
            W_task = tasks[t].get("W")

            band_ck: dict[str, dict] = {}
            for a in alphas:
                band = functional_cp_band(band_scores, calib_scores, a)
                if band is None:
                    rows.append({
                        "task": t, "arm": arm, "model": kind, "alpha": a,
                        "skip_reason": f"성공 판 부족 (band {len(band_scores)} / "
                                       f"calib {len(calib_scores)} < {MIN_BAND_EPS})"})
                    continue
                recs = []
                for e, sc in test_scores:
                    ft = fire_step(sc, band["delta"])
                    rec = {
                        "task": t, "arm": arm, "model": kind, "alpha": a,
                        "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
                        "succ": e.succ, "y": e.y, "T": e.T,
                        "W": None if W_task is None else int(W_task),
                        "fired": ft is not None,
                        "t_fire": ft,
                        "relpos": None if ft is None else
                                  round(ft / max(e.T - 1, 1), 4),
                        "steps_before_end": None if ft is None else int(e.T - 1 - ft),
                        "fire_phase": None if ft is None else
                                      tasks[t]["phase_names"].get(int(e.phase[ft]),
                                                                  str(int(e.phase[ft]))),
                        "max_score": round(float(sc.max()), 4),
                    }
                    recs.append(rec)
                    detail.append(rec)
                row = {"task": t, "instruction": tasks[t]["instruction"], "arm": arm,
                       "model": kind, "alpha": a, "band_L": band["L"],
                       "bw": round(band["bw"], 4),
                       "n_train_ep": len(seqs), "n_band_succ": len(band_scores),
                       "n_calib_succ": len(calib_scores),
                       "train_scenes": ",".join(map(str, sp["scenes"]["train"])),
                       "calib_scenes": ",".join(map(str, sp["scenes"]["calib"])),
                       "test_scenes": ",".join(map(str, sp["scenes"]["test"])),
                       "truncate": args.truncate_train,
                       "n_trunc_dropped": tasks[t].get("n_trunc_dropped", 0),
                       "skip_reason": ""}
                row.update(summarize([r for r in recs]))
                row.update(tdm)
                rows.append(row)
                band_ck[f"{a:.2f}"] = {"mu": band["mu"].astype(np.float32),
                                       "sd": band["sd"].astype(np.float32),
                                       "bw": band["bw"],
                                       "delta": band["delta"].astype(np.float32)}
            if band_ck:
                bands_ck[t] = band_ck

        ckpts[gname] = {
            "arm": arm, "model": kind, "group": gname,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": input_dim, "hidden": args.hidden,
            "std_mean": mu, "std_std": sd,
            "feature": {"layer": args.layer, "denoise": args.denoise, "seg": args.seg,
                        "layer_idx": tasks[gtasks[0]]["layer_idx"],
                        "denoise_idx": tasks[gtasks[0]]["denoise_idx"],
                        "seg_idx": tasks[gtasks[0]]["seg_idx"],
                        "dim": input_dim},
            "cp_bands": bands_ck,
            "tasks": gtasks,
            "shards": [tasks[t]["shard"] for t in gtasks],   # basename 만 (docs/04 §8)
            "train": {"epochs": args.epochs, "lr": args.lr, "lambda_reg": args.lambda_reg,
                      "grad_clip": args.grad_clip, "batch_size": args.batch_size,
                      "seed": args.seed, "band_mu": args.band_mu},
            "truncate": {
                "mode": args.truncate_train,
                "rollout_W": {t: tasks[t].get("W") for t in gtasks},
                "phase_caps": {t: tasks[t].get("phase_caps") for t in gtasks},
                "n_dropped": {t: tasks[t].get("n_trunc_dropped", 0) for t in gtasks},
            },
        }

    # 풀링 행 (arm/model/α 별 전체 test 합산)
    pooled_td = {}
    for t_d in TD_GRID:
        a_, n_ = td_auroc(td_pool[t_d])
        pooled_td[f"auroc_td{t_d}"], pooled_td[f"n_td{t_d}"] = a_, n_
    for a in alphas:
        recs = [r for r in detail if r["alpha"] == a]
        if recs:
            row = {"task": "__pooled__", "instruction": "", "arm": arm, "model": kind,
                   "alpha": a, "truncate": args.truncate_train, "skip_reason": ""}
            row.update(summarize(recs))
            row.update(pooled_td)
            rows.append(row)

    for gname, payload in ckpts.items():
        slug = "all" if gname == "__all__" else gname
        torch.save(payload, out_dir / f"detector_{arm}_{kind}_{slug}.pt")
    return rows, detail, ckpts


def run_loto(kind: str, tasks: dict, splits: dict, args,
             out_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """leave-one-task-out (zero-shot task 전이) arm.

    질문: **학습에 없던 task 에서 detector 가 작동하나, phase 단위 길이 절제가 전이를
    살리나** (SAFE 논문의 unseen 전이 주장 재검 — 우리 seen18 재현은 unseen 0.434).

    fold 규약 (held-out task h 하나당 1 fold):
      학습   = h 를 뺀 나머지 task 들의 **train scene** episode (기존 scene split 재사용).
      절제   = train task 별 자기 W/phase cap (mixed arm 과 동일 — cap 은 task-로컬).
               h 는 **항상 full** (절제도, 학습·보정도 없음).
      표준화 = train(=나머지 task) 통계.
      CP밴드 = h 데이터를 쓸 수 없으므로 **train task 들의 성공 판을 전부 풀링**해
               μ_t/σ_t (band_mu 출처 규약 동일) · bw(calib 성공 풀) 를 잡는다.
               시간축은 절대 record index, 밴드 길이 초과 시 마지막값 유지(plateau).
      test   = h 의 **전 episode** (train/calib/test 분할 무시 = 한 판도 안 쓴 zero-shot).

    W(=timer 기준선·tpr_before_W·lead_vs_W) 는 h 자신의 성공 길이 통계라 zero-shot 이
    아니다 — **oracle 참조용 지표**로만 읽는다 (열 이름은 다른 arm 과 맞추려고 그대로).
    """
    alphas = args.alphas
    rows: list[dict] = []
    detail: list[dict] = []
    ckpts: dict[str, dict] = {}
    td_pool: dict[int, list[tuple[float, int]]] = {t_d: [] for t_d in TD_GRID}
    timer_all: list[dict] = []

    all_tasks = sorted(tasks)
    if len(all_tasks) < 2:
        raise SystemExit("--arm loto 는 task 2개 이상 필요")

    for held in all_tasks:
        tr_tasks = [t for t in all_tasks if t != held]
        tr_eps = [e for t in tr_tasks for e in splits[t]["train"]]
        test_eps = tasks[held].get("all_eps")
        if test_eps is None:
            raise SystemExit("loto: held-out 전 episode 사본이 없다 (run() 배선 확인)")
        base_row = {"task": held, "instruction": tasks[held]["instruction"],
                    "arm": "loto", "model": kind, "truncate": args.truncate_train}
        if not tr_eps:
            rows.append({**base_row, "alpha": None, "skip_reason": "train 판 0"})
            continue
        if not test_eps:
            rows.append({**base_row, "alpha": None, "skip_reason": "test 판 0"})
            continue

        mu, sd = standardizer(tr_eps)
        seqs = [(apply_std(e, mu, sd), e.y) for e in tr_eps]
        input_dim = seqs[0][0].shape[1]
        n_fail_tr = sum(1 for _, y in seqs if y == 1)
        print(f"  [train] arm=loto model={kind} heldout={held} "
              f"n_train={len(seqs)} (fail {n_fail_tr}, {len(tr_tasks)} task) "
              f"n_test={len(test_eps)} dim={input_dim}", flush=True)
        if n_fail_tr == 0 or n_fail_tr == len(seqs):
            print(f"  [skip] loto/{kind}/{held}: train 이 단일 클래스", flush=True)
            rows.append({**base_row, "alpha": None, "skip_reason": "train single-class"})
            continue

        model = train_detector(kind, seqs, input_dim, args.epochs, args.lr, args.hidden,
                               args.lambda_reg, args.grad_clip, args.batch_size,
                               args.seed, verbose=not args.quiet)

        # CP 밴드 출처 = train task 풀링 (held-out 판은 단 한 개도 안 쓴다).
        band_src = [e for t in tr_tasks for e in splits[t]["train"] if e.succ == 1] \
            if args.band_mu == "train" else \
            [e for t in tr_tasks for e in splits[t]["calib"] if e.succ == 1]
        calib_src = [e for t in tr_tasks for e in splits[t]["calib"] if e.succ == 1]
        band_scores = [score_seq(model, apply_std(e, mu, sd)) for e in band_src]
        calib_scores = [score_seq(model, apply_std(e, mu, sd)) for e in calib_src]

        test_scores = [(e, score_seq(model, apply_std(e, mu, sd))) for e in test_eps]
        tdm = td_metrics(test_scores)              # α 무관 (밴드 안 씀)
        for t_d in TD_GRID:
            td_pool[t_d].extend(td_pairs(test_scores, t_d))
        W_task = tasks[held].get("W")              # oracle 참조용 (아래 docstring 참조)

        band_ck: dict[str, dict] = {}
        for a in alphas:
            band = functional_cp_band(band_scores, calib_scores, a)
            if band is None:
                rows.append({**base_row, "alpha": a,
                             "skip_reason": f"성공 판 부족 (band {len(band_scores)} / "
                                            f"calib {len(calib_scores)} < {MIN_BAND_EPS})"})
                continue
            recs = []
            for e, sc in test_scores:
                ft = fire_step(sc, band["delta"])
                rec = {
                    "task": held, "arm": "loto", "model": kind, "alpha": a,
                    "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
                    "succ": e.succ, "y": e.y, "T": e.T,
                    "W": None if W_task is None else int(W_task),
                    "fired": ft is not None,
                    "t_fire": ft,
                    "relpos": None if ft is None else round(ft / max(e.T - 1, 1), 4),
                    "steps_before_end": None if ft is None else int(e.T - 1 - ft),
                    "fire_phase": None if ft is None else
                                  tasks[held]["phase_names"].get(int(e.phase[ft]),
                                                                 str(int(e.phase[ft]))),
                    "max_score": round(float(sc.max()), 4),
                }
                recs.append(rec)
                detail.append(rec)
            row = {**base_row, "alpha": a, "band_L": band["L"],
                   "bw": round(band["bw"], 4),
                   "n_train_ep": len(seqs), "n_band_succ": len(band_scores),
                   "n_calib_succ": len(calib_scores),
                   # loto 는 scene 이 아니라 task 로 fold 를 가른다 — 열 이름은 공용
                   "train_scenes": f"{len(tr_tasks)}tasks",
                   "calib_scenes": f"{len(tr_tasks)}tasks",
                   "test_scenes": "all(heldout)",
                   "n_trunc_dropped": sum(tasks[t].get("n_trunc_dropped", 0)
                                          for t in tr_tasks),
                   "skip_reason": ""}
            row.update(summarize(recs))
            row.update(tdm)
            rows.append(row)
            band_ck[f"{a:.2f}"] = {"mu": band["mu"].astype(np.float32),
                                   "sd": band["sd"].astype(np.float32),
                                   "bw": band["bw"],
                                   "delta": band["delta"].astype(np.float32)}

        # 같은 fold 의 무feature 기준선 (held-out 전 episode 위).
        trecs = timer_records(test_eps, held, W_task, tasks[held]["phase_names"])
        for r in trecs:
            r["arm"] = "loto"
        if trecs:
            timer_all += trecs
            detail += trecs
            trow = {**base_row, "alpha": None, "model": "timer", "skip_reason": ""}
            trow.update(summarize(trecs))
            rows.append(trow)

        ckpts[held] = {
            "arm": "loto", "model": kind, "group": held,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": input_dim, "hidden": args.hidden,
            "std_mean": mu, "std_std": sd,
            "feature": {"layer": args.layer, "denoise": args.denoise, "seg": args.seg,
                        "layer_idx": tasks[held]["layer_idx"],
                        "denoise_idx": tasks[held]["denoise_idx"],
                        "seg_idx": tasks[held]["seg_idx"],
                        "dim": input_dim},
            "cp_bands": {held: band_ck} if band_ck else {},
            "tasks": tr_tasks,                 # 학습에 쓴 task (held-out 은 제외)
            "heldout_task": held,
            "shards": [tasks[t]["shard"] for t in tr_tasks],   # basename 만 (docs/04 §8)
            "train": {"epochs": args.epochs, "lr": args.lr, "lambda_reg": args.lambda_reg,
                      "grad_clip": args.grad_clip, "batch_size": args.batch_size,
                      "seed": args.seed, "band_mu": args.band_mu,
                      "band_source": "pooled train tasks (zero-shot)"},
            "truncate": {
                "mode": args.truncate_train,
                "rollout_W": {t: tasks[t].get("W") for t in tr_tasks},
                "phase_caps": {t: tasks[t].get("phase_caps") for t in tr_tasks},
                "n_dropped": {t: tasks[t].get("n_trunc_dropped", 0) for t in tr_tasks},
                "heldout": "full (절제 없음)",
            },
        }

    # 풀링 행 (전 fold 합산)
    pooled_td = {}
    for t_d in TD_GRID:
        a_, n_ = td_auroc(td_pool[t_d])
        pooled_td[f"auroc_td{t_d}"], pooled_td[f"n_td{t_d}"] = a_, n_
    for a in alphas:
        recs = [r for r in detail if r["alpha"] == a]
        if recs:
            row = {"task": "__pooled__", "instruction": "", "arm": "loto", "model": kind,
                   "alpha": a, "truncate": args.truncate_train, "skip_reason": ""}
            row.update(summarize(recs))
            row.update(pooled_td)
            rows.append(row)
    if timer_all:
        row = {"task": "__pooled__", "instruction": "", "arm": "loto", "model": "timer",
               "alpha": None, "truncate": args.truncate_train, "skip_reason": ""}
        row.update(summarize(timer_all))
        rows.append(row)

    for held, payload in ckpts.items():
        torch.save(payload, out_dir / f"detector_loto_{kind}_{held}.pt")
    return rows, detail, ckpts


# =============================================================================
# arm: loko-cell (scene-local leave-one-jitter-out, 셀별 detector)
# =============================================================================

REGISTRY_COLS = ("instruction", "slug", "scene", "jitter", "registered",
                 "n_pool_other", "n_pool_fail", "n_target_fail", "n_target_succ",
                 "n_succ_calib", "reason", "ckpt_rel")

LOKO_EVAL_SETS = (
    # (이름, in_train) — in_train=1 은 그 집합이 학습에 들어갔다는 뜻(=상한 지표).
    ("target_j_fail", 1),
    ("target_j_succ", 0),
    ("pool_other", 1),
)


def _jstrat_auroc(scene_scored: list[tuple[Episode, np.ndarray]]) -> dict:
    """j 별 고정시각(JSTRAT_TD) AUROC. 성공·실패가 공존하지 않는 j 는 채점 제외."""
    by_j: dict[int, list[tuple[Episode, np.ndarray]]] = {}
    for e, sc in scene_scored:
        by_j.setdefault(int(e.jitter), []).append((e, sc))
    vals: dict[int, float] = {}
    unscored = 0
    for j2 in sorted(by_j):
        a, _n = td_auroc(td_pairs(by_j[j2], JSTRAT_TD))
        if a is None:                      # 단일 클래스(또는 t_d 까지 남은 판 없음)
            unscored += 1
        else:
            vals[j2] = float(a)
    return {
        f"auroc_td{JSTRAT_TD}_jstrat_mean":
            round(float(np.mean(list(vals.values()))), 4) if vals else None,
        "n_j_scored": len(vals), "n_j_unscored": unscored,
        "jstrat_detail": "|".join(f"j{j2}:{v:.2f}" for j2, v in sorted(vals.items())),
    }


def run_loko_cells(kind: str, tasks: dict, args, out_dir: Path,
                   cells: list[tuple[str, int, int]]
                   ) -> tuple[list[dict], list[dict], list[dict]]:
    """셀 (slug, scene s, jitter j) 하나마다 독립 detector + 발화 시뮬.

    학습 pool = 같은 scene 의 **다른 j 전판**(pool_other) + **대상 j 의 실패판**.
    대상 j 의 성공판은 학습에서 뺀다 (success-blind) — 위양성을 그 판으로 재기 때문.
    CP 밴드는 pool_other 성공판 위에서 LOO/k-fold (`loo_cp_band`).
    절제(`--truncate-train`)는 학습 pool 에만, 평가 시퀀스는 항상 full.

    반환 = (요약행, 상세기록, registry 행). registry 는 **미등록 셀도 사유와 함께**
    한 줄씩 남긴다 (무음 탈락 금지).
    """
    rows: list[dict] = []
    detail: list[dict] = []
    registry: list[dict] = []
    loko_root = out_dir / "loko"

    # 안전장치: 대상 slug 의 shard 에 지터 축이 실제로 있나 (oven/washer 는 legacy
    # jitter_reset_idx 가 전부 0 이라 셀이 전부 한 칸으로 뭉개진다).
    for slug in sorted({c[0] for c in cells}):
        info = tasks.get(slug)
        if info is None:
            continue
        jvals = {int(e.jitter) for e in info["all_eps"]}
        if len(jvals) <= 1:
            raise SystemExit(
                f"[loko] {slug}: jitter 축 없음 (값 {sorted(jvals)}) — v6 jitter_idx 열로 "
                "추출된 shard 인지 확인(oven/washer 는 jitter_reset_idx 가 전부 0)")

    for slug, s, j in cells:
        info = tasks.get(slug)
        base_reg = {"instruction": (info or {}).get("instruction", slug), "slug": slug,
                    "scene": s, "jitter": j, "registered": 0, "n_pool_other": 0,
                    "n_pool_fail": 0, "n_target_fail": 0, "n_target_succ": 0,
                    "n_succ_calib": 0, "reason": "", "ckpt_rel": ""}
        base_row = {"task": slug, "instruction": (info or {}).get("instruction", slug),
                    "arm": "loko-cell", "model": kind, "truncate": args.truncate_train,
                    "scene": s, "jitter": j}
        if info is None:
            base_reg["reason"] = "shard_not_loaded"
            registry.append(base_reg)
            rows.append({**base_row, "alpha": None, "skip_reason": "shard_not_loaded"})
            print(f"[loko] {slug} s{s} j{j}: shard 미로드 — 건너뜀", flush=True)
            continue

        eps = info["all_eps"]
        scene_eps = [e for e in eps if e.scene == s]
        cell_eps = [e for e in scene_eps if e.jitter == j]
        pool_other = [e for e in scene_eps if e.jitter != j]
        t_fail = [e for e in cell_eps if e.succ == 0]
        t_succ = [e for e in cell_eps if e.succ == 1]
        n_pool_fail = sum(1 for e in pool_other if e.succ == 0)
        n_calib_succ = sum(1 for e in pool_other if e.succ == 1)
        train_pool = pool_other + t_fail
        base_reg.update({"n_pool_other": len(pool_other), "n_pool_fail": n_pool_fail,
                         "n_target_fail": len(t_fail), "n_target_succ": len(t_succ),
                         "n_succ_calib": n_calib_succ})

        # ---- 게이트 (순서 고정) --------------------------------------------
        reason = ""
        if not cell_eps:
            reason = "cell_empty"
        elif n_pool_fail < int(args.min_pool_fail):
            reason = f"pool_fail<{int(args.min_pool_fail)}"
        elif len({e.y for e in train_pool}) < 2:
            reason = "single-class"
        elif n_calib_succ < int(args.min_calib_succ):
            # conformal (1−α) 분위가 정의되려면 n ≥ 1/α − 1 (α=0.1 → 9).
            reason = f"calib_succ<{int(args.min_calib_succ)}"
        if reason:
            base_reg["reason"] = reason
            registry.append(base_reg)
            rows.append({**base_row, "alpha": None, "skip_reason": reason})
            print(f"[loko] {slug} s{s} j{j}: 미등록 ({reason}) "
                  f"pool={len(pool_other)}(fail {n_pool_fail}) "
                  f"target={len(t_fail)}F/{len(t_succ)}S", flush=True)
            continue

        # ---- 절제 (학습 pool 의 성공 판 기준; 평가는 항상 full) --------------
        W = rollout_cap(train_pool)
        caps = phase_dwell_caps(train_pool)
        tr_eps: list[Episode] = []
        n_drop = 0
        for e in train_pool:
            te = truncate_episode(e, args.truncate_train, W, caps)
            if te is None:
                n_drop += 1
            else:
                tr_eps.append(te)
        if len({e.y for e in tr_eps}) < 2:
            base_reg["reason"] = "single-class(trunc)"
            registry.append(base_reg)
            rows.append({**base_row, "alpha": None, "skip_reason": "single-class(trunc)"})
            print(f"[loko] {slug} s{s} j{j}: 미등록 (절제 후 단일 클래스)", flush=True)
            continue

        mu, sd = standardizer(tr_eps)
        seqs = [(apply_std(e, mu, sd), e.y) for e in tr_eps]
        input_dim = seqs[0][0].shape[1]
        train_ep_ids = sorted({int(e.ep_id) for e in tr_eps})
        print(f"[loko] {slug} s{s} j{j}: train {len(seqs)} "
              f"(pool_other {len(pool_other)} + target_fail {len(t_fail)}, "
              f"drop {n_drop}) | calib_succ {n_calib_succ} | W={W} dim={input_dim}",
              flush=True)
        model = train_detector(kind, seqs, input_dim, args.epochs, args.lr, args.hidden,
                               args.lambda_reg, args.grad_clip, args.batch_size,
                               args.seed, verbose=not args.quiet)

        # ---- CP 밴드: pool_other 성공 판(절제 반영본) 위 LOO/k-fold ----------
        calib_eps = [e for e in tr_eps if e.succ == 1]
        calib_scores = [score_seq(model, apply_std(e, mu, sd)) for e in calib_eps]
        bands = {a: loo_cp_band(calib_scores, a, args.cp_folds) for a in args.alphas}
        cp_kind = "loo" if int(args.cp_folds) <= 0 else f"kfold-{int(args.cp_folds)}"

        # ---- 평가 (full 시퀀스) ---------------------------------------------
        sc_map = {int(e.ep_id): score_seq(model, apply_std(e, mu, sd)) for e in scene_eps}
        jstrat = _jstrat_auroc([(e, sc_map[int(e.ep_id)]) for e in scene_eps])
        groups = {"target_j_fail": t_fail, "target_j_succ": t_succ,
                  "pool_other": pool_other}

        band_ck: dict[str, dict] = {}
        for a in args.alphas:
            band = bands[a]
            if band is None:
                rows.append({**base_row, "alpha": a,
                             "skip_reason": f"성공 판 부족 (calib {len(calib_scores)} "
                                            f"< {MIN_BAND_EPS})"})
                continue
            band_ck[f"{a:.2f}"] = {"mu": band["mu"].astype(np.float32),
                                   "sd": band["sd"].astype(np.float32),
                                   "bw": band["bw"],
                                   "delta": band["delta"].astype(np.float32)}
            for name, in_train in LOKO_EVAL_SETS:
                group = groups[name]
                if not group:
                    rows.append({**base_row, "alpha": a, "eval_set": name,
                                 "in_train": in_train, "skip_reason": "eval 판 0"})
                    continue
                recs = []
                for e in group:
                    sc = sc_map[int(e.ep_id)]
                    ft = fire_step(sc, band["delta"])
                    rec = {
                        "task": slug, "arm": "loko-cell", "model": kind, "alpha": a,
                        "eval_set": name, "in_train": in_train,
                        "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
                        "jitter": e.jitter, "succ": e.succ, "y": e.y, "T": e.T,
                        "W": None if W is None else int(W),
                        "fired": ft is not None, "t_fire": ft,
                        "relpos": None if ft is None else round(ft / max(e.T - 1, 1), 4),
                        "steps_before_end": None if ft is None else int(e.T - 1 - ft),
                        "fire_phase": None if ft is None else
                                      info["phase_names"].get(int(e.phase[ft]),
                                                              str(int(e.phase[ft]))),
                        "max_score": round(float(sc.max()), 4),
                    }
                    recs.append(rec)
                    detail.append(rec)
                row = {**base_row, "alpha": a, "eval_set": name, "in_train": in_train,
                       "band_L": band["L"], "bw": round(band["bw"], 4),
                       "n_train_ep": len(seqs), "n_band_succ": len(calib_scores),
                       "n_calib_succ": len(calib_scores),
                       "train_scenes": f"s{s}/other-j", "calib_scenes": f"loo({cp_kind})",
                       "test_scenes": f"s{s}/j{j}" if name != "pool_other"
                                      else f"s{s}/other-j",
                       "n_trunc_dropped": n_drop, "skip_reason": ""}
                row.update(summarize(recs))
                row.update(td_metrics([(e, sc_map[int(e.ep_id)]) for e in group]))
                row.update(jstrat)
                row.update(fire_percentiles(recs))
                rows.append(row)

        # ---- timer 기준선 (무feature; W 는 학습 pool 성공 기준) ---------------
        for name, in_train in LOKO_EVAL_SETS:
            group = groups[name]
            if not group:
                continue
            trecs = timer_records(group, slug, W, info["phase_names"])
            for r in trecs:
                r["arm"] = "loko-cell"
                r["eval_set"] = name
                r["in_train"] = in_train
            detail += trecs
            trow = {**base_row, "alpha": None, "model": "timer", "eval_set": name,
                    "in_train": in_train, "skip_reason": ""}
            trow.update(summarize(trecs))
            trow.update(fire_percentiles(trecs))
            rows.append(trow)

        # ---- 체크포인트 + registry ------------------------------------------
        ck_rel = Path("loko") / slug / f"s{s}" / f"j{j}" / \
            f"detector_pertask_{kind}_{slug}.pt"
        ck_path = out_dir / ck_rel
        ck_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arm": "loko-cell", "model": kind, "group": slug,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": input_dim, "hidden": args.hidden,
            "std_mean": mu, "std_std": sd,
            "feature": {"layer": args.layer, "denoise": args.denoise, "seg": args.seg,
                        "layer_idx": info["layer_idx"],
                        "denoise_idx": info["denoise_idx"],
                        "seg_idx": info["seg_idx"], "dim": input_dim},
            # serve(`src/failure_online/online_failure.py`) 계약: task(=slug) → α → 밴드
            "cp_bands": {slug: band_ck} if band_ck else {},
            "tasks": [slug],
            "shards": [info["shard"]],                 # basename 만 (docs/04 §8)
            "train": {"epochs": args.epochs, "lr": args.lr, "lambda_reg": args.lambda_reg,
                      "grad_clip": args.grad_clip, "batch_size": args.batch_size,
                      "seed": args.seed, "band_mu": "loo(pool_other succ)"},
            "truncate": {"mode": args.truncate_train, "rollout_W": {slug: W},
                         "phase_caps": {slug: {str(k): v for k, v in caps.items()}},
                         "n_dropped": {slug: n_drop},
                         "eval": "full (절제 없음)"},
            "loko": {"instruction": info["instruction"], "slug": slug,
                     "scene": s, "jitter": j,
                     "n_pool_other": len(pool_other), "n_target_fail": len(t_fail),
                     "n_calib_succ": len(calib_scores), "cp": cp_kind,
                     "n_target_succ_excluded": len(t_succ),
                     "train_ep_ids": train_ep_ids},
        }
        torch.save(payload, ck_path)
        base_reg.update({"registered": 1, "reason": "",
                         "n_succ_calib": len(calib_scores),
                         "ckpt_rel": ck_rel.as_posix()})
        registry.append(base_reg)

    return rows, detail, registry


def write_registry(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(REGISTRY_COLS)]
    for r in rows:
        lines.append("\t".join("" if r.get(c) is None else str(r.get(c, ""))
                               for c in REGISTRY_COLS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# TSV
# =============================================================================

TSV_COLS = (["task", "instruction", "arm", "model", "alpha", "truncate", "n_test_ep",
             "n_fail", "n_succ", "tpr", "fpr", "tpr_before_W", "lead_vs_W", "W",
             "median_relpos_fail", "median_steps_before_end_fail", "median_relpos_fp",
             "fire_phase_dist", "length_auroc", "maxscore_auroc"]
            + [f"auroc_td{t}" for t in TD_GRID] + [f"n_td{t}" for t in TD_GRID]
            + ["band_L", "bw", "n_train_ep", "n_band_succ", "n_calib_succ",
               "n_trunc_dropped", "train_scenes", "calib_scenes", "test_scenes",
               "skip_reason"]
            # loko-cell 전용 (다른 arm 은 빈 칸). 기존 열 순서는 건드리지 않는다.
            + ["scene", "jitter", "eval_set", "in_train",
               f"auroc_td{JSTRAT_TD}_jstrat_mean", "n_j_scored", "n_j_unscored",
               "jstrat_detail", "t_fire_p25", "t_fire_p50", "t_fire_p75", "n_fired"])


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_COLS)]
    for r in rows:
        lines.append("\t".join("" if r.get(c) is None else str(r.get(c, ""))
                               for c in TSV_COLS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# self-test (합성 데이터 — 원격 실행 전 로컬 게이트)
# =============================================================================

def _write_synth_shard(path: Path, n_scene=10, n_noise=6, dim=32, seed=0,
                       onset=0.4, signal=1.6, sig_off=0, sig_rand_sign=False,
                       n_jitter=0, succ_jitter=None, instruction=None):
    """shard 계약과 동일한 NPZ 를 합성한다. 실패 판은 onset 이후 일부 축이 이동.

    `sig_off` = 실패 신호가 실리는 축 offset. task 마다 다르게 주면 **공유 신호 없음**
    (loto 전이 음성 대조).
    `sig_rand_sign` = 실패 이동의 부호를 판마다 무작위로. 다른 task 에서 학습한 고정
    readout 의 투영이 ± 대칭이 되어 held-out AUROC 가 기대값 0.5 로 모인다 (부호가
    고정이면 우연히 −방향에 걸려 0.5 에서 크게 벗어난다 — 음성 대조가 불안정).
    `n_jitter` = 0 이면 **jitter 열을 쓰지 않는다**(2축 legacy shard = 기존 케이스와
    바이트 단위로 같은 난수열). >0 이면 scene × jitter × noise 3축.
    `succ_jitter` = 그 j 는 항상 성공 (j-층화 채점에서 단일 클래스 j 를 만들기 위한 장치).
    """
    rng = np.random.default_rng(seed)
    layers = [0, 2, 4, 8, 10, 12, 15]
    K, S = 4, 4
    Xs, ep_id, scene, noise, rec_idx, succ, phase, ep_len = [], [], [], [], [], [], [], []
    jitter: list[np.ndarray] = []
    codebook = {"reach": 0, "grasp": 1, "transport": 2, "release": 3}
    e = 0
    jits = list(range(n_jitter)) if n_jitter > 0 else [None]
    for s in range(n_scene):
        for jt in jits:
            for nz in range(n_noise):
                fail = bool(rng.random() < 0.45)
                if jt is not None and succ_jitter is not None and jt == int(succ_jitter):
                    fail = False
                T = int(rng.integers(18, 26)) if fail else int(rng.integers(10, 18))
                base = rng.normal(0, 1, size=(T, dim)).astype(np.float32)
                base += rng.normal(0, 0.5, size=(1, dim)).astype(np.float32)  # scene 효과
                if fail:
                    t0 = int(onset * T)
                    sgn = -1.0 if (sig_rand_sign and rng.random() < 0.5) else 1.0
                    base[t0:, sig_off:sig_off + 4] += sgn * signal
                blk = np.zeros((T, len(layers), K, S, dim), dtype=np.float16)
                # layer 12 × denoise 3 × seg "all" 만 신호를 담고 나머지는 잡음
                blk[:] = rng.normal(0, 1, size=blk.shape).astype(np.float16)
                blk[:, layers.index(12), K - 1, S - 1, :] = base.astype(np.float16)
                Xs.append(blk)
                ep_id.append(np.full(T, e, np.int32))
                scene.append(np.full(T, s, np.int16))
                noise.append(np.full(T, nz, np.int16))
                rec_idx.append(np.arange(T, dtype=np.int16))
                succ.append(np.full(T, 0 if fail else 1, np.int8))
                ph = np.minimum((np.arange(T) * 4) // T, 3).astype(np.int16)
                phase.append(ph)
                ep_len.append(np.full(T, T, np.int16))
                if jt is not None:
                    jitter.append(np.full(T, jt, np.int16))
                e += 1
    meta = {"instruction": instruction or path.stem, "capture_layers": layers,
            "segment_names": ["state", "future", "action", "all"],
            "phase_codebook": codebook, "denoise_k": K, "dim": dim, "tier": "segA"}
    arrs = dict(X=np.concatenate(Xs), ep_id=np.concatenate(ep_id),
                scene=np.concatenate(scene), noise=np.concatenate(noise),
                rec_idx=np.concatenate(rec_idx), succ=np.concatenate(succ),
                phase_code=np.concatenate(phase), ep_len=np.concatenate(ep_len),
                meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    if jitter:
        arrs["jitter"] = np.concatenate(jitter)
    np.savez(path, **arrs)


def _synth_case(root: Path, tag: str, seed: int, signal: float, onset: float,
                n_scene: int, n_noise: int, sig_offs=(0, 0),
                sig_rand_sign: bool = False, n_jitter: int = 0,
                succ_jitter=None) -> Path:
    """합성 shard 2개(=task 2개)를 담은 디렉터리. sig_offs 가 다르면 공유 신호 없음."""
    sd = root / f"segA_{tag}"
    sd.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("SynthTaskA", "SynthTaskB")):
        _write_synth_shard(sd / f"{name}.npz", n_scene=n_scene, n_noise=n_noise,
                           seed=seed + i, onset=onset, signal=signal,
                           sig_off=sig_offs[i], sig_rand_sign=sig_rand_sign,
                           n_jitter=n_jitter, succ_jitter=succ_jitter)
    return sd


def _synth_run(args, shard_dir: Path, out_dir: Path, truncate: str,
               epochs: int, arm: str = "pertask", extra: dict | None = None) -> list[dict]:
    """합성 케이스 1회 실행 → sim_rows.json 의 행 목록."""
    a = argparse.Namespace(**vars(args))
    a.shard_dir, a.out = shard_dir, out_dir
    a.truncate_train = truncate
    a.epochs = epochs
    a.self_test = False
    a.arm = arm
    a.models = "lstm"
    a.quiet = True
    a.train_scenes, a.calib_scenes, a.test_scenes = 8, 3, 3
    for k, v in (extra or {}).items():
        setattr(a, k, v)
    rc = run(a)
    if rc != 0:
        raise RuntimeError(f"self-test 하위 실행 실패 (truncate={truncate})")
    return json.loads((out_dir / "sim_rows.json").read_text())


def _read_tsv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def _write_cells_tsv(path: Path, rows: list[dict], cols: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(cols)]
    lines += ["\t".join(str(r.get(c, "")) for c in cols) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _selftest_cell_tsv_reader(root: Path) -> list[str]:
    """read_cell_tsv 계약 단위 점검 (torch 안 씀 — 즉시 끝난다)."""
    fails: list[str] = []
    known = {"PPCC_bread": "PPCC/bread", "SynthTaskA": "SynthTaskA"}
    d = root / "celltsv"

    # (1) instruction 에 '/' 가 있어도 slug 로 매칭 + jitter_idx 우선
    p1 = _write_cells_tsv(d / "ok.tsv",
                          [{"instruction": "PPCC/bread", "scene_idx": 3,
                            "jitter_idx": 2, "jitter_reset_idx": 2, "noise_idx": 0}],
                          ["instruction", "scene_idx", "jitter_idx",
                           "jitter_reset_idx", "noise_idx"])
    rows, by_slug = read_cell_tsv(p1, known, "(test1)")
    if not (len(rows) == 1 and rows[0]["slug"] == "PPCC_bread"
            and rows[0]["jitter"] == 2 and set(by_slug) == {"PPCC_bread"}):
        fails.append(f"(tsv) 슬래시 instruction 매칭 실패: {rows}")

    # (2) legacy jitter_reset_idx 단독 — 값이 두 종류 이상이면 허용
    p2 = _write_cells_tsv(d / "legacy.tsv",
                          [{"slug": "SynthTaskA", "scene_idx": 1,
                            "jitter_reset_idx": 4, "noise_idx": 2},
                           {"slug": "SynthTaskA", "scene_idx": 1,
                            "jitter_reset_idx": 0, "noise_idx": 3}],
                          ["slug", "scene_idx", "jitter_reset_idx", "noise_idx"])
    rows2, _ = read_cell_tsv(p2, known, "(test2)")
    if not (len(rows2) == 2 and rows2[0]["jitter"] == 4):
        fails.append(f"(tsv) legacy jitter_reset_idx 실패: {rows2}")

    # (2b) legacy 단독인데 값이 전부 같으면 지터 축이 없는 것 → fail-loud
    p2b = _write_cells_tsv(d / "legacy_flat.tsv",
                           [{"slug": "SynthTaskA", "scene_idx": 1,
                             "jitter_reset_idx": 0, "noise_idx": 2},
                            {"slug": "SynthTaskA", "scene_idx": 2,
                             "jitter_reset_idx": 0, "noise_idx": 3}],
                           ["slug", "scene_idx", "jitter_reset_idx", "noise_idx"])
    try:
        read_cell_tsv(p2b, known, "(test2b)")
        fails.append("(tsv) jitter_reset_idx 가 전부 0 인데 통과했다")
    except SystemExit:
        pass

    # (3) 두 열이 어긋나면 **jitter_idx 를 채택**하고 로그만 남긴다 (v6 실데이터 규약:
    #     oven/washer 는 jitter_reset_idx 가 전부 0 이라 불일치가 정상)
    p3 = _write_cells_tsv(d / "both.tsv",
                          [{"grid_instruction": "PPCC/bread", "slug": "PPCC_bread",
                            "scene_idx": 1, "jitter_idx": 3, "jitter_reset_idx": 0,
                            "noise_idx": 2, "machine": "kanu", "sig": "deadbeef"}],
                          ["grid_instruction", "slug", "machine", "scene_idx",
                           "jitter_idx", "noise_idx", "jitter_reset_idx", "sig"])
    rows3, _ = read_cell_tsv(p3, known, "(test3)")
    if not (len(rows3) == 1 and rows3[0]["jitter"] == 3
            and rows3[0]["slug"] == "PPCC_bread"
            and rows3[0]["instruction"] == "PPCC/bread"):
        fails.append(f"(tsv) jitter_idx 우선 채택 실패: {rows3}")

    # (4) 어느 shard 와도 매칭 안 되는 행은 fail-loud
    p4 = _write_cells_tsv(d / "unmatched.tsv",
                          [{"grid_instruction": "NoSuchTask", "scene_idx": 0,
                            "jitter_idx": 0, "noise_idx": 0}],
                          ["grid_instruction", "scene_idx", "jitter_idx", "noise_idx"])
    try:
        read_cell_tsv(p4, known, "(test4)")
        fails.append("(tsv) 미매칭 행인데 통과했다")
    except SystemExit:
        pass

    # (5) 지터 열이 아예 없으면 fail-loud
    p5 = _write_cells_tsv(d / "nojit.tsv",
                          [{"slug": "SynthTaskA", "scene_idx": 0, "noise_idx": 0}],
                          ["slug", "scene_idx", "noise_idx"])
    try:
        read_cell_tsv(p5, known, "(test5)")
        fails.append("(tsv) 지터 열 없는데 통과했다")
    except SystemExit:
        pass
    return fails


def _selftest_loo_band() -> list[str]:
    """loo_cp_band: bw 가 유한하고 α 가 커지면 단조 감소하는가."""
    fails: list[str] = []
    rng = np.random.default_rng(0)
    succ = [np.abs(rng.normal(0.3, 0.1, size=int(rng.integers(8, 15))))
            for _ in range(12)]
    prev = None
    got = []
    for a in (0.05, 0.1, 0.2, 0.3):
        band = loo_cp_band(succ, a, 0)
        if band is None:
            fails.append(f"(loo) α={a}: 밴드 None (표본 12판인데)")
            continue
        if not np.isfinite(band["bw"]) or not np.all(np.isfinite(band["delta"])):
            fails.append(f"(loo) α={a}: bw/delta 에 비유한값")
        if band["folds"] != 12 or band["n_cal"] != 12:
            fails.append(f"(loo) α={a}: folds/n_cal = {band['folds']}/{band['n_cal']} "
                         "(LOO 면 12/12 이어야)")
        got.append((a, round(band["bw"], 4)))
        if prev is not None and band["bw"] > prev + 1e-12:
            fails.append(f"(loo) α={a}: bw {band['bw']:.4f} > 이전 {prev:.4f} — "
                         "(1−α) 분위 단조성 위반")
        prev = band["bw"]
    if loo_cp_band(succ[:2], 0.1, 0) is not None:
        fails.append("(loo) 성공 판 2개인데 밴드가 나왔다 (MIN_BAND_EPS 위반)")
    kb = loo_cp_band(succ, 0.1, 4)
    if kb is None or kb["folds"] != 4:
        fails.append(f"(loo) k-fold(4) 밴드 실패: {None if kb is None else kb['folds']}")
    print(f"  (b) LOO 밴드 bw(α↑) : {got}  (단조 감소 기대)")
    return fails


def _pooled(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r.get("task") == "__pooled__" and r.get("model") != "timer"
            and r.get("tpr") is not None]


def self_test(args) -> int:
    """합성 데이터 게이트.

    (1) 기존 게이트 — 신호 있는 합성에서 TPR > FPR.
    (2) 길이 절제 게이트 — 합성 2종:
        (a) feature 신호 O + 길이 confound  → truncate=rollout 에서도 조기 AUROC 유지
        (b) feature 신호 X + 길이 confound  → truncate=rollout 이면 조기 AUROC ≈ 0.5,
            full 학습이면 (길이만 보고) 종반 검출은 살아난다.
    합성 잡음을 감안해 임계는 느슨하게 잡고, 어긋나면 이유를 찍고 fail-loud.
    """
    epochs = min(args.epochs, 8)
    t_early = 10          # (a) 의 onset(0.25·T ≈ 4~6) 이후이면서 W 보다 한참 앞
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sig_dir = _synth_case(root, "sig", args.seed, signal=2.0, onset=0.25,
                              n_scene=14, n_noise=8)
        len_dir = _synth_case(root, "lenonly", args.seed + 100, signal=0.0, onset=0.4,
                              n_scene=14, n_noise=8)

        print("\n=== [self-test 1/8] 신호 O · truncate=none (기존 게이트) ===")
        rows_a_none = _synth_run(args, sig_dir, root / "out_a_none", "none", epochs)
        print("\n=== [self-test 2/8] 신호 O · truncate=rollout ===")
        rows_a_tr = _synth_run(args, sig_dir, root / "out_a_rollout", "rollout", epochs)
        print("\n=== [self-test 3/8] 신호 X(길이만) · truncate=none ===")
        rows_b_none = _synth_run(args, len_dir, root / "out_b_none", "none", epochs)
        print("\n=== [self-test 4/8] 신호 X(길이만) · truncate=rollout ===")
        rows_b_tr = _synth_run(args, len_dir, root / "out_b_rollout", "rollout", epochs)

        def _td(rows) -> float | None:
            v = [r.get(f"auroc_td{t_early}") for r in _pooled(rows)
                 if r.get(f"auroc_td{t_early}") is not None]
            return float(v[0]) if v else None

        def _best_gap(rows) -> float | None:
            g = [r["tpr"] - (r["fpr"] or 0.0) for r in _pooled(rows)]
            return max(g) if g else None

        def _late(rows) -> float | None:
            v = [r.get("maxscore_auroc") for r in _pooled(rows)
                 if r.get("maxscore_auroc") is not None]
            return float(v[0]) if v else None

        fails: list[str] = []

        pooled = _pooled(rows_a_none)
        if not pooled:
            fails.append("(1) 신호 O/none: pooled 유효 행 0")
        print("\n[self-test] (1) 신호 O · none — pooled (TPR > FPR 이어야 통과)")
        for r in pooled:
            good = r["tpr"] > (r["fpr"] or 0.0)
            if not good and r["alpha"] <= 0.25:
                fails.append(f"(1) α={r['alpha']}: TPR {r['tpr']} ≤ FPR {r['fpr']}")
            print(f"  arm={r['arm']:7s} model={r['model']:4s} α={r['alpha']:.2f} "
                  f"TPR={r['tpr']} FPR={r['fpr']} relpos={r['median_relpos_fail']} "
                  f"{'OK' if good else 'FAIL'}")

        a_tr, b_tr, b_none = _td(rows_a_tr), _td(rows_b_tr), _td(rows_b_none)
        gap_b_none, gap_b_tr = _best_gap(rows_b_none), _best_gap(rows_b_tr)
        late_b_none, late_b_tr = _late(rows_b_none), _late(rows_b_tr)
        print(f"\n[self-test] 절제 게이트 (t_d={t_early} 조기 AUROC / 종반 maxscore AUROC)")
        print(f"  (a) 신호 O  rollout 절제 : auroc_td{t_early}={_fmt(a_tr)}  (> 0.65 기대)")
        print(f"  (b) 신호 X  rollout 절제 : auroc_td{t_early}={_fmt(b_tr)}  (≲ 0.65 기대)"
              f" | 종반={_fmt(late_b_tr)} TPR−FPR(best α)={_fmt(gap_b_tr)}")
        print(f"  (b) 신호 X  none        : auroc_td{t_early}={_fmt(b_none)}  "
              f"| 종반={_fmt(late_b_none)} "
              f"TPR−FPR(best α)={_fmt(gap_b_none)} (> 0.10 기대)")

        if a_tr is None:
            fails.append(f"(a) 신호 O/rollout: auroc_td{t_early} 계산 불가 (표본 부족)")
        elif a_tr <= 0.65:
            fails.append(f"(a) 신호 O/rollout: 조기 AUROC {a_tr} ≤ 0.65 — 절제가 "
                         "진짜 feature 신호까지 죽였다")
        if b_tr is None:
            fails.append(f"(b) 신호 X/rollout: auroc_td{t_early} 계산 불가")
        elif b_tr > 0.65:
            fails.append(f"(b) 신호 X/rollout: 조기 AUROC {b_tr} > 0.65 — 길이만 있는 "
                         "합성인데 조기 분리가 나온다 (절제 누수 의심)")
        # (b) 에 길이 confound 가 실제로 있고 종반에는 읽힌다는 것의 근거:
        #   ① timer(무feature 길이 기준선)가 (b) 에서 거의 완벽히 분리한다 — 데이터 sanity
        #   ② full 학습 detector 도 종반 발화로 양의 TPR−FPR 를 낸다 (조기 td 는 ≈0.5)
        timer_gap = None
        for r in rows_b_none:
            if r.get("task") == "__pooled__" and r.get("model") == "timer" \
                    and r.get("tpr") is not None:
                timer_gap = r["tpr"] - (r["fpr"] or 0.0)
        print(f"  (b) timer 기준선(길이만)  : TPR−FPR={_fmt(timer_gap)} (> 0.5 기대)")
        if timer_gap is None or timer_gap <= 0.5:
            fails.append(f"(b) timer 기준선 TPR−FPR={timer_gap} — 합성에 길이 confound "
                         "가 없다. 대조 자체가 성립하지 않는다")
        if gap_b_none is None:
            fails.append("(b) 신호 X/none: pooled 유효 행 0")
        elif gap_b_none <= 0.10:
            fails.append(f"(b) 신호 X/none: TPR−FPR {gap_b_none:.3f} ≤ 0.10 — full 학습 "
                         "detector 가 길이 신호조차 못 잡았다 (대조 무의미)")

        # ---- (c) loto (task 전이) 게이트 ----------------------------------
        # 공유 신호 O(두 합성 task 가 같은 축으로 실패) → held-out AUROC > 0.65,
        # 공유 신호 X(task 마다 다른 축) → ≈ 0.5. 전자만 되면 전이 배선이 살아있고,
        # 후자에서 높게 나오면 held-out 판이 학습·보정에 샜다는 뜻.
        nosh_dir = _synth_case(root, "loto_nosh", args.seed + 200, signal=2.0,
                               onset=0.25, n_scene=14, n_noise=8, sig_offs=(0, 8),
                               sig_rand_sign=True)
        print("\n=== [self-test 5/8] loto · 공유 신호 O ===")
        rows_c_sh = _synth_run(args, sig_dir, root / "out_c_shared", "none", epochs,
                               arm="loto")
        print("\n=== [self-test 6/8] loto · 공유 신호 X (다른 축) ===")
        rows_c_no = _synth_run(args, nosh_dir, root / "out_c_noshare", "none", epochs,
                               arm="loto")
        c_sh, c_no = _td(rows_c_sh), _td(rows_c_no)
        print(f"\n[self-test] loto 게이트 (t_d={t_early} held-out AUROC, pooled)")
        print(f"  (c) 공유 신호 O : {_fmt(c_sh)}  (> 0.65 기대)")
        print(f"  (c) 공유 신호 X : {_fmt(c_no)}  (0.35~0.65 기대)")
        if c_sh is None:
            fails.append("(c) loto/공유O: auroc 계산 불가 (표본 부족)")
        elif c_sh <= 0.65:
            fails.append(f"(c) loto/공유O: held-out AUROC {c_sh} ≤ 0.65 — 공유 신호가 "
                         "있는데 전이가 안 된다 (배선 의심)")
        if c_no is None:
            fails.append("(c) loto/공유X: auroc 계산 불가")
        elif not (0.35 <= c_no <= 0.65):
            fails.append(f"(c) loto/공유X: held-out AUROC {c_no} 이 0.35~0.65 밖 — "
                         "공유 신호 없는 합성인데 전이가 보인다 (held-out 누수 의심)")

        # ---- (d~g) loko-cell 게이트 ---------------------------------------
        # 3축(scene × jitter × noise) 합성. j=0 은 **항상 성공** → j-층화 채점에서
        # 반드시 제외되어야 하는 단일 클래스 j 가 결정적으로 생긴다.
        lk_dir = _synth_case(root, "loko", args.seed + 300, signal=2.0, onset=0.25,
                             n_scene=3, n_noise=8, n_jitter=4, succ_jitter=0)
        cells_cols = ["instruction", "scene_idx", "jitter_idx", "noise_idx"]
        cells = [
            # 같은 셀을 noise 2행으로 → dedupe 되어 1셀이어야 한다
            {"instruction": "SynthTaskA", "scene_idx": 0, "jitter_idx": 1, "noise_idx": 0},
            {"instruction": "SynthTaskA", "scene_idx": 0, "jitter_idx": 1, "noise_idx": 3},
            {"instruction": "SynthTaskA", "scene_idx": 1, "jitter_idx": 2, "noise_idx": 0},
            # 존재하지 않는 지터 → cell_empty 로 registry 에 남아야 한다
            {"instruction": "SynthTaskA", "scene_idx": 2, "jitter_idx": 9, "noise_idx": 0},
        ]
        cells_tsv = _write_cells_tsv(root / "loko_cells.tsv", cells, cells_cols)
        lk_out = root / "out_loko"
        print("\n=== [self-test 7/8] loko-cell · 셀별 detector ===")
        rows_lk = _synth_run(args, lk_dir, lk_out, "none", epochs, arm="loko-cell",
                             extra={"loko_cells_tsv": str(cells_tsv),
                                    "min_pool_fail": 3, "min_calib_succ": 6,
                                    "cp_folds": 0})
        print("\n=== [self-test 8/8] loko-cell · 게이트 미달(min-calib-succ 999) ===")
        lk_out2 = root / "out_loko_gate"
        _synth_run(args, lk_dir, lk_out2, "none", epochs, arm="loko-cell",
                   extra={"loko_cells_tsv": str(cells_tsv), "min_pool_fail": 3,
                          "min_calib_succ": 999, "cp_folds": 0})

        reg = _read_tsv_rows(lk_out / "cell_registry.tsv")
        reg2 = _read_tsv_rows(lk_out2 / "cell_registry.tsv")
        det = json.loads((lk_out / "sim_detail.json").read_text())
        print("\n[self-test] loko-cell 게이트")
        print(f"  (a) registry {len(reg)}행 — "
              + " | ".join(f"s{r['scene']}j{r['jitter']}:"
                           f"{'등록' if r['registered'] == '1' else r['reason']}"
                           for r in reg))

        # (a) 게이트 미달 셀이 사유와 함께 registry 에 남는가 (무음 탈락 금지)
        if len(reg) != 3:
            fails.append(f"(a) registry 행 {len(reg)} != 대상 셀 3 (noise dedupe 실패?)")
        empty = [r for r in reg if r["reason"] == "cell_empty"]
        if len(empty) != 1:
            fails.append(f"(a) cell_empty 행 {len(empty)}개 (존재하지 않는 j 하나 기대)")
        if not any(r["registered"] == "1" for r in reg):
            fails.append("(a) 등록된 셀이 하나도 없다 — 합성 표본/게이트 확인")
        bad2 = [r for r in reg2 if r["registered"] == "1"]
        if bad2:
            fails.append(f"(a) min_calib_succ=999 인데 등록된 셀 {len(bad2)}개")
        if not any(r["reason"].startswith("calib_succ<999") for r in reg2):
            fails.append(f"(a) calib_succ<999 사유 행이 없다: "
                         f"{[r['reason'] for r in reg2]}")

        # (c) 대상 j 성공판이 학습 pool 에 안 들어갔는가 (ep_id 집합 직접 대조)
        checked = 0
        for r in reg:
            if r["registered"] != "1":
                continue
            ck = torch.load(lk_out / r["ckpt_rel"], map_location="cpu",
                            weights_only=False)
            tr_ids = set(int(v) for v in ck["loko"]["train_ep_ids"])
            tgt_succ = {int(d["ep_id"]) for d in det["episodes"]
                        if d.get("eval_set") == "target_j_succ"
                        and d.get("task") == r["slug"]
                        and int(d.get("scene", -1)) == int(r["scene"])
                        and int(d.get("jitter", -1)) == int(r["jitter"])}
            if not tgt_succ:
                fails.append(f"(c) s{r['scene']}j{r['jitter']}: target_j_succ 판이 0 "
                             "— 검증이 공허하다")
                continue
            leak = tgt_succ & tr_ids
            checked += 1
            print(f"  (c) s{r['scene']}j{r['jitter']}: target_j_succ {len(tgt_succ)}판 "
                  f"∩ train {len(tr_ids)}판 = {len(leak)} (0 기대)")
            if leak:
                fails.append(f"(c) s{r['scene']}j{r['jitter']}: 대상 j 성공판 "
                             f"{sorted(leak)} 가 학습 pool 에 들어갔다")
        if checked == 0:
            fails.append("(c) 검증할 등록 셀이 없다")

        # (d) j-층화 AUROC 가 단일클래스 j(=항상 성공인 j0)를 제외하는가
        jrows = [r for r in rows_lk if r.get("arm") == "loko-cell"
                 and r.get("model") != "timer" and r.get("n_j_scored") is not None]
        if not jrows:
            fails.append("(d) j-층화 열이 붙은 행이 없다")
        for r in jrows[:1] or []:
            print(f"  (d) jstrat: mean={r.get(f'auroc_td{JSTRAT_TD}_jstrat_mean')} "
                  f"scored={r['n_j_scored']} unscored={r['n_j_unscored']} "
                  f"detail={r.get('jstrat_detail')}")
        for r in jrows:
            if int(r["n_j_unscored"]) < 1:
                fails.append(f"(d) s{r['scene']}j{r['jitter']}: 단일클래스 j0 가 있는데 "
                             f"n_j_unscored={r['n_j_unscored']}")
            if "j0:" in str(r.get("jstrat_detail") or ""):
                fails.append(f"(d) s{r['scene']}j{r['jitter']}: 항상 성공인 j0 가 "
                             f"채점됐다 ({r['jstrat_detail']})")
            if int(r["n_j_scored"]) < 1:
                fails.append(f"(d) s{r['scene']}j{r['jitter']}: 채점된 j 가 0")

        # in_train 표기 계약 (target_j_succ 만 학습 밖)
        for r in jrows:
            want = 0 if r.get("eval_set") == "target_j_succ" else 1
            if int(r["in_train"]) != want:
                fails.append(f"(d) eval_set={r.get('eval_set')} 의 in_train="
                             f"{r['in_train']} (기대 {want})")

        # ---- 셀 TSV 리더 · LOO 밴드 단위 점검 -------------------------------
        print("\n[self-test] 셀 TSV 리더 · LOO 밴드")
        fails += _selftest_cell_tsv_reader(root)
        fails += _selftest_loo_band()

        if fails:
            print("\n[self-test] FAIL")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("\n[self-test] PASS")
        return 0


# =============================================================================
# main
# =============================================================================

def run(args) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [s.strip() for s in args.shards.split(",") if s.strip()] if args.shards else None
    paths = discover_shards(shard_dir, only)

    # 셀 TSV 매칭용 slug ↔ instruction 지도. `--shards` 로 일부만 골랐어도 TSV 가 다른
    # shard 를 가리킬 수 있으므로 **디렉터리 전체** 로 만든다 (meta_json 만 읽는다).
    slug_instr = shard_slug_index(discover_shards(shard_dir, None))

    exclude_cells: set[tuple[str, int, int, int]] = set()
    exclude_cells_by_slug: dict[str, set] = {}
    if args.exclude_cells_tsv:
        ex_rows, _ = read_cell_tsv(args.exclude_cells_tsv, slug_instr,
                                   "--exclude-cells-tsv")
        for r in ex_rows:
            key = (r["slug"], r["scene"], r["jitter"], r["noise"])
            exclude_cells.add(key)
            exclude_cells_by_slug.setdefault(r["slug"], set()).add(key)
        print(f"[exclude] 셀 {len(exclude_cells)}개 로드 "
              f"({ {k: len(v) for k, v in sorted(exclude_cells_by_slug.items())} })",
              flush=True)

    loko_cells: list[tuple[str, int, int]] = []
    if args.arm == "loko-cell":
        if not args.loko_cells_tsv:
            raise SystemExit("--arm loko-cell 은 --loko-cells-tsv 가 필수")
        lk_rows, _ = read_cell_tsv(args.loko_cells_tsv, slug_instr, "--loko-cells-tsv")
        # 같은 (slug, scene, jitter) 가 noise 별로 여러 행이면 셀 단위로 dedupe.
        loko_cells = sorted({(r["slug"], r["scene"], r["jitter"]) for r in lk_rows})
        print(f"[loko] 셀 TSV 행 {len(lk_rows)} → 대상 셀 {len(loko_cells)}개 "
              f"(slug {len(set(c[0] for c in loko_cells))}종)", flush=True)

    tasks: dict[str, dict] = {}
    splits: dict[str, dict] = {}
    t0 = time.time()
    for p in paths:
        eps, spec = load_shard_episodes(p, args.layer, args.denoise, args.seg)
        slug = p.stem
        if args.arm == "loko-cell":
            # scene-local arm 은 scene 분할을 쓰지 않는다 (셀 = (instruction, scene, j)).
            # v6 scene 단위 shard 는 scene 열이 상수라 split_scenes 가 애초에 불가능하다.
            # W/dwell cap 은 아래에서 part["train"](=전 episode) 위에서 잡히고, 실제
            # 학습 pool 기준 재계산은 run_loko_cells 가 셀마다 따로 한다.
            sc_split = {"train": sorted({e.scene for e in eps}), "calib": [], "test": []}
            part = {"train": list(eps), "calib": [], "test": []}
        else:
            sc_split = split_scenes(slug, [e.scene for e in eps], args.train_scenes,
                                    args.calib_scenes, args.test_scenes, args.seed)
            part = {k: [e for e in eps if e.scene in set(v)] for k, v in sc_split.items()}
        # eval 대상 셀(slug, scene, jitter, noise)은 train/calib 에서 제외 — in-sample 방지.
        # scene 자체는 남긴다(45 §3: scene 노출 허용, held-out 은 episode 축). test 에
        # 떨어진 셀은 그대로 둔다(평가 자체가 목적).
        n_excl = {"train": 0, "calib": 0}
        if exclude_cells:
            for k in ("train", "calib"):
                before = len(part[k])
                part[k] = [e for e in part[k]
                           if (slug, e.scene, e.jitter, e.noise) not in exclude_cells]
                n_excl[k] = before - len(part[k])
            if exclude_cells_by_slug.get(slug) and sum(n_excl.values()) == 0 and \
                    not any(e.jitter >= 0 for e in eps):
                raise SystemExit(f"{slug}: --exclude-cells-tsv 지정됐으나 shard 에 jitter "
                                 "열이 없어 셀 매칭 불가 (2축 legacy shard)")
            print(f"[exclude] {slug}: eval 셀 {len(exclude_cells_by_slug.get(slug, ()))}개 "
                  f"→ train −{n_excl['train']} / calib −{n_excl['calib']}", flush=True)
        tasks[slug] = {
            "n_excluded": n_excl,
            "instruction": spec.instruction, "shard": p.name,
            "phase_names": spec.phase_names, "dim": spec.dim,
            "layer_idx": spec.layer_idx, "denoise_idx": spec.denoise_idx,
            "seg_idx": spec.seg_idx, "layers": spec.layers,
        }
        splits[slug] = {**part, "scenes": sc_split}
        if args.arm in ("loto", "loko-cell"):
            # held-out fold 의 test = 전 episode(절제 전 원본). 아래 apply_truncation 은
            # train/calib 리스트를 절제 사본으로 갈아끼우므로 여기서 원본을 붙들어 둔다.
            # loko-cell 은 scene split 자체를 쓰지 않고 이 원본 위에서 셀을 자른다.
            tasks[slug]["all_eps"] = eps
        # W/cap 은 truncate 모드와 무관하게 **항상** 계산한다 — timer 기준선·tpr_before_W
        # 가 모든 run 에서 필요하고, task 별(mixed arm 에서도 task 별)로 잡는다.
        W = rollout_cap(part["train"])
        caps = phase_dwell_caps(part["train"])
        tasks[slug]["W"] = W
        tasks[slug]["phase_caps"] = {str(k): v for k, v in caps.items()}
        dropped = apply_truncation(splits[slug], args.truncate_train, W, caps)
        tasks[slug]["n_trunc_dropped"] = int(sum(dropped.values()))
        if args.truncate_train != "none":
            print(f"[trunc] {slug}: mode={args.truncate_train} W={W} caps={caps} "
                  f"dropped(train/calib)={dropped['train']}/{dropped['calib']}", flush=True)
        elif W is not None:
            print(f"[trunc] {slug}: mode=none (W={W} 는 timer 기준선용으로만 사용)",
                  flush=True)
        sp = splits[slug]
        print(f"[load] {slug}: ep {len(eps)} (fail {sum(1 for e in eps if e.y == 1)}) "
              f"| train {len(sp['train'])} / calib {len(sp['calib'])} "
              f"(succ {sum(1 for e in sp['calib'] if e.succ == 1)}) "
              f"/ test {len(sp['test'])} | scenes {sc_split}", flush=True)

    dims = {t["dim"] for t in tasks.values()}
    if len(dims) != 1:
        raise SystemExit(f"shard 마다 dim 이 다르다 {sorted(dims)} — mixed arm 불가")

    arms = ["pertask", "mixed"] if args.arm == "both" else [args.arm]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ("lstm", "mlp")]
    if bad:
        raise SystemExit(f"--models 에 알 수 없는 값 {bad} (lstm|mlp)")

    rows: list[dict] = []
    detail: list[dict] = []
    registry: list[dict] = []
    for arm in arms:
        for kind in models:
            if arm == "loto":
                r, d, _ = run_loto(kind, tasks, splits, args, out_dir)
            elif arm == "loko-cell":
                r, d, reg = run_loko_cells(kind, tasks, args, out_dir, loko_cells)
                # registry 는 model 무관(셀 게이트 결과) — 첫 model 것만 원장으로 남긴다.
                if kind == models[0]:
                    registry += reg
            else:
                r, d, _ = run_arm(arm, kind, tasks, splits, args, out_dir)
            rows += r
            detail += d

    # timer 기준선 (무feature): "t ≥ W 면 발화". detector 가 이것보다 못하면 학습한 것은
    # 길이뿐이다. arm/α 무관이라 task 당 한 행 + 풀링 한 행.
    # (loto 는 fold 마다 held-out 전 episode 위 timer 행을 run_loto 안에서 이미 낸다)
    # (loko-cell 도 셀마다 자기 W 로 timer 행을 run_loko_cells 안에서 낸다)
    timer_all: list[dict] = []
    for t in (sorted(tasks) if not set(arms) & {"loto", "loko-cell"} else []):
        recs = timer_records(splits[t]["test"], t, tasks[t].get("W"),
                             tasks[t]["phase_names"])
        if not recs:
            continue
        timer_all += recs
        row = {"task": t, "instruction": tasks[t]["instruction"], "arm": "-",
               "model": "timer", "alpha": None, "truncate": args.truncate_train,
               "skip_reason": ""}
        row.update(summarize(recs))
        rows.append(row)
    if timer_all:
        row = {"task": "__pooled__", "instruction": "", "arm": "-", "model": "timer",
               "alpha": None, "truncate": args.truncate_train, "skip_reason": ""}
        row.update(summarize(timer_all))
        rows.append(row)
        detail += timer_all

    write_tsv(rows, out_dir / "sim_summary.tsv")
    if args.arm == "loko-cell":
        write_registry(registry, out_dir / "cell_registry.tsv")
        n_reg = sum(1 for r in registry if r.get("registered"))
        print(f"[loko] registry {len(registry)} 셀 → 등록 {n_reg} / 미등록 "
              f"{len(registry) - n_reg} (cell_registry.tsv)", flush=True)
    (out_dir / "sim_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    payload = {
        "config": {
            "shards": [p.name for p in paths],          # basename 만 (docs/04 §8)
            "layer": args.layer, "denoise": args.denoise, "seg": args.seg,
            "arm": args.arm, "models": models, "alphas": list(args.alphas),
            "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
            "hidden": args.hidden, "lambda_reg": args.lambda_reg,
            "grad_clip": args.grad_clip, "batch_size": args.batch_size,
            "band_mu": args.band_mu,
            "truncate_train": args.truncate_train,
            "truncate_caps": {t: {"rollout_W": tasks[t].get("W"),
                                  "phase_caps": tasks[t].get("phase_caps"),
                                  "n_dropped": tasks[t].get("n_trunc_dropped", 0)}
                              for t in sorted(tasks)},
            "td_grid": list(TD_GRID),
            "split": {"train": args.train_scenes, "calib": args.calib_scenes,
                      "test": args.test_scenes, "unit": "scene"},
            "exclude_cells": {
                "tsv": Path(args.exclude_cells_tsv).name if args.exclude_cells_tsv else None,
                "n_cells": len(exclude_cells),
                "n_excluded": {t: tasks[t].get("n_excluded") for t in sorted(tasks)}},
            **({"loko": {
                # 파일명은 basename 만 (docs/04 §8 — 절대경로 기록 금지)
                "cells_tsv": Path(args.loko_cells_tsv).name if args.loko_cells_tsv
                             else None,
                "n_cells": len(loko_cells),
                "n_registered": sum(1 for r in registry if r.get("registered")),
                "gates": {"min_pool_fail": int(args.min_pool_fail),
                          "min_calib_succ": int(args.min_calib_succ)},
                "cp": "loo" if int(args.cp_folds) <= 0 else f"kfold-{int(args.cp_folds)}",
                "cp_folds": int(args.cp_folds),
                "jstrat_td": JSTRAT_TD,
                "eval_sets": [n for n, _ in LOKO_EVAL_SETS],
                "train": "pool_other(같은 scene 의 다른 j 전판) + 대상 j 실패판",
                "excluded_from_train": "대상 j 성공판 (success-blind)",
                "eval_seq": "full (절제 없음)"}}
               if args.arm == "loko-cell" else {}),
            **({"loto": {"folds": {h: [t for t in sorted(tasks) if t != h]
                                   for h in sorted(tasks)},
                         "test": "held-out task 전 episode (full, zero-shot)",
                         "band": "train task 성공 판 풀링",
                         "W": "held-out 자기 성공 통계 = oracle 참조용"}}
               if args.arm == "loto" else {}),
            "label": "y=1 은 failure (SAFE 규약)",
        },
        "scene_split": {t: splits[t]["scenes"] for t in splits},
        "episodes": detail,
    }
    (out_dir / "sim_detail.json").write_text(json.dumps(payload, ensure_ascii=False),
                                             encoding="utf-8")

    print(f"\n[summary] {time.time()-t0:.0f}s — {len(rows)} 행 → {out_dir.name}/")
    hdr = f"{'task':22s} {'arm':7s} {'model':5s} {'α':>5s} {'TPR':>5s} {'FPR':>5s} " \
          f"{'preW':>5s} {'lead':>5s} {'relpos':>6s} {'left':>5s} {'lenA':>5s} " \
          f"{'td10':>5s} {'td20':>5s}"
    print(hdr)
    for r in rows:
        if r.get("tpr") is None:
            continue
        print(f"{str(r['task'])[:22]:22s} {r['arm']:7s} {r['model']:5s} "
              f"{_fmt(r['alpha']):>5s} {r['tpr']:5.2f} {(r['fpr'] or 0):5.2f} "
              f"{_fmt(r.get('tpr_before_W')):>5s} "
              f"{_fmt(r.get('lead_vs_W'), 1):>5s} "
              f"{_fmt(r['median_relpos_fail']):>6s} "
              f"{_fmt(r['median_steps_before_end_fail'],1):>5s} "
              f"{_fmt(r['length_auroc']):>5s} "
              f"{_fmt(r.get('auroc_td10')):>5s} {_fmt(r.get('auroc_td20')):>5s}")
    return 0


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", help="Tier A shard 디렉터리 (<slug>.npz)")
    ap.add_argument("--out", help="출력 디렉터리")
    ap.add_argument("--shards", default="", help="slug 콤마목록 (기본: 디렉터리 전부)")
    ap.add_argument("--layer", type=int, default=12, help="물리 capture layer 번호")
    ap.add_argument("--denoise", type=int, default=-1,
                    help="denoise step index (-1 = 마지막 k)")
    ap.add_argument("--seg", default="all",
                    help="token segment 이름 (state|future|action|all)")
    ap.add_argument("--arm", default="both",
                    choices=("both", "pertask", "mixed", "loto", "loko-cell"),
                    help="both=pertask+mixed. loto=leave-one-task-out (task 전이), "
                         "loko-cell=scene-local leave-one-jitter-out (셀별 detector) "
                         "— 각각 단독 실행")
    ap.add_argument("--models", default="lstm,mlp", help="lstm,mlp")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3", help="CP 유의수준(FPR 목표)")
    ap.add_argument("--train-scenes", type=int, default=6)
    ap.add_argument("--calib-scenes", type=int, default=2)
    ap.add_argument("--test-scenes", type=int, default=2)
    ap.add_argument("--exclude-cells-tsv", default="",
                    help="train/calib 에서 제외할 eval 셀 TSV (열: slug|grid_instruction"
                         "|instruction, scene_idx, jitter_idx(우선)|jitter_reset_idx, "
                         "noise_idx; 나머지 열은 무시)")
    ap.add_argument("--loko-cells-tsv", default="",
                    help="--arm loko-cell 의 대상 셀 TSV (열 계약은 "
                         "--exclude-cells-tsv 와 동일; (slug,scene,jitter) 로 dedupe)")
    ap.add_argument("--min-pool-fail", type=int, default=3,
                    help="loko-cell 게이트: pool_other 의 최소 실패 판 수")
    ap.add_argument("--min-calib-succ", type=int, default=9,
                    help="loko-cell 게이트: CP 밴드용 최소 성공 판 수 "
                         "(conformal (1−α) 분위 정의에 n ≥ 1/α − 1; α=0.1 → 9)")
    ap.add_argument("--cp-folds", type=int, default=0,
                    help="loko-cell CP 밴드 fold 수 (0 = episode LOO, >0 = k-fold)")
    ap.add_argument("--truncate-train", default="none",
                    choices=("none", "rollout", "phase-gt"),
                    help="학습 데이터 길이 절제 (TRAIN·CALIB 만; TEST 는 항상 full). "
                         "rollout=성공 record 수 ceil(μ+1σ) 로 앞부분만, "
                         "phase-gt=phase 별 성공 dwell ceil(μ+1σ) cap. 기본 none")
    ap.add_argument("--band-mu", default="train", choices=("train", "calib"),
                    help="δ_t 의 μ/σ 출처 (bw 는 항상 calib 성공 판)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lambda-reg", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="epoch 로그 끄기")
    ap.add_argument("--self-test", action="store_true",
                    help="합성 데이터로 파이프라인 검증 (TPR > FPR)")
    args = ap.parse_args()
    args.alphas = [float(x) for x in str(args.alphas).split(",") if str(x).strip()]

    if args.self_test:
        return self_test(args)
    if not args.shard_dir or not args.out:
        ap.error("--shard-dir 와 --out 은 필수 (--self-test 제외)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
