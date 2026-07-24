#!/usr/bin/env python3
"""exp4-3 분리도 지도: 모델별 캡처 계약 → 공통 rolls 계약 어댑터 (유일한 모델별 코드).

공통 rolls 계약 (sweep 코어가 소비하는 3+α 키):
  { name, success:int, length:n, dit:[n,L,D], dit_k:[n,L,K,D], phases:list[n],
    capture_layers:list[L], episode_idx, inference_seed, scenario_seed }

디스패치는 hidden_states record 의 ndim 으로 (feature_kind 는 로깅·검증용):
  ndim=4 [L,K,T,D]  → full-token (N1.5 / N1.6 perT) — 기존 load_rollout_fulltoken(T mean)
  ndim=3 [L,K,D]    → pooled (N1.6 pooled / π0.5 expert) — K mean 이 dit
  ndim=2 [L,D]      → slot (Cosmos action-슬롯) — K=1 로 확장

D·L 이 셀 내 불일치하면 raise (혼입 게이트). 사용은 승준 노드(~/anaconda3/bin/python).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/steer"))

from fit_phase_conceptor_n15 import load_rollout_fulltoken  # noqa: E402


def _load_pooled(d: dict, pkl_path) -> dict:
    """record [L,K,D] (T 이미 pool — N1.6 pooled / π0.5 expert 캡처)."""
    hs = d.get("hidden_states") or []
    if not hs:
        raise ValueError(f"{pkl_path}: hidden_states 비어 있음")
    dit_k = np.stack([np.asarray(r, dtype=np.float32) for r in hs], axis=0)  # [n,L,K,D]
    if dit_k.ndim != 4:
        raise ValueError(f"{pkl_path}: pooled 기대 [n,L,K,D], got {dit_k.shape}")
    phases = list(d.get("feature_phases") or [])
    if len(phases) != dit_k.shape[0]:
        raise ValueError(f"{pkl_path}: feature_phases {len(phases)} != records {dit_k.shape[0]}")
    return {"dit_k": dit_k, "dit": dit_k.mean(axis=2), "phases": phases,
            "length": int(dit_k.shape[0])}


def _load_slot(d: dict, pkl_path) -> dict:
    """record [L,D] (Cosmos action-슬롯). K=1 확장."""
    hs = d.get("hidden_states") or []
    if not hs:
        raise ValueError(f"{pkl_path}: hidden_states 비어 있음")
    dit = np.stack([np.asarray(r, dtype=np.float32) for r in hs], axis=0)  # [n,L,D]
    if dit.ndim != 3:
        raise ValueError(f"{pkl_path}: slot 기대 [n,L,D], got {dit.shape}")
    phases = list(d.get("feature_phases") or [])
    if len(phases) != dit.shape[0]:
        raise ValueError(f"{pkl_path}: feature_phases {len(phases)} != records {dit.shape[0]}")
    return {"dit_k": dit[:, :, None, :], "dit": dit, "phases": phases,
            "length": int(dit.shape[0])}


def load_one(pkl_path: Path, capture_layers_override=None) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    rec0 = np.asarray((d.get("hidden_states") or [None])[0])
    if rec0.ndim == 4:
        r = load_rollout_fulltoken(d, pkl_path, "mean")
    elif rec0.ndim == 3:
        r = _load_pooled(d, pkl_path)
    elif rec0.ndim == 2:
        r = _load_slot(d, pkl_path)
    else:
        raise ValueError(f"{pkl_path}: 알 수 없는 record ndim={rec0.ndim}")
    cl = (r.get("capture_layers") or d.get("capture_layers")
          or d.get("layer_indices") or capture_layers_override)
    if cl is None:
        raise ValueError(f"{pkl_path}: capture_layers 부재 — --capture-layers 로 명시하라")
    r["capture_layers"] = [int(x) for x in cl]
    if len(r["capture_layers"]) != r["dit"].shape[1]:
        raise ValueError(f"{pkl_path}: capture_layers {len(r['capture_layers'])} != L {r['dit'].shape[1]}")
    r["name"] = Path(pkl_path).name
    r["success"] = int(d.get("episode_success", 0))
    r["feature_kind"] = d.get("feature_kind")
    for k in ("episode_idx", "inference_seed", "scenario_seed"):
        v = d.get(k)  # 키가 존재하나 None 인 경우(예: inference_seed 미지정) 대비
        r[k] = int(v) if v is not None else -1
    return r


def load_cell_rolls(manifest: Path, cell: str, capture_layers_override=None) -> list:
    """manifest tsv(pkl \t label [\t scene]) 에서 cell 행만 로드. label 이 내장값 override."""
    rows = []
    for line in Path(manifest).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        if p.parent.name != cell:
            continue
        rows.append((p, int(parts[1])))
    if not rows:
        raise SystemExit(f"manifest 에 cell={cell} 행 없음: {manifest}")
    missing = [str(p) for p, _ in rows if not p.exists()]
    if missing:
        raise SystemExit(f"pkl 누락 {len(missing)}개: {missing[:3]}")
    rolls = []
    contract = None
    for p, label in rows:
        r = load_one(p, capture_layers_override)
        r["success"] = label
        sig = (tuple(r["capture_layers"]), r["dit"].shape[2])  # (L 목록, D)
        if contract is None:
            contract = sig
        elif sig != contract:
            raise SystemExit(f"{p}: 캡처 계약 혼입 {sig} != {contract}")
        rolls.append(r)
    return rolls
