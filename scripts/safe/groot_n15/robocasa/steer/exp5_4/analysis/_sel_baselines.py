#!/usr/bin/env python3
"""exp5-4 — 자명 특징 baseline 점수 생성기 ([S,J] 행렬).

학습 축(활성 mean-diff) 과 **동일한 split·동일한 top-1 절차**로 비교하기 위해
점수만 만들어 주고 선택은 `_sel_common.score_select` 가 한다.

점수 종류
  · act_norm_L{L}      : record 0 활성의 L2 노름 (layer별)              [배포가능]
  · a0_pos_norm        : 첫 executed action 의 (dx,dy,dz) 노름          [배포가능]
  · a0_full_norm       : 첫 action 6-dim(위치+회전) 노름                [배포가능]
  · chunk_speed_mean   : 첫 chunk(=executed 5 step) 평균 ‖a_t‖          [배포가능]
  · chunk_tv           : chunk 내부 total variation Σ‖a_{t+1}−a_t‖      [배포가능]
  · chunk_jerk         : chunk 내부 2차 차분 Σ‖a_{t+2}−2a_{t+1}+a_t‖    [배포가능]
  · seed_only          : noise seed 값 자체 (음성 대조 — 반드시 무효)   [음성대조]
  · oracle_handle_cos  : 첫 action 이동방향 vs (손잡이−초기 eef) cos    [privileged]

주의
  · GR00T 표준은 chunk 16 예측 / 5 실행 → csv 한 행 = env step, 첫 chunk = 앞 5행.
  · oracle_handle_cos 는 손잡이 좌표(시뮬 특권 정보) + rollout pkl 초기 상태를
    쓰므로 **배포 불가**. 표에서 별도 구획으로 표기한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _sel_common import build_ep_index, read_actions

# cell → rollout csv 루트 (원격 실측 경로)
ROLLOUT_ROOTS = {
    "pq3_drawer_right": "~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/"
                        "scene_matched_exp41/pq3_drawer_right/raw_rollouts",
    "pq3_ppcc_beer": "~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/"
                     "scene_matched_exp41/pq3_ppcc_beer/raw_rollouts",
    "exp41_mixer": "~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/"
                   "exp41_mixer/raw_rollouts",
    "exp53_mixer_sm": "~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/"
                      "exp5_3_mixer_sm/raw_rollouts",
}

DEPLOYABLE = ("a0_pos_norm", "a0_full_norm", "chunk_speed_mean", "chunk_tv", "chunk_jerk")


def activation_norm_scores(A):
    """[S,J] — record 0 활성 L2 노름."""
    return np.linalg.norm(A, axis=2)


def seed_only_scores(E_seeds, S):
    """[S,J] — seed 값 자체 (모든 scene 에서 동일 순위 → 음성 대조)."""
    return np.tile(np.asarray(E_seeds, float)[None, :], (S, 1))


def action_scores(cell: str, E, chunk: int = 5, root: str | None = None):
    """rollout csv 에서 chunk 기반 점수들 계산. E = [S,J] episode index.

    반환 (scores: {name: [S,J]}, note: str)
    """
    r = Path((root or ROLLOUT_ROOTS.get(cell, "")).replace("~", str(Path.home())))
    if not r.exists():
        return {}, f"rollout 루트 없음: {r}"
    S, J = E.shape
    idx = build_ep_index(r)
    names = ("a0_pos_norm", "a0_full_norm", "chunk_speed_mean", "chunk_tv", "chunk_jerk")
    out = {n: np.full((S, J), np.nan) for n in names}
    n_ok = n_miss = 0
    for i in range(S):
        for j in range(J):
            f = idx.get(int(E[i, j]))
            if f is None:
                n_miss += 1
                continue
            act, _hdr = read_actions(f, chunk)
            if act is None or len(act) < 2:
                n_miss += 1
                continue
            n_ok += 1
            pos, full = act[:, :3], act[:, :6]
            out["a0_pos_norm"][i, j] = np.linalg.norm(pos[0])
            out["a0_full_norm"][i, j] = np.linalg.norm(full[0])
            out["chunk_speed_mean"][i, j] = np.linalg.norm(full, axis=1).mean()
            d1 = np.diff(full, axis=0)
            out["chunk_tv"][i, j] = np.linalg.norm(d1, axis=1).sum()
            if len(full) >= 3:
                d2 = np.diff(full, n=2, axis=0)
                out["chunk_jerk"][i, j] = np.linalg.norm(d2, axis=1).sum()
    note = f"csv {n_ok}/{S*J} 판 사용 (결측 {n_miss}), chunk={chunk}행"
    return out, note


# ── privileged oracle: 손잡이 기하 ──────────────────────────────────────
def _quat_to_R(q):
    """robosuite 규약 (x,y,z,w) 쿼터니언 → 회전행렬."""
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_handle_tsv(path):
    geom = {}
    for ln in Path(path).expanduser().read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) < 15 or "__ERROR__" in ln:
            continue
        geom[int(p[0])] = dict(h=np.array([float(p[1]), float(p[2]), float(p[3])]),
                               b=np.array([float(p[4]), float(p[5]), float(p[6])]))
    return geom


def handle_oracle_scores(cell, E, scenes, geom, chunk=5, root=None, pkl_per_scene=True):
    """첫 action 이동방향과 (손잡이−초기 eef) 방향의 cos. 점수는 '작을수록 좋음'
    규약에 맞추려고 −cos 를 돌려준다(즉 cos 큰 판이 top-1).

    scene 당 pkl 1개만 열어 초기 상태(eef_pos_rel, base_position/rotation)를 얻는다
    (같은 seed scene 이면 초기 상태 동일 — 스크립트가 2판 대조로 검증).
    """
    import pickle
    r = Path((root or ROLLOUT_ROOTS.get(cell, "")).replace("~", str(Path.home())))
    S, J = E.shape
    idx = build_ep_index(r)
    out = np.full((S, J), np.nan)
    diag = dict(scene_state_check=None, n_scene_geom=0, mean_cos=None)
    cos_all = []
    for i in range(S):
        s = scenes[i]
        if s not in geom:
            continue
        # 초기 상태: 이 scene 첫 episode 의 pkl
        st0 = None
        for j in range(J):
            f = idx.get(int(E[i, j]))
            if f is None:
                continue
            pk = f.with_suffix(".pkl")
            if not pk.exists():
                continue
            with open(pk, "rb") as fh:
                d = pickle.load(fh)
            st0 = d["states"][0]
            break
        if st0 is None:
            continue
        diag["n_scene_geom"] += 1
        R = _quat_to_R(st0["observation.state.base_rotation"])
        bpos = np.asarray(st0["observation.state.base_position"], float)
        eef = np.asarray(st0["observation.state.eef_pos_rel"], float)
        tgt = R.T @ (geom[s]["h"] - bpos) - eef        # base frame 에서 손잡이 방향
        tn = np.linalg.norm(tgt)
        if tn == 0:
            continue
        tgt = tgt / tn
        for j in range(J):
            f = idx.get(int(E[i, j]))
            if f is None:
                continue
            act, _ = read_actions(f, chunk)
            if act is None:
                continue
            v = act[0, :3]
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            c = float(v @ tgt / vn)
            out[i, j] = -c
            cos_all.append(c)
    if cos_all:
        diag["mean_cos"] = float(np.mean(cos_all))
    return out, diag
