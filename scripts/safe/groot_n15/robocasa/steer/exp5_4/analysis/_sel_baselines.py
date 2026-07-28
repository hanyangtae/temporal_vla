#!/usr/bin/env python3
"""exp5-4 — 자명 특징 baseline 점수 생성기 ([S,J] 행렬).

학습 축(활성 mean-diff) 과 **동일한 split·동일한 top-1 절차**로 비교하기 위해
점수만 만들어 주고 선택은 `_sel_common.score_select` 가 한다.

★ 2026-07-28 정정 (Codex Gate2 리뷰): rollout **csv 한 행 = replan 1회의 첫 action**
   (실측: ep0 csv 62행 = record 62개, env_step 311 = 62×5+1, n_action_steps=5).
   따라서 csv 5행을 읽으면 "서로 다른 replan 5개의 첫 action" 이지 첫 chunk 가 아니다.
   chunk 내부 통계량은 **pkl `actions[0]`** (action.end_effector_position [16,3] +
   rotation [16,3]) 에서만 정확히 계산할 수 있다 → pkl 경로로 전환.

점수 종류
  · act_norm_L{L}      : record 0 활성의 L2 노름 (layer별)                [배포가능]
  · a0_pos_norm        : chunk 첫 step 의 (dx,dy,dz) 노름                 [배포가능]
  · a0_full_norm       : chunk 첫 step 6-dim(위치+회전) 노름              [배포가능]
  · chunk_speed_mean   : 첫 chunk(16 step) 평균 ‖a_t‖                     [배포가능]
  · chunk_tv           : 첫 chunk 내부 total variation Σ‖a_{t+1}−a_t‖     [배포가능]
  · chunk_jerk         : 첫 chunk 내부 2차 차분 Σ‖a_{t+2}−2a_{t+1}+a_t‖   [배포가능]
  · seed_only          : noise seed 값 자체 (음성 대조 — 반드시 무효)     [음성대조]
  · oracle_handle_cos  : chunk 첫 step 이동방향 vs (손잡이−초기 eef) cos   [privileged]

pkl 은 판당 ~250MB (활성 포함) 이라 cell 당 160판 로드에 수 분 걸린다 →
결과는 `--pkl-cache` NPZ 로 캐시한다.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from _sel_common import build_ep_index

# cell → rollout 루트 (원격 실측 경로)
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

CHUNK_NAMES = ("a0_pos_norm", "a0_full_norm", "chunk_speed_mean", "chunk_tv", "chunk_jerk")
DEPLOYABLE = CHUNK_NAMES


def activation_norm_scores(A):
    """[S,J] — record 0 활성 L2 노름."""
    return np.linalg.norm(A, axis=2)


def seed_only_scores(seeds, S):
    """[S,J] — seed 값 자체 (모든 scene 에서 같은 순위 → 음성 대조)."""
    return np.tile(np.asarray(seeds, float)[None, :], (S, 1))


def _chunk_from_pkl(d):
    """pkl actions[0] → chunk [16,6] (eef 위치 3 + 회전 3)."""
    a0 = d["actions"][0]
    pos = np.asarray(a0["action.end_effector_position"], float)
    rot = np.asarray(a0["action.end_effector_rotation"], float)
    return np.concatenate([pos, rot], axis=1)


def pkl_pass(cell, E, scenes, root=None, cache=None, geom=None, verbose=True):
    """cell 의 모든 판에 대해 pkl 을 한 번 훑어 chunk 점수(+oracle) 계산.

    반환 (scores dict[name]=[S,J], diag dict)
    """
    S, J = E.shape
    names = list(CHUNK_NAMES)
    out = {n: np.full((S, J), np.nan) for n in names}
    oracle = np.full((S, J), np.nan)
    diag = dict(n_ok=0, n_miss=0, n_total=S * J, mean_oracle_cos=None,
                init_state_check=None, source="pkl actions[0] (16-step chunk)")

    cpath = Path(cache).expanduser() if cache else None
    if cpath and cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        if all(n in z for n in names) and z[names[0]].shape == (S, J):
            for n in names:
                out[n] = z[n]
            if "oracle_handle_cos" in z:
                oracle = z["oracle_handle_cos"]
                if np.isfinite(oracle).any():
                    diag["mean_oracle_cos"] = float(-np.nanmean(oracle))
            diag.update(dict(n_ok=int(np.isfinite(out[names[0]]).sum()),
                             n_miss=int((~np.isfinite(out[names[0]])).sum()),
                             cached=str(cpath)))
            if verbose:
                print(f"  [cache] {cpath} 재사용 ({diag['n_ok']}/{S*J}판)")
            return out, oracle, diag

    r = Path((root or ROLLOUT_ROOTS.get(cell, "")).replace("~", str(Path.home())))
    if not r.exists():
        diag["error"] = f"rollout 루트 없음: {r}"
        return out, oracle, diag
    idx = build_ep_index(r)

    cos_all, state_ref = [], {}
    for i in range(S):
        for j in range(J):
            f = idx.get(int(E[i, j]))
            pk = f.with_suffix(".pkl") if f is not None else None
            if pk is None or not pk.exists():
                diag["n_miss"] += 1
                continue
            with open(pk, "rb") as fh:
                d = pickle.load(fh)
            ch = _chunk_from_pkl(d)
            if ch.shape[0] < 3:
                diag["n_miss"] += 1
                continue
            diag["n_ok"] += 1
            out["a0_pos_norm"][i, j] = np.linalg.norm(ch[0, :3])
            out["a0_full_norm"][i, j] = np.linalg.norm(ch[0])
            out["chunk_speed_mean"][i, j] = np.linalg.norm(ch, axis=1).mean()
            out["chunk_tv"][i, j] = np.linalg.norm(np.diff(ch, axis=0), axis=1).sum()
            out["chunk_jerk"][i, j] = np.linalg.norm(np.diff(ch, n=2, axis=0), axis=1).sum()

            st0 = d["states"][0]
            key = (i,)
            eef = np.asarray(st0["observation.state.eef_pos_rel"], float)
            if key in state_ref:
                dd = float(np.abs(state_ref[key] - eef).max())
                diag["init_state_check"] = max(diag["init_state_check"] or 0.0, dd)
            else:
                state_ref[key] = eef
            if geom is not None and scenes[i] in geom:
                R = _quat_to_R(st0["observation.state.base_rotation"])
                bpos = np.asarray(st0["observation.state.base_position"], float)
                tgt = R.T @ (geom[scenes[i]]["h"] - bpos) - eef
                tn, vn = np.linalg.norm(tgt), np.linalg.norm(ch[0, :3])
                if tn > 0 and vn > 0:
                    c = float(ch[0, :3] @ (tgt / tn) / vn)
                    oracle[i, j] = -c          # '작을수록 선택' 규약 → −cos
                    cos_all.append(c)
    if cos_all:
        diag["mean_oracle_cos"] = float(np.mean(cos_all))
    if cpath:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cpath, oracle_handle_cos=oracle, **out)
        diag["cached"] = str(cpath)
    return out, oracle, diag


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
