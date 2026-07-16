"""GR00T N1.5 RoboCasa — wrong-grasp 에피소드의 VL activation 분리 분석.

질문: wrong-grasp(목표물 대신 distractor 파지, event labeler 7번째 phase) 에피소드의
VL pathway activation 이 나머지 데이터(succ + other-fail)와 분포가 다른가? 특히
**발생 전(pre-onset) reach phase** 에서 이미 다른가 (online failure-type 식별 근거,
docs/steering/14 의 중심 미해결 문제).

주장 상한 (사전 고정, confound gate 7):
  단일 cell·고정 관측에서 VL 은 이미지의 결정적 함수 → 최강 주장은
  "VL representation 이 wrong-grasp 를 grasp 이벤트 전에 선형 판독 가능하게 인코딩"
  (failure-predictive monitor). "goal 오독이 VL 에서 발생한다"는 인과 주장 불가.

사전등록 스펙 (플랜 ~/.claude/plans/wrong-grasp-binary-dewdrop.md, 실행 전 freeze):
  - Primary (확증 1개): VL × W_pre × wg vs rest-all. budget = 포함 에피소드 global-min
    reach count (탈락 0 원칙 — 탈락은 selection bias). episode-pooled LOO-AUROC,
    episode-permutation p (n_perm=1000, 소표본이면 exact 전수), α=0.05.
    LOO 의 역할 = 점수 생성 장치 (D=2048 ≫ n → in-sample LDA 는 무작위 라벨도 AUROC 1.0,
    검정력 0). 유의성 판정은 permutation p 가 담당.
  - Positive control: VL × W_at × wg vs succ(grasp+transport). 분리 실패 → 파이프라인 무효.
  - Layer 특이성 (secondary): W_pre × rest-all, DiT 7 + VL 프로파일, max-T permutation
    familywise 보정 (공유 permutation).
  - wg vs other-fail: exact permutation 병기하되 **descriptive 전용** (null 상단 ~0.85+).
  - dwell-baseline gate 는 두지 않음: wg 의 reach dwell 이 길어 count AUROC ~0.9+ 가
    예상되고, equal-budget pool 이 count 경로를 이미 차단 → gate 로 쓰면 통과 불가능한
    결함 기준. dwell AUROC 는 context 통계로만 병기.
  - Robustness: ① dwell-matched subset ② comparator 내 score↔dwell Spearman
    ③ budget sweep ④ 라벨-fit 없는 succ-기준 diag-Mahalanobis anomaly (vl_anomaly_score 방식).
  - Leakage 사다리: W_early(절대 초기 k) → W_pre → onset-정렬 t_rel 곡선 (진단 전용).

메커니즘 반영 추가 (2026-07-16 census·영상 실증 후):
  wg 는 초기 오독이 아니라 [bread 정상 파지 → 운반 중 drop → bread 시야 밖 소실 →
  재탐색 → distractor 파지] 의 2차 사건. 갈림은 drop 물리(bread 가 counter 에 남나 /
  바닥으로 사라지나)가 결정 — 영상 3+2건 실증. 따라서:
  - W_pre 무신호는 "가설 반증"이 아니라 메커니즘상 당연 (초기엔 아무 문제 없음).
  - 올바른 event-matched 질문 = **W_postdrop**: drop 직후 ~ 다음 파지 이벤트 전
    재탐색 reach 구간에서, 이후 wg 로 가는 에피소드 vs bread 재획득 에피소드 비교.
    비교군 = drop 경험자만 (같은 event-state). drop 없는 wg(ep39 류)는 제외·별도 보고.
  - 해석 가드레일: 여기서의 분리는 "target 소실 상태의 시각 판독"일 가능성이 지배적
    (bread 가시성) — online trigger(target-lost 검출)로는 유용, "VL 오독 원인" 증거 아님.

``phase_separation.py`` 의 검증된 primitive 를 import 재사용 (수정하지 않음):
load_rollout, phase_records, equal_budget_pool, rank_auroc, loo_auroc, _lda_project.
LOO permutation 은 fold-별 PCA projection 이 y-무관임을 이용해 캐시(FoldProjector)로
가속하되, 수학은 ps.loo_auroc 와 동일해야 한다 (tests 에서 동치 assert).

READ-ONLY on raw_rollouts. 출력만 analysis/wrong_grasp_vl_separation/<cell>/ 아래.
실행: 원격 anaconda python (torch 필요, scipy 없음, OMP cap).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import phase_separation as ps  # noqa: E402  (검증된 primitive 재사용)

SEED = 0
WG_PHASE = "wrong-grasp"
REACH_PHASE = "reach-to-object"
AT_COMPARATOR_PHASES = ("grasp", "transport")
EXACT_MAX = 20000  # 가능한 라벨링 수가 이 이하면 exact permutation 전수 열거

CLAIM_CEILING = (
    "단일 cell·고정 관측: VL은 이미지의 결정적 함수 → 'VL representation이 wrong-grasp를 "
    "grasp 이벤트 전에 선형 판독 가능하게 인코딩'(failure-predictive monitor)까지만. "
    "'goal 오독이 VL에서 발생' 인과 주장 불가."
)
SCOPE_NOTE = "1 cell (1 task×1 object×1 seed×1 instruction) — 일반화 불가, cell-내 존재 증명."


# --------------------------------------------------------------------------- #
# 에피소드 분류·윈도 (순수 함수, 테스트 대상)
# --------------------------------------------------------------------------- #
def classify_episode(roll: dict) -> str:
    """"wg" (wrong-grasp record ≥1) | "succ" | "other_fail"."""
    if any(p == WG_PHASE for p in roll["phases"]):
        return "wg"
    return "succ" if roll["success"] else "other_fail"


def first_wg_index(roll: dict) -> int | None:
    for i, p in enumerate(roll["phases"]):
        if p == WG_PHASE:
            return i
    return None


def first_phase_index(roll: dict, phase: str) -> int | None:
    for i, p in enumerate(roll["phases"]):
        if p == phase:
            return i
    return None


def load_rollout_with_events(pkl_path: Path) -> dict:
    """ps.load_rollout + event 필드 (drop/grasp record 인덱스, record 단위 실증 완료)."""
    import pickle

    roll = ps.load_rollout(pkl_path)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    roll["drop_steps"] = [int(v) for v in (d.get("drop_steps") or [])]
    roll["grasp_steps"] = [int(v) for v in (d.get("grasp_steps") or [])]
    return roll


def postdrop_window(roll: dict) -> tuple[list[int], str]:
    """W_postdrop: drop 직후 ~ 다음 획득 이벤트 전의 reach 레코드 (event-matched).

    wg  : anchor = 첫 wg record **이전 마지막** drop, 끝 = 첫 wg record.
          drop 없는 wg (ep39 류: insert 후 재탐색 중 wg) 는 제외 ("no_drop_before_wg").
    비-wg: anchor = 첫 drop, 끝 = 그 뒤 첫 grasp 이벤트 (없으면 에피소드 끝).
          drop 이 없는 에피소드는 event-state 부재로 제외 ("no_drop").
    반환: (reach record 인덱스 리스트, 상태 문자열)
    """
    drops = roll.get("drop_steps") or []
    phases = roll["phases"]
    fw = first_wg_index(roll)
    if fw is not None:
        pre_drops = [s for s in drops if s < fw]
        if not pre_drops:
            return [], "no_drop_before_wg"
        a, end = pre_drops[-1], fw
    else:
        if not drops:
            return [], "no_drop"
        a = drops[0]
        later_grasps = [g for g in (roll.get("grasp_steps") or []) if g > a]
        end = later_grasps[0] if later_grasps else len(phases)
    idx = [i for i in range(a + 1, min(end, len(phases))) if phases[i] == REACH_PHASE]
    return idx, "ok" if idx else "empty_window"


def window_indices(roll: dict, window: str, k: int | None = None) -> list[int]:
    """에피소드에서 윈도에 속한 record 인덱스.

    W_pre  : wg 에피소드 = 첫 wg record **이전** reach 레코드 (인과/pre-onset),
             비-wg = reach 레코드 전체.
    W_at   : wg = wrong-grasp 레코드, 비-wg = grasp+transport 레코드 (양성 대조용).
    W_early: 클래스 무관 절대 첫 k 레코드.
    """
    phases = roll["phases"]
    if window == "W_pre":
        cut = first_wg_index(roll)
        rng = range(len(phases)) if cut is None else range(cut)
        return [i for i in rng if phases[i] == REACH_PHASE]
    if window == "W_at":
        if any(p == WG_PHASE for p in phases):
            return [i for i, p in enumerate(phases) if p == WG_PHASE]
        return [i for i, p in enumerate(phases) if p in AT_COMPARATOR_PHASES]
    if window == "W_early":
        assert k is not None and k > 0
        return list(range(min(k, len(phases))))
    raise ValueError(f"unknown window {window}")


def episode_vectors(roll: dict, layer_key, idx: list[int]) -> list[np.ndarray]:
    if layer_key == "VL":
        return [roll["vl"][i] for i in idx]
    return [roll["dit"][i, layer_key, :] for i in idx]


def build_design(rolls: list[dict], labels: list[int], window: str, layer_key,
                 budget: int | str = "auto", k: int | None = None) -> dict | None:
    """에피소드당 1벡터 설계행렬.

    budget="auto": 포함(윈도 record ≥1) 에피소드들의 global-min count — 탈락 0 원칙.
    budget=int  : count < budget 에피소드는 탈락 (탈락 수를 meta 로 정직하게 보고;
                  budget sweep 전용, primary 아님).
    반환 None = 클래스 한쪽이 비어 설계 불가.
    """
    entries = []  # (roll, y, idx)
    dropped = {"empty": [], "under_budget": []}
    for roll, y in zip(rolls, labels):
        idx = window_indices(roll, window, k)
        if not idx:
            dropped["empty"].append(roll["name"])
            continue
        entries.append((roll, y, idx))
    if not entries:
        return None
    counts = np.array([len(idx) for _, _, idx in entries])
    b = int(counts.min()) if budget == "auto" else int(budget)
    kept = []
    for roll, y, idx in entries:
        if len(idx) < b:
            dropped["under_budget"].append(roll["name"])
            continue
        kept.append((roll, y, idx))
    ys = np.array([y for _, y, _ in kept])
    if len(np.unique(ys)) < 2 or int((ys == 1).sum()) < 2 or int((ys == 0).sum()) < 2:
        return None
    X = np.stack([ps.equal_budget_pool(episode_vectors(roll, layer_key, idx), b)
                  for roll, _, idx in kept], axis=0)
    kept_counts = np.array([len(idx) for _, _, idx in kept], dtype=np.float64)
    return {
        "X": X, "y": ys, "budget": b,
        "names": [roll["name"] for roll, _, _ in kept],
        "counts": kept_counts,
        # dwell(count) AUROC 는 gate 아님 — context 통계 (잔존 부분집합 위에서 계산)
        "dwell_auroc": ps.rank_auroc(kept_counts, ys),
        "n_pos": int((ys == 1).sum()), "n_neg": int((ys == 0).sum()),
        "dropped": dropped,
    }


def build_postdrop_design(rolls: list[dict], layer_key, comparator_cls=("succ", "other_fail")) -> dict | None:
    """W_postdrop event-matched 설계행렬 (에피소드당 1벡터).

    포함 = postdrop_window 상태 "ok" 인 에피소드만 (event-state 매칭 — drop 없는
    에피소드는 비교 불가 상태라 정의상 제외, budget-탈락과 다름). budget = global-min.
    """
    entries = []
    excluded: dict[str, list[str]] = {}
    for roll in rolls:
        cls = classify_episode(roll)
        if cls != "wg" and cls not in comparator_cls:
            continue
        idx, status = postdrop_window(roll)
        if status != "ok":
            excluded.setdefault(status, []).append(f"{roll['name']}({cls})")
            continue
        entries.append((roll, 1 if cls == "wg" else 0, idx, cls))
    ys = np.array([y for _, y, _, _ in entries])
    if len(entries) == 0 or int((ys == 1).sum()) < 2 or int((ys == 0).sum()) < 2:
        return None
    b = min(len(idx) for _, _, idx, _ in entries)
    X = np.stack([ps.equal_budget_pool(episode_vectors(roll, layer_key, idx), b)
                  for roll, _, idx, _ in entries], axis=0)
    counts = np.array([len(idx) for _, _, idx, _ in entries], dtype=np.float64)
    return {
        "X": X, "y": ys, "budget": b, "counts": counts,
        "names": [roll["name"] for roll, _, _, _ in entries],
        "cls": [cls for _, _, _, cls in entries],
        "dwell_auroc": ps.rank_auroc(counts, ys),
        "n_pos": int((ys == 1).sum()), "n_neg": int((ys == 0).sum()),
        "excluded": excluded,
    }


# --------------------------------------------------------------------------- #
# LOO 점수 캐시 (ps.loo_auroc 와 수학 동일 — fold PCA 가 y-무관임을 이용)
# --------------------------------------------------------------------------- #
class FoldProjector:
    """fold 별 (mu, PCA-P, 투영된 train/test) 를 1회 계산해 permutation 을 가속.

    ps._lda_project 와 동일: n_pc=min(30, D, n_tr-1), mu/SVD 는 train 만으로,
    LDA 방향 = 투영공간 (mu1-mu0)/||·||. train 에 한 클래스가 비면 score 0.0
    (ps.loo_auroc 과 동일 처리).
    """

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        self.n = n
        self.tr_proj: list[np.ndarray] = []  # [n](n-1, n_pc)
        self.te_proj: list[np.ndarray] = []  # [n](n_pc,)
        for i in range(n):
            tr = np.ones(n, dtype=bool)
            tr[i] = False
            Xtr = X[tr]
            mu = Xtr.mean(axis=0)
            Xtr_c = Xtr - mu
            n_pc = int(min(30, X.shape[1], Xtr.shape[0] - 1))
            if n_pc <= 0:
                self.tr_proj.append(np.zeros((Xtr.shape[0], 1)))
                self.te_proj.append(np.zeros(1))
                continue
            _, _, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
            P = Vt[:n_pc].T
            self.tr_proj.append(Xtr_c @ P)
            self.te_proj.append((X[i] - mu) @ P)

    def loo_scores(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        scores = np.empty(self.n, dtype=np.float64)
        for i in range(self.n):
            ytr = np.delete(y, i)
            if len(np.unique(ytr)) < 2:
                scores[i] = 0.0
                continue
            Xtr_p = self.tr_proj[i]
            mu1 = Xtr_p[ytr == 1].mean(axis=0)
            mu0 = Xtr_p[ytr == 0].mean(axis=0)
            d = mu1 - mu0
            nd = np.linalg.norm(d)
            scores[i] = 0.0 if nd < 1e-12 else float(self.te_proj[i] @ (d / nd))
        return scores

    def loo_auroc(self, y: np.ndarray) -> float:
        return ps.rank_auroc(self.loo_scores(y), np.asarray(y))


def _perm_labelings(y: np.ndarray, n_perm: int, rng: np.random.Generator):
    """exact 가능하면 전수 열거, 아니면 n_perm 회 셔플. (kind, iterable, total)"""
    y = np.asarray(y)
    n, n1 = len(y), int((y == 1).sum())
    n_exact = math.comb(n, n1)
    if n_exact <= EXACT_MAX:
        def gen():
            for pos in itertools.combinations(range(n), n1):
                yp = np.zeros(n, dtype=y.dtype)
                yp[list(pos)] = 1
                yield yp
        return "exact", gen(), n_exact
    return "sampled", (rng.permutation(y) for _ in range(n_perm)), n_perm


def perm_stats(fp: FoldProjector, y: np.ndarray, n_perm: int,
               rng: np.random.Generator) -> dict:
    """관측 LOO-AUROC + permutation p (양측, |auroc-0.5| 기준) + null95 상단."""
    y = np.asarray(y)
    obs = fp.loo_auroc(y)
    obs_dev = abs(obs - 0.5)
    kind, labelings, total = _perm_labelings(y, n_perm, rng)
    devs = np.empty(total)
    for j, yp in enumerate(labelings):
        devs[j] = abs(fp.loo_auroc(yp) - 0.5)
    if kind == "exact":
        p = float((devs >= obs_dev - 1e-12).sum()) / total  # 관측 라벨링 포함 (exact)
    else:
        p = (1.0 + float((devs >= obs_dev - 1e-12).sum())) / (1.0 + total)
    return {"auroc": obs, "p_perm": p, "null95_upper": 0.5 + float(np.percentile(devs, 95)),
            "perm_kind": kind, "n_perm_effective": int(total)}


def max_t_family(fps: dict, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """layer family (공유 permutation) max-T familywise 보정.

    fps: {layer_label: FoldProjector} — 모든 layer 가 같은 에피소드/순서(y 공유) 가정.
    반환: familywise null95 + layer 별 (auroc, p_fw).
    """
    y = np.asarray(y)
    labels = list(fps.keys())
    obs = {lab: fps[lab].loo_auroc(y) for lab in labels}
    kind, labelings, total = _perm_labelings(y, n_perm, rng)
    max_devs = np.empty(total)
    for j, yp in enumerate(labelings):
        max_devs[j] = max(abs(fps[lab].loo_auroc(yp) - 0.5) for lab in labels)
    out = {"null95_fw_upper": 0.5 + float(np.percentile(max_devs, 95)),
           "perm_kind": kind, "n_perm_effective": int(total), "layers": {}}
    for lab in labels:
        dev = abs(obs[lab] - 0.5)
        cnt = float((max_devs >= dev - 1e-12).sum())
        p_fw = cnt / total if kind == "exact" else (1.0 + cnt) / (1.0 + total)
        out["layers"][lab] = {"auroc": obs[lab], "p_familywise": p_fw}
    return out


# --------------------------------------------------------------------------- #
# Robustness 통계
# --------------------------------------------------------------------------- #
def diag_mahal_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """라벨-fit 없는 보조 점수: 음성(neg=비교군) 분포 기준 대각 Mahalanobis.

    neg 에피소드 자신은 leave-one-out 으로 자기 자신을 참조분포에서 제외.
    (vl_anomaly_score.py 의 성공분포-기준 anomaly 방식을 y 에 일반화.)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    neg_idx = np.flatnonzero(y == 0)
    scores = np.empty(len(y))
    for i in range(len(y)):
        R = X[neg_idx[neg_idx != i]] if y[i] == 0 else X[neg_idx]
        mu = R.mean(axis=0)
        var = R.var(axis=0) + 1e-8
        scores[i] = float(np.sqrt(((X[i] - mu) ** 2 / var).mean()))
    return scores


def mahal_perm_stats(X: np.ndarray, y: np.ndarray, n_perm: int,
                     rng: np.random.Generator) -> dict:
    obs = ps.rank_auroc(diag_mahal_scores(X, y), y)
    kind, labelings, total = _perm_labelings(y, n_perm, rng)
    devs = np.empty(total)
    for j, yp in enumerate(labelings):
        devs[j] = abs(ps.rank_auroc(diag_mahal_scores(X, yp), yp) - 0.5)
    obs_dev = abs(obs - 0.5)
    if kind == "exact":
        p = float((devs >= obs_dev - 1e-12).sum()) / total
    else:
        p = (1.0 + float((devs >= obs_dev - 1e-12).sum())) / (1.0 + total)
    return {"auroc": obs, "p_perm": p, "null95_upper": 0.5 + float(np.percentile(devs, 95)),
            "perm_kind": kind}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """scipy 없는 Spearman = rank 변환 후 Pearson."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1)
        # tie 평균순위
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(np.asarray(a, dtype=np.float64)), ranks(np.asarray(b, dtype=np.float64))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def trel_curve(rolls: list[dict], layer_key, max_back: int = 12) -> dict:
    """onset-정렬 single-record AUROC 곡선 (진단 전용, 검정 아님).

    wg = 첫 wg record 에 정렬, comparator(succ) = 첫 grasp record 에 정렬.
    t_rel = -1..-max_back. 각 t_rel 에서 유효 에피소드만 (anchor+t_rel >= 0).
    """
    out = {}
    anchored = []
    for roll in rolls:
        cls = classify_episode(roll)
        if cls == "wg":
            anchored.append((roll, first_wg_index(roll), 1))
        elif cls == "succ":
            a = first_phase_index(roll, "grasp")
            if a is not None:
                anchored.append((roll, a, 0))
    for t in range(-1, -max_back - 1, -1):
        X, y = [], []
        for roll, a, lab in anchored:
            i = a + t
            if i < 0:
                continue
            X.append(episode_vectors(roll, layer_key, [i])[0])
            y.append(lab)
        y = np.array(y)
        if int((y == 1).sum()) < 2 or int((y == 0).sum()) < 2:
            out[str(t)] = {"auroc": None, "n_wg": int((y == 1).sum()), "n_cmp": int((y == 0).sum())}
            continue
        a_ = ps.loo_auroc(np.stack(X).astype(np.float64), y)
        out[str(t)] = {"auroc": a_, "n_wg": int((y == 1).sum()), "n_cmp": int((y == 0).sum())}
    return out


# --------------------------------------------------------------------------- #
# census / 메인
# --------------------------------------------------------------------------- #
def census(rolls: list[dict]) -> dict:
    per = []
    for roll in rolls:
        cls = classify_episode(roll)
        fw = first_wg_index(roll)
        pd_idx, pd_status = postdrop_window(roll)
        per.append({
            "name": roll["name"], "cls": cls, "len": roll["length"],
            "first_wg_idx": fw,
            "pre_onset_reach": len(window_indices(roll, "W_pre")),
            "reach_total": sum(1 for p in roll["phases"] if p == REACH_PHASE),
            "drop_steps": roll.get("drop_steps"), "grasp_steps": roll.get("grasp_steps"),
            "postdrop_reach": len(pd_idx), "postdrop_status": pd_status,
            "phase_composition": {p: roll["phases"].count(p) for p in dict.fromkeys(roll["phases"])},
        })
    n = {c: sum(1 for e in per if e["cls"] == c) for c in ("succ", "other_fail", "wg")}
    return {"n": n, "per_episode": per}


COMPARISONS = ("rest_all", "succ", "other_fail")


def comparison_labels(rolls: list[dict], comparison: str):
    """(subset_rolls, labels y=1 wg). comparison = 비교군 정의."""
    sub, y = [], []
    for roll in rolls:
        cls = classify_episode(roll)
        if cls == "wg":
            sub.append(roll); y.append(1)
        elif comparison == "rest_all":
            sub.append(roll); y.append(0)
        elif comparison == "succ" and cls == "succ":
            sub.append(roll); y.append(0)
        elif comparison == "other_fail" and cls == "other_fail":
            sub.append(roll); y.append(0)
    return sub, y


def main() -> None:
    ap = argparse.ArgumentParser(description="wrong-grasp VL activation 분리 분석")
    ap.add_argument("--run-dir", required=True, help="raw_rollouts dir (task 상위)")
    ap.add_argument("--cell", default="ppcc_bread", help="cell 디렉토리 이름")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-perm", type=int, default=1000, help="primary/family permutation 수")
    ap.add_argument("--n-perm-grid", type=int, default=200, help="탐색 그리드 permutation 수")
    ap.add_argument("--budgets", default="auto,5,8,13", help="budget sweep (auto=global-min)")
    ap.add_argument("--k-early", default="5,10")
    ap.add_argument("--min-wg", type=int, default=3)
    ap.add_argument("--expect", default=None, help="census assert 'succ,other_fail,wg' (예: 48,5,7)")
    ap.add_argument("--smoke", action="store_true", help="census 만 출력하고 종료")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cell_dirs = sorted(run_dir.glob(f"*/{args.cell}")) or [run_dir / args.cell]
    cell_dir = next((d for d in cell_dirs if d.is_dir()), None)
    if cell_dir is None:
        raise SystemExit(f"cell dir not found: {args.cell} under {run_dir}")
    out_dir = Path(args.out) if args.out else (
        run_dir.parent / "analysis" / "wrong_grasp_vl_separation" / args.cell)

    rolls = [load_rollout_with_events(p) for p in sorted(cell_dir.glob("*.pkl"))]
    if not rolls:
        raise SystemExit(f"no pkl under {cell_dir}")
    cen = census(rolls)
    print(f"[census] {args.cell}: {cen['n']}")
    for e in cen["per_episode"]:
        if e["cls"] == "wg":
            print(f"  wg {e['name']} first_wg={e['first_wg_idx']} pre_reach={e['pre_onset_reach']}")

    if args.expect:
        exp = [int(v) for v in args.expect.split(",")]
        got = [cen["n"]["succ"], cen["n"]["other_fail"], cen["n"]["wg"]]
        assert got == exp, f"census mismatch: expected {exp}, got {got}"
        print(f"[census] expect OK: succ={exp[0]} other_fail={exp[1]} wg={exp[2]}")

    if args.smoke:
        # smoke: budget 규칙 확정에 필요한 분포까지 출력
        for cls in ("succ", "other_fail", "wg"):
            dw = sorted(e["pre_onset_reach"] for e in cen["per_episode"] if e["cls"] == cls)
            print(f"  reach dwell [{cls}] n={len(dw)} min={dw[0] if dw else '-'} "
                  f"med={dw[len(dw)//2] if dw else '-'} max={dw[-1] if dw else '-'}")
        for e in cen["per_episode"]:
            if e["cls"] == "other_fail":
                print(f"  other_fail {e['name']} phases={e['phase_composition']}")
        for e in cen["per_episode"]:
            if e["postdrop_status"] == "ok":
                print(f"  postdrop[{e['cls']}] {e['name']} reach={e['postdrop_reach']} "
                      f"drops={e['drop_steps']}")
            elif e["cls"] == "wg":
                print(f"  postdrop[wg-EXCLUDED] {e['name']} status={e['postdrop_status']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "census.json").write_text(json.dumps(cen, indent=2))
        print(f"[smoke] -> {out_dir / 'census.json'}")
        return

    if cen["n"]["wg"] < args.min_wg:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "census.json").write_text(json.dumps(cen, indent=2))
        print(f"[skip] wg={cen['n']['wg']} < min {args.min_wg} — census 만 기록")
        return

    rng = np.random.default_rng(SEED)
    cap = rolls[0]["capture_layers"]
    layer_keys = list(range(rolls[0]["dit"].shape[1])) + (["VL"] if rolls[0]["vl"] is not None else [])
    layer_labels = {**{str(i): f"DiT-L{cap[i]}" for i in range(len(cap))}, "VL": "VL"}
    k_early = [int(v) for v in args.k_early.split(",")]

    results = {
        "spec_frozen": {
            "primary": "VL x W_pre x wg vs rest_all, budget=global-min(탈락0), "
                       f"LOO-AUROC + episode-perm p (n_perm={args.n_perm}), alpha=0.05",
            "positive_control": "VL x W_at x wg vs succ — 실패 시 파이프라인 무효",
            "layer_family": "W_pre x rest_all max-T familywise (공유 perm)",
            "other_fail_role": "exact perm 병기, descriptive 전용 (확증 불가)",
            "dwell_gate": "없음 — context 통계로만 병기 (equal-budget pool 이 count 경로 차단)",
            "event_matched": "W_postdrop: drop 직후~다음 획득 전 reach, wg vs drop-경험 비교군 "
                             "(메커니즘 실증 후 추가 — W_pre 무신호는 메커니즘상 당연으로 재해석). "
                             "분리는 target-가시성 판독일 가능성 지배적 — 오독 원인 증거 아님.",
            "claim_ceiling": CLAIM_CEILING,
            "scope": SCOPE_NOTE,
        },
        "cell": args.cell, "run_dir": str(run_dir),
        "capture_layers": cap, "layer_labels": layer_labels,
        "census": cen,
    }

    # ---------------- Primary: VL × W_pre × rest_all ----------------
    sub, y = comparison_labels(rolls, "rest_all")
    d_primary = build_design(sub, y, "W_pre", "VL", budget="auto")
    assert d_primary is not None, "primary design failed"
    fp_primary = FoldProjector(d_primary["X"])
    stats = perm_stats(fp_primary, d_primary["y"], args.n_perm, rng)
    signal = abs(stats["auroc"] - 0.5) > (stats["null95_upper"] - 0.5) and stats["p_perm"] < 0.05
    results["primary"] = {
        **stats, "budget": d_primary["budget"],
        "n_wg": d_primary["n_pos"], "n_cmp": d_primary["n_neg"],
        "dropped": d_primary["dropped"],
        "dwell_auroc_context": d_primary["dwell_auroc"],
        "verdict": "SIGNAL" if signal else "NO-SIGNAL",
    }
    print(f"[primary] VL W_pre wg({d_primary['n_pos']}) vs rest({d_primary['n_neg']}) "
          f"budget={d_primary['budget']}: auroc={stats['auroc']:.3f} p={stats['p_perm']:.4f} "
          f"null95={stats['null95_upper']:.3f} dwell(ctx)={d_primary['dwell_auroc']:.3f} "
          f"-> {results['primary']['verdict']}")

    # ---------------- Positive control: VL × W_at × succ ----------------
    sub_at, y_at = comparison_labels(rolls, "succ")
    d_at = build_design(sub_at, y_at, "W_at", "VL", budget="auto")
    if d_at is None:
        results["positive_control"] = {"status": "design failed"}
    else:
        fp_at = FoldProjector(d_at["X"])
        st_at = perm_stats(fp_at, d_at["y"], args.n_perm_grid, rng)
        ok = abs(st_at["auroc"] - 0.5) > (st_at["null95_upper"] - 0.5)
        results["positive_control"] = {**st_at, "budget": d_at["budget"],
                                       "n_wg": d_at["n_pos"], "n_cmp": d_at["n_neg"],
                                       "pipeline_valid": bool(ok)}
        print(f"[positive-control] VL W_at: auroc={st_at['auroc']:.3f} "
              f"null95={st_at['null95_upper']:.3f} -> pipeline_valid={ok}")

    # ---------------- Layer family: W_pre × rest_all, max-T ----------------
    fps = {}
    designs_by_layer = {}
    for lk in layer_keys:
        d = build_design(sub, y, "W_pre", lk, budget="auto")
        if d is None:
            continue
        designs_by_layer[layer_labels[str(lk)]] = d
        fps[layer_labels[str(lk)]] = FoldProjector(d["X"])
    fam = max_t_family(fps, d_primary["y"], args.n_perm, rng)
    results["layer_family"] = fam
    print("[layer-family] W_pre rest_all (max-T fw null95=%.3f):" % fam["null95_fw_upper"])
    for lab, v in sorted(fam["layers"].items(), key=lambda kv: abs(kv[1]["auroc"] - 0.5), reverse=True):
        print(f"    {lab:8s} auroc={v['auroc']:.3f} p_fw={v['p_familywise']:.4f}")

    # ---------------- 탐색 그리드 (exploratory — verdict 없음) ----------------
    grid = {}
    windows = [("W_pre", None)] + [("W_early", k) for k in k_early]
    for cmp_name in COMPARISONS:
        sub_c, y_c = comparison_labels(rolls, cmp_name)
        grid[cmp_name] = {}
        for window, k in windows:
            wname = window if k is None else f"{window}{k}"
            entry = {"layers": {}}
            for lk in layer_keys:
                d = build_design(sub_c, y_c, window, lk, budget="auto", k=k)
                if d is None:
                    continue
                fp = FoldProjector(d["X"])
                st = perm_stats(fp, d["y"], args.n_perm_grid, rng)
                entry["layers"][layer_labels[str(lk)]] = {
                    **st, "budget": d["budget"], "n_wg": d["n_pos"], "n_cmp": d["n_neg"],
                    "dwell_auroc": d["dwell_auroc"]}
            grid[cmp_name][wname] = entry
    results["grid"] = grid

    # ---------------- Robustness ----------------
    rb = {}
    # ① dwell-matched subset: reach dwell 최장 comparator 10개 (primary 설계 재사용)
    cmp_idx = np.flatnonzero(d_primary["y"] == 0)
    order = cmp_idx[np.argsort(-d_primary["counts"][cmp_idx])][:10]
    keep = np.concatenate([np.flatnonzero(d_primary["y"] == 1), order])
    Xm, ym = d_primary["X"][keep], d_primary["y"][keep]
    cm = d_primary["counts"][keep]
    fpm = FoldProjector(Xm)
    stm = perm_stats(fpm, ym, args.n_perm, rng)
    rb["dwell_matched"] = {**stm, "n_wg": int((ym == 1).sum()), "n_cmp": int((ym == 0).sum()),
                           "dwell_auroc": ps.rank_auroc(cm, ym)}
    # ② comparator 내 latent score ↔ reach dwell Spearman
    scores0 = fp_primary.loo_scores(d_primary["y"])
    mask_c = d_primary["y"] == 0
    rb["score_dwell_spearman_cmp"] = spearman(scores0[mask_c], d_primary["counts"][mask_c])
    # ③ budget sweep
    sweep = {}
    for bs in args.budgets.split(","):
        b = "auto" if bs == "auto" else int(bs)
        d = build_design(sub, y, "W_pre", "VL", budget=b)
        if d is None:
            sweep[bs] = {"status": "design failed (class empty after drop)"}
            continue
        fp = FoldProjector(d["X"])
        st = perm_stats(fp, d["y"], args.n_perm_grid, rng)
        sweep[bs] = {**st, "budget": d["budget"], "n_wg": d["n_pos"], "n_cmp": d["n_neg"],
                     "n_dropped": len(d["dropped"]["under_budget"]),
                     "dwell_auroc": d["dwell_auroc"]}
    rb["budget_sweep"] = sweep
    # ④ 라벨-fit 없는 diag-Mahalanobis anomaly (rest 분포 기준)
    rb["mahal_anomaly"] = mahal_perm_stats(d_primary["X"], d_primary["y"], args.n_perm_grid, rng)
    results["robustness"] = rb
    print(f"[robustness] dwell_matched auroc={stm['auroc']:.3f} p={stm['p_perm']:.4f} | "
          f"score~dwell rho={rb['score_dwell_spearman_cmp']:.3f} | "
          f"mahal auroc={rb['mahal_anomaly']['auroc']:.3f}")

    # ---------------- Event-matched: W_postdrop (메커니즘 반영 재설계) ----------------
    pd_out = {}
    for pd_name, keep in [("drop_all", ("succ", "other_fail")), ("drop_succ", ("succ",))]:
        d = build_postdrop_design(rolls, "VL", comparator_cls=keep)
        if d is None:
            pd_out[pd_name] = {"status": "design failed (표본 부족)"}
            continue
        fp = FoldProjector(d["X"])
        st = perm_stats(fp, d["y"], args.n_perm, rng)
        pd_out[pd_name] = {**st, "budget": d["budget"], "n_wg": d["n_pos"], "n_cmp": d["n_neg"],
                           "comparator_cls": list(d["cls"]), "names": d["names"],
                           "dwell_auroc": d["dwell_auroc"], "excluded": d["excluded"]}
        print(f"[postdrop:{pd_name}] wg({d['n_pos']}) vs cmp({d['n_neg']}) budget={d['budget']}: "
              f"auroc={st['auroc']:.3f} p={st['p_perm']:.4f} null95={st['null95_upper']:.3f} "
              f"dwell(ctx)={d['dwell_auroc']:.3f} excluded={ {k: len(v) for k, v in d['excluded'].items()} }")
    # layer 프로파일 (drop_all, max-T)
    pd_fps = {}
    for lk in layer_keys:
        d = build_postdrop_design(rolls, lk, comparator_cls=("succ", "other_fail"))
        if d is not None:
            pd_fps[layer_labels[str(lk)]] = (FoldProjector(d["X"]), d["y"])
    if pd_fps:
        y_pd = next(iter(pd_fps.values()))[1]
        fam_pd = max_t_family({k: v[0] for k, v in pd_fps.items()}, y_pd, args.n_perm, rng)
        pd_out["layer_family_drop_all"] = fam_pd
        print("[postdrop layer-family] (max-T fw null95=%.3f):" % fam_pd["null95_fw_upper"])
        for lab, v in sorted(fam_pd["layers"].items(), key=lambda kv: abs(kv[1]["auroc"] - 0.5),
                             reverse=True):
            print(f"    {lab:8s} auroc={v['auroc']:.3f} p_fw={v['p_familywise']:.4f}")
    results["postdrop"] = pd_out

    # ---------------- t_rel 곡선 (진단) ----------------
    results["trel"] = {"VL": trel_curve(rolls, "VL")}
    print("[trel] VL onset-aligned:",
          {t: (None if v["auroc"] is None else round(v["auroc"], 3))
           for t, v in results["trel"]["VL"].items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wg_vl_separation.json").write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out_dir / 'wg_vl_separation.json'}")

    _plots(results, out_dir)


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def _plots(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # 원격 matplotlib 부재 시 JSON 만으로 충분
        print(f"[plot] skipped: {e}")
        return

    # 1) layer profile (W_pre × rest_all + familywise band)
    fam = results["layer_family"]
    labs = list(fam["layers"].keys())
    vals = [fam["layers"][b]["auroc"] for b in labs]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["tab:red" if b == "VL" else "tab:blue" for b in labs]
    ax.bar(labs, vals, color=colors)
    ax.axhline(0.5, color="gray", lw=0.8)
    hi = fam["null95_fw_upper"]
    ax.axhspan(1 - hi, hi, color="gray", alpha=0.15, label=f"max-T fw null95 [{1-hi:.2f},{hi:.2f}]")
    ax.set_ylim(0, 1)
    ax.set_ylabel("LOO AUROC (wg vs rest)")
    ax.set_title("W_pre layer profile (VL=red)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_profile_wpre.png", dpi=150)
    plt.close(fig)

    # 2) t_rel 곡선
    tr = results["trel"]["VL"]
    ts = sorted((int(t) for t in tr), reverse=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = [t for t in ts if tr[str(t)]["auroc"] is not None]
    ax.plot(xs, [tr[str(t)]["auroc"] for t in xs], marker="o")
    for t in xs:
        ax.annotate(str(tr[str(t)]["n_wg"]), (t, tr[str(t)]["auroc"]), fontsize=6,
                    textcoords="offset points", xytext=(0, 5))
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set_xlabel("t_rel (records before onset; wg=first-wg, succ=first-grasp)")
    ax.set_ylabel("single-record LOO AUROC")
    ax.set_title("VL onset-aligned separation (diagnostic; n_wg annotated)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "trel_curve.png", dpi=150)
    plt.close(fig)

    # 3) budget sensitivity
    sw = results["robustness"]["budget_sweep"]
    keys = [k for k in sw if "auroc" in sw[k]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(len(keys)), [sw[k]["auroc"] for k in keys], marker="o", label="latent")
    ax.plot(range(len(keys)), [sw[k]["dwell_auroc"] for k in keys], marker="s",
            ls="--", label="dwell(context)")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(b={sw[k]['budget']},drop={sw[k]['n_dropped']})" for k in keys],
                       fontsize=7)
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("W_pre budget sensitivity (VL)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "budget_sensitivity.png", dpi=150)
    plt.close(fig)

    # 4) reach dwell 분포
    cen = results["census"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for xi, cls in enumerate(("succ", "other_fail", "wg")):
        dw = [e["pre_onset_reach"] for e in cen["per_episode"] if e["cls"] == cls]
        ax.scatter([xi] * len(dw), dw, alpha=0.5, s=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["succ", "other_fail", "wg"])
    ax.set_ylabel("pre-onset reach records")
    ax.set_title("reach dwell by class (context; equal-budget controls this)")
    fig.tight_layout()
    fig.savefig(out_dir / "dwell_distributions.png", dpi=150)
    plt.close(fig)
    print(f"[plot] 4 figures -> {out_dir}")


if __name__ == "__main__":
    main()
