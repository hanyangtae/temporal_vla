"""학습된 run을 다시 돌려 상태열을 얻는다 — 리포트(잠재·silhouette)와 영상(트랙) 공용.

레거시 viz_infer/viz_common을 대체한다. sys.path 조작도, 존재하지 않는 PlainAE import도
없다: 새 `phase/` 모델·설정만으로 체크포인트를 복원한다.

    bundle = load_run("runs/ae-log_likelihood-s0")
    z      = latents(bundle, steps.x)        # AE/SAE 잠재 [N,d]
    path   = states(bundle, steps.x)         # 상태열 [N] (cls는 phase 예측)
    mp     = phase_mapping(bundle, tr.x, tr.meta["phase_id"].to_numpy())  # state→phase
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phase.analysis.colormap import N_PHASE, state_to_phase
from phase.data.splits import load_split
from phase.models import build_classifier, build_model


def load_cfg(run_dir) -> dict:
    """체크포인트에서 config만 꺼낸다 — 모델을 복원하지 않아도 되는 호출용(예: 데이터 설정)."""
    return torch.load(Path(run_dir) / "model.pt", map_location="cpu",
                      weights_only=False)["config"]


def load_run(run_dir, device="cpu") -> dict:
    """runs/<run>/model.pt → 복원된 모델 번들. cls면 분류기, 아니면 AE/SAE+kmeans 중심."""
    ck = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    mcfg = cfg["model"]
    name = mcfg["name"]
    if name == "cls":
        model = build_classifier(mcfg, n_class=ck.get("n_class", mcfg.get("n_class")))
        model.load_state_dict(ck["state_dict"])
        model.eval().to(device)
        return {"kind": "cls", "name": name, "model": model, "cfg": cfg}

    model = build_model(mcfg)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return {"kind": "cluster", "name": name, "model": model, "cfg": cfg,
            "centers": np.asarray(ck["kmeans_centers"]),
            "active": None if ck.get("active") is None else np.asarray(ck["active"])}


@torch.no_grad()
def latents(bundle, x) -> np.ndarray:
    """AE/SAE 잠재 [N, d]. cls에는 잠재 개념이 없다 (호출하지 말 것)."""
    if bundle["kind"] == "cls":
        raise TypeError("cls 번들에는 latent가 없습니다")
    return bundle["model"].latent(x).cpu().numpy()


def assign_states(bundle, z) -> np.ndarray:
    """잠재 [N,d] → kmeans 상태열 [N] (최근접 중심 = KMeans.predict). active 부분공간만."""
    from sklearn.metrics import pairwise_distances_argmin
    active = bundle["active"]
    za = z if active is None else z[:, active]
    return pairwise_distances_argmin(za, bundle["centers"]).astype(np.int64)


@torch.no_grad()
def predict_phase(bundle, x) -> np.ndarray:
    """cls 예측 phase_id [N]."""
    return bundle["model"].predict(x).cpu().numpy().astype(np.int64)


def states(bundle, x) -> np.ndarray:
    """모델의 상태열 [N]. cls는 phase 예측, AE/SAE는 kmeans 최근접 중심.

    AE/SAE는 인코더가 시점별이고 배정도 스텝 독립이라 미래를 안 본다 → 구조적 causal.
    """
    if bundle["kind"] == "cls":
        return predict_phase(bundle, x)
    return assign_states(bundle, latents(bundle, x))


def phase_mapping(bundle, train_x, train_phase, n_phase=N_PHASE):
    """AE/SAE의 state→GT-phase 다수결 매핑 (train 적합, 누수 방지). cls는 None(이미 phase)."""
    if bundle["kind"] == "cls":
        return None
    st = assign_states(bundle, latents(bundle, train_x))
    return state_to_phase(st, np.asarray(train_phase), len(bundle["centers"]), n_phase)


# ---------------------------------------------------------------- 데이터 헬퍼
def load_splits(data_cfg, device="cpu", which=("train", "val", "test")) -> dict:
    """run의 data 설정(dict)으로 split들을 로드한다."""
    d = data_cfg
    return {s: load_split(d["root"], s, d["split_file"], d["layer"], d["denoise"],
                          d["n_comp"], d["whiten"], d["success_only"], device)
            for s in which}


def episode_rows(steps, episode_id) -> np.ndarray:
    """steps에서 한 에피소드의 행 인덱스 [n] (원래 (episode_id, t) 정렬 유지)."""
    return np.flatnonzero(steps.meta["episode_id"].to_numpy() == episode_id)


# ---------------------------------------------------------------- 다른 split 데이터 먹이기
# run마다 학습 split이 다르고 **split마다 PCA 기저가 다르다**(phase.data.build.make_tag).
# 그래서 어떤 run에 다른 split의 데이터를 넣으려면 그 run 자신의 기저로 다시 투영해야 한다.
# 리포트 여러 개가 "모든 run을 같은 스텝에서 평가한다"를 위해 이 경로를 공유한다.
def read_eval_steps(root, split_file, split, layer, denoise, success_only=False):
    """평가에 쓸 **원본 특징**과 meta — 아직 어떤 PCA도 안 거친 상태.

    run마다 기저가 달라 투영 전 단계에서 한 번만 읽어 공유한다 (1536차원이라 재독이 비싸다).
    """
    from phase.data import source
    raw, meta = source.read_steps(root, split_file, split, layer, denoise, success_only)
    return np.asarray(raw), meta


def split_phase_ids(root, split_file, layer, denoise, n_comp=64, whiten=True,
                    splits=("train", "val", "test")):
    """그 split_file **전체**(기본 train+val+test)의 phase_id 열을 이어붙여 돌려준다.

    "원본 분포에 살아 있는 phase"를 정하는 데 쓴다 — 평가 split만 보면 "표본이 작아서 없다"와
    "클래스가 죽어서 없다"가 섞인다. 캐시의 parquet만 읽어 특징(1536차원)은 건드리지 않는다.
    """
    import pandas as pd

    from phase.data.build import derived_dir, make_tag
    d = derived_dir(root, make_tag(layer, denoise, n_comp, whiten, split_file))
    out = []
    for sp in splits:
        p = d / f"{sp}.steps.parquet"
        if p.exists():
            out.append(pd.read_parquet(p, columns=["phase_id"])["phase_id"].to_numpy())
        else:                                        # 캐시가 없으면 원본에서 (느린 경로)
            out.append(read_eval_steps(root, split_file, sp, layer, denoise)[1]
                       ["phase_id"].to_numpy())
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def train_episodes(bundle, root) -> set:
    """그 run이 학습·검증에 쓴 에피소드 id 집합. 평가에서 누수를 뺄 때 쓴다."""
    import json
    p = Path(root) / "splits" / bundle["cfg"]["data"]["split_file"]
    if not p.exists():
        return set()
    d = json.loads(p.read_text())
    return set(d.get("train", [])) | set(d.get("val", []))


def project_raw(bundle, raw, root) -> np.ndarray:
    """원본 특징 [N,F] → 그 run의 PCA 기저로 투영한 입력 [N,n_comp]."""
    from phase.data import pca
    from phase.data.build import derived_dir, make_tag
    dc = bundle["cfg"]["data"]
    tag = make_tag(dc["layer"], dc["denoise"], dc["n_comp"], dc["whiten"], dc["split_file"])
    f = derived_dir(root, tag) / "pca.npz"
    if not f.exists():
        raise FileNotFoundError(f"PCA 기저가 없습니다: {f} — 그 split의 캐시를 먼저 만드세요")
    return pca.apply(raw, {k: v for k, v in np.load(f).items()}, dc["whiten"])


def active_mask(bundle):
    """[d] bool — 클러스터링이 실제로 쓴 부분공간. 전 차원을 쓰면 None."""
    a = bundle.get("active")
    return None if a is None else np.asarray(a, bool)


def masked_latents(bundle, x) -> np.ndarray:
    """**이미 그 run의 기저로 투영된** 입력 x → 잠재 [N,d] (active 부분공간만).

    run 자신의 split 캐시(`load_splits`의 `st.x`)를 먹일 때 쓴다 — 그 x는 이미 맞는 기저라
    재투영이 필요 없다. 다른 split의 원본 특징을 먹일 때는 `latents_from_raw`(재투영 포함)를 쓸 것.

    **주의**: 반환은 이미 마스킹된 잠재다. 여기 결과를 `assign_states`에 다시 넣으면 SAE에서
    두 번 마스킹돼 차원이 어긋난다 — 상태열은 마스킹 **전** 잠재로 뽑을 것.
    """
    z = latents(bundle, x)
    m = active_mask(bundle)
    return z if m is None else z[:, m]


def latents_from_raw(bundle, raw, root, device="cpu") -> np.ndarray:
    """원본 특징 → 그 run의 잠재 [N,d]. SAE는 클러스터링과 같은 active 부분공간만 남긴다.

    kmeans 중심도 active 부분공간에 있으므로(assign_states 참고), 중심과 거리를 재려면
    반드시 이 함수의 반환과 차원을 맞춰야 한다.
    """
    x = torch.from_numpy(np.ascontiguousarray(project_raw(bundle, raw, root))).to(device)
    return masked_latents(bundle, x)
