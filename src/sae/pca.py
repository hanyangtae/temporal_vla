"""PCA + whitening (train split 전용).

출처: task_classification@88543a2 `phase/data/pca.py`
      (https://github.com/robots-oh/task_classification)
이식 근거: docs/steering/29_sae_port_review.md §5.A 마지막 항목("선택 사용")

★ **G1 에선 미사용 — 원본 1536-d 를 SAE 에 직접 넣는다.**
   이유: 동료 세팅은 1536→PCA64 로 압축한 뒤 m=128 사전을 얹어 **PCA64 대비만** 과완비였다.
   scene-feature 를 분리하려면 원본 D 에 대해 overcomplete(m = 4~8×D) 여야 한다
   (docs/steering/30_sae_g1_port_handout.md §0 용어, §2.2-2, §4 Phase C1).
   표준화는 PCA 대신 **train-split feature-wise mean/std** 만 쓴다(핸드아웃 §4 B4).
   코드는 나중 비교 실험(압축 입력 대조군)을 위해 확보만 해 둔다. 원본 대비 로직 변경 없음.

--- 원본 모듈 설명 (보존) ---
[왜 whitening인가 — stage0 진단]
    layer 12에서 PC1은 phase 설명력 η²=0.261(분산 13.2%)인데 PC5는 η²=0.215로 거의
    맞먹으면서 분산은 2.2%뿐이다. 재구성 손실은 분산에 비례해 방향을 배분하므로,
    whitening 없이는 phase 정보를 담은 저분산 방향이 버려진다. 동시에 PC2(분산 11.1%)는
    η²_cell=0.168인 nuisance 방향이라 오히려 과대표집된다.
"""
from __future__ import annotations

import numpy as np
import torch


def fit(x, n_comp, device="cpu"):
    """[N, F] → (stats, evr). 공분산 고유분해(N > F일 때 F×F가 싸다).

    [반환]
        stats  {"mu": [F], "V": [n_comp, F], "sqrt_lam": [n_comp]}
        evr    [n_comp] 성분별 설명 분산 비율
    """
    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    lam_all, V = torch.linalg.eigh(cov.double())            # 오름차순
    lam = lam_all.flip(0)[:n_comp].clamp_min(1e-8)
    V = V.flip(1)[:, :n_comp].T.contiguous()                # [n_comp, F]
    evr = (lam / lam_all.clamp_min(0).sum()).cpu().numpy()

    return ({"mu": mu.cpu().numpy(),
             "V": V.float().cpu().numpy(),
             "sqrt_lam": lam.sqrt().float().cpu().numpy()}, evr)


def apply(x, stats, whiten=True):
    """[N, F] → [N, n_comp] float32. whiten이면 각 성분을 sqrt(고윳값)으로 나눈다."""
    z = (x - stats["mu"]) @ stats["V"].T
    return (z / stats["sqrt_lam"] if whiten else z).astype(np.float32)
