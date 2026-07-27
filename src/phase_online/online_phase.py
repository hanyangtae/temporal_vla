"""gr00t DiT layer-12 residual stream → 실시간 action-phase 추론 (self-contained).

task_classification 의 사후 이산화 파이프라인을 **추론 경로만** 그대로 옮긴 것.
학습 코드(패키지)에 의존하지 않고 torch/numpy 만으로 동작하도록 encoder forward 와
KMeans 배정을 직접 구현한다 — gr00t inference 루프 안에서 바로 import 해 쓰기 위함.

파이프라인 (한 step):
    h  : gr00t DiT block layer 12 의 residual stream, 마지막 denoise step, [1536] (또는 [B,1536])
      → x = ((h - mu) @ V.T) / sqrt_lam          # PCA-64 whiten   (pca.npz)
      → z = encoder(x)                            # AE=dense16 / SAE=top-k(16) of relu, 128
      → k = argmin_k ||z[active] - center_k||²    # KMeans 최근접 (model.pt: kmeans_centers)
      → phase = cluster_phase_map[k]              # 비지도 클러스터 → 사람이 읽는 phase 이름

주의:
  * **layer 12 / 마지막 denoise step** 의 residual stream 이어야 체크포인트/PCA 통계와 정렬된다.
  * GMM 은 공분산이 저장돼 있지 않아 재현 불가 → 배정은 KMeans 를 쓴다.
  * cluster→이름 매핑은 val 배정의 다수결(purity 동봉). 클러스터 id 자체가 발견된 상태다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class _AEEncoder(torch.nn.Module):
    """AE 인코더: Linear(64,256)-GELU-Linear(256,256)-GELU-Linear(256,16). latent=출력(활성화 없음)."""

    def __init__(self, input_dim, d, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU())
        self.head = torch.nn.Linear(hidden, d)

    def forward(self, x):
        return self.head(self.net(x))


class _SAEEncoder(torch.nn.Module):
    """SAE 인코더: h = top-k(ReLU(Linear(64,128))). k 개만 남기고 나머지 0 (값 보존)."""

    def __init__(self, input_dim, m, k):
        super().__init__()
        self.net = torch.nn.Linear(input_dim, m)
        self.m, self.k = m, k

    def forward(self, x):
        h = F.relu(self.net(x))
        if self.k < self.m:
            thr = torch.topk(h, self.k, dim=-1).values[..., -1:]
            h = torch.where(h >= thr, h, torch.zeros_like(h))
        return h


class OnlinePhaseClassifier:
    """체크포인트 + PCA + 매핑을 로드해 residual stream → phase 를 배정한다.

    사용:
        clf = OnlinePhaseClassifier.from_run(
            run_dir="task_classification/runs/ae-log_likelihood-s0",
            pca_path="task_classification/datasets_local/phase_cls_pq3/derived/L12-D3-pca64w/pca.npz",
            map_path="task_classification/runs/cluster_phase_map.json",   # 선택
            device="cuda")
        out = clf.infer(h)         # h: [1536] 또는 [B,1536]  (layer12, 마지막 denoise step)
        # out = {"cluster": 13, "phase": "reach-to-object", "purity": 1.0}  (배치면 리스트)
    """

    def __init__(self, encoder, kmeans_centers, active, pca, cluster_map=None,
                 device="cpu", run_name=""):
        self.device = torch.device(device)
        self.enc = encoder.to(self.device).eval()
        self.active = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self.centers = torch.as_tensor(kmeans_centers, dtype=torch.float32,
                                       device=self.device)          # [K, active_dim]
        self.mu = torch.as_tensor(pca["mu"], dtype=torch.float32, device=self.device)
        self.V = torch.as_tensor(pca["V"], dtype=torch.float32, device=self.device)  # [64,1536]
        self.sqrt_lam = torch.as_tensor(pca["sqrt_lam"], dtype=torch.float32, device=self.device)
        self.cluster_map = cluster_map or {}
        self.run_name = run_name

    # ---------------------------------------------------------------- 로딩
    @classmethod
    def from_run(cls, run_dir, pca_path, map_path=None, device="cpu"):
        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        cfg_model = ckpt["config"]["model"]
        sd = ckpt["state_dict"]
        name = cfg_model["name"]
        in_dim = cfg_model["input_dim"]

        if name == "ae":
            enc = _AEEncoder(in_dim, cfg_model["latent_dim"], cfg_model["encoder"]["hidden"])
            enc.load_state_dict({k[len("enc."):]: v for k, v in sd.items() if k.startswith("enc.")})
        elif name == "sae":
            enc = _SAEEncoder(in_dim, cfg_model["latent_dim"], cfg_model["encoder"]["k"])
            enc.load_state_dict({k[len("enc."):]: v for k, v in sd.items() if k.startswith("enc.")})
        else:
            raise ValueError(f"지원하지 않는 모델: {name} (ae/sae 만)")

        pca = np.load(pca_path)
        cluster_map = None
        if map_path is not None and Path(map_path).exists():
            allmaps = json.loads(Path(map_path).read_text())
            # {run_name: {cluster: {...}}} 또는 {cluster: {...}} 둘 다 허용
            cluster_map = allmaps.get(run_dir.name, allmaps)
        return cls(enc, ckpt["kmeans_centers"], ckpt["active"],
                   {k: pca[k] for k in pca.files}, cluster_map, device, run_dir.name)

    # ---------------------------------------------------------------- 추론
    @torch.no_grad()
    def _cluster(self, h):
        """h [B,1536] → cluster id [B] (KMeans 최근접, 유클리드)."""
        h = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        if h.ndim == 1:
            h = h[None]
        x = ((h - self.mu) @ self.V.T) / self.sqrt_lam          # PCA-64 whiten [B,64]
        z = self.enc(x)                                         # latent [B,d]
        z = z[:, self.active]                                   # active 부분공간
        d2 = torch.cdist(z, self.centers)                       # [B,K]
        return d2.argmin(1)

    @torch.no_grad()
    def infer(self, h):
        """residual stream → phase. 단일 입력이면 dict, 배치면 list[dict]."""
        ks = self._cluster(h).tolist()
        outs = []
        for k in ks:
            info = self.cluster_map.get(str(k), self.cluster_map.get(k, {}))
            outs.append({"cluster": int(k),
                         "phase": info.get("name"),
                         "purity": info.get("purity")})
        return outs[0] if len(outs) == 1 else outs


if __name__ == "__main__":
    # 스모크 테스트: 임의 1536-d 입력으로 파이프라인이 도는지 확인
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="task_classification")
    p.add_argument("--run", default="runs/ae-log_likelihood-s0")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    root = Path(a.root)
    clf = OnlinePhaseClassifier.from_run(
        run_dir=root / a.run,
        pca_path=root / "datasets_local/phase_cls_pq3/derived/L12-D3-pca64w/pca.npz",
        map_path=root / "runs/cluster_phase_map.json",
        device=a.device)
    g = torch.Generator().manual_seed(0)
    h = torch.randn(4, 1536, generator=g)
    print(f"[{clf.run_name}] random-input smoke test:")
    for o in clf.infer(h):
        print("  ", o)
