"""SAFE 식 failure detector 의 **온라인 1-step 전진** 경로 (self-contained).

학습·시뮬 코드 = `scripts/analysis/grid_phase/failure_detector_sim.py`
(그 자체가 `scripts/safe/groot_n16/robocasa/analyze/pathway_lstm_detector.py` +
`vis/core/lstm.py` 의 아키텍처·손실·정규화 이식본). 여기서는 그 스크립트가 저장한
`detector_<arm>_<model>_<slug|all>.pt` 를 읽어 **serve 안에서 한 step 씩** 점수를 낸다.
학습 스크립트를 import 하지 않는다 (serve 컨테이너에 분석 스크립트 의존을 만들지 않기
위해) — 대신 아키텍처를 동일하게 복제하고, self-test 가 전체-시퀀스 forward 와
1-step 전진의 수치 일치를 검증한다.

한 step 의 계산 (`failure_detector_sim` 의 feature 좌표 규약과 동일):
    hidden [L, K, T, D]  (all_token_full 캡처)
      → layer  : ckpt feature.layer(**물리 layer 번호**)를 serve 의 capture_layers 에서 역산
      → denoise: ckpt feature.denoise (-1 = 마지막 k)
      → segment: ckpt feature.seg (state|future|action|all) 의 토큰축 mean → [D]
      → 표준화 : (x - std_mean) / std_std
      → score  : LSTM 1-step (h,c 캐리) | MLP per-step 출력의 **누적평균**
      → fired  : score > δ_t,  δ = ckpt cp_bands[task][α] 의 delta (t 가 밴드 길이를
                 넘으면 마지막 값 유지 = sim 의 `fire_step` 규약)

episode 경계에서 `reset()` 을 부르지 않으면 (h,c)·누적평균·step 카운터가 이어져
점수가 오염된다 — serve 는 `/reset` 에서 호출한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# extract_grid_matrix.token_segments / fit_phase_conceptor._token_segments 와 같은 규약.
N_FUTURE = 32  # state 1 + future 32 + action H = T


class LSTMDetector(nn.Module):
    """단층 LSTM + linear + sigmoid (failure_detector_sim.LSTMDetector 와 동일 구조)."""

    cumulative = False

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # [B,T,D] → [B,T]
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


class MLPDetector(nn.Module):
    """per-step MLP→scalar→sigmoid. 검출 score 는 출력의 누적평균 (SAFE-MLP 규약)."""

    cumulative = True

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):  # [B,T,D] → [B,T]
        return torch.sigmoid(self.net(x)).squeeze(-1)


def token_segment_slice(seg: str, T: int, n_future: int = N_FUTURE) -> slice:
    """세그먼트 이름 → 토큰축 slice. 'all' 은 전체 토큰."""
    if seg == "all":
        return slice(0, T)
    if seg == "state":
        return slice(0, 1)
    if seg == "future":
        return slice(1, 1 + n_future)
    if seg == "action":
        horizon = T - 1 - n_future
        if horizon <= 0:
            raise ValueError(
                f"segment 'action': 토큰수 T={T} 가 1+{n_future}+H 가정에 안 맞는다")
        return slice(T - horizon, T)
    raise ValueError(f"알 수 없는 segment '{seg}' (state|future|action|all)")


class OnlineFailureDetector:
    """detector 체크포인트 + CP 밴드를 로드해 step 별 failure score/발화를 낸다.

    사용:
        det = OnlineFailureDetector.from_checkpoint("detector_mixed_lstm_all.pt",
                                                    alpha=0.2, task=None)
        li = det.resolve_layer_index(capture_layers)   # 기동 시 preflight
        det.reset()                                    # episode 경계
        out = det.step(det.feature_from_hidden(hidden, li))
        # out = {"score": 0.71, "fired": True, "delta": 0.63, "t": 4}
    """

    def __init__(self, model, std_mean, std_std, feature, band, meta, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.std_mean = np.asarray(std_mean, dtype=np.float32)
        self.std_std = np.asarray(std_std, dtype=np.float32)
        self.feature = dict(feature)
        self.band = band  # {"mu","sd","bw","delta"} — delta 만 발화 판정에 쓴다
        self.meta = dict(meta)
        self.cumulative = bool(getattr(model, "cumulative", False))
        self.reset()

    # ---------------------------------------------------------------- 로딩
    @classmethod
    def from_checkpoint(cls, path, alpha: float = 0.2, task: str | None = None,
                        device: str = "cpu") -> "OnlineFailureDetector":
        path = Path(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "input_dim", "std_mean", "std_std", "feature",
                    "cp_bands", "model"):
            if key not in ck:
                raise ValueError(
                    f"{path.name}: detector ckpt 에 '{key}' 없음 — "
                    "failure_detector_sim.py 가 저장한 파일이 맞나")
        kind = str(ck["model"])
        input_dim = int(ck["input_dim"])
        hidden = int(ck.get("hidden", 256))
        if kind == "lstm":
            model = LSTMDetector(input_dim, hidden)
        elif kind == "mlp":
            model = MLPDetector(input_dim, hidden)
        else:
            raise ValueError(f"{path.name}: 알 수 없는 detector model '{kind}' (lstm|mlp)")
        model.load_state_dict({k: torch.as_tensor(v) for k, v in ck["state_dict"].items()})

        feature = dict(ck["feature"])
        if int(feature.get("dim", input_dim)) != input_dim:
            raise ValueError(
                f"{path.name}: feature.dim {feature.get('dim')} != input_dim {input_dim}")
        std_mean = np.asarray(ck["std_mean"], dtype=np.float32).ravel()
        std_std = np.asarray(ck["std_std"], dtype=np.float32).ravel()
        if std_mean.shape != (input_dim,) or std_std.shape != (input_dim,):
            raise ValueError(
                f"{path.name}: 표준화 통계 shape {std_mean.shape}/{std_std.shape} "
                f"!= (input_dim={input_dim},)")

        bands = ck["cp_bands"] or {}
        if not bands:
            raise ValueError(f"{path.name}: cp_bands 가 비어 있다 — 발화 임계 δ 를 만들 수 없다")
        if task is None:
            if len(bands) != 1:
                raise ValueError(
                    f"{path.name}: cp_bands 에 task 가 {len(bands)}개 "
                    f"({sorted(bands)}) — --failure-task 로 어느 밴드를 쓸지 지정할 것 "
                    "(mixed arm 은 밴드가 task 별로 보정된다)")
            task = next(iter(bands))
        if task not in bands:
            raise ValueError(f"{path.name}: cp_bands 에 task '{task}' 없음 (있는 것: {sorted(bands)})")
        akey = f"{float(alpha):.2f}"
        per_alpha = bands[task]
        if akey not in per_alpha:
            raise ValueError(
                f"{path.name}: task '{task}' 에 α={akey} 밴드 없음 "
                f"(있는 것: {sorted(per_alpha)})")
        raw = per_alpha[akey]
        band = {k: np.asarray(v, dtype=np.float32).ravel() if k != "bw" else float(v)
                for k, v in raw.items()}
        if "delta" not in band or band["delta"].size == 0:
            raise ValueError(f"{path.name}: α={akey} 밴드에 delta 가 없다")

        meta = {
            "ckpt": path.name,          # basename 만 (docs/04 §8 — 절대경로 기록 금지)
            "arm": ck.get("arm"), "model": kind, "group": ck.get("group"),
            "task": task, "alpha": float(alpha), "bw": band["bw"],
            "band_L": int(band["delta"].size),
            "tasks": list(ck.get("tasks") or []),
            "shards": list(ck.get("shards") or []),
            "hidden": hidden, "input_dim": input_dim,
            "layer": feature.get("layer"), "denoise": feature.get("denoise"),
            "seg": feature.get("seg"),
        }
        return cls(model, std_mean, std_std, feature, band, meta, device=device)

    # ---------------------------------------------------------------- 좌표
    def resolve_layer_index(self, capture_layers) -> int:
        """물리 layer(ckpt feature.layer) → hidden_states 의 L축 인덱스 (fail-loud)."""
        layers = [int(v) for v in (capture_layers or [])]
        want = int(self.feature["layer"])
        if want not in layers:
            raise RuntimeError(
                f"failure-detector: layer {want} 가 capture layers {layers} 에 없음 "
                f"(--groot-dit-capture-layers 에 {want} 포함 필요)")
        return layers.index(want)

    def feature_from_hidden(self, hidden, layer_idx: int) -> np.ndarray:
        """hidden [L,K,T,D] → 표준화 **전** feature [D] (layer×denoise×segment mean)."""
        arr = np.asarray(hidden)
        if arr.ndim != 4:
            raise RuntimeError(
                f"failure-detector: hidden ndim={arr.ndim} — all_token_full [L,K,T,D] 필요 "
                "(--groot-dit-token-pool all_token_full)")
        d = int(self.feature.get("denoise", -1))
        ki = arr.shape[1] - 1 if d < 0 else d
        if not (0 <= ki < arr.shape[1]):
            raise RuntimeError(
                f"failure-detector: denoise {d} 범위 밖 (K={arr.shape[1]})")
        sl = token_segment_slice(str(self.feature.get("seg", "all")), int(arr.shape[2]))
        feat = arr[layer_idx, ki, sl, :].astype(np.float32).mean(axis=0)
        if feat.shape[0] != self.std_mean.shape[0]:
            raise RuntimeError(
                f"failure-detector: feature dim {feat.shape[0]} != 학습 dim "
                f"{self.std_mean.shape[0]}")
        return feat

    # ---------------------------------------------------------------- 전진
    def reset(self) -> None:
        """episode 경계: LSTM (h,c)·누적평균·step 카운터 초기화."""
        self._hc = None
        self._sum = 0.0
        self._t = 0

    @torch.no_grad()
    def step(self, feat: np.ndarray) -> dict:
        """표준화 전 feature [D] → {"score","fired","delta","t"} 를 내고 상태를 1 전진."""
        x = (np.asarray(feat, dtype=np.float32) - self.std_mean) / self.std_std
        xb = torch.from_numpy(np.ascontiguousarray(x)).float().view(1, 1, -1).to(self.device)
        if self.cumulative:
            raw = float(torch.sigmoid(self.model.net(xb)).view(-1)[0])
            self._sum += raw
            score = self._sum / (self._t + 1)
        else:
            out, self._hc = self.model.lstm(xb, self._hc)
            score = float(torch.sigmoid(self.model.fc(out)).view(-1)[0])
        delta = self.band["delta"]
        # t 가 밴드 길이를 넘으면 마지막 밴드 유지 (sim 의 fire_step 과 동일 규약)
        d_t = float(delta[min(self._t, delta.size - 1)])
        rec = {"score": float(score), "fired": bool(score > d_t), "delta": d_t,
               "t": int(self._t)}
        self._t += 1
        return rec

    # -------------------------------------------------- 롤백/재전진 (per-step 게이팅)
    def snapshot(self) -> dict:
        """현재 detector 상태를 떠 놓는다 (docs/steering/47 per-step 게이팅).

        serve 는 1차 무개입 pass 의 x_t 로 발화를 판정한 뒤, 발화하면 이 스냅샷으로
        되돌려 2차 steered pass 의 x_t' 로 다시 전진시킨다. LSTM (h,c) 는 이후 step 이
        같은 텐서를 덮어쓰지 않도록 반드시 clone 한다. MLP 경로는 `_hc` 가 None 이다.
        """
        hc = self._hc
        if hc is None:
            hc_snap = None
        else:
            if not (isinstance(hc, tuple) and len(hc) == 2):
                raise RuntimeError(
                    f"failure-detector: snapshot 불가 — _hc 가 (h,c) 튜플이 아님 ({type(hc)})")
            hc_snap = (hc[0].detach().clone(), hc[1].detach().clone())
        return {"_hc": hc_snap, "_sum": float(self._sum), "_t": int(self._t)}

    def restore(self, snap: dict) -> None:
        """`snapshot()` 이 뜬 상태로 되돌린다 (fail-loud: 키 누락이면 즉시 예외)."""
        if not isinstance(snap, dict):
            raise RuntimeError(
                f"failure-detector: restore 인자가 dict 가 아님 ({type(snap)})")
        for key in ("_hc", "_sum", "_t"):
            if key not in snap:
                raise RuntimeError(
                    f"failure-detector: restore snapshot 에 '{key}' 없음 "
                    f"(있는 것: {sorted(snap)}) — snapshot() 산출물이 맞나")
        hc = snap["_hc"]
        if hc is None:
            self._hc = None
        else:
            if not (isinstance(hc, tuple) and len(hc) == 2):
                raise RuntimeError(
                    f"failure-detector: restore 의 _hc 가 (h,c) 튜플이 아님 ({type(hc)})")
            # 복원본을 또 clone: 같은 스냅샷으로 여러 번 되돌려도 서로 오염되지 않게.
            self._hc = (hc[0].detach().clone(), hc[1].detach().clone())
        self._sum = float(snap["_sum"])
        self._t = int(snap["_t"])

    # ---------------------------------------------------------------- 지문
    def spec(self) -> dict:
        return dict(self.meta)


# ---------------------------------------------------------------------------
# self-test: 합성 ckpt 로 로드 → 1-step 전진 == 전체-시퀀스 forward → reset 확인
# ---------------------------------------------------------------------------

def _synthetic_ckpt(path: Path, kind: str, dim: int = 16, layers=(0, 4, 12),
                    K: int = 4, T: int = 49, band_L: int = 6) -> None:
    torch.manual_seed(0)
    model = LSTMDetector(dim, 8) if kind == "lstm" else MLPDetector(dim, 8)
    rng = np.random.default_rng(0)
    torch.save({
        "arm": "mixed", "model": kind, "group": "__all__",
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": dim, "hidden": 8,
        "std_mean": rng.normal(0, 0.1, dim).astype(np.float32),
        "std_std": np.abs(rng.normal(1.0, 0.05, dim)).astype(np.float32),
        "feature": {"layer": 12, "denoise": -1, "seg": "all",
                    "layer_idx": list(layers).index(12), "denoise_idx": K - 1,
                    "seg_idx": 3, "dim": dim},
        "cp_bands": {"SynthTask": {"0.20": {
            "mu": np.full(band_L, 0.4, np.float32),
            "sd": np.full(band_L, 0.1, np.float32),
            "bw": 1.5,
            "delta": np.linspace(0.55, 0.45, band_L).astype(np.float32)}}},
        "tasks": ["SynthTask"], "shards": ["SynthTask.npz"],
        "train": {"epochs": 1},
    }, path)


def _self_test() -> int:
    import tempfile

    ok = True
    layers = (0, 4, 12)
    K, T, dim, steps = 4, 49, 16, 9
    with tempfile.TemporaryDirectory() as td:
        for kind in ("lstm", "mlp"):
            ck = Path(td) / f"detector_mixed_{kind}_all.pt"
            _synthetic_ckpt(ck, kind, dim=dim, layers=layers, K=K, T=T)
            det = OnlineFailureDetector.from_checkpoint(ck, alpha=0.2)
            li = det.resolve_layer_index(list(layers))
            assert li == 2, li

            rng = np.random.default_rng(1)
            hidden = [rng.normal(0, 1, (len(layers), K, T, dim)).astype(np.float16)
                      for _ in range(steps)]
            feats = np.stack([det.feature_from_hidden(h, li) for h in hidden])

            det.reset()
            online = np.array([det.step(f)["score"] for f in feats])

            # ground truth = 전체 시퀀스 forward (sim.score_seq 와 같은 계산)
            xs = torch.from_numpy(
                ((feats - det.std_mean) / det.std_std).astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                raw = det.model(xs).squeeze(0).numpy().astype(np.float64)
            batch = np.cumsum(raw) / np.arange(1, len(raw) + 1) if det.cumulative else raw
            err = float(np.max(np.abs(online - batch)))
            good = err < 1e-5
            ok &= good
            print(f"[self-test] {kind}: online vs batch max|Δ|={err:.2e} "
                  f"{'OK' if good else 'FAIL'}")

            # 밴드 plateau: t 가 band_L(6) 을 넘어도 마지막 δ 유지
            det.reset()
            rec = [det.step(f) for f in feats]
            last = det.band["delta"][-1]
            plateau = all(abs(r["delta"] - last) < 1e-6 for r in rec[6:])
            ok &= plateau
            print(f"[self-test] {kind}: δ plateau {'OK' if plateau else 'FAIL'} "
                  f"(t≥6 δ={rec[-1]['delta']:.4f}, band_L={det.meta['band_L']})")

            # reset 이 상태를 정말 지우는지 (첫 step 점수 동일 재현)
            det.reset()
            s0 = det.step(feats[0])["score"]
            same = abs(s0 - rec[0]["score"]) < 1e-9
            ok &= same
            print(f"[self-test] {kind}: reset 재현 {'OK' if same else 'FAIL'}")

            # snapshot/restore: 중간에서 뜬 상태로 되돌리면 이후 점수가 bit 일치
            mid = 4
            det.reset()
            for f in feats[:mid]:
                det.step(f)
            snap = det.snapshot()
            tail_a = [det.step(f)["score"] for f in feats[mid:]]
            det.restore(snap)
            tail_b = [det.step(f)["score"] for f in feats[mid:]]
            bitsame = all(a == b for a, b in zip(tail_a, tail_b))
            # 원 시퀀스(rec) 와도 일치해야 한다 = 스냅샷이 진짜 그 시점 상태
            vs_full = all(a == r["score"] for a, r in zip(tail_a, rec[mid:]))
            ok &= bitsame and vs_full
            print(f"[self-test] {kind}: snapshot/restore bit 일치 "
                  f"{'OK' if bitsame else 'FAIL'} / 원시퀀스 대조 "
                  f"{'OK' if vs_full else 'FAIL'} (mid={mid})")

            # 발화 후 재전진 시나리오: 같은 t 에서 다른 입력(x') 로 다시 전진해도
            # 상태가 x' 기준으로만 커밋되는지 (스냅샷 복원 → 다른 feature → 원복 확인)
            det.restore(snap)
            alt = feats[mid] + 0.5
            det.step(alt)
            det.restore(snap)
            after = det.step(feats[mid])["score"]
            clean = (after == rec[mid]["score"])
            ok &= clean
            print(f"[self-test] {kind}: 롤백 후 재전진 오염 없음 "
                  f"{'OK' if clean else 'FAIL'}")

            # 미등록 α / task 는 fail-loud
            for bad in (dict(alpha=0.05), dict(task="NoSuchTask")):
                try:
                    OnlineFailureDetector.from_checkpoint(ck, **bad)
                    print(f"[self-test] {kind}: {bad} 가 통과됨 FAIL")
                    ok = False
                except ValueError:
                    pass
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
