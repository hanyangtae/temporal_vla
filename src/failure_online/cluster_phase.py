"""activation cluster 기반 **online phase 판정기** (self-contained).

학습·번들 export 코드 = `scripts/analysis/grid_phase/ae_cluster.py`
(`export_bundle()` 이 포맷의 정본). 여기서는 그 NPZ 를 읽어 **serve 안에서 한 record
씩** raw-1536 feature 를 cluster 에 배정한다. 분석 스크립트를 import 하지 않는다
(serve 컨테이너에 분석 의존을 만들지 않기 위해) — 대신 encoder 수식을 arch json 대로
복제하고, `__main__` self-test 가 numpy 참조 구현과 수치 일치를 검증한다.

한 record 의 계산 (docs/steering/47 후속, cluster-k8 라운드):
    hidden [L, K, T, D]  (all_token_full 캡처)
      → layer  : 번들 feature_spec.layer(**물리 layer 번호**)를 capture_layers 에서 역산
      → denoise: 번들 feature_spec.denoise_index (학습 shard 와 같은 k)
      → segment: "all" = 49 토큰 mean → x [1536]
      → 표준화 : (x - mu) / scalar_std   (스칼라 std — 축별 whitening 아님)
      → encode : Linear(1536→256) GELU Linear(256→256) GELU Linear(256→latent)
      → 배정   : 해당 slug 의 centers 최근접 (argmin L2) → phase 이름 "c{idx}"

detector 와 달리 **상태가 없다** (per-record 독립) — episode 경계 reset 불필요.
좌표(layer·denoise·segment)는 `OnlineFailureDetector.feature_from_hidden` 과 동일해야
하며, 토큰 세그먼트 slice 는 그 모듈의 `token_segment_slice` 를 그대로 재사용한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# `python3 src/failure_online/cluster_phase.py` (self-test) 로 직접 돌릴 때도 절대 import 가
# 되도록 repo root 를 먼저 얹는다 — serve 경로에서는 이미 sys.path 에 있어 no-op.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.failure_online.online_failure import token_segment_slice  # noqa: E402

# ae_cluster.export_bundle 의 arch json 이 쓰는 연산 이름
_OPS = {"Linear", "GELU"}


class ClusterPhaseAssigner:
    """AE cluster 번들(NPZ)을 로드해 raw-1536 feature → phase 이름("c3")을 낸다.

    사용:
        asg = ClusterPhaseAssigner.from_bundle("ae_bundle_k8.npz", task="OpenDrawer")
        li = asg.resolve_layer_index(capture_layers)     # 기동 시 preflight
        out = asg.assign(asg.feature_from_hidden(hidden, li))
        # out = {"name": "c3", "idx": 3, "dist": 1.83}
    """

    def __init__(self, mu, scalar_std, encoder, centers, slug, feature, meta,
                 device="cpu"):
        self.device = torch.device(device)
        self.mu = np.asarray(mu, dtype=np.float32).ravel()
        self.scalar_std = float(scalar_std)
        if not np.isfinite(self.scalar_std) or self.scalar_std <= 0:
            raise ValueError(f"cluster-phase: scalar_std={self.scalar_std} 가 비정상 (>0 필요)")
        self.encoder = encoder.to(self.device).eval()
        self.centers = np.asarray(centers, dtype=np.float32)
        if self.centers.ndim != 2:
            raise ValueError(f"cluster-phase: centers ndim={self.centers.ndim} (=[k,latent] 필요)")
        self._centers_t = torch.from_numpy(
            np.ascontiguousarray(self.centers)).float().to(self.device)
        self.slug = str(slug)
        self.feature = dict(feature)
        self.meta = dict(meta)

    # ---------------------------------------------------------------- 로딩
    @classmethod
    def from_bundle(cls, path, task: str | None = None,
                    device: str = "cpu") -> "ClusterPhaseAssigner":
        """`ae_cluster.py --export-bundle` 산출 NPZ → assigner (fail-loud)."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            keys = set(f.files)
            for key in ("mu", "scalar_std", "arch", "k", "latent", "slugs"):
                if key not in keys:
                    raise ValueError(
                        f"{path.name}: 번들에 '{key}' 없음 — "
                        "ae_cluster.py --export-bundle 산출물이 맞나")
            mu = np.asarray(f["mu"], dtype=np.float32).ravel()
            scalar_std = float(np.asarray(f["scalar_std"]).reshape(-1)[0])
            arch = json.loads(str(f["arch"]))
            k = int(np.asarray(f["k"]).reshape(-1)[0])
            latent = int(np.asarray(f["latent"]).reshape(-1)[0])
            slugs = [str(s) for s in np.asarray(f["slugs"]).ravel().tolist()]
            prov = (json.loads(str(f["provenance"]))
                    if "provenance" in keys else {})
            # slug 선택: 지정 없고 번들에 1개뿐이면 그것, 아니면 명시 필수
            if task is None:
                if len(slugs) != 1:
                    raise ValueError(
                        f"{path.name}: 번들에 slug 가 {len(slugs)}개 ({slugs}) — "
                        "--cluster-phase-task 로 어느 instruction 의 centers 를 쓸지 지정할 것 "
                        "(centers 는 instruction 별로 따로 적합된다)")
                task = slugs[0]
            task = str(task)
            if task not in slugs:
                raise ValueError(
                    f"{path.name}: slug '{task}' 없음 (있는 것: {slugs})")
            ckey = f"centers.{task}"
            if ckey not in keys:
                raise ValueError(f"{path.name}: '{ckey}' 배열 없음 (slugs 는 {slugs})")
            centers = np.asarray(f[ckey], dtype=np.float32)
            if centers.shape != (k, latent):
                raise ValueError(
                    f"{path.name}: centers[{task}] shape {centers.shape} != ({k}, {latent})")
            state = {kk: np.asarray(f[kk], dtype=np.float32)
                     for kk in keys if kk.startswith("enc.")}

        encoder, used = _build_encoder(arch, state, path.name)
        unused = sorted(set(state) - used)
        if unused:
            raise ValueError(
                f"{path.name}: encoder state 에 arch 가 안 쓰는 키가 남음 {unused} — "
                "arch json 과 state_dict 이 어긋난다")

        fspec = dict(prov.get("feature_spec") or {})
        input_dim = int(arch.get("input_dim", mu.shape[0]))
        if mu.shape != (input_dim,):
            raise ValueError(f"{path.name}: mu shape {mu.shape} != (input_dim={input_dim},)")
        feature = {
            "layer": int(fspec.get("layer", 12)),
            "denoise_index": int(fspec.get("denoise_index", -1)),
            "seg": str(fspec.get("segment", "all")).split("(")[0],
            "dim": input_dim,
        }
        meta = {
            "bundle": path.name,       # basename 만 (docs/04 §8 — 절대경로 기록 금지)
            "slug": task, "slugs": slugs, "k": k, "latent": latent,
            "input_dim": input_dim,
            "layer": feature["layer"], "denoise_index": feature["denoise_index"],
            "seg": feature["seg"],
            "git_commit": prov.get("git_commit"),
            "shard_dir": prov.get("shard_dir_basename"),
            "seed": prov.get("seed"),
        }
        return cls(mu, scalar_std, encoder, centers, task, feature, meta, device=device)

    # ---------------------------------------------------------------- 좌표
    def resolve_layer_index(self, capture_layers) -> int:
        """물리 layer(번들 feature_spec.layer) → hidden_states 의 L축 인덱스 (fail-loud)."""
        layers = [int(v) for v in (capture_layers or [])]
        want = int(self.feature["layer"])
        if want not in layers:
            raise RuntimeError(
                f"cluster-phase: layer {want} 가 capture layers {layers} 에 없음 "
                f"(--groot-dit-capture-layers 에 {want} 포함 필요)")
        return layers.index(want)

    def feature_from_hidden(self, hidden, layer_idx: int) -> np.ndarray:
        """hidden [L,K,T,D] → 표준화 **전** feature [D] (layer×denoise×segment mean).

        `OnlineFailureDetector.feature_from_hidden` 과 같은 좌표 규약 — 세그먼트 slice 는
        같은 헬퍼를 쓰고, denoise 는 번들이 기록한 학습 시 인덱스를 그대로 쓴다
        (음수면 마지막 k).
        """
        arr = np.asarray(hidden)
        if arr.ndim != 4:
            raise RuntimeError(
                f"cluster-phase: hidden ndim={arr.ndim} — all_token_full [L,K,T,D] 필요 "
                "(--groot-dit-token-pool all_token_full)")
        d = int(self.feature.get("denoise_index", -1))
        ki = arr.shape[1] - 1 if d < 0 else d
        if not (0 <= ki < arr.shape[1]):
            raise RuntimeError(
                f"cluster-phase: denoise {d} 범위 밖 (K={arr.shape[1]}) — 학습 shard 와 "
                "denoise 축 크기가 다르다")
        sl = token_segment_slice(str(self.feature.get("seg", "all")), int(arr.shape[2]))
        feat = arr[layer_idx, ki, sl, :].astype(np.float32).mean(axis=0)
        if feat.shape[0] != self.mu.shape[0]:
            raise RuntimeError(
                f"cluster-phase: feature dim {feat.shape[0]} != 번들 dim {self.mu.shape[0]}")
        return feat

    # ---------------------------------------------------------------- 배정
    @torch.no_grad()
    def encode(self, feats: np.ndarray) -> np.ndarray:
        """raw feature [D] 또는 [n,D] → latent [latent] / [n,latent]."""
        x = np.asarray(feats, dtype=np.float32)
        single = x.ndim == 1
        xb = x.reshape(1, -1) if single else x
        if xb.ndim != 2 or xb.shape[1] != self.mu.shape[0]:
            raise RuntimeError(
                f"cluster-phase: encode 입력 shape {x.shape} != [*, {self.mu.shape[0]}]")
        xs = (xb - self.mu) / np.float32(self.scalar_std)
        z = self.encoder(
            torch.from_numpy(np.ascontiguousarray(xs)).float().to(self.device))
        z = z.cpu().numpy().astype(np.float32)
        return z[0] if single else z

    @torch.no_grad()
    def assign(self, feat: np.ndarray) -> dict:
        """표준화 전 feature [D] → {"name","idx","dist"} (최근접 center)."""
        out = self.assign_batch(np.asarray(feat, dtype=np.float32).reshape(1, -1))
        return out[0]

    @torch.no_grad()
    def assign_batch(self, feats: np.ndarray) -> list[dict]:
        """raw feature [n,D] → 배정 dict 리스트 (배치 경로, 수식은 assign 과 동일)."""
        z = np.atleast_2d(self.encode(np.asarray(feats, dtype=np.float32)))
        zt = torch.from_numpy(np.ascontiguousarray(z)).float().to(self.device)
        d = torch.cdist(zt, self._centers_t)          # [n, k]
        dist, idx = torch.min(d, dim=1)
        return [{"name": f"c{int(i)}", "idx": int(i), "dist": float(v)}
                for i, v in zip(idx.cpu().numpy(), dist.cpu().numpy())]

    # ---------------------------------------------------------------- 지문
    def spec(self) -> dict:
        return dict(self.meta)


def _build_encoder(arch: dict, state: dict, name: str) -> tuple[nn.Module, set]:
    """arch json 의 encoder 층 목록 → nn.Sequential + 사용한 state 키 집합 (fail-loud)."""
    layers_spec = arch.get("encoder")
    if not isinstance(layers_spec, list) or not layers_spec:
        raise ValueError(f"{name}: arch.encoder 가 비었거나 리스트가 아님")
    mods: list[nn.Module] = []
    used: set = set()
    for i, spec in enumerate(layers_spec):
        op = str(spec.get("op"))
        if op not in _OPS:
            raise ValueError(f"{name}: arch.encoder[{i}] op '{op}' 미지원 (허용 {sorted(_OPS)})")
        if op == "GELU":
            # 번들 arch 는 "exact (erf), torch nn.GELU default" 를 명시한다.
            approx = str(arch.get("gelu", "exact"))
            if "tanh" in approx.lower():
                raise ValueError(f"{name}: gelu='{approx}' — tanh 근사는 미지원(학습은 exact)")
            mods.append(nn.GELU())
            continue
        key = str(spec.get("state_key"))
        wk, bk = f"{key}.weight", f"{key}.bias"
        for kk in (wk, bk):
            if kk not in state:
                raise ValueError(
                    f"{name}: encoder state 에 '{kk}' 없음 "
                    f"(있는 것: {sorted(state)})")
        w, b = state[wk], state[bk]
        fan_in, fan_out = int(spec["in"]), int(spec["out"])
        if w.shape != (fan_out, fan_in) or b.shape != (fan_out,):
            raise ValueError(
                f"{name}: '{key}' shape {w.shape}/{b.shape} != "
                f"({fan_out},{fan_in})/({fan_out},)")
        lin = nn.Linear(fan_in, fan_out)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
            lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)))
        mods.append(lin)
        used |= {wk, bk}
    return nn.Sequential(*mods), used


# ---------------------------------------------------------------------------
# self-test: 합성 번들 → 로드 → encoder·배정이 numpy 참조 구현과 일치하는지
# ---------------------------------------------------------------------------

def _synthetic_bundle(path: Path, dim: int = 32, hidden: int = 8, latent: int = 4,
                      k: int = 5, slugs=("SlugA", "SlugB"), denoise_index: int = 3):
    rng = np.random.default_rng(0)
    payload = {
        "mu": rng.normal(0, 0.3, dim).astype(np.float32),
        "scalar_std": np.asarray(1.7, dtype=np.float32),
        "k": np.asarray(k, dtype=np.int32),
        "latent": np.asarray(latent, dtype=np.int32),
        "slugs": np.asarray(list(slugs), dtype=np.str_),
    }
    shapes = {"enc.net.0": (hidden, dim), "enc.net.2": (hidden, hidden),
              "enc.head": (latent, hidden)}
    for key, (o, i) in shapes.items():
        payload[f"{key}.weight"] = rng.normal(0, 0.2, (o, i)).astype(np.float32)
        payload[f"{key}.bias"] = rng.normal(0, 0.1, o).astype(np.float32)
    for s in slugs:
        payload[f"centers.{s}"] = rng.normal(0, 1.0, (k, latent)).astype(np.float32)
    payload["arch"] = np.asarray(json.dumps({
        "encoder": [
            {"op": "Linear", "in": dim, "out": hidden, "state_key": "enc.net.0"},
            {"op": "GELU"},
            {"op": "Linear", "in": hidden, "out": hidden, "state_key": "enc.net.2"},
            {"op": "GELU"},
            {"op": "Linear", "in": hidden, "out": latent, "state_key": "enc.head"},
        ],
        "gelu": "exact (erf), torch nn.GELU default",
        "preprocess": "x_std = (x - mu) / scalar_std  (축별 whitening 아님)",
        "assign": "argmin_c ||enc(x_std) - centers[c]||^2",
        "hidden": hidden, "latent": latent, "input_dim": dim,
        "state_keys": sorted(shapes),
    }, ensure_ascii=False))
    payload["provenance"] = np.asarray(json.dumps({
        "script": "synthetic", "seed": 0, "shard_dir_basename": "synth",
        "feature_spec": {"layer": 12, "denoise_index": denoise_index,
                         "segment": "all(49-token mean)",
                         "layer_axis_index": 5, "dim": dim},
        "git_commit": "0" * 8,
    }, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return payload


def _numpy_reference(payload: dict, x: np.ndarray, slug: str) -> tuple[int, float]:
    """번들 배열만으로 다시 계산한 참조 배정 (torch 경로와 대조용)."""
    from math import erf

    xs = (x.astype(np.float64) - payload["mu"]) / float(payload["scalar_std"])
    gelu = np.vectorize(lambda v: 0.5 * v * (1.0 + erf(v / np.sqrt(2.0))))
    h = xs @ payload["enc.net.0.weight"].T.astype(np.float64) + payload["enc.net.0.bias"]
    h = gelu(h)
    h = h @ payload["enc.net.2.weight"].T.astype(np.float64) + payload["enc.net.2.bias"]
    h = gelu(h)
    z = h @ payload["enc.head.weight"].T.astype(np.float64) + payload["enc.head.bias"]
    cent = payload[f"centers.{slug}"].astype(np.float64)
    d = np.linalg.norm(cent - z[None, :], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def _self_test() -> int:
    import tempfile

    ok = True
    dim, K, T, layers = 32, 4, 49, (0, 2, 4, 8, 10, 12, 15)
    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "ae_bundle_k5.npz"
        payload = _synthetic_bundle(bp, dim=dim)
        asg = ClusterPhaseAssigner.from_bundle(bp, task="SlugA")
        li = asg.resolve_layer_index(list(layers))
        good = li == 5
        ok &= good
        print(f"[self-test] layer_idx={li} (12 → 5) {'OK' if good else 'FAIL'}")

        rng = np.random.default_rng(1)
        hidden = [rng.normal(0, 1, (len(layers), K, T, dim)).astype(np.float16)
                  for _ in range(7)]
        feats = np.stack([asg.feature_from_hidden(h, li) for h in hidden])

        # feature 좌표: denoise_index=3(=K-1), seg 'all' 토큰 mean 과 직접 대조
        ref_feat = hidden[0][li, 3, :, :].astype(np.float32).mean(axis=0)
        good = np.allclose(feats[0], ref_feat, atol=0, rtol=0)
        ok &= good
        print(f"[self-test] feature 좌표(L12·k3·49토큰 mean) {'OK' if good else 'FAIL'}")

        # encoder 수식: torch 경로 vs numpy 참조 (배정 idx·거리)
        errs, mism = [], 0
        for f in feats:
            out = asg.assign(f)
            ri, rd = _numpy_reference(payload, f, "SlugA")
            mism += int(out["idx"] != ri)
            errs.append(abs(out["dist"] - rd))
        good = mism == 0 and max(errs) < 1e-4
        ok &= good
        print(f"[self-test] encoder/배정 vs numpy 참조: idx 불일치 {mism}, "
              f"max|Δdist|={max(errs):.2e} {'OK' if good else 'FAIL'}")

        # 배치 경로 == 단건 경로
        batch = asg.assign_batch(feats)
        single = [asg.assign(f) for f in feats]
        good = all(a["idx"] == b["idx"] and abs(a["dist"] - b["dist"]) < 1e-6
                   for a, b in zip(batch, single))
        ok &= good
        print(f"[self-test] batch == single {'OK' if good else 'FAIL'}")

        # 이름 규약
        good = all(o["name"] == f"c{o['idx']}" for o in batch)
        ok &= good
        print(f"[self-test] phase 이름 'c{{idx}}' {'OK' if good else 'FAIL'}")

        # slug 미지정(2개 이상) / 미등록 slug / 미포함 layer 는 fail-loud
        for kw, exc in ((dict(), ValueError), (dict(task="NoSuch"), ValueError)):
            try:
                ClusterPhaseAssigner.from_bundle(bp, **kw)
                print(f"[self-test] {kw} 가 통과됨 FAIL")
                ok = False
            except exc:
                pass
        try:
            asg.resolve_layer_index([0, 4, 8])
            print("[self-test] layer 미포함이 통과됨 FAIL")
            ok = False
        except RuntimeError:
            pass
        print("[self-test] fail-loud 가드 OK")

        # 4D 아닌 hidden 은 거부
        try:
            asg.feature_from_hidden(np.zeros((K, T, dim), np.float32), li)
            print("[self-test] hidden ndim 가드 통과됨 FAIL")
            ok = False
        except RuntimeError:
            pass

        # slug 1개 번들은 task 생략 가능
        bp1 = Path(td) / "ae_bundle_single.npz"
        _synthetic_bundle(bp1, dim=dim, slugs=("Only",))
        a1 = ClusterPhaseAssigner.from_bundle(bp1)
        good = a1.slug == "Only"
        ok &= good
        print(f"[self-test] 단일 slug 자동 선택 {'OK' if good else 'FAIL'}")

    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
