"""best-of-N 재샘플 후보를 고르는 **(scene, phase) 조건부 LLR 채점기** (self-contained).

per-step 게이트가 발화한 record 에서 DiT-only 재샘플 후보를 n 개 뽑았을 때, 각 후보의
raw-1536 활성화를 AE latent 로 접어 "실패 가우시안 대비 성공 가우시안" 로그우도비로
점수를 매긴다 (`llr` 이 작을수록 성공 쪽). serve 는 llr argmin 후보를 실행한다.

연산자 등록 단위는 **(scene, phase)** 다 — 번들은 task 당 1개(NPZ)이고 scene 은 키에
들어간다. serve 는 기동 인자(--llr-scene)로 scene 을 고정하고 phase 만 online 으로 정한다.

`cluster_phase.ClusterPhaseAssigner` 와 **같은 AE 계열**(같은 encoder 명명·같은 exact
erf GELU)이며, 여기서는 serve 에 torch 의존을 더하지 않기 위해 numpy 만 쓴다. 분석
스크립트를 import 하지 않는다 — 대신 `__main__` self-test 가 왕복 로드를 검증한다.

한 후보의 계산:
    hidden [L,K,T,D] → (serve 쪽 기존 추출 헬퍼: layer×denoise×49토큰 mean) → x [1536]
      → 표준화 : z = (x - scaler_mu) / scaler_std     (스칼라 std — 축별 whitening 아님)
      → encode : Linear(1536→256) GELU Linear(256→256) GELU Linear(256→16)
      → 중심화 : z~ = latent - succ_mean[e]           (등록 엔트리 e 의 성공 중심 기준 좌표)
      → 점수   : llr = logN(z~; mu_f, cov_f) - logN(z~; mu_s, cov_s)
      → OOD    : max(log_s, log_f) < ood_lo[e] 이면 그 entry 에 대해 후보 기각

serve 의 실제 채점 경로는 `score_nearest(vec, scene)` 다 — **"현재 cluster phase" 로
entry 를 고르지 않는다**. 발화 시점의 online cluster 가 등록 entry 와 전면 불일치해
(scene, phase) 조회는 전 케이스 fallback 으로 퇴화했기 때문(연산자 설계 세션 실측).
대신 그 scene 의 모든 등록 entry 중 **OOD 가 아니면서 succ_mean 이 latent 공간에서 가장
가까운** entry 에 후보를 배정하고 그 entry 의 llr 을 쓴다. `score(vec, scene, phase)` 는
진단용으로 남긴다.

★ NPZ 계약 (이 docstring 이 단일 출처 — export 코드는 이 키들을 그대로 쓸 것)
    meta               : json str — {"task":..., "ae_ref":..., "scenes":[...], "phases":[...]}
    scaler_mu          : float32 [1536]
    scaler_std         : float32 scalar (>0)
    enc.0.weight/bias  : float32 [256,1536] / [256]
    enc.2.weight/bias  : float32 [256,256]  / [256]
    enc.4.weight/bias  : float32 [16,256]   / [16]
    registered         : str 배열 — 점수를 낼 수 있는 (scene, phase) 엔트리 이름들.
                         원소 형식 **"s{scene}__c{k}"** (예: "s3__c2").
    각 등록 엔트리 e ("s3__c2" 등) 에 대해:
      succ_mean.<e>    : float32 [16]      — 성공 latent 평균 (중심화 기준점)
      mu_s.<e>         : float32 [16]      — 중심화 좌표에서의 성공 가우시안 평균
      cov_s.<e>        : float32 [16,16]   — 성공 공분산 (raw — 정칙화는 로드 시)
      mu_f.<e>         : float32 [16]      — 실패 가우시안 평균
      cov_f.<e>        : float32 [16,16]   — 실패 공분산 (raw)
      ood_lo.<e>       : float32 scalar    — train max(logN_s,logN_f) 의 5퍼센타일

공분산은 로드 시 대칭화 후 1e-3·tr(Σ)/16·I 로 정칙화하고 cholesky 를 미리 잡는다
(특이 공분산으로 serve 가 런타임에 죽는 것 방지 — 로드 시점 fail-loud).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LATENT_DIM = 16
INPUT_DIM = 1536
# cluster_phase.py 번들과 동일한 encoder 층 명명 (Linear-GELU-Linear-GELU-Linear)
_ENC_KEYS = ("enc.0", "enc.2", "enc.4")
_COV_REG = 1e-3  # 정칙화 계수: Σ ← Σ + _COV_REG·tr(Σ)/d·I
_ERF = np.vectorize(math.erf, otypes=[np.float64])


def normalize_scene(scene) -> str:
    """scene 을 번들 키 표기 "s{n}" 으로 정규화 (int·"3"·"s3" 수용, 그 외 fail-loud)."""
    if scene is None:
        raise ValueError("llr: scene 이 None — (scene, phase) 등록 단위라 scene 필수")
    if isinstance(scene, (int, np.integer)) and not isinstance(scene, bool):
        return f"s{int(scene)}"
    s = str(scene).strip()
    body = s[1:] if s[:1].lower() == "s" else s
    if not body.isdigit():
        raise ValueError(f"llr: scene {scene!r} 형식 불명 (int 또는 's3' 필요)")
    return f"s{int(body)}"


def entry_key(scene, phase) -> str:
    """(scene, phase) → 번들 등록 엔트리 키 "s{scene}__c{k}"."""
    p = str(phase).strip()
    if not p:
        raise ValueError("llr: phase 가 빈 문자열")
    return f"{normalize_scene(scene)}__{p}"


def _gelu(x: np.ndarray) -> np.ndarray:
    """exact GELU (erf) — cluster_phase.py 의 torch ``nn.GELU()`` 기본과 같은 수식."""
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)))


def _regularize(cov: np.ndarray, name: str) -> np.ndarray:
    """대칭화 + 1e-3·tr/d·I 정칙화 (fail-loud: PD 아니면 로드 시 에러)."""
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (LATENT_DIM, LATENT_DIM):
        raise ValueError(f"llr: '{name}' shape {c.shape} != ({LATENT_DIM},{LATENT_DIM})")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"llr: '{name}' 에 비유한 값")
    c = 0.5 * (c + c.T)
    tr = float(np.trace(c))
    if not (tr > 0):
        raise ValueError(f"llr: '{name}' trace={tr} (>0 필요)")
    return c + (_COV_REG * tr / LATENT_DIM) * np.eye(LATENT_DIM)


def _solve_lower(low: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L y = b 전방대입 (scipy 없이 — serve/원격 env 에 scipy 가 없다)."""
    n = low.shape[0]
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = (b[i] - low[i, :i] @ y[:i]) / low[i, i]
    return y


class _Gaussian:
    """cholesky 사전계산 다변량 정규 — logpdf 만 제공."""

    def __init__(self, mu: np.ndarray, cov: np.ndarray, name: str):
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        if self.mu.shape != (LATENT_DIM,):
            raise ValueError(f"llr: '{name}' mu shape {self.mu.shape} != ({LATENT_DIM},)")
        self.cov = _regularize(cov, name)
        try:
            self.chol = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as exc:  # 정칙화 후에도 실패 = 번들 이상
            raise ValueError(f"llr: '{name}' 공분산이 정칙화 후에도 PD 아님") from exc
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self.chol))))
        self._const = LATENT_DIM * math.log(2.0 * math.pi)

    def logpdf(self, z: np.ndarray) -> float:
        d = np.asarray(z, dtype=np.float64).ravel() - self.mu
        y = _solve_lower(self.chol, d)
        return float(-0.5 * (self._const + self._logdet + float(y @ y)))


class LLRScorer:
    """번들 NPZ → 후보 활성화 [1536] × (scene, phase) → llr / OOD 판정.

    사용:
        sc = LLRScorer.from_bundle("llr_bundle.npz")
        if sc.registered(3, "c2"):
            out = sc.score(vec1536, 3, "c2")   # {"llr","log_s","log_f","ood_reject"}
    """

    def __init__(self, mu, std, weights, biases, entries, meta):
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        if self.mu.shape != (INPUT_DIM,):
            raise ValueError(f"llr: scaler_mu shape {self.mu.shape} != ({INPUT_DIM},)")
        self.std = float(std)
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"llr: scaler_std={self.std} 가 비정상 (>0 필요)")
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float64).ravel() for b in biases]
        # key "s3__c2" -> {"succ_mean","gs","gf","ood_lo"}
        # (속성명이 table 인 이유: 공개 API 는 entries(scene) 메서드다)
        self.table = dict(entries)
        self.meta = dict(meta)

    # ---------------------------------------------------------------- 로딩
    @classmethod
    def from_bundle(cls, path) -> "LLRScorer":
        """★ 위 docstring 의 NPZ 계약대로 로드 (누락 키는 전부 fail-loud)."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            keys = set(f.files)
            for key in ("meta", "scaler_mu", "scaler_std", "registered"):
                if key not in keys:
                    raise ValueError(f"{path.name}: 번들에 '{key}' 없음 — LLR 번들이 맞나")
            meta = json.loads(str(f["meta"]))
            mu = np.asarray(f["scaler_mu"], dtype=np.float64).ravel()
            std = float(np.asarray(f["scaler_std"]).reshape(-1)[0])

            shapes = [(256, INPUT_DIM), (256, 256), (LATENT_DIM, 256)]
            weights, biases = [], []
            for key, (o, i) in zip(_ENC_KEYS, shapes):
                wk, bk = f"{key}.weight", f"{key}.bias"
                for kk in (wk, bk):
                    if kk not in keys:
                        raise ValueError(f"{path.name}: encoder 키 '{kk}' 없음")
                w = np.asarray(f[wk], dtype=np.float64)
                b = np.asarray(f[bk], dtype=np.float64).ravel()
                if w.shape != (o, i) or b.shape != (o,):
                    raise ValueError(
                        f"{path.name}: '{key}' shape {w.shape}/{b.shape} != "
                        f"({o},{i})/({o},)")
                weights.append(w)
                biases.append(b)

            names = [str(s) for s in np.asarray(f["registered"]).ravel().tolist()]
            if not names:
                raise ValueError(f"{path.name}: registered 가 비었음")
            entries = {}
            for e in names:
                if "__" not in e:
                    raise ValueError(
                        f"{path.name}: registered 원소 '{e}' 형식 위반 "
                        "(‘s{scene}__c{k}’ 필요 — (scene, phase) 등록 단위)")
                need = (f"succ_mean.{e}", f"mu_s.{e}", f"cov_s.{e}",
                        f"mu_f.{e}", f"cov_f.{e}", f"ood_lo.{e}")
                for kk in need:
                    if kk not in keys:
                        raise ValueError(
                            f"{path.name}: 엔트리 '{e}' 의 '{kk}' 없음 "
                            "(registered 와 배열이 어긋난다)")
                sm = np.asarray(f[f"succ_mean.{e}"], dtype=np.float64).ravel()
                if sm.shape != (LATENT_DIM,):
                    raise ValueError(
                        f"{path.name}: succ_mean.{e} shape {sm.shape} != ({LATENT_DIM},)")
                entries[e] = {
                    "succ_mean": sm,
                    "gs": _Gaussian(f[f"mu_s.{e}"], f[f"cov_s.{e}"], f"cov_s.{e}"),
                    "gf": _Gaussian(f[f"mu_f.{e}"], f[f"cov_f.{e}"], f"cov_f.{e}"),
                    "ood_lo": float(np.asarray(f[f"ood_lo.{e}"]).reshape(-1)[0]),
                }
        return cls(mu, std, weights, biases, entries, meta)

    # ---------------------------------------------------------------- 조회
    def registered(self, scene, phase) -> bool:
        """(scene, phase) 에 가우시안이 적합돼 있는지 (미등록이면 점수를 낼 수 없다).

        scene·phase 표기가 깨져 키를 못 만드는 경우도 "미등록"으로 본다 (serve 는
        사유를 extras 에 남기고 후보 0 fallback — 조용한 오채점보다 낫다).
        """
        if not phase:
            return False
        try:
            key = entry_key(scene, phase)
        except ValueError:
            return False
        return key in self.table

    def scenes(self) -> list[str]:
        """번들에 등록된 scene 목록 ("s1","s3" …)."""
        return sorted({e.split("__", 1)[0] for e in self.table})

    def spec(self) -> dict:
        return {
            "registered": sorted(self.table),
            "scenes": self.scenes(),
            "latent": LATENT_DIM,
            "input_dim": INPUT_DIM,
            **{k: v for k, v in self.meta.items() if k in ("task", "ae_ref")},
        }

    # ---------------------------------------------------------------- 채점
    def encode(self, vec: np.ndarray) -> np.ndarray:
        """raw feature [1536] → latent [16] (표준화 + 3층 MLP, exact erf GELU)."""
        x = np.asarray(vec, dtype=np.float64).ravel()
        if x.shape != (INPUT_DIM,):
            raise ValueError(f"llr: encode 입력 shape {x.shape} != ({INPUT_DIM},)")
        h = (x - self.mu) / self.std
        h = _gelu(self.weights[0] @ h + self.biases[0])
        h = _gelu(self.weights[1] @ h + self.biases[1])
        return self.weights[2] @ h + self.biases[2]

    def score(self, vec: np.ndarray, scene, phase) -> dict:
        """raw feature [1536] × (scene, phase) → {"llr","log_s","log_f","ood_reject"}.

        llr>0 = 실패 가우시안 쪽 — serve 는 llr argmin 후보를 실행한다.
        """
        key = entry_key(scene, phase)
        ent = self.table.get(key)
        if ent is None:
            raise KeyError(f"llr: 엔트리 '{key}' 미등록 (등록: {sorted(self.table)})")
        z = self.encode(vec) - ent["succ_mean"]
        log_s = ent["gs"].logpdf(z)
        log_f = ent["gf"].logpdf(z)
        return {
            "llr": float(log_f - log_s),
            "log_s": float(log_s),
            "log_f": float(log_f),
            "ood_reject": bool(max(log_s, log_f) < ent["ood_lo"]),
        }

    def entries(self, scene) -> list[str]:
        """해당 scene 의 등록 entry 키 목록 (정렬). scene 표기가 깨지면 ValueError."""
        pref = normalize_scene(scene) + "__"
        return sorted(e for e in self.table if e.startswith(pref))

    def score_nearest(self, vec: np.ndarray, scene) -> dict:
        """raw feature [1536] × scene → 가장 가까운 **비-OOD** entry 로 배정해 채점.

        반환 {"entry": str|None, "llr": float|None, "log_s","log_f","ood_reject"}.
        배정 규칙: entry e 마다 z~ = latent - succ_mean[e] 로 log_s/log_f 를 구하고
        max(log_s,log_f) < ood_lo[e] 인 e 는 후보 기각. 남은 e 중 latent 공간
        유클리드 거리 ||latent - succ_mean[e]|| 최소인 e 에 배정한다. 전 entry 에서
        기각이면 entry=None, llr=None, ood_reject=True (log_s/log_f 는 거리 최소
        entry 의 값 — 진단용). scene 에 entry 가 하나도 없으면 ValueError (fail-loud).
        """
        keys = self.entries(scene)
        if not keys:
            raise ValueError(
                f"llr: scene '{normalize_scene(scene)}' 에 등록 entry 없음 "
                f"(번들 scenes={self.scenes()})")
        latent = self.encode(vec)
        rows = []
        for e in keys:
            ent = self.table[e]
            z = latent - ent["succ_mean"]
            log_s = ent["gs"].logpdf(z)
            log_f = ent["gf"].logpdf(z)
            rows.append((e, float(np.linalg.norm(z)), log_s, log_f,
                         bool(max(log_s, log_f) < ent["ood_lo"])))
        keep = [r for r in rows if not r[4]]
        if not keep:
            _, _, log_s, log_f, _ = min(rows, key=lambda r: r[1])
            return {"entry": None, "llr": None, "log_s": float(log_s),
                    "log_f": float(log_f), "ood_reject": True}
        e, _, log_s, log_f, _ = min(keep, key=lambda r: r[1])
        return {"entry": e, "llr": float(log_f - log_s), "log_s": float(log_s),
                "log_f": float(log_f), "ood_reject": False}


# ---------------------------------------------------------------------------
# 스모크용 dummy 번들 (무작위 파라미터, seed 고정) — 배선 점검 전용, 의미 없는 점수
# ---------------------------------------------------------------------------

def make_dummy_bundle(path, entries=(("s1", "c0"), ("s1", "c2"), ("s3", "c2")),
                      seed: int = 0):
    """무작위 파라미터 LLR 번들 생성 (계약 키 전부 채움).

    ``entries`` = [(scene, phase), ...] — scene 은 int 도 가능 (키는 정규화된다).
    반환 = 저장한 payload dict.
    """
    rng = np.random.default_rng(seed)
    ekeys = [entry_key(s, p) for s, p in entries]
    payload = {
        "meta": np.asarray(json.dumps({
            "task": "DummyTask", "ae_ref": "synthetic",
            "scenes": sorted({normalize_scene(s) for s, _ in entries}),
            "phases": sorted({str(p) for _, p in entries}),
        }, ensure_ascii=False)),
        "scaler_mu": rng.normal(0, 0.3, INPUT_DIM).astype(np.float32),
        "scaler_std": np.asarray(1.7, dtype=np.float32),
        "registered": np.asarray(ekeys, dtype=np.str_),
    }
    for key, (o, i) in zip(_ENC_KEYS, [(256, INPUT_DIM), (256, 256), (LATENT_DIM, 256)]):
        payload[f"{key}.weight"] = rng.normal(0, 1.0 / math.sqrt(i), (o, i)).astype(np.float32)
        payload[f"{key}.bias"] = rng.normal(0, 0.05, o).astype(np.float32)
    for e in ekeys:
        payload[f"succ_mean.{e}"] = rng.normal(0, 0.5, LATENT_DIM).astype(np.float32)
        for tag in ("s", "f"):
            a = rng.normal(0, 1.0, (LATENT_DIM, LATENT_DIM))
            payload[f"mu_{tag}.{e}"] = rng.normal(0, 0.3, LATENT_DIM).astype(np.float32)
            payload[f"cov_{tag}.{e}"] = ((a @ a.T) / LATENT_DIM).astype(np.float32)
        payload[f"ood_lo.{e}"] = np.asarray(-60.0, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return payload


# ---------------------------------------------------------------------------
# self-test: dummy 번들 왕복 로드 + score 유한성 + 정칙화 대칭성
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "llr_dummy.npz"
        payload = make_dummy_bundle(bp)
        sc = LLRScorer.from_bundle(bp)

        good = sorted(sc.table) == ["s1__c0", "s1__c2", "s3__c2"]
        ok &= good
        print(f"[self-test] 왕복 로드 registered={sorted(sc.table)} {'OK' if good else 'FAIL'}")

        good = sc.scenes() == ["s1", "s3"]
        ok &= good
        print(f"[self-test] scenes()={sc.scenes()} {'OK' if good else 'FAIL'}")

        # scene 표기 정규화: int / "3" / "s3" 동치, 그 외는 거부
        good = (normalize_scene(3) == normalize_scene("3") == normalize_scene("s3") == "s3"
                and entry_key(3, "c2") == "s3__c2")
        ok &= good
        print(f"[self-test] scene 정규화(int/'3'/'s3') {'OK' if good else 'FAIL'}")

        # (scene, phase) 교차 등록: s3__c0 는 없다 (phase 만으로 판정하면 안 됨)
        good = (sc.registered(3, "c2") and sc.registered("s1", "c0")
                and not sc.registered(3, "c0") and not sc.registered(9, "c2")
                and not sc.registered("s3", None) and not sc.registered("bad", "c2"))
        ok &= good
        print(f"[self-test] registered(scene, phase) 교차 판정 {'OK' if good else 'FAIL'}")

        rng = np.random.default_rng(7)
        vecs = rng.normal(0, 1.0, (8, INPUT_DIM)).astype(np.float32)
        outs = [sc.score(v, 3, "c2") for v in vecs]
        good = all(np.isfinite([o["llr"], o["log_s"], o["log_f"]]).all() for o in outs)
        ok &= good
        print(f"[self-test] score 유한성 (n=8) {'OK' if good else 'FAIL'}")

        good = all(abs(o["llr"] - (o["log_f"] - o["log_s"])) < 1e-9 for o in outs)
        ok &= good
        print(f"[self-test] llr == log_f - log_s {'OK' if good else 'FAIL'}")

        # 같은 phase 라도 scene 이 다르면 다른 엔트리 (scene 축이 실제로 먹는지)
        d = max(abs(sc.score(v, 3, "c2")["llr"] - sc.score(v, 1, "c2")["llr"]) for v in vecs)
        good = d > 1e-6
        ok &= good
        print(f"[self-test] scene 축 유효 (max|Δllr| s3 vs s1 = {d:.3e}) "
              f"{'OK' if good else 'FAIL'}")

        # score_nearest: scene 의 entry 목록·거리 규칙·OOD 기각
        good = sc.entries(1) == ["s1__c0", "s1__c2"] and sc.entries("s3") == ["s3__c2"]
        ok &= good
        print(f"[self-test] entries(scene) {'OK' if good else 'FAIL'}")

        mism = 0
        for v in vecs:
            out = sc.score_nearest(v, 1)
            lat = sc.encode(v)
            # 참조: 비-OOD entry 중 ||latent - succ_mean|| 최소
            cand = []
            for e in sc.entries(1):
                ent = sc.table[e]
                z = lat - ent["succ_mean"]
                ls, lf = ent["gs"].logpdf(z), ent["gf"].logpdf(z)
                if max(ls, lf) >= ent["ood_lo"]:
                    cand.append((float(np.linalg.norm(z)), e, lf - ls))
            if not cand:
                mism += int(out["entry"] is not None or not out["ood_reject"])
                continue
            _, ref_e, ref_llr = min(cand)
            mism += int(out["entry"] != ref_e or abs(out["llr"] - ref_llr) > 1e-9)
        good = mism == 0
        ok &= good
        print(f"[self-test] score_nearest 배정=최근접 비-OOD entry (불일치 {mism}) "
              f"{'OK' if good else 'FAIL'}")

        # 전-OOD: 그 scene 의 모든 entry ood_lo 를 올리면 entry=None·llr=None·기각
        saved = {e: sc.table[e]["ood_lo"] for e in sc.entries(1)}
        for e in saved:
            sc.table[e]["ood_lo"] = 1e9
        outs_n = [sc.score_nearest(v, 1) for v in vecs]
        good = all(o["entry"] is None and o["llr"] is None and o["ood_reject"]
                   and np.isfinite([o["log_s"], o["log_f"]]).all() for o in outs_n)
        ok &= good
        print(f"[self-test] score_nearest 전-OOD 기각 {'OK' if good else 'FAIL'}")
        for e, v in saved.items():
            sc.table[e]["ood_lo"] = v

        # entry 없는 scene 은 fail-loud
        try:
            sc.score_nearest(vecs[0], 7)
            print("[self-test] entry 없는 scene 이 통과됨 FAIL")
            ok = False
        except ValueError:
            print("[self-test] entry 없는 scene 거부 OK")

        # 정칙화: 대칭 + PD + 원본 대비 tr 증가율 = _COV_REG
        c0 = payload["cov_s.s3__c2"].astype(np.float64)
        reg = sc.table["s3__c2"]["gs"].cov
        sym = float(np.abs(reg - reg.T).max())
        eig = float(np.linalg.eigvalsh(reg).min())
        ratio = float(np.trace(reg) / np.trace(0.5 * (c0 + c0.T)) - 1.0)
        good = sym < 1e-12 and eig > 0 and abs(ratio - _COV_REG) < 1e-9
        ok &= good
        print(f"[self-test] 정칙화 대칭 {sym:.1e}·min eig {eig:.3e}·Δtr {ratio:.2e} "
              f"{'OK' if good else 'FAIL'}")

        # cholesky 전방대입이 numpy solve 와 일치
        z = rng.normal(0, 1.0, LATENT_DIM)
        gs = sc.table["s3__c2"]["gs"]
        ref = float(-0.5 * (LATENT_DIM * math.log(2 * math.pi)
                            + float(np.log(np.linalg.det(reg)))
                            + (z - gs.mu) @ np.linalg.solve(reg, z - gs.mu)))
        got = gs.logpdf(z)
        good = abs(got - ref) < 1e-6
        ok &= good
        print(f"[self-test] logpdf vs numpy 참조 |Δ|={abs(got-ref):.2e} "
              f"{'OK' if good else 'FAIL'}")

        # OOD: ood_lo 를 아주 높이면 전부 기각되어야 한다
        sc.table["s3__c2"]["ood_lo"] = 1e9
        good = all(sc.score(v, 3, "c2")["ood_reject"] for v in vecs)
        ok &= good
        print(f"[self-test] ood_reject 발화 {'OK' if good else 'FAIL'}")

        # 미등록 엔트리·잘못된 scene 표기·입력 dim 은 fail-loud
        for fn, exc in ((lambda: sc.score(vecs[0], 3, "c0"), KeyError),
                        (lambda: sc.score(vecs[0], "bad", "c2"), ValueError),
                        (lambda: sc.encode(np.zeros(3)), ValueError)):
            try:
                fn()
                print("[self-test] fail-loud 가드 통과됨 FAIL")
                ok = False
            except exc:
                pass
        print("[self-test] fail-loud 가드 OK")

        # 키 누락 / registered 형식 위반 번들은 로드 거부
        bad = dict(payload)
        bad.pop("mu_f.s1__c0")
        bp2 = Path(td) / "llr_bad.npz"
        np.savez_compressed(bp2, **bad)
        bad2 = dict(payload)
        bad2["registered"] = np.asarray(["c2"], dtype=np.str_)
        bp3 = Path(td) / "llr_bad_fmt.npz"
        np.savez_compressed(bp3, **bad2)
        for p in (bp2, bp3):
            try:
                LLRScorer.from_bundle(p)
                print(f"[self-test] 불량 번들 {p.name} 이 통과됨 FAIL")
                ok = False
            except ValueError:
                pass
        print("[self-test] 불량 번들(키 누락·registered 형식) 거부 OK")

    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
