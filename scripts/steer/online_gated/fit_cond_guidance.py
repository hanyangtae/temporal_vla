#!/usr/bin/env python3
"""condg — 상태-조건부 대조 guidance 연산자 fit (phase별 W_s/W_f + 게이트 τ).

스펙 단일 출처: `docs/steering/44_cond_guidance_operator.md` (§1 연산자, §2 등록 게이트,
§3 대조군, §4 저장 규약). 정확식(전처리·릿지·고정B·AUROC)의 원본 구현은
`scripts/analysis/grid_phase/cond_margin.py` 이고, 이 스크립트는 그 수식을 **그대로**
재사용한다 (아래 "cond_margin 대비 차이" 참조).

연산자 (44 §1):
  φ(s) = [eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)] 9 + 인접 record 차분 속도 9 = 18
         (첫 record 속도 = 0). scene 별 z-score (mp, sp+1e-8; train record 풀).
  h    = DiT L12 · 마지막 denoise(step 3) · 49토큰 mean. scene 별 중심화 (train 성공 record
         평균 mh; 해당 scene 에 성공이 없으면 그 scene 전체 평균).
  릿지(무절편)  W = (PᵀP + λI)⁻¹ PᵀX,  λ = 1e-3·n  (n = 그 클래스 train record 행수).
  margin        m = ‖h̃ − φ̃W_s‖² − ‖h̃ − φ̃W_f‖²   (m 클수록 실패 쪽 — 부호 고정, 뒤집지 않음)
  개입(serve)   m > τ 일 때만  d̂ = normalize(φ̃W_f − φ̃W_s),
                h̃′ = h̃ − β·⟨h̃ − φ̃W_s, d̂⟩·d̂

등록 게이트 (44 §2, **개정 — 5-seed split CV**):
  · 배포 W_s/W_f 와 중심화 stats 는 **전체 fit 셋**으로 fit 한다 (split 무관).
  · 게이트는 seed 0..K-1 (기본 K=5) 각각 scene-층화 6:4 split 로 임시 W 를 학습해
    held-out 고정B margin AUROC 와 길이단독 AUROC 를 얻고,
      등록 = margin AUROC **중앙값 > 길이단독 중앙값** AND **과반(5 중 3) seed 에서 margin>길이**.
  · 경성 표본 하한 = 전체 fit 셋 클래스당 episode ≥ 8 (미달이면 registered=False).
    클래스당 < 15 면 등록하되 small_sample=True 딱지를 붙인다.
  · τ = 전 seed held-out 성공 episode 고정B margin 을 **pool** 한 분포의 90퍼센타일.
    B = seed 별 B (train 성공 dwell 25퍼센타일, max(3,·)) 의 중앙값(int) — 진단용.
미달 cell 은 registered=False 로 기록만 하고 serve 는 무시(identity).
  · `--force-register` (탐색 라운드 전용) 를 주면 이 안전장치를 해제하고 W 가 fit 된 전
    phase 를 등록한다. 게이트 판정은 계속 계산해 gate_registered/gate_reason 에 남긴다.
    W 자체를 못 뽑은 phase(경성 하한 미달 등) 는 그대로 identity 잔존.

변형 (44 §3):
  condg     처치.
  condg_pl  episode 라벨을 **scene-층화 순열** 후 동일 절차. 등록 게이트는 우회 —
            처치가 등록한 phase 는 위약도 강제 등록(τ 는 자기 캘리브레이션). AUROC 는 기록.
  condg_hs  ablation "성공-모방 단독" h̃′ = (1−β)h̃ + β·φ̃W_s. W·τ·B 가 처치와 **동일**하고
            적용식만 다르므로 NPZ 내용은 condg 와 같고 meta 의 mode 만 "hs" 다.

cond_margin.py 대비 차이 (전부 44 문서가 요구한 것):
  1. split 이 scene **층화** 6:4 (cond_margin 은 비층화 전역 순열) — 44 §2.
  2. glob 이 `n*` (cond_margin 은 `n[0-4]`) 이고, 실제 셀 선택은 fit manifest 로 한다.
     manifest 가 없으면 s0–4 × n0–4 fallback = cond_margin 과 동일 집합.
  3. phase 별 rng 를 (seed, crc32(phase)) 로 고정 — 처치/위약이 **같은 split** 을 쓰게
     하기 위함 (cond_margin 은 phase 순서에 의존하는 단일 rng 스트림).
  4. AUROC 를 3종(고정B margin / 길이단독 / record) 다 산출하고 등록 판정에 쓴다.
  5. 게이트가 단일 split 이 아니라 5-seed CV 이고, 배포 W 는 전체 셋 fit 이다 (위 참조).
  전처리·릿지·B·margin·auroc 함수는 라인 단위로 동일하다.

셀 선택 vs scene 그룹 (v4 지터 격자):
  · **선택** 키 = 평탄 cell id si (= base_scene*100 + k, base 재사용 셀 = +99).
    manifest·경로 glob·pkl `scene_idx` 가 전부 이 값이다.
  · **그룹** 키 = base scene (평탄 si // 100). `--v4-jitter` 를 주면 scene 통계
    (mh/mp/sp)·층화 분할·위약 라벨 순열·NPZ `scene{sc}__*` 키가 base scene 으로 접힌다.
    serve 러너가 base scene(0-4)을 post 하므로 이게 serve 계약과 맞는다.
  · ⚠ grid_root 아래에 plan_id 디렉토리가 여럿 섞여 있으면(v4 지터 격자 vs v2/v1 구
    격자) **평탄 si 가 충돌**한다 — v2 base scene1 과 v4 scene0·k=1 이 둘 다
    `scene_idx=1`. 이 경우 (scene_idx, noise_idx) 튜플 선택은 조용히 오귀속되므로
    반드시 경로 명시 모드 `--episode-manifest` 를 쓴다 (glob·셀 필터 미사용,
    scene 그룹 키 = manifest 의 base_scene 열, 라벨 정본 = manifest label).

사용 (승준 노드, pkl 있는 곳):
  ~/anaconda3/bin/python scripts/steer/online_gated/fit_cond_guidance.py \
      --slug OpenDrawer_right --phases reach,grasp,pull

산출: <out-dir>/<variant>/condg.npz + metadata.json,  <out-dir>/fit_summary.json
"""
from __future__ import annotations

import os

# BLAS 스레드 cap — 공유 노드. numpy import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse  # noqa: E402
import glob as globmod  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

try:  # rollout pkl 은 torch 텐서를 담고 있어 unpickle 에 torch 가 필요할 수 있다.
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch 없는 환경에서도 import 자체는 통과시킨다
    torch = None

_CLUSTER_MOD = None


def _cluster_api():
    """cluster phase 라벨 어댑터(scripts/fit/fit_phase_conceptor.py)를 **지연** import.

    중복 구현 금지 — 번들 로드·좌표 검사·배정은 전부 그쪽 공용 함수다. cluster 모드가
    아닐 때는 부르지 않으므로 이 스크립트의 기존 의존성은 그대로다.
    """
    global _CLUSTER_MOD
    if _CLUSTER_MOD is None:
        repo = Path(__file__).resolve().parents[3]
        for pth in (str(repo), str(repo / "scripts/fit")):
            if pth not in sys.path:
                sys.path.insert(0, pth)
        import fit_phase_conceptor as _m  # noqa: PLC0415
        _CLUSTER_MOD = _m
    return _CLUSTER_MOD


# cond_margin.py 와 동일한 slug → grid 경로 매핑
TASKMAP = {"OpenDrawer_left": "OpenDrawer/left", "OpenDrawer_right": "OpenDrawer/right",
           "DishwasherRack_out": "DishwasherRack/out", "OvenRack_out": "OvenRack/out",
           "PPCC_candle": "PPCC/candle", "PPCC_bread": "PPCC/bread",
           "PPCC_marshmallow": "PPCC/marshmallow", "PPCC_jug": "PPCC/jug",
           "CoffeeSetupMug": "CoffeeSetupMug"}

DEFAULT_GRID = "~/datasets/temporal_vla_store/groot/n15/grid/"
DEFAULT_VARIANTS = "condg,condg_pl,condg_hs"
FALLBACK_CELLS = {(s, n) for s in range(5) for n in range(5)}   # s5m5
MIN_PHASE_RECORDS = 4       # cond_margin: episode 당 phase record 4개 미만이면 제외
MIN_EPS_PER_CLASS = 7       # cond_margin: 클래스당 episode 7 미만이면 phase 자체를 건너뜀
MIN_TRAIN_EPS = 15          # 44 §2 권장 표본 — 미달이면 등록하되 small_sample 딱지
HARD_MIN_EPS = 8            # 44 §2 개정: 전체 fit 셋 클래스당 episode 경성 하한
GATE_SEEDS = 5              # 게이트 CV split seed 수 (0..GATE_SEEDS-1)
TRAIN_FRAC = 0.6
GATE_SCHEME = "cv5_median_majority"
CELL_RE = re.compile(r"/s(\d+)/n(\d+)/")
SPEC = "44"
V4_SCENE_STRIDE = 100       # v4 지터 격자: 평탄 cell id = base_scene*100 + k (base 재사용 = +99)


def fold_scene(si: int, v4_jitter: bool) -> int:
    """평탄 cell id → 그룹 키.

    두 층위를 구분한다 (v4 지터 격자):
      · **선택(=평탄 si)** — manifest·경로 glob·pkl 의 `scene_idx` 는 전부 평탄 id
        (base_scene*100 + k, base 셀은 +99). 셀 필터는 이 평탄 id 로 그대로 매칭한다.
      · **그룹(=base scene)** — scene 통계(mh/mp/sp)·층화 분할·라벨 순열·NPZ
        `scene{sc}__*` 키는 base scene(평탄//100) 단위로 접는다. serve 러너가
        base scene(0-4)을 post 하기 때문.
    v4_jitter=False 면 항등 (기존 s0-4 격자와 완전 동일)."""
    return si // V4_SCENE_STRIDE if v4_jitter else si


# ------------------------------------------------------------------ 정확식 (cond_margin 이식)
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """rank AUROC (pos 가 높을 확률). cond_margin.auroc 와 동일."""
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ridge(X_list, P_list) -> np.ndarray:
    """무절편 릿지 W = (PᵀP + λI)⁻¹PᵀX, λ = 1e-3·n. cond_margin.ridge 와 동일."""
    X = np.concatenate(X_list)
    P = np.concatenate(P_list)
    lam = 1e-3 * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)


# ------------------------------------------------------------------ 입력
def load_fit_cells(path: Path | None) -> tuple[set[tuple[int, int]], str]:
    """fit manifest 에서 (scene_idx, noise_idx) 집합을 뽑는다.

    manifest 포맷이 레포 안에 하나로 굳어 있지 않아 3종을 자동 판별한다.
      (a) 헤더에 scene_idx/noise_idx 열이 있는 index 계열 tsv
      (b) fit_setm 계열 headerless `pkl경로 \\t label \\t scene` → 경로에서 s*/n* 파싱
      (c) replay_cells 계열 headerless `scene_idx \\t noise_idx \\t env_seed ...`
    파일이 없으면 s0–4 × n0–4 fallback (+경고).

    v4 지터 격자에서는 scene_idx 열이 **평탄 si**(base*100+k, base 셀=+99)를 담는다.
    여기서 뽑는 셀 집합은 pkl 경로·`scene_idx` 와 대조하는 **선택** 키이므로 평탄 si
    그대로 두고 접지 않는다 (접기는 --v4-jitter 의 그룹 단계에서만). 선택=평탄 si,
    그룹=base scene.
    """
    if path is None or not path.exists():
        print(f"[warn] fit manifest 없음 ({path}) — s0-4 × n0-4 fallback 사용", flush=True)
        return set(FALLBACK_CELLS), "fallback_s5m5"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if not lines:
        print(f"[warn] fit manifest 비어 있음 ({path}) — s0-4 × n0-4 fallback", flush=True)
        return set(FALLBACK_CELLS), "fallback_s5m5"
    head = lines[0].split("\t")
    cells: set[tuple[int, int]] = set()
    if "scene_idx" in head and "noise_idx" in head:      # (a)
        ci, ni = head.index("scene_idx"), head.index("noise_idx")
        for ln in lines[1:]:
            p = ln.split("\t")
            cells.add((int(p[ci]), int(p[ni])))
        return cells, "header_scene_noise"
    first = head[0]
    if first.endswith(".pkl") or "/" in first:           # (b)
        for ln in lines:
            m = CELL_RE.search(ln.split("\t")[0])
            if m:
                cells.add((int(m.group(1)), int(m.group(2))))
        if cells:
            return cells, "path_scene_noise"
    try:                                                  # (c)
        for ln in lines:
            p = ln.split("\t")
            cells.add((int(p[0]), int(p[1])))
        return cells, "leading_two_ints"
    except (ValueError, IndexError):
        pass
    print(f"[warn] fit manifest 포맷 판별 실패 ({path}) — s0-4 × n0-4 fallback", flush=True)
    return set(FALLBACK_CELLS), "fallback_s5m5"


def _features(d: dict, layer: int, denoise_step: int) -> tuple[np.ndarray, np.ndarray]:
    """rollout dict → (H, P). 추출식은 cond_margin 과 라인 단위로 동일."""
    cap = [int(x) for x in d["capture_layers"]]
    H = np.stack([np.asarray(h[cap.index(layer), denoise_step], dtype=np.float32).mean(0)
                  for h in d["hidden_states"]])
    st = [np.concatenate([np.asarray(s["observation.state.eef_pos_rel"]),
                          np.asarray(s["observation.state.eef_quat_rel"]),
                          np.asarray(s["observation.state.gripper_qpos"])])
          for s in d["states"]]
    P0 = np.stack(st).astype(np.float64)
    P = np.hstack([P0, np.vstack([np.zeros((1, 9)), np.diff(P0, axis=0)])])
    return H.astype(np.float64), P


def load_episodes(slug: str, grid_root: Path, cells: set[tuple[int, int]],
                  layer: int, denoise_step: int, v4_jitter: bool = False,
                  cluster_feat: bool = False) -> list[dict]:
    """rollout pkl → episode 레코드. 특징 추출식은 cond_margin 과 동일.

    셀 **선택**(경로 사전필터·pkl scene_idx 대조·중복 제거)은 언제나 평탄 si 로 한다.
    v4_jitter=True 면 레코드의 `scene`(=그룹 키)만 base scene 으로 접고, 평탄 id 는
    `cell_si` 로 보존한다 (로그·provenance 용)."""
    pat = str(grid_root / f"*/*/{TASKMAP[slug]}/s*/n*/base/rollout.pkl")
    pkls = sorted(globmod.glob(pat))
    # 경로 기반 사전 필터 (pkl 로드 비용 절감). 최종 판정은 pkl 안의 scene_idx/noise_idx.
    pre = []
    for p in pkls:
        m = CELL_RE.search(p)
        if m is None or (int(m.group(1)), int(m.group(2))) in cells:
            pre.append(p)
    print(f"# {slug}: pkl {len(pkls)}개 → fit cell 후보 {len(pre)}개", flush=True)

    eps: list[dict] = []
    seen_cells: set[tuple[int, int]] = set()
    for p in pre:
        with open(p, "rb") as f:
            d = pickle.load(f)
        cell = (int(d["scene_idx"]), int(d["noise_idx"]))
        if cell not in cells:
            continue
        if cell in seen_cells:      # 다중 머신 수집 task 의 중복 셀 제거 (첫 등장만)
            continue
        seen_cells.add(cell)
        H, P = _features(d, layer, denoise_step)
        eps.append({
            # scene = 그룹 키(v4 면 base scene), cell_si = 평탄 cell id 원본
            "scene": fold_scene(int(d["scene_idx"]), v4_jitter),
            "cell_si": int(d["scene_idx"]), "noise": int(d["noise_idx"]),
            "success": int(d["episode_success"]),
            "H": H, "P": P,
            "phases": list(d["feature_phases"]),
            # cluster 모드용 판정 feature (L12 · 마지막 denoise · 49토큰 mean). --layer/
            # --denoise-step 과 무관하게 번들 학습 좌표로 고정 뽑는다.
            **({"cluster_feat": _cluster_api().cluster_features_from_pkl(d, Path(p).name)}
               if cluster_feat else {}),
            # docs/04 규약 — 출처는 절대경로가 아니라 내용 지문으로 남긴다.
            "sig": hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16],
        })
    return eps


def parse_episode_manifest(path: Path) -> list[dict]:
    """경로 기반 episode manifest(헤더 tsv) → 행 dict 목록.

    필수 열: `pkl_path` (grid_root 기준 상대 또는 절대), `label` (1=succ/0=fail),
    `base_scene` (0-4, scene 그룹 키). 선택 열: `cell_si`, `noise_idx`.
    numpy 없이도 동작하는 순수 파싱부 (단위 검증용으로 분리)."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise SystemExit(f"episode manifest 비어 있음: {path}")
    head = lines[0].split("\t")
    need = ("pkl_path", "label", "base_scene")
    missing = [c for c in need if c not in head]
    if missing:
        raise SystemExit(f"episode manifest 필수 열 없음 {missing} (head={head})")
    idx = {c: head.index(c) for c in head}
    rows: list[dict] = []
    for ln in lines[1:]:
        p = ln.split("\t")
        r = {"pkl_path": p[idx["pkl_path"]].strip(),
             "label": int(p[idx["label"]]),
             "base_scene": int(p[idx["base_scene"]])}
        for opt in ("cell_si", "noise_idx"):
            if opt in idx and idx[opt] < len(p) and p[idx[opt]].strip() != "":
                r[opt] = int(p[idx[opt]])
        rows.append(r)
    return rows


def load_episodes_manifest(manifest: Path, grid_root: Path,
                           layer: int, denoise_step: int,
                           cluster_feat: bool = False) -> list[dict]:
    """경로 명시 manifest 로 episode 를 고른다 (plan_id 혼재 grid_root 안전 경로).

    grid_root 아래에 plan_id 디렉토리가 여럿 섞여 있으면 (v4=지터 격자, v2/v1=구
    격자) **평탄 si 가 충돌한다**: v2 base scene1 과 v4 scene0·k=1 이 둘 다
    `scene_idx=1` 이라 (scene_idx, noise_idx) 튜플 선택은 조용히 오귀속된다.
    이 모드는 glob·셀 필터를 전혀 쓰지 않고 manifest 가 나열한 pkl 만 읽으며,
    scene 그룹 키는 `base_scene` 열을 그대로 쓴다 (--v4-jitter 의 //100 폴딩과
    같은 결과지만 출처가 명시적).

    label 정본 = manifest. pkl 내부 `episode_success` 와 어긋나면 fail-loud.
    """
    rows = parse_episode_manifest(manifest)
    print(f"# episode manifest {manifest.name}: {len(rows)}행 (경로 명시 모드 — "
          f"glob/셀 필터 미사용)", flush=True)
    eps: list[dict] = []
    mismatch: list[str] = []
    for r in rows:
        p = Path(r["pkl_path"]).expanduser()
        if not p.is_absolute():
            p = grid_root / p
        if not p.exists():
            raise SystemExit(f"episode manifest 의 pkl 없음: {r['pkl_path']}")
        with open(p, "rb") as f:
            d = pickle.load(f)
        pkl_succ = int(d["episode_success"])
        if pkl_succ != r["label"]:
            mismatch.append(f"{r['pkl_path']}: manifest label={r['label']} vs "
                            f"pkl episode_success={pkl_succ}")
        H, P = _features(d, layer, denoise_step)
        eps.append({
            "scene": int(r["base_scene"]),          # 그룹 키 = manifest 명시 base scene
            "cell_si": int(r.get("cell_si", d["scene_idx"])),   # 평탄 si (기록용)
            "noise": int(r.get("noise_idx", d["noise_idx"])),
            "success": int(r["label"]),             # 라벨 정본 = manifest
            "H": H, "P": P,
            "phases": list(d["feature_phases"]),
            **({"cluster_feat": _cluster_api().cluster_features_from_pkl(d, p.name)}
               if cluster_feat else {}),
            "sig": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        })
    if mismatch:
        raise SystemExit("episode manifest label 불일치 %d건 (fail-loud):\n  %s"
                         % (len(mismatch), "\n  ".join(mismatch[:20])))
    return eps


# ------------------------------------------------------------------ 분할·라벨
def stratified_train_idx(rows: list[tuple], seed: int, phase: str) -> set[int]:
    """scene 층화 6:4 episode 분할 (44 §2). scene 마다 순열 후 앞 60% 를 train.

    rng 는 (seed, crc32(phase)) 로 고정 — phase 처리 순서와 무관하고, 처치/위약이
    같은 split 을 쓴다 (라벨만 다른 대조가 되도록)."""
    rng = np.random.default_rng([seed, zlib.crc32(phase.encode("utf-8"))])
    tr: set[int] = set()
    for sc in sorted({r[0] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[0] == sc]
        order = rng.permutation(len(idx))
        n_tr = int(len(idx) * TRAIN_FRAC)
        tr.update(idx[j] for j in order[:n_tr])
    return tr


def scene_stratified_permute(eps: list[dict], seed: int) -> list[int]:
    """episode 성공 라벨을 scene 안에서만 순열 (44 §3 위약). 반환 = episode 순 라벨 리스트."""
    rng = np.random.default_rng([seed, 12345])
    labels = [e["success"] for e in eps]
    out = list(labels)
    for sc in sorted({e["scene"] for e in eps}):
        idx = [i for i, e in enumerate(eps) if e["scene"] == sc]
        vals = np.array([labels[i] for i in idx])
        vals = vals[rng.permutation(len(vals))]
        for i, v in zip(idx, vals):
            out[i] = int(v)
    return out


# ------------------------------------------------------------------ phase fit 부품
def _scene_stats(rows: list[tuple], sub: set[int] | None
                 ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """scene 별 중심화 파라미터 (mh: 성공-우선, mp/sp: 전체). sub=None 이면 전체 셋 기준."""
    stats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sc in sorted({r[0] for r in rows}):
        if sub is None:
            g = [r for r in rows if r[0] == sc]
        else:
            g = [rows[i] for i in sub if rows[i][0] == sc] or [r for r in rows if r[0] == sc]
        ref_h = np.concatenate([r[2] for r in g if r[1] == 1]) if any(
            r[1] == 1 for r in g) else np.concatenate([r[2] for r in g])
        ref_p = np.concatenate([r[3] for r in g])
        stats[sc] = (ref_h.mean(0), ref_p.mean(0), ref_p.std(0) + 1e-8)
    return stats


def _global_stats(rows: list[tuple], sub: set[int] | None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """unseen scene 용 fallback 중심화 파라미터 (성공 기준 mh). sub=None 이면 전체 셋."""
    g = list(rows) if sub is None else ([rows[i] for i in sub] or list(rows))
    ref_h = np.concatenate([r[2] for r in g if r[1] == 1]) if any(
        r[1] == 1 for r in g) else np.concatenate([r[2] for r in g])
    ref_p = np.concatenate([r[3] for r in g])
    return ref_h.mean(0), ref_p.mean(0), ref_p.std(0) + 1e-8


def _prep_fn(rows: list[tuple], stats: dict):
    def prep(i):
        sc, su, H, P, _ = rows[i]
        mh, mp, sp = stats[sc]
        return su, H - mh, (P - mp) / sp
    return prep


def _fit_W(rows: list[tuple], sub: list[int], stats: dict) -> np.ndarray:
    prep = _prep_fn(rows, stats)
    pr = [prep(i) for i in sub]
    return ridge([p[1] for p in pr], [p[2] for p in pr])


def _cv_seed(rows: list[tuple], phase: str, seed: int) -> dict | None:
    """한 CV seed: train split 으로 임시 W → held-out 고정B margin / 길이 AUROC.

    반환 None = 이 seed 는 train 에 한쪽 클래스가 없어 대조 불가."""
    tr = stratified_train_idx(rows, seed, phase)
    tr_s = [i for i in tr if rows[i][1] == 1]
    tr_f = [i for i in tr if rows[i][1] == 0]
    if not tr_s or not tr_f:
        return None
    stats = _scene_stats(rows, tr)
    prep = _prep_fn(rows, stats)
    Ws = _fit_W(rows, tr_s, stats)
    Wf = _fit_W(rows, tr_f, stats)

    # 길이 공정화: 고정 예산 B = train 성공 dwell 의 25퍼센타일 (전 episode 동일 초반 창)
    succ_dw = sorted(len(rows[i][2]) for i in tr_s)
    B = max(3, succ_dw[len(succ_dw) // 4])

    ep_m: dict[int, list] = {0: [], 1: []}
    fixB_m: dict[int, list] = {0: [], 1: []}
    rec_m: dict[int, list] = {0: [], 1: []}
    dwell: dict[int, list] = {0: [], 1: []}
    for i in range(len(rows)):
        if i in tr:
            continue
        su, X, P = prep(i)
        m = ((X - P @ Ws) ** 2).sum(1) - ((X - P @ Wf) ** 2).sum(1)
        ep_m[su].append(float(np.mean(m)))
        if len(m) >= B:
            fixB_m[su].append(float(np.mean(m[:B])))
        rec_m[su].append(m)
        dwell[su].append(len(m))
    return {
        "seed": seed, "B": int(B),
        "auroc_margin_fixB": auroc(np.array(fixB_m[0]), np.array(fixB_m[1])),
        "auroc_len": auroc(np.array(dwell[0], float), np.array(dwell[1], float)),
        "auroc_episode": auroc(np.array(ep_m[0]), np.array(ep_m[1])),
        "auroc_record": auroc(np.concatenate(rec_m[0]) if rec_m[0] else np.array([]),
                              np.concatenate(rec_m[1]) if rec_m[1] else np.array([])),
        "fixB_succ": list(fixB_m[1]),
        "n_train_succ": len(tr_s), "n_train_fail": len(tr_f),
        "n_heldout_succ": len(ep_m[1]), "n_heldout_fail": len(ep_m[0]),
    }


def _median(v: list[float]) -> float:
    a = np.array([x for x in v if np.isfinite(x)], dtype=np.float64)
    return float(np.median(a)) if a.size else float("nan")


# ------------------------------------------------------------------ phase fit
def fit_phase(eps: list[dict], labels: list[int], phase: str, seed: int,
              min_train_eps: int, gate_seeds: int = GATE_SEEDS,
              min_eps_per_class: int = MIN_EPS_PER_CLASS) -> dict | None:
    """한 phase 의 W_s/W_f/B/τ/AUROC.

    배포 W·stats = **전체 fit 셋** fit. 등록 게이트 = gate_seeds 개 split CV 의
    (margin AUROC 중앙값 > 길이 중앙값) AND (과반 seed 에서 margin>길이) AND 표본 하한.
    미달이면 registered=False 인 dict, 아예 불가면 None."""
    rows = []
    for k, e in enumerate(eps):
        idx = [i for i, ph in enumerate(e["phases"]) if ph == phase]
        if len(idx) >= MIN_PHASE_RECORDS:
            rows.append((e["scene"], int(labels[k]), e["H"][idx], e["P"][idx], k))
    if not rows:
        return None
    ns = sum(r[1] for r in rows)
    nf = len(rows) - ns
    base = {"phase": phase, "n_eps": len(rows), "n_eps_succ": int(ns), "n_eps_fail": int(nf)}
    if ns < min_eps_per_class or nf < min_eps_per_class:
        print(f"  [{phase}] episode 부족 s/f={ns}/{nf} (유효 하한 {min_eps_per_class}"
              f", 기본 {MIN_EPS_PER_CLASS}) — W fit 자체를 건너뜀", flush=True)
        return {**base, "registered": False,
                "skip_reason": f"episode 부족 s/f={ns}/{nf} (<{min_eps_per_class})"}
    # 하한을 기본값 아래로 내려 fit 한 경우 = [극소표본]. 기존 [소표본](min_train_eps)과 별개 딱지.
    sub_floor = min(ns, nf) < MIN_EPS_PER_CLASS

    # --- 배포 연산자: 전체 fit 셋 (split 무관) ------------------------------------
    all_s = [i for i, r in enumerate(rows) if r[1] == 1]
    all_f = [i for i, r in enumerate(rows) if r[1] == 0]
    if not all_s or not all_f:
        return {**base, "registered": False,
                "skip_reason": "한쪽 클래스 없음 — 대조 불가"}
    stats = _scene_stats(rows, None)
    Ws = _fit_W(rows, all_s, stats)
    Wf = _fit_W(rows, all_f, stats)

    # --- 게이트: gate_seeds 개 split CV -------------------------------------------
    cv = [c for c in (_cv_seed(rows, phase, s) for s in range(gate_seeds)) if c is not None]
    a_marg = [c["auroc_margin_fixB"] for c in cv]
    a_lens = [c["auroc_len"] for c in cv]
    med_marg, med_len = _median(a_marg), _median(a_lens)
    wins = sum(1 for m, l in zip(a_marg, a_lens)
               if np.isfinite(m) and np.isfinite(l) and m > l)
    need_wins = gate_seeds // 2 + 1          # 5 → 3 (44 §2 개정 "≥3/5")

    # τ = 전 seed held-out 성공 고정B margin pool 의 90퍼센타일, B = seed B 중앙값
    pooled = [x for c in cv for x in c["fixB_succ"]]
    tau = float(np.percentile(np.array(pooled), 90)) if pooled else float("nan")
    B = int(round(_median([c["B"] for c in cv]))) if cv else 0

    small_sample = min(ns, nf) < min_train_eps
    reasons = []
    if not cv:
        reasons.append(f"CV seed {gate_seeds}개 모두 train 한쪽 클래스 없음")
    else:
        if not (np.isfinite(med_marg) and np.isfinite(med_len) and med_marg > med_len):
            reasons.append(f"margin 중앙값 {med_marg:.3f} ≤ 길이 중앙값 {med_len:.3f}")
        if wins < need_wins:
            reasons.append(f"margin>길이 seed {wins}/{len(cv)} < {need_wins}")
    if min(ns, nf) < HARD_MIN_EPS:
        reasons.append(f"클래스당 episode {ns}/{nf} < {HARD_MIN_EPS} (경성 하한)")
    if not np.isfinite(tau):
        reasons.append("held-out 성공 고정B 표본 0 — τ 산출 불가")
    return {
        **base, "registered": not reasons,
        "skip_reason": "; ".join(reasons) if reasons else "",
        "small_sample": bool(small_sample),
        "sub_floor": bool(sub_floor), "min_eps_per_class": int(min_eps_per_class),
        "W_s": Ws, "W_f": Wf, "B": B, "tau": tau, "sign": 1.0,
        # 대표값 = CV 중앙값 (기존 키 이름 유지)
        "auroc_margin_fixB": med_marg, "auroc_len": med_len,
        "auroc_record": _median([c["auroc_record"] for c in cv]),
        "auroc_episode": _median([c["auroc_episode"] for c in cv]),
        "auroc_margin_seeds": [float(x) for x in a_marg],
        "auroc_len_seeds": [float(x) for x in a_lens],
        "gate_seeds": int(gate_seeds), "gate_seeds_ok": int(len(cv)),
        "gate_wins": int(wins), "gate_need_wins": int(need_wins),
        "gate_scheme": GATE_SCHEME,
        "B_seeds": [int(c["B"]) for c in cv],
        "n_fit_succ": int(ns), "n_fit_fail": int(nf),
        "n_train_succ": int(_median([c["n_train_succ"] for c in cv])) if cv else 0,
        "n_train_fail": int(_median([c["n_train_fail"] for c in cv])) if cv else 0,
        "n_heldout_succ": int(_median([c["n_heldout_succ"] for c in cv])) if cv else 0,
        "n_heldout_fail": int(_median([c["n_heldout_fail"] for c in cv])) if cv else 0,
        "n_tau_pool": len(pooled),
        "stats": stats,
        # global fallback (미지 scene 용): 전체 fit 셋 record 풀
        "global": _global_stats(rows, None),
    }


# ------------------------------------------------------------------ 저장
def save_npz(out_dir: Path, variant: str, mode: str, results: dict[str, dict],
             meta: dict) -> Path:
    """NPZ (docs/04 규약: 절대경로 미기록, 입력 출처는 sig 리스트로)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr: dict[str, np.ndarray] = {}
    phases_out = []
    for ph, r in results.items():
        phases_out.append(ph)
        arr[f"{ph}__registered"] = np.array(bool(r.get("registered", False)))
        for key in ("auroc_margin_fixB", "auroc_len", "auroc_record", "auroc_episode"):
            arr[f"{ph}__{key}"] = np.array(float(r.get(key, np.nan)))
        arr[f"{ph}__tau"] = np.array(float(r.get("tau", np.nan)))
        arr[f"{ph}__B"] = np.array(int(r.get("B", 0)))
        arr[f"{ph}__sign"] = np.array(float(r.get("sign", 1.0)))
        # 게이트 CV 진단 (44 §2 개정)
        arr[f"{ph}__auroc_margin_seeds"] = np.array(
            r.get("auroc_margin_seeds", []), dtype=np.float64)
        arr[f"{ph}__auroc_len_seeds"] = np.array(
            r.get("auroc_len_seeds", []), dtype=np.float64)
        arr[f"{ph}__small_sample"] = np.array(bool(r.get("small_sample", False)))
        if "W_s" not in r:
            continue
        arr[f"{ph}__W_s"] = r["W_s"].astype(np.float32)
        arr[f"{ph}__W_f"] = r["W_f"].astype(np.float32)
        mh_g, mp_g, sp_g = r["global"]
        arr[f"{ph}__mh_global"] = mh_g.astype(np.float32)
        arr[f"{ph}__mp_global"] = mp_g.astype(np.float32)
        arr[f"{ph}__sp_global"] = sp_g.astype(np.float32)
        scenes = sorted(r["stats"])
        arr[f"{ph}__scenes"] = np.array(scenes, dtype=np.int32)
        for sc in scenes:
            mh, mp, sp = r["stats"][sc]
            arr[f"{ph}__scene{sc}__mh"] = mh.astype(np.float32)
            arr[f"{ph}__scene{sc}__mp"] = mp.astype(np.float32)
            arr[f"{ph}__scene{sc}__sp"] = sp.astype(np.float32)
    arr["phases"] = np.array(phases_out, dtype="<U64")
    arr["registered_phases"] = np.array(
        [p for p in phases_out if results[p].get("registered")], dtype="<U64")
    full_meta = {**meta, "variant": variant, "mode": mode}
    arr["meta_json"] = np.array(json.dumps(full_meta, ensure_ascii=False))
    path = out_dir / "condg.npz"
    np.savez_compressed(path, **arr)
    (out_dir / "metadata.json").write_text(
        json.dumps(full_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def force_register(results: dict[str, dict]) -> tuple[list[str], list[str]]:
    """--force-register: 게이트 판정을 무시하고 W 가 fit 된 전 phase 를 등록시킨다.

    탐색 라운드 전용 — 44 §2 의 "미달 cell = identity" 안전장치를 해제한다.
    게이트 판정 자체는 계속 계산·보존한다 (`gate_registered` / `gate_reason`).
    경성 표본 하한 등으로 **W 조차 못 뽑은** phase 는 강제 등록 대상이 아니고
    serve 에서 identity 로 남는다 (반환값 둘째 원소).

    반환: (강제 등록된 phase 목록, W 부재로 identity 잔존하는 phase 목록)
    """
    forced, identity = [], []
    for ph, r in results.items():
        r["gate_registered"] = bool(r.get("registered"))
        r["gate_reason"] = r.get("skip_reason", "")
        if "W_s" not in r:
            identity.append(ph)
            continue
        if not r["gate_registered"]:
            forced.append(ph)
        r["registered"] = True
        r["forced_register"] = not r["gate_registered"]
        r["skip_reason"] = ""
    return forced, identity


def _spread(v: list[float]) -> str:
    """중앙값(min–max) 문자열. 5-seed CV 진단용."""
    a = np.array([x for x in v if np.isfinite(x)], dtype=np.float64)
    if not a.size:
        return "-"
    return f"{np.median(a):.2f}({a.min():.2f}–{a.max():.2f})"


def print_table(variant: str, results: dict[str, dict]) -> None:
    print(f"\n[{variant}] phase 표 — 게이트 {GATE_SCHEME} (held-out; fail>succ 방향)", flush=True)
    print(f"  {'phase':<14}{'margin 중앙(min–max)':>22}{'길이 중앙(min–max)':>22}"
          f"{'win':>6}{'B':>5}{'tau':>10}  {'reg':<5} 표본(fit s/f)", flush=True)
    for ph, r in results.items():
        if "W_s" not in r:
            print(f"  {ph:<14}{'-':>22}{'-':>22}{'-':>6}{'-':>5}{'-':>10}  "
                  f"{'no':<5} ← {r.get('skip_reason', '')}", flush=True)
            continue
        flag = "  [소표본]" if r.get("small_sample") else ""
        if r.get("sub_floor"):
            flag += (f"  [극소표본] 클래스당 episode 하한을 "
                     f"{r.get('min_eps_per_class')} 로 낮춰 fit (기본 {MIN_EPS_PER_CLASS})")
        if r.get("forced_register"):
            flag += (f"  [force-register] 게이트 판정 무시: {ph}"
                     f"(원판정 FAIL: {r.get('gate_reason', '')})")
        win = f"{r.get('gate_wins', 0)}/{r.get('gate_seeds_ok', 0)}"
        print(f"  {ph:<14}{_spread(r.get('auroc_margin_seeds', [])):>22}"
              f"{_spread(r.get('auroc_len_seeds', [])):>22}{win:>6}{r['B']:>5d}"
              f"{r['tau']:>10.2f}  {'YES' if r['registered'] else 'no':<5} "
              f"({r.get('n_fit_succ', 0)}/{r.get('n_fit_fail', 0)})" + flag
              + (f"  ← {r['skip_reason']}" if r["skip_reason"] else ""), flush=True)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slug", required=True, help=f"task slug (가능: {sorted(TASKMAP)})")
    ap.add_argument("--phases", required=True, help="콤마 목록 (예: reach,grasp,pull)")
    ap.add_argument("--grid-root", type=Path, default=Path(DEFAULT_GRID),
                    help=f"grid rollout 루트 (기본 {DEFAULT_GRID})")
    ap.add_argument("--cells-tsv", type=Path, default=None,
                    help="정본 fit cell manifest (기본 "
                         "outputs/steer/online_pipe/manifests/<slug>_s5m5.tsv)")
    ap.add_argument("--cell-scenes", default=None,
                    help="manifest 대신 명시 셀: scene 범위 (예 0-9). --cell-noises 와 쌍. "
                         "설계 판정(c): 게이트 검증 체제 = s0-9 × n0-4 (44 문서 개정)")
    ap.add_argument("--cell-noises", default=None, help="noise 범위 (예 0-4)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="기본 outputs/steer/online_pipe/<slug>/condg_s5m5/")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"생성할 변형 (기본 {DEFAULT_VARIANTS})")
    ap.add_argument("--seed", type=int, default=0,
                    help="위약 라벨 순열·기타 rng seed (기본 0). 게이트 split seed 는 "
                         "0..--gate-seeds-1 로 고정")
    ap.add_argument("--gate-seeds", type=int, default=GATE_SEEDS,
                    help=f"등록 게이트 CV split seed 수 (기본 {GATE_SEEDS})")
    ap.add_argument("--layer", type=int, default=12, help="DiT 물리 layer (기본 12)")
    ap.add_argument("--denoise-step", type=int, default=3,
                    help="denoise step 인덱스 (기본 3 = 마지막)")
    ap.add_argument("--min-train-eps", type=int, default=MIN_TRAIN_EPS,
                    help=f"등록 게이트의 클래스당 train episode 하한 (기본 {MIN_TRAIN_EPS})")
    ap.add_argument("--min-eps-per-class", type=int, default=MIN_EPS_PER_CLASS,
                    help=f"클래스당 최소 episode 수 (기본 {MIN_EPS_PER_CLASS}). 탐색 라운드에서 "
                         "소표본 케이스(예: 성공 5판)를 그래도 fit 하려면 낮춘다 — "
                         "[극소표본] 딱지가 결과에 남는다")
    ap.add_argument("--episode-manifest", type=Path, default=None,
                    help="경로 명시 episode manifest (헤더 tsv: pkl_path·label·base_scene "
                         "[·cell_si·noise_idx]). 지정하면 glob/셀 필터를 쓰지 않고 나열된 "
                         "pkl 만 로드하고 scene 그룹 키로 base_scene 열을 쓴다. plan_id 가 "
                         "섞인 grid_root(v4 지터 vs v2/v1)에서는 평탄 si 가 충돌하므로 "
                         "이 모드를 쓸 것")
    ap.add_argument("--v4-jitter", action="store_true",
                    help="v4 지터 격자 — pkl 의 평탄 cell id(s=base_scene*100+k, base=+99)를 "
                         "base scene(//100)으로 접어 scene 통계·층화·등록을 base scene "
                         "단위로 묶는다 (셀 선택 자체는 평탄 si 그대로)")
    ap.add_argument("--cluster-bundle", type=Path, default=None,
                    help="ae_cluster.py --export-bundle NPZ. 지정 시 phase 라벨을 GT "
                         "feature_phases 대신 activation cluster 배정으로 치환한다 "
                         "(판정 좌표 = DiT L12 · 마지막 denoise · 49토큰 mean, --layer/"
                         "--denoise-step 과 무관). 이때 --phases 에는 클러스터 라벨을 넘긴다 "
                         "(예: --phases c0,c1,c2,c3,c4,c5,c6,c7). ⚠ split 시드가 "
                         "(seed, crc32(phase)) 라 phase 문자열이 바뀌면 6:4 분할이 통째로 "
                         "달라진다 — GT 라벨로 낸 기준선과 직접 비교 불가(기준선 리셋).")
    ap.add_argument("--cluster-task", default=None,
                    help="--cluster-bundle 안의 slug (centers.<slug>). 미지정 시 --slug 사용")
    ap.add_argument("--force-register", action="store_true",
                    help="게이트 판정을 무시하고 W가 fit된 전 phase를 등록 (탐색 라운드 전용 "
                         "— 45 스펙의 \"미달 cell=identity\" 안전장치 해제)")
    args = ap.parse_args()

    if args.slug not in TASKMAP:
        raise SystemExit(f"알 수 없는 slug {args.slug} (가능: {sorted(TASKMAP)})")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ("condg", "condg_pl", "condg_hs")]
    if bad:
        raise SystemExit(f"알 수 없는 variant {bad}")
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    grid_root = args.grid_root.expanduser()
    cells_tsv = (args.cells_tsv or
                 Path(f"outputs/steer/online_pipe/manifests/{args.slug}_s5m5.tsv")).expanduser()
    out_dir = (args.out_dir or
               Path(f"outputs/steer/online_pipe/{args.slug}/condg_s5m5")).expanduser()

    if args.episode_manifest is not None:
        # 경로 명시 모드 — plan_id 혼재로 평탄 si 가 충돌하는 grid_root 의 유일한 안전 경로.
        cells, cells_mode = set(), f"episode_manifest:{args.episode_manifest.name}"
    elif args.cell_scenes is not None or args.cell_noises is not None:
        if not (args.cell_scenes and args.cell_noises):
            raise SystemExit("--cell-scenes 와 --cell-noises 는 쌍으로 지정")

        def _rng(spec: str) -> list[int]:
            a, b = (spec.split("-") + [spec])[:2]
            return list(range(int(a), int(b) + 1))

        cells = {(s, n) for s in _rng(args.cell_scenes) for n in _rng(args.cell_noises)}
        cells_mode = f"explicit_s{args.cell_scenes}_n{args.cell_noises}"
    else:
        cells, cells_mode = load_fit_cells(cells_tsv)
    # 셀 선택(cells)은 평탄 si 기준 그대로 — --cell-scenes/--cell-noises 나 manifest 의
    # scene_idx 열이 평탄 si 를 담고 있어야 pkl glob 필터와 맞는다. 접기는 그룹 단계에서만.
    use_cluster = args.cluster_bundle is not None
    if args.episode_manifest is not None:
        eps = load_episodes_manifest(args.episode_manifest.expanduser(), grid_root,
                                     args.layer, args.denoise_step,
                                     cluster_feat=use_cluster)
    else:
        eps = load_episodes(args.slug, grid_root, cells, args.layer, args.denoise_step,
                            v4_jitter=args.v4_jitter, cluster_feat=use_cluster)
    if not eps:
        raise SystemExit(f"fit cell 에 해당하는 rollout 0개 (slug={args.slug})")
    phase_label_source = "gt:feature_phases"
    if use_cluster:
        # phase 라벨 = activation cluster 배정 (로더 반환 직후 치환 — 이후 로직은 phase
        # 문자열만 보므로 무수정). ⚠ split rng 가 crc32(phase) 라 GT 라벨 라운드와
        # 분할이 달라진다 = 기준선 리셋 (도움말 참조).
        api = _cluster_api()
        bundle = api.load_cluster_bundle(args.cluster_bundle)
        ctask = args.cluster_task or args.slug
        api.relabel_all_by_cluster(eps, bundle, ctask, tag=f" {args.slug}")
        phase_label_source = api.cluster_label_source(bundle)
        names = api.cluster_phase_names(bundle)
        bad_ph = [p for p in phases if p not in names]
        if bad_ph:
            raise SystemExit(f"--cluster-bundle 모드에서 --phases {bad_ph} 는 클러스터 라벨이 "
                             f"아니다 (가능: {','.join(names)})")
    n_s = sum(e["success"] for e in eps)
    used_cells = sorted((e["cell_si"], e["noise"]) for e in eps)   # provenance = 평탄 si
    print(f"[{args.slug}] episode {len(eps)} (succ {n_s} / fail {len(eps) - n_s}) "
          f"scene {sorted({e['scene'] for e in eps})} cells_mode={cells_mode}", flush=True)
    if args.v4_jitter or args.episode_manifest is not None:
        fold_summary = {}
        for e in eps:
            g = fold_summary.setdefault(e["scene"], [set(), 0])
            g[0].add((e["cell_si"], e["noise"]))
            g[1] += 1
        tag = "episode-manifest" if args.episode_manifest is not None else "v4-jitter"
        print(f"  [{tag}] base scene별 접기: " + ", ".join(
            f"s{sc}={len(v[0])}셀/{v[1]}ep" for sc, v in sorted(fold_summary.items())),
            flush=True)

    base_meta = {
        "spec": SPEC, "lambda": "1e-3*n", "seed": args.seed, "slug": args.slug,
        "phases_requested": phases, "layer": args.layer, "denoise_step": args.denoise_step,
        "feature": "DiT L%d · denoise %d · 49-token mean" % (args.layer, args.denoise_step),
        "state_phi": "[eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)] + 차분속도 = 18d",
        "split": f"게이트 전용 CV: seed 0..{args.gate_seeds - 1}, episode 6:4 scene 층화, "
                 "rng([seed, crc32(phase)])",
        "fit_scope": "배포 W_s/W_f·scene stats·global fallback = 전체 fit 셋 (split 무관)",
        "gate_scheme": GATE_SCHEME,
        "gate": f"{args.gate_seeds}-seed CV — 고정B margin AUROC 중앙값 > 길이단독 중앙값 "
                f"AND 과반({args.gate_seeds // 2 + 1}/{args.gate_seeds}) seed 에서 margin>길이 "
                f"AND 전체 fit 셋 클래스당 episode ≥ {HARD_MIN_EPS}",
        "gate_seeds": args.gate_seeds,
        "force_register": bool(args.force_register),
        "force_register_note": ("게이트 판정 무시하고 W fit된 전 phase 등록 (탐색 라운드 전용). "
                                "판정 자체는 gate_registered/gate_reason 로 보존"
                                if args.force_register else ""),
        "small_sample_rule": f"클래스당 episode < {args.min_train_eps} 이면 등록하되 "
                             "small_sample=True 딱지",
        "min_eps_per_class": int(args.min_eps_per_class),
        "sub_floor_rule": (f"클래스당 episode 하한을 기본 {MIN_EPS_PER_CLASS} 아래"
                           f"({args.min_eps_per_class})로 내림 — fit 된 phase 에 "
                           "sub_floor=True 딱지"
                           if args.min_eps_per_class < MIN_EPS_PER_CLASS else ""),
        "tau": f"{args.gate_seeds}-seed held-out 성공 episode 고정B margin pool 의 90퍼센타일",
        "B": "seed 별 max(3, train 성공 dwell 25퍼센타일) 의 중앙값(int)",
        "sign_note": "margin 부호 고정(m 클수록 실패) — 역전 cell 은 게이트가 배제",
        "fit_cells": [[int(s), int(n)] for s, n in used_cells],
        "fit_cells_source": cells_mode,
        "phase_label_source": phase_label_source,
        "input_sigs": [e["sig"] for e in eps],   # docs/04 — 경로 아닌 내용 지문
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "docs/steering/44_cond_guidance_operator.md + "
                  "scripts/analysis/grid_phase/cond_margin.py",
    }
    if args.v4_jitter:   # 기본 off 일 때 meta 를 건드리지 않도록 켰을 때만 추가
        base_meta["v4_jitter"] = True
        base_meta["scene_grouping"] = ("셀 선택=평탄 si(base*100+k, base=+99) / "
                                       "그룹·NPZ scene 키=base scene(평탄//100)")
        base_meta["fit_base_scenes"] = sorted({int(e["scene"]) for e in eps})
    if args.episode_manifest is not None:
        base_meta["episode_manifest"] = args.episode_manifest.name   # 절대경로 미기록
        base_meta["scene_grouping"] = ("episode manifest 의 base_scene 열 = 그룹·NPZ scene 키 "
                                       "(plan_id 혼재 시 평탄 si 충돌 회피)")
        base_meta["fit_base_scenes"] = sorted({int(e["scene"]) for e in eps})
        base_meta["label_source"] = "episode manifest 의 label 열 (pkl success 와 대조 검증)"

    labels_treat = [e["success"] for e in eps]
    res_treat: dict[str, dict] = {}
    for ph in phases:
        r = fit_phase(eps, labels_treat, ph, args.seed, args.min_train_eps, args.gate_seeds,
                      args.min_eps_per_class)
        if r is None:
            print(f"  [{ph}] phase 없음(codebook) — 건너뜀", flush=True)
            continue
        res_treat[ph] = r
    if not res_treat:
        raise SystemExit("처리 가능한 phase 0개")
    # 기본 하한 미만 표본으로 fit 된 phase 가 하나라도 있으면 NPZ meta 에도 남긴다
    base_meta["sub_floor"] = any(r.get("sub_floor") for r in res_treat.values())
    if args.force_register:
        forced, identity = force_register(res_treat)
        print(f"  [force-register] 게이트 판정 무시 — 강제 등록 {sorted(forced) or '없음'}",
              flush=True)
        for ph in forced:
            print(f"    · {ph} (원판정 FAIL: {res_treat[ph].get('gate_reason', '')})", flush=True)
        if identity:
            print(f"    · W 산출 실패로 identity 잔존(강제 등록 불가): {sorted(identity)}",
                  flush=True)
    print_table("condg", res_treat)
    reg_phases = {ph for ph, r in res_treat.items() if r.get("registered")}
    print(f"  등록 phase: {sorted(reg_phases) or '없음'}", flush=True)

    written = []
    summary: dict[str, dict] = {}
    if "condg" in variants:
        p = save_npz(out_dir / "condg", "condg", "margin_gated_contrast", res_treat, base_meta)
        written.append(p)
    if "condg_hs" in variants:
        # W·τ·B 는 처치와 동일. 적용식만 h̃′ = (1−β)h̃ + β·φ̃W_s (44 §3) → mode 로만 구분.
        p = save_npz(out_dir / "condg_hs", "condg_hs", "success_mimic", res_treat,
                     {**base_meta, "note": "condg 와 동일한 W/τ/B — 적용식만 성공-모방 단독"})
        written.append(p)
    if "condg_pl" in variants:
        labels_pl = scene_stratified_permute(eps, args.seed)
        res_pl: dict[str, dict] = {}
        for ph in res_treat:
            r = fit_phase(eps, labels_pl, ph, args.seed, args.min_train_eps, args.gate_seeds,
                          args.min_eps_per_class)
            if r is None:
                continue
            # 44 §3: 위약은 자기 등록 게이트를 **우회**하고, 등록 집합을 처치와 일치시킨다.
            #   - 처치가 등록한 phase → 강제 등록 (게이트에 걸려 identity 가 되면
            #     타이밍-섭동 대조 기능을 잃는다).
            #   - 처치가 등록 안 한 phase → 자기 AUROC 가 좋아도 미등록 (처치가 손대지
            #     않는 cell 에 위약만 개입하면 짝이 맞지 않는다).
            # 자기 게이트 판정은 own_gate_* 로 남긴다.
            r["own_gate_registered"] = bool(r.get("registered"))
            r["own_gate_reason"] = r.get("skip_reason", "")
            r["registered"] = ph in reg_phases and "W_s" in r
            r["forced_register"] = bool(r["registered"] and not r["own_gate_registered"])
            r["skip_reason"] = "" if r["registered"] else (
                "처치 미등록 phase — 위약도 미등록(짝맞춤)" if ph not in reg_phases
                else r.get("skip_reason", ""))
            if ph in reg_phases and "W_s" not in r:
                print(f"  [warn] 위약 {ph}: W 산출 실패로 강제 등록 불가 "
                      f"({r.get('own_gate_reason')})", flush=True)
            res_pl[ph] = r
        print_table("condg_pl", res_pl)
        p = save_npz(out_dir / "condg_pl", "condg_pl", "margin_gated_contrast",
                     res_pl, {**base_meta,
                              "placebo": "episode 라벨 scene-층화 순열, 등록 게이트 우회",
                              "placebo_labels": [int(x) for x in labels_pl]})
        written.append(p)
        summary["condg_pl"] = {ph: _row_summary(r) for ph, r in res_pl.items()}

    summary["condg"] = {ph: _row_summary(r) for ph, r in res_treat.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_summary.json").write_text(
        json.dumps({**base_meta, "variants": variants,
                    "registered_phases": sorted(reg_phases),
                    "results": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in written:
        print(f"[write] {p.relative_to(out_dir.parent) if out_dir.parent in p.parents else p.name}",
              flush=True)
    print(f"FIT_CONDG_DONE {args.slug}", flush=True)


def _row_summary(r: dict) -> dict:
    """JSON 요약 (행렬 제외)."""
    return {k: v for k, v in r.items() if k not in ("W_s", "W_f", "stats", "global")}


if __name__ == "__main__":
    sys.exit(main())
