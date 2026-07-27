#!/usr/bin/env python
"""exp5/G1 — SAE 입력 빌더: rollout pkl → layer 별 per-token 행렬 + 행 메타.

핸드아웃 `docs/steering/30_sae_g1_port_handout.md` §4 Phase B 구현.

입력 소스 2종 (택1):
  (a) `--manifest` (fit30 경로, 기존): 탭 3열 tsv + `--cell` 필터.
  (b) `--scan-dir`  (scene-matched 컬렉션, 신규): manifest tsv 가 없는 디렉토리를 glob.
      라벨은 파일명 `...--ep{N}--succ{0|1}.pkl` 에서 읽고 pkl 내부 `episode_success` 와
      **대조**(불일치 시 즉시 에러). scenario_seed/inference_seed/layout/style 는 pkl 내부값.

split 축 2종 (`--split-by`):
  - `episode` (기본, fit30 하위호환): episode 단위, layout×style 층화.
  - `scene`   (scene-matched 전용 권장): **scenario_seed 단위** held-out. 같은 scene 의 다른
    inference_seed 판이 train/test 양쪽에 들어가면 scene probe 가 누수로 무의미해진다
    (scene-matched 격자는 scene 당 8판이므로 episode split 은 사실상 전부 누수).

**실행 위치 = 승준**(pkl 이 거기 있음). 그래서 이 파일은 **numpy/torch/stdlib 만** 쓴다
(승준 base python3 에는 torch 가 없으므로 `~/anaconda3/bin/python`, scipy·sklearn 없음).

핵심 계약 (2026-07-27 원격 정찰 실측):
  - manifest tsv: 헤더 없음, `#` 주석, 탭 3열 = pkl경로 / label(1=succ) / scenario_seed.
    cell 필터는 pkl 경로의 **부모 디렉토리 이름** (fit_mean_diff.load_cell_rolls:403 관례).
  - pkl["hidden_states"] = list[record], record = torch.Tensor fp16 [L=7, K=4, T=49, D=1536].
  - pkl["capture_layers"] = [0,2,4,8,10,12,15] → 물리 layer 는 반드시 `cap.index(L)` 로 인덱싱
    (리스트 위치 != layer 번호. 하드코딩 금지 — 핸드아웃 §6-7).
  - K(denoise) 축은 **평균** — fit_mean_diff.py:420-421 `np.asarray(rec).mean(axis=1)` 관례와 동일.
  - T=49 = state[0:1] + future[1:33] + action[33:49] (fit_phase_conceptor SEGMENTS 규약).
  - pkl["feature_phases"] len == record 수, ep_meta.layout_id/style_id, 최상위 scenario_seed·
    episode_idx·episode_success·inference_seed.

동료 레포(robots-oh task_classification) 빌더(`make_phase_dataset.py:102-109`)를 **재사용하지
않는 이유**: 그 빌더는 토큰 축을 평균으로 없앤다. 우리는 per-token 보존이 필수다
(memory `feedback-no-rollout-pooling`: phase 는 timestep 구분이 load-bearing).
PCA(동료 `phase/data/pca.py`)도 쓰지 않는다 — 원본 1536-d 에 overcomplete SAE 를 얹는 게
이번 재설계의 전제(핸드아웃 §2.2-2). 표준화는 train-split feature-wise mean/std 만.

메모리 주의: `np.asarray(rec[li], dtype=np.float32)` — **layer 인덱싱을 float32 변환보다 먼저**.
record 전체를 float32 로 올리면 7배 낭비(4배 dtype × layer 수).

산출 (--out-dir):
  X_L{L}.npz    : X [N_rows, 1536] fp16          (원시값. 표준화는 학습 시 적용)
  stats_L{L}.npz: mean [1536] / std [1536] float32 (train split 행만으로 계산)
  meta.npz      : 행 단위 메타 (아래 META_FIELDS) + phase 코드북
  split.json    : episode → split 배정표 (재현성)

사용 예 (fit30 / manifest):
  ~/anaconda3/bin/python scripts/scene_sae/build_sae_inputs.py \
      --manifest ~/datasets/.../manifests_fit30/pq3_drawer_left/fit_manifest.tsv \
      --cell pq3_drawer_left --layers 0,2,8,10,12 \
      --out-dir ~/workspace/temporal_vla/outputs/scene_sae/pq3_drawer_left

사용 예 (scene-matched 160판 / scan-dir + scene split):
  ~/anaconda3/bin/python scripts/scene_sae/build_sae_inputs.py \
      --scan-dir ~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/scene_matched_exp41/\
pq3_drawer_right/raw_rollouts/OpenDrawer/pq3_drawer_right \
      --split-by scene --layers 8,10,12 \
      --out-dir ~/workspace/temporal_vla/outputs/scene_sae/scene_matched_drawer_right

  ⚠ 메모리: 160판 ≈ 60만+ 행 × 1536 → layer 당 fp16 ~1.8GB (X 저장 시 concat 사본 포함
  일시 ~2 배). 승준 여유를 감안한 **기본 권장은 3 layer (8,10,12)**. 더 필요하면
  `--layers 0,2` 처럼 나눠서 **같은 --out-dir 에 이어서** 실행하면 된다 (meta/split 은
  동일 seed·동일 episode 목록이므로 재기록돼도 동일).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

# 토큰 세그먼트 (N1.5 DiT T=49) — fit_phase_conceptor_n15.SEGMENTS 규약과 동일.
SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))

META_FIELDS = ("episode_idx", "record_idx", "token_idx", "token_seg", "phase_code",
               "success", "scenario_seed", "inference_seed", "layout_id", "style_id", "split")

SPLIT_NAMES = ("train", "val", "test")

# scan-dir 파일명 규약: `task7--ep{N}--succ{0|1}.pkl` (prefix 는 임의).
SCAN_RE = re.compile(r"--ep(\d+)--succ([01])\.pkl$")


# --------------------------------------------------------------------- manifest
def read_manifest(manifest: Path, cell: str) -> list[dict]:
    """manifest tsv → [{pkl, label, scene}] (fit_mean_diff.load_cell_rolls:393-410 관례)."""
    rows = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if cell and p.parent.name != cell:
            continue
        rows.append({"pkl": p, "label": int(parts[1]),
                     "scene": parts[2] if len(parts) > 2 else ""})
    if not rows:
        raise SystemExit(f"manifest 에 cell={cell} 행 없음: {manifest}")
    missing = [str(r["pkl"]) for r in rows if not r["pkl"].exists()]
    if missing:
        raise SystemExit(f"pkl 누락 {len(missing)}개: {missing[:3]}")
    return rows


def scan_dir(root: Path, pattern: str) -> list[dict]:
    """manifest 없는 컬렉션: 디렉토리 glob → [{pkl, label, scene}] (scene 은 pkl 로드 시 채움).

    라벨 = 파일명 `--succ{0|1}`. pkl 내부 `episode_success` 와의 대조는 로드 시점에
    (`--strict-label`) 수행한다 — 여기서 pkl 을 열면 400MB × N 을 두 번 읽게 된다.
    """
    files = sorted(root.glob(pattern))
    rows = []
    for p in files:
        m = SCAN_RE.search(p.name)
        if m is None:
            raise SystemExit(f"파일명 규약 위반 (…--ep{{N}}--succ{{0|1}}.pkl): {p.name}")
        rows.append({"pkl": p, "label": int(m.group(2)), "scene": "",
                     "name_episode_idx": int(m.group(1))})
    if not rows:
        raise SystemExit(f"scan-dir 에 pkl 없음: {root}/{pattern}")
    rows.sort(key=lambda r: r["name_episode_idx"])
    dup = {r["name_episode_idx"] for r in rows}
    if len(dup) != len(rows):
        raise SystemExit(f"episode_idx 중복 (파일명 기준) — {len(rows)} 파일 / {len(dup)} idx")
    return rows


def seg_of_token(t: int) -> int:
    for i, (_n, lo, hi) in enumerate(SEGMENTS):
        if lo <= t < hi:
            return i
    raise ValueError(f"token {t} 세그먼트 밖 (T 계약 위반)")


def token_seg_vector(T: int) -> np.ndarray:
    return np.asarray([seg_of_token(t) for t in range(T)], dtype=np.int8)


# ------------------------------------------------------------------------ split
def assign_splits(eps: list[dict], seed: int) -> dict[int, int]:
    """episode 단위 split 배정 (record/token split 은 자기상관 누수 — 핸드아웃 §6-2).

    층화 = scene 그룹 (layout_id, style_id) — fit30 drawer_left 에서 두 필드는 완전 공선이라
    사실상 layout 5종. 그룹 안에서 succ/fail 을 번갈아 배열해(성공률 쏠림 방지) 앞에서부터
    train → val → test 로 자른다. **모든 layout 이 train 과 test 양쪽에 나타나야** scene probe
    (라벨=layout) 가 held-out 평가 가능하므로:
      n == 1  → train (그 layout 은 probe 평가 불가, 경고)
      n == 2  → train 1 / test 1   (실측 layout (2,2) 케이스 — val 없음)
      n >= 3  → test = max(1, round(.20n)), val = max(1, round(.133n)) if n>=5 else 0,
                나머지 train
    실측 drawer_left(6/2/7/8/7 ep)에서 정확히 train 20 / val 4 / test 6 이 된다.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = {}
    for e in eps:
        groups.setdefault((e["layout_id"], e["style_id"]), []).append(e)

    out: dict[int, int] = {}
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        g = groups[key]
        succ = sorted([e for e in g if e["success"] == 1], key=lambda e: e["episode_idx"])
        fail = sorted([e for e in g if e["success"] != 1], key=lambda e: e["episode_idx"])
        rng.shuffle(succ)
        rng.shuffle(fail)
        order = []
        for i in range(max(len(succ), len(fail))):      # succ/fail 번갈아 = 성공률 균형
            if i < len(succ):
                order.append(succ[i])
            if i < len(fail):
                order.append(fail[i])
        n = len(order)
        if n == 1:
            n_test = n_val = 0
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test = max(1, int(round(0.20 * n)))
            n_val = max(1, int(round(0.1333 * n))) if n >= 5 else 0
            n_val = min(n_val, max(0, n - n_test - 1))
        n_train = n - n_test - n_val
        for i, e in enumerate(order):
            out[e["episode_idx"]] = 0 if i < n_train else (1 if i < n_train + n_val else 2)
    return out


def _spread_slots(n: int, n_val: int, n_test: int) -> list[int]:
    """길이 n 의 split 슬롯을 균등 간격으로 배치 (val/test 를 먼저 찍고 나머지 train).

    난수 없이 결정적. **주의(2026-07-27 리뷰 #3)**: 이 함수를 layout 그룹 round-robin 으로
    섞은 전역 순서와 zip 하면 슬롯 간격(n/cnt)과 그룹 주기(n_groups)가 공약수를 가질 때
    **에일리어싱**이 생긴다 — 20 scene = layout 2그룹 × 10 이면 test 슬롯이 2,8,12,18 로
    전부 짝수 위치 = 한 그룹에 몰린다. 그래서 이제 이 함수는 **그룹 내부**(같은 layout 안,
    성공률 오름차순)에서만 쓰고, 그룹 간 배분은 `_alloc_counts` 비례배분이 담당한다.
    (회귀 테스트: tests/test_scene_sae_build.py::test_spread_slots_round_robin_aliases)
    """
    slots: list[int | None] = [None] * n
    for code, cnt in ((2, n_test), (1, n_val)):
        for k in range(cnt):
            pos = int(round((k + 0.5) * n / max(cnt, 1))) % n
            while slots[pos] is not None:
                pos = (pos + 1) % n
            slots[pos] = code
    return [0 if s is None else s for s in slots]


def _alloc_counts(sizes: list[int], total: int, caps: list[int]) -> list[int]:
    """그룹 크기 비례 배분 (최대잉여법) — 각 그룹 cap 을 넘지 않게.

    층화의 핵심: test/val 정원을 layout 그룹에 **명시적으로** 나눠 준다. round-robin +
    전역 슬롯 방식은 주기 공명으로 한 그룹에 몰릴 수 있었다 (리뷰 #3).
    """
    tot = sum(sizes)
    if tot <= 0 or total <= 0:
        return [0] * len(sizes)
    quota = [total * s / tot for s in sizes]
    base = [min(int(q), c) for q, c in zip(quota, caps)]
    order = sorted(range(len(sizes)),
                   key=lambda i: (-(quota[i] - int(quota[i])), -sizes[i], i))
    rem = total - sum(base)
    while rem > 0:
        progressed = False
        for i in order:
            if rem <= 0:
                break
            if base[i] < caps[i]:
                base[i] += 1
                rem -= 1
                progressed = True
        if not progressed:                              # 모든 그룹이 cap 포화
            break
    return base


def assign_scene_splits(eps: list[dict], seed: int,
                        val_frac: float = 0.10, test_frac: float = 0.20) -> dict[int, int]:
    """**scenario_seed 단위** split → {scenario_seed: split_code}.

    scene-matched 격자(scene 20 × inference_seed 8)에서 episode 단위 split 을 쓰면 같은
    scene 의 다른 denoise seed 판이 train/test 양쪽에 들어가 scene probe 가 누수로 뻥튀기된다.
    그래서 scene 자체를 held-out 한다. n=20 → train 14 / val 2 / test 4.

    층화 (2026-07-27 리뷰 #3 개정 — 구 방식은 round-robin×균등슬롯 에일리어싱으로
    test 가 단일 layout 에 몰릴 수 있었다):
      (1) layout×style 그룹으로 나눈다.
      (2) test/val 정원을 그룹 **크기 비례**로 명시 배분(`_alloc_counts`, cap = size-1 →
          모든 그룹이 train 에 최소 1 개는 남는다).
      (3) 그룹 **안에서** 성공률 오름차순 정렬 후 `_spread_slots` 로 균등 간격 배정
          (같은 그룹에서 성공률 극단만 뽑히는 것 방지).
    seed 는 성공률 동률 tie-break 에만 쓰이며 배정은 결정적으로 재현된다.
    """
    rng = np.random.default_rng(seed)
    scenes: dict[int, dict] = {}
    for e in eps:
        s = scenes.setdefault(int(e["scenario_seed"]),
                              {"n": 0, "succ": 0, "layout": e["layout_id"], "style": e["style_id"]})
        s["n"] += 1
        s["succ"] += int(e["success"] == 1)

    groups: dict[tuple, list[int]] = {}
    for sc, info in scenes.items():
        groups.setdefault((info["layout"], info["style"]), []).append(sc)
    for key in groups:                                   # 그룹 내 = 성공률 오름차순
        tie = {sc: float(rng.random()) for sc in groups[key]}
        groups[key].sort(key=lambda sc: (scenes[sc]["succ"] / max(scenes[sc]["n"], 1), tie[sc], sc))

    n = len(scenes)
    if n < 3:
        raise SystemExit(f"scene 수 {n} — scene held-out split 불가 (>=3 필요)")
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    n_val = min(n_val, max(0, n - n_test - 1))

    gkeys = sorted(groups, key=lambda k: (str(k[0]), str(k[1])))
    sizes = [len(groups[k]) for k in gkeys]
    caps_test = [max(0, s - 1) for s in sizes]           # 그룹당 train 최소 1 확보
    test_g = _alloc_counts(sizes, n_test, caps_test)
    caps_val = [max(0, s - 1 - t) for s, t in zip(sizes, test_g)]
    val_g = _alloc_counts(sizes, n_val, caps_val)

    out: dict[int, int] = {}
    for gi, key in enumerate(gkeys):
        g = groups[key]
        slots = _spread_slots(len(g), val_g[gi], test_g[gi])
        for i, sc in enumerate(g):
            out[sc] = slots[i]
    return out


def split_group_coverage(item2split: dict, group_of: dict) -> dict[str, list]:
    """split 별로 등장하는 layout×style 그룹 목록 (진단 + coverage assert 근거)."""
    cov: dict[str, set] = {name: set() for name in SPLIT_NAMES}
    for item, s in item2split.items():
        cov[SPLIT_NAMES[s]].add(group_of[item])
    return {name: sorted(str(g) for g in v) for name, v in cov.items()}


def assert_split_group_coverage(item2split: dict, group_of: dict, axis: str) -> dict[str, list]:
    """각 split 이 **가능한 한** ≥2 layout 그룹을 덮는지 검사 (리뷰 #3).

    단일 그룹에 몰린 split 은 "scene probe held-out" 이 사실상 한 layout 만 보는 것이라
    결과 해석이 무의미해진다. 요구치는 min(2, 전체 그룹 수, 그 split 의 항목 수) —
    그룹이 1 개뿐이거나 split 항목이 1 개면 자동 통과.
    """
    cov = split_group_coverage(item2split, group_of)
    n_groups = len({group_of[i] for i in item2split})
    counts = {name: sum(1 for _i, s in item2split.items() if SPLIT_NAMES[s] == name)
              for name in SPLIT_NAMES}
    bad = []
    for name in SPLIT_NAMES:
        need = min(2, n_groups, counts[name])
        if len(cov[name]) < need:
            bad.append(f"{name}: 그룹 {cov[name]} ({len(cov[name])}개) < 요구 {need} "
                       f"(항목 {counts[name]}개)")
    if bad:
        raise SystemExit(f"split({axis}) layout coverage 위반 — " + " | ".join(bad))
    return cov


# ------------------------------------------------------------------- 검증/지문
def validate_episodes(eps: list[dict], require_scene_meta: bool) -> None:
    """수집 스캔 무결성 검증 (리뷰 #9).

    ① (scenario_seed, inference_seed) 쌍 유일 — 중복이면 같은 판을 두 번 세는 것.
    ② seed = -1 금지 (pkl 에 필드가 없었다는 뜻 → scene split 이 조용히 무의미해진다).
    ③ 같은 scene 안의 layout_id/style_id 일관 — scene = fixture 고정이 계약이므로
      한 scene 에서 layout 이 갈리면 수집이 섞인 것.
    """
    errs: list[str] = []
    if require_scene_meta:
        bad_seed = [e["episode_idx"] for e in eps
                    if int(e["scenario_seed"]) < 0 or int(e["inference_seed"]) < 0]
        if bad_seed:
            errs.append(f"scenario_seed/inference_seed = -1 인 episode {len(bad_seed)}개: "
                        f"{bad_seed[:5]}")
        seen: dict[tuple, int] = {}
        dups = []
        for e in eps:
            key = (int(e["scenario_seed"]), int(e["inference_seed"]))
            if key in seen:
                dups.append(f"{key} = ep{seen[key]} & ep{e['episode_idx']}")
            seen[key] = e["episode_idx"]
        if dups:
            errs.append(f"(scenario_seed, inference_seed) 중복 {len(dups)}건: {dups[:5]}")
    fixture: dict[int, tuple] = {}
    mixed = []
    for e in eps:
        sc = int(e["scenario_seed"])
        if sc < 0:                      # seed 미기록 컬렉션(fit30 일부)은 scene 개념이 없다
            continue
        f = (int(e["layout_id"]), int(e["style_id"]))
        if sc in fixture and fixture[sc] != f:
            mixed.append(f"scene {sc}: {fixture[sc]} vs {f} (ep{e['episode_idx']})")
        fixture.setdefault(sc, f)
    if mixed:
        errs.append(f"같은 scene 내 layout/style 불일치 {len(mixed)}건: {mixed[:5]}")
    if errs:
        raise SystemExit("스캔 검증 실패 — " + " | ".join(errs))


def row_fingerprint(eps: list[dict]) -> str:
    """행 구성 지문 = sha256(정렬된 'episode_idx:n_records' 목록)[:12] (리뷰 #10).

    같은 --out-dir 에 layer 를 나눠 이어서 빌드할 때, 사이에 수집이 바뀌면 X_L*.npz 끼리
    행이 어긋난다(조용한 오정렬 = 모든 하류 결과 무효). meta.npz 와 각 X_L*.npz 에 같은
    지문을 박아 두고 이어쓰기·로드 시점에 대조한다.
    """
    payload = ";".join(f"{int(e['episode_idx'])}:{int(e['n_records'])}"
                       for e in sorted(eps, key=lambda x: int(x["episode_idx"])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="rollout pkl → layer 별 per-token SAE 입력")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="fit30 manifest tsv (기존 경로)")
    src.add_argument("--scan-dir", type=Path,
                     help="manifest 없는 컬렉션 디렉토리 (파일명 --ep{N}--succ{0|1}.pkl)")
    ap.add_argument("--scan-glob", default="*.pkl", help="--scan-dir 패턴 (기본 *.pkl)")
    ap.add_argument("--cell", default="",
                    help="pkl 부모 디렉토리 이름. --manifest 필수, --scan-dir 이면 디렉토리명 기본값")
    ap.add_argument("--split-by", default="episode", choices=["episode", "scene"],
                    help="episode=기존 층화 split / scene=scenario_seed 단위 held-out")
    ap.add_argument("--layers", default="0,2,8,10,12",
                    help="물리 layer 목록. capture_layers 와의 교집합만 사용 "
                         "(scene-matched 160판 권장 기본: 8,10,12)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="X 저장 dtype")
    ap.add_argument("--seed", type=int, default=424101, help="split 배정 seed (exp4-1 관례)")
    ap.add_argument("--limit-eps", type=int, default=0, help=">0 이면 앞 N episode 만 (smoke)")
    ap.add_argument("--strict-label", default="auto", choices=["auto", "on", "off"],
                    help="파일명/manifest 라벨 vs pkl episode_success 대조. auto=scan-dir 에서 on")
    args = ap.parse_args()

    want_layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = np.float16 if args.dtype == "fp16" else np.float32

    if args.manifest is not None:
        if not args.cell:
            ap.error("--manifest 모드에서는 --cell 이 필요합니다")
        cell = args.cell
        rows = read_manifest(args.manifest.expanduser(), cell)
        source = {"mode": "manifest", "manifest": str(args.manifest)}
    else:
        scan_root = args.scan_dir.expanduser()
        cell = args.cell or scan_root.name
        rows = scan_dir(scan_root, args.scan_glob)
        source = {"mode": "scan-dir", "scan_dir": str(scan_root), "scan_glob": args.scan_glob}
    strict = args.strict_label == "on" or (args.strict_label == "auto" and args.manifest is None)
    if args.limit_eps:
        rows = rows[: args.limit_eps]
    print(f"[build] source={source['mode']} cell={cell} episodes={len(rows)} "
          f"layers={want_layers} split_by={args.split_by} strict_label={strict}", flush=True)

    # ---- 1 pass: 로드하며 layer 별 X 누적 + 행 메타 수집
    X_acc: dict[int, list[np.ndarray]] = {L: [] for L in want_layers}
    meta_acc: dict[str, list[np.ndarray]] = {f: [] for f in META_FIELDS if f != "split"}
    phase_codebook: dict[str, int] = {}
    eps_info: list[dict] = []
    contract = None
    t_start = time.time()

    for i, m in enumerate(rows):
        with open(m["pkl"], "rb") as f:
            d = pickle.load(f)
        hs = d.get("hidden_states") or []
        if not hs:
            raise SystemExit(f"{m['pkl']}: hidden_states 비어 있음")
        cap = list(d.get("capture_layers") or [])
        if not cap:
            raise SystemExit(f"{m['pkl']}: capture_layers 없음 — layer 인덱싱 불가")
        miss = [L for L in want_layers if L not in cap]
        if miss:
            raise SystemExit(f"{m['pkl']}: capture_layers={cap} 에 요청 layer {miss} 없음")
        li = {L: cap.index(L) for L in want_layers}      # 물리 layer → 리스트 위치

        phases = list(d.get("feature_phases") or [])
        if len(phases) != len(hs):
            raise SystemExit(f"{m['pkl']}: feature_phases {len(phases)} != records {len(hs)}")

        shape0 = tuple(np.shape(hs[0]))
        here = (tuple(cap), shape0[1], shape0[2], shape0[3])   # (cap, K, T, D)
        if contract is None:
            contract = here
            print(f"[build] contract capture_layers={cap} K={here[1]} T={here[2]} D={here[3]}",
                  flush=True)
        elif here != contract:
            raise SystemExit(f"{m['pkl']}: pkl 계약 불일치 {here} != {contract}")
        K, T, D = contract[1], contract[2], contract[3]
        seg_vec = token_seg_vector(T)

        ep_meta = d.get("ep_meta") or {}
        ep_success = int(d.get("episode_success", -1))
        # 파일명/manifest 라벨 vs pkl 내부 episode_success 대조 (scan-dir 기본 on)
        if strict:
            if ep_success not in (0, 1):
                raise SystemExit(f"{m['pkl']}: episode_success 없음/이상 ({ep_success}) — 대조 불가")
            if ep_success != int(m["label"]):
                raise SystemExit(f"{m['pkl']}: 라벨 불일치 — 파일명/manifest={m['label']} "
                                 f"vs pkl episode_success={ep_success}")
        elif ep_success in (0, 1) and ep_success != int(m["label"]):
            print(f"[warn] {m['pkl'].name}: 라벨 불일치 manifest={m['label']} "
                  f"pkl={ep_success} (manifest 우선 — fit30 관례)", flush=True)
        if "name_episode_idx" in m and int(d.get("episode_idx", m["name_episode_idx"])) != \
                m["name_episode_idx"]:
            raise SystemExit(f"{m['pkl']}: episode_idx 불일치 — 파일명={m['name_episode_idx']} "
                             f"vs pkl={d.get('episode_idx')}")

        ep = {
            "episode_idx": int(d.get("episode_idx", m.get("name_episode_idx", i))),
            "success": int(m["label"]),          # manifest 라벨 override (fit_phase_conceptor 관례)
            "episode_success": ep_success,
            "scenario_seed": int(d.get("scenario_seed", -1)),
            "inference_seed": int(d.get("inference_seed", -1)),
            "layout_id": int(ep_meta.get("layout_id", -1)),
            "style_id": int(ep_meta.get("style_id", -1)),
            "n_records": len(hs),
            "pkl": str(m["pkl"]),
        }
        eps_info.append(ep)

        for L in want_layers:
            idx = li[L]
            # ★ layer 인덱싱을 float32 변환 **전에** (record 전체 변환 시 메모리 7배)
            chunk = np.empty((len(hs) * T, D), dtype=save_dtype)
            for r, rec in enumerate(hs):
                a = np.asarray(rec[idx], dtype=np.float32)       # [K, T, D]
                x = a.mean(axis=0)                               # K(denoise) 평균 — fit_mean_diff:421
                chunk[r * T:(r + 1) * T] = x.astype(save_dtype)
            X_acc[L].append(chunk)

        n_rec = len(hs)
        rec_idx = np.repeat(np.arange(n_rec, dtype=np.int32), T)
        tok_idx = np.tile(np.arange(T, dtype=np.int16), n_rec)
        for ph in phases:
            phase_codebook.setdefault(str(ph), len(phase_codebook))
        ph_code = np.repeat(
            np.asarray([phase_codebook[str(p)] for p in phases], dtype=np.int16), T)
        meta_acc["episode_idx"].append(np.full(n_rec * T, ep["episode_idx"], np.int32))
        meta_acc["record_idx"].append(rec_idx)
        meta_acc["token_idx"].append(tok_idx)
        meta_acc["token_seg"].append(np.tile(seg_vec, n_rec))
        meta_acc["phase_code"].append(ph_code)
        meta_acc["success"].append(np.full(n_rec * T, ep["success"], np.int8))
        meta_acc["scenario_seed"].append(np.full(n_rec * T, ep["scenario_seed"], np.int64))
        meta_acc["inference_seed"].append(np.full(n_rec * T, ep["inference_seed"], np.int64))
        meta_acc["layout_id"].append(np.full(n_rec * T, ep["layout_id"], np.int32))
        meta_acc["style_id"].append(np.full(n_rec * T, ep["style_id"], np.int32))

        del d, hs
        print(f"[build] ({i+1}/{len(rows)}) ep{ep['episode_idx']} succ={ep['success']} "
              f"rec={n_rec} layout={ep['layout_id']} style={ep['style_id']} "
              f"scene={ep['scenario_seed']} inf={ep['inference_seed']} "
              f"({time.time()-t_start:.0f}s)", flush=True)

    # ---- split
    # 두 축을 **한 번의 pkl pass 에서 모두** 기록한다. X 재생성이 비싼데(160판 × 400MB) 축만
    # 바꿔 다시 도는 건 낭비고, 두 축은 쓰임이 다르다:
    #   split_episode : scene 안에서 나눔 → scenario_seed(20클래스) probe 가 평가 가능한 유일한 축.
    #                   단 SAE 를 이 split 으로 학습하면 test scene 을 이미 본 상태(그건 "SAE 가
    #                   scene 을 인코딩하나" 질문에는 오히려 정상, "새 scene 일반화"에는 누수).
    #   split_scene   : scenario_seed 자체를 held-out → 새 scene 일반화용. test scene 클래스가
    #                   train 에 없으므로 scenario_seed 라벨 probe 는 여기서 평가 불가
    #                   (layout/success/phase 라벨 probe 용).
    # `split` = --split-by 로 고른 기본 축 (하위 도구는 이것만 봐도 동작).
    # 스캔 무결성 (리뷰 #9) — split 배정 **전에** 막는다 (오염된 scene 축은 조용히 무의미)
    validate_episodes(eps_info,
                      require_scene_meta=(args.split_by == "scene" or args.manifest is None))

    ep2split_ep = assign_splits(eps_info, args.seed)
    scene2split: dict[int, int] = {}
    try:
        scene2split = assign_scene_splits(eps_info, args.seed)
    except SystemExit:
        if args.split_by == "scene":
            raise
    ep2split_sc = ({e["episode_idx"]: scene2split[int(e["scenario_seed"])] for e in eps_info}
                   if scene2split else {})
    ep2split = ep2split_sc if args.split_by == "scene" else ep2split_ep

    meta = {f: np.concatenate(v) for f, v in meta_acc.items()}

    def _split_col(mapping: dict[int, int]) -> np.ndarray:
        col = np.zeros(len(meta["episode_idx"]), dtype=np.int8)
        for e_idx, s in mapping.items():
            col[meta["episode_idx"] == e_idx] = s
        return col

    split_row = _split_col(ep2split)
    meta["split"] = split_row
    meta["split_episode"] = _split_col(ep2split_ep)
    if ep2split_sc:
        meta["split_scene"] = _split_col(ep2split_sc)

    # scene split 이면 scene 누수 0 을 **여기서 강제** (핵심 계약 — 조용히 새면 probe 가 뻥튀기)
    scene_of_split: dict[int, set] = {0: set(), 1: set(), 2: set()}
    for e in eps_info:
        scene_of_split[ep2split[e["episode_idx"]]].add(int(e["scenario_seed"]))
    leaked = sorted((scene_of_split[0] | scene_of_split[1]) & scene_of_split[2])
    if args.split_by == "scene" and leaked:
        raise SystemExit(f"scene 누수 {len(leaked)}개 (train/val ∩ test): {leaked[:5]}")

    # split 별 layout coverage (리뷰 #3) — 한 split 이 단일 layout 에 몰리면 에러
    group_of_ep = {e["episode_idx"]: (int(e["layout_id"]), int(e["style_id"])) for e in eps_info}
    coverage = {"episode": assert_split_group_coverage(ep2split_ep, group_of_ep, "episode")}
    if scene2split:
        group_of_scene = {}
        for e in eps_info:
            group_of_scene.setdefault(int(e["scenario_seed"]),
                                      (int(e["layout_id"]), int(e["style_id"])))
        coverage["scene"] = assert_split_group_coverage(scene2split, group_of_scene, "scene")

    fingerprint = row_fingerprint(eps_info)              # 리뷰 #10
    prev_meta = out_dir / "meta.npz"
    if prev_meta.exists():                              # 같은 out-dir 이어쓰기 (layer 분할 실행)
        try:
            prev = np.load(prev_meta, allow_pickle=False)
            prev_fp = str(prev["row_fingerprint"]) if "row_fingerprint" in prev.files else None
        except Exception as exc:                        # noqa: BLE001
            raise SystemExit(f"기존 meta.npz 를 읽을 수 없음: {prev_meta} ({exc})")
        if prev_fp is None:
            print(f"[warn] 기존 meta.npz 에 row_fingerprint 없음 (구 빌드) — 대조 생략", flush=True)
        elif prev_fp != fingerprint:
            raise SystemExit(
                f"이어쓰기 행 구성 불일치 — 기존 meta.npz fingerprint={prev_fp} vs 이번 "
                f"{fingerprint}. 같은 --out-dir 에 다른 episode 집합/record 수를 섞으면 "
                f"X_L*.npz 행이 어긋난다. 새 --out-dir 를 쓰거나 기존 산출물을 지울 것.")

    split_json = {
        "seed": args.seed, "cell": cell, "split_by": args.split_by, "source": source,
        "ratio_target": ({"train": 0.70, "val": 0.10, "test": 0.20} if args.split_by == "scene"
                         else {"train": 0.667, "val": 0.133, "test": 0.20}),
        "scene_split": {str(sc): SPLIT_NAMES[s] for sc, s in sorted(scene2split.items())}
        if scene2split else
        {str(sc): sorted(SPLIT_NAMES[s] for s in {ep2split[e["episode_idx"]] for e in eps_info
                                                  if int(e["scenario_seed"]) == sc})
         for sc in sorted({int(e["scenario_seed"]) for e in eps_info})},
        "scene_leakage_test_vs_trainval": leaked,
        "row_fingerprint": fingerprint,
        "layout_coverage": coverage,
        "split_columns": ["split", "split_episode"] + (["split_scene"] if ep2split_sc else []),
        "episodes": [{**{k: v for k, v in e.items() if k != "pkl"},
                      "split": SPLIT_NAMES[ep2split[e["episode_idx"]]],
                      "split_episode": SPLIT_NAMES[ep2split_ep[e["episode_idx"]]],
                      **({"split_scene": SPLIT_NAMES[ep2split_sc[e["episode_idx"]]]}
                         if ep2split_sc else {}),
                      "pkl": e["pkl"]} for e in eps_info],
    }
    (out_dir / "split.json").write_text(json.dumps(split_json, indent=2, ensure_ascii=False))

    np.savez_compressed(
        out_dir / "meta.npz",
        phase_codebook=np.asarray(json.dumps(phase_codebook, ensure_ascii=False)),
        split_names=np.asarray(SPLIT_NAMES),
        segment_names=np.asarray([s[0] for s in SEGMENTS]),
        row_fingerprint=np.asarray(fingerprint),         # 리뷰 #10 (X_L*.npz 와 대조)
        split_by=np.asarray(args.split_by),
        **meta)

    # ---- layer 별 X + train-split 표준화 통계
    tr = meta["split"] == 0
    for L in want_layers:
        X = np.concatenate(X_acc[L], axis=0)
        X_acc[L] = []                                   # 즉시 해제 (메모리)
        if len(X) != len(split_row):
            raise SystemExit(f"L{L}: 행수 {len(X)} != meta {len(split_row)}")
        np.savez(out_dir / f"X_L{L}.npz", X=X, row_fingerprint=np.asarray(fingerprint))
        Xtr = X[tr].astype(np.float32)                  # 표준화 통계는 **train 행만** (누수 방지)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)               # 상수 feature 보호
        np.savez(out_dir / f"stats_L{L}.npz", mean=mu.astype(np.float32),
                 std=sd.astype(np.float32), n_train_rows=np.int64(int(tr.sum())))
        print(f"[build] saved X_L{L} shape={X.shape} dtype={X.dtype} "
              f"train_rows={int(tr.sum())}", flush=True)
        del X, Xtr

    # ---- 요약
    n_rows = len(split_row)
    print("\n===== 요약 =====")
    print(f"episodes={len(eps_info)}  records={sum(e['n_records'] for e in eps_info)}  rows={n_rows}")
    for s, name in enumerate(SPLIT_NAMES):
        sel = split_row == s
        eps_s = [e for e in eps_info if ep2split[e["episode_idx"]] == s]
        print(f"  {name:5s}: ep={len(eps_s):3d} (succ {sum(e['success'] for e in eps_s)}) "
              f"rows={int(sel.sum()):8d}")
    n_scenes = len({int(e["scenario_seed"]) for e in eps_info})
    n_inf = len({int(e["inference_seed"]) for e in eps_info})
    print(f"  scene(scenario_seed)={n_scenes}  inference_seed={n_inf}  "
          f"split_by={args.split_by} (기본 split 컬럼)")
    print(f"    scene train/val/test = {len(scene_of_split[0])}/{len(scene_of_split[1])}/"
          f"{len(scene_of_split[2])}  누수(test∩trainval)={len(leaked)}")
    print(f"    meta 컬럼: {split_json['split_columns']} "
          f"— scenario_seed 라벨 probe 는 split_episode 축에서만 평가 가능")
    print("  layout 분포 (ep 기준):")
    lay = {}
    for e in eps_info:
        k = (e["layout_id"], e["style_id"])
        lay.setdefault(k, [0, 0, 0])[ep2split[e["episode_idx"]]] += 1
    for k in sorted(lay, key=lambda x: (str(x[0]), str(x[1]))):
        print(f"    layout={k[0]} style={k[1]}: train/val/test = {lay[k]}")
    for axis, cov in coverage.items():
        print(f"  layout coverage ({axis} 축): " +
              "  ".join(f"{n}={len(cov[n])}그룹" for n in SPLIT_NAMES))
    print(f"  row_fingerprint = {fingerprint} (meta.npz · X_L*.npz 공통)")
    print(f"  success rows: succ={int((meta['success'] == 1).sum())} "
          f"fail={int((meta['success'] != 1).sum())}")
    print(f"  phases={len(phase_codebook)} {sorted(phase_codebook)}")
    print(f"  산출물: {len(want_layers)} × X_L*.npz + {len(want_layers)} × stats_L*.npz "
          f"+ meta.npz + split.json  → {out_dir}")
    print("  (완료 판정은 로그 문자열이 아니라 위 산출물 **개수**로 — 핸드아웃 §6-6)")


if __name__ == "__main__":
    sys.exit(main())
