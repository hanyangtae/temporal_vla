#!/usr/bin/env python3
"""setM_seg 트리 → 세그먼트 게인 변형 트리 (기본: future-only).

왜 필요한가
-----------
`fit_setm.py:save_segment_npz` 의 NPZ 는 `alpha0_seg_mask` 를 **0/1 플래그가 아니라
세그먼트별 게인 승수**로 쓴다 (serve `SetpointSteering` 이 `β·mask·(h·r̂−s)·r̂`).
future-only arm 은 state·action 게인을 0 으로 죽인 같은 연산자다 — 재fit 이 아니라
mask 만 바꾸면 되고, 그래야 treatment 와 방향(r̂)·setpoint(s_t)가 **bitwise 동일**해서
"어느 토큰 세그먼트에 걸었나"만 달라진 대조가 된다.

위약(setM_pl)의 mask 는 dose-match 스케일(예 [1.95, 2.0, 1.18])이라 0/1 로 덮으면
dose 매칭이 깨진다 → **기존 mask 에 선택 벡터를 곱한다** ([1.95,2.0,1.18] → [0,2.0,0]).

사용
----
    python scripts/steer/online_gated/make_seg_mask_variant.py \
        --src outputs/steer/online_pipe/OvenRack_out/setM_seg \
        --out outputs/steer/online_pipe/OvenRack_out/setM_seg_fut
    # 위약도 같은 변형
    ... --src .../setM_seg_pl --out .../setM_seg_fut_pl

    python scripts/steer/online_gated/make_seg_mask_variant.py --self-test

계약 (docs/04 §1): 변형도 연산자다 → 출력 디렉토리마다 `config.json` 을 남긴다.
입력 sig 는 원본 config.json 의 `train_episode_sigs` 를 승계한다 (같은 rollout 에서
나온 연산자이므로). params 에 `seg_mask`·`variant`·`variant_src_opsig` 를 기록해
armsig 가 원본과 갈리게 한다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# fit_setm.SEGMENTS 와 같은 규약 (N1.5 DiT T=49: state 1 + future 32 + action 16).
# 이름 정보가 metadata.json 에 없을 때의 폴백 판정에만 쓴다.
DEFAULT_SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))
REQUIRED_KEYS = ("alpha0_v_seg", "alpha0_s_tok", "alpha0_seg_bounds", "alpha0_seg_mask")


def segment_names(npz_dir: Path, bounds: np.ndarray) -> list[str]:
    """세그먼트 이름 목록. metadata.json 의 `segments` 가 정본, 없으면 경계로 폴백."""
    meta_path = npz_dir / "metadata.json"
    if meta_path.exists():
        try:
            names = json.loads(meta_path.read_text(encoding="utf-8")).get("segments")
        except json.JSONDecodeError as exc:  # 손상된 메타는 조용히 넘기지 않는다
            raise SystemExit(f"{meta_path}: metadata.json 파싱 실패 ({exc})") from exc
        if names:
            if len(names) != len(bounds):
                raise SystemExit(
                    f"{meta_path}: segments {len(names)}개 != seg_bounds {len(bounds)}개")
            return [str(n) for n in names]
    want = [[lo, hi] for _n, lo, hi in DEFAULT_SEGMENTS]
    if [list(map(int, b)) for b in bounds] == want:
        return [n for n, _lo, _hi in DEFAULT_SEGMENTS]
    raise SystemExit(
        f"{npz_dir}: 세그먼트 이름을 알 수 없다 — metadata.json 의 'segments' 가 없고 "
        f"seg_bounds {bounds.tolist()} 가 기본 규약 {want} 과도 다르다 "
        "(추측해서 게인을 0 으로 죽이면 조용히 엉뚱한 arm 이 된다)")


def selector(names: list[str], keep: str) -> np.ndarray:
    if keep not in names:
        raise SystemExit(f"세그먼트 {keep!r} 없음 (있는 것: {names})")
    return np.asarray([1.0 if n == keep else 0.0 for n in names], dtype=np.float32)


def convert_one(src_dir: Path, dst_dir: Path, keep: str, allow_missing_config: bool) -> dict:
    """dit_L* 디렉토리 하나 변환. 반환 = 요약 dict."""
    with np.load(src_dir / "conceptors.npz") as z:
        arrays = {k: z[k] for k in z.files}
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise SystemExit(
            f"{src_dir}: setpoint_seg NPZ 가 아니다 — 없는 키 {missing} "
            "(--steering-op setpoint_seg 계약: alpha0_v_seg/s_tok/seg_bounds/seg_mask)")
    bounds = np.asarray(arrays["alpha0_seg_bounds"])
    names = segment_names(src_dir, bounds)
    old_mask = np.asarray(arrays["alpha0_seg_mask"], dtype=np.float32).reshape(-1)
    if len(old_mask) != len(names):
        raise SystemExit(f"{src_dir}: seg_mask {len(old_mask)} != segments {len(names)}")
    new_mask = (old_mask * selector(names, keep)).astype(np.float32)
    if not np.any(new_mask > 0):
        raise SystemExit(
            f"{src_dir}: 변형 결과 mask 가 전부 0 — 원본 {keep} 게인이 0 이었다 (무개입 arm)")
    arrays["alpha0_seg_mask"] = new_mask

    dst_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_dir / "conceptors.npz", **arrays)

    # metadata.json 승계 + 변형 표시
    meta = {}
    if (src_dir / "metadata.json").exists():
        meta = json.loads((src_dir / "metadata.json").read_text(encoding="utf-8"))
    meta.update({
        "seg_mask": [float(x) for x in new_mask],
        "seg_mask_src": [float(x) for x in old_mask],
        "variant": f"{keep}_only",
        "variant_tool": "scripts/steer/online_gated/make_seg_mask_variant.py",
    })
    (dst_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # config.json — docs/04 §1: 출처(input sig) 는 연산자 디렉토리 안에 있어야 한다.
    src_cfg_path = src_dir / "config.json"
    if not src_cfg_path.exists():
        if not allow_missing_config:
            raise SystemExit(
                f"{src_dir}: config.json 없음 — 출처 없는 연산자는 변형하지 않는다 "
                "(docs/04 §1). 정말 필요하면 --allow-missing-config")
    else:
        from src.utils.operator_config import write_operator_config  # noqa: PLC0415

        src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
        sigs = src_cfg.get("train_episode_sigs") or []
        params = {
            k: v for k, v in src_cfg.items()
            if k not in ("op_type", "train_episode_sigs", "n_train_episodes",
                         "train_episode_fingerprint")
        }
        params.update({
            "seg_mask": [float(x) for x in new_mask],
            "variant": f"{keep}_only",
            "variant_src_fingerprint": src_cfg.get("train_episode_fingerprint"),
        })
        write_operator_config(dst_dir, op_type=src_cfg.get("op_type", "setm"),
                              input_sigs=sigs, params=params)
    return {"dir": dst_dir.name,
            "mask": [float(x) for x in new_mask], "src_mask": [float(x) for x in old_mask]}


def convert_tree(src: Path, out: Path, keep: str, allow_missing_config: bool) -> int:
    if not src.is_dir():
        raise SystemExit(f"src 없음: {src}")
    npz_dirs = sorted(p.parent for p in src.rglob("conceptors.npz"))
    if not npz_dirs:
        raise SystemExit(f"{src}: conceptors.npz 가 하나도 없다")
    n = 0
    for d in npz_dirs:
        rel = d.relative_to(src)
        info = convert_one(d, out / rel, keep, allow_missing_config)
        print(f"[variant] {rel}  mask {info['src_mask']} -> {info['mask']}", flush=True)
        n += 1
    # phase 디렉토리 옆의 부가 파일(fit 요약 등)은 옮기지 않는다 — 연산자 자체만 복제.
    print(f"[variant] {n} 개 연산자 → {out} (keep={keep})", flush=True)
    return n


# ---------------------------------------------------------------- self-test
def _self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "setM_seg_pl"
        T, D, S = 49, 8, 3
        rng = np.random.default_rng(0)
        for phase in ("reach", "grasp"):
            d = src / phase / "dit_L12"
            d.mkdir(parents=True)
            np.savez_compressed(
                d / "conceptors.npz",
                alpha0_v_seg=rng.normal(size=(S, D)).astype(np.float32),
                alpha0_s_tok=rng.normal(size=(T,)).astype(np.float32),
                alpha0_seg_bounds=np.asarray([[0, 1], [1, 33], [33, 49]], dtype=np.int32),
                alpha0_seg_mask=np.asarray([1.95, 2.0, 1.18], dtype=np.float32),
            )
            (d / "metadata.json").write_text(json.dumps(
                {"op": "setpoint_seg", "segments": ["state", "future", "action"],
                 "seg_mask": [1.95, 2.0, 1.18]}))
            (d / "config.json").write_text(json.dumps(
                {"op_type": "setm", "op": "setpoint_seg", "beta": 1.0,
                 "train_episode_sigs": ["a" * 16, "b" * 16],
                 "n_train_episodes": 2, "train_episode_fingerprint": "deadbeef1234"}))
        out = root / "setM_seg_fut_pl"
        n = convert_tree(src, out, "future", allow_missing_config=False)
        assert n == 2, n
        for phase in ("reach", "grasp"):
            sd, dd = src / phase / "dit_L12", out / phase / "dit_L12"
            with np.load(sd / "conceptors.npz") as a, np.load(dd / "conceptors.npz") as b:
                assert sorted(a.files) == sorted(b.files), (a.files, b.files)
                np.testing.assert_allclose(b["alpha0_seg_mask"], [0.0, 2.0, 0.0])
                for k in ("alpha0_v_seg", "alpha0_s_tok", "alpha0_seg_bounds"):
                    np.testing.assert_array_equal(a[k], b[k])  # 방향·setpoint 는 불변
            cfg = json.loads((dd / "config.json").read_text())
            assert cfg["variant"] == "future_only", cfg
            assert cfg["train_episode_sigs"] == ["a" * 16, "b" * 16], cfg
            assert cfg["seg_mask"] == [0.0, 2.0, 0.0], cfg
        # 처치(mask=ones) → [0,1,0]
        tsrc = root / "setM_seg"
        shutil.copytree(src, tsrc)
        for p in tsrc.rglob("conceptors.npz"):
            with np.load(p) as z:
                arr = {k: z[k] for k in z.files}
            arr["alpha0_seg_mask"] = np.ones(S, dtype=np.float32)
            np.savez_compressed(p, **arr)
        convert_tree(tsrc, root / "setM_seg_fut", "future", allow_missing_config=False)
        with np.load(root / "setM_seg_fut" / "reach" / "dit_L12" / "conceptors.npz") as z:
            np.testing.assert_allclose(z["alpha0_seg_mask"], [0.0, 1.0, 0.0])
    print("[self-test] OK — future-only 변형: 게인만 곱해지고 v_seg/s_tok 불변")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="원본 트리 (setM_seg | setM_seg_pl)")
    ap.add_argument("--out", type=Path, help="출력 트리 (setM_seg_fut | setM_seg_fut_pl)")
    ap.add_argument("--segment", default="future",
                    help="살릴 세그먼트 이름 (기본 future — state/action 게인을 0 으로)")
    ap.add_argument("--allow-missing-config", action="store_true",
                    help="원본에 config.json 이 없어도 진행 (출처 끊김 — 비권장)")
    ap.add_argument("--self-test", action="store_true", help="합성 NPZ 로 변형 로직 검증")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.src is None or args.out is None:
        ap.error("--src 와 --out 필요 (또는 --self-test)")
    if args.out.resolve() == args.src.resolve():
        ap.error("--out 이 --src 와 같다 (원본 덮어쓰기 금지)")
    convert_tree(args.src, args.out, args.segment, args.allow_missing_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
