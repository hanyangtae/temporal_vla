#!/usr/bin/env python
"""kanu 수집 219판 클린 영상 재생성 배치 러너 (**로컬 kanu 머신, robocasa 컨테이너 경유**).

`replay_clean_video.py` 를 에피소드당 **fresh 프로세스**(docker exec 1회)로 띄우고
워커 N개로 병렬 처리한다. 결과는 results.jsonl 에 append 되며, 이미 성공 기록이 있는
stem 은 건너뛴다(**resume**).

## 규약
- ep 당 fresh 프로세스: gym.make 를 한 프로세스에서 연속 생성하면 scene 이 오염되므로
  (`replay_clean_video.py` 상단 재현 규약) 워커는 항상 docker exec 를 새로 띄운다.
- GPU: EGL 렌더에만 쓰고, 완전히 빈 GPU 1개를 `--gpu` 로 지정 (MUJOCO_EGL_DEVICE_ID).
- CPU: 공유 64코어 머신 → 워커당 BLAS thread 1로 cap, 워커 수 기본 6.
- 충실도: `--eef-tol` 는 넉넉히 두어 mp4 는 남기고, **교체 자격은 eef_max_dev_m == 0
  (비트 일치)** 로 별도 판정한다 (push_clean_videos.sh). 발산 판은 삭제하지 않고 목록만 남긴다.

## 사용
    setsid nohup python scripts/analysis/grid_phase/replay_batch_runner.py \
      --bundle-dir <dir> --out-dir <dir> --gpu 0 --workers 6 \
      >> <log> 2>&1 &

완료 판정: 1패스 끝에 `KANU_REPLAY_PASS_DONE`. 219판 전체 완주 마커(`KANU_REPLAY_ALL_DONE`)는
이 러너를 반복 호출하는 `replay_orchestrate_kanu.sh` 가 찍는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

CONTAINER_REPO = os.environ.get("REPLAY_CONTAINER_REPO",
                                "/temporal_vla.claude-worktree-placeholder")
HOST_REPO = os.environ.get("REPLAY_HOST_REPO", "")
if not HOST_REPO:  # 기본: 이 파일이 속한 repo 루트 (main repo → /temporal_vla)
    HOST_REPO = str(Path(__file__).resolve().parents[3])
    CONTAINER_REPO = "/temporal_vla" if HOST_REPO.endswith("temporal_vla") \
        else CONTAINER_REPO

_lock = threading.Lock()


def _in_container(p: Path) -> str:
    """호스트 절대경로 → robocasa 컨테이너 경로 (repo 트리는 /temporal_vla 로 마운트)."""
    s = str(p)
    if s.startswith(HOST_REPO):
        return CONTAINER_REPO + s[len(HOST_REPO):]
    raise SystemExit(f"컨테이너에서 보이지 않는 경로: {s}")


def run_one(bundle: Path, out_dir: Path, gpu: int, eef_tol: float,
            container: str, timeout: int) -> dict:
    cmd = [
        "docker", "exec",
        "-e", "MUJOCO_GL=egl",
        # robocasa 컨테이너 EGL 은 default 1대만 열거(2026-08-20 실측 — NVML 복구와
        # 무관). v2 수집·exp6 replay 40/40 재현이 전부 이 default 디바이스에서 성립했으므로
        # 동일 경로 사용. --gpu 는 EGL 에 강제되지 않는다 (eef 게이트가 정합을 보증).
        "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        container, "python",
        f"{CONTAINER_REPO}/scripts/analysis/grid_phase/replay_clean_video.py",
        "--bundle", _in_container(bundle),
        "--out-dir", _in_container(out_dir),
        "--eef-tol", str(eef_tol),
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"stem": bundle.name.replace(".bundle.pkl", "").replace(".pkl", ""), "error": "timeout",
                "sec": round(time.time() - t0, 1)}
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("REPLAY_JSON ")), None)
    if line is None:
        return {"stem": bundle.name.replace(".bundle.pkl", "").replace(".pkl", ""),
                "error": f"rc={r.returncode}",
                "tail": (r.stdout[-800:] + "\n" + r.stderr[-800:]).strip(),
                "sec": round(time.time() - t0, 1)}
    info = json.loads(line[len("REPLAY_JSON "):])
    info["sec"] = round(time.time() - t0, 1)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eef-tol", type=float, default=0.01)
    ap.add_argument("--container", default="robocasa")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    bdir = Path(args.bundle_dir).resolve()      # 컨테이너 경로 변환에 절대경로 필요
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    # v2 번들(국내투고 추출)은 <machine>_<instr>_s<i>_n<j>.pkl 이름 — 두 규약 다 잡는다.
    bundles = sorted(set(bdir.glob("*.bundle.pkl")) | set(bdir.glob("kanu_*.pkl")))
    done: set[str] = set()
    if res_path.exists():
        for ln in res_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in d:                       # 실패분은 재시도
                done.add(d.get("stem", ""))
    todo = [b for b in bundles
            if b.name.replace(".bundle.pkl", "").replace(".pkl", "") not in done]
    total = len(bundles)
    print(f"[batch] 번들 {total}판 / 완료기록 {len(done)} / 이번 실행 {len(todo)}판 "
          f"workers={args.workers} gpu={args.gpu}", flush=True)

    n_ok = len(done)
    n_div = 0
    n_err = 0
    counter = {"i": 0}

    def task(b: Path) -> None:
        info = run_one(b, out_dir, args.gpu, args.eef_tol, args.container, args.timeout)
        with _lock:
            counter["i"] += 1
            with open(res_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
            nonlocal n_ok, n_div, n_err
            if "error" in info:
                n_err += 1
                tag = f"ERROR {info['error']}"
            elif info.get("diverged"):
                n_div += 1
                tag = "DIVERGED"
            else:
                n_ok += 1
                tag = "ok"
            print(f"[batch] {counter['i']}/{len(todo)} (누적 완료 {n_ok}/{total}) "
                  f"{info['stem']} {tag} eef_max={info.get('eef_max_dev_m')} "
                  f"frames={info.get('frames')}/{info.get('expected_frames')} "
                  f"{info.get('sec')}s", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(task, todo))

    errs = []
    if res_path.exists():
        seen: dict[str, dict] = {}
        for ln in res_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                seen[d.get("stem", "")] = d           # 마지막 시도 기준
        errs = [s for s, d in seen.items() if "error" in d or d.get("diverged")]
    print(f"[batch] 경과 {round(time.time() - t0, 1)}s  ok={n_ok} diverged={n_div} "
          f"error={n_err}", flush=True)
    if errs:
        print("[batch] 미교체 대상(발산/실패): " + ", ".join(sorted(errs)), flush=True)
    print("KANU_REPLAY_PASS_DONE", flush=True)   # 전체 완주 마커는 orchestrate 쪽(ALL_DONE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
