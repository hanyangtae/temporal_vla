"""RoboCasa 데이터셋 평가 유틸리티.

에피소드 평가 로직, 병렬화 인프라, 결과 출력 함수를 제공한다.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import signal
import sys
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from path_setup import configure_repo_paths

configure_repo_paths(include_robocasa=True)

import numpy as np
import robocasa.utils.lerobot_utils as LU
import robosuite
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from tqdm import tqdm

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger(__name__)


# ── 메모리 안전장치 ────────────────────────────────────────────────────────

def safe_worker_count(requested: int, env_mem_gb: float = 1.5, reserve_gb: float = 4.0) -> int:
    """가용 메모리 기준으로 안전한 워커 수를 반환. psutil 없으면 요청값 그대로."""
    if not _HAS_PSUTIL:
        return requested
    available_gb = psutil.virtual_memory().available / 1024 ** 3
    max_by_mem = max(1, int((available_gb - reserve_gb) / env_mem_gb))
    if requested > max_by_mem:
        logger.warning(
            "[메모리 안전장치] 가용 %.1fGB → 워커 %d→%d으로 제한 "
            "(env당 %.1fGB, 예약 %.1fGB). --reserve-gb/--env-mem-gb 로 조정 가능.",
            available_gb, requested, max_by_mem, env_mem_gb, reserve_gb,
        )
        return max_by_mem
    return requested


def log_memory_info(workers: int):
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    logger.info(
        "[메모리] 전체 %.1fGB | 가용 %.1fGB | 사용률 %.0f%% | 워커: %d",
        mem.total / 1024 ** 3, mem.available / 1024 ** 3, mem.percent, workers,
    )


# ── 워커 초기화 ────────────────────────────────────────────────────────────

_worker_env = None


def _sigint_ignore():
    """워커 프로세스에서 SIGINT 무시 + faulthandler 활성화."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    faulthandler.enable(file=sys.stderr)


def _init_episode_worker(env_kwargs: dict):
    """Pool initializer: 워커 프로세스당 env를 한 번만 생성."""
    _sigint_ignore()
    global _worker_env
    _worker_env = robosuite.make(**env_kwargs)


# ── 에피소드 평가 ──────────────────────────────────────────────────────────

def _check_success(env) -> bool:
    result = env._check_success()
    return bool(result.get("task", False)) if isinstance(result, dict) else bool(result)


def _load_initial_state(dataset: Path, ep_num: int) -> dict:
    return {
        "states": LU.get_episode_states(dataset, ep_num)[0],
        "model": LU.get_episode_model_xml(dataset, ep_num),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset, ep_num)),
    }


def eval_episode_from_final_state(env, dataset: Path, ep_num: int) -> bool:
    """최종 상태만 로드해 성공 여부 체크 (빠른 모드)."""
    states = LU.get_episode_states(dataset, ep_num)
    reset_to(env, _load_initial_state(dataset, ep_num))
    reset_to(env, {"states": states[-1]})
    return _check_success(env)


def eval_episode_with_actions(env, dataset: Path, ep_num: int) -> dict:
    """open-loop 액션 재생으로 평가 (궤적 발산 통계 포함)."""
    states = LU.get_episode_states(dataset, ep_num)
    actions = LU.get_episode_actions(dataset, ep_num)
    reset_to(env, _load_initial_state(dataset, ep_num))

    traj_len = states.shape[0]
    first_success_step = None
    divergences = []

    for t in range(traj_len):
        env.step(actions[t])
        if t < traj_len - 1:
            playback = np.array(env.sim.get_state().flatten())
            if not np.all(np.equal(states[t + 1], playback)):
                divergences.append(float(np.linalg.norm(states[t + 1] - playback)))
        if _check_success(env) and first_success_step is None:
            first_success_step = t

    return {
        "success": _check_success(env),
        "first_success_step": first_success_step,
        "traj_len": traj_len,
        "mean_divergence": float(np.mean(divergences)) if divergences else 0.0,
        "max_divergence": float(np.max(divergences)) if divergences else 0.0,
    }


# ── 병렬화 워커 함수 (모듈 최상위에 있어야 pickle 가능) ───────────────────

def _eval_episode_worker(args: tuple) -> dict:
    ep_num, dataset_path_str, use_actions = args
    dataset_path = Path(dataset_path_str)
    try:
        if use_actions:
            r = eval_episode_with_actions(_worker_env, dataset_path, ep_num)
        else:
            r = {"success": eval_episode_from_final_state(_worker_env, dataset_path, ep_num)}
        r["ep_num"] = ep_num
        return r
    except Exception as e:
        return {"ep_num": ep_num, "success": False, "error": str(e)}


def _evaluate_dataset_worker(args: tuple) -> dict:
    dataset_path_str, n, use_actions = args
    faulthandler.enable(file=sys.stderr)
    try:
        return evaluate_dataset(Path(dataset_path_str), n=n, use_actions=use_actions, silent=True)
    except Exception:
        # 워커 내에서 잡히는 예외는 로그로 남기고 re-raise
        traceback.print_exc(file=sys.stderr)
        raise


# ── 단일 데이터셋 평가 ────────────────────────────────────────────────────

def _make_env_kwargs(dataset_path: Path) -> dict:
    meta = LU.get_env_metadata(dataset_path)
    kwargs = meta["env_kwargs"]
    kwargs["env_name"] = meta["env_name"]
    kwargs.update(
        has_renderer=False,
        renderer="mjviewer",
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    return kwargs


def _eval_sequential(env_kwargs, dataset_path, total, use_actions, silent) -> list[dict]:
    env = robosuite.make(**env_kwargs)
    results = []
    for ep_num in tqdm(range(total), desc="에피소드 평가", disable=silent):
        try:
            if use_actions:
                r = eval_episode_with_actions(env, dataset_path, ep_num)
            else:
                r = {"success": eval_episode_from_final_state(env, dataset_path, ep_num)}
            r["ep_num"] = ep_num
        except Exception as e:
            if not silent:
                logger.error("ep %d 실패: %s", ep_num, e)
            r = {"ep_num": ep_num, "success": False, "error": str(e)}
        results.append(r)
    env.close()
    return results


def _eval_parallel(env_kwargs, dataset_path, total, use_actions, silent, workers) -> list[dict]:
    args_list = [(ep_num, str(dataset_path), use_actions) for ep_num in range(total)]
    results = []
    _spawn = multiprocessing.get_context("spawn")
    pool = _spawn.Pool(
        processes=workers,
        initializer=_init_episode_worker,
        initargs=(env_kwargs,),
    )
    try:
        it = pool.imap_unordered(_eval_episode_worker, args_list)
        for r in tqdm(it, total=total, desc="에피소드 병렬 평가", disable=silent):
            if "error" in r and not silent:
                logger.error("ep %d 실패: %s", r["ep_num"], r["error"])
            results.append(r)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    results.sort(key=lambda r: r["ep_num"])
    return results


def evaluate_dataset(
    dataset_path: Path,
    n: int = None,
    use_actions: bool = False,
    workers: int = 1,
    silent: bool = False,
) -> dict:
    """데이터셋 전체에 대한 성공률 평가."""
    env_kwargs = _make_env_kwargs(dataset_path)
    if not silent:
        logger.info("태스크: %s", env_kwargs["env_name"])

    episodes = LU.get_episodes(dataset_path)
    if n is not None:
        episodes = episodes[:n]
    total = len(episodes)

    if workers > 1:
        results = _eval_parallel(env_kwargs, dataset_path, total, use_actions, silent, workers)
    else:
        results = _eval_sequential(env_kwargs, dataset_path, total, use_actions, silent)

    n_success = sum(r["success"] for r in results)
    summary = {
        "dataset": str(dataset_path),
        "task": env_kwargs["env_name"],
        "mode": "actions" if use_actions else "states",
        "total_episodes": total,
        "n_success": n_success,
        "success_rate": n_success / total if total > 0 else 0.0,
        "episodes": results,
    }
    if use_actions:
        divs = [r["mean_divergence"] for r in results if "mean_divergence" in r]
        summary["mean_divergence"] = float(np.mean(divs)) if divs else 0.0
    return summary


# ── 전체 데이터셋 탐색 + 일괄 평가 ────────────────────────────────────────

def find_all_datasets(base_path: Path, task_type: str = None) -> list[Path]:
    """base_path/{atomic|composite}/*/*/lerobot 패턴으로 유효한 데이터셋 탐색."""
    patterns = (
        [f"{task_type}/*/*/lerobot"] if task_type
        else ["atomic/*/*/lerobot", "composite/*/*/lerobot"]
    )
    datasets = []
    for pattern in patterns:
        datasets.extend(
            p for p in sorted(base_path.glob(pattern))
            if (p / "extras" / "dataset_meta.json").exists()
        )
    return datasets


def run_all_datasets(
    datasets: list[Path],
    n: int = None,
    use_actions: bool = False,
    workers: int = 1,
    task_callback=None,
) -> tuple[list[dict], list[dict]]:
    """여러 데이터셋 평가. (summaries, failed) 반환.

    task_callback: 태스크 하나가 완료될 때마다 호출되는 콜백.
                   signature: callback(summary: dict, completed: list[dict])
    """
    if workers > 1:
        all_summaries, failed = _run_parallel_datasets(datasets, n, use_actions, workers, task_callback)
    else:
        all_summaries, failed = _run_sequential_datasets(datasets, n, use_actions, task_callback)

    ds_order = {str(p): i for i, p in enumerate(datasets)}
    all_summaries.sort(key=lambda s: ds_order.get(s["dataset"], 9999))
    return all_summaries, failed


def _run_parallel_datasets(datasets, n, use_actions, workers, task_callback=None) -> tuple[list, list]:
    all_summaries, failed = [], []
    futures_map = {}
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_sigint_ignore,
        mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        for ds_path in datasets:
            f = executor.submit(_evaluate_dataset_worker, (str(ds_path), n, use_actions))
            futures_map[f] = ds_path

        for i, future in enumerate(as_completed(futures_map), 1):
            ds_path = futures_map[future]
            task_name = ds_path.parts[-3]
            try:
                summary = future.result()
                logger.info("[%d/%d] %s 완료", i, len(datasets), task_name)
                log_summary(summary)
                all_summaries.append(summary)
                if task_callback:
                    task_callback(summary, list(all_summaries))
            except Exception:
                tb = traceback.format_exc()
                logger.error("[%d/%d] %s 실패:\n%s", i, len(datasets), task_name, tb)
                failed.append({"dataset": str(ds_path), "error": tb})

    except KeyboardInterrupt:
        logger.warning("평가 중단됨. 워커 종료 중...")
        executor.shutdown(wait=False, cancel_futures=True)
        if _HAS_PSUTIL:
            for child in psutil.Process(os.getpid()).children(recursive=True):
                child.kill()
        logger.info("종료 완료.")
    else:
        executor.shutdown(wait=True)
    return all_summaries, failed


def _run_sequential_datasets(datasets, n, use_actions, task_callback=None) -> tuple[list, list]:
    all_summaries, failed = [], []
    for i, ds_path in enumerate(datasets):
        task_name = ds_path.parts[-3]
        logger.info("[%d/%d] %s", i + 1, len(datasets), task_name)
        try:
            summary = evaluate_dataset(ds_path, n=n, use_actions=use_actions)
            log_summary(summary)
            all_summaries.append(summary)
            if task_callback:
                task_callback(summary, list(all_summaries))
        except KeyboardInterrupt:
            logger.warning("평가 중단됨.")
            break
        except Exception:
            tb = traceback.format_exc()
            logger.error("실패:\n%s", tb)
            failed.append({"dataset": str(ds_path), "error": tb})
    return all_summaries, failed


# ── 결과 출력 ──────────────────────────────────────────────────────────────

def log_summary(summary: dict, show_failures: bool = True):
    rate = summary["success_rate"]
    lines = [
        "=" * 50,
        f"태스크:       {summary['task']}",
        f"데이터셋:     {summary['dataset']}",
        f"평가 모드:    {summary['mode']}",
        f"총 에피소드:  {summary['total_episodes']}",
        f"성공 에피소드: {summary['n_success']} / {summary['total_episodes']}",
        f"성공률:       {rate:.1%}",
    ]
    if "mean_divergence" in summary:
        lines.append(f"평균 궤적 발산: {summary['mean_divergence']:.6f}")

    if show_failures:
        failed_eps = [r["ep_num"] for r in summary.get("episodes", []) if not r["success"]]
        if failed_eps:
            lines.append(f"실패 에피소드: {failed_eps}")

    lines.append("=" * 50)

    msg = "\n".join(lines)
    if rate > 0.5:
        logger.info(msg)
    else:
        logger.warning(msg)


def log_all_summary(summaries: list[dict]):
    total_eps = sum(s["total_episodes"] for s in summaries)
    total_ok = sum(s["n_success"] for s in summaries)
    overall = total_ok / total_eps if total_eps > 0 else 0.0

    lines = [
        "=" * 70,
        "  전체 평가 결과 요약",
        "=" * 70,
        f"{'태스크':<45} {'성공/전체':>12} {'성공률':>8}",
        "-" * 70,
    ]
    for task_type in ["atomic", "composite"]:
        typed = [s for s in summaries if task_type in s["dataset"]]
        if not typed:
            continue
        lines.append(f"  [{task_type.upper()}]")
        for s in typed:
            lines.append(
                f"  {s['task']:<43} {s['n_success']}/{s['total_episodes']:>8} {s['success_rate']:>7.1%}"
            )
    lines += [
        "-" * 70,
        f"  {'전체 합계':<43} {total_ok}/{total_eps:>8} {overall:>7.1%}",
        "=" * 70,
    ]

    msg = "\n".join(lines)
    if overall >= 0.5:
        logger.info(msg)
    else:
        logger.warning(msg)


# 하위 호환을 위한 alias
print_summary = log_summary
print_all_summary = log_all_summary
