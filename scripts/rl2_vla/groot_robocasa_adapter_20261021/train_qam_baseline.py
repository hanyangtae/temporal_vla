"""OpenDrawer prefix-critic QAM baseline 학습 (host ``.venvs/rl2_iid``, JAX). 정본 규약: CONTRACT.md §11.

입력 = ``extract_opendrawer_latents.py`` 가 만든 추출 폴더 (검증 loader: baseline_data.load_extraction).
transition = episode 시간순 (γ=0.99, H16 예측 / 5 실행, 마지막 transition terminal discount 0).

- optimizer 는 train 92 episode transition 만 본다. val 10 episode 는 평가 전용 (gradient 없음).
- 사전 선언 val metric: ``val/action_mse_exec_prefix`` (전체 val set, 고정 val seed). 낮을수록 좋음,
  동률이면 이른 step. step 0 / eval_every 배수 / 마지막 step 에서 평가·checkpoint 저장.
- checkpoint 는 step 별 새 폴더 (tmp → rename, 기존 폴더·tmp 폴더 있으면 거부, 삭제 없음). sha256·bytes 를
  기록한다. 중간 checkpoint 는 전부 보존한다.
- disk preflight: 첫 checkpoint 크기를 잰 뒤 (남은 checkpoint 수 × 최대 크기) 를 쓰고도 6GiB 이상 남는지
  매 저장 후 확인한다. 모자라면 아무것도 지우지 않고 ``FAILED.json`` (DiskBudget) 로 멈춘다.
- run_manifest 의 경로 인자는 이식 가능한 참조 (root alias + 상대경로 + sha) 로만 기록한다 (docs/04 §6-1).
- selection receipt 는 평가마다 새 파일 (O_EXCL). 최종 receipt 는 ``selection_final.json``.
- progress.jsonl 을 log_every 마다 flush+fsync (메인 모니터용). 비유한 loss/metric 이면 즉시 멈추고
  ``FAILED.json`` 을 남긴다.

실행 (repo root, host):
  CUDA_VISIBLE_DEVICES=<gpu> .venvs/rl2_iid/bin/python \
    scripts/rl2_vla/groot_robocasa_adapter_20261021/train_qam_baseline.py \
    --extraction outputs/rl2_vla/<extraction_dir> --out outputs/rl2_vla/<new_run_dir>
smoke (짧게):
  ... --steps 20 --eval-every 10 --log-every 5 --profile smoke_reduced_network --batch-size 8 [--allow-subset-smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parents[2]
for _p in (ADAPTER_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline_data import (  # noqa: E402
    EXECUTE_STEPS,
    GAMMA,
    canonical_json,
    sha256_bytes,
    sha256_path,
    write_json_exclusive,
)
from contract import ContractError  # noqa: E402
from data_contract import AUX_SPLIT_V2_REL, OPENDRAWER_PACKAGE_REL  # noqa: E402

VAL_METRIC = "val/action_mse_exec_prefix"
TRAIN_VERSION = "groot_opendrawer_prefix_qam_baseline_rl2_reward/v3"
CODE_FILES = ("train_qam_baseline.py", "baseline_data.py", "qam_prefix.py", "transitions.py",
              "data_contract.py", "contract.py", "composition.py", "rl2_expert_rewards.py")


MIN_FREE_BYTES = 6 * 1024 ** 3   # checkpoint 를 다 쓴 뒤에도 남길 여유 (6 GiB)


class NonFinite(ContractError):
    pass


class DiskBudget(ContractError):
    pass


def _disk_free(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def disk_preflight(ckpt_root: Path, *, ckpt_bytes: int, remaining: int,
                   min_free: int = MIN_FREE_BYTES) -> dict:
    """남은 checkpoint (remaining 개 × ckpt_bytes) 를 쓴 뒤에도 min_free 이상 남는지. 아니면 DiskBudget."""
    free = _disk_free(ckpt_root)
    need = int(ckpt_bytes) * int(remaining)
    rec = {"free_bytes": free, "remaining_checkpoints": int(remaining), "checkpoint_bytes": int(ckpt_bytes),
           "projected_remaining_bytes": need, "min_free_bytes": int(min_free),
           "free_after_projection_bytes": free - need}
    if free - need < min_free:
        raise DiskBudget(f"disk 부족: free {free/1e9:.2f}GB − 남은 checkpoint {remaining}×"
                         f"{ckpt_bytes/1e9:.3f}GB < 여유 {min_free/1e9:.2f}GB (삭제 없이 중단)")
    return rec


# ---------------- 이식 가능한 경로 참조 (산출물에 절대경로 금지, docs/04 §6-1) ----------------

def path_ref(path: Path | str | None, roots: dict[str, Path], *, hash_file: bool = False) -> dict | None:
    """경로 → {root alias, 상대경로} (+파일 sha256). 어느 root 아래에도 없으면 basename 만 남긴다."""
    if path is None:
        return None
    p = Path(path).resolve()
    ref: dict = {"root": "unresolved", "name": p.name}
    for alias, root in roots.items():
        try:
            ref = {"root": alias, "rel": p.relative_to(Path(root).resolve()).as_posix()}
            break
        except ValueError:
            continue
    if hash_file and p.is_file():
        ref["sha256"] = sha256_path(p)
    return ref


def portable_args(a: argparse.Namespace, roots: dict[str, Path]) -> dict:
    """CLI 인자 기록본. 경로 인자는 path_ref 로 바꾸고, 그 밖의 값은 그대로 둔다."""
    out = {}
    for k, v in vars(a).items():
        if k == "data_root":
            from scripts.path_setup import DATA_ROOT

            out[k] = {"root": "datasets_root", "is_path_setup_default":
                      Path(v).resolve() == Path(DATA_ROOT).resolve()}
        elif isinstance(v, Path):
            out[k] = path_ref(v, roots, hash_file=(k == "split"))
        else:
            out[k] = v
    return out


def scrub_paths(text: str, roots: dict[str, Path]) -> str:
    """오류 메시지 안의 root 절대경로를 alias 로 바꾼다 (FAILED.json·progress 기록용). 긴 root 먼저."""
    for alias, root in sorted(roots.items(), key=lambda kv: -len(str(kv[1]))):
        for form in {str(root), str(Path(root).resolve())}:
            text = text.replace(form, f"<{alias}>")
    return text


def eval_steps(total: int, every: int) -> list[int]:
    """평가·저장 step: 0, every, 2·every, ..., total (total 포함, 중복 없음)."""
    if total < 1 or every < 1:
        raise ContractError(f"steps {total} / eval_every {every} 는 1 이상")
    return sorted({0, total, *range(every, total + 1, every)})


def select_best(evals: list[dict], metric: str = VAL_METRIC) -> dict:
    """val metric 최소 (동률 → 이른 step). 비유한 값이 있으면 거부. train 지표는 보지 않는다."""
    if not evals:
        raise ContractError("평가 기록 없음")
    for e in evals:
        if not np.isfinite(e["metrics"][metric]):
            raise NonFinite(f"step {e['step']}: {metric} 비유한")
    return min(evals, key=lambda e: (e["metrics"][metric], e["step"]))


def _fsync_append(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def save_checkpoint(trainer, ckpt_root: Path, *, feature_signature: str) -> dict:
    """step 별 새 폴더에 저장 (tmp 폴더 → rename). 대상이 있으면 거부 (덮어쓰기 없음)."""
    step = trainer.steps_done
    final = ckpt_root / f"step_{step:07d}"
    if final.exists():
        raise ContractError(f"{final} 이미 존재 — checkpoint 덮어쓰기 금지")
    tmp = ckpt_root / f".tmp_step_{step:07d}_{os.getpid()}"
    if tmp.exists():   # 이전 실패의 잔재일 수 있다 — 지우지 않고 거부 (메인이 확인)
        raise ContractError(f"예상하지 않은 tmp checkpoint 폴더 {tmp.name} 존재 — 삭제하지 않고 중단")
    tmp.mkdir()
    trainer.save(tmp, feature_signature=feature_signature)
    files = {p.name: {"sha256": sha256_path(p), "bytes": p.stat().st_size} for p in sorted(tmp.iterdir())}
    os.rename(tmp, final)  # 같은 파일시스템, final 미존재 확인 후
    return {"step": step, "dir": final.name, "files": files,
            "bytes": int(sum(v["bytes"] for v in files.values()))}


def build_manifest_for_purpose(split: Path, data_root, feature_signature: str, out: Path) -> dict:
    """baseline purpose manifest 를 조립·저장한 뒤 fail-closed loader 로 다시 검증한다."""
    from composition import FEATURE_CAPTURE_POINT
    from data_contract import load_finetune_data_manifest, package_smoke_manifest

    fe = {"status": "confirmed", "feature_signature": feature_signature,
          "capture_point": FEATURE_CAPTURE_POINT, "denoise_step": 0}
    m = package_smoke_manifest(split, feature_extraction=fe, data_root=data_root,
                               mode="published_package_subset_for_baseline")
    p = out / "data_manifest.json"
    write_json_exclusive(p, m)
    loaded = load_finetune_data_manifest(p, feature_signature=feature_signature,
                                         purpose="published_package_subset_for_baseline")
    return loaded["membership_claim"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OpenDrawer prefix-critic QAM baseline 학습 (validation-only 선택)")
    ap.add_argument("--extraction", required=True, type=Path, help="extract_opendrawer_latents.py 출력 폴더")
    ap.add_argument("--out", required=True, type=Path, help="새 run 폴더 (비어 있지 않으면 거부)")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / AUX_SPLIT_V2_REL)
    ap.add_argument("--data-root", type=Path, default=None, help="dataset root (기본 path_setup.DATA_ROOT)")
    ap.add_argument("--profile", default="production", choices=("production", "smoke_reduced_network"))
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=50_000, help="평가·checkpoint 간격 (step 0·마지막 포함)")
    ap.add_argument("--log-every", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=42, help="agent init + train sampler seed")
    ap.add_argument("--val-seed", type=int, default=42, help="val metric 고정 seed (평가마다 동일)")
    ap.add_argument("--allow-subset-smoke", action="store_true",
                    help="102 episode 미만 추출로 smoke (production 선택 receipt 아님으로 표시)")
    ap.add_argument("--skip-source-verify", action="store_true",
                    help="parquet 원본 대조 생략 (dataset 없는 host 용; receipt 에 기록)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"--out {a.out} 이 비어 있지 않음 (덮어쓰기 금지)")
    eval_at = eval_steps(a.steps, a.eval_every)
    if a.log_every < 1:
        raise SystemExit("--log-every >= 1")
    if a.data_root is None:
        from scripts.path_setup import DATA_ROOT

        a.data_root = DATA_ROOT

    import jax

    from baseline_data import load_extraction
    from composition import SPEC
    from contract import load_checkpoint_contract, raw_to_model_index, resolve_cached_checkpoint_dir
    from qam_prefix import PrefixQAMTrainer
    from transitions import TrainSampler, build_split

    a.out.mkdir(parents=True, exist_ok=True)
    roots = {"repo": REPO_ROOT, "datasets_root": Path(a.data_root), "run": a.out}
    t_start = time.time()
    cc = load_checkpoint_contract(resolve_cached_checkpoint_dir(a.data_root))
    modality = json.loads((Path(a.data_root) / OPENDRAWER_PACKAGE_REL / "meta" / "modality.json").read_text())
    raw_index = raw_to_model_index(modality["action"])
    ld = load_extraction(a.extraction, a.split, codec=cc.codec, raw_index=raw_index, data_root=a.data_root,
                         verify_source=not a.skip_source_verify, allow_subset=a.allow_subset_smoke)
    production = (not ld.receipt["subset"]) and a.profile == "production" and not a.skip_source_verify
    claim = build_manifest_for_purpose(a.split, a.data_root, SPEC.signature(), a.out)
    from rl2_expert_rewards import REWARD_CONTRACT, build_rl2_split, save_reward_sidecar
    train, train_rewards, train_reward_audit = build_rl2_split(ld.train, split="train", gamma=GAMMA)
    val, val_rewards, val_reward_audit = build_rl2_split(ld.val, split="val", gamma=GAMMA)
    reward_sidecar = save_reward_sidecar(a.out / "raw_and_rl2_rewards.npz", {**train_rewards, **val_rewards})
    write_json_exclusive(a.out / "reward_transition_audit.json", {"contract": REWARD_CONTRACT,
        "sidecar": reward_sidecar, "rows": train_reward_audit + val_reward_audit})
    sampler = TrainSampler(train, val=val, seed=a.seed)   # train/val episode·run_group 겹치면 거부

    overrides = {"batch_size": a.batch_size}   # 기록용 (sampler 가 실제 batch 를 정한다)
    tr = PrefixQAMTrainer.create(profile=a.profile, gamma=GAMMA, seed=a.seed,
                                 data_label="opendrawer_published_package_subset_92train_rl2_expert_baseline",
                                 execute_steps=EXECUTE_STEPS, **overrides)
    base_flags = tr.flags
    tr.flags = lambda feature_signature: dict(base_flags(feature_signature), reward_contract=REWARD_CONTRACT)
    run_manifest = {
        "version": TRAIN_VERSION, "production_run": production, "args": portable_args(a, roots),
        "extraction": {"dir_name": a.extraction.name, **{k: v for k, v in ld.receipt.items() if k != "episodes"}},
        "extraction_episodes_sha256": sha256_bytes(canonical_json(ld.receipt["episodes"])),
        "membership_claim": claim,
        "reward_contract": REWARD_CONTRACT, "reward_sidecar": reward_sidecar,
        "transitions": {"train_rows": len(train), "val_rows": len(val), "gamma": GAMMA,
                        "execute_steps": EXECUTE_STEPS,
                        "train_terminal_rows": int(train.data["terminals"].sum()),
                        "train_partial_rows": int((train.data["executed_m"] < EXECUTE_STEPS).sum()),
                        "val_terminal_rows": int(val.data["terminals"].sum())},
        "validation_use": "checkpoint selection only (no optimizer update, no hparam search)",
        "val_metric": {"name": VAL_METRIC, "direction": "min", "tie_break": "earliest step",
                       "val_seed": a.val_seed, "set": "whole val split", "batch_size": a.batch_size},
        "disk_policy": {"min_free_bytes": MIN_FREE_BYTES, "keep_all_checkpoints": True,
                        "check": "after every checkpoint: free - remaining*max_checkpoint_bytes >= min_free"},
        "eval_and_checkpoint_steps": eval_at,
        "agent_flags": tr.flags(SPEC.signature()),
        "code_sha256": {f: sha256_path(ADAPTER_DIR / f) for f in CODE_FILES},
        "git_head": _git_head(), "jax_devices": [str(d) for d in jax.devices()],
    }
    write_json_exclusive(a.out / "run_manifest.json", run_manifest)
    progress = a.out / "progress.jsonl"
    ckpt_root = a.out / "checkpoints"
    sel_root = a.out / "selection"
    ckpt_root.mkdir()
    sel_root.mkdir()
    evals, ckpts = [], []
    _fsync_append(progress, {"event": "start", "time": time.time(), "train_rows": len(train),
                             "val_rows": len(val), "setup_s": round(time.time() - t_start, 2)})

    def do_eval_and_save():
        t0 = time.time()
        m = tr.evaluate(val, batch_size=a.batch_size, seed=a.val_seed)
        bad = [k for k, v in m.items() if not np.isfinite(v)]
        if bad:
            raise NonFinite(f"step {tr.steps_done}: 비유한 val metric {bad}")
        ck = save_checkpoint(tr, ckpt_root, feature_signature=SPEC.signature())
        rec = {"step": tr.steps_done, "metrics": m, "checkpoint": ck, "eval_s": round(time.time() - t0, 2)}
        evals.append(rec)
        ckpts.append(ck)
        best = select_best(evals)
        write_json_exclusive(sel_root / f"after_step_{tr.steps_done:07d}.json", {
            "val_metric": VAL_METRIC, "best_step_so_far": best["step"],
            "best_value_so_far": best["metrics"][VAL_METRIC],
            "best_checkpoint_sha256": best["checkpoint"]["files"], "evaluated_steps": [e["step"] for e in evals]})
        projected = ck["bytes"] * len(eval_at)
        _fsync_append(progress, {"event": "eval", "step": tr.steps_done, "time": time.time(),
                                 "walltime_s": round(time.time() - t_start, 1), **m,
                                 "checkpoint_dir": ck["dir"], "checkpoint_bytes": ck["bytes"],
                                 "projected_total_checkpoint_bytes": projected})
        if len(ckpts) < len(eval_at):   # 마지막 저장 뒤에는 더 쓸 checkpoint 가 없다
            disk = disk_preflight(ckpt_root, ckpt_bytes=max(c["bytes"] for c in ckpts),
                                  remaining=len(eval_at) - len(ckpts))
            _fsync_append(progress, {"event": "disk", "step": tr.steps_done, **disk})

    try:
        if _disk_free(ckpt_root) < MIN_FREE_BYTES:
            raise DiskBudget(f"시작 시 disk free {_disk_free(ckpt_root)/1e9:.2f}GB < {MIN_FREE_BYTES/1e9:.2f}GB")
        do_eval_and_save()                                     # step 0
        t_log, s_log = time.time(), 0
        for i in range(1, a.steps + 1):
            info = tr.step(sampler.sample(a.batch_size))       # 비유한 loss → ContractError
            if i % a.log_every == 0 or i == a.steps:
                now = time.time()
                _fsync_append(progress, {"event": "train", "step": tr.steps_done, "time": now,
                                         "walltime_s": round(now - t_start, 1),
                                         "steps_per_s": round((i - s_log) / max(now - t_log, 1e-9), 3), **info})
                t_log, s_log = now, i
            if i in eval_at:
                do_eval_and_save()
    except ContractError as exc:
        err = scrub_paths(str(exc), roots)
        write_json_exclusive(a.out / "FAILED.json", {"step": tr.steps_done, "error": err,
                                                     "type": type(exc).__name__, "time": time.time()})
        _fsync_append(progress, {"event": "failed", "step": tr.steps_done, "error": err})
        print(f"FAILED at step {tr.steps_done}: {exc}", file=sys.stderr)
        return 2

    best = select_best(evals)
    final = {
        "version": TRAIN_VERSION, "production_run": production, "val_metric": VAL_METRIC,
        "selection_rule": "argmin val metric over evaluated checkpoints, tie -> earliest step; validation only",
        "selected_step": best["step"], "selected_value": best["metrics"][VAL_METRIC],
        "selected_checkpoint": best["checkpoint"], "evaluations": evals,
        "total_checkpoint_bytes": int(sum(c["bytes"] for c in ckpts)),
        "walltime_s": round(time.time() - t_start, 1),
        "run_manifest_sha256": sha256_path(a.out / "run_manifest.json"),
        "not_claimed": ["online rollout success rate", "gate1 composition with GR00T",
                        "historical 100_demos membership", "TRQAM adaptation"],
    }
    write_json_exclusive(a.out / "selection_final.json", final)
    _fsync_append(progress, {"event": "done", "step": tr.steps_done, "selected_step": best["step"],
                             "walltime_s": final["walltime_s"]})
    print(json.dumps({k: final[k] for k in ("selected_step", "selected_value", "total_checkpoint_bytes",
                                            "walltime_s", "production_run")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
