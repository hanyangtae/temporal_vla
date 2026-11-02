"""SYNTHETIC CPU smoke: prefix-critic QAM 실제 학습 API 를 합성 transition 으로 돌린다.

이 스크립트의 결과는 **합성 데이터 smoke** 다. 데이터셋 학습도, 실제 GR00T latent 학습도,
GPU parity 도 아니다. 관측 1024 / action 12 / H16 / flow 4 / 실행 m=5 계약은 그대로 쓰고,
망 크기만 ``smoke_reduced_network`` 로 줄인다 (production 망은 생성·1-step shape 점검만).

실행 (repo root, host CPU):
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
    .venvs/rl2_iid/bin/python scripts/rl2_vla/groot_robocasa_adapter_20261021/smoke_prefix_qam.py \
    --out outputs/rl2_vla/groot_prefix_qam_synthetic_smoke_20261021
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parents[2]
for p in (ADAPTER_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from composition import SPEC  # noqa: E402
from contract import ACTION_HORIZON, ENV_ACTION_DIM  # noqa: E402
from transitions import EpisodeInput, TrainSampler, build_split, sparse_success_rewards  # noqa: E402

SYNTHETIC_LABEL = "synthetic_smoke_not_dataset_training_not_gpu_parity"


def synthetic_episodes(seed: int, *, split: str, lengths, execute_steps: int = 5) -> list[EpisodeInput]:
    """합성 episode. 성공=terminal(마지막 step reward 1), 실패=truncated(+latent(T)).

    action chunk 는 expert 처럼 episode 끝 이후 위치를 0 padding·chunk_valid=False 로 둔다.
    """
    rng = np.random.default_rng(seed)
    eps = []
    for j, T in enumerate(lengths):
        steps = np.arange(0, T, execute_steps)
        success = j % 2 == 0
        acts = rng.uniform(-1, 1, size=(len(steps), ACTION_HORIZON, ENV_ACTION_DIM)).astype(np.float32)
        valid = (steps[:, None] + np.arange(ACTION_HORIZON)[None]) < T
        acts[~valid] = 0.0
        eps.append(EpisodeInput(
            episode_id=f"synthetic_{split}_{seed}_{j}", run_group=f"synthetic_{split}_{seed}_{j}",
            split=split, length=int(T), end_reason="terminal" if success else "truncated",
            decision_steps=steps,
            latents=(rng.normal(size=(len(steps), 1024)) * 0.5).astype(np.float32),
            action_chunks=acts, rewards=sparse_success_rewards(T, success), chunk_valid=valid,
            final_latent=None if success else (rng.normal(size=1024) * 0.5).astype(np.float32)))
    return eps


def synthetic_sets(seed: int = 0, gamma: float = 0.99, execute_steps: int = 5):
    train = build_split(synthetic_episodes(seed, split="train", lengths=[13, 23, 10, 17, 31, 9],
                                           execute_steps=execute_steps),
                        split="train", execute_steps=execute_steps, gamma=gamma)
    val = build_split(synthetic_episodes(seed + 1, split="val", lengths=[12, 21],
                                         execute_steps=execute_steps),
                      split="val", execute_steps=execute_steps, gamma=gamma)
    return train, val


def _tree_diff(a, b):
    import jax

    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(la, lb))


def run(out: Path, *, steps: int, seed: int, production_check: bool) -> dict:
    import jax

    from qam_prefix import PrefixQAMTrainer

    gamma = 0.99
    train, val = synthetic_sets(seed, gamma)
    tr = PrefixQAMTrainer.create(profile="smoke_reduced_network", gamma=gamma, seed=seed,
                                 data_label=SYNTHETIC_LABEL)
    p0 = jax.tree_util.tree_map(np.asarray, tr.agent.network.params)
    sampler = TrainSampler(train, val=val, seed=seed)
    t0 = time.time()
    hist = tr.fit(sampler, steps=steps, batch_size=8, val_set=val, eval_every=steps, log_every=max(1, steps // 5))
    fit_s = time.time() - t0
    p1 = tr.agent.network.params
    changed = {k: _tree_diff(p0[k], p1[k]) for k in sorted(p0)}
    finite = all(np.isfinite(v) for h in hist for k, v in h.items() if k != "step")

    ckpt_dir = out / "smoke_reduced_network"
    path = tr.save(ckpt_dir, feature_signature=SPEC.signature())
    re = PrefixQAMTrainer.load(ckpt_dir, tr.steps_done, feature_signature=SPEC.signature())
    reload_diff = _tree_diff(tr.agent.network.params, re.agent.network.params)
    obs = (np.random.default_rng(seed + 7).normal(size=(3, 1024)) * 0.5).astype(np.float32)
    noisy = np.random.default_rng(seed + 8).normal(size=(3, 192)).astype(np.float32)
    v_a = [tr.velocity_fn()(obs, noisy, t) for t in range(4)]
    v_b = [re.velocity_fn()(obs, noisy, t) for t in range(4)]
    vel_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(v_a, v_b))
    res = {
        "label": SYNTHETIC_LABEL,
        "profile": "smoke_reduced_network",
        "contract": {"ob_dim": 1024, "action_dim": 12, "horizon": 16, "flow_steps": 4,
                     "execute_steps": 5, "gamma": gamma},
        "train_rows": len(train), "val_rows": len(val),
        "partial_rows_train": int(np.sum(train.data["executed_m"] < 5)),
        "terminal_rows_train": int(np.sum(train.data["terminals"])),
        "truncated_end_rows_train": int(np.sum(train.data["truncated_end"])),
        "steps": tr.steps_done, "fit_seconds": round(fit_s, 2),
        "first_log": hist[0], "last_logs": hist[-2:], "all_finite": bool(finite),
        "param_max_abs_change": changed,
        "checkpoint": str(path.relative_to(out)),
        "reload_param_max_abs_diff": reload_diff,
        "reload_velocity_max_abs_diff": vel_diff,
        "velocity_shape": list(v_a[0].shape),
    }
    if production_check:
        t1 = time.time()
        prod = PrefixQAMTrainer.create(profile="production", gamma=gamma, seed=seed, data_label=SYNTHETIC_LABEL)
        n_params = int(sum(np.asarray(x).size for k, v in prod.agent.network.params.items()
                           if "target" not in k for x in jax.tree_util.tree_leaves(v)))
        info = prod.step(TrainSampler(train, val=val, seed=seed + 1).sample(16))
        res["production_network_check"] = {
            "hidden_dims": list(prod.agent.config["actor_hidden_dims"]), "num_qs": prod.agent.config["num_qs"],
            "non_target_param_count": n_params, "one_step_batch16_finite": all(np.isfinite(list(info.values()))),
            "critic_loss": info["critic/critic_loss"], "seconds": round(time.time() - t1, 2),
            "note": "1 update only, synthetic batch — shape/finite check, not training"}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="새 출력 폴더 (기존 가중치 경로 금지)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-production-check", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"--out {a.out} 이 비어 있지 않음 (덮어쓰기 금지)")
    a.out.mkdir(parents=True, exist_ok=True)
    res = run(a.out, steps=a.steps, seed=a.seed, production_check=not a.skip_production_check)
    (a.out / "smoke_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
