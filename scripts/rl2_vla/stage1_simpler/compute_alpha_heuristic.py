#!/usr/bin/env python3
"""RL2-VLA CP alpha 선택 휴리스틱(논문 §V-C) 재산출 — 우리 플랫폼 재학습 SAFE 검출기용.

수집한 rollout 전체에 대해 **eval 시점과 동일한 절차**로 SAFE-LSTM 점수 궤적을 계산하고,
alpha 별 CP band 발동 여부로 episode-level TPR/TNR/balanced accuracy 를 구해
수집 seed 별 top-3 alpha 를 뽑는다. 출력 JSON 구조는 저자 번들
(`RL2_CoVer_VLA/simpler/bashes/rl2_cp_alphas_combined.json`) 과 동일하다.

충실성(재구현 최소화):
  - feature 집계: SAFE `failure_prob.data.open_pizero.load_rollouts` 를 그대로 사용.
    (N,T,H,E)=(1,10,5,1024) → INTACT-pi0 redundant 첫 horizon 슬라이싱 `[:, :, 1:, :]`
    → horizon mean → diffusion idx 0 → 첫 샘플. 이는 eval 경로
    `run_simpler_eval_with_openpi.py:409-418` 과 축·순서가 동일함을 확인했다.
  - CP band 로드/스트레치: eval 이 쓰는 `rl2_utils.load_failure_detection_model` 을 직접 import.
    (38 → 38*n_action_steps=152 로 np.interp 스트레치)
  - 점수: LSTM 은 단방향·dropout 0 이라 전체 시퀀스 1회 forward 의 t 번째 출력이
    eval 의 "prefix 누적 후 마지막 timestep" 과 동일하다. `--verify` 로
    `rl2_utils.check_failure_prediction_lstm` 과 수치 일치를 실측 검증한다.
  - 발동 판정: 에피소드 내 어느 chunk k 에서든 score >= band[k*n_action_steps] 이면 실패 예측.

train/val split 재현:
  SAFE train.py 와 같은 순서(seed_everything(0) → load_rollouts → seed_everything(train.seed)
  → split_rollouts)로 60/40 split 을 재현한다. `--verify-split-wandb <run.wandb>` 로 학습 당시
  offline wandb 기록의 roc_auc/model_{train,val_seen}_tq* 와 대조해 split 동일성을 실측 확인할 수 있다
  (20260807/123421 run 기준 8/8 지표가 소수점 이하 완전 일치).

사용 예:
  python compute_alpha_heuristic.py \
      --checkpoint-dir .../logs/open_pizero-bridge-lstm-ours_cpTrue/20260807/123421 \
      --out RL2-VLA/experiments/rl2_cp_alphas_combined_ours.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
RL2_ROOT = REPO_ROOT / "RL2-VLA"
SAFE_ROOT = RL2_ROOT / "third_party" / "SAFE"

for p in (str(SAFE_ROOT), str(RL2_ROOT / "RL2_CoVer_VLA"), str(RL2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# eval 시점 함수 (CP band 로드/스트레치 + LSTM 점수) 를 그대로 사용
from simpler.rl2_utils import (  # noqa: E402
    check_failure_prediction_lstm,
    load_failure_detection_model,
)
from failure_prob.data import load_rollouts, split_rollouts  # noqa: E402
from failure_prob.utils.random import seed_everything  # noqa: E402

# train.py 와 동일한 수집-seed 추출 규칙 (…/task_seedX/episode.pkl)
_SEED_RE = re.compile(r"_seed(\d+)$")


def get_rollout_seed(rollout) -> str:
    subfolder = os.path.basename(os.path.dirname(rollout.mp4_path.replace(".mp4", ".pkl")))
    m = _SEED_RE.search(subfolder)
    return m.group(1) if m else "?"


def build_eval_cfg(checkpoint_dir: str, alpha: float, n_action_steps: int):
    """`load_failure_detection_model` 이 요구하는 최소 eval cfg."""
    return OmegaConf.create(
        {
            "failure_checkpoint_dir": checkpoint_dir,
            "use_taskwise_cp_band": False,
            "failure_cp_alpha": alpha,
            "n_action_steps": n_action_steps,
        }
    )


def load_all_rollouts(train_cfg, data_path: str | None):
    """SAFE train.py 와 동일한 순서·시드로 rollout 을 적재하고 split 을 재현한다."""
    cfg = train_cfg.copy()
    if data_path is not None:
        cfg.dataset.data_path = data_path
    cfg.dataset.load_to_cuda = False  # 점수 계산 시 배치 단위로 GPU 로 올린다
    cfg.train.log_precomputed = False
    cfg.train.log_precomputed_only = False

    seed_everything(0)  # train.py: rollout 적재 전
    all_rollouts = load_rollouts(cfg)

    seed_everything(int(cfg.train.seed))  # train.py: split 직전 재시드
    splits = split_rollouts(cfg, all_rollouts)
    return all_rollouts, splits


@torch.no_grad()
def score_rollouts(model, rollouts, batch_size: int = 64, device: str = "cuda"):
    """rollout 별 per-chunk failure score 궤적 (eval 의 timestep 별 값과 동일)."""
    model_dtype = next(model.parameters()).dtype
    scores = []
    for i in range(0, len(rollouts), batch_size):
        chunk = rollouts[i : i + batch_size]
        lens = [r.hidden_states.shape[0] for r in chunk]
        dim = chunk[0].hidden_states.shape[-1]
        feats = torch.zeros(len(chunk), max(lens), dim, dtype=model_dtype, device=device)
        for j, r in enumerate(chunk):
            feats[j, : lens[j]] = r.hidden_states.to(device=device, dtype=model_dtype)
        out = model({"features": feats})[:, :, 0]  # (B, T)
        for j, n in enumerate(lens):
            scores.append(out[j, :n].float().cpu().numpy())
    return scores


@torch.no_grad()
def verify_against_eval_fn(model, rollouts, scores, cp_band, n_verify: int, device: str):
    """eval 의 check_failure_prediction_lstm 과 배치 점수가 같은지 실측 확인."""
    max_abs = 0.0
    for r, s in zip(rollouts[:n_verify], scores[:n_verify]):
        all_features: list[torch.Tensor] = []
        for t in range(r.hidden_states.shape[0]):
            prob, _ = check_failure_prediction_lstm(
                hidden_states_last_token=r.hidden_states[t].to(device),
                failure_model=model,
                cp_band=cp_band,
                timestep=t,
                all_features=all_features,
            )
            max_abs = max(max_abs, abs(prob - float(s[t])))
    return max_abs


def episode_flags(scores, rollouts, cp_band, n_action_steps: int, tol: float = 2e-4):
    """에피소드별 '실패 예측 발동' 여부 (eval 과 동일: 어느 chunk 든 score >= band).

    반환: (flags, n_near_ties) — near-tie = 판정을 좌우하는 margin
    (max_k (score_k - band_k)) 의 절대값이 tol 미만인 **에피소드 수**.
    0 이면 float32 배치 오차(~2e-5)가 어떤 에피소드의 발동 판정도 바꿀 수 없다.
    """
    flags, n_near = [], 0
    for s in scores:
        idx = np.minimum(np.arange(len(s)) * n_action_steps, len(cp_band) - 1)
        decisive = float(np.max(s - cp_band[idx]))
        n_near += int(abs(decisive) < tol)
        flags.append(decisive >= 0)
    return flags, n_near


def verify_split_against_wandb(wandb_run_path: str, model, splits, batch_size: int, device: str):
    """학습 당시 wandb 기록의 roc_auc 와 재현 split 의 roc_auc 를 대조 (split 동일성 검증)."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    from failure_prob.utils.metrics import compute_roc_by_quantiles

    ds = datastore.DataStore()
    ds.open_for_scan(wandb_run_path)
    logged = {}
    while True:
        try:
            data = ds.scan_data()
        except AssertionError:
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            for item in rec.history.item:
                key = "/".join(item.nested_key) if item.nested_key else item.key
                m = re.fullmatch(r"roc_auc/model_(?P<split>.+)_tq(?P<q>[0-9.]+)", key)
                if m:  # 마지막 epoch 값이 최종적으로 남는다
                    logged.setdefault(m["split"], {})[float(m["q"])] = float(item.value_json)

    print(f"\n[verify-split] roc_auc 지표 대조 (wandb: {wandb_run_path})")
    print(f"  {'split':<10}{'tq':>6}{'ours':>12}{'wandb':>12}{'diff':>11}")
    worst = 0.0
    for split, rollouts in splits.items():
        quantiles = sorted(logged.get(split, {}))
        if not quantiles:
            continue
        scores = score_rollouts(model, rollouts, batch_size, device)
        auc_by_q, _, _ = compute_roc_by_quantiles(scores, rollouts, quantiles)
        for q in quantiles:
            ref = logged[split][q]
            diff = abs(auc_by_q[q] - ref)
            worst = max(worst, diff)
            print(f"  {split:<10}{q:>6}{auc_by_q[q]:>12.6f}{ref:>12.6f}{diff:>11.2e}")
    print(f"  → max diff = {worst:.2e} (0 이면 split·점수 파이프라인이 학습 당시와 동일)")
    return worst


def metrics_by_seed(rollouts, flags):
    """수집 seed 별 TPR(실패→실패) / TNR(성공→성공) / balanced accuracy."""
    agg = defaultdict(lambda: {"tp": 0, "fn": 0, "tn": 0, "fp": 0})
    for r, f in zip(rollouts, flags):
        b = agg[get_rollout_seed(r)]
        if r.episode_success == 0:  # 실패 에피소드 (positive class)
            b["tp" if f else "fn"] += 1
        else:
            b["fp" if f else "tn"] += 1
    out = {}
    for seed, b in agg.items():
        n_fail, n_succ = b["tp"] + b["fn"], b["tn"] + b["fp"]
        tpr = b["tp"] / n_fail if n_fail else float("nan")
        tnr = b["tn"] / n_succ if n_succ else float("nan")
        out[seed] = {
            "tpr": tpr,
            "tnr": tnr,
            "bal_acc": 0.5 * (tpr + tnr),
            "n_fail": n_fail,
            "n_succ": n_succ,
        }
    return out


def run(args):
    ckpt_dir = str(Path(args.checkpoint_dir).resolve())
    train_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))

    all_rollouts, splits = load_all_rollouts(train_cfg, args.data_path)
    print(f"\n[data] {len(all_rollouts)} rollouts, splits: "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    alphas = sorted(
        np.load(os.path.join(ckpt_dir, "cp_band_by_alpha.npy"), allow_pickle=True).item().keys()
    )
    if args.alphas:
        alphas = [float(a) for a in args.alphas]

    # 모델 + band 는 eval 경로 함수로 로드 (alpha 마다 band 가 다름)
    model, cp_band0 = load_failure_detection_model(
        build_eval_cfg(ckpt_dir, alphas[0], args.n_action_steps), curr_task=None
    )
    model.eval()

    if args.verify_split_wandb:
        verify_split_against_wandb(
            args.verify_split_wandb, model, splits, args.batch_size, args.device
        )

    eval_sets = {"all": all_rollouts}
    if "val_seen" in splits:
        eval_sets["val_seen"] = splits["val_seen"]

    scores_by_set = {k: score_rollouts(model, v, args.batch_size, args.device)
                     for k, v in eval_sets.items()}

    if args.verify > 0:
        max_abs = verify_against_eval_fn(
            model, eval_sets["all"], scores_by_set["all"], cp_band0, args.verify, args.device
        )
        print(f"\n[verify] batched score vs check_failure_prediction_lstm: "
              f"max |diff| = {max_abs:.3e} over {args.verify} episodes "
              f"(허용 {args.verify_tol:.0e}; 차이는 batched cuDNN LSTM 의 float32 오차)")
        assert max_abs < args.verify_tol, "배치 점수가 eval 함수와 불일치"

    # alpha × set × seed 지표
    results = {k: defaultdict(dict) for k in eval_sets}
    n_near_total = 0
    for alpha in alphas:
        _, cp_band = load_failure_detection_model(
            build_eval_cfg(ckpt_dir, alpha, args.n_action_steps), curr_task=None
        )
        for set_name, rollouts in eval_sets.items():
            flags, n_near = episode_flags(
                scores_by_set[set_name], rollouts, cp_band, args.n_action_steps, args.verify_tol
            )
            n_near_total += n_near if set_name == "all" else 0
            for seed, m in metrics_by_seed(rollouts, flags).items():
                results[set_name][seed][alpha] = m
    print(f"\n[verify] 결정 margin 이 ±{args.verify_tol:.0e} 이내인 에피소드 수 = {n_near_total} "
          f"(alpha 전체 합) → 0 이면 수치 오차가 발동 판정을 바꿀 수 없음")

    for set_name in eval_sets:
        print(f"\n{'='*88}\n[{set_name}] alpha 별 episode-level 지표\n{'='*88}")
        for seed in sorted(results[set_name], key=lambda s: ["42", "0", "7"].index(s)
                           if s in ("42", "0", "7") else 99):
            per_seed = results[set_name][seed]
            n_fail = per_seed[alphas[0]]["n_fail"]
            n_succ = per_seed[alphas[0]]["n_succ"]
            print(f"\n  collection seed {seed}  (fail={n_fail}, succ={n_succ})")
            print(f"  {'alpha':>7} {'TPR':>7} {'TNR':>7} {'BalAcc':>8}")
            for a in alphas:
                m = per_seed[a]
                print(f"  {a:>7} {m['tpr']:>7.3f} {m['tnr']:>7.3f} {m['bal_acc']*100:>8.2f}")

    # 산출 JSON: 저자 번들과 동일 구조 (top-3 alpha / bal_acc[%])
    primary = args.select_on if args.select_on in results else "all"
    alpha_out, balacc_out = {}, {}
    for seed in ["42", "0", "7"]:
        if seed not in results[primary]:
            continue
        ranked = sorted(results[primary][seed].items(), key=lambda kv: (-kv[1]["bal_acc"], kv[0]))
        top3 = ranked[: args.top_k]
        alpha_out[seed] = [float(a) for a, _ in top3]
        balacc_out[seed] = [round(m["bal_acc"] * 100, 1) for _, m in top3]

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"alpha": {"combined": alpha_out}, "bal_acc": {"combined": balacc_out}}, f, indent=4)
    print(f"\n[out] selection set = {primary}  →  {out_path}")
    print(json.dumps({"alpha": {"combined": alpha_out}, "bal_acc": {"combined": balacc_out}}, indent=4))

    if args.detail_json:
        detail = {
            s: {seed: {str(a): m for a, m in per_a.items()} for seed, per_a in by_seed.items()}
            for s, by_seed in results.items()
        }
        Path(args.detail_json).write_text(json.dumps(detail, indent=2))
        print(f"[out] detail → {args.detail_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="SAFE 학습 로그 디렉토리 (config.yaml/model_final.ckpt/cp_band_by_alpha.npy)")
    ap.add_argument("--data-path", default=None, help="rollout 루트 (기본: config.yaml 의 dataset.data_path)")
    ap.add_argument("--out", required=True, help="출력 JSON 경로")
    ap.add_argument("--detail-json", default=None, help="alpha×seed 전체 지표를 저장할 부가 JSON")
    ap.add_argument("--alphas", nargs="*", default=None, help="평가할 alpha 목록 (기본: npy 의 전체)")
    ap.add_argument("--select-on", default="val_seen", choices=["val_seen", "all"],
                    help="top-k 선정에 쓸 집합 (val_seen 재현 실패 시 all 로 폴백)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--n-action-steps", type=int, default=4, help="수집·eval 의 cfg.n_action_steps")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--verify", type=int, default=2, help="eval 함수와 대조 검증할 에피소드 수 (0=skip)")
    ap.add_argument("--verify-tol", type=float, default=2e-4, help="배치-vs-eval 점수 허용 오차 및 near-tie 기준")
    ap.add_argument("--verify-split-wandb", default=None,
                    help="학습 당시 offline wandb run 파일(run-*.wandb). roc_auc 대조로 split 재현을 검증")
    ap.add_argument("--device", default="cuda")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
