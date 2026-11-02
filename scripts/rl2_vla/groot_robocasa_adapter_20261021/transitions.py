"""Chunk transition builder (episode 별 시간순) + expert target + train/val 분리 sampler.

규칙 (CONTRACT.md §7 정본):
- 결정 시점 t_i = i·m (m = 실제 실행 step 수, 1..16; 기존 수집 m=5). 마지막 chunk 는 m_i = T − t_i.
- reward_i   = Σ_{k<m_i} γ^k · r[t_i + k]          (실행된 m_i step 만)
- next_obs_i = latent(t_i + m_i)                    (같은 episode 안의 다음 결정 latent)
- discount_i = γ^{m_i} · (1 − terminal_i)           (bootstrap 계수, trainer 가 그대로 곱한다)
- next_executed_m_i = m (명목 실행 길이) if has_next else 0. 다음 결정에서 정책은 시간 한도를 모르므로
  명목 m 을 실행한다 → bootstrap critic 은 이 길이의 prefix mask 로 평가한다. 데이터의 다음 행 m_{i+1}
  (종료/절단으로 짧아진 partial) 을 bootstrap 에 쓰지 않는다 (qam_prefix.py, CONTRACT.md §10).
- terminal_i = 마지막 chunk 이고 end_reason == "terminal" (env 종료). "truncated"(timeout) 는
  terminal 이 아니므로 bootstrap 하며, 이때 latent(T) 가 없으면 거부한다.
- transition 은 episode 단위로 다 만든 뒤에만 섞는다. 섞인 배열에서 index+1 로 다음 관측을 찾지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from composition import FEATURE_DIM
from contract import ACTION_HORIZON, ActionCodec, ContractError, ENV_ACTION_DIM, reorder_raw_to_model

END_REASONS = ("terminal", "truncated")


def sparse_success_rewards(length: int, success: bool) -> np.ndarray:
    """reward 계약 'sparse_success_terminal': 성공 episode 의 마지막 env step 에만 1."""
    r = np.zeros(int(length), dtype=np.float64)
    if success:
        r[-1] = 1.0
    return r


def expert_chunk_targets(raw_actions: np.ndarray, decision_steps: Sequence[int], *,
                         codec: ActionCodec, raw_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """데이터셋 expert action (T, 12; raw LeRobot 순서) → 결정 시점별 (16, 12) 정규화 모델순서 target.

    target 은 데이터셋 action 그 자체를 재배열·정규화한 것이다 (GR00T 예측으로 대체하지 않는다).
    episode 끝을 넘는 위치는 0 으로 채우고 valid=False 로 표시한다 (다음 episode 로 넘어가지 않음).
    """
    raw = np.asarray(raw_actions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != ENV_ACTION_DIM:
        raise ContractError(f"expert raw action shape {raw.shape} != (T, 12)")
    norm = codec.normalize(reorder_raw_to_model(raw, raw_index))
    T = len(raw)
    steps = np.asarray(decision_steps, dtype=np.int64)
    chunks = np.zeros((len(steps), ACTION_HORIZON, ENV_ACTION_DIM), dtype=np.float32)
    valid = np.zeros((len(steps), ACTION_HORIZON), dtype=bool)
    for i, t in enumerate(steps):
        if not 0 <= t < T:
            raise ContractError(f"decision step {t} 이 episode [0, {T}) 밖")
        n = min(ACTION_HORIZON, T - t)
        chunks[i, :n] = norm[t:t + n]
        valid[i, :n] = True
    return chunks, valid


@dataclass(frozen=True)
class EpisodeInput:
    episode_id: str
    run_group: str
    split: str                     # train | val
    length: int                    # 실행된 env step 수 T
    end_reason: str                # terminal | truncated
    decision_steps: np.ndarray     # (n,) = 0, m, 2m, ... < T
    latents: np.ndarray            # (n, 1024) 각 결정 시점 decoder-입력 feature
    action_chunks: np.ndarray      # (n, 16, 12) 정규화 모델순서 chunk (rollout=정책 chunk, expert=target)
    rewards: np.ndarray            # (T,) env step 별 reward (계약은 manifest reward_contract)
    chunk_valid: np.ndarray | None = None   # (n, 16) chunk 위치 유효 여부 (None=전부 유효)
    final_latent: np.ndarray | None = None  # (1024,) latent(T); truncated bootstrap 에 필요


FIELDS = ("observations", "actions", "chunk_valid", "executed_mask", "executed_m", "rewards",
          "discounts", "terminals", "truncated_end", "next_observations", "has_next",
          "next_executed_m", "decision_step", "episode_id", "run_group")


def build_episode_transitions(ep: EpisodeInput, *, execute_steps: int, gamma: float) -> dict:
    m = int(execute_steps)
    if not 1 <= m <= ACTION_HORIZON:
        raise ContractError(f"execute_steps {m} not in [1, {ACTION_HORIZON}]")
    if not 0.0 < gamma <= 1.0:
        raise ContractError(f"gamma {gamma} not in (0, 1]")
    if ep.end_reason not in END_REASONS:
        raise ContractError(f"{ep.episode_id}: end_reason {ep.end_reason!r} (terminal|truncated 명시)")
    T = int(ep.length)
    steps = np.asarray(ep.decision_steps, dtype=np.int64)
    want = np.arange(0, T, m, dtype=np.int64)
    if T <= 0 or not np.array_equal(steps, want):
        raise ContractError(f"{ep.episode_id}: decision_steps 가 0, {m}, ... < {T} 와 다름")
    n = len(steps)
    lat = np.asarray(ep.latents, dtype=np.float32)
    if lat.shape != (n, FEATURE_DIM) or not np.all(np.isfinite(lat)):
        raise ContractError(f"{ep.episode_id}: latents shape {lat.shape} != ({n}, {FEATURE_DIM})")
    acts = np.asarray(ep.action_chunks, dtype=np.float32)
    if acts.shape != (n, ACTION_HORIZON, ENV_ACTION_DIM):
        raise ContractError(f"{ep.episode_id}: action_chunks shape {acts.shape}")
    valid = (np.ones((n, ACTION_HORIZON), bool) if ep.chunk_valid is None
             else np.asarray(ep.chunk_valid, dtype=bool))
    r = np.asarray(ep.rewards, dtype=np.float64)
    if r.shape != (T,) or not np.all(np.isfinite(r)):
        raise ContractError(f"{ep.episode_id}: rewards shape {r.shape} != ({T},)")

    out = {k: [] for k in FIELDS}
    for i, t in enumerate(steps):
        mi = min(m, T - int(t))
        last = int(t) + mi == T
        terminal = last and ep.end_reason == "terminal"
        if not last:
            nxt, has_next = lat[i + 1], True        # steps[i+1] == t + m (위에서 검사)
        elif terminal:
            nxt, has_next = np.zeros(FEATURE_DIM, np.float32), False
        else:
            if ep.final_latent is None:
                raise ContractError(f"{ep.episode_id}: truncated 마지막 chunk 의 latent(T={T}) 없음 "
                                    "— bootstrap 불가, 거부")
            nxt = np.asarray(ep.final_latent, dtype=np.float32)
            if nxt.shape != (FEATURE_DIM,) or not np.all(np.isfinite(nxt)):
                raise ContractError(f"{ep.episode_id}: final_latent shape {nxt.shape}")
            has_next = True
        if not valid[i, :mi].all():
            raise ContractError(f"{ep.episode_id}@{t}: 실행된 chunk 위치가 valid 아님")
        disc = gamma ** np.arange(mi)
        executed = np.zeros(ACTION_HORIZON, bool)
        executed[:mi] = True
        out["observations"].append(lat[i])
        out["actions"].append(acts[i])
        out["chunk_valid"].append(valid[i])
        out["executed_mask"].append(executed)
        out["executed_m"].append(mi)
        out["rewards"].append(float(np.dot(disc, r[t:t + mi])))
        out["discounts"].append(0.0 if terminal else float(gamma ** mi))
        out["terminals"].append(terminal)
        out["truncated_end"].append(last and not terminal)
        out["next_observations"].append(nxt)
        out["has_next"].append(has_next)
        out["next_executed_m"].append(m if has_next else 0)
        out["decision_step"].append(int(t))
        out["episode_id"].append(ep.episode_id)
        out["run_group"].append(ep.run_group)
    return {k: np.asarray(v) for k, v in out.items()}


@dataclass(frozen=True)
class TransitionSet:
    split: str
    data: Mapping[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.data["observations"])

    @property
    def episode_ids(self) -> set:
        return set(self.data["episode_id"].tolist())

    @property
    def run_groups(self) -> set:
        return set(self.data["run_group"].tolist())


def build_split(episodes: Sequence[EpisodeInput], *, split: str, execute_steps: int,
                gamma: float) -> TransitionSet:
    if split not in ("train", "val"):
        raise ContractError(f"split {split!r} not in train|val")
    eps = list(episodes)
    if not eps:
        raise ContractError(f"{split}: episode 없음")
    ids = [e.episode_id for e in eps]
    if len(set(ids)) != len(ids):
        raise ContractError(f"{split}: episode_id 중복")
    if any(e.split != split for e in eps):
        raise ContractError(f"{split}: 다른 split episode 혼입")
    parts = [build_episode_transitions(e, execute_steps=execute_steps, gamma=gamma) for e in eps]
    data = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    return TransitionSet(split, data)


class TrainSampler:
    """optimizer 용 sampler. train split 만 받고, val 과 episode/run_group 이 겹치면 거부한다.

    sample 은 이미 만들어진 transition 행만 고른다 (next_observations 는 행에 저장된 값).
    """

    def __init__(self, train: TransitionSet, *, val: TransitionSet | None, seed: int):
        if not isinstance(train, TransitionSet) or train.split != "train":
            raise ContractError("optimizer loader 는 train split 만 허용 (val 투입 금지)")
        if val is not None:
            if val.split != "val":
                raise ContractError("val 인자는 val split 이어야 함")
            if train.episode_ids & val.episode_ids:
                raise ContractError("train/val episode 겹침")
            if train.run_groups & val.run_groups:
                raise ContractError("train/val run_group 겹침")
        self._train = train
        self._rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> dict:
        idx = self._rng.integers(0, len(self._train), size=int(batch_size))
        return {k: v[idx] for k, v in self._train.data.items()}


def val_batches(val: TransitionSet, batch_size: int):
    """평가 전용 순차 batch (optimizer 에 넘기지 않는다)."""
    if val.split != "val":
        raise ContractError("val_batches 는 val split 만")
    for s in range(0, len(val), int(batch_size)):
        yield {k: v[s:s + batch_size] for k, v in val.data.items()}
