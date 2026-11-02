"""QAM trainer 통합: 실행 prefix critic + 행별 discount (GR00T N1.5 RoboCasa 계약).

upstream 은 수정하지 않는다. ``RL2-VLA/third_party/qam`` 의 ``QAMAgent``·``Value``·``ActorVectorField``·
``TrainState``·``save_agent`` 를 그대로 import 하고, critic 이 닿는 메서드만 subclass 에서 바꾼다.
정본 설명은 CONTRACT.md §10.

설계 요약
- actor: H16×12 = 192 차원 전체 chunk 를 예측한다 (GR00T 와 같은 chunk, BC target 도 16 전체).
- critic: Q(s, a ⊙ m, m). m 은 실행 prefix mask (16,), 앞 k 개만 1. a 의 tail 은 critic 안에서 0 을 곱해
  지우고, mask 자체를 입력에 붙여 duration 을 조건으로 준다. 따라서
  * tail 값을 바꿔도 Q 가 bit 단위로 같고, ∂Q/∂a_tail 이 정확히 0 이다.
  * mask 를 받지 않는 critic 호출(upstream fql/edit/best-of-n 경로)은 인자 부족으로 실패한다 (fail closed).
- TD target = rewards_i + discounts_i · Q_target(s'_i, π(s'_i), m_next).
  discounts_i = γ^{m_i}(1−terminal_i) 는 transition 이 준 값을 그대로 한 번만 곱한다 (γ^H 재곱 없음).
  m_next 는 행의 next_executed_m (= 명목 m, 종료 행은 0) 이다.
- critic 학습 행의 mask 는 실제 실행 길이 m_i (마지막 partial chunk 는 m_i < m).
- actor 개선(adjoint terminal 조건)·sample ranking·평가는 명목 m mask 를 쓴다. 정책은 시간 한도를
  관측하지 않으므로 배포 시 항상 명목 m 을 실행한다.
- BC flow loss 는 chunk_valid 로 mask 한다. episode 끝을 넘는 padding tail 은 감독하지 않는다.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contract import (
    ACTION_HORIZON,
    CONFIG_SHA256,
    CONTRACT_KEY,
    ContractError,
    ENV_ACTION_DIM,
    FEATURE_INTERFACE,
    METADATA_SHA256,
    QAM_FLOW_STEPS,
    QAM_OB_DIM,
    validate_qam_flags,
)

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("GROOT_ADAPTER_REPO_ROOT") or ADAPTER_DIR.parents[2])
QAM_SRC = REPO_ROOT / "RL2-VLA" / "third_party" / "qam"


def _load_upstream():
    """upstream qam 모듈을 수정 없이 불러온다. agents/__init__ (다른 agent 전부 import) 는 건너뛴다."""
    if not (QAM_SRC / "agents" / "qam.py").is_file():
        raise ContractError(f"upstream QAM 소스 없음: {QAM_SRC}")
    if str(QAM_SRC) not in sys.path:
        sys.path.insert(0, str(QAM_SRC))
    import utils.flax_utils as fu  # noqa: E402  (upstream)
    import utils.networks as nets  # noqa: E402  (upstream)
    for mod in (fu, nets):
        if Path(mod.__file__).resolve().parent != (QAM_SRC / "utils").resolve():
            raise ContractError(f"'utils' 이름 충돌: {mod.__file__} 는 upstream qam 이 아님")
    spec = importlib.util.spec_from_file_location("rl2_qam_upstream_agent", QAM_SRC / "agents" / "qam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fu, nets


_UP, _FU, _NETS = _load_upstream()

import flax  # noqa: E402
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

QAMAgent = _UP.QAMAgent
upstream_get_config = _UP.get_config
ModuleDict, TrainState = _FU.ModuleDict, _FU.TrainState
Value, ActorVectorField = _NETS.Value, _NETS.ActorVectorField

PREFIX_CONTRACT_KEY = "prefix_critic_contract"
PREFIX_CONTRACT = {
    "version": "groot_qam_prefix_critic/v1",
    "critic_input": "concat(obs, actions*repeat(exec_mask, action_dim), exec_mask)",
    "critic_row_mask": "executed_m (actual executed prefix, partial final chunk < m)",
    "bootstrap_next_mask": "next_executed_m (= nominal execute_steps if has_next else 0)",
    "actor_and_ranking_mask": "nominal execute_steps",
    "discount_source": "per_row_discounts_gamma_pow_m_i_times_not_terminal (applied once)",
    "bc_target": "full H16 expert chunk, loss masked by chunk_valid",
}
EXECUTE_STEPS_DEFAULT = 5  # 기존 RoboCasa 수집/실행 = chunk 16 예측 / 5 실행

# RL2-VLA/third_party/qam/train_latents.sh (공개 Bridge latent 레시피) 에서 바꾼 값만.
PUBLIC_RECIPE = {
    "inv_temp": 0.1, "fql_alpha": 0.0, "edit_scale": 0.0, "action_chunking": True,
    "horizon_length": 4,  # 공개 레시피 값 (아래 GROOT_FIXED 로 16 대체)
}
# GR00T 계약 고정값 (CONTRACT.md §4). flow 4 = GR00T num_inference_timesteps.
GROOT_FIXED = {"horizon_length": ACTION_HORIZON, "flow_steps": QAM_FLOW_STEPS, "best_of_n": 1,
               "action_chunking": True, "fql_alpha": 0.0, "edit_scale": 0.0, "residual": False}
PROFILES = {
    # 공개 기본 망 (512×4, ensemble 10). 입력/출력 차원은 계약 그대로.
    "production": {},
    # CPU smoke 전용 축소 망. 관측 1024 / action 12 / H16 / flow 4 계약은 그대로 둔다.
    "smoke_reduced_network": {"actor_hidden_dims": (32, 32), "value_hidden_dims": (32, 32),
                              "num_qs": 2, "batch_size": 8},
}
# profile 별 override 허용 key (계약 key 는 어떤 profile 에서도 바꿀 수 없다).
_TUNABLE = {"lr", "batch_size", "inv_temp", "tau", "rho", "clip_adj", "clip_grad",
            "use_target_grad", "target_actor", "actor_layer_norm", "value_layer_norm"}
_SMOKE_TUNABLE = _TUNABLE | {"actor_hidden_dims", "value_hidden_dims", "num_qs"}


def prefix_mask(m, horizon: int = ACTION_HORIZON):
    """정수 m (스칼라 또는 배열) → (..., H) float32 prefix mask (앞 m 개 1)."""
    m = jnp.asarray(m)
    return (jnp.arange(horizon) < m[..., None]).astype(jnp.float32)


class PrefixMaskedValue(nn.Module):
    """upstream ``Value`` 를 감싼 실행 prefix critic. mask 없이는 호출할 수 없다."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    num_ensembles: int
    horizon: int
    action_dim: int

    @nn.compact
    def __call__(self, observations, actions, exec_mask):
        mask = exec_mask.astype(actions.dtype)
        if mask.shape[-1] != self.horizon or actions.shape[-1] != self.horizon * self.action_dim:
            raise ContractError(f"critic 입력 shape actions {actions.shape} mask {mask.shape}")
        masked = actions * jnp.repeat(mask, self.action_dim, axis=-1)
        return Value(hidden_dims=self.hidden_dims, layer_norm=self.layer_norm,
                     num_ensembles=self.num_ensembles, name="value")(
            observations, jnp.concatenate([masked, mask], axis=-1))


def _check_supported(config: Mapping[str, Any]) -> None:
    """masked critic 경로가 구현·검증되지 않은 branch 는 생성 단계에서 거부한다."""
    if not config.get("action_chunking"):
        raise ContractError("prefix critic 은 action_chunking=True 만 지원")
    if float(config.get("fql_alpha", 0.0)) != 0.0:
        raise ContractError("fql_alpha>0 (one-step actor 의 critic 호출) 은 prefix mask 미구현 → 거부")
    if float(config.get("edit_scale", 0.0)) != 0.0:
        raise ContractError("edit_scale>0 (edit actor 의 critic 호출) 은 prefix mask 미구현 → 거부")
    if config.get("residual"):
        raise ContractError("residual=True 는 검증하지 않은 경로 → 거부")
    m, H = config.get("execute_steps"), config.get("horizon_length")
    if not isinstance(m, int) or not isinstance(H, int) or not 1 <= m <= H:
        raise ContractError(f"execute_steps {m!r} 는 1..horizon_length({H}) 정수여야 함")
    if int(config.get("best_of_n", 1)) < 1:
        raise ContractError("best_of_n >= 1")


class PrefixCriticQAMAgent(QAMAgent):
    """upstream QAMAgent 에서 critic 이 닿는 경로만 바꾼 subclass."""

    # ---- helpers ----
    def _critic(self, name, observations, actions, exec_mask, params=None):
        return self.network.select(name)(observations, actions, exec_mask, params=params)

    def nominal_mask(self, batch_shape) -> jnp.ndarray:
        base = prefix_mask(self.config["execute_steps"], self.config["horizon_length"])
        return jnp.broadcast_to(base, tuple(batch_shape) + (self.config["horizon_length"],))

    def _flat(self, actions):
        return jnp.reshape(actions, (actions.shape[0], -1))

    # ---- critic ----
    def td_target(self, batch, rng, next_actions=None):
        """r_i + d_i · Q_target(s'_i, π(s'_i), m_next). d_i 는 batch 값을 그대로 쓴다."""
        if next_actions is None:
            next_actions = self.sample_actions(batch["next_observations"], rng=rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self._critic("target_critic", batch["next_observations"], next_actions,
                               batch["next_exec_mask"])
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)
        return batch["rewards"] + batch["discounts"] * next_q

    def critic_loss(self, batch, grad_params, rng):
        target_q = jax.lax.stop_gradient(self.td_target(batch, rng))
        q = self._critic("critic", batch["observations"], self._flat(batch["actions"]),
                         batch["exec_mask"], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()
        return critic_loss, {"critic_loss": critic_loss, "q_mean": q.mean(), "q_max": q.max(),
                             "q_min": q.min(), "target_mean": target_q.mean()}

    # ---- actor ----
    def terminal_adjoint(self, obs, x1):
        """adjoint 초기값 −inv_temp·∇_a Q(s, a, m_nominal). tail 성분은 정확히 0."""
        name = "target_critic" if self.config["use_target_grad"] else "critic"
        mask = self.nominal_mask(obs.shape[:-1])
        if self.config["clip_adj"]:
            f = lambda o, y: self._critic(name, o, jnp.clip(y, -1.0, 1.0), mask).mean(axis=0).sum()
        else:
            f = lambda o, y: self._critic(name, o, y, mask).mean(axis=0).sum()
        return -jax.grad(f, 1)(obs, x1) * self.config["inv_temp"]

    @partial(jax.jit, static_argnames=("flow_steps",))
    def adj_matching(self, obs, rng, flow_steps=None):
        # upstream QAMAgent.adj_matching 과 같고 adjoint 초기값만 terminal_adjoint (masked critic).
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps
        action_dim = self.config["action_dim"] * self.config["horizon_length"]
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))
        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")
        h = 1 / flow_steps
        xs, ts = [x], []
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                v = self.network.select("actor_fast")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:
                x = x + h * actor_slow(obs, x, t)
            xs.append(x)
            ts.append(t)
        adj = self.terminal_adjoint(obs, xs[-1])
        pre_adj_info = {"adj_max": jnp.abs(adj).max(), "adj_std": jnp.abs(adj).std(),
                        "adj_mean": jnp.abs(adj).mean()}
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)

            adj = adj + h * jax.vjp(fn, xs[i])[1](adj)[0]
            adjs.append(adj)
        return (jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0),
                jnp.stack(ts, axis=0), pre_adj_info)

    def bc_flow_loss(self, batch, grad_params, rng):
        """full H16 expert chunk flow-matching. chunk_valid 가 False 인 padding 위치는 감독하지 않는다."""
        vm = jnp.repeat(batch["chunk_valid"], self.config["action_dim"], axis=-1)
        # padding 위치는 loss 만 가리면 부족하다: x_t 입력을 통해 valid 차원 예측에 새어 들어간다.
        # 그래서 target 을 고정값 0 (expert_chunk_targets 의 padding 값) 으로 바꾼 뒤 보간한다.
        x_1 = self._flat(batch["actions"]) * vm
        x_rng, t_rng = jax.random.split(rng)
        x_0 = jax.random.normal(x_rng, x_1.shape)
        t = jax.random.uniform(t_rng, (x_1.shape[0], 1))
        x_t = (1 - t) * x_0 + t * x_1
        pred = self.network.select("actor_slow")(batch["observations"], x_t, t, params=grad_params)
        per_row = (jnp.square(pred - (x_1 - x_0)) * vm).sum(-1) / jnp.maximum(vm.sum(-1), 1.0)
        return per_row.mean()

    def actor_loss(self, batch, grad_params, rng):
        rng, bc_rng, adj_rng = jax.random.split(rng, 3)
        flow_loss = self.bc_flow_loss(batch, grad_params, bc_rng)
        obs = batch["observations"]
        xs, adjs, ts, pre_adj_info = self.adj_matching(obs, adj_rng)
        h = 1 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))
        observations = jnp.repeat(obs[None], self.config["flow_steps"], axis=0)
        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)
        vf_base = actor_slow(observations, xs, ts)
        adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)
        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))
        return flow_loss + adj_loss, {"flow_loss": flow_loss, "fast_loss": adj_loss,
                                      "adj_loss": adj_loss, **pre_adj_info}

    # ---- sampling / ranking / eval ----
    def rank_candidates(self, observations, actions):
        """(.., N, obs), (.., N, 192) → (.., N) critic 점수 (명목 m mask)."""
        mask = self.nominal_mask(actions.shape[:-1])
        return self._critic("critic", observations, actions, mask).mean(axis=0)

    @jax.jit
    def sample_actions(self, observations, rng):
        n = self.config["best_of_n"]
        action_dim = self.config["action_dim"] * self.config["horizon_length"]
        bshape = observations.shape[: -len(self.config["ob_dims"])]
        noises = jax.random.normal(rng, (*bshape, n, action_dim))
        observations = jnp.repeat(observations[..., None, :], n, axis=-2)
        model = "slow" if self.config["inv_temp"] == 0.0 else "fast"
        actions = self.compute_flow_actions(observations, noises, model=model)
        if n == 1:
            return actions[..., 0, :]
        idx = jnp.argmax(self.rank_candidates(observations, actions), axis=-1)
        return jnp.take_along_axis(actions, idx[..., None, None], axis=-2)[..., 0, :]

    @partial(jax.jit, static_argnames=("model",))
    def flow_velocity(self, observations, x, i, model="fast"):
        """ComposedFlowSampler 용 velocity. t = i/flow_steps (GR00T t_k = k/4 와 같은 시간)."""
        t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
        return sum(self.network.select(f"actor_{m}")(observations, x, t) for m in model.split(","))

    def compute_val_metrics(self, batch, rng):
        _, info = self.total_loss(batch, self.network.params, rng)
        pred = self.sample_actions(batch["observations"], rng=rng)
        gt = self._flat(batch["actions"])
        vm = jnp.repeat(batch["chunk_valid"], self.config["action_dim"], axis=-1)
        em = jnp.repeat(self.nominal_mask(gt.shape[:-1]), self.config["action_dim"], axis=-1) * vm
        sq = jnp.square(pred - gt)
        return {"val/critic_loss": info["critic/critic_loss"], "val/q_mean": info["critic/q_mean"],
                "val/flow_loss": info["actor/flow_loss"], "val/adj_loss": info["actor/adj_loss"],
                "val/action_mse_valid": (sq * vm).sum() / jnp.maximum(vm.sum(), 1.0),
                "val/action_mse_exec_prefix": (sq * em).sum() / jnp.maximum(em.sum(), 1.0)}

    # ---- construction ----
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        # upstream QAMAgent.create 와 같고 critic 정의·init 인자만 PrefixMaskedValue 로 바뀐다.
        config = dict(config)
        _check_supported(config)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_times = ex_actions[..., :1]
        H = config["horizon_length"]
        full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        full_action_dim = full_actions.shape[-1]
        ex_mask = prefix_mask(config["execute_steps"], H)
        if config["edit_target_entropy"] is None:
            config["edit_target_entropy"] = -config["edit_target_entropy_multiplier"] * full_action_dim
        critic_def = PrefixMaskedValue(hidden_dims=tuple(config["value_hidden_dims"]),
                                       layer_norm=config["value_layer_norm"],
                                       num_ensembles=config["num_qs"], horizon=H,
                                       action_dim=ex_actions.shape[-1])
        actor_def = ActorVectorField(hidden_dims=tuple(config["actor_hidden_dims"]),
                                     layer_norm=config["actor_layer_norm"], action_dim=full_action_dim)
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions, ex_mask)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions, ex_mask)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        if config["clip_grad"]:
            tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0), optax.adam(learning_rate=config["lr"]))
        else:
            tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})["params"]
        network = TrainState.create(network_def, params, tx=tx)
        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_actor_slow"] = params["modules_actor_slow"]
        config["ob_dims"] = ex_observations.shape
        config["action_dim"] = ex_actions.shape[-1]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ---------------------------------------------------------------------------
# config / batch
# ---------------------------------------------------------------------------

def make_config(profile: str, *, gamma: float, execute_steps: int = EXECUTE_STEPS_DEFAULT,
                **overrides) -> dict:
    if profile not in PROFILES:
        raise ContractError(f"profile {profile!r} not in {sorted(PROFILES)}")
    allowed = _SMOKE_TUNABLE if profile == "smoke_reduced_network" else _TUNABLE
    bad = sorted(set(overrides) - allowed)
    if bad:
        raise ContractError(f"{profile}: override 금지 key {bad} (계약/망 크기 고정)")
    if not 0.0 < float(gamma) <= 1.0:
        raise ContractError(f"gamma {gamma} not in (0, 1]")
    cfg = upstream_get_config().to_dict()
    cfg.update(PUBLIC_RECIPE)
    cfg.update(GROOT_FIXED)
    cfg.update(PROFILES[profile])
    cfg.update(overrides)
    cfg["discount"] = float(gamma)       # 기록용. TD target 은 batch 의 행별 discounts 만 쓴다.
    cfg["execute_steps"] = int(execute_steps)
    cfg["prefix_critic"] = True
    cfg["profile"] = profile
    _check_supported(cfg)
    return cfg


BATCH_KEYS = ("observations", "actions", "exec_mask", "chunk_valid", "rewards", "discounts",
              "next_observations", "next_exec_mask")


def to_qam_batch(rows: Mapping[str, np.ndarray], *, execute_steps: int, gamma: float,
                 rtol: float = 1e-6) -> dict:
    """transitions.FIELDS 행 → trainer batch. 행 의미가 계약과 다르면 거부한다 (host 측 검사)."""
    obs = np.asarray(rows["observations"], np.float32)
    B = obs.shape[0]
    acts = np.asarray(rows["actions"], np.float32)
    if obs.shape != (B, QAM_OB_DIM) or acts.shape != (B, ACTION_HORIZON, ENV_ACTION_DIM):
        raise ContractError(f"batch shape obs {obs.shape} actions {acts.shape}")
    em = np.asarray(rows["executed_m"], np.int64)
    emask = np.asarray(rows["executed_mask"], bool)
    valid = np.asarray(rows["chunk_valid"], bool)
    ar = np.arange(ACTION_HORIZON)
    if not np.all((em >= 1) & (em <= execute_steps)):
        raise ContractError(f"executed_m 범위 밖 (1..{execute_steps}): {np.unique(em)}")
    if not np.array_equal(emask, ar[None] < em[:, None]):
        raise ContractError("executed_mask 가 executed_m prefix 와 다름")
    if np.any(emask & ~valid):
        raise ContractError("실행 prefix 가 chunk_valid 밖")
    term = np.asarray(rows["terminals"], bool)
    has_next = np.asarray(rows["has_next"], bool)
    if np.any(term == has_next):
        raise ContractError("terminal ⇔ ¬has_next 불일치")
    disc = np.asarray(rows["discounts"], np.float64)
    want = np.where(term, 0.0, float(gamma) ** em)
    if not np.allclose(disc, want, rtol=rtol, atol=0.0):
        raise ContractError("discounts 가 γ^{m_i}(1−terminal) 와 다름 (γ^H 고정·재곱 금지)")
    nm = np.asarray(rows["next_executed_m"], np.int64)
    if not np.array_equal(nm, np.where(has_next, execute_steps, 0)):
        raise ContractError("next_executed_m 는 has_next 면 명목 m, 아니면 0")
    nxt = np.asarray(rows["next_observations"], np.float32)
    r = np.asarray(rows["rewards"], np.float32)
    for name, a in (("observations", obs), ("actions", acts), ("next_observations", nxt), ("rewards", r)):
        if not np.all(np.isfinite(a)):
            raise ContractError(f"{name} 비유한")
    if np.any(nxt[~has_next] != 0):
        raise ContractError("종료 행 next_observations 는 0 이어야 함")
    return {"observations": obs, "actions": acts, "exec_mask": emask.astype(np.float32),
            "chunk_valid": valid.astype(np.float32), "rewards": r,
            "discounts": disc.astype(np.float32), "next_observations": nxt,
            "next_exec_mask": (ar[None] < nm[:, None]).astype(np.float32)}


# ---------------------------------------------------------------------------
# provenance / checkpoint
# ---------------------------------------------------------------------------

def contract_provenance(feature_signature: str) -> dict:
    return {"config_sha256": CONFIG_SHA256, "metadata_sha256": METADATA_SHA256,
            "feature_signature": feature_signature, "feature_interface": FEATURE_INTERFACE,
            "flow_time_convention": "t=i/flow_steps, noise(t=0)->action(t=1), v=action-noise",
            "action_space": "groot_normalized_active12"}


def _jsonable(x):
    if isinstance(x, Mapping):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _ex_inputs():
    return jnp.zeros((QAM_OB_DIM,), jnp.float32), jnp.zeros((ENV_ACTION_DIM,), jnp.float32)


class PrefixQAMTrainer:
    """실제 학습 API: create → fit/step → evaluate → save/load → velocity_fn."""

    def __init__(self, agent: PrefixCriticQAMAgent, *, gamma: float, data_label: str):
        self.agent = agent
        self.gamma = float(gamma)
        self.execute_steps = int(agent.config["execute_steps"])
        self.data_label = str(data_label)
        self.steps_done = 0

    @classmethod
    def create(cls, *, profile: str, gamma: float, seed: int, data_label: str,
               execute_steps: int = EXECUTE_STEPS_DEFAULT, **overrides) -> "PrefixQAMTrainer":
        cfg = make_config(profile, gamma=gamma, execute_steps=execute_steps, **overrides)
        ex_obs, ex_act = _ex_inputs()
        return cls(PrefixCriticQAMAgent.create(seed, ex_obs, ex_act, cfg), gamma=gamma,
                   data_label=data_label)

    def batch(self, rows) -> dict:
        return to_qam_batch(rows, execute_steps=self.execute_steps, gamma=self.gamma)

    def step(self, rows) -> dict:
        self.agent, info = self.agent.update(self.batch(rows))
        self.steps_done += 1
        out = {k: float(v) for k, v in info.items()}
        bad = [k for k, v in out.items() if not np.isfinite(v)]
        if bad:
            raise ContractError(f"비유한 loss/metric: {bad}")
        return out

    def evaluate(self, val_set, *, batch_size: int, seed: int = 0) -> dict:
        from transitions import val_batches  # val 전용 순차 loader (optimizer 에 넘기지 않음)

        acc, n = {}, 0
        for i, rows in enumerate(val_batches(val_set, batch_size)):
            m = self.agent.compute_val_metrics(self.batch(rows), jax.random.PRNGKey(seed + i))
            w = len(rows["observations"])
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + float(v) * w
            n += w
        return {k: v / n for k, v in acc.items()}

    def fit(self, sampler, *, steps: int, batch_size: int, val_set=None, eval_every: int = 0,
            log_every: int = 1) -> list[dict]:
        hist = []
        for i in range(1, int(steps) + 1):
            info = self.step(sampler.sample(batch_size))
            if i % log_every == 0 or i == steps:
                hist.append({"step": self.steps_done, **info})
            if val_set is not None and eval_every and (i % eval_every == 0 or i == steps):
                hist.append({"step": self.steps_done, **self.evaluate(val_set, batch_size=batch_size)})
        return hist

    def flags(self, feature_signature: str) -> dict:
        cfg = _jsonable(dict(self.agent.config))
        cfg["ob_dims"] = list(cfg["ob_dims"])
        return {"agent": cfg, CONTRACT_KEY: contract_provenance(feature_signature),
                PREFIX_CONTRACT_KEY: dict(PREFIX_CONTRACT, execute_steps=self.execute_steps,
                                          gamma=self.gamma),
                "data_label": self.data_label, "steps_done": self.steps_done,
                "run_group": "groot_robocasa_prefix_qam"}

    def save(self, out_dir: str | Path, *, feature_signature: str) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        flags = self.flags(feature_signature)
        validate_qam_flags(flags, feature_signature=feature_signature)
        path = Path(_FU.save_agent(self.agent, str(d), self.steps_done))
        (d / f"flags_{self.steps_done}.json").write_text(json.dumps(flags, indent=1, sort_keys=True))
        return path

    @classmethod
    def load(cls, out_dir: str | Path, step: int, *, feature_signature: str) -> "PrefixQAMTrainer":
        d = Path(out_dir)
        flags = json.loads((d / f"flags_{step}.json").read_text())
        validate_qam_flags(flags, feature_signature=feature_signature)
        pc = flags.get(PREFIX_CONTRACT_KEY)
        if not isinstance(pc, Mapping) or pc.get("version") != PREFIX_CONTRACT["version"]:
            raise ContractError("prefix critic 계약 없는 checkpoint (full-chunk critic QAM 등) 거부")
        cfg = dict(flags["agent"])
        for k in ("actor_hidden_dims", "value_hidden_dims"):
            cfg[k] = tuple(cfg[k])
        if cfg.get("execute_steps") != pc.get("execute_steps"):
            raise ContractError("execute_steps 기록 불일치")
        cfg.pop("ob_dims"), cfg.pop("action_dim")
        ex_obs, ex_act = _ex_inputs()
        agent = PrefixCriticQAMAgent.create(0, ex_obs, ex_act, cfg)
        agent = _FU.restore_agent_with_file(agent, str(d / f"params_{step}.pkl"))
        tr = cls(agent, gamma=pc["gamma"], data_label=flags["data_label"])
        tr.steps_done = int(flags["steps_done"])
        return tr

    def velocity_fn(self):
        """ComposedFlowSampler(qam_fn=...) callback: (feature (B,1024), noisy (B,192), t int) → (B,192)."""
        agent = self.agent
        if agent.config["flow_steps"] != QAM_FLOW_STEPS:
            raise ContractError(f"flow_steps {agent.config['flow_steps']} != GR00T {QAM_FLOW_STEPS}")
        model = "slow" if agent.config["inv_temp"] == 0.0 else "fast"

        def qam_fn(feature, noisy, t):
            if not 0 <= int(t) < QAM_FLOW_STEPS:
                raise ContractError(f"flow step {t} 범위 밖")
            v = agent.flow_velocity(jnp.asarray(feature, jnp.float32), jnp.asarray(noisy, jnp.float32),
                                    int(t), model=model)
            return np.array(v, dtype=np.float32)  # 쓰기 가능한 복사본 (torch.as_tensor 경고 방지)

        return qam_fn
