"""Public RL2 successful-expert rewards; raw extraction remains immutable.
Synthetic terminal truncates the critic's effective prefix, not physical data.
Recorded absorbing decision rows are retained with zero targets.
"""
import io
import numpy as np
from baseline_data import audit_rewards, write_exclusive, sha256_path
from contract import ContractError
from transitions import build_split, TransitionSet

REWARD_CONTRACT = {
    "name": "rl2_success_expert_last3/v1",
    "source": "rl2-vla/octo@0af7423a9922e8b6be004257b954cc1454eb2deb/octo/data/dataset.py:412-438",
    "consumer": "RL2-VLA/third_party/qam/envs/bridge_utils.py:109-135",
    "raw_fields": "raw_reward and raw_done retained without relabeling",
    "derived_fields": "rl2_reward=-1 before final3,0 final3; td_mask=0 final3; td_mask_last_step=0 finalonly",
    "gamma": 0.99,
    "scope": "published successful expert demonstrations only; never failed or unknown rollout",
    "terminal_meaning": "synthetic RL2 success-repeat convention, not observed physical termination",
    "chunk_adaptation": "H16 BC targets unchanged; critic prefix ends at first synthetic terminal inclusive; subsequent recorded decisions remain zero-reward terminal samples",
    "sampling": "chronological stride5 decision latents, not original Octo random subsampling",
}

def reward_fields(raw_reward, raw_done, *, source_role):
    if source_role != "successful_expert":
        raise ContractError("RL2 expert relabeling refuses failed/unknown rollout")
    raw = np.asarray(raw_reward, dtype=np.float32)
    done = np.asarray(raw_done, dtype=bool)
    audit_rewards(raw, done)
    T = len(raw)
    reward = -np.ones(T, np.float32); reward[-3:] = 0
    mask = np.ones(T, np.float32); mask[-3:] = 0
    last = np.ones(T, np.float32); last[-1] = 0
    return dict(raw_reward=raw.copy(), raw_done=done.copy(), rl2_reward=reward,
                rl2_td_mask=mask, rl2_td_mask_last_step=last)

def build_rl2_split(episodes, *, split, gamma=.99, execute_steps=5):
    if gamma != .99 or execute_steps != 5:
        raise ContractError("Pinned baseline gamma=.99, execute_steps=5")
    native = build_split(episodes, split=split, gamma=gamma, execute_steps=execute_steps)
    data = {k: np.array(v, copy=True) for k,v in native.data.items()}
    sidecar = {}; audit = []; offset = 0
    for ep in episodes:
        # The validated package's raw done is final-only; loader has verified it against parquet.
        done = np.zeros(ep.length, bool); done[-1] = True
        fields = reward_fields(ep.rewards, done, source_role="successful_expert")
        prefix = ep.episode_id.rsplit("/",1)[-1]
        for key,value in fields.items(): sidecar[prefix+"__"+key] = value
        for j,t in enumerate(ep.decision_steps):
            t = int(t); row = offset+j; observed = min(execute_steps,ep.length-t)
            zeros = np.flatnonzero(fields["rl2_td_mask"][t:t+observed] == 0)
            terminal = bool(len(zeros)); effective = int(zeros[0])+1 if terminal else observed
            data["rewards"][row] = np.dot(gamma**np.arange(effective), fields["rl2_reward"][t:t+effective])
            data["executed_m"][row] = effective
            data["executed_mask"][row] = np.arange(16) < effective
            data["discounts"][row] = 0. if terminal else gamma**effective
            data["terminals"][row] = terminal
            data["has_next"][row] = not terminal
            data["next_executed_m"][row] = 0 if terminal else execute_steps
            if terminal: data["next_observations"][row] = 0
            audit.append(dict(episode_id=ep.episode_id, decision_step=t,
                              observed_prefix=observed, rl2_effective_prefix=effective,
                              synthetic_terminal=terminal))
        offset += len(ep.decision_steps)
    return TransitionSet(split, data), sidecar, audit

def save_reward_sidecar(path, arrays):
    b=io.BytesIO(); np.savez_compressed(b, **arrays)
    write_exclusive(path,b.getvalue())
    return {"file":path.name,"sha256":sha256_path(path),"bytes":path.stat().st_size}
