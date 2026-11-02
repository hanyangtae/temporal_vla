import json
from pathlib import Path
import numpy as np
import pytest
from composition import SPEC
from qam_prefix import PrefixQAMTrainer
from qam_torch_actor import export_actor, TorchQAMVelocity
from contract import ContractError

@pytest.mark.parametrize("profile",["smoke_reduced_network","production"])
def test_jax_torch_velocity_parity_all_flow_steps(tmp_path,profile):
    tr=PrefixQAMTrainer.create(profile=profile,gamma=.99,seed=42,data_label="synthetic_parity_only")
    ck=tmp_path/'checkpoint';tr.save(ck,feature_signature=SPEC.signature())
    dst=tmp_path/'actor';export_actor(tr,ck,dst,diagnostic=True)
    actor=TorchQAMVelocity(dst)
    rng=np.random.default_rng(42);features=rng.normal(size=(3,1024)).astype(np.float32)*5
    noisy=rng.normal(size=(3,192)).astype(np.float32)
    for t in range(4):
        a=tr.velocity_fn()(features,noisy,t);b=actor(features,noisy,t)
        np.testing.assert_allclose(b,a,rtol=2e-5,atol=3e-5)
    assert actor.calls==4
    with pytest.raises(ContractError):actor(features,noisy,4)
    with pytest.raises(ContractError):actor(features[:,:7],noisy,0)
    with (dst/'actor.npz').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ContractError):TorchQAMVelocity(dst)

def test_actual_rl2_smoke_checkpoint_real_latent_parity():
    root=Path(__file__).resolve().parents[4]
    base=root/'outputs/rl2_vla/groot_opendrawer_rl2_reward_full_data_smoke_20261021'
    dst=root/'outputs/rl2_vla/groot_rl2_actor_smoke_export_20261021'
    zpath=root/'outputs/analysis/groot_robocasa_lineage_20261021/lerobot_latent_smoke.npz'
    if not dst.exists():pytest.skip('actual diagnostic export unavailable')
    tr=PrefixQAMTrainer.load(base/'checkpoints/step_0000004',4,feature_signature=SPEC.signature())
    with np.load(zpath,allow_pickle=False) as z:
        candidates=[z[k] for k in z.files if z[k].ndim==2 and z[k].shape[-1]==1024]
        features=candidates[0][:3]
    noisy=np.random.default_rng(7).normal(size=(len(features),192)).astype(np.float32)
    actor=TorchQAMVelocity(dst)
    for t in range(4):np.testing.assert_allclose(actor(features,noisy,t),tr.velocity_fn()(features,noisy,t),rtol=2e-5,atol=3e-5)
    assert actor.meta['source_checkpoint']['step']==4 and actor.meta['diagnostic_only']
