import copy
import json

import numpy as np
import pytest

from composition import FEATURE_CAPTURE_POINT, SPEC
from contract import CONFIG_SHA256, METADATA_SHA256
from data_contract import (
    CONTRACT_VERSION,
    FINETUNE_LINEAGE,
    LOCAL_OPENDRAWER,
    DataContractBlocked,
    check_record_timing,
    feature_from_record,
    load_finetune_data_manifest,
    load_trqam_rollout_manifest,
)

SIG = SPEC.signature()
SMOKE = "published_package_subset_for_smoke"
FULL = "original_full_reproduction"


def _common():
    return {
        "contract_version": CONTRACT_VERSION,
        "checkpoint": {"config_sha256": CONFIG_SHA256, "metadata_sha256": METADATA_SHA256},
        "feature_extraction": {"status": "confirmed", "feature_signature": SIG,
                               "capture_point": FEATURE_CAPTURE_POINT, "denoise_step": 0},
    }


def finetune_manifest():
    m = _common()
    m.update({
        "dataset_identity": {"status": "confirmed", "dataset_soup": "pretrain_human300",
                             "registry_commit": FINETUNE_LINEAGE["registry_commit"],
                             "filter_key": "100_demos", "evidence": ["report.md"]},
        "episode_membership": {"status": "confirmed", "mode": "published_package_subset_for_smoke",
                               "historical_membership_claimed": False,
                               "package_inventory": {
                                   "info_sha256": LOCAL_OPENDRAWER["info_sha256"],
                                   "episodes_jsonl_sha256": LOCAL_OPENDRAWER["episodes_jsonl_sha256"],
                                   "package_episode_count": 102},
                               "aux_split_sha256": "0" * 64},
        "normalization_source": "checkpoint_metadata",
        "action_target": "expert_dataset_action",
        "split_policy": {"status": "confirmed", "decided_by": "main", "unit": "episode"},
        "episodes": [{"episode_id": "a", "split": "train", "run_group": "g1"},
                     {"episode_id": "b", "split": "val", "run_group": "g2"}],
    })
    return m


EP = {"split": "train", "has_feature": True, "end_reason": "truncated",
      "feature_capture_point": FEATURE_CAPTURE_POINT,
      "feature_axes": ["denoise_step", "model_token", "feature_dim"], "record_shape": [4, 49, 1024]}


def rollout_manifest():
    m = _common()
    m.update({
        "scenario": {"status": "confirmed"},
        "reward_contract": {"status": "confirmed", "name": "sparse_success_terminal", "gamma": 0.99},
        "transition": {"execute_steps": 5},
        "episodes": [
            dict(EP, episode_id="h1", run_group="base", source_role="historical_base",
                 success=True, end_reason="terminal", scenario_seed=1, noise_id=0),
            dict(EP, episode_id="x1", run_group="exp", source_role="historical_expert",
                 scenario_seed=2),
            dict(EP, episode_id="c1", run_group="chg", source_role="changed_failure",
                 success=False, scenario_seed=3, noise_id=1),
            dict(EP, episode_id="c2", run_group="chg_val", split="val",
                 source_role="changed_failure", success=False, scenario_seed=4, noise_id=2),
        ],
        "evaluation": {"status": "confirmed", "episode_ids": ["e1", "e2"],
                       "run_groups": ["eval"], "scenario_seeds": [100000, 100001],
                       "noise_ids": [9], "reset_ids": []},
    })
    return m


def _write(tmp_path, m):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    return p


def test_lineage_constants():
    assert FINETUNE_LINEAGE["dataset_soup"] == "pretrain_human300"
    assert FINETUNE_LINEAGE["tasks"] == 300 and FINETUNE_LINEAGE["nominal_demos"] == 30000
    assert LOCAL_OPENDRAWER["episodes"] == 102 and LOCAL_OPENDRAWER["frames"] == 20488
    assert LOCAL_OPENDRAWER["splits"] == {"train": "0:102"}


def test_valid_manifests_pass(tmp_path):
    out = load_finetune_data_manifest(_write(tmp_path, finetune_manifest()), feature_signature=SIG,
                                      purpose=SMOKE)
    assert out["membership_claim"]["original_membership_confirmed"] is False
    load_trqam_rollout_manifest(_write(tmp_path, rollout_manifest()), feature_signature=SIG)


def test_current_finetune_state_is_blocked_on_membership(tmp_path):
    m = finetune_manifest()
    m["episode_membership"]["status"] = "unresolved"  # 2026-10-21 실제 상태
    for purpose in (SMOKE, FULL):
        with pytest.raises(DataContractBlocked, match="episode_membership"):
            load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=purpose)


def _del(path):
    def f(m):
        node = m
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
    return f


@pytest.mark.parametrize("mutate", [
    _del(("dataset_identity",)),
    lambda m: m["dataset_identity"].update(status="unresolved"),
    lambda m: m["dataset_identity"].update(evidence=[]),
    lambda m: m["dataset_identity"].update(dataset_soup="Tabletop24k"),
    lambda m: m["dataset_identity"].update(dataset_soup="target500"),
    lambda m: m["dataset_identity"].update(registry_commit="main"),
    _del(("episode_membership",)),
    lambda m: m.update(action_target="groot_prediction"),
    lambda m: m["split_policy"].update(status="pending"),
    _del(("split_policy",)),
    lambda m: m.update(normalization_source="dataset_recomputed"),
    lambda m: m["episodes"].append({"episode_id": "c", "split": "test", "run_group": "g3"}),
    lambda m: m["episodes"].append({"episode_id": "a", "split": "train", "run_group": "g1"}),
    lambda m: m["episodes"].append({"episode_id": "d", "split": "train", "run_group": "g2"}),
    lambda m: m.update(episodes=[{"episode_id": "b", "split": "val", "run_group": "g2"}]),
    lambda m: m["checkpoint"].update(config_sha256="0" * 64),
    lambda m: m["feature_extraction"].update(feature_signature="ffff"),
    lambda m: m["feature_extraction"].update(capture_point="action_head.model.transformer_blocks[11]"),
    lambda m: m["feature_extraction"].update(status="draft"),
    lambda m: m.update(contract_version="groot_robocasa_adapter_20261021/v1"),
])
def test_finetune_manifest_fail_closed(tmp_path, mutate):
    m = finetune_manifest()
    mutate(m)
    with pytest.raises(DataContractBlocked):
        load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=SMOKE)


def _ep(i, **kw):
    return lambda m: m["episodes"][i].update(**kw)


@pytest.mark.parametrize("mutate,match", [
    (_ep(2, source_role="changed_success", success=True), "changed_success"),
    (_ep(2, success=True), "success is False"),
    (_ep(2, success=None), "success"),
    (lambda m: m["episodes"][2].pop("success"), "success"),
    (_ep(2, source_role="changed_unknown"), "source_role"),
    (_ep(0, end_reason="done"), "end_reason"),
    (_ep(0, has_feature=False), "feature"),
    (_ep(0, split="eval"), "split"),
    (_ep(3, run_group="chg"), "run_group"),                       # train/val 같은 group
    (lambda m: m["scenario"].update(status="tbd"), "scenario"),
    (lambda m: m.pop("reward_contract"), "reward_contract"),
    (lambda m: m.update(transition={"execute_steps": 17}), "execute_steps"),
    (lambda m: m["evaluation"].update(episode_ids=["c1"]), "episode_ids"),
    (lambda m: m["evaluation"].update(run_groups=["base"]), "run_groups"),
    (lambda m: m["evaluation"].update(scenario_seeds=[3]), "scenario_seeds"),
    (lambda m: m["evaluation"].update(noise_ids=[1]), "noise_ids"),
    (lambda m: m["evaluation"].update(scenario_seeds=[]), "evaluation"),
    (lambda m: m["evaluation"].update(status="draft"), "evaluation"),
    # 기존 rollout pkl 의 block residual capture 는 decoder 입력 계약이 아님
    (_ep(0, feature_axes=["layer", "denoise_step", "model_token", "feature_dim"],
         record_shape=[7, 4, 49, 1536]), "block residual"),
    (_ep(0, feature_axes=["layer", "denoise_step", "feature_dim"], record_shape=[16, 4, 1536]),
     "block residual"),
    (_ep(0, feature_axes=["model_token", "feature_dim"], record_shape=[49, 1536]), "block residual"),
    (_ep(0, feature_capture_point="action_head.model.transformer_blocks[15]"), "capture_point"),
    (_ep(0, feature_axes=["model_token", "feature_dim"], record_shape=[49, 1024]), "denoise"),
    (_ep(0, feature_axes=["denoise_step", "action_token", "feature_dim"],
         record_shape=[4, 17, 1024]), "16"),
])
def test_rollout_manifest_fail_closed(tmp_path, mutate, match):
    m = rollout_manifest()
    mutate(m)
    with pytest.raises(DataContractBlocked, match=match):
        load_trqam_rollout_manifest(_write(tmp_path, m), feature_signature=SIG)


def test_feature_from_full_token_record():
    h = np.random.default_rng(0).normal(size=(4, 49, 1024)).astype(np.float32)
    out = feature_from_record(h, axes=["denoise_step", "model_token", "feature_dim"],
                              capture_point=FEATURE_CAPTURE_POINT)
    np.testing.assert_allclose(out, h[0, 33:49].mean(0), rtol=1e-6)
    np.testing.assert_allclose(out, h[0, -16:].mean(0), rtol=1e-6)


def test_feature_from_pooled_and_action_token_records():
    rng = np.random.default_rng(1)
    pooled = rng.normal(size=(1024,)).astype(np.float32)
    np.testing.assert_array_equal(
        feature_from_record(pooled, axes=["feature_dim"], capture_point=FEATURE_CAPTURE_POINT,
                            denoise_step_captured=0), pooled)
    tok = rng.normal(size=(16, 1024)).astype(np.float32)
    np.testing.assert_allclose(
        feature_from_record(tok, axes=["action_token", "feature_dim"],
                            capture_point=FEATURE_CAPTURE_POINT, denoise_step_captured=0),
        tok.mean(0), rtol=1e-6)
    with pytest.raises(DataContractBlocked, match="denoise"):  # 몇 번째 call 인지 미상
        feature_from_record(pooled, axes=["feature_dim"], capture_point=FEATURE_CAPTURE_POINT)


def test_existing_block_residual_record_rejected_not_converted():
    h = np.random.default_rng(2).normal(size=(7, 4, 49, 1536)).astype(np.float32)
    with pytest.raises(DataContractBlocked, match="block residual"):
        feature_from_record(h, axes=["layer", "denoise_step", "model_token", "feature_dim"],
                            capture_point=FEATURE_CAPTURE_POINT)


def test_record_timing():
    check_record_timing([0, 5, 10, 15], 5)
    for bad in ([5, 10], [0, 5, 11], [0, 16]):
        with pytest.raises(DataContractBlocked):
            check_record_timing(bad, 5)
    with pytest.raises(DataContractBlocked):
        check_record_timing([0, 17], 17)
