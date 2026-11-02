import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from conftest import REPO_ROOT, UPSTREAM_STATE_ACTION, extract_source
from contract import (
    ACTION_KEYS,
    CONTRACT_KEY,
    FEATURE_INTERFACE,
    RAW_LEROBOT_ACTION_SLICES,
    ActionCodec,
    ContractError,
    build_action_codec,
    load_checkpoint_contract,
    raw_to_model_index,
    reorder_model_to_raw,
    reorder_raw_to_model,
    validate_model_config,
    validate_qam_flags,
)

OPENDRAWER_META = Path.home() / (".cache/temporal_vla/datasets/robocasa/v1.0/pretrain/atomic/"
                                 "OpenDrawer/20250819/lerobot/meta")


@pytest.fixture(scope="module")
def ckpt(cached_checkpoint_dir):
    return load_checkpoint_contract(cached_checkpoint_dir)


@pytest.fixture(scope="module")
def raw_config(cached_checkpoint_dir):
    return json.loads((cached_checkpoint_dir / "config.json").read_text())


@pytest.fixture(scope="module")
def metadata(cached_checkpoint_dir):
    return json.loads((cached_checkpoint_dir / "experiment_cfg/metadata.json").read_text())


# ---------------------------------------------------------------- checkpoint config
def test_cached_checkpoint_passes_contract(ckpt):
    assert ckpt.codec.min.shape == (12,)
    assert ckpt.codec.modes.count("binary") == 2


@pytest.mark.parametrize("path,value", [
    (("action_dim",), 7),
    (("action_head_cfg", "action_dim"), 7),
    (("action_horizon",), 4),
    (("action_head_cfg", "action_horizon"), 4),
    (("action_head_cfg", "num_inference_timesteps"), 10),
    (("action_head_cfg", "input_embedding_dim"), 1024),
    (("action_head_cfg", "diffusion_model_cfg", "attention_head_dim"), 32),
    (("action_head_cfg", "diffusion_model_cfg", "num_layers"), 12),
    (("action_head_cfg", "diffusion_model_cfg", "output_dim"), 1536),
    (("action_head_cfg", "hidden_size"), 1536),
    (("action_head_cfg", "backbone_embedding_dim"), 1536),
    (("model_type",), "gr00t_n1_6"),
])
def test_rejects_wrong_checkpoint_config(raw_config, path, value):
    cfg = copy.deepcopy(raw_config)
    node = cfg
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(ContractError):
        validate_model_config(cfg)


def test_rejects_unpinned_checkpoint(tmp_path, raw_config, metadata):
    (tmp_path / "experiment_cfg").mkdir()
    cfg = dict(raw_config, transformers_version="x")  # 내용 1바이트라도 다르면 sha 불일치
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "experiment_cfg/metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ContractError, match="sha"):
        load_checkpoint_contract(tmp_path)
    load_checkpoint_contract(tmp_path, require_pinned=False)  # 구조 계약은 통과


# ---------------------------------------------------------------- action order / masks
def _data_config_literals():
    src = (REPO_ROOT / "configs/policies/robocasa_n15_panda_omron_data_config.py").read_text()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("action_keys", "action_normalization_modes"):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def test_action_order_matches_training_data_config_not_metadata_order(metadata):
    lit = _data_config_literals()
    keys = [f"action.{k}" for k, _, _ in ACTION_KEYS]
    assert keys == lit["action_keys"]
    assert [m for _, _, m in ACTION_KEYS] == [lit["action_normalization_modes"][k] for k in keys]
    # metadata dict 삽입 순서는 학습 concat 순서와 다르다 → 의존하면 안 됨
    assert list(metadata["new_embodiment"]["statistics"]["action"]) != [k for k, _, _ in ACTION_KEYS]


def test_action_order_matches_lerobot_profile():
    prof = yaml.safe_load((REPO_ROOT / "configs/checkpoints/"
                           "lerobot_groot_n15__robocasa365_ckpt120000.yaml").read_text())
    assert [int(a["dims"]) for a in prof["action_layout"]] == [d for _, d, _ in ACTION_KEYS]
    assert prof["model_specific"]["checkpoint_subfolder"].endswith("checkpoint-120000")
    assert prof["model_specific"]["native_action_unapply"] is True


def test_masks(ckpt):
    am = ActionCodec.active_mask()
    assert am.shape == (32,) and am[:12].all() and not am[12:].any()
    assert np.flatnonzero(ckpt.codec.binary_mask).tolist() == [6, 11]
    assert np.flatnonzero(ckpt.codec.degenerate_mask).tolist() == [10]  # base_motion[3] min=max=0


# ---------------------------------------------------------------- codec parity vs upstream
@pytest.fixture(scope="module")
def upstream_normalizer():
    ns = {"torch": torch}
    exec(extract_source(UPSTREAM_STATE_ACTION, "Normalizer", ast.ClassDef), ns)
    return ns["Normalizer"]


def test_codec_matches_upstream_normalizer(ckpt, metadata, upstream_normalizer):
    stats = metadata["new_embodiment"]["statistics"]["action"]
    rng = np.random.default_rng(0)
    raw = rng.uniform(-1.2, 1.2, size=(5, 16, 12))
    norm_ours = ckpt.codec.normalize(raw)
    model_out = rng.uniform(-1.5, 1.5, size=(5, 16, 32))
    inv_ours = ckpt.codec.unapply(model_out)
    off = 0
    for key, dims, mode in ACTION_KEYS:
        n = upstream_normalizer(mode, {k: v for k, v in stats[key].items()})
        sl = slice(off, off + dims)
        ref = n.forward(torch.tensor(raw[..., sl], dtype=torch.float64)).numpy()
        np.testing.assert_allclose(norm_ours[..., sl], ref, atol=1e-6, err_msg=key)
        ref_inv = n.inverse(torch.tensor(model_out[..., sl], dtype=torch.float64)).numpy()
        np.testing.assert_allclose(inv_ours[..., sl], ref_inv, atol=1e-6, err_msg=key)
        off += dims


def test_unapply_ignores_padding_and_rejects_bad_shapes(ckpt):
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, size=(2, 16, 32))
    b = a.copy()
    b[..., 12:] = rng.normal(size=(2, 16, 20)) * 1e3
    np.testing.assert_array_equal(ckpt.codec.unapply(a), ckpt.codec.unapply(b))
    b[..., 12:] = np.nan  # padding 의 nan 도 읽지 않음
    np.testing.assert_array_equal(ckpt.codec.unapply(a), ckpt.codec.unapply(b))
    for bad in [(2, 4, 32), (2, 16, 7), (2, 16, 12), (16,)]:
        with pytest.raises(ContractError):
            ckpt.codec.unapply(np.zeros(bad))
    c = a.copy()
    c[0, 0, 3] = np.inf
    with pytest.raises(ContractError):
        ckpt.codec.unapply(c)


def test_binary_and_degenerate_unapply(ckpt):
    a = np.zeros((1, 16, 32))
    a[..., 6] = 0.51
    a[..., 11] = 0.49
    a[..., 10] = 0.9
    raw = ckpt.codec.unapply(a)
    assert set(np.unique(raw[..., 6])) == {1.0} and set(np.unique(raw[..., 11])) == {0.0}
    assert np.all(raw[..., 10] == 0.0)
    env = ckpt.codec.split_env_action(raw[0, 0])
    assert {k: v.shape for k, v in env.items()} == {
        "action.end_effector_position": (3,), "action.end_effector_rotation": (3,),
        "action.gripper_close": (1,), "action.base_motion": (4,), "action.control_mode": (1,)}


def test_env_and_lerobot_drift_guards():
    gym = (REPO_ROOT / "src/benchmarks/robocasa/robocasa/wrappers/gym_wrapper.py").read_text()
    assert '-1.0 if input_action["action.gripper_close"] < 0.5 else 1.0' in gym
    assert '-1.0 if input_action["action.control_mode"] < 0.5 else 1.0' in gym
    pol = (REPO_ROOT / "lerobot/src/lerobot/policies/groot/modeling_groot.py").read_text()
    assert "actions = actions[:, :, :original_action_dim]" in pol


def test_out_of_range_fraction(ckpt):
    a = np.zeros((1, 16, 32))
    a[..., 0] = 1.5
    assert ckpt.codec.out_of_range_fraction(a) == pytest.approx(1 / 9)


def test_codec_rejects_missing_or_reshaped_metadata(metadata):
    m = copy.deepcopy(metadata)
    del m["new_embodiment"]["statistics"]["action"]["control_mode"]
    with pytest.raises(ContractError):
        build_action_codec(m)
    m = copy.deepcopy(metadata)
    m["new_embodiment"]["modalities"]["action"]["gripper_close"]["shape"] = [2]
    with pytest.raises(ContractError):
        build_action_codec(m)


# ---------------------------------------------------------------- QAM flags
SIG = "0123456789abcdef"


def _good_flags():
    from contract import CONFIG_SHA256, METADATA_SHA256

    return {
        "seed": 0,
        "agent": {"ob_dims": [1024], "action_dim": 12, "horizon_length": 16,
                  "action_chunking": True, "flow_steps": 4, "best_of_n": 1},
        CONTRACT_KEY: {
            "config_sha256": CONFIG_SHA256, "metadata_sha256": METADATA_SHA256,
            "feature_signature": SIG,
            "feature_interface": FEATURE_INTERFACE,
            "flow_time_convention": "t=i/flow_steps, noise(t=0)->action(t=1), v=action-noise",
            "action_space": "groot_normalized_active12",
        },
    }


def test_good_qam_flags_pass():
    validate_qam_flags(_good_flags(), feature_signature=SIG)


@pytest.mark.parametrize("mutate", [
    lambda f: f["agent"].update(action_dim=7),
    lambda f: f["agent"].update(horizon_length=4),
    lambda f: f["agent"].update(ob_dims=[1536]),  # block residual QAM
    lambda f: f["agent"].update(ob_dims=[2048]),
    lambda f: f["agent"].update(ob_dims=[512]),
    lambda f: f[CONTRACT_KEY].update(feature_interface="dit_block11_residual_action_mean"),
    lambda f: f["agent"].update(flow_steps=10),
    lambda f: f.pop(CONTRACT_KEY),
    lambda f: f[CONTRACT_KEY].update(feature_signature="ffff"),
    lambda f: f[CONTRACT_KEY].update(config_sha256="0" * 64),
    lambda f: f.update(env_name="bridge"),
    lambda f: f.update(bridge_dataset_dir="/mnt/hdd/oxe"),
])
def test_bad_qam_flags_rejected(mutate):
    f = _good_flags()
    mutate(f)
    with pytest.raises(ContractError):
        validate_qam_flags(f, feature_signature=SIG)


def test_real_pi0_bridge_flags_rejected():
    paths = sorted(Path(REPO_ROOT / "outputs").glob("rl2_trqam_risk_mix_20261014/*/train_seed42/flags.json"))
    if not paths:
        pytest.skip("로컬 pi0 Bridge QAM flags 없음")
    flags = json.loads(paths[0].read_text())
    with pytest.raises(ContractError, match="Bridge"):
        validate_qam_flags(flags, feature_signature=SIG)


def test_cached_config_has_decoder_input_1024_and_internal_1536(raw_config):
    head = raw_config["action_head_cfg"]
    assert head["diffusion_model_cfg"]["output_dim"] == 1024 == head["hidden_size"]
    assert head["input_embedding_dim"] == 1536 == 32 * 48  # 내부 DiT 폭 검사는 유지
    assert head["backbone_embedding_dim"] == 2048


def test_block_residual_qam_flags_message():
    f = _good_flags()
    f["agent"]["ob_dims"] = [1536]
    with pytest.raises(ContractError, match="다른 인터페이스"):
        validate_qam_flags(f, feature_signature=SIG)


# ---------------------------------------------------------------- raw LeRobot12 → model order
def test_raw_to_model_index_explicit():
    idx = raw_to_model_index()
    # model = eef(raw5:8) rot(raw8:11) grip(raw11) base(raw0:4) mode(raw4)
    assert idx.tolist() == [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]


def test_raw_reorder_roundtrip_and_values():
    idx = raw_to_model_index()
    raw = np.random.default_rng(0).normal(size=(3, 20, 12))
    model = reorder_raw_to_model(raw, idx)
    np.testing.assert_array_equal(model[..., 0:3], raw[..., 5:8])
    np.testing.assert_array_equal(model[..., 3:6], raw[..., 8:11])
    np.testing.assert_array_equal(model[..., 6], raw[..., 11])
    np.testing.assert_array_equal(model[..., 7:11], raw[..., 0:4])
    np.testing.assert_array_equal(model[..., 11], raw[..., 4])
    np.testing.assert_array_equal(reorder_model_to_raw(model, idx), raw)
    np.testing.assert_array_equal(reorder_raw_to_model(reorder_model_to_raw(model, idx), idx), model)


@pytest.mark.parametrize("bad", [
    dict(RAW_LEROBOT_ACTION_SLICES, gripper_close=(10, 11)),           # 겹침
    dict(RAW_LEROBOT_ACTION_SLICES, base_motion=(0, 3)),               # 폭 오류
    {k: v for k, v in RAW_LEROBOT_ACTION_SLICES.items() if k != "control_mode"},
])
def test_raw_slices_rejected(bad):
    with pytest.raises(ContractError):
        raw_to_model_index(bad)


def test_raw_slices_match_local_opendrawer_modality_and_stats():
    mod = OPENDRAWER_META / "modality.json"
    if not mod.is_file():
        pytest.skip("로컬 OpenDrawer LeRobot meta 없음")
    action = json.loads(mod.read_text())["action"]
    assert {k: (v["start"], v["end"]) for k, v in action.items()} == RAW_LEROBOT_ACTION_SLICES
    assert raw_to_model_index(action).tolist() == raw_to_model_index().tolist()
    st = json.loads((OPENDRAWER_META / "stats.json").read_text())["action"]
    lo, hi = np.asarray(st["min"]), np.asarray(st["max"])
    assert np.all(lo[0:4] == 0) and np.all(hi[0:4] == 0)        # base_motion: 이 task 에선 0
    assert lo[4] == hi[4] == -1.0                                # control_mode 상수
    assert set(np.r_[lo[11], hi[11]]) == {-1.0, 1.0}             # gripper 이진 ±1
