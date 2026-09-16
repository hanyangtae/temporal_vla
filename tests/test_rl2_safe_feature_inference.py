"""Validate original SAFE transforms and loader without importing simulation modules."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SIMPLER = ROOT / 'RL2-VLA/RL2_CoVer_VLA/simpler'


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('horizon', [0.0, 1.0, 'mean', 'concat-2'])
@pytest.mark.parametrize('denoise', [0.0, 1.0, 'mean', 'concat-2'])
def test_all_16_features_match_original_safe(horizon, denoise):
    ns = extract(SIMPLER / 'run_simpler_eval_with_openpi.py',
                 {'_stage2_safe_feature', '_stage2_context'}, {'np': np})
    original = extract(ROOT / 'RL2-VLA/third_party/SAFE/failure_prob/data/utils.py',
                       {'process_tensor_idx_rel', 'parse_and_index_tensor_last'}, {'np': np})
    values = np.random.default_rng(7).normal(size=(1, 10, 5, 1024)).astype(np.float32)
    transform = original['process_tensor_idx_rel']
    expected = transform(transform(values[:, :, 1:, :], horizon), denoise)[0]
    model = SimpleNamespace(safe_horizon_idx_rel=horizon, safe_diff_idx_rel=denoise,
                            safe_input_dim=expected.size)
    actual = ns['_stage2_safe_feature'](values, model)
    np.testing.assert_allclose(actual, expected)
    # QAM must retain its original feature regardless of selected SAFE transform.
    np.testing.assert_allclose(ns['_stage2_context'](values), values[0, 0, 1:].mean(axis=0))


@pytest.mark.parametrize('unit,expected_length', [('environment_step', 150), ('chunk', 600)])
def test_loader_uses_trained_dimension_task_alpha_and_time_unit(tmp_path, unit, expected_length):
    OmegaConf.save(OmegaConf.create(dict(input_dim=4096, horizon_idx_rel='concat-2',
        diff_idx_rel='concat-2', model={'name': 'lstm'}, cp_band_time_unit=unit,
        selected_cp_alpha_by_task={'simpler_stack_cube': .15})), tmp_path / 'config.yaml')
    np.save(tmp_path / 'cp_bands_by_task.npy', {'simpler_stack_cube': {.15: np.arange(150) / 150}})
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, state): assert state == {'ours': True}
    requested = []
    def model_factory(cfg, dim): requested.append(dim); return Model()
    ns = extract(SIMPLER / 'rl2_utils.py', {'load_failure_detection_model'},
        dict(np=np, os=os, OmegaConf=OmegaConf, get_model_safe=model_factory,
             torch=SimpleNamespace(load=lambda *args, **kw: {'ours': True})))
    cfg = SimpleNamespace(failure_checkpoint_dir=str(tmp_path), use_taskwise_cp_band=True,
                          task_suite_name='simpler_stack_cube', failure_cp_alpha=.2, n_action_steps=4)
    model, band = ns['load_failure_detection_model'](cfg, 'widowx_stack_cube')
    assert requested == [4096]
    assert model.safe_selected_cp_alpha == .15
    assert model.safe_horizon_idx_rel == model.safe_diff_idx_rel == 'concat-2'
    assert len(band) == expected_length
    if unit == 'environment_step': np.testing.assert_array_equal(band, np.arange(150) / 150)


def test_safe_rejects_feature_dimension_mismatch():
    ns = extract(SIMPLER / 'run_simpler_eval_with_openpi.py', {'_stage2_safe_feature'}, {'np': np})
    with pytest.raises(ValueError, match='mismatch'):
        ns['_stage2_safe_feature'](np.zeros((1, 10, 5, 1024)), SimpleNamespace(safe_input_dim=4096))
