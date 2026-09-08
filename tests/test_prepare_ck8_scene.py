import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.grid_phase.prepare_ck8_scene import prepare
from src.failure_online.cluster_phase import _synthetic_bundle


def test_prepare_streams_layer12_last_denoise_and_preserves_gt(tmp_path):
    bundle = tmp_path / "bundle.npz"
    _synthetic_bundle(bundle, dim=1536, hidden=4, latent=2, k=8,
                      slugs=("Demo",), denoise_index=3)
    n, layers, tokens = 5, [0, 2, 4, 8, 10, 12, 15], 4
    x = np.zeros((n, len(layers), 4, tokens, 1536), dtype=np.float32)
    x[:, layers.index(12), 3] = 99.0
    x[:, layers.index(12), 3, 3] = 2.0
    # Other layers/steps/tokens must not leak into the compact result.
    x[:, 0, 0] = 99.0
    shard = tmp_path / "Demo__s0.npz"
    meta = {"capture_layers": layers, "segment_names": ["state", "future", "action", "all"],
            "phase_codebook": {"gt": 0}}
    np.savez_compressed(
        shard, X=x, ep_id=np.repeat(np.arange(2), [2, 3]),
        scene=np.zeros(n, dtype=np.int16), noise=np.array([0, 0, 1, 1, 1], dtype=np.int16),
        jitter=np.zeros(n, dtype=np.int16), rec_idx=np.array([0, 1, 0, 1, 2]),
        succ=np.array([1, 1, 0, 0, 0], dtype=np.int8),
        ep_len=np.array([2, 2, 3, 3, 3], dtype=np.int16),
        phase_code=np.array([4, 4, 5, 6, 6], dtype=np.int16),
        meta_json=np.asarray(json.dumps(meta)))
    out = tmp_path / "prepared.npz"
    result = prepare(shard, bundle, "Demo", out, batch=2)
    assert result["rows"] == n
    with np.load(out, allow_pickle=False) as z:
        got = z["X"]
        assert got.shape == (n, 1, 4, 1, 1536)
        assert np.all(got[:, 0, :3] == 0)
        assert np.all(got[:, 0, 3, 0] == 2)
        np.testing.assert_array_equal(z["gt_phase_code"], [4, 4, 5, 6, 6])
        assert set(json.loads(str(z["meta_json"].item()))["phase_codebook"]) == {f"c{i}" for i in range(8)}


def test_prepare_rejects_fortran_x(tmp_path):
    bundle = tmp_path / "bundle.npz"
    _synthetic_bundle(bundle, dim=1536, hidden=4, latent=2, k=8,
                      slugs=("Demo",), denoise_index=3)
    x = np.asfortranarray(np.zeros((1, 7, 4, 49, 1536), dtype=np.float32))
    meta = json.dumps({"capture_layers": [0, 2, 4, 8, 10, 12, 15], "segment_names": ["all"]})
    shard = tmp_path / "bad.npz"
    np.savez(shard, X=x, ep_id=[0], scene=[0], noise=[0], jitter=[0],
             rec_idx=[0], succ=[0], ep_len=[1], phase_code=[0], meta_json=meta)
    try:
        prepare(shard, bundle, "Demo", tmp_path / "out.npz")
    except ValueError as e:
        assert "Fortran" in str(e)
    else:
        raise AssertionError("Fortran X should be rejected")
