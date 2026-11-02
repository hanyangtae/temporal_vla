"""Online MIL failure gate for the fold27 serve (adapter only; task_classification sources are untouched).

Contract (task_classification 33e5e04, PR11; examples/{load_model,score_test_episodes}.py,
train/train_mil_pooling_cell.py, script/build_grid_cell_shards.py LAYER, DENOISE = 12, 3):
  hidden  = DiT transformer_blocks[12] output at the LAST denoise call (k=3), all 49 tokens, stored fp16
  x       = token_mean: float32 mean over 49 tokens -> float16 -> float32   (MilPoolingTrainer._token_mean)
  z       = ((x - mu) / sd - z_mu) / z_sd                                    (fit_transform/apply_transform, raw)
  w_F     = sigmoid(net.frame_logits(z))     per inference step; attention a is training-only, never used here
  fired   = w_F > fire_threshold (0.5)       evaluated every step (per-step gate, same as SAFE serve)
The frame score has no recurrent state, so step t depends only on step t's hidden -> causal by construction.
`first_fire_t` is kept for the MIL "first crossing" metric (fires.tsv) but does not latch the gate.
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[4]
TC = R / 'task_classification'
EXPECT_TC_COMMIT = '33e5e0495d8918b19e87da105bff735d2d80461d'
LAYER, DENOISE, N_TOKENS, DIM = 12, 3, 49, 1536


class MilContractError(RuntimeError):
    pass


def _import_model():
    if str(TC) not in sys.path:
        sys.path.insert(0, str(TC))
    from failure_detection_multi_instance_learning.model.mil_pooling_model import MilPoolingModel
    return MilPoolingModel


def tc_commit():
    """HEAD of the task_classification checkout (plain .git dir or submodule gitfile)."""
    g = TC / '.git'
    if g.is_file():
        g = (TC / g.read_text().split(':', 1)[1].strip()).resolve()
    return (g / 'HEAD').read_text().strip()


def token_mean(feature):
    """[N, 49, D] (any float) -> [N, D] float32, identical rounding to the trainer/shard."""
    f = np.asarray(feature)
    return f.astype(np.float32).mean(1).astype(np.float16).astype(np.float32)


def apply_transform(tf, x):
    if tf.get('kind', 'raw') != 'raw':
        raise MilContractError(f'only raw transform supported, got {tf.get("kind")}')
    z = (x - tf['mu']) / tf['sd']
    return ((z - tf['z_mu']) / tf['z_sd']).astype(np.float32)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_checkpoint(ck, *, require_target_jitter=None):
    """Fail closed on anything outside the PR11 attn/raw contract."""
    if ck.get('pooling') != 'attn' or ck.get('input') != 'raw':
        raise MilContractError(f'expected attn/raw, got {ck.get("pooling")}/{ck.get("input")}')
    tf = ck['transform']
    for k in ('mu', 'sd', 'z_mu', 'z_sd'):
        if np.asarray(tf[k]).shape != (DIM,):
            raise MilContractError(f'transform {k} shape {np.asarray(tf[k]).shape}')
    thr = ck['config']['fire_threshold']
    if not (0 < float(thr) < 1):
        raise MilContractError(f'bad fire_threshold {thr}')
    if require_target_jitter is not None and list(ck['cell']['target_jitter']) != [int(require_target_jitter)]:
        raise MilContractError(f'target_jitter {ck["cell"]["target_jitter"]} != [{require_target_jitter}]')
    return float(thr)


class MilGate:
    """Same surface as src.failure_online.OnlineFailureDetector as used by serve_safe_qam.py."""

    def __init__(self, net, ck, model_path, threshold=None):
        import torch
        self._torch = torch
        self.net, self.ck = net.eval(), ck
        self.tf = {k: np.asarray(v, dtype=np.float32) for k, v in ck['transform'].items() if k in ('mu', 'sd', 'z_mu', 'z_sd')}
        self.tf['kind'] = ck['transform'].get('kind', 'raw')
        ck_thr = validate_checkpoint(ck)
        if threshold is not None and float(threshold) != ck_thr:
            raise MilContractError(f'threshold override {threshold} != checkpoint {ck_thr}; not allowed')
        self.threshold = ck_thr
        self.model_sha256 = sha256(model_path)
        self.reset()

    @classmethod
    def from_run(cls, run_dir, *, expected_sha256=None, require_target_jitter=None):
        import torch
        run_dir = Path(run_dir)
        path = run_dir / 'model.pt' if run_dir.is_dir() else run_dir
        if expected_sha256 is not None and sha256(path) != expected_sha256:
            raise MilContractError(f'model.pt sha mismatch {path}')
        net = _import_model().load_model(path, device='cpu')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint(ck, require_target_jitter=require_target_jitter)
        return cls(net, ck, path)

    # -- feature
    def feature_from_hidden(self, hidden, layer_idx=None):
        """hidden from serve hook on transformer_blocks[12]: [1,4,49,1536] or [4,49,1536] (fp16).
        `layer_idx` kept for interface parity with OnlineFailureDetector; only layer-12 input is accepted."""
        a = np.asarray(hidden)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.shape != (4, N_TOKENS, DIM):
            raise MilContractError(f'hidden shape {a.shape} != (4,{N_TOKENS},{DIM}) (layer {LAYER}, all denoise)')
        return token_mean(a[DENOISE][None])[0]

    def scores(self, x):
        """[T, D] token-mean features -> [T] w_F (batch; reference for the online path)."""
        with self._torch.no_grad():
            z = self._torch.from_numpy(apply_transform(self.tf, np.asarray(x, np.float32)))
            return self._torch.sigmoid(self.net.frame_logits(z)).numpy()

    # -- online state
    def reset(self):
        self._t = 0
        self._first_fire = None

    def step(self, feat):
        w = float(self.scores(np.asarray(feat, np.float32)[None])[0])
        fired = w > self.threshold
        if fired and self._first_fire is None:
            self._first_fire = self._t
        rec = {'score': w, 'fired': bool(fired), 'delta': self.threshold, 't': self._t, 'first_fire_t': self._first_fire}
        self._t += 1
        return rec

    def snapshot(self):
        return {'_t': self._t, '_first_fire': self._first_fire}

    def restore(self, snap):
        self._t, self._first_fire = int(snap['_t']), snap['_first_fire']

    def spec(self):
        c = self.ck['cell']
        return {'kind': 'mil_attn_raw', 'model_sha256': self.model_sha256, 'threshold': self.threshold,
                'layer': LAYER, 'denoise': DENOISE, 'token_pool': 'mean49_f32_f16_f32', 'gate': 'per_step w_F>thr',
                'cells': c.get('cells'), 'target_jitter': c.get('target_jitter'), 'train_jitter': c.get('train_jitter'),
                'seed': self.ck.get('seed'), 'best': self.ck.get('best'), 'train': self.ck['config'].get('train'),
                'tc_commit_expected': EXPECT_TC_COMMIT}
