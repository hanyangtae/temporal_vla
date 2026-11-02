"""CPU tests for eval/runtime (fold27 online MIL + QAM/TRQAM evaluator). No GPU, no docker, no network.

    .venvs/rl2_iid/bin/python -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_eval_runtime.py -v
torch-dependent tests skip (with reason) when torch is absent; a skip is not a pass.
"""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path

RT = Path(__file__).resolve().parents[1] / 'eval' / 'runtime'
sys.path.insert(0, str(RT))
import common as C            # noqa: E402
import build_manifest as B    # noqa: E402
import run_eval_arm as RA     # noqa: E402
import launch_group as LG     # noqa: E402
import restore_actor as RS    # noqa: E402

HAS_TORCH = importlib.util.find_spec('torch') is not None
SPEC = json.loads((C.R / C.FOLD_SPEC).read_text())
FOLD = next(f for f in SPEC['folds'] if f['fold_id'] == 'OpenDrawer-left_s2_j2')
SRV_FOLD = next(f for f in SPEC['folds'] if f['eval_host'] == 'srv50')


# ====================================================================== fake GR00T head (CPU)
if HAS_TORCH:
    import numpy as np
    import torch
    from torch import nn
    sys.path.insert(0, str(C.ADAPTER))
    from composition import SPEC as FSPEC, ComposedFlowSampler
    import gated_action as GA

    class BF(dict):
        def __init__(self, data=None):
            super().__init__(data or {})

        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e

    class Block(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.scale = nn.Parameter(torch.full((1536,), 0.9 + 0.01 * i)); self.bias = nn.Parameter(torch.randn(1536) * 0.01)

        def forward(self, h, enc, temb):
            return torch.tanh(h * self.scale + self.bias + enc.mean(1, keepdim=True) + temb[:, None])

    class FakeDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(Block(i) for i in range(16)); self.proj = nn.Linear(1536, 1024)

        def forward(self, hidden_states, encoder_hidden_states, timestep):
            temb = (timestep.float() / 1000.0)[:, None].expand(-1, 1536)
            h = hidden_states
            for blk in self.transformer_blocks:
                h = blk(h, encoder_hidden_states, temb)
            return self.proj(h)

    class Dec(nn.Module):
        def __init__(self):
            super().__init__(); self.dec = nn.Linear(1024, 32)

        def forward(self, x, emb):
            return self.dec(x)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__(); self.lin = nn.Linear(32, 1536)

        def forward(self, actions, t, emb):
            return self.lin(actions) + (t.float() / 1000.0)[:, None, None]

    class FakeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(action_horizon=16, action_dim=32, add_pos_embed=True, input_embedding_dim=1536, hidden_size=1024)
            self.num_inference_timesteps, self.num_timestep_buckets, self.action_horizon = 4, 1000, 16
            self.state_lin = nn.Linear(64, 1536); self.position_embedding = nn.Embedding(16, 1536)
            self.future_tokens = nn.Embedding(32, 1536); self.model = FakeDiT(); self.action_decoder = Dec(); self.action_encoder = Enc()

        def process_backbone_output(self, bo):
            bo['backbone_features'] = bo['backbone_features'] * 1.0
            return bo

        def state_encoder(self, state, emb):
            return self.state_lin(state)

    def upstream_get_action():
        path = C.R / 'lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py'
        text = path.read_text()
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == 'get_action')
        ns = {'torch': torch, 'BatchFeature': BF}
        exec(textwrap.dedent(ast.get_source_segment(text, node)), ns)
        return ns['get_action']

    def inputs(seed=0):
        g = torch.Generator().manual_seed(seed)
        return BF({'backbone_features': torch.randn(1, 7, 1536, generator=g)}), \
            BF({'state': torch.randn(1, 1, 64, generator=g), 'embodiment_id': torch.zeros(1, dtype=torch.long)})

    class CountingQAM:
        def __init__(self, value=0.3):
            self.calls, self.value = [], value

        def __call__(self, feature, noisy, t):
            self.calls.append(int(t))
            return np.full(noisy.shape, self.value, np.float32)

    class ScriptGate:
        """MilGate surface with a scripted fire sequence; records every feature it was asked to score."""
        def __init__(self, fires, consume_rng=False):
            self.fires, self.consume_rng, self.features, self.resets = list(fires), consume_rng, [], 0
            self.reset()

        def reset(self):
            self._t, self._first = 0, None; self.resets += 1

        def feature_from_hidden(self, hidden, _layer=None):
            import mil_gate as G
            return G.token_mean(np.asarray(hidden)[0][3][None])[0]

        def step(self, feat):
            if self.consume_rng:
                torch.rand(1)
            self.features.append(np.asarray(feat).copy())
            fired = bool(self.fires[min(self._t, len(self.fires) - 1)])
            if fired and self._first is None:
                self._first = self._t
            r = {'score': 0.9 if fired else 0.1, 'fired': fired, 't': self._t, 'first_fire_t': self._first, 'delta': .5}
            self._t += 1
            return r

        def snapshot(self):
            return (self._t, self._first)

        def restore(self, s):
            self._t, self._first = s


@unittest.skipUnless(HAS_TORCH, 'torch not installed in this python')
class GatedActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.head = FakeHead().eval()
        cls.up = staticmethod(upstream_get_action())

    def native(self):
        return lambda bo, ai: self.up(self.head, bo, ai)

    def reference(self, seed, n=3):
        """Plain upstream get_action sequence: actions + RNG after each call + block12 d3 hidden."""
        torch.manual_seed(seed)
        out, rng, hid = [], [], []
        for i in range(n):
            buf = []
            h = self.head.model.transformer_blocks[12].register_forward_hook(lambda m, a, o: buf.append(o.detach().float().numpy().astype(np.float16)[0]))
            bo, ai = inputs(i)
            out.append(self.up(self.head, bo, ai)['action_pred']); rng.append(torch.random.get_rng_state()); hid.append(buf[3]); h.remove()
        return out, rng, hid

    def run_ga(self, mode, fires, seed=7, n=3, qam=None, gate=None):
        qam = qam or CountingQAM()
        gate = gate or ScriptGate(fires)
        recs = []
        ga = GA.GatedAction(self.head, self.native(), gate, ComposedFlowSampler(FSPEC, qam_fn=None if mode == 'base' else qam, qam_weight=.5),
                            mode, use_cuda=False, emit=recs.append)
        torch.manual_seed(seed)
        out, rng = [], []
        for i in range(n):
            bo, ai = inputs(i)
            out.append(ga(bo, ai)['action_pred']); rng.append(torch.random.get_rng_state())
        return ga, gate, qam, out, rng, recs

    def test_ungated_is_exact_vanilla_and_rng_untouched(self):
        ref, ref_rng, _ = self.reference(7)
        for mode in ('base', 'gated'):
            ga, gate, qam, out, rng, recs = self.run_ga(mode, [0, 0, 0])
            for i in range(3):
                self.assertTrue(torch.equal(out[i], ref[i]), mode)
                self.assertTrue(torch.equal(rng[i], ref_rng[i]), mode)
            self.assertEqual(qam.calls, [])
            self.assertTrue(all(not r['gate'] and r['noise_sha256_composed'] is None and r['qam_delta_max_normalized'] == 0 for r in recs))
            self.assertTrue(ga.state['off_identity_verified'])

    def test_gate_called_every_inference_on_first_pass_hidden(self):
        _, _, hid = self.reference(7)
        import mil_gate as G
        ga, gate, qam, out, rng, recs = self.run_ga('gated', [0, 1, 0])
        self.assertEqual(ga.state['requests'], 3)
        self.assertEqual(ga.state['gate_calls'], 3)
        self.assertEqual([r['mil']['t'] for r in recs], [0, 1, 2])
        pre = [gate.features[0], gate.features[1], gate.features[3]]          # [1] fired -> one bookkeeping post step at index 2
        self.assertEqual(len(gate.features), 4)
        for i in range(3):
            self.assertTrue(np.array_equal(pre[i], G.token_mean(hid[i][None])[0]), i)   # L12 D3 token mean of the vanilla pass

    def test_gated_composes_from_same_initial_noise(self):
        ref, ref_rng, _ = self.reference(7)
        ga, gate, qam, out, rng, recs = self.run_ga('gated', [0, 1, 0])
        self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[2], ref[2]))   # ungated rows stay vanilla
        self.assertFalse(torch.equal(out[1], ref[1]))
        for i in range(3):
            self.assertTrue(torch.equal(rng[i], ref_rng[i]), f'RNG drift after request {i}')
        self.assertEqual(qam.calls, [0, 1, 2, 3])
        r = recs[1]
        self.assertTrue(r['gate'] and r['noise_sha256_composed'] == r['noise_sha256_first_pass'] and r['first_fire_t'] == 1)
        # independent recomputation: same RNG position -> ComposedFlowSampler(gate=1) directly
        torch.manual_seed(7)
        self.up(self.head, *inputs(0))
        direct, _ = ComposedFlowSampler(FSPEC, qam_fn=CountingQAM(), qam_weight=.5).sample(self.head, *inputs(1), decision_env_step=5, gate=1.)
        self.assertTrue(torch.equal(direct, out[1]))

    def test_always_mode_and_qam_steps(self):
        ga, gate, qam, out, rng, recs = self.run_ga('always', [0, 0, 0])
        self.assertEqual(qam.calls, [0, 1, 2, 3] * 3)
        self.assertEqual(ga.state['composed_calls'], 3)

    def test_reset_clears_gate_state(self):
        ga, gate, qam, out, rng, recs = self.run_ga('gated', [0, 1, 1])
        self.assertEqual(ga.state['first_fire_t'], 1)
        r0 = gate.resets
        ga.reset()
        self.assertEqual(gate.resets, r0 + 1)
        self.assertEqual(gate._t, 0)
        self.assertIsNone(ga.state['first_fire_t'])
        self.assertEqual(ga.state['episode_resets'], 1)

    def test_rng_consuming_gate_fails_closed(self):
        with self.assertRaises(GA.GateRuntimeError):
            self.run_ga('gated', [0], n=1, gate=ScriptGate([0], consume_rng=True))

    def test_mode_requires_actor(self):
        with self.assertRaises(GA.GateRuntimeError):
            GA.GatedAction(self.head, self.native(), ScriptGate([0]), ComposedFlowSampler(FSPEC, qam_fn=None, qam_weight=.5), 'gated', use_cuda=False)

    def test_real_mil_gate_scores_match_batch(self):
        import mil_gate as G
        sys.path.insert(0, str(G.TC))
        from failure_detection_multi_instance_learning.model.mil_pooling_model import MilPoolingModel
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(0)
            net = MilPoolingModel(1536, 'attn')
            rng = np.random.default_rng(2)
            tf = {'kind': 'raw', 'mu': rng.normal(size=1536).astype(np.float32), 'sd': np.abs(rng.normal(size=1536)).astype(np.float32) + .5,
                  'z_mu': np.zeros(1536, np.float32), 'z_sd': np.ones(1536, np.float32)}
            net.save_model(Path(d) / 'model.pt', config={'fire_threshold': 0.5, 'train': {}}, transform=tf, input='raw',
                           cell={'cells': ['x'], 'target_jitter': [2], 'train_jitter': [0, 1, 3, 4]}, seed=0, best={})
            gate = G.MilGate.from_run(Path(d))
            _, _, hid = self.reference(7)
            ga, _, _, _, _, recs = self.run_ga('gated', None, gate=gate)
            batch = gate.scores(np.stack([G.token_mean(h[None])[0] for h in hid]))
            self.assertTrue(np.allclose([r['mil']['score'] for r in recs], batch, atol=1e-6))
            self.assertEqual([r['gate'] for r in recs], [bool(s > .5) for s in batch])


# ====================================================================== manifest / pins / seeds (stdlib)
class Sandbox:
    """Temporary MIL run + actor dirs inside the repo (gitignored outputs/tmp) so manifests can pin repo-relative paths."""
    def __init__(self):
        base = C.R / 'outputs/tmp'
        base.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix='f27rt_', dir=base))

    def fresh(self, *parts):
        d = self.dir.joinpath(*parts)
        k = 0
        while d.exists():
            k += 1; d = self.dir.joinpath(*parts[:-1], f'{parts[-1]}.{k}')
        d.mkdir(parents=True)
        return d

    def mil(self, fold, contract=True, target_j=None):
        d = self.fresh('mil', fold['fold_id'])
        (d / 'model.pt').write_bytes(b'fake-mil-' + fold['fold_id'].encode())
        if contract:
            (d / 'FOLD_CONTRACT.json').write_text(json.dumps({'fold_id': fold['fold_id'], 'target_j': fold['target_j'] if target_j is None else target_j,
                                                              'model_sha256': C.sha256_path(d / 'model.pt'), 'shared_checkpoint_reused': False,
                                                              'tc_commit_expected': C.TC_COMMIT}))
        return d

    def actor(self, name, kind, fold_id=None, production=True, steps=C.COMMON_STEPS, diagnostic=None):
        d = self.fresh('actors', name)
        (d / 'actor.npz').write_bytes(name.encode())
        (d / 'actor.json').write_text(json.dumps({'diagnostic_only': (not production) if diagnostic is None else diagnostic, 'flags': {'data_label': 'x'}}))
        prov = {'kind': kind, 'common_steps': steps, 'production': production, 'smoke': not production, 'actor_sha256': C.sha256_path(d / 'actor.npz')}
        if fold_id:
            prov['fold_id'] = fold_id
        (d / 'runtime_provenance.json').write_text(json.dumps(prov))
        return d

    def production_actors(self, fold):
        return {'mil_qam65': self.actor('common', 'common'), 'mil_trqam_basic': self.actor(f'{fold["fold_id"]}_b', 'trqam_basic', fold['fold_id']),
                'mil_trqam_pbrs': self.actor(f'{fold["fold_id"]}_p', 'trqam_pbrs', fold['fold_id']),
                'mil_trqam_mix': self.actor(f'{fold["fold_id"]}_m', 'trqam_mix', fold['fold_id'])}

    def spec_with(self, mutate):
        s = copy.deepcopy(SPEC)
        mutate(next(f for f in s['folds'] if f['fold_id'] == FOLD['fold_id']))
        p = self.dir / f'spec_{len(list(self.dir.glob("spec_*")))}.json'
        p.write_text(json.dumps(s))
        return p

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.addCleanup(self.sb.close)

    def build(self, fold=FOLD, **kw):
        out = kw.pop('out', self.sb.dir / 'eval')
        actors = kw.pop('actors', None) or self.sb.production_actors(fold)
        mil = kw.pop('mil', None) or self.sb.mil(fold)
        return B.build(fold['fold_id'], mil, actors, out, **kw)

    def test_same_40_rows_seeds_plans(self):
        actors, mil = self.sb.production_actors(FOLD), self.sb.mil(FOLD)
        path, m = self.build(actors=actors, mil=mil)
        self.assertEqual([r['n'] for r in m['rows']], list(range(40)))
        ev = {r['n']: r for r in FOLD['eval']}
        for r in m['rows']:
            self.assertEqual((r['env_seed'], r['inference_seed'], r['plan_id'], r['j'], r['sig'], r['base_success']),
                             (ev[r['n']]['env_seed'], ev[r['n']]['inference_seed'], ev[r['n']]['plan_id'], FOLD['target_j'],
                              ev[r['n']]['sig'], ev[r['n']]['success']))
        self.assertEqual(set(m['plans']), {FOLD['source_plan_n0_9'], '748863ce9233', '2f26fb046fc4'})
        for pid, p in m['plans'].items():
            self.assertEqual(p['sha256'], C.sha256_path(C.R / p['path']))
        self.assertEqual(m['host'], 'kanu'); self.assertEqual(m['identity']['lang'], 'Open the left drawer.')
        self.assertEqual(m['identity']['layout'], 5)
        self.assertEqual(set(m['arms']), set(C.PRODUCTION_ARMS))
        self.assertTrue(all(a['mode'] == 'gated' for a in m['arms'].values()))
        B.build(FOLD['fold_id'], mil, actors, self.sb.dir / 'eval')     # identical rebuild accepted (write-once)
        RA.verify_manifest(path, 'mil_qam65')                         # host-side file/sha/plan/tc checks

    def test_all_27_folds_build_on_their_archive_host(self):
        hosts = {}
        for f in SPEC['folds']:
            _, m = self.build(fold=f)
            hosts[m['host']] = hosts.get(m['host'], 0) + 1
            self.assertEqual(len(m['rows']), 40)
        self.assertEqual(hosts, {'kanu': 6, 'srv48': 9, 'srv50': 12})

    def test_missing_or_wrong_sha_refused(self):
        actors = self.sb.production_actors(FOLD)
        del actors['mil_trqam_mix']
        with self.assertRaises(C.RuntimeContractError):
            self.build(actors=actors)
        a = self.sb.actor('bad', 'common'); (a / 'actor.npz').write_bytes(b'tampered')
        acts = {'mil_qam65': a, **{k: self.sb.actor(k + 'x', v[1], FOLD['fold_id']) for k, v in C.PRODUCTION_ARMS.items() if k != 'mil_qam65'}}
        with self.assertRaises(C.RuntimeContractError):
            self.build(actors=acts)
        with self.assertRaises(C.RuntimeContractError):     # shared48 / no fold contract
            self.build(mil=self.sb.mil({'fold_id': 'nocontract', 'target_j': 2}, contract=False))

    def test_production_refuses_smoke_and_wrong_fold_actor(self):
        acts = self.sb.production_actors(FOLD)
        acts['mil_qam65'] = self.sb.actor('smoke20', 'smoke', production=False, steps=20)
        with self.assertRaises(C.RuntimeContractError):
            self.build(actors=acts)
        acts = self.sb.production_actors(FOLD)
        acts['mil_trqam_basic'] = self.sb.actor('other_b', 'trqam_basic', 'OpenDrawer-left_s2_j4')
        with self.assertRaises(C.RuntimeContractError):
            self.build(actors=acts)
        with self.assertRaises(C.RuntimeContractError):     # MIL contract for another target j
            self.build(mil=self.sb.mil({'fold_id': FOLD['fold_id'] + 'x', 'target_j': 2}, target_j=4))

    def test_seed_and_plan_fail_closed(self):
        def env_seed(f): f['eval'][3]['env_seed'] += 1
        def inf_seed(f): f['eval'][25]['inference_seed'] += 1
        def plan(f): f['eval'][12]['plan_id'] = '2f26fb046fc4'
        def src(f): f['eval'][0]['seed_source'] = 'derived'
        def missing(f): f['eval'][5]['inference_seed'] = None
        for mut in (env_seed, inf_seed, plan, src, missing):
            with self.assertRaises((C.RuntimeContractError, Exception), msg=mut.__name__):
                self.build(spec_path=self.sb.spec_with(mut), out=self.sb.dir / f'eval_{mut.__name__}')

    def test_smoke_location_and_rows(self):
        with self.assertRaises(C.RuntimeContractError):     # production cannot use a subset
            self.build(rows_subset=[0])
        smoke_actor = self.sb.actor('smoke20', 'smoke', production=False, steps=20)
        with self.assertRaises(C.RuntimeContractError):     # smoke outside SMOKE_ROOT
            self.build(smoke=True, rows_subset=[0], actors={'smoke_gated': smoke_actor})
        root = C.R / C.SMOKE_ROOT / ('unittest_' + self.sb.dir.name)
        self.addCleanup(shutil.rmtree, root, True)
        path, m = self.build(smoke=True, rows_subset=[0, 10, 20], out=root,
                             actors={'smoke_base': None, 'smoke_gated': smoke_actor}, mil=self.sb.mil(FOLD, contract=False))
        self.assertEqual([r['n'] for r in m['rows']], [0, 10, 20])
        RA.verify_manifest(path, 'smoke_gated')
        with self.assertRaises(C.RuntimeContractError):
            RA.verify_manifest(path, 'mil_qam65')
        sys.argv = ['x', '--manifest', str(path)]
        import verify_fold
        with self.assertRaises(C.RuntimeContractError):
            verify_fold.main(['--manifest', str(path)])


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(); self.addCleanup(self.sb.close)
        self.path, self.m = B.build(FOLD['fold_id'], self.sb.mil(FOLD), self.sb.production_actors(FOLD), self.sb.dir / 'eval')
        self.out = self.path.parent / 'mil_qam65'

    def finish_row(self, n, attempt='a0', success=1):
        row = self.m['rows'][n]
        g = self.out / 'rows' / f'n{n:02d}' / attempt / 'grid' / 'x' / f's{self.m["scene"]}' / f'j{row["j"]}' / f'n{n}' / 'mil_qam65'
        g.mkdir(parents=True)
        (g / 'meta.json').write_text(json.dumps({'episode_success': success, 'steps': 100, 'inference_seed': row['inference_seed']}))
        (g / 'traj.csv').write_text('a,b\n1,2\n')
        item = {'fold_id': self.m['fold_id'], 'arm': 'mil_qam65', 'n': n, 'attempt': attempt, 'env_seed': row['env_seed'],
                'inference_seed': row['inference_seed'], 'meta': str((g / 'meta.json').relative_to(self.out)),
                'files_sha256': RA.row_files_sha(g / 'meta.json')}
        with (self.out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(item) + '\n')
        return g

    def test_resume_runs_missing_only(self):
        self.out.mkdir(parents=True)
        for n in (0, 1, 7, 39):
            self.finish_row(n)
        (self.out / 'rows/n02/a0').mkdir(parents=True)          # crashed attempt without meta -> kept, new attempt next
        done = RA.load_progress(self.out, self.m, 'mil_qam65')
        self.assertEqual(sorted(done), [0, 1, 7, 39])
        self.assertEqual([r['n'] for r in RA.pending_rows(self.m, done)], [n for n in range(40) if n not in (0, 1, 7, 39)])
        self.assertEqual(RA.next_attempt(self.out, 2).name, 'a1')
        self.assertEqual(RA.next_attempt(self.out, 3).name, 'a0')

    def test_tampered_finished_row_refused(self):
        self.out.mkdir(parents=True)
        g = self.finish_row(4)
        (g / 'traj.csv').write_text('changed\n')
        with self.assertRaises(C.RuntimeContractError):
            RA.load_progress(self.out, self.m, 'mil_qam65')

    def test_orphan_duplicate_and_seed_drift_refused(self):
        self.out.mkdir(parents=True)
        self.finish_row(5)
        orphan = self.out / 'rows/n06/a0/grid/m'; orphan.mkdir(parents=True); (orphan / 'meta.json').write_text('{}')
        with self.assertRaises(C.RuntimeContractError):
            RA.load_progress(self.out, self.m, 'mil_qam65')
        shutil.rmtree(self.out / 'rows/n06')
        line = (self.out / 'progress.jsonl').read_text()
        (self.out / 'progress.jsonl').write_text(line + line)
        with self.assertRaises(C.RuntimeContractError):
            RA.load_progress(self.out, self.m, 'mil_qam65')
        d = json.loads(line); d['inference_seed'] += 1
        (self.out / 'progress.jsonl').write_text(json.dumps(d) + '\n')
        with self.assertRaises(C.RuntimeContractError):
            RA.load_progress(self.out, self.m, 'mil_qam65')

    def test_code_change_refused(self):
        m = json.loads(self.path.read_text())
        m['code']['runtime/gated_action.py'] = '0' * 64
        with self.assertRaises(C.RuntimeContractError):
            B.check_code_hashes(m['code'])


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(); self.addCleanup(self.sb.close)

    def test_kanu_and_srv_commands(self):
        has_np = importlib.util.find_spec('numpy') is not None     # kanu docker helpers (adapter) import numpy
        for fold in ((FOLD, SRV_FOLD) if has_np else (SRV_FOLD,)):
            path, m = B.build(fold['fold_id'], self.sb.mil(fold), self.sb.production_actors(fold), self.sb.dir / 'eval')
            m['_arm'] = 'mil_trqam_pbrs'
            out = path.parent / 'mil_trqam_pbrs'; att = out / 'rows/n25/a0'
            row = m['rows'][25]
            cmd = RA.collect_command(m, row, out, att, 1, 8890, 'c', m['host'], 'msha')
            val = lambda k: cmd[cmd.index(k) + 1]
            self.assertEqual((val('--seed'), val('--inference-seed'), val('--noise-idx'), val('--jitter-idx'), val('--scene-idx')),
                             (str(row['env_seed']), str(row['inference_seed']), '25', str(fold['target_j']), str(fold['scene'])))
            self.assertEqual(val('--plan-json'), '/temporal_vla/' + C.PLAN_FILES['2f26fb046fc4'])
            self.assertEqual((val('--n-action-steps'), val('--expect-chunk-len'), val('--max-episode-steps')), ('5', '16', '720'))
            self.assertEqual(val('--grid-instruction'), fold['instruction'])
            self.assertEqual(val('--canonical-instruction'), m['identity']['lang'])
            serve, env = RA.serve_command(m, 'mil_trqam_pbrs', out, 1, 8890, 'srv', m['host'], C.SRV_SERVE_PY)
            self.assertEqual(serve[serve.index('--mode') + 1], 'gated')
            self.assertNotIn('--smoke', serve)
            if m['host'] == 'kanu':
                self.assertEqual(cmd[:2], ['docker', 'run']); self.assertIsNone(env)
                self.assertIn('--log-driver', serve)
            else:
                self.assertEqual(cmd[:3], ['docker', 'exec', '-w']); self.assertIn('robocasa', cmd)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
                self.assertTrue(serve[0].endswith('envs/lerobot_050_groot/bin/python'))

    def test_slot_caps_and_host_jobs(self):
        self.assertEqual(len(LG.plan_slots('kanu', [0, 1, 2], 2, 8860)), 6)
        self.assertEqual(len(LG.plan_slots('srv48', [2], 6, 8890)), 6)
        for host, g, s in (('kanu', [0, 1, 2, 3], 2), ('kanu', [0], 3), ('srv50', [1, 2], 6), ('srv48', [2], 7)):
            with self.assertRaises(C.RuntimeContractError):
                LG.plan_slots(host, g, s, 8860)
        path, _ = B.build(FOLD['fold_id'], self.sb.mil(FOLD), self.sb.production_actors(FOLD), self.sb.dir / 'eval')
        with self.assertRaises(C.RuntimeContractError):
            LG.check_jobs([(str(path), 'mil_qam65')], 'srv48')
        self.assertFalse(LG.check_jobs([(str(path), 'mil_qam65')], 'kanu'))

    def test_production_runner_requires_harness(self):
        path, _ = B.build(FOLD['fold_id'], self.sb.mil(FOLD), self.sb.production_actors(FOLD), self.sb.dir / 'eval')
        env = dict(os.environ); os.environ.pop('HARNESS_RECEIPT', None)
        try:
            rc = RA.main(['--manifest', str(path), '--arm', 'mil_qam65', '--gpu', '0', '--port', '8860'])
        finally:
            os.environ.clear(); os.environ.update(env)
        self.assertEqual(rc, 2)
        self.assertTrue(list((path.parent / 'mil_qam65' / 'failures').glob('*.json')))


class RestoreTest(unittest.TestCase):
    def test_smoke_run_is_not_production(self):
        smoke = C.R / C.COMMON_QAM_WORK / 'scratch_smoke'
        if not (smoke / 'COMPLETE.json').exists():
            self.skipTest('smoke run absent')
        with self.assertRaises(C.RuntimeContractError):
            RS.check_common_run(smoke)

    def test_manifest_leakage(self):
        ok = {'experiment': 'atomic65_common_qam_20261029', 'mode': 'full', 'n_tasks': 65,
              'episodes': [{'episode_id': 'robocasa/v1.0/pretrain/atomic/X/episode_000001', 'split': 'train'}]}
        RS.check_manifest_leakage(ok)
        for bad in ({'episode_id': 'robocasa/v1.0/target/atomic/X/episode_000001', 'split': 'train'},
                    {'episode_id': 'robocasa/v1.0/pretrain/atomic/X/episode_000002', 'split': 'test'}):
            m = copy.deepcopy(ok); m['episodes'].append(bad)
            with self.assertRaises(C.RuntimeContractError):
                RS.check_manifest_leakage(m)
        with self.assertRaises(C.RuntimeContractError):
            RS.check_manifest_leakage(dict(ok, mode='subset'))

    def test_fold_leakage_all_folds(self):
        for f in SPEC['folds']:
            RS.check_fold_leakage(f)
        f = copy.deepcopy(FOLD); f['fit'][0] = dict(f['fit'][0], sig=f['eval'][0]['sig'])
        with self.assertRaises(C.RuntimeContractError):
            RS.check_fold_leakage(f)

    def test_serve_actor_meta_gate(self):
        import serve_mil_qam as S
        prod = {'diagnostic_only': False, 'flags': {'data_label': 'official'}, 'runtime_provenance': {'production': True, 'smoke': False, 'common_steps': C.COMMON_STEPS}}
        S.validate_actor_meta(prod, smoke=False)
        for bad in ({'diagnostic_only': True}, {'runtime_provenance': {'production': False, 'smoke': True, 'common_steps': 20}},
                    {'flags': {'data_label': 'x_smoke'}}):
            with self.assertRaises(C.RuntimeContractError):
                S.validate_actor_meta({**prod, **bad}, smoke=False)
        S.validate_actor_meta({'diagnostic_only': True, 'runtime_provenance': {'smoke': True}}, smoke=True)

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None and HAS_TORCH, 'jax/torch absent')
    def test_smoke_actor_export_and_torch_velocity(self):
        smoke = C.R / C.COMMON_QAM_WORK / 'scratch_smoke'
        if not (smoke / 'checkpoints/step_0000020/params_20.pkl').exists():
            self.skipTest('smoke checkpoint absent')
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        with tempfile.TemporaryDirectory(dir=C.R / 'outputs/tmp') as d:
            out = Path(d) / 'smoke_actor'
            RS.main(['smoke', '--run', str(smoke), '--step', '20', '--out', str(out)])
            prov = json.loads((out / 'runtime_provenance.json').read_text())
            self.assertTrue(prov['smoke'] and not prov['production'])
            self.assertTrue(json.loads((out / 'actor.json').read_text())['diagnostic_only'])
            from qam_torch_actor import TorchQAMVelocity
            from qam_prefix import PrefixQAMTrainer
            v = TorchQAMVelocity(out)
            tr = PrefixQAMTrainer.load(smoke / 'checkpoints/step_0000020', 20, feature_signature=C.FEATURE_SIGNATURE)
            rng = np.random.default_rng(0)
            f, x = rng.normal(size=(1, 1024)).astype(np.float32), rng.normal(size=(1, 192)).astype(np.float32)
            for t in range(4):
                self.assertTrue(np.allclose(v(f, x, t), tr.velocity_fn()(f, x, t), atol=1e-4))
            with self.assertRaises(C.RuntimeContractError):     # smoke checkpoint through the production path
                RS.check_common_run(smoke)


# ====================================================================== production connector (eval_pipeline.py)
import eval_pipeline as EP   # noqa: E402

COMMON_RUN = C.R / C.COMMON_QAM_WORK / 'scratch_run'


@unittest.skipUnless((COMMON_RUN / 'COMPLETE.json').exists(), 'common 500k run absent')
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init_sha = C.sha256_path(COMMON_RUN / 'checkpoints/step_0500000/params_500000.pkl')

    def setUp(self):
        self.sb = Sandbox(); self.addCleanup(self.sb.close)
        self.p = EP.Pipeline(root=self.sb.dir / 'prod', mil_root=self.sb.dir / 'mil', trqam_root=self.sb.dir / 'trqam')
        c = self.sb.actor('common_tmp', 'common')
        dst = self.p.actor_dir(None, 'mil_qam65'); dst.parent.mkdir(parents=True); shutil.copytree(c, dst)
        self.old_sha = EP.COMMON_ACTOR_SHA
        EP.COMMON_ACTOR_SHA = C.sha256_path(dst / 'actor.npz')
        self.addCleanup(setattr, EP, 'COMMON_ACTOR_SHA', self.old_sha)

    def mil(self, fold):
        d = self.sb.dir / 'mil' / fold['fold_id'] / 'attn/raw/s0'; d.mkdir(parents=True)
        (d / 'model.pt').write_bytes(b'm' + fold['fold_id'].encode())
        (d / 'FOLD_CONTRACT.json').write_text(json.dumps({'fold_id': fold['fold_id'], 'target_j': fold['target_j'], 'shared_checkpoint_reused': False,
                                                          'model_sha256': C.sha256_path(d / 'model.pt'), 'tc_commit_expected': C.TC_COMMIT}))

    def done(self, fold, arm, smoke=False, steps=50000):
        short = EP.TRQAM_ARMS[arm]
        run = self.p.trqam_run(fold['fold_id'], arm); (run / 'checkpoint').mkdir(parents=True)
        (run / 'checkpoint' / f'params_{steps}.pkl').write_bytes(arm.encode())
        fp = {'fold_id': fold['fold_id'], 'arm': short, 'steps': steps, 'init': {'step': 500000, 'params_sha256': self.init_sha}}
        (run / 'DONE.json').write_text(json.dumps({'fold_id': fold['fold_id'], 'arm': short, 'episodes': 160, 'alpha': None, 'beta': None,
                                                   'harness_receipt': 'kanu:6:x', 'smoke': smoke, 'fingerprint': fp,
                                                   'params_sha256': C.sha256_path(run / 'checkpoint' / f'params_{steps}.pkl')}))
        out = self.p.actor_dir(fold['fold_id'], arm); out.mkdir(parents=True)          # pre-exported (export path = restore_actor, tested above)
        (out / 'actor.npz').write_bytes(b'a' + arm.encode()); (out / 'actor.json').write_text(json.dumps({'diagnostic_only': False, 'flags': {}}))
        (out / 'runtime_provenance.json').write_text(json.dumps({'kind': f'trqam_{short}', 'fold_id': fold['fold_id'], 'common_steps': C.COMMON_STEPS,
            'production': True, 'smoke': False, 'actor_sha256': C.sha256_path(out / 'actor.npz'),
            'params_sha256': C.sha256_path(run / 'checkpoint' / f'params_{steps}.pkl')}))

    def test_per_fold_ready_without_waiting_for_all(self):
        self.mil(FOLD)
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id']), 'waiting')
        for arm in EP.TRQAM_ARMS:
            self.done(FOLD, arm)
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id']), 'ready')
        m = json.loads((self.p.fdir(FOLD['fold_id']) / 'manifest.json').read_text())
        self.assertEqual([r['n'] for r in m['rows']], list(range(40)))
        self.assertEqual(m['arms']['mil_qam65']['actor_sha256'], EP.COMMON_ACTOR_SHA)
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id']), 'ready')               # idempotent, immutable
        self.assertEqual(self.p.jobs('kanu'), [(FOLD['fold_id'], a) for a in EP.ARM_ORDER])
        self.assertEqual(self.p.jobs('srv50'), [])
        other = next(f for f in SPEC['folds'] if f['eval_host'] == 'kanu' and f['fold_id'] != FOLD['fold_id'])
        self.assertEqual(self.p.prepare_fold(other['fold_id']), 'waiting')             # others still waiting
        (self.p.fdir(FOLD['fold_id']) / 'mil_qam65').mkdir(); (self.p.fdir(FOLD['fold_id']) / 'mil_qam65/RUN_COMPLETE.json').write_text('{}')
        EP.fail_marker(self.p.root / 'queue/kanu/failed', f'{FOLD["fold_id"]}.mil_trqam_pbrs', RuntimeError('x'))
        self.assertEqual(self.p.jobs('kanu'), [(FOLD['fold_id'], 'mil_trqam_basic'), (FOLD['fold_id'], 'mil_trqam_mix')])
        self.assertEqual(len(self.p.jobs('kanu', retry_failed=True)), 3)
        st = self.p.write_state()
        self.assertEqual(st['hosts']['kanu'][FOLD['fold_id']], 'evaluating 1/4')
        self.assertEqual({h: len(v) for h, v in st['hosts'].items()}, EP.HOST_FOLDS)
        m2 = self.p.fdir(FOLD['fold_id']) / 'manifest.json'; os.chmod(m2, 0o644); m2.write_text(m2.read_text() + ' ')
        with self.assertRaises(C.RuntimeContractError):
            self.p.ready(FOLD['fold_id'])

    def test_failures_are_explicit_and_sticky(self):
        self.mil(FOLD)
        for arm in EP.TRQAM_ARMS:
            self.done(FOLD, arm, smoke=(arm == 'mil_trqam_mix'))
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id']), 'failed')
        self.assertEqual(len(self.p.prepare_failed(FOLD['fold_id'])), 1)
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id']), 'failed')               # no silent retry
        self.assertEqual(len(self.p.prepare_failed(FOLD['fold_id'])), 1)
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id'], retry_failed=True), 'failed')
        self.assertEqual(len(self.p.prepare_failed(FOLD['fold_id'])), 2)
        self.assertFalse((self.p.fdir(FOLD['fold_id']) / 'READY.json').exists())

    def test_wrong_init_and_partial_export_refused(self):
        self.mil(FOLD)
        for arm in EP.TRQAM_ARMS:
            self.done(FOLD, arm)
        d = json.loads((self.p.trqam_run(FOLD['fold_id'], 'mil_trqam_basic') / 'DONE.json').read_text())
        d['fingerprint']['init']['params_sha256'] = '0' * 64                              # e.g. 10/13-task init
        (self.p.trqam_run(FOLD['fold_id'], 'mil_trqam_basic') / 'DONE.json').write_text(json.dumps(d))
        self.assertEqual(self.p.prepare_fold(FOLD['fold_id']), 'failed')
        f = next(x for x in SPEC['folds'] if x['eval_host'] == 'srv48')
        self.mil(f)
        for arm in EP.TRQAM_ARMS:
            self.done(f, arm)
        os.remove(self.p.actor_dir(f['fold_id'], 'mil_trqam_mix') / 'runtime_provenance.json')
        self.assertEqual(self.p.prepare_fold(f['fold_id']), 'failed')
        self.assertTrue((self.p.actor_dir(f['fold_id'], 'mil_trqam_mix') / 'actor.npz').exists())   # never deleted


class ResourceWaitTest(unittest.TestCase):
    LIVE = {'a': {'machine': 'kanu', 'gpus': [6, 7], 'serves': 1, 'session': 'collect-topup-fork', 'kind': 'collect'},
            'b': {'machine': 'kanu', 'gpus': [5], 'serves': 2, 'session': 'rl-fold27-smoke', 'kind': 'eval'},
            'c': {'machine': 'srv50', 'gpus': [3], 'serves': 6, 'session': 'collect-topup-fork', 'kind': 'collect'}}

    def test_admission_rules(self):
        self.assertTrue(EP.admissible(self.LIVE, 'kanu', [5], 2, 'fold27-eval'))     # foreign session on GPU5
        self.assertTrue(EP.admissible(self.LIVE, 'kanu', [0], 2, 'fold27-eval'))     # cap 3 already held
        self.assertTrue(EP.admissible(self.LIVE, 'srv50', [2], 6, 'fold27-eval'))    # srv all-session cap 1
        self.assertEqual(EP.admissible(self.LIVE, 'srv48', [2], 6, 'fold27-eval'), [])
        free = {k: v for k, v in self.LIVE.items() if k != 'b'}
        self.assertEqual(EP.admissible(free, 'kanu', [5], 2, 'fold27-eval'), [])
        self.assertTrue(EP.admissible(free, 'kanu', [5], 3, 'fold27-eval'))

    def test_wait_never_forces(self):
        states = [self.LIVE, self.LIVE, {k: v for k, v in self.LIVE.items() if k != 'b'}]
        sleeps = []
        ok = EP.wait_resources('kanu', [5], 2, 'fold27-eval', poll=1, state_fn=lambda: states.pop(0), busy_fn=lambda g: [],
                               sleep=sleeps.append)
        self.assertTrue(ok); self.assertEqual(len(sleeps), 2)
        busy = [[5], []]
        ok = EP.wait_resources('kanu', [5], 2, 'fold27-eval', poll=1, state_fn=lambda: {}, busy_fn=lambda g: busy.pop(0), sleep=sleeps.append)
        self.assertTrue(ok); self.assertEqual(len(sleeps), 3)

    def test_port_wait(self):
        import socket
        with socket.socket() as busy:
            busy.bind(('127.0.0.1', 0)); busy.listen(); port = busy.getsockname()[1]
            self.assertFalse(EP.wait_port_free(port, timeout=0, sleep=lambda s: None))
        self.assertTrue(EP.wait_port_free(port, timeout=0))

    def test_queue_requires_harness(self):
        env = dict(os.environ)
        os.environ.pop('HARNESS_RECEIPT', None); os.environ.update(HARNESS_MACHINE='kanu', GPUS='5', SERVES_PER_GPU='2', HARNESS_SESSION='x')
        try:
            with self.assertRaises(C.RuntimeContractError):
                EP.queue(EP.Pipeline(root=C.R / 'outputs/tmp/none'))
        finally:
            os.environ.clear(); os.environ.update(env)


# ====================================================================== smoke watcher (smoke_verify.py)
import smoke_verify as SV   # noqa: E402

SMOKE_S1 = C.R / C.SMOKE_ROOT / 's1/OpenDrawer-left_s2_j2'


@unittest.skipUnless((SMOKE_S1 / 'smoke_base/progress.jsonl').exists() and len(SV.lines(SMOKE_S1 / 'smoke_base/progress.jsonl')) == 2
                     and len(SV.lines(SMOKE_S1 / 'smoke_always/progress.jsonl')) == 2, 'real smoke s1 base/always rows absent')
class SmokeVerifyTest(unittest.TestCase):
    """Runs on copies of the real finished smoke_base/smoke_always rows (the live dir is never touched)."""
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='unittest_sv_', dir=C.R / C.SMOKE_ROOT))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.d = self.root / SMOKE_S1.name; self.d.mkdir()
        shutil.copy2(SMOKE_S1 / 'manifest.json', self.d / 'manifest.json')
        for arm in ('smoke_base', 'smoke_always'):
            shutil.copytree(SMOKE_S1 / arm, self.d / arm, symlinks=True)
        self.mp = self.d / 'manifest.json'

    def v(self):
        return SV.validate(manifest=self.mp, arms=('smoke_base', 'smoke_always'))

    def test_real_rows_pass(self):
        errs, table, allowed = self.v()
        self.assertEqual(errs, []); self.assertEqual(len(table), 4); self.assertEqual(allowed, [])
        self.assertEqual(table['smoke_base/n0']['success'], table['smoke_base/n0']['archive'])

    def test_missing_row(self):
        p = self.d / 'smoke_always/progress.jsonl'
        p.write_text(p.read_text().splitlines()[0] + '\n')
        errs, _, _ = self.v()
        self.assertTrue(any('rows' in e or 'orphan' in e for e in errs), errs)

    def test_bad_decision(self):
        f = next((self.d / 'smoke_base').glob('*.decisions.jsonl'))
        ds = SV.lines(f); ds[3]['gate'] = True; ds[5]['qam_delta_max_normalized'] = 0.5
        f.write_text(''.join(json.dumps(x) + '\n' for x in ds))
        errs, _, _ = self.v()
        self.assertTrue(any('gate rule' in e for e in errs) and any('ungated action modified' in e for e in errs), errs)
        f = next((self.d / 'smoke_always').glob('*.decisions.jsonl'))
        ds = SV.lines(f); ds[0]['noise_sha256_composed'] = '0' * 64
        f.write_text(''.join(json.dumps(x) + '\n' for x in ds))
        errs, _, _ = self.v()
        self.assertTrue(any('initial noise' in e for e in errs), errs)

    def test_failure_markers(self):
        fd = self.d / 'smoke_base/failures'
        (fd / '1.json').write_text(json.dumps({'error': SV.ALLOWED_STARTUP_ERROR, 'type': 'OSError', 'time': 1.0}))   # before everything
        errs, _, allowed = self.v()
        self.assertEqual(errs, []); self.assertEqual(len(allowed), 1)
        (fd / '2.json').write_text(json.dumps({'error': SV.ALLOWED_STARTUP_ERROR, 'type': 'OSError', 'time': time.time() + 10}))  # after episodes
        (fd / '3.json').write_text(json.dumps({'error': "RuntimeError('serve exited 1')", 'type': 'RuntimeError', 'time': 1.0}))
        errs, _, _ = self.v()
        self.assertEqual(len([e for e in errs if 'failure marker' in e]), 2, errs)

    def test_watch_unit_failure_and_timeout(self):
        out = self.root / 'out'; out.mkdir()
        rc = SV.main(['--manifest', str(self.mp), '--out', str(out), '--unit', 'u', '--poll', '0'],
                     unit_fn=lambda u: {'ActiveState': 'failed', 'Result': 'exit-code', 'LoadState': 'loaded'}, sleep=lambda s: None)
        self.assertEqual(rc, 3)
        rc = SV.main(['--manifest', str(self.mp), '--out', str(out), '--unit', 'u', '--poll', '0'],
                     unit_fn=lambda u: {'ActiveState': 'inactive', 'Result': 'success', 'LoadState': 'loaded'}, sleep=lambda s: None)
        self.assertEqual(rc, 3)
        rc = SV.main(['--manifest', str(self.mp), '--out', str(out), '--timeout', '0', '--poll', '0'], sleep=lambda s: None)
        self.assertEqual(rc, 3)
        self.assertEqual(len(list(out.glob('SMOKE_VERIFY_FAILED.*.json'))), 3)
        self.assertFalse((out / 'SMOKE_VERIFIED.json').exists())

    def test_pass_writes_marker_only_on_pass(self):
        out = self.root / 'out'; out.mkdir()
        shutil.copytree(self.d / 'smoke_always', self.d / 'smoke_gated')     # wrong mode for gated rows -> must FAIL
        rc = SV.main(['--manifest', str(self.mp), '--out', str(out), '--poll', '0'], sleep=lambda s: None)
        self.assertEqual(rc, 1); self.assertFalse((out / 'SMOKE_VERIFIED.json').exists())


if __name__ == '__main__':
    unittest.main()
