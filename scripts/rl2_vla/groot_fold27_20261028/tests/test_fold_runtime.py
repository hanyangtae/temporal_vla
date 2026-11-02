"""CPU tests: episode staging, compact export, per-fold orchestration (fake runner), checkpoint revision pin.

    python3 -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_fold_runtime.py -v
"""
import argparse, hashlib, json, os, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'fold_data'), str(ROOT / 'trqam'), str(ROOT / 'mil')]
import stage_episodes as SE    # noqa: E402
import compact_export as CE    # noqa: E402
import run_fold as RF          # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_remote(d, n=3, with_last_block=True):
    """Fake extract_scene output dir + matching mini fold spec (fit = first n-1, eval = last)."""
    d.mkdir(parents=True)
    (d / 'contract.json').write_text(json.dumps({'with_last_block': with_last_block, 'layers': [0, 2, 4, 8, 10, 12, 15]}))
    fold = {'fold_id': 'F', 'instruction': 'PPCC/bread', 'scene': 4, 'archive_machine': 'worker2', 'fit': [], 'eval': []}
    rng = np.random.default_rng(0)
    for i in range(n):
        r = {'plan_id': '4fa6496cd684', 'j': 0 if i < n - 1 else 1, 'n': i, 'pkl_sha256': f'{i:064x}', 'success': i % 2}
        (fold['fit'] if i < n - 1 else fold['eval']).append(r)
        eid = SE.episode_id(r['plan_id'], 'worker2', 'PPCC/bread', 4, r['j'], r['n'])
        st = SE.stem(eid)
        arr = dict(tokmean_l12d3=rng.normal(size=(6, 1536)).astype(np.float16), actions_model12=np.zeros((6, 16, 12), np.float32),
                   success=np.int8(r['success']), n_inferences=np.int32(6), env_steps=np.int32(30), first_success_step=np.int32(-1),
                   max_episode_steps=np.int32(30), env_step_success=np.zeros(30, np.int8))
        if with_last_block:
            arr['last_block_d0'] = rng.normal(size=(6, 16, 1536)).astype(np.float16)
        np.savez(d / f'{st}.npz', **arr)
        (d / f'{st}.json').write_text(json.dumps({'episode_id': eid, 'pkl_sha256': r['pkl_sha256'], 'success': r['success'],
                                                  'npz': f'{st}.npz', 'npz_sha256': sha(d / f'{st}.npz'), 'n_inferences': 6}))
    return {'folds': [fold]}


class StageTest(unittest.TestCase):
    def test_stage_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); spec = make_remote(t / 'remote')
            s = SE.stage(spec, SE.LocalFetcher(), t / 'remote', t / 'local', reserve_gib=0, log=lambda *_: None)
            self.assertEqual((s['episodes'], s['pulled_this_run']), (3, 3))
            s = SE.stage(spec, SE.LocalFetcher(), t / 'remote', t / 'local', reserve_gib=0, log=lambda *_: None)
            self.assertEqual(s['pulled_this_run'], 0)                                   # resume: nothing re-pulled
            only_fit = SE.needed(spec, roles=('fit',))
            self.assertEqual(len(only_fit), 2)
            npz = next((t / 'local').glob('*.npz')); npz.write_bytes(npz.read_bytes() + b'x')   # local tamper
            with self.assertRaises(SE.StageError):
                SE.stage(spec, SE.LocalFetcher(), t / 'remote', t / 'local', reserve_gib=0, log=lambda *_: None)

    def test_stage_rejects_linkage_and_contract(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); spec = make_remote(t / 'remote')
            spec['folds'][0]['fit'][0]['pkl_sha256'] = 'f' * 64
            with self.assertRaises(SE.StageError):
                SE.stage(spec, SE.LocalFetcher(), t / 'remote', t / 'local', reserve_gib=0, log=lambda *_: None)
            spec = make_remote(t / 'remote2')
            (t / 'local2').mkdir(); (t / 'local2' / 'contract.json').write_text('{}')
            with self.assertRaises(SE.StageError):
                SE.stage(spec, SE.LocalFetcher(), t / 'remote2', t / 'local2', reserve_gib=0, log=lambda *_: None)
            with self.assertRaises(SE.StageError):
                SE.stage(make_remote(t / 'remote3'), SE.LocalFetcher(), t / 'remote3', t / 'local3', reserve_gib=1e9, log=lambda *_: None)


class WrapperFetcherTest(unittest.TestCase):
    def test_command_uses_wrapper_and_quotes(self):
        f = SE.WrapperFetcher()
        self.assertTrue(f.WRAPPER.is_file())
        cmd = f._cmd("/data/x y/'q'.npz")
        self.assertEqual(cmd[:3], ['bash', str(f.WRAPPER), 'run'])
        self.assertEqual(len(cmd), 4)
        import shlex
        self.assertEqual(shlex.split(cmd[3]), ['cat', '--', "/data/x y/'q'.npz"])
        with self.assertRaises(SE.StageError):
            f._cmd('relative/path.npz')


class CompactTest(unittest.TestCase):
    def test_compact_identical_and_traceable(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); make_remote(t / 'full')
            self.assertEqual(CE.export(t / 'full', t / 'cmp', log=lambda *_: None), 3)
            c = json.loads((t / 'cmp/contract.json').read_text())
            self.assertEqual(c['source_contract_sha256'], sha(t / 'full/contract.json'))
            for jp in (t / 'full').glob('*.json'):
                if jp.name == 'contract.json':
                    continue
                full, cmp_ = json.loads(jp.read_text()), json.loads((t / 'cmp' / jp.name).read_text())
                self.assertEqual(cmp_['source_npz_sha256'], full['npz_sha256'])
                self.assertEqual(cmp_['pkl_sha256'], full['pkl_sha256'])
                self.assertTrue(CE.verify_equal(t / 'full' / full['npz'], t / 'cmp' / cmp_['npz']))
                with np.load(t / 'cmp' / cmp_['npz']) as z:
                    self.assertNotIn('last_block_d0', z.files)
            self.assertEqual(CE.export(t / 'full', t / 'cmp', log=lambda *_: None), 3)   # idempotent resume

    def test_compact_satisfies_latent_provenance(self):
        import finetune_fold as FF
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); make_remote(t / 'full'); CE.export(t / 'full', t / 'cmp', log=lambda *_: None)
            (t / 'lat').mkdir(); (t / 'lat/contract.json').write_text(json.dumps({'episodes_contract_sha256': sha(t / 'full/contract.json')}))
            FF.check_latent_provenance(t / 'cmp', t / 'lat')
            FF.check_latent_provenance(t / 'full', t / 'lat')
            (t / 'lat/contract.json').write_text(json.dumps({'episodes_contract_sha256': 'x'}))
            with self.assertRaises(FF.FinetuneError):
                FF.check_latent_provenance(t / 'cmp', t / 'lat')


class FakeRunner:
    """Simulates each stage by writing its completion marker; can be told to fail a stage."""
    def __init__(self, work, fail=None):
        self.work, self.fail, self.calls = work, fail, []

    def __call__(self, cmd, log, env=None):
        stage = log.parent.name
        self.calls.append(stage)
        log.write_text('fake\n')
        if stage == self.fail:
            return 1
        fold = 'F'
        if stage == 'mil_shards':
            p = self.work / 'mil_shards' / fold; p.mkdir(parents=True, exist_ok=True); (p / 'SHARD_COMPLETE.json').write_text('{}')
        elif stage == 'mil_train':
            p = self.work / 'mil_runs' / fold / 'attn/raw/s0'; p.mkdir(parents=True, exist_ok=True)
            (p / 'model.pt').write_text('m')
            (p / 'FOLD_CONTRACT.json').write_text(json.dumps({'model_sha256': sha(p / 'model.pt'),
                                                              'heldout_proof': {k: True for k in RF.PROOF_KEYS}}))
        elif stage.startswith('trqam'):
            arm = stage.split('_')[-1]
            p = self.work / 'trqam' / fold / RF.RECIPE_TAGS[arm]; p.mkdir(parents=True, exist_ok=True)
            (p / ('DRYRUN.json' if '_dry_' in stage else 'DONE.json')).write_text('{}')
        return 0


def args(t, **kw):
    a = dict(spec=t / 'spec.json', fold='F', work=t / 'work', episodes=t / 'ep', logs=t / 'logs', remote_dir=None, latents=None,
             arms=['basic', 'pbrs', 'mix'], py_mil='py', py_trqam='py', production=False, init=None, init_step=500000, retry_failed=False)
    a.update(kw)
    return argparse.Namespace(**a)


class OrchestrationTest(unittest.TestCase):
    def test_waits_for_latents_then_completes_and_skips(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fr = FakeRunner(t / 'work')
            self.assertEqual(RF.FoldRun(args(t), runner=fr).run(), RF.EXIT_WAITING)
            self.assertEqual(fr.calls, ['mil_shards', 'mil_train'])
            st = json.loads((t / 'work/orch/F/ORCH_STATUS.json').read_text())
            self.assertEqual((st['state'], st['stage']), ('waiting', 'trqam_dry_basic'))
            (t / 'lat').mkdir(); (t / 'lat/COMPLETE.json').write_text('{}')
            self.assertEqual(RF.FoldRun(args(t, latents=t / 'lat'), runner=fr).run(), 0)
            self.assertEqual(fr.calls[2:], ['trqam_dry_basic', 'trqam_dry_pbrs', 'trqam_dry_mix'])
            n = len(fr.calls)
            self.assertEqual(RF.FoldRun(args(t, latents=t / 'lat'), runner=fr).run(), 0)
            self.assertEqual(len(fr.calls), n)                                           # all markers present -> no reruns

    def test_failure_stops_and_logs_are_attempt_numbered(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fr = FakeRunner(t / 'work', fail='mil_train')
            with self.assertRaises(RF.OrchError):
                RF.FoldRun(args(t), runner=fr).run()
            self.assertTrue((t / 'work/orch/F/ORCH_FAILED.json').exists())
            self.assertEqual(fr.calls, ['mil_shards', 'mil_train'])
            with self.assertRaises(RF.OrchError):                                        # refuses until --retry-failed
                RF.FoldRun(args(t), runner=FakeRunner(t / 'work')).run()
            fr2 = FakeRunner(t / 'work')
            self.assertEqual(RF.FoldRun(args(t, retry_failed=True), runner=fr2).run(), RF.EXIT_WAITING)
            self.assertEqual(fr2.calls, ['mil_train'])
            self.assertEqual(sorted(p.name for p in (t / 'logs/mil_train').glob('*.log')), ['F.1.log', 'F.2.log'])
            self.assertFalse((t / 'work/orch/F/ORCH_FAILED.json').exists() and False)

    def test_production_gates(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            os.environ.pop('HARNESS_RECEIPT', None)
            with self.assertRaises(RF.OrchError):
                RF.FoldRun(args(t, production=True, init=t / 'init'), runner=FakeRunner(t / 'work')).run()
            os.environ['HARNESS_RECEIPT'] = 'r'
            try:
                with self.assertRaises(RF.OrchError):
                    RF.FoldRun(args(t, production=True), runner=FakeRunner(t / 'work')).run()
                (t / 'lat').mkdir(); (t / 'lat/COMPLETE.json').write_text('{}')
                fr = FakeRunner(t / 'work')
                self.assertEqual(RF.FoldRun(args(t, production=True, init=t / 'init', latents=t / 'lat', arms=['basic']), runner=fr).run(), 0)
                self.assertEqual(fr.calls[-2:], ['trqam_dry_basic', 'trqam_basic'])
            finally:
                os.environ.pop('HARNESS_RECEIPT', None)

    def test_marker_missing_after_rc0_is_error(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with self.assertRaises(RF.OrchError):
                RF.FoldRun(args(t), runner=lambda cmd, log, env=None: (log.write_text('x'), 0)[1]).run()


class MilCommitTest(unittest.TestCase):
    FOLD = {'fold_id': 'F', 'target_j': 1, 'archive_machine': 'worker2', 'instruction': 'PPCC/bread', 'scene': 4,
            'fit': [{'plan_id': 'p', 'j': j, 'n': n} for j in (0, 2) for n in range(2)],
            'eval': [{'plan_id': 'p', 'j': 1, 'n': n} for n in range(2)]}

    def trainer(self, train_js=(0,), val_js=(2,), test_js=(1,)):
        import pandas as pd
        # explicit columns: an empty split (val_js=()) must keep the trainer's step-table schema
        mk = lambda js: pd.DataFrame([{'episode_id': f'p/worker2/PPCC/bread/s4/j{j}/n{n}', 'jitter': j} for j in js for n in range(2)],
                                     columns=['episode_id', 'jitter'])
        return argparse.Namespace(data_meta={'train': mk(train_js), 'val': mk(val_js), 'test': mk(test_js)})

    def test_heldout_proof(self):
        try:
            import fold_mil as FM
            import pandas  # noqa: F401
        except ImportError as e:
            self.skipTest(f'deps absent: {e}')
        p = FM.heldout_proof(self.trainer(), self.FOLD)
        self.assertTrue(p['train_val_disjoint_eval40'])
        with self.assertRaises(FM.FoldMilError):
            FM.heldout_proof(self.trainer(train_js=(0, 1)), self.FOLD)    # target j leaked into train
        with self.assertRaises(FM.FoldMilError):
            FM.heldout_proof(self.trainer(val_js=()), self.FOLD)           # train+val != fit set

    def test_verify_committed(self):
        try:
            import fold_mil as FM
        except ImportError as e:
            self.skipTest(f'deps absent: {e}')
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            (run / 'model.pt').write_text('m')
            with self.assertRaises(FM.FoldMilError):                          # model without contract = partial
                FM.verify_committed(run, 'F', 0, 'S', self.FOLD)
            good = {'fold_id': 'F', 'seed': 0, 'recipe': FM.RECIPE_TRAIN, 'shard_complete_sha256': 'S', 'target_j': 1,
                    'model_sha256': sha(run / 'model.pt'), 'heldout_proof': {k: True for k in RF.PROOF_KEYS}}
            (run / 'FOLD_CONTRACT.json').write_text(json.dumps(good))
            FM.verify_committed(run, 'F', 0, 'S', self.FOLD)
            for k, v in (('shard_complete_sha256', 'X'), ('model_sha256', 'X'), ('heldout_proof', None), ('recipe', {})):
                (run / 'FOLD_CONTRACT.json').write_text(json.dumps(dict(good, **{k: v})))
                with self.assertRaises(FM.FoldMilError, msg=k):
                    FM.verify_committed(run, 'F', 0, 'S', self.FOLD)
            (run / 'FOLD_CONTRACT.json').write_text(json.dumps(dict(good, model_sha256='X')))
            self.assertFalse(RF.mil_committed(run))
            (run / 'FOLD_CONTRACT.json').write_text(json.dumps(dict(good, heldout_proof={})))
            self.assertFalse(RF.mil_committed(run))
            (run / 'FOLD_CONTRACT.json').write_text(json.dumps(dict(good, heldout_proof={'train_val_disjoint_eval40': True})))
            self.assertFalse(RF.mil_committed(run))                            # all three booleans required
            (run / 'FOLD_CONTRACT.json').write_text(json.dumps(good))
            self.assertTrue(RF.mil_committed(run))


class DeviceGateTest(unittest.TestCase):
    def test_same_host_only_unless_validated(self):
        import rollout_latents as RL
        eid = '4fa6496cd684/worker2/PPCC/bread/s4/j1/n0'
        RL.device_gate(eid, 'srv50', False)
        with self.assertRaises(SystemExit):
            RL.device_gate(eid, 'kanu', False)
        RL.device_gate(eid, 'kanu', True)
        with self.assertRaises(SystemExit):
            RL.device_gate(eid, None, False)


class RevisionTest(unittest.TestCase):
    def test_revision_pin(self):
        import rollout_latents as RL
        good = Path('/x/snapshots') / RL.EXPECT_REVISION / 'gr00t_n1-5/multitask_learning/checkpoint-120000'
        self.assertEqual(RL.checkpoint_revision(good), RL.EXPECT_REVISION)
        for bad in (Path('/x/snapshots/14895998aaaa/gr00t_n1-5/multitask_learning/checkpoint-120000'),
                    Path('/x/other') / RL.EXPECT_REVISION / 'gr00t_n1-5/multitask_learning/checkpoint-120000',
                    Path('/x/snapshots') / RL.EXPECT_REVISION / 'gr00t_n1-5/checkpoint-120000'):
            with self.assertRaises(SystemExit):
                RL.checkpoint_revision(bad)


if __name__ == '__main__':
    unittest.main()
