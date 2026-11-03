"""t49_host_chain_v2: per-scene archive kind/root/pin on top of the unchanged v1 chain (fake scene from test_t49_host_chain).
Covers config validation (v1/v2 loaders reject each other's configs), full candle-only order with a v2 archive,
success_rule required for v2 archives, kind mismatch stop, module constant / cfg binding restored, freeze records archives,
and the shipped srv50_candle_v2 config is candle-only with a NEW out root."""
import json, shutil, sys, tempfile, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT), str(ROOT / 'fold_data')]
import t49_host_chain as CH          # noqa: E402
import t49_host_chain_v2 as C2       # noqa: E402
import test_t49_host_chain as T1     # noqa: E402  (Env fixture)
import stage_episodes as SE          # noqa: E402

PIN = 'a' * 64


def to_v2(e, archive_kind='fold27_extract49_v2', contract_kind='fold27_extract49_v2', rule='collector_v2'):
    c = e.complete['contract']
    c.update(kind=contract_kind, extractor49_sha256=PIN)
    if rule is not None:
        c['success_rule'] = rule
    cb = json.dumps(c).encode()
    (e.arch / T1.NAME / 'contract.json').write_bytes(cb); e.complete['contract_sha256'] = T1.sha(cb)
    root = e.cfg.pop('source49_root')
    e.cfg['kind'] = C2.KIND
    e.cfg['scenes'][0].update(archive_kind=archive_kind, source49_root=root, extractor49_sha256=PIN)
    e.write_cfg()
    return e


def chain(e, **kw):
    return C2.ChainV2(e.repo / 'o' / 'cfg.json', repo=e.repo, fetcher=SE.LocalFetcher(), run=e.run, sleep=lambda s: None,
                      stagers=kw.pop('stagers', lambda d: []), disk=lambda p: types.SimpleNamespace(free=10 ** 15), poll_s=1, **kw)


class ChainV2Test(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.t)

    def env(self, **kw):
        return to_v2(T1.Env(Path(tempfile.mkdtemp(dir=self.t))), **kw)

    def test_config_validation(self):
        e = self.env()
        C2.load_config(e.repo / 'o/cfg.json', e.repo)
        with self.assertRaises(CH.ChainError):
            CH.load_config(e.repo / 'o/cfg.json', e.repo)                     # v1 loader refuses v2 config
        v1 = T1.Env(Path(tempfile.mkdtemp(dir=self.t)))
        with self.assertRaises(CH.ChainError):
            C2.load_config(v1.repo / 'o/cfg.json', v1.repo)                   # v2 loader refuses v1 config
        muts = [lambda c: c.update(source49_root='/x'), lambda c: c['scenes'][0].pop('extractor49_sha256'),
                lambda c: c['scenes'][0].update(archive_kind='fold27_extract49_v3'),
                lambda c: c['scenes'][0].update(source49_root='relative/root'),
                lambda c: c['scenes'][0].update(source49_root='/d/fold27_extract49_v1_20261103'),
                lambda c: c.update(out_root='outputs/rl2_vla/fold27_trqam_t49_srv50_independent_20261103'),
                lambda c: c.update(out_root='outputs/rl2_vla/fold27_trqam_t49_hosts_20261103'),
                lambda c: c.update(gpus=[1, 2])]
        for mut in muts:
            x = self.env(); mut(x.cfg); x.write_cfg()
            with self.assertRaises(CH.ChainError):
                C2.load_config(x.repo / 'o/cfg.json', x.repo)

    def test_full_chain_with_v2_archive(self):
        e = self.env(); e.finish_archive()
        ch = chain(e); st = ch.run(); ch._lock.close()
        self.assertEqual(st['status'], 'done'); self.assertEqual(st['kind'], C2.KIND)
        self.assertEqual(e.kinds(), ['stage', 'latents', 'smoke_basic', 'smoke_pbrs', 'smoke_mix', 'coord'])
        sc = e.calls[0]
        self.assertEqual(sc[sc.index('--remote-dir') + 1], f'{e.cfg["scenes"][0]["source49_root"]}/{T1.NAME}')
        self.assertEqual(CH.EXTRACT_KIND, 'fold27_extract49_v1')                # module constant restored
        self.assertNotIn('source49_root', ch.cfg); self.assertNotIn('extractor49_sha256', ch.cfg)
        fz = json.loads((ch.sd / 'FREEZE.json').read_text())
        self.assertEqual(fz['archives'][T1.NAME][0], 'fold27_extract49_v2'); self.assertIn('chain_v2', fz['tools'])
        self.assertIn('chain', fz['tools'])                                      # v1 chain code pinned too

    def test_v2_archive_without_success_rule_stops(self):
        e = self.env(rule=None); e.finish_archive()
        ch = chain(e); ch.sd.mkdir(parents=True)
        with self.assertRaisesRegex(CH.ChainError, 'success_rule'):
            ch.archive_gate(ch.cfg['scenes'][0])
        self.assertEqual(CH.EXTRACT_KIND, 'fold27_extract49_v1')

    def test_kind_mismatch_stops(self):
        e = self.env(contract_kind='fold27_extract49_v1'); e.finish_archive()     # scene says v2, archive is v1
        ch = chain(e); ch.sd.mkdir(parents=True)
        with self.assertRaisesRegex(CH.ChainError, 'kind'):
            ch.archive_gate(ch.cfg['scenes'][0])
        self.assertEqual(CH.EXTRACT_KIND, 'fold27_extract49_v1')

    def test_v1_kind_scene_supported(self):
        e = self.env(archive_kind='fold27_extract49_v1', contract_kind='fold27_extract49_v1', rule=None); e.finish_archive()
        ch = chain(e); ch.sd.mkdir(parents=True)
        self.assertEqual(json.loads(ch.archive_gate(ch.cfg['scenes'][0]))['kind'], 'fold27_extract49_v1')

    def test_shipped_candle_config(self):
        cfg_p = REPO / 'outputs/analysis/groot_fold27_20261028/t49_chain_20261103/srv50_candle_v2.json'
        cfg, spec = C2.load_config(cfg_p, REPO)
        self.assertEqual([s['key'] for s in cfg['scenes']], ['PPCC/candle|s0'])
        self.assertEqual(cfg['scenes'][0]['archive_kind'], 'fold27_extract49_v2')
        self.assertEqual(cfg['scenes'][0]['extractor49_sha256'],
                         CH.sha_file(ROOT / 'fold_data/extract_scene49_v2.py'))      # pin == current v2 extractor
        others = [json.loads((cfg_p.parent / n).read_text()) for n in ('srv48.json', 'srv50.json', 'srv50_independent.json')]
        self.assertTrue(all(cfg['out_root'] != o['out_root'] and cfg['stage_root'] != o['stage_root']
                            and cfg['latents_root'] != o['latents_root'] for o in others))
        src = json.loads((REPO / cfg['sources']).read_text())
        self.assertEqual(list(src), ['PPCC/candle|s0'])                                  # no jug/marsh retrain


if __name__ == '__main__':
    unittest.main()
