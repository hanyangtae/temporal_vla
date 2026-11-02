"""CPU tests for finetune_fold gates: 3-way pkl linkage, latent provenance, init (no smoke), reward defaults, AE split."""
import argparse, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'trqam')]
import finetune_fold as FF   # noqa: E402
import ae_potential as AP    # noqa: E402

A, B = 'a' * 64, 'b' * 64


class LinkageTest(unittest.TestCase):
    def test_all_three_must_match(self):
        FF.check_linkage('e', {'pkl_sha256': A}, {'episode_id': 'e', 'pkl_sha256': A}, {'episode_id': 'e', 'pkl_sha256': A})
        for spec, ext, lat in ((B, A, A), (A, B, A), (A, A, B)):       # each single-side mismatch rejected
            with self.assertRaises(FF.FinetuneError):
                FF.check_linkage('e', {'pkl_sha256': spec}, {'episode_id': 'e', 'pkl_sha256': ext}, {'episode_id': 'e', 'pkl_sha256': lat})
        with self.assertRaises(FF.FinetuneError):
            FF.check_linkage('e', {'pkl_sha256': A}, {'episode_id': 'e', 'pkl_sha256': A}, {'episode_id': 'x', 'pkl_sha256': A})

    def test_latent_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            ep, la = Path(d) / 'ep', Path(d) / 'la'; ep.mkdir(); la.mkdir()
            (ep / 'contract.json').write_text(json.dumps({'with_last_block': True}))
            (la / 'contract.json').write_text(json.dumps({'episodes_contract_sha256': FF.sha(ep / 'contract.json')}))
            FF.check_latent_provenance(ep, la)
            (la / 'contract.json').write_text(json.dumps({'episodes_contract_sha256': 'x'}))
            with self.assertRaises(FF.FinetuneError):
                FF.check_latent_provenance(ep, la)

    def test_init_rejects_smoke_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'flags_500000.json').write_text(json.dumps({'data_label': 'original_human_atomic65_scratch', 'steps_done': 500000}))
            self.assertEqual(FF.check_init(d, 500000), 'original_human_atomic65_scratch')
            (d / 'flags_20.json').write_text(json.dumps({'data_label': 'atomic65_smoke', 'steps_done': 20}))
            with self.assertRaises(FF.FinetuneError):
                FF.check_init(d, 20)
            (d / 'flags_100.json').write_text(json.dumps({'data_label': 'atomic65', 'steps_done': 50}))
            with self.assertRaises(FF.FinetuneError):
                FF.check_init(d, 100)


class RewardDefaultTest(unittest.TestCase):
    def ns(self, **kw):
        base = dict(arm='basic', alpha=None, beta=None, reward_confirmation=None, dry_run=True)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_defaults_recorded_not_confirmed(self):
        a = self.ns(arm='pbrs'); r = FF.reward_gate(a)
        self.assertEqual((a.alpha, a.beta, r['decision']), (0.1, None, 'main_default_existing_recipe'))
        a = self.ns(arm='mix'); r = FF.reward_gate(a)
        self.assertEqual((a.alpha, a.beta, r['decision']), (0.1, 0.0396, 'main_default_existing_recipe'))
        self.assertNotIn('confirmation', r)
        with self.assertRaises(FF.FinetuneError):
            FF.reward_gate(self.ns(arm='mix', alpha=0.1))
        with self.assertRaises(FF.FinetuneError):
            FF.reward_gate(self.ns(arm='basic', alpha=0.1))


class AESplitTest(unittest.TestCase):
    def test_split_within_fit_stratified(self):
        eps = [{'episode_id': f'e{i}', 'success': i < 3} for i in range(160)]     # rare S3 fold
        tr, va = AP.ae_val_split(eps)
        self.assertEqual(len(tr) + len(va), 160)
        self.assertFalse({e['episode_id'] for e in tr} & {e['episode_id'] for e in va})
        self.assertGreaterEqual(sum(e['success'] for e in tr), 1)
        self.assertEqual(sum(not e['success'] for e in va), round(157 * .2))
        self.assertEqual(AP.ae_val_split(eps)[1], va)      # deterministic


if __name__ == '__main__':
    unittest.main()
