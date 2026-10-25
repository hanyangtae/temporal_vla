#!/usr/bin/env python3
"""noise_extension_check: append-only nold→noise_limit admission (stdlib unittest).

Run: python3 scripts/utils/tests/test_noise_extension.py
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import groot_harness as gh  # noqa: E402

KEY = 'OpenDrawer/left'
OTHER = 'OpenDrawer/right'
POLICY = {'machines': {'kanu': {}, 'srv48': {}, 'srv50': {}}}


def jitters():
    return [{'reset_idx': i, 'lat': 0.0, 'back': 0.0} for i in range(5)]


def plan(nseeds):
    return {'model': 'groot', 'version': 'n15', 'ckpt': 'ck', 'capture_layers': [8],
            'denoise_k': 4, 'token_mode': 'all',
            'instructions': {KEY: [1, 2], OTHER: [3]},
            'scenes': {KEY: [{'layout': 1, 'style': 1}, {'layout': 2, 'style': 2}],
                       OTHER: [{'layout': 3, 'style': 3}]},
            'jitters': {KEY: [jitters(), jitters()], OTHER: [jitters()]},
            'extra': {'env_kwargs': {'a': 1}, 'env_names': {KEY: 'E'}, 'instruction_text': {KEY: 't'}},
            'noise_seeds': list(range(1000, 1000 + nseeds))}


def cells(key, sid, noises):
    return {f'{key}|s{sid}|j{j}|n{n}' for j in range(5) for n in noises}


class NoiseExtensionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(gh, 'policy', return_value=POLICY)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_check(self, nold, nnew, new=None, contract_count=None, noise_limit=None, done=None):
        old = plan(nold)
        new = new if new is not None else plan(nnew)
        plan_path = self.dir / 'plan.json'
        plan_path.write_text(json.dumps(new))
        contract = {'noise_count': nnew if contract_count is None else contract_count,
                    'allowed_instructions': [KEY], 'excluded_instructions': [OTHER],
                    'selected_scenes': [{'instruction': KEY, 'scene': 1}],
                    'machine': 'kanu', 'jitter_count': 5,
                    'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()}
        if done is None:
            # Everything except the appended noise of the selected scene is already done.
            done = cells(KEY, 0, range(nnew)) | cells(KEY, 1, range(nold))
        done_path = self.dir / 'done.txt'
        done_path.write_text('\n'.join(sorted(done)) + '\n')
        return gh.noise_extension_check(contract, old, new, plan_path, KEY,
                                        nnew if noise_limit is None else noise_limit, done_path)

    def test_legacy_n10_to_n20(self):
        r = self.run_check(10, 20)
        self.assertEqual(r['noise_range'], [10, 19])
        self.assertEqual(r['new_episodes'], 5 * 10)
        self.assertEqual(r['remaining_episodes'], 5 * 10)

    def test_n20_to_n40(self):
        r = self.run_check(20, 40)
        self.assertEqual(r['noise_range'], [20, 39])
        self.assertEqual(r['new_episodes'], 5 * 20)
        self.assertEqual(r['remaining_episodes'], 5 * 20)

    def test_changed_prefix_rejected(self):
        new = plan(40)
        new['noise_seeds'][3] = 99999
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.run_check(20, 40, new=new)

    def test_duplicate_seed_rejected(self):
        new = plan(40)
        new['noise_seeds'][30] = new['noise_seeds'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_check(20, 40, new=new)

    def test_changed_environment_rejected(self):
        for field, mutate in (('extra', lambda p: p['extra']['env_kwargs'].update(a=2)),
                              ('jitters', lambda p: p['jitters'][KEY][1][0].update(lat=0.1)),
                              ('scenes', lambda p: p['scenes'][KEY][0].update(style=9)),
                              ('capture_layers', lambda p: p.update(capture_layers=[10]))):
            new = plan(40)
            mutate(new)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f'changed {field}'):
                self.run_check(20, 40, new=new)

    def test_wrong_noise_count_rejected(self):
        cases = {'contract_mismatch': dict(contract_count=30),
                 'limit_mismatch': dict(noise_limit=30),
                 'not_append': dict(new=plan(20))}
        for name, kw in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'append-only'):
                nnew = 20 if name == 'not_append' else 40
                self.run_check(20, nnew, **kw)
        with self.assertRaisesRegex(ValueError, 'append-only'):
            self.run_check(0, 20)

    def test_original_cell_unskip_rejected(self):
        base = cells(KEY, 0, range(40)) | cells(KEY, 1, range(20))
        # Old noise of the selected scene must stay skipped (no backfill) ...
        with self.assertRaisesRegex(ValueError, 'DONE_LIST'):
            self.run_check(20, 40, done=base - {f'{KEY}|s1|j2|n5'})
        # ... and so must the appended noise of out-of-scope scenes.
        with self.assertRaisesRegex(ValueError, 'DONE_LIST'):
            self.run_check(20, 40, done=base - {f'{KEY}|s0|j0|n30'})


if __name__ == '__main__':
    unittest.main()
