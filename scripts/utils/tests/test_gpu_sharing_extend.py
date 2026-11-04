"""gpu_sharing_extend: exact expiry-only extension of an existing v1/v2 contract (fake probe; no nvidia-smi/docker/proc)."""
import hashlib, json, sys, unittest
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(UTILS), str(Path(__file__).resolve().parent)]
import gpu_sharing_authorization as SA   # noqa: E402
import gpu_sharing_extend as X           # noqa: E402
import test_gpu_sharing_v2 as V          # noqa: E402  (fixture only, unchanged)


class ExtendTest(unittest.TestCase):
    setUp = V.V2Test.setUp
    args = V.V2Test.args

    def ext(self, src=None, add=86400, expect=None, out=None, **kw):
        src = src or self.path
        exp = SA.load(src)['expires_at'] if expect is None else expect
        return X.extend(src, add, exp, 'user: 그거 24시간 늘려', out or self.t / 'new.json', probe=self.p, **kw)

    def body(self, p):
        return json.loads(Path(p).read_text())

    def test_v2_exact_extension_only_expiry_changes(self):
        before = self.path.read_bytes()
        r = self.ext()
        old, new = self.body(self.path), self.body(self.t / 'new.json')
        self.assertEqual(new['expires_at'], old['expires_at'] + 86400)
        self.assertEqual({k: v for k, v in new.items() if k not in ('expires_at', 'extensions')},
                         {k: v for k, v in old.items() if k != 'expires_at'})
        self.assertEqual(self.path.read_bytes(), before)                        # source untouched without --install-over
        e = new['extensions'][0]
        self.assertEqual(e['from_sha256'], hashlib.sha256(before).hexdigest()); self.assertEqual(e['add_seconds'], 86400.0)
        self.assertEqual([x['pid'] for x in e['live']['pinned_live']], [1101996])
        self.assertEqual(r['changed_keys'], ['expires_at', 'extensions'])
        self.assertEqual(e['live']['other_gpu_pids_not_judged'], [])            # container members are classified
        a2 = SA.load(self.t / 'new.json')                                       # official loader + checks accept it
        self.p.t = old['expires_at'] + 10
        SA.authorized_processes(a2, 'srv50', [3], 'fold27-v2-eval', 'eval', 2, self.p)
        with self.assertRaisesRegex(SA.AuthError, 'expire'):
            SA.authorized_processes(SA.load(self.path), 'srv50', [3], 'fold27-v2-eval', 'eval', 2, self.p)

    def test_v1_and_our_serves_recorded_not_judged(self):
        self.p.procs[5000] = V.P(1004, 1004, 1376000000, 1, [5000], '0::/user.slice', 'python serve.py')
        self.p.apps.append({'pid': 5000, 'used_mib': 6000})
        r = self.ext(src=self.t / 'v1.json')
        self.assertIn(5000, r['live']['other_gpu_pids_not_judged'])
        self.assertEqual(self.body(self.t / 'new.json')['expires_at'], self.body(self.t / 'v1.json')['expires_at'] + 86400)

    def refused(self, msg, **kw):
        with self.assertRaisesRegex(SA.AuthError, msg):
            self.ext(**kw)
        self.assertFalse((self.t / 'new.json').exists())

    def test_refusals_write_nothing(self):
        exp = SA.load(self.path)['expires_at']
        self.refused('expected', expect=exp + 1)
        self.refused('> 0', add=0)
        self.refused('cap', add=72 * 3600)
        for mut, msg in [(lambda: setattr(self.p, 'host', 'srv48'), 'host'),
                         (lambda: setattr(self.p, 'boot', 'x'), 'boot id'),
                         (lambda: self.p.procs[1101996].update(start_ticks=1), 'start_ticks'),
                         (lambda: self.p.procs[1101996].update(cmd='python other.py'), 'cmdline_sha256'),
                         (lambda: self.p.procs[1101996].update(uid=5), 'uid'),
                         (lambda: self.p.procs.pop(1101996), 'gone'),
                         (lambda: setattr(self.p, 'apps', [a for a in self.p.apps if a['pid'] != 1101996]), 'not on GPU'),
                         (lambda: self.p.apps[0].update(used_mib=99999), 'authorized'),
                         (lambda: self.p.procs[3630028].update(start_ticks=1), 'container'),
                         (lambda: setattr(self.p, 't', exp), 'already expired')]:
            self.setUp(); mut()
            self.refused(msg)

    def test_existing_out_refused(self):
        (self.t / 'new.json').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.ext()
        self.assertEqual((self.t / 'new.json').read_text(), 'keep')

    def test_install_over_keeps_original_and_swaps(self):
        before = self.path.read_bytes(); sha = hashlib.sha256(before).hexdigest()
        r = self.ext(install_over=self.path)
        keep = Path(r['installed']['original_kept'])
        self.assertEqual(keep.read_bytes(), before); self.assertIn(sha[:16], keep.name)
        self.assertEqual(self.path.read_bytes(), (self.t / 'new.json').read_bytes())
        self.assertEqual(SA.load(self.path)['expires_at'], json.loads(before)['expires_at'] + 86400)
        self.assertEqual(list(self.t.glob('.*.tmp')), [])

    def test_install_over_refused_if_target_changed(self):
        tgt = self.t / 'other.json'; tgt.write_text('{"x":1}')
        with self.assertRaisesRegex(SA.AuthError, 'changed'):
            self.ext(install_over=tgt)
        self.assertEqual(tgt.read_text(), '{"x":1}')

    def test_second_extension_chains_and_needs_new_expect(self):
        self.ext()
        old = SA.load(self.path)['expires_at']
        with self.assertRaisesRegex(SA.AuthError, 'expected'):
            self.ext(src=self.t / 'new.json', expect=old, out=self.t / 'new2.json')

    def test_cli_refusal_rc2(self):
        rc = X.main(['--auth', str(self.t / 'missing.json'), '--add-seconds', '86400', '--expect-expires-at', '1',
                     '--approval', 'x', '--out', str(self.t / 'o.json')])
        self.assertEqual(rc, 2); self.assertFalse((self.t / 'o.json').exists())


if __name__ == '__main__':
    unittest.main()
