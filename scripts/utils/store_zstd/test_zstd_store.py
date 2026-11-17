"""Tests for zstd_store.py (stdlib unittest + real zstd CLI).

run:  python3 -m unittest scripts/utils/store_zstd/test_zstd_store.py -v
"""
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zstd_store as zs  # noqa: E402

P40, PA, PB = '2f26fb046fc4', 'aaaaaaaaaaaa', 'bbbbbbbbbbbb'


def blob(seed, n=300000):
    r = random.Random(seed)
    words = [bytes(r.getrandbits(8) for _ in range(8)) for _ in range(64)]
    return b''.join(r.choice(words) for _ in range(n // 8))


def mk_rollout(root, plan, key, s, j, n, env_seed, reset=None, lat=None, back=None, payload=None,
               arm='base', with_meta=True, pkl_sha_in_meta=True):
    d = os.path.join(root, plan, 'kanu', key, 's%d' % s, 'j%d' % j, 'n%d' % n, arm)
    os.makedirs(d)
    data = payload if payload is not None else blob(hash((plan, key, s, j, n)) & 0xffff)
    with open(os.path.join(d, 'rollout.pkl'), 'wb') as f:
        f.write(data)
    with open(os.path.join(d, 'traj.csv'), 'w') as f:
        f.write('t,x\n0,1\n')
    if with_meta:
        meta = {'plan_id': plan, 'grid_instruction': key, 'scene_idx': s, 'jitter_idx': j, 'noise_idx': n,
                'env_seed': env_seed, 'inference_seed': 1300000 + n}
        if reset is not None:
            meta['jitter_reset_idx'] = reset
        if lat is not None:
            meta['base_lat'], meta['base_back'] = lat, back
        if pkl_sha_in_meta:
            meta['pkl_sha256'] = hashlib.sha256(data).hexdigest()
        with open(os.path.join(d, 'meta.json'), 'w') as f:
            json.dump(meta, f)
    return d


def run_cli(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = zs.main(list(argv))
    return rc, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix='zstd_store_test_'))
        self.root = os.path.join(self.tmp, 'grid')
        os.makedirs(self.root)
        self.out = os.path.join(self.tmp, 'manifest')
        self.state = os.path.join(self.tmp, 'state')
        self.lockdir = os.path.join(self.tmp, 'locks')
        os.makedirs(self.lockdir)
        zs.HOOKS.clear()

    def tearDown(self):
        zs.HOOKS.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, *extra, out=None):
        rc, txt = run_cli('build', '--root', self.root, '--out', out or self.out, '--protect-plan', P40,
                          '--min-size', '1000', '--progress-every', '0', *extra)
        return rc, txt

    def manifest(self, out=None):
        out = out or self.out
        with open(os.path.join(out, 'manifest.tsv.sha256')) as f:
            sha = f.read().split()[0]
        rows = zs.load_manifest(os.path.join(out, 'manifest.tsv'), sha)[1]
        return sha, {r['rel_path']: r for r in rows}

    def excl(self, out=None):
        with open(os.path.join(out or self.out, 'exclusions.tsv')) as f:
            next(f)
            return {l.split('\t')[0]: l.split('\t')[1] for l in f}

    def summary(self, out=None):
        with open(os.path.join(out or self.out, 'summary.json')) as f:
            return json.load(f)

    def run_mode(self, mode, sha, *extra, state=None):
        return run_cli('run', '--manifest', os.path.join(self.out, 'manifest.tsv'), '--manifest-sha256', sha,
                       '--state-dir', state or self.state, '--mode', mode, '--min-age-sec', '0',
                       '--min-free-gb', '0', '--lock-dir', self.lockdir, *extra)


class TestBuildScope(Base):
    def fixture(self):
        # protected n40 plan: drawer s2 j0 (seed 100005, reset 3), oven s1 j1 (seed 100017, lat .1)
        mk_rollout(self.root, P40, 'OpenDrawer/left', 2, 0, 20, 100005, reset=3)
        mk_rollout(self.root, P40, 'OvenRack/out-left', 1, 1, 21, 100017, reset=0, lat=0.1, back=0.0)
        # same world in other plan, n0..19 -> protected
        self.same = mk_rollout(self.root, PA, 'OpenDrawer/left', 2, 0, 5, 100005, reset=3)
        # same oven world with renamed side key -> protected (family match)
        self.renamed = mk_rollout(self.root, PB, 'OvenRack/out-right', 0, 4, 12, 100017, reset=0, lat=0.1, back=0.0)
        # same scene, other jitter -> compress
        self.otherj = mk_rollout(self.root, PA, 'OpenDrawer/left', 2, 1, 5, 100005, reset=7)
        # index-only coincidence: same key/s/j, different kitchen -> compress
        self.idx = mk_rollout(self.root, PB, 'OpenDrawer/left', 2, 0, 3, 100999, reset=3)
        # meta without jitter info on protected scene -> wildcard -> protected
        self.nojit = mk_rollout(self.root, PB, 'OpenDrawer/left', 4, 2, 1, 100005)
        # missing meta -> not a rollout dir
        self.nometa = mk_rollout(self.root, PA, 'PPCC/jug', 0, 0, 0, 101250, with_meta=False)
        # meta sha mismatch
        self.badsha = mk_rollout(self.root, PA, 'PPCC/bread', 0, 0, 0, 100466, reset=1)
        with open(os.path.join(self.badsha, 'rollout.pkl'), 'ab') as f:
            f.write(b'x')
        # excluded dirs
        for d in ('shard_robocasa', 'derived', 'checkpoints', 'videos'):
            dd = os.path.join(self.root, PA, d, 'x')
            os.makedirs(dd)
            with open(os.path.join(dd, 'rollout.pkl'), 'wb') as f:
                f.write(blob(1))
        # symlinks: file link and dir link; their targets must be protected
        self.ftarget = mk_rollout(self.root, PA, 'PPCC/candle', 0, 0, 0, 100477, reset=1)
        os.symlink(os.path.join(self.ftarget, 'rollout.pkl'), os.path.join(self.root, PA, 'link.pkl'))
        self.dtarget = mk_rollout(self.root, PA, 'PPCC/candle', 0, 0, 1, 100477, reset=1)
        os.symlink(self.dtarget, os.path.join(self.root, PB, 'dirlink'))
        # hardlink
        self.hl = mk_rollout(self.root, PA, 'PPCC/apple', 0, 0, 0, 100521, reset=1)
        os.link(os.path.join(self.hl, 'rollout.pkl'), os.path.join(self.tmp, 'outside_hardlink.pkl'))
        # existing .zst sibling
        self.zsib = mk_rollout(self.root, PA, 'PPCC/apple', 0, 0, 1, 100521, reset=1)
        open(os.path.join(self.zsib, 'rollout.pkl.zst'), 'wb').close()
        self.plain = mk_rollout(self.root, PA, 'PPCC/apple', 0, 0, 2, 100521, reset=1)

    def rel(self, d, fn='rollout.pkl'):
        return os.path.relpath(os.path.join(d, fn), self.root)

    def test_scope(self):
        self.fixture()
        rc, txt = self.build('--expect-protected', '2', '--expect-protected-worlds', '2')
        self.assertEqual(rc, 0, txt)
        _, m = self.manifest()
        ex = self.excl()
        self.assertEqual(set(m), {self.rel(self.otherj), self.rel(self.idx), self.rel(self.plain)})
        self.assertEqual(ex[self.rel(self.same)], 'protected_world_match')
        self.assertEqual(ex[self.rel(self.renamed)], 'protected_world_match')
        self.assertEqual(ex[self.rel(self.nojit)], 'protected_world_match')
        self.assertEqual(ex[self.rel(self.nometa)], 'not_rollout_dir')
        self.assertEqual(ex[self.rel(self.badsha)], 'meta_sha_mismatch')
        self.assertEqual(ex[self.rel(self.ftarget)], 'symlink_target')
        self.assertEqual(ex[self.rel(self.dtarget)], 'symlink_target')
        self.assertEqual(ex[self.rel(self.hl)], 'hardlink_preserve')
        self.assertEqual(ex[self.rel(self.zsib)], 'zst_sibling_exists')
        self.assertEqual(ex[self.rel(self.plain, 'traj.csv')], 'suffix_not_included')
        for d in ('shard_robocasa', 'derived', 'checkpoints', 'videos'):
            self.assertEqual(ex['%s/%s/' % (PA, d)], 'excluded_dir_' + d)
        p40 = [k for k in ex if k.startswith(P40)]
        self.assertTrue(p40 and all(ex[k] == 'protected_plan' for k in p40))
        self.assertNotIn('%s/link.pkl' % PA, m)
        s = self.summary()
        self.assertEqual(s['index_only_coincidences_kept'], 1)
        self.assertEqual(s['symlinks'], 2)
        self.assertEqual(s['cross_plan_world_match_dirs'], 3)

    def test_expect_gate(self):
        self.fixture()
        rc, txt = self.build('--expect-protected', '1800')
        self.assertEqual(rc, 1)
        self.assertIn('refusing', txt)
        self.assertFalse(os.path.exists(os.path.join(self.out, 'manifest.tsv')))
        rc, txt = self.build('--expect-protected-worlds', '45', out=self.out + '2')
        self.assertEqual(rc, 1)

    def test_symlink_inside_protected_plan_protects_target(self):
        src = mk_rollout(self.root, PA, 'PPCC/jug', 1, 0, 3, 101250, reset=2)
        mk_rollout(self.root, P40, 'OpenDrawer/left', 2, 0, 20, 100005, reset=3)
        d = os.path.join(self.root, P40, 'kanu', 'PPCC/jug', 's1', 'j0', 'n3')
        os.makedirs(d)
        os.symlink(src, os.path.join(d, 'base'))
        rc, txt = self.build('--expect-protected', '2')
        self.assertEqual(rc, 0, txt)
        _, m = self.manifest()
        self.assertNotIn(self.rel(src), m)
        self.assertIn(self.excl()[self.rel(src)], ('symlink_target', 'protected_world_match'))
        self.assertEqual(self.summary()['protected_coords_by_how'], {'physical': 1, 'symlink': 1})

    def test_external_symlink_into_root_protects_target(self):
        mk_rollout(self.root, P40, 'OpenDrawer/left', 2, 0, 20, 100005, reset=3)
        a = mk_rollout(self.root, PA, 'PPCC/jug', 1, 0, 3, 101250, reset=2)
        b = mk_rollout(self.root, PA, 'PPCC/jug', 1, 0, 4, 101250, reset=2)
        c = mk_rollout(self.root, PB, 'PPCC/jug', 1, 0, 5, 101250, reset=2)
        ws = os.path.join(self.tmp, 'workspace')
        os.makedirs(ws)
        os.symlink(os.path.join(a, 'rollout.pkl'), os.path.join(ws, 'a.pkl'))
        os.symlink(os.path.join(self.root, PB), os.path.join(ws, 'planB'))      # whole plan alias
        os.symlink('/nonexistent/grid/plan', os.path.join(ws, 'dangling'))       # ignored
        rc, txt = self.build('--external-link-scan', ws)
        self.assertEqual(rc, 0, txt)
        _, m = self.manifest()
        self.assertEqual(set(m), {self.rel(b)})
        ex = self.excl()
        self.assertEqual(ex[self.rel(a)], 'symlink_target')
        self.assertEqual(ex[self.rel(c)], 'symlink_target')
        self.assertEqual(self.summary()['external_links_into_root'], 2)

    def test_no_hash_manifest_refused_by_run(self):
        self.fixture()
        rc, _ = self.build('--no-hash')
        self.assertEqual(rc, 0)
        sha = open(os.path.join(self.out, 'manifest.tsv.sha256')).read().split()[0]
        rc, txt = self.run_mode('verify', sha)
        self.assertEqual(rc, 1)
        self.assertIn('no-hash', txt)


class TestRun(Base):
    def setUp(self):
        super().setUp()
        mk_rollout(self.root, P40, 'OpenDrawer/left', 2, 0, 20, 100005, reset=3)
        self.d1 = mk_rollout(self.root, PA, 'PPCC/apple', 0, 0, 1, 100521, reset=1)
        self.d2 = mk_rollout(self.root, PA, 'PPCC/apple', 0, 0, 2, 100521, reset=1)
        self.d3 = mk_rollout(self.root, PA, 'PPCC/apple', 0, 0, 3, 100521, reset=1,
                             payload=os.urandom(200000))  # incompressible
        os.chmod(os.path.join(self.d1, 'rollout.pkl'), 0o640)
        os.utime(os.path.join(self.d1, 'rollout.pkl'), ns=(1_600_000_000_000_000_000, 1_600_000_000_123_456_789))
        rc, txt = self.build('--expect-protected', '1')
        self.assertEqual(rc, 0, txt)
        self.sha, self.m = self.manifest()
        self.p1, self.p2, self.p3 = (os.path.join(d, 'rollout.pkl') for d in (self.d1, self.d2, self.d3))
        self.orig = {p: open(p, 'rb').read() for p in (self.p1, self.p2, self.p3)}

    def zsha(self, z):
        return zs.decompress_sha256(z)[0]

    def leftovers(self):
        out = []
        for dp, _, fs in os.walk(self.root):
            out += [os.path.join(dp, f) for f in fs if '.zst.tmp-' in f]
        return out

    def test_verify_mode_touches_nothing(self):
        before = {p: os.lstat(p) for p in self.orig}
        rc, txt = self.run_mode('verify', self.sha)
        self.assertEqual(rc, 0, txt)
        for p in self.orig:
            self.assertEqual(zs.stat_sig(os.lstat(p)), zs.stat_sig(before[p]))
            self.assertFalse(os.path.exists(p + '.zst'))
        self.assertEqual(self.leftovers(), [])
        rep = json.loads(run_cli('report', '--state-dir', self.state)[1])
        self.assertEqual(rep['by_status'], {'verified': 2, 'skip_incompressible': 1})
        self.assertEqual(rep['saved_actual_bytes'], 0)
        self.assertGreater(rep['verify_would_save_bytes'], 0)

    def test_execute_roundtrip_and_meta(self):
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 0, txt)
        for p in (self.p1, self.p2):
            self.assertFalse(os.path.exists(p))
            self.assertEqual(self.zsha(p + '.zst'), hashlib.sha256(self.orig[p]).hexdigest())
        st = os.lstat(self.p1 + '.zst')
        self.assertEqual(st.st_mode & 0o777, 0o640)
        self.assertEqual(st.st_mtime_ns, 1_600_000_000_123_456_789)
        self.assertTrue(os.path.exists(self.p3))  # incompressible kept
        self.assertFalse(os.path.exists(self.p3 + '.zst'))
        rep = json.loads(run_cli('report', '--state-dir', self.state)[1])
        exp = sum(len(self.orig[p]) - os.lstat(p + '.zst').st_size for p in (self.p1, self.p2))
        self.assertEqual(rep['saved_actual_bytes'], exp)
        self.assertEqual(self.leftovers(), [])
        # rerun is a no-op
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 0)
        self.assertIn('"todo": 0', txt)

    def test_corruption_detected(self):
        def corrupt(tmp, **_):
            with open(tmp, 'r+b') as f:
                f.seek(os.path.getsize(tmp) // 2)
                b = f.read(1)
                f.seek(-1, 1)
                f.write(bytes([b[0] ^ 0xff]))
        zs.HOOKS['after_compress'] = corrupt
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 1)
        for p in self.orig:
            self.assertEqual(open(p, 'rb').read(), self.orig[p])
            self.assertFalse(os.path.exists(p + '.zst'))
        self.assertEqual(self.leftovers(), [])
        self.assertIn('fail_verify', txt)

    def test_source_changed_after_manifest(self):
        st = os.lstat(self.p1)
        with open(self.p1, 'r+b') as f:
            f.write(b'Z')
        os.utime(self.p1, ns=(st.st_atime_ns, st.st_mtime_ns))  # same size+mtime, ctime differs
        rc, txt = self.run_mode('execute', self.sha)
        self.assertTrue(os.path.exists(self.p1))
        self.assertFalse(os.path.exists(self.p1 + '.zst'))
        self.assertFalse(os.path.exists(self.p2))  # unaffected file still processed

    def test_source_changed_during_processing(self):
        def mutate(full, **_):
            if full == self.p1:
                with open(full, 'ab') as f:
                    f.write(b'late write')
        zs.HOOKS['before_finalize'] = mutate
        rc, txt = self.run_mode('execute', self.sha)
        self.assertTrue(os.path.exists(self.p1))
        self.assertFalse(os.path.exists(self.p1 + '.zst'))
        self.assertIn('recheck:stat_changed', txt)
        self.assertEqual(self.leftovers(), [])

    def test_crash_after_finalize_then_resume(self):
        class Crash(BaseException):
            pass

        def boom(**_):
            raise Crash()
        zs.HOOKS['after_finalize'] = boom
        with self.assertRaises(Crash):
            self.run_mode('execute', self.sha)
        first = [p for p in (self.p1, self.p2) if os.path.exists(p + '.zst')]
        self.assertEqual(len(first), 1)
        self.assertTrue(os.path.exists(first[0]))  # source kept across crash
        zs.HOOKS.clear()
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 0, txt)
        for p in (self.p1, self.p2):
            self.assertFalse(os.path.exists(p))
            self.assertEqual(self.zsha(p + '.zst'), hashlib.sha256(self.orig[p]).hexdigest())
        self.assertIn('done_resumed', txt)

    def test_crash_mid_compress_cleans_tmp_on_restart(self):
        class Crash(BaseException):
            pass

        def boom(**_):
            raise Crash()
        zs.HOOKS['after_compress'] = boom
        with self.assertRaises(Crash):
            self.run_mode('execute', self.sha)
        self.assertEqual(self.leftovers(), [])  # cleaned in except path
        # simulate SIGKILL leftover: journal begin with an orphan tmp
        j = zs.Journal(os.path.join(self.state, 'journal.jsonl'))
        orphan = os.path.join(self.d2, '.rollout.pkl.zst.tmp-999.1')
        open(orphan, 'wb').write(b'junk')
        j.write(ev='begin', rel=os.path.relpath(self.p2, self.root), tmp=orphan, mode='execute')
        zs.HOOKS.clear()
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 0, txt)
        self.assertFalse(os.path.exists(orphan))
        self.assertFalse(os.path.exists(self.p2))

    def test_foreign_zst_conflict_preserved(self):
        open(self.p1 + '.zst', 'wb').write(b'not ours')
        rc, txt = self.run_mode('execute', self.sha)
        self.assertTrue(os.path.exists(self.p1))
        self.assertEqual(open(self.p1 + '.zst', 'rb').read(), b'not ours')
        self.assertIn('conflict_final_exists', txt)

    def test_rsync_partial_defers(self):
        open(os.path.join(self.d1, '.rollout.pkl.AbC123'), 'wb').close()
        rc, txt = self.run_mode('execute', self.sha)
        self.assertTrue(os.path.exists(self.p1))
        self.assertIn('deferred', txt)

    def test_open_writer_defers(self):
        with open(self.p1, 'rb'):
            # our own pid is skipped by design; spawn a holder process
            import subprocess
            h = subprocess.Popen([sys.executable, '-c', 'import sys,time;f=open(sys.argv[1],"rb");print(1,flush=True);time.sleep(30)',
                                  self.p1], stdout=subprocess.PIPE)
            h.stdout.readline()
            try:
                rc, txt = self.run_mode('execute', self.sha)
            finally:
                h.kill()
                h.wait()
        self.assertTrue(os.path.exists(self.p1))
        self.assertIn('open_by_process', txt)

    def test_low_disk_stops(self):
        rc, txt = run_cli('run', '--manifest', os.path.join(self.out, 'manifest.tsv'), '--manifest-sha256', self.sha,
                          '--state-dir', self.state, '--mode', 'execute', '--min-age-sec', '0',
                          '--min-free-gb', '1000000', '--lock-dir', self.lockdir)
        self.assertEqual(rc, zs.EXIT_LOW_DISK)
        for p in self.orig:
            self.assertTrue(os.path.exists(p))
            self.assertFalse(os.path.exists(p + '.zst'))

    def test_stop_file(self):
        os.makedirs(self.state)
        open(os.path.join(self.state, 'STOP'), 'w').close()
        rc, _ = self.run_mode('execute', self.sha)
        self.assertEqual(rc, zs.EXIT_INTERRUPTED)
        self.assertTrue(all(os.path.exists(p) for p in self.orig))

    def test_lock(self):
        os.makedirs(self.state)
        fd = zs._flock(os.path.join(self.state, 'LOCK'))
        try:
            rc, txt = self.run_mode('execute', self.sha)
        finally:
            os.close(fd)
        self.assertEqual(rc, zs.EXIT_LOCKED)
        other = os.path.join(self.tmp, 'state2')
        name = 'zstd_store_%s.lock' % hashlib.sha256(self.root.encode()).hexdigest()[:12]
        fd = zs._flock(os.path.join(self.lockdir, name))
        try:
            rc, _ = self.run_mode('execute', self.sha, state=other)
        finally:
            os.close(fd)
        self.assertEqual(rc, zs.EXIT_LOCKED)
        self.assertTrue(all(os.path.exists(p) for p in self.orig))

    def test_manifest_tamper_refused(self):
        rc, txt = self.run_mode('execute', '0' * 64)
        self.assertEqual(rc, 1)
        mp = os.path.join(self.out, 'manifest.tsv')
        with open(mp, 'a') as f:
            f.write('x\n')
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 1)
        self.assertTrue(all(os.path.exists(p) for p in self.orig))

    def test_state_dir_bound_to_mode(self):
        rc, _ = self.run_mode('verify', self.sha)
        self.assertEqual(rc, 0)
        rc, txt = self.run_mode('execute', self.sha)
        self.assertEqual(rc, 1)
        self.assertIn('bound', txt)

    def test_symlink_swapped_in_after_manifest(self):
        os.unlink(self.p1)
        target = os.path.join(self.tmp, 'elsewhere.pkl')
        open(target, 'wb').write(self.orig[self.p1])
        os.symlink(target, self.p1)
        rc, txt = self.run_mode('execute', self.sha)
        self.assertTrue(os.path.islink(self.p1))
        self.assertTrue(os.path.exists(target))
        self.assertFalse(os.path.exists(self.p1 + '.zst'))

    def test_link_created_after_manifest_is_skipped(self):
        ws = os.path.join(self.tmp, 'ws')
        os.makedirs(ws)
        os.symlink(self.d1, os.path.join(ws, 'alias'))
        rc, txt = self.run_mode('execute', self.sha, '--external-link-scan', ws)
        self.assertEqual(rc, 0, txt)
        self.assertIn('"skip_link_target_now": 1', txt)
        self.assertTrue(os.path.exists(self.p1))
        self.assertFalse(os.path.exists(self.p1 + '.zst'))
        self.assertFalse(os.path.exists(self.p2))

    def test_sample(self):
        scratch = os.path.join(self.tmp, 'scratch')
        rel = os.path.relpath(self.p1, self.root)
        before = zs.stat_sig(os.lstat(self.p1))
        rc, txt = run_cli('sample', '--root', self.root, '--scratch', scratch, '--rel', rel, '--min-free-gb', '0')
        self.assertEqual(rc, 0, txt)
        out = json.loads(txt)
        self.assertTrue(out['all_roundtrip_ok'])
        self.assertEqual(os.listdir(scratch), [])
        self.assertEqual(zs.stat_sig(os.lstat(self.p1)), before)


if __name__ == '__main__':
    unittest.main()
