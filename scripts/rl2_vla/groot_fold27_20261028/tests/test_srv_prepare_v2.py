"""srv_prepare_v2.sh remote command strings, executed locally (SRV_PREPARE_LOCAL_DEPLOY) against a throwaway deploy clone that
holds the deploy commit history but NOT the AE commit: git-bundle route -> worktree -> blob check; record counting."""
import shutil, subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SH = ROOT / 'fold_data/srv_prepare_v2.sh'
O = 'outputs/analysis/groot_fold27_20261028'
DEPLOY_SHA, AE_SHA = 'c354946c222a0c1be4c4b6a8029eef06231b5a36', '8070d4c5a581e5734304c763a1c1684de5d957cd'
BUNDLES = REPO / O / 'git_bundles'


def sh(*a, cwd=None, env=None, check=True):
    return subprocess.run(list(a), cwd=cwd, env=env, capture_output=True, text=True, check=check)


@unittest.skipUnless((BUNDLES / 'BUNDLES.json').exists(), 'git bundles not built')
class SrvPrepareV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Path(tempfile.mkdtemp(prefix='srvprep_v2_'))
        d = cls.d = cls.t / 'deploy'
        sh('git', 'init', '-q', str(d))
        sh('git', 'fetch', '-q', '--depth', '30', f'file://{REPO}', DEPLOY_SHA, cwd=d)     # no alternates: real isolation
        sh('git', 'update-ref', '--no-deref', 'HEAD', DEPLOY_SHA, cwd=d)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.t)

    def run_step(self, host, step, check=True):
        import os
        env = dict(os.environ, SRV_PREPARE_LOCAL_DEPLOY=str(self.d))
        return sh('bash', str(SH), host, step, env=env, check=check)

    def test_1_bundle_route_worktree_and_blobs(self):
        self.assertNotEqual(sh('git', 'cat-file', '-e', f'{AE_SHA}^{{commit}}', cwd=self.d, check=False).returncode, 0)
        r = self.run_step('srv50', 'ae-worktree', check=False)
        self.assertNotEqual(r.returncode, 0)                                    # no commit yet -> refuses, no source copied
        self.assertFalse((self.d / '.worktrees').exists())
        (self.d / O / 'git_bundles').mkdir(parents=True)
        shutil.copy(BUNDLES / 'trqam_retention.bundle', self.d / O / 'git_bundles/')   # stands in for the bundle transfer
        self.assertIn('AE_COMMIT_FETCHED', self.run_step('srv50', 'ae-fetch-bundle').stdout)
        out = self.run_step('srv50', 'ae-worktree').stdout
        n = int(out.split('AE_CLOSURE_BLOBS_OK ')[1].split('/')[0])
        self.assertGreaterEqual(n, 20); self.assertIn(f'{n}/{n}', out)
        wt = self.d / '.worktrees/trqam-retention'
        for f in ('scripts/rl2_vla/ae_cluster_reward/tc_import.py', 'scripts/rl2_vla/stage2_cluster_reward/clusters.py',
                  'scripts/analysis/grid_phase/ae_cluster.py', 'manuscript/numbers.md'):
            self.assertEqual((wt / f).read_bytes(), (REPO / '.worktrees/trqam-retention' / f).read_bytes())
        # tampering a closure file is caught
        p = wt / 'manuscript/numbers.md'; orig = p.read_bytes(); p.write_bytes(orig + b'x')
        self.assertNotEqual(self.run_step('srv50', 'ae-worktree', check=False).returncode, 0)
        p.write_bytes(orig)
        self.assertEqual(self.run_step('srv50', 'ae-worktree').returncode, 0)

    def test_2_bundle_sha_gate(self):
        b = self.d / O / 'git_bundles/trqam_retention.bundle'
        if not b.exists():
            self.skipTest('order')
        keep = b.read_bytes(); b.write_bytes(keep + b'\0')
        r = self.run_step('srv50', 'ae-fetch-bundle', check=False)
        self.assertNotEqual(r.returncode, 0); self.assertIn('BUNDLE_SHA_MISMATCH', r.stdout)
        b.write_bytes(keep)

    def test_3_verify_counts_episode_records_not_json(self):
        for name in ('srv48_dishrack_left_s3', 'srv48_dishrack_right_s1', 'srv48_ovenrack_left_s2'):
            d = self.d / O / 'full_episodes' / name; d.mkdir(parents=True, exist_ok=True)
            (d / 'contract.json').write_text('{}')
            for i in range(200):
                (d / f'{i:024x}.json').write_text('{}'); (d / f'{i:024x}.npz').write_bytes(b'n')
        c = self.d / 'outputs/rl2_vla/fold27_mil_20261030_continuation/compact_verified'; c.mkdir(parents=True, exist_ok=True)
        (c / 'contract.json').write_text('{}')
        out = self.run_step('srv48', 'verify').stdout
        self.assertEqual(out.count('records=200'), 3); self.assertNotIn('records=201', out)
        orphan = self.d / O / 'full_episodes/srv48_ovenrack_left_s2' / f'{0:024x}.npz'
        orphan.unlink()                                                         # json without npz is not a record
        self.assertNotEqual(self.run_step('srv48', 'verify', check=False).returncode, 0)
        orphan.write_bytes(b'n')

    def test_0_push_ssh_route_into_separate_clone(self):
        d2 = self.t / 'deploy2'
        sh('git', 'init', '-q', str(d2)); sh('git', 'fetch', '-q', '--depth', '30', f'file://{REPO}', DEPLOY_SHA, cwd=d2)
        sh('git', 'update-ref', '--no-deref', 'HEAD', DEPLOY_SHA, cwd=d2)
        import os
        r = sh('bash', str(SH), 'srv48', 'ae-pack-rc', env=dict(os.environ, SRV_PREPARE_LOCAL_DEPLOY=str(d2)))
        self.assertIn('AE_COMMIT_FETCHED', r.stdout)
        self.assertEqual(sh('git', 'rev-parse', 'HEAD', cwd=d2).stdout.strip(), DEPLOY_SHA)          # deploy HEAD untouched
        r = sh('bash', str(SH), 'srv48', 'ae-worktree', env=dict(os.environ, SRV_PREPARE_LOCAL_DEPLOY=str(d2)))
        self.assertIn('AE_CLOSURE_BLOBS_OK', r.stdout)

    def test_5_shared_common_dir_original_untouched(self):
        """Real layout: deploy = worktree of the original checkout (shared .git). Route must leave the original's
        HEAD/branch/index/status unchanged; only the pin ref, objects and worktree metadata are added."""
        import hashlib, os
        orig = self.t / 'original'
        sh('git', 'init', '-q', str(orig)); sh('git', 'fetch', '-q', '--depth', '30', f'file://{REPO}', DEPLOY_SHA, cwd=orig)
        sh('git', 'update-ref', 'refs/heads/eval_whole_pipe_1', DEPLOY_SHA, cwd=orig)
        sh('git', 'symbolic-ref', 'HEAD', 'refs/heads/eval_whole_pipe_1', cwd=orig); sh('git', 'read-tree', 'HEAD', cwd=orig)
        dep = self.t / 'shared_deploy'
        sh('git', 'worktree', 'add', '-q', '--no-checkout', '--detach', str(dep), DEPLOY_SHA, cwd=orig)

        def snap():
            return (sh('git', 'symbolic-ref', 'HEAD', cwd=orig).stdout, sh('git', 'rev-parse', 'HEAD', cwd=orig).stdout,
                    hashlib.sha256((orig / '.git/index').read_bytes()).hexdigest(),
                    hashlib.sha256(sh('git', 'status', '--porcelain', cwd=orig).stdout.encode()).hexdigest(),
                    sorted(p.name for p in orig.iterdir()))
        before = snap()
        env = dict(os.environ, SRV_PREPARE_LOCAL_DEPLOY=str(dep))
        self.assertEqual(sh('git', 'rev-parse', '--path-format=absolute', '--git-common-dir', cwd=dep).stdout.strip(), str(orig / '.git'))
        self.assertIn('AE_COMMIT_FETCHED', sh('bash', str(SH), 'srv48', 'ae-pack-rc', env=env).stdout)
        self.assertIn('AE_CLOSURE_BLOBS_OK', sh('bash', str(SH), 'srv48', 'ae-worktree', env=env).stdout)
        self.assertEqual(snap(), before)
        self.assertEqual(sh('git', 'rev-parse', 'refs/fold27/pin/trqam_retention', cwd=orig).stdout.strip(), AE_SHA)   # disclosed
        wts = sh('git', 'worktree', 'list', '--porcelain', cwd=orig).stdout
        self.assertIn(str(dep / '.worktrees/trqam-retention'), wts)
        self.assertIn('common=', sh('bash', str(SH), 'srv48', 'check', env=env).stdout)

    def test_4_no_source_push_data_anywhere(self):
        src = SH.read_text()
        self.assertNotIn('push-data', [l.split('#')[0] for l in src.splitlines()].__str__())
        self.assertIn('git worktree add --detach', src)
        self.assertNotIn('git push', src)                          # only the remote_compute.sh channel is used
        self.assertNotIn('ssh://', src)


if __name__ == '__main__':
    unittest.main()
