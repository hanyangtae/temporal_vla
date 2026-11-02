"""iso_env_container plan/gate + run_eval_arm_v2 explicit container selection (no docker needed: inspect dicts injected)."""
import copy, json, os, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/runtime'))
import common as C               # noqa: E402
import run_eval_arm as RA        # noqa: E402
import iso_env_container as ISO  # noqa: E402
import run_eval_arm_v2 as V2     # noqa: E402

DEPLOY = str(C.R)
IMG = 'sha256:920c3d7c2e7394b359b9fb39a0aa54ea3f4fcf964aeff484b26c80191e477dc5'
ORIG = {'Name': '/robocasa', 'Image': IMG,
        'HostConfig': {'Binds': ['/home/u/.cache/temporal_vla:/cache:rw', '/home/u/pkt_ws/temporal_vla:/temporal_vla:rw'],
                       'NetworkMode': 'host', 'IpcMode': 'private', 'ShmSize': 67108864},
        'Config': {'Env': ['NVIDIA_VISIBLE_DEVICES=2', 'PYTHONPATH=/temporal_vla:/temporal_vla/src/benchmarks/robocasa', 'VLA_CACHE_ROOT=/cache'],
                   'User': 'junhyeong', 'WorkingDir': '/temporal_vla', 'Entrypoint': ['/temporal_vla/scripts/start_vnc.sh']}}


def iso(gpu=3, src=DEPLOY, rw=True, image=IMG, running=True, labels=None, env_gpu=None):
    return {'Name': f'/{ISO.iso_name(gpu)}', 'Image': image, 'State': {'Running': running},
            'Mounts': [{'Destination': '/cache', 'Source': '/home/u/.cache/temporal_vla', 'RW': True},
                       {'Destination': '/temporal_vla', 'Source': src, 'RW': rw}],
            'Config': {'Env': [f'NVIDIA_VISIBLE_DEVICES={gpu if env_gpu is None else env_gpu}', 'VLA_CACHE_ROOT=/cache'],
                       'Labels': labels if labels is not None else {'fold27.deploy': os.path.realpath(DEPLOY), 'fold27.gpu': str(gpu)}}}


class IsoEnvTest(unittest.TestCase):
    def test_plan_mirrors_original_but_isolates_repo_and_gpu(self):
        argv = ISO.plan(ORIG, 3, DEPLOY)
        s = ' '.join(argv)
        self.assertEqual(argv[:5], ['docker', 'run', '-d', '--name', 'fold27-iso-env-gpu3'])
        self.assertIn(f'-v {os.path.realpath(DEPLOY)}:/temporal_vla:rw', s)
        self.assertIn('-v /home/u/.cache/temporal_vla:/cache:rw', s)
        self.assertNotIn('/home/u/pkt_ws/temporal_vla:/temporal_vla', s)              # original repo not mounted
        self.assertIn('-e NVIDIA_VISIBLE_DEVICES=3', s); self.assertNotIn('NVIDIA_VISIBLE_DEVICES=2', s)
        self.assertIn('--gpus device=3', s); self.assertIn('--network host', s); self.assertIn('--user junhyeong', s)
        self.assertEqual(argv[-2:], [IMG, 'infinity'])
        self.assertIn('--entrypoint sleep', s); self.assertNotIn('start_vnc', s)
        self.assertIn('-e PYTHONPATH=/temporal_vla:/temporal_vla/src/benchmarks/robocasa', s)
        bad = copy.deepcopy(ORIG); bad['HostConfig']['Binds'][1] = f'{DEPLOY}:/temporal_vla:rw'
        with self.assertRaises(ISO.GateError):
            ISO.plan(bad, 3, DEPLOY)

    def test_gate(self):
        self.assertEqual(ISO.gate(iso(), 3, DEPLOY, IMG), IMG)
        for info, gpu, img in ((iso(src='/home/u/pkt_ws/temporal_vla'), 3, None), (iso(rw=False), 3, None),
                               (iso(env_gpu=2), 3, None), (iso(), 2, None), (iso(image='sha256:x'), 3, IMG),
                               (iso(running=False), 3, None), (iso(labels={}), 3, None)):
            with self.assertRaises(ISO.GateError):
                ISO.gate(info, gpu, DEPLOY, img)
        two = iso(); two['Mounts'].append({'Destination': '/temporal_vla', 'Source': DEPLOY, 'RW': True})
        with self.assertRaises(ISO.GateError):
            ISO.gate(two, 3, DEPLOY)


class RunEvalArmV2Test(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.saved = (C.SRV_ENV_CONTAINER, RA.srv_env_preflight)
        RA.srv_env_preflight = lambda out: IMG                   # stands in for the original preflight (docker inspect)

    def tearDown(self):
        C.SRV_ENV_CONTAINER, RA.srv_env_preflight = self.saved
        shutil.rmtree(self.t)

    def man(self, host):
        p = self.t / f'{host}.json'; p.write_text(json.dumps({'host': host})); return str(p)

    def args(self, host, *extra):
        return ['--manifest', self.man(host), '--arm', 'mil_qam65', '--gpu', '3', '--port', '8896', *extra]

    def test_srv_requires_explicit_isolated_container(self):
        with self.assertRaises(Exception):
            V2.configure(self.args('srv50'), inspect=lambda n: iso())
        with self.assertRaises(Exception):
            V2.configure(self.args('srv50', '--env-container', 'robocasa', '--expect-repo', DEPLOY), inspect=lambda n: iso())
        with self.assertRaises(Exception):
            V2.configure(self.args('srv50', '--env-container', 'fold27-iso-env-gpu3', '--expect-repo', '/elsewhere'), inspect=lambda n: iso())
        self.assertEqual(C.SRV_ENV_CONTAINER, 'robocasa')       # nothing changed on refusal

    def test_selection_propagates_to_preflight_and_exec(self):
        seen = []
        rest = V2.configure(self.args('srv50', '--env-container', 'fold27-iso-env-gpu3', '--expect-repo', DEPLOY, '--expect-image', IMG),
                            inspect=lambda n: (seen.append(n), iso())[1])
        self.assertNotIn('--env-container', rest); self.assertIn('--gpu', rest)
        self.assertEqual(C.SRV_ENV_CONTAINER, 'fold27-iso-env-gpu3')
        self.assertEqual(RA.srv_env_preflight('out'), IMG); self.assertEqual(seen, ['fold27-iso-env-gpu3'])
        # wrong GPU env inside the container -> preflight (also the per-episode check) fails
        V2.RA.srv_env_preflight = lambda out: IMG
        V2.configure(self.args('srv50', '--env-container', 'fold27-iso-env-gpu3', '--expect-repo', DEPLOY),
                     inspect=lambda n: iso(env_gpu=2))
        with self.assertRaises(RuntimeError):
            RA.srv_env_preflight('out')

    def test_kanu_refuses_selection(self):
        a = self.args('kanu'); self.assertEqual(V2.configure(a), a)
        with self.assertRaises(Exception):
            V2.configure(self.args('kanu', '--env-container', 'x'))


if __name__ == '__main__':
    unittest.main()
