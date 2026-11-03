"""One-time GPU sharing contract (gpu_sharing_authorization.py) + its only consumer hook in groot_harness.probe_gpus.
Fake host probe only: no nvidia-smi, no /proc of real processes, no GPU, no harness ledger of any real machine."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

UTILS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTILS))
import gpu_sharing_authorization as SA  # noqa: E402

spec = importlib.util.spec_from_file_location('groot_harness', UTILS / 'groot_harness.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

CMD = '/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 flow_policy_prev/flow.py'
UUID = 'GPU-9fdab52c-2f3e-38e1-1fcd-855c290ef8c3'
BOOT = '521dd1f3-71bf-437d-8fdb-11505e397d7f'
SESSION = 'fold27-v2-eval'


def ident(cmd, uid=0, start=1374316127):
    return {'uid': uid, 'start_ticks': start, 'cmdline_sha256': hashlib.sha256(cmd.replace(' ', '\0').encode()).hexdigest(), 'cmdline': cmd}


class FakeProbe:
    def __init__(self):
        self.host, self.boot, self.t = 'worker2', BOOT, 1000.0
        self.procs = {1101996: ident(CMD)}
        self.g = {'uuid': UUID, 'free_mib': 60057, 'total_mib': 81920, 'apps': [{'pid': 1101996, 'used_mib': 20970}]}
        self.fail = False

    def boot_id(self): return self.boot
    def hostname(self): return self.host
    def now(self): return self.t
    def proc(self, pid): return copy.deepcopy(self.procs.get(int(pid)))

    def gpu(self, index):
        if self.fail:
            raise SA.AuthError(f'GPU{index}: nvidia-smi query failed')
        return copy.deepcopy(self.g)


def args(out, **kw):
    a = dict(machine='srv50', gpu=3, gpu_uuid=UUID, session=SESSION, experiment='fold27', pid=1101996, expect_uid=0,
             expect_cmd=CMD, max_used_mib=22000, max_serves=2, kind='eval', per_serve_mib=6118, reserve_mib=8192,
             expires_in_h=24, approval='user: srv50 GPU3 one-time sharing for this fold27 experiment', out=out)
    a.update(kw)
    return argparse.Namespace(**a)


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.p = FakeProbe()
        self.path = self.t / 'auth.json'
        SA.prepare(args(self.path), self.p)
        self.auth = SA.load(self.path)

    def ok(self, **kw):
        req = dict(machine='srv50', gpus=[3], session=SESSION, kind='eval', serves=2)
        req.update(kw)
        return SA.authorized_processes(self.auth, req['machine'], req['gpus'], req['session'], req['kind'], req['serves'], self.p)

    def refused(self, msg, **kw):
        with self.assertRaisesRegex(SA.AuthError, msg):
            self.ok(**kw)

    def test_prepare_pins_full_identity(self):
        f = self.auth['foreign'][0]
        self.assertEqual((f['pid'], f['uid'], f['start_ticks'], f['cmdline']), (1101996, 0, 1374316127, CMD))
        self.assertEqual((self.auth['gpu_uuid'], self.auth['boot_id'], self.auth['max_serves']), (UUID, BOOT, 2))
        with self.assertRaises(FileExistsError):
            SA.prepare(args(self.path), self.p)                                   # never overwrites a contract

    def test_prepare_refuses_mismatched_expectations(self):
        for i, (kw, msg) in enumerate(((dict(expect_cmd=CMD + ' --x'), 'cmdline'), (dict(expect_uid=1000), 'uid'),
                                       (dict(gpu_uuid='GPU-x'), 'uuid'), (dict(max_serves=7), 'max_serves'),
                                       (dict(max_used_mib=1000), 'already uses'))):
            with self.assertRaisesRegex(SA.AuthError, msg):
                SA.prepare(args(self.t / f'x{i}.json', **kw), self.p)
            self.assertFalse((self.t / f'x{i}.json').exists())                  # refused before anything is written
        self.p.g['apps'].append({'pid': 5, 'used_mib': 10})
        with self.assertRaisesRegex(SA.AuthError, 'exactly the approved PID'):
            SA.prepare(args(self.t / 'y.json'), self.p)
        self.p.g['apps'].pop(); self.p.host = 'worker1'
        with self.assertRaisesRegex(SA.AuthError, 'authorized host'):
            SA.prepare(args(self.t / 'z.json'), self.p)

    def test_exact_identity_accepted(self):
        r = self.ok()
        self.assertEqual(r['approved_present'], [1101996]); self.assertEqual(r['required_mib'], 2 * 6118 + 8192)

    def test_scope_refusals(self):
        self.refused('GPUs', gpus=[2]); self.refused('GPUs', gpus=[3, 4]); self.refused('session', session='fold27-trqam-v2')
        self.refused('machine', machine='srv48'); self.refused('kind', kind='collect'); self.refused('serves', serves=3)
        self.p.host = 'worker1'; self.refused('host')

    def test_identity_refusals(self):
        for mut, msg in ((lambda: self.p.procs[1101996].update(start_ticks=1374399999), 'start_ticks'),
                         (lambda: self.p.procs[1101996].update(uid=1000), 'uid'),
                         (lambda: self.p.procs.__setitem__(1101996, ident(CMD + ' --other')), 'cmdline_sha256'),
                         (lambda: self.p.procs.pop(1101996), 'unverifiable'),
                         (lambda: self.p.g['apps'].append({'pid': 777, 'used_mib': 100}), 'unapproved compute process PID 777'),
                         (lambda: self.p.g.update(uuid='GPU-other'), 'uuid'),
                         (lambda: setattr(self.p, 'boot', 'other-boot'), 'boot id'),
                         (lambda: setattr(self.p, 't', self.auth['expires_at'] + 1), 'expired')):
            self.setUp()
            mut()
            self.refused(msg)

    def test_memory_refusals(self):
        self.p.g['free_mib'] = 2 * 6118 + 8192 - 1; self.refused('required')
        self.setUp(); self.p.g['apps'][0]['used_mib'] = 22001; self.refused('uses 22001')
        self.setUp(); self.p.fail = True; self.refused('query failed')

    def test_ended_task_is_plain_empty_gpu_and_never_a_replacement(self):
        self.p.g['apps'] = []; self.p.procs.pop(1101996)
        self.assertEqual(self.ok()['approved_present'], [])
        self.p.procs[2000000] = ident(CMD, start=1400000000)                     # same command, new task
        self.p.g['apps'] = [{'pid': 2000000, 'used_mib': 20000}]
        self.refused('unapproved compute process PID 2000000')
        self.p.procs[1101996] = ident(CMD, start=1400000001)                     # PID reused by a new task
        self.p.g['apps'] = [{'pid': 1101996, 'used_mib': 20000}]
        self.refused('start_ticks')

    def test_load_requires_pinned_identity_and_policy_bounds(self):
        a = json.loads(self.path.read_text())
        bad = copy.deepcopy(a); bad['foreign'] = [{'pid': 1101996}]
        (self.t / 'b1.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(SA.AuthError, 'PID alone'):
            SA.load(self.t / 'b1.json')
        bad = copy.deepcopy(a); bad['max_serves'] = 7
        (self.t / 'b2.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(SA.AuthError, 'max_serves'):
            SA.load(self.t / 'b2.json')
        bad = copy.deepcopy(a); del bad['expires_at']
        (self.t / 'b3.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(SA.AuthError, 'missing'):
            SA.load(self.t / 'b3.json')

    def test_matches_identity(self):
        self.assertTrue(SA.matches_identity(self.auth, 1101996, self.p))
        self.assertFalse(SA.matches_identity(self.auth, 777, self.p))
        self.p.procs[1101996]['start_ticks'] += 1
        self.assertFalse(SA.matches_identity(self.auth, 1101996, self.p))


class HarnessHookTest(unittest.TestCase):
    REQ = {'machine': 'srv50', 'gpus': [3], 'serves': 2, 'session': SESSION, 'kind': 'eval'}

    def test_default_path_unchanged(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(h, 'check_gpu_processes') as probe:
            h.probe_gpus(self.REQ)
        probe.assert_called_once_with([3])

    def test_contract_path_accepts_and_refuses(self):
        t = Path(tempfile.mkdtemp()); p = FakeProbe(); path = t / 'auth.json'
        SA.prepare(args(path), p)
        with patch.dict(os.environ, {'HARNESS_SHARING_AUTH': str(path)}), patch.object(SA, 'Probe', return_value=p), \
                patch.object(h, 'check_gpu_processes') as default_probe:
            h.probe_gpus(self.REQ)                                                # exact identity: admitted
            default_probe.assert_not_called()
            p.g['apps'].append({'pid': 777, 'used_mib': 100})
            with self.assertRaisesRegex(ValueError, 'sharing authorization refused: .*777'):
                h.probe_gpus(self.REQ)
            p.g['apps'].pop()
            with self.assertRaisesRegex(ValueError, 'GPUs'):
                h.probe_gpus({**self.REQ, 'gpus': [2]})
        with patch.dict(os.environ, {'HARNESS_SHARING_AUTH': str(t / 'missing.json')}):
            with self.assertRaisesRegex(ValueError, 'sharing authorization refused'):
                h.probe_gpus(self.REQ)

    def test_run_keeps_reservation_when_contract_fails_at_release(self):
        t = Path(tempfile.mkdtemp()); root = t / 'leases'
        calls = []

        def probe(req):
            calls.append(req['gpus'])
            if len(calls) == 3:
                raise ValueError('sharing authorization refused: unapproved compute process PID 777')
        a = argparse.Namespace(machine='kanu', gpus='5', serves=1, session='s', kind='eval', command=['--', sys.executable, '-c', 'pass'])
        with patch.object(h, 'broker', side_effect=lambda r: h.transact(root, r)), patch.object(h, 'probe_gpus', side_effect=probe), \
                patch.object(h.socket, 'gethostname', return_value='kanu'), patch.object(h.signal, 'signal'), \
                patch.object(h, 'policy', return_value={'coordinator_host': 'kanu', 'machines': {'kanu': {
                    'max_gpus_all_sessions': 3, 'max_serves_per_gpu': 2, 'hostname': 'kanu'}}}):
            self.assertEqual(h.run(a), 0)
        state = json.loads((root / 'harness_slots.json').read_text())
        self.assertEqual(len(state), 1)                                           # receipt retained, never released early
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
