"""v2 sharing contract: pinned process + pinned foreign CONTAINER INSTANCE with dynamically changing verified children
+ multiple sessions (eval, existing fold27 training). Fake host probe only (no nvidia-smi, docker or real /proc)."""
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

spec = importlib.util.spec_from_file_location('groot_harness_v2t', UTILS / 'groot_harness.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

UUID = 'GPU-9fdab52c-2f3e-38e1-1fcd-855c290ef8c3'
BOOT = '521dd1f3-71bf-437d-8fdb-11505e397d7f'
CID = '4ef7ae84e3da499cbbe725775fbf121c3737231bd4a5232264f423a89319327b'
IMG = 'sha256:3e08b634ae0396d1ffff046027ec716914a3d041aa4c541106eeb28575926e54'
ROOT_CMD = '/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 flow_policy_prev/flow.py'
SESS = {'fold27-v2-eval': {'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118},
        'fold27-trqam-v2': {'allowed_kind': 'eval', 'max_serves': 1, 'per_serve_mib': 12000}}


def P(uid, gid, start, ppid, nspid, cgroup, cmd):
    return {'uid': uid, 'gid': gid, 'start_ticks': start, 'ppid': ppid, 'nspid': nspid, 'cgroup': cgroup, 'cmd': cmd}


class Probe2:
    def __init__(self):
        self.host, self.boot, self.t, self.fail = 'worker2', BOOT, 1000.0, False
        dock = f'0::/docker/{CID}'
        self.procs = {1101996: P(0, 0, 1374316127, 1, [1101996], '0::/system.slice/isaac.scope', ROOT_CMD),
                      3630028: P(1019, 1018, 1375757165, 3629999, [3630028, 1], dock, 'bash'),
                      3666545: P(1019, 1018, 1375773000, 3630028, [3666545, 40], dock, 'python -B -'),
                      3677646: P(1019, 1018, 1375777899, 3666545, [3677646, 41], dock, 'python -B -')}
        self.apps = [{'pid': 1101996, 'used_mib': 20970}]
        self.free = 60057

    def boot_id(self): return self.boot
    def hostname(self): return self.host
    def now(self): return self.t

    def proc(self, pid):
        p = self.procs.get(int(pid))
        if p is None:
            return None
        return {'uid': p['uid'], 'start_ticks': p['start_ticks'], 'cmdline': p['cmd'],
                'cmdline_sha256': hashlib.sha256(p['cmd'].replace(' ', '\0').encode()).hexdigest()}

    def status(self, pid):
        p = self.procs.get(int(pid))
        if p is None:
            return None
        return {'PPid': f'\t{p["ppid"]}', 'Uid': f'\t{p["uid"]}\t{p["uid"]}\t{p["uid"]}\t{p["uid"]}',
                'Gid': f'\t{p["gid"]}\t{p["gid"]}', 'NSpid': '\t' + '\t'.join(map(str, p['nspid']))}

    def cgroup(self, pid):
        p = self.procs.get(int(pid))
        return None if p is None else p['cgroup'] + '\n'

    def gpu(self, index):
        if self.fail:
            raise SA.AuthError('nvidia-smi failed')
        return {'uuid': UUID, 'free_mib': self.free, 'total_mib': 81920, 'apps': [a for a in self.apps if a['pid'] in self.procs]}


def inspect(**kw):
    i = {'Id': CID, 'Name': '/jolly_wright', 'Image': IMG, 'State': {'Running': True, 'Pid': 3630028, 'StartedAt': '2026-11-03T11:02:09.728560927Z'},
         'Config': {'User': '1019:1018'}, 'HostConfig': {'DeviceRequests': [{'Driver': '', 'Count': 0, 'DeviceIDs': [UUID], 'Capabilities': [['gpu']]}]}}
    for k, v in kw.items():
        cur = i
        *path, last = k.split('.')
        for p in path:
            cur = cur[p]
        cur[last] = v
    return [i]


class V2Test(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.p = Probe2()
        v1 = self.t / 'v1.json'
        SA.prepare(argparse.Namespace(machine='srv50', gpu=3, gpu_uuid=UUID, session='fold27-v2-eval', experiment='fold27', pid=1101996,
                                      expect_uid=0, expect_cmd=ROOT_CMD, max_used_mib=22000, max_serves=2, kind='eval', per_serve_mib=6118,
                                      reserve_mib=8192, expires_in_h=24, approval='user one-time', out=v1), self.p)
        self.ins = self.t / 'inspect.json'; self.ins.write_text(json.dumps(inspect()))
        self.path = self.t / 'v2.json'
        SA.prepare_v2(self.args(), self.p)
        self.auth = SA.load(self.path)
        self.p.apps = [{'pid': 1101996, 'used_mib': 20970}, {'pid': 3666545, 'used_mib': 1200}, {'pid': 3677646, 'used_mib': 800}]

    def args(self, **kw):
        a = dict(machine='srv50', gpu=3, gpu_uuid=UUID, experiment='fold27', sessions=json.dumps(SESS), container_inspect=self.ins,
                 expect_container_id=CID, expect_image=IMG, expect_member_uid=1019, expect_member_gid=1018, container_max_used_mib=8000,
                 approval='user: srv50 GPU3 experiment one-time incl. sj container', pinned_from=self.t / 'v1.json',
                 launcher_evidence='docker run pid 3625178 uid 1019 --gpus device=' + UUID, reserve_mib=8192, expires_in_h=24, out=self.path)
        a.update(kw)
        return argparse.Namespace(**a)

    def ok(self, session='fold27-v2-eval', serves=2, **kw):
        return SA.authorized_processes(self.auth, kw.get('machine', 'srv50'), kw.get('gpus', [3]), session, kw.get('kind', 'eval'), serves, self.p)

    def refused(self, msg, **kw):
        with self.assertRaisesRegex(SA.AuthError, msg):
            self.ok(**kw)

    # ---------------- acceptance ----------------
    def test_changing_verified_children_accepted_for_eval_and_train(self):
        r = self.ok()
        self.assertEqual(r['approved_present'], [1101996]); self.assertEqual(sorted(r['container_members'][CID]), [3666545, 3677646])
        self.assertEqual(self.ok(session='fold27-trqam-v2', serves=1)['required_mib'], 12000 + 8192)
        self.p.procs[3699999] = P(1019, 1018, 1375800000, 3630028, [3699999, 50], f'0::/docker/{CID}', 'python -B -')
        self.p.apps.append({'pid': 3699999, 'used_mib': 100})                      # a NEW child of the SAME instance
        self.assertIn(3699999, self.ok()['container_members'][CID])

    # ---------------- container identity refusals ----------------
    def mutate_child(self, **kw):
        self.p.procs[3677646].update(kw)

    def test_member_refusals(self):
        cases = [(dict(cgroup='0::/docker/' + 'f' * 64), 'not in the pinned container cgroup'),
                 (dict(cgroup='0::/system.slice/user.scope'), 'not in the pinned container cgroup'),
                 (dict(uid=1015), 'uid/gid'), (dict(gid=1014), 'uid/gid'), (dict(nspid=[3677646]), 'nested pid namespace'),
                 (dict(start_ticks=1375700000), 'started before the container init'),
                 (dict(ppid=1248667), 'lineage')]
        for kw, msg in cases:
            self.setUp(); self.mutate_child(**kw)
            self.refused(msg)

    def test_host_process_same_uid_same_command_refused(self):
        self.p.procs[4000000] = P(1019, 1018, 1375900000, 1248667, [4000000], '0::/user.slice/user-1019.slice', 'python -B -')
        self.p.apps.append({'pid': 4000000, 'used_mib': 50})
        self.refused('unapproved compute process PID 4000000')

    def test_other_container_refused(self):
        other = 'a' * 64
        self.p.procs[4100000] = P(1019, 1018, 1375900000, 4099999, [4100000, 5], f'0::/docker/{other}', 'python -B -')
        self.p.procs[4099999] = P(1019, 1018, 1375890000, 1, [4099999, 1], f'0::/docker/{other}', 'bash')
        self.p.apps.append({'pid': 4100000, 'used_mib': 50})
        self.refused('unapproved compute process PID 4100000')

    def test_restarted_container_refused(self):
        self.p.procs[3630028]['start_ticks'] = 1375999999                         # new init instance, same pid
        self.refused('init identity changed')
        self.setUp(); self.p.procs.pop(3630028)
        self.refused('init identity changed')

    def test_reused_pid_in_lineage_refused(self):
        self.p.procs[3666545]['start_ticks'] = 1375790000                         # parent now newer than its child
        self.refused('lineage')

    def test_aggregate_container_memory(self):
        self.p.apps[1]['used_mib'] = 7300
        self.refused('container 4ef7ae84e3da uses 8100 MiB > authorized 8000')

    def test_pinned_root_still_strict(self):
        self.p.procs[1101996]['start_ticks'] += 1
        self.refused('pinned identity')

    def test_scope_and_memory(self):
        self.refused('not in authorized sessions', session='fold27-other')
        self.refused('serves 2 exceeds authorized 1', session='fold27-trqam-v2', serves=2)
        self.refused('GPUs', gpus=[2]); self.refused('machine', machine='srv48')
        self.p.free = 2 * 6118 + 8192 - 1; self.refused('required')
        self.setUp(); self.p.t = self.auth['expires_at']; self.refused('expired')
        self.setUp(); self.p.boot = 'x'; self.refused('boot id')
        self.setUp(); self.p.fail = True; self.refused('nvidia-smi failed')

    def test_approved_now_for_guardian(self):
        ok, v = SA.approved_now(self.auth, self.p.apps, self.p)
        self.assertEqual(ok, {1101996, 3666545, 3677646}); self.assertEqual(v, [])
        self.p.apps[1]['used_mib'] = 7300
        ok, v = SA.approved_now(self.auth, self.p.apps, self.p)
        self.assertEqual(ok, {1101996}); self.assertTrue(v)                       # whole over-limit container untolerated
        self.p.t = self.auth['expires_at'] + 1
        self.assertEqual(SA.approved_now(self.auth, self.p.apps, self.p)[0], set())

    # ---------------- prepare / load ----------------
    def test_prepare_v2_refusals(self):
        bad = [(dict(), dict(Id='b' * 64), 'inspect Id'), (dict(), dict(Image='sha256:x'), 'inspect Image'),
               (dict(), {'HostConfig.DeviceRequests': [{'Count': -1, 'DeviceIDs': None}]}, 'GPU devices'),
               (dict(), {'HostConfig.DeviceRequests': [{'Count': 0, 'DeviceIDs': [UUID, 'GPU-other']}]}, 'GPU devices'),
               (dict(), {'Config.User': '1019'}, 'Config.User'), (dict(), {'State.Running': False}, 'not running')]
        for i, (akw, ikw, msg) in enumerate(bad):
            ins = self.t / f'ins{i}.json'; ins.write_text(json.dumps(inspect(**ikw)))
            with self.assertRaisesRegex(SA.AuthError, msg):
                SA.prepare_v2(self.args(container_inspect=ins, out=self.t / f'o{i}.json', **akw), self.p)
            self.assertFalse((self.t / f'o{i}.json').exists())
        self.p.procs[3630028]['nspid'] = [3630028]
        with self.assertRaisesRegex(SA.AuthError, 'NSpid'):
            SA.prepare_v2(self.args(out=self.t / 'n.json'), self.p)
        self.setUp(); self.p.procs[3630028]['cgroup'] = '0::/docker/' + 'c' * 64
        with self.assertRaisesRegex(SA.AuthError, 'init cgroup'):
            SA.prepare_v2(self.args(out=self.t / 'c.json'), self.p)
        self.setUp(); self.p.procs[777] = P(5, 5, 1, 1, [777], '0::/x', 'x'); self.p.apps.append({'pid': 777, 'used_mib': 1})
        with self.assertRaisesRegex(SA.AuthError, 'unapproved processes now'):
            SA.prepare_v2(self.args(out=self.t / 'u.json'), self.p)

    def test_load_v2_requires_explicit_bounds(self):
        a = json.loads(self.path.read_text())
        for mut, msg in ((lambda x: x['sessions']['fold27-trqam-v2'].pop('per_serve_mib'), 'per_serve_mib'),
                         (lambda x: x['sessions']['fold27-v2-eval'].update(max_serves=7), 'max_serves'),
                         (lambda x: x['containers'][0].update(cgroup=f'/docker/{CID}'), 'cgroup'),
                         (lambda x: x['containers'][0].update(devices=[UUID, 'GPU-x']), 'exactly the authorized GPU'),
                         (lambda x: x['containers'][0].update(id=CID[:12]), '64-hex'),
                         (lambda x: x['containers'][0]['init'].pop('start_ticks'), 'init needs')):
            b = copy.deepcopy(a); mut(b)
            q = self.t / f'bad{len(msg)}.json'; q.write_text(json.dumps(b))
            with self.assertRaisesRegex(SA.AuthError, msg):
                SA.load(q)

    def test_harness_hook_uses_v2(self):
        req = {'machine': 'srv50', 'gpus': [3], 'serves': 1, 'session': 'fold27-trqam-v2', 'kind': 'eval'}
        with patch.dict(os.environ, {'HARNESS_SHARING_AUTH': str(self.path)}), patch.object(SA, 'Probe', return_value=self.p), \
                patch.object(h, 'check_gpu_processes') as default_probe:
            h.probe_gpus(req); default_probe.assert_not_called()
            self.p.procs[3677646]['cgroup'] = '0::/user.slice'
            with self.assertRaisesRegex(ValueError, 'sharing authorization refused'):
                h.probe_gpus(req)


if __name__ == '__main__':
    unittest.main()
