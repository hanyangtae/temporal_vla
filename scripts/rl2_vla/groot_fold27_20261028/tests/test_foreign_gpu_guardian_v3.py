"""foreign_gpu_guardian_v3 with a v2 contract (pinned root + pinned foreign container instance). Fake OS / NVML only.
All v1 guardian tests are re-run on v3 with the approved container's changing children present on the GPU; plus:
container members tolerated and never signalled; another container / host same-uid process / restarted container /
over-limit container -> foreign -> only OUR jobs stopped; config session outside the contract refused; v1 contract on v3
behaves like v2."""
import hashlib, json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import test_foreign_gpu_guardian as T     # noqa: E402  (v1 fixture + tests)
import test_foreign_gpu_guardian_v2 as T2  # noqa: E402  (approved-root fixture values)
import foreign_gpu_guardian_v3 as G3      # noqa: E402

CID = '4ef7ae84e3da499cbbe725775fbf121c3737231bd4a5232264f423a89319327b'


class GProbe3(T2.GProbe):
    def status(self, pid):
        p = self.o.procs.get(pid)
        if p is None:
            return None
        gid, ns = getattr(p, 'gid', p.uid), getattr(p, 'nsl', [p.pid])
        return {'PPid': f'\t{p.ppid}', 'Uid': f'\t{p.uid}\t{p.uid}', 'Gid': f'\t{gid}\t{gid}', 'NSpid': '\t' + '\t'.join(map(str, ns))}

    def cgroup(self, pid):
        p = self.o.procs.get(pid)
        return None if p is None else (p.cgroup or '0::/') + '\n'


class ContainerSharingTest(T.GuardianTest):
    def setUp(self):
        super().setUp()
        o = self.o
        o.add(600, 60, 1, T2.APPROVED_CMD, uid=0, cgroup='0::/system.slice/isaac.scope'); o.gpu.append(600)
        dock = f'0::/docker/{CID}'
        init = o.add(700, 70, 1, 'bash', uid=1019, cgroup=dock); init.gid, init.nsl = 1018, [700, 1]
        kid = o.add(701, 71, 700, 'python -B -', uid=1019, cgroup=dock); kid.gid, kid.nsl = 1018, [701, 9]
        o.gpu.append(701)
        self.auth = {'kind': 'groot_gpu_sharing_exception_v2', 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 1, 'gpu_uuid': T2.UUID,
                     'experiment': 'fold27', 'sessions': {T.SESSION: {'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118},
                                                          'fold27-trqam-v2': {'allowed_kind': 'eval', 'max_serves': 1, 'per_serve_mib': 12000}},
                     'reserve_mib': 8192, 'expires_at': o.t + 3600, 'boot_id': o.bootv, 'prepared_at': o.t, 'approval': 'user one-time',
                     'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': T2.APPROVED_CMD, 'max_used_mib': 22000,
                                  'cmdline_sha256': hashlib.sha256(T2.APPROVED_CMD.encode()).hexdigest()}],
                     'containers': [{'id': CID, 'name': '/jolly_wright', 'image_id': 'sha256:img', 'started_at': 'x', 'cgroup': dock,
                                     'devices': [T2.UUID], 'member_uid': 1019, 'member_gid': 1018, 'max_used_mib': 8000,
                                     'init': {'pid': 700, 'start_ticks': 70, 'uid': 1019, 'gid': 1018}, 'launcher_evidence': ''}]}
        self.ap = self.t / 'state_auth_v2.json'; self.ap.write_text(json.dumps(self.auth)); self.files[self.ap] = self.ap.read_bytes()
        self.probe = GProbe3(o); self.probe.used[701] = 1500

    def g(self, **kw):
        return G3.GuardianV3({**self.cfg, 'sharing_authorization': str(self.ap), **kw}, self.o, self.t / 'state', probe=self.probe)

    def never(self, *pids):
        hit = {k[0] if not isinstance(k[0], tuple) else k[0][1] for k in self.o.kills}
        for p in pids:
            self.assertNotIn(p, hit); self.assertIn(p, self.o.procs)

    def act(self, g):
        g.poll(); conf = g.poll(); self.assertTrue(conf)
        return g.act(conf)

    def test_container_children_tolerated_and_change(self):
        g = self.g()
        self.assertIsNone(g.poll())
        kid = self.o.add(702, 72, 700, 'python -B -', uid=1019, cgroup=f'0::/docker/{CID}'); kid.gid, kid.nsl = 1018, [702, 10]
        self.o.gpu.append(702); self.o.remove(701)                         # children come and go inside the instance
        self.assertEqual(g.run(max_polls=4), 'no_action')
        self.assertEqual(self.o.kills, []); self.never(600, 700, 702)

    def test_other_container_is_foreign(self):
        other = '0::/docker/' + 'b' * 64
        x = self.o.add(800, 80, 1, 'python -B -', uid=1019, cgroup=other); x.gid, x.nsl = 1018, [800, 3]
        self.o.gpu.append(800)
        self.assertEqual(self.act(self.g()), 'released'); self.never(600, 700, 701, 800)

    def test_host_same_uid_same_command_is_foreign(self):
        x = self.o.add(810, 81, 1, 'python -B -', uid=1019, cgroup='0::/user.slice/user-1019.slice'); x.gid = 1018
        self.o.gpu.append(810)
        self.act(self.g()); self.never(600, 701, 810)

    def test_restarted_container_is_foreign(self):
        self.o.procs[700].start = 99
        self.act(self.g()); self.never(600, 700, 701)

    def test_over_limit_container_is_foreign(self):
        self.probe.used[701] = 8001
        self.act(self.g()); self.never(600, 701)

    def test_session_outside_contract_refused(self):
        with self.assertRaises(SystemExit):
            G3.GuardianV3({**self.cfg, 'session': 'fold27-other', 'sharing_authorization': str(self.ap)}, self.o, self.t / 'state', probe=self.probe)


class V1ContractOnV3(T2.SharingTest):
    """Every v2 sharing test (v1 contract, pinned root only) passes on v3 unchanged."""

    def g(self, **kw):
        return G3.GuardianV3({**self.cfg, 'sharing_authorization': str(self.ap), **kw}, self.o, self.t / 'state', probe=self.probe)


if __name__ == '__main__':
    unittest.main()
