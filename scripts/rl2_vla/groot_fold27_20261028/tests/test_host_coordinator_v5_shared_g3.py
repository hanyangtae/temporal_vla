"""host_fold_coordinator_v5_shared_g3: every v5_shared test re-run with the v3 guardian (script + config kind), plus:
a v2 guardian is refused; a kind-V1 training contract (session == --session) trains all arms; a v1 contract for another
session is refused. Fake probes only (same fixtures as test_host_coordinator_v5_shared)."""
import hashlib, json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT), str(ROOT.parents[2] / 'scripts/utils')]
import test_host_coordinator_v5_shared as TS     # noqa: E402
import host_fold_coordinator_v5_shared as HS     # noqa: E402
import host_fold_coordinator_v5_shared_g3 as G3  # noqa: E402

HC = HS.HC
V3_SCRIPT = 'scripts/rl2_vla/groot_fold27_20261028/trqam/train_guardian_v3.py'


class G3Test(TS.SharedTest):
    def setUp(self):
        super().setUp()
        self.gconf['kind'] = G3.GUARD_KIND
        Path(self.gcfg).write_text(json.dumps(self.gconf))
        self.gp.argvv[1] = V3_SCRIPT

    def shared(self, demand=lambda: 0, harness=None, extra=()):
        self.clock = clock = [0.0]

        def sleep(s):
            clock[0] += s
            self.conc.append((clock[0], len([k for k, v in self.ledger.items() if v['session'] == 'fold27-trqam-v2'])))
        sp = self.spawn()

        def spawn(cmd, **kw):
            self.envs.append(kw['env']); return sp(cmd, **kw)
        argv = self.argv('--gpus', '5', *extra) + ['--eval-root', str(self.t / 'eval'), '--sharing-auth', str(self.ap),
                                                   '--train-guardian-pid', '4242', '--train-guardian-start', '77',
                                                   '--train-guardian-config', self.gcfg]
        co = G3.HostCoordinatorV5SharedG3(HS.parse(argv), popen=self.fake_popen(), run=None, spawn=spawn, demand=demand,
                                          harness=harness or (lambda: self.ledger), pending=lambda: (3, ''), sleep=sleep,
                                          now=lambda: clock[0], probe=self.sp, gprobe=self.gp)
        orig = co.run

        def go():
            try:
                return orig()
            finally:
                if co._lock:
                    co._lock.close(); co._lock = None
        co.run = go
        return co

    def test_v2_guardian_refused(self):
        self.gconf['kind'] = 'fold27_train_guardian_v2'; Path(self.gcfg).write_text(json.dumps(self.gconf))
        self.gp.argvv[1] = V3_SCRIPT.replace('_v3', '_v2')
        self.refused()

    def v1_contract(self, session='fold27-trqam-v2'):
        a = {k: self.auth[k] for k in ('machine', 'hostname', 'gpu', 'gpu_uuid', 'experiment', 'reserve_mib', 'expires_at',
                                       'boot_id', 'prepared_at', 'approval', 'foreign')}
        a.update(kind='groot_gpu_sharing_exception_v1', session=session, allowed_kind='eval', max_serves=1, per_serve_mib=7168,
                 per_serve_basis='measured_4584_x1.5_rounded_7168')
        self.ap.write_text(json.dumps(a))

    def test_v1_training_contract_trains_all_arms(self):
        self.v1_contract()
        res = self.shared().run()
        self.assertEqual(res['done_folds'][1:], self.todo())
        self.assertTrue(all(e['HARNESS_SHARING_AUTH'] == str(self.ap) for e in self.envs))

    def test_v1_contract_other_session_refused(self):
        self.v1_contract(session='fold27-v2-eval')                            # the eval contract is not a training scope
        with self.assertRaises(HC.CoordError):
            self.shared()

    def test_v1_contract_new_foreign_never_launches(self):
        self.v1_contract()
        self.sp.extra = [{'pid': 777, 'used_mib': 100}]                        # e.g. a leftover eval serve: never approved
        self.refused()


if __name__ == '__main__':
    unittest.main()
