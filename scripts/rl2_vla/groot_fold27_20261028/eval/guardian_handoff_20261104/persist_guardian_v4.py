"""guardian_v4 in an explicit user 'persist' mode (new file; guardian_v4 / v1-v3 / watchdog / handoff unchanged).

User decision (srv50 GPU1, 2026-11-04, relayed by main): once our eval runs, a NEW foreign GPU process must NOT by itself
stop it. Memory protection stays. Others are never signalled. In this mode:
  * foreign identity is NOT a stop condition: every non-owned GPU process is tolerated, recorded each poll
    ('persist_foreign_present': pid/start/uid/cmd/used MiB), never owned -> never signalled (v1 signals owned only).
  * memory protection, separated from identity: while a foreign process is present AND our consumer runs, GPU free memory
    below cfg['reserve_mib'] for confirm_polls consecutive polls -> the unchanged v1 act() stops ONLY our verified jobs
    (prevents an OOM of the other task caused by us) and releases officially. Low free memory with no foreign process
    (our own usage) is not acted on. An NVML query failure means observe nothing, act on nothing (unchanged rule).
  * adoption (a NEW consumer): unchanged v4 handshake; the last poll must have succeeded with no low-memory streak, and
    by default NO foreign process may be present (cfg 'adopt_with_foreign': false) -- persist covers running work only,
    launch admission stays with the harness/wait_resources rules.
  * everything else = guardian_v4 (HOLD, watchdog, fence, heartbeat, follower marks, release). The file name contains
    'guardian_v4.py' so consumer_watchdog's identity check of a hung guardian still matches.
Required config keys (fail closed otherwise): 'foreign_policy': 'persist', 'persist_approval': <user statement verbatim>,
'reserve_mib': int > 0. 'sharing_authorization' must be absent (a contract has its own semantics; do not mix).

    python persist_guardian_v4.py --config <cfg.json> --state-dir <NEW dir> [--observe-only]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardian_v4 as G4   # noqa: E402  (unchanged)


class PersistGuardianV4(G4.GuardianV4):
    def __init__(self, cfg, osi, state_dir, observe_only=False, probe=None, take_lock=True):
        if cfg.get('foreign_policy') != 'persist':
            raise SystemExit("foreign_policy must be 'persist' (explicit user mode)")
        if not str(cfg.get('persist_approval') or '').strip():
            raise SystemExit('persist_approval (user statement) required')
        if not isinstance(cfg.get('reserve_mib'), int) or cfg['reserve_mib'] <= 0:
            raise SystemExit('reserve_mib (int > 0) required: memory protection is not optional')
        if cfg.get('sharing_authorization'):
            raise SystemExit('persist mode does not combine with a sharing contract')
        super().__init__(cfg, osi, state_dir, observe_only, probe, take_lock)
        self.low_mem, self.foreign_now = 0, []

    def poll(self):
        self.last_ok, self._qfail = False, False
        self.seen = {}
        try:
            g = self.probe.gpu(self.gpu)                       # pids + used + free in one NVML snapshot
        except Exception as e:   # noqa: BLE001 - uncertainty: observe nothing, act on nothing
            self.event('gpu_query_failed', error=str(e)[:300]); self.low_mem = 0; return None
        self.refresh()
        used = {a['pid']: a['used_mib'] for a in g['apps']}
        foreign = []
        for p in used:
            if self.owned_pid(p):
                continue
            st = self.o.start(p)
            if st is None:                                     # vanished between NVML and /proc
                continue
            foreign.append([p, st, self.o.uid(p), (self.o.cmdline(p) or '')[:160], used[p]])
        self.foreign_now = foreign
        if foreign:
            self.event('persist_foreign_present', gpu=self.gpu, pids=foreign, free_mib=g['free_mib'])
        need = int(self.c.get('confirm_polls', 2))
        confirmed = None
        if foreign and g['free_mib'] < int(self.c['reserve_mib']):
            self.low_mem += 1
            self.event('persist_headroom_low', free_mib=g['free_mib'], reserve_mib=self.c['reserve_mib'], polls=self.low_mem)
            if self.low_mem >= need:
                if self.read_consumer() is None:
                    self.event('headroom_low_while_idle', pids=foreign)
                else:
                    confirmed = [(p, s) for p, s, *_ in foreign]   # the reason; act() signals only OUR owned identities
        else:
            self.low_mem = 0
        self.last_ok = not self._qfail and not self.low_mem and confirmed is None and \
            (not foreign or bool(self.c.get('adopt_with_foreign', False)))
        return confirmed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true')
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = PersistGuardianV4(cfg, G4.OS4(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir,
                          a.observe_only)
    print(json.dumps({'mode': 'persist', 'reserve_mib': cfg['reserve_mib'], 'pid': g.me}), flush=True)
    g.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
