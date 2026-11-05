"""Exclusive-GPU TRAINING guardian (new file; train_guardian v1/v2/v3, harness, train_queue, finetune all unchanged).

Scope: fold27 TRAINING only (train_queue_v2.sh / smoke_v2b.sh under the official groot harness) on ONE GPU that is
empty at admission (no sharing contract). Eval consumers keep their own guardians (guardian_v4 / persist_guardian_v4);
the 2026-11-04 eval 'persist' approval is NOT inherited: persist for training needs its own config statement.

Three modes, one state dir (<sd>), one lock (<sd>/guardian.lock, flock, held by exactly one actor):

watch     (long-lived guardian, started once, detached)
  ownership  unchanged v1/v2 proof: exactly one harness receipt of our session on [gpu] (kind eval, launcher_host here,
             consumer_pid recorded), launcher = our `groot_harness.py run --session S --gpus G --kind eval` (uid, start
             no later than receipt.created + skew), consumer = our training entry (train_queue_v2 / smoke_v2b / finetune)
             below the launcher, owned = launcher descendants by (pid, start ticks), sticky. Unproven -> ALERT +
             ADMISSION_UNSAFE, no signal (v2).
             Never owned, never signalled (with their subtrees): this actor itself, and THIS launch's watchdog, which
             launch spawns before exec and is therefore a child of the launcher pid -- proven by uid + argv (watchdog
             mode, this --config/--state-dir, --launch-pid/--launch-start = owner launcher) + its own armed file naming
             that exact (pid, start) on this boot (own_watchdog). A look-alike without that proof stays owned.
  stop rules (the ONLY reasons our training is stopped; each must hold confirm_polls consecutive polls):
             own_mem   NVML used by our proven PIDs > min(receipt serves, max_jobs) x per_job_mib
             headroom  GPU free < reserve_mib (with or without a visible foreign process: either way the GPU is short)
             foreign   ONLY with foreign_policy 'stop': an unproven GPU process confirmed
             fail-closed (v2/v3, unchanged): GPU query failing / UUID mismatch / identity checks raising while our
             training runs; host or boot changed
  foreign_policy 'persist' (training): a foreign process arriving while our training runs is recorded every poll
             (pid/start/uid/cmd/MiB) and never by itself stops us; memory protection above stays. Never signalled
             (only proven own identities are ever signalled; the OS refuses other uids anyway).
  action     unchanged v2 act(): freeze own launcher ancestors (wait_launch / coordinator) -> SIGSTOP queue -> SIGTERM
             trainers (finetune_fold_v2 atomic resume states) -> SIGKILL leftovers -> stop queue -> wait own absence ->
             official `groot_harness.py release` only if still ours, else receipt KEPT + ADMISSION_UNSAFE. Then HOLD.json
             (blocks every new admission until main inspects) and the guardian exits.
  normal end the harness launcher exits on its own (finished, failed or killed). The cached proof is kept while any
             owned identity lives (memory protection continues; the dead launcher pid is never walked again: PID reuse).
             When every owned identity is gone: GPU probed (no owned PID, no own-uid training process), then
               receipt gone                     -> released_by_harness (normal path)
               receipt still there, entry equal -> released_by_guardian via `groot_harness.py release` (the harness
                 keeps it when ANY compute process is on the GPU at exit, e.g. a foreign one that arrived meanwhile)
               anything unprovable              -> receipt kept, retried; after confirm_polls ADMISSION_UNSAFE
             -> COMPLETED.<receipt>.json, guardian idle again (next launch can be admitted).
  heartbeat  <sd>/heartbeat.json every poll: guardian identity, boot, time, phase (watch | acting | stopped), receipt,
             launcher, last_ok (last poll clean: verified, no foreign, no breach), completed receipts.

launch    (admission wrapper; the training command is exec'd by it, so its pid becomes the harness launcher)
  fail closed, all must hold or nothing is started:
    refuse (rc 3): config invalid; env GPUS/HARNESS_MACHINE/HARNESS_SESSION != config; HARNESS_SHARING_AUTH or
      HARNESS_RECEIPT set; SERVES_PER_GPU not in 1..max_jobs; command not `bash <train_queue_v2.sh|smoke_v2b.sh> ...`;
      train_queue_v2 without explicit --per-gpu k <= SERVES_PER_GPU (or with --gpus); pending ALERT / ADMISSION_UNSAFE /
      HOLD; no heartbeat, guardian identity dead/other boot, guardian argv not `watch` on this --config/--state-dir or
      observe-only; heartbeat older than hang_s; GPU UUID differs; watchdog not armed in time
    wait (rc 2 + '[harness] BLOCKED', so wait_launch.py retries): guardian busy with a receipt / last poll not clean;
      ANY harness receipt on this machine+GPU; ANY compute process on the GPU; free < serves x per_job_mib +
      reserve_mib; harness/NVML query failed
  then: ADMIT record, a detached watchdog (setsid, outside the training tree) for THIS launch identity, wait for its
  armed file, admission re-checked, exec. The harness itself re-checks the empty GPU at and after admission (unchanged).

watchdog  (one per launch; guardian loss)
  launch alive and guardian alive with a fresh heartbeat (hang_s; acting: act_deadline_s) -> nothing.
  guardian alive but heartbeat stale for confirm_polls+1 polls -> TERM then KILL that guardian (our uid, identity and
  argv re-checked before each signal). Guardian identity gone for confirm_polls+1 polls spanning confirm_s -> take the
  guardian lock (busy -> a live actor exists -> defer), build this guardian on the same state dir, prove ownership from
  the ledger exactly as the guardian does (launcher must be THIS launch) and run the same act() -> HOLD.
  Launch ended -> exit once the guardian heartbeat (newer than the end) no longer carries this launcher (completion
  recorded), or HOLD exists; guardian lost after the end -> exit with an event (receipt left for main).

    python3 train_guardian_exclusive.py watch    --config <cfg.json> --state-dir <dir> [--observe-only] [--once]
    python3 train_guardian_exclusive.py snapshot --config <cfg.json> --state-dir <dir>
    GPUS=G HARNESS_MACHINE=M HARNESS_SESSION=S SERVES_PER_GPU=k \
      python3 train_guardian_exclusive.py launch --config <cfg.json> --state-dir <dir> -- bash <train entry> ...
    (watchdog is started by launch)
Config kind 'fold27_train_guardian_exclusive_v1': host, hostname, gpu, gpu_uuid, session, per_job_mib, per_job_basis,
max_jobs (<= 6), reserve_mib, foreign_policy ('persist' needs persist_approval verbatim + persist_scope 'training'),
optional confirm_polls (>= 2), poll_s, kill_wait_s, release_wait_s, clock_skew_s, hang_s, act_deadline_s, confirm_s,
arm_wait_s, end_wait_s, harness. No sharing_authorization.
"""
import argparse, fcntl, json, os, signal, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_guardian_v3 as TG3     # noqa: E402  (unchanged; v2 / v1 below it unchanged)

TG2, TG1, SA = TG3.TG2, TG3.TG1, TG3.SA
KIND = 'fold27_train_guardian_exclusive_v1'
SELF = 'train_guardian_exclusive.py'
POLICIES = ('persist', 'stop')
ENTRIES = ('train_queue_v2.sh', 'smoke_v2b.sh')
BLOCK = '[harness] BLOCKED'
HB, HOLD = 'heartbeat.json', 'HOLD.json'
PENDING = ('ALERT.json', 'ADMISSION_UNSAFE.json', HOLD)
NVSMI_S, HARNESS_S = 60, 120          # subprocess timeouts: SA.Probe.gpu (x2 calls) / foreign_gpu_guardian.OS.harness
MAX_SLOTS = 6                         # srv harness slot limit per GPU (configs/harness/gpu_policy.json)


def write_json(p, body):
    p = Path(p)
    tmp = p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(body, indent=1, default=str)); os.replace(tmp, p)


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def real(p):
    return str(Path(p).resolve()) if p else None


def validate(cfg):
    def need(ok, why):
        if not ok:
            raise SystemExit(f'exclusive training guardian config: {why}')
    need(cfg.get('kind') == KIND, f'kind must be {KIND}')
    for k in ('host', 'hostname', 'session', 'gpu_uuid', 'per_job_basis'):
        need(isinstance(cfg.get(k), str) and cfg[k].strip(), f'{k} (non-empty string) required')
    for k in ('gpu', 'per_job_mib', 'max_jobs', 'reserve_mib'):
        need(type(cfg.get(k)) is int and cfg[k] >= (0 if k == 'gpu' else 1), f'{k} (int) required')
    need(cfg['max_jobs'] <= MAX_SLOTS, f'max_jobs <= {MAX_SLOTS}')
    need(not cfg.get('sharing_authorization'), 'no sharing contract here (shared GPUs use train_guardian_v3)')
    need(cfg.get('foreign_policy') in POLICIES, f'foreign_policy must be explicitly one of {POLICIES}')
    if cfg['foreign_policy'] == 'persist':
        need(str(cfg.get('persist_approval') or '').strip(), 'persist_approval (user statement verbatim) required')
        need(cfg.get('persist_scope') == 'training', "persist_scope must be 'training' (eval approvals are not inherited)")
    need(int(cfg.get('confirm_polls', 2)) >= 2, 'confirm_polls >= 2')
    return cfg


def deadlines(c):
    poll = float(c.get('poll_s', 10))
    hang = float(c.get('hang_s', 3 * poll + 2 * NVSMI_S + 2 * HARNESS_S + 60))
    act = float(c.get('act_deadline_s', 2 * float(c.get('kill_wait_s', 60)) + float(c.get('release_wait_s', 600))
                      + 3 * HARNESS_S + 2 * NVSMI_S + 60))
    return hang, act


def argv_of(osi, pid):
    return (osi.cmdline(pid) or '').split()


def flag(argv, f):
    return argv[argv.index(f) + 1] if f in argv and argv.index(f) + 1 < len(argv) else None


def guardian_argv_ok(osi, pid, cfg_path, sd):
    a = argv_of(osi, pid)
    return (any(x.endswith(SELF) for x in a) and 'watch' in a and '--observe-only' not in a
            and real(flag(a, '--config')) == real(cfg_path) and real(flag(a, '--state-dir')) == real(sd))


def take_lock(path):
    fh = open(path, 'a')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


# ====================================================================================================== guardian
class ExclusiveTrainGuardian(TG3.TrainGuardianV3):
    def __init__(self, cfg, osi, state_dir, probe=None, observe_only=False, me=None, lock=True, cfg_path=None):
        validate(cfg)
        self.cfg_path, self.excluded = cfg_path, set()   # cfg_path: needed to prove our own watchdog (own_watchdog)
        # same attributes as TrainGuardian/V2/V3.__init__ (their contract checks are the only part not applicable)
        self.c, self.o, self.sd = cfg, osi, Path(state_dir)
        self.sd.mkdir(parents=True, exist_ok=True)
        self.gpu, self.session, self.boot0 = int(cfg['gpu']), cfg['session'], osi.boot()
        self.auth = {'gpu_uuid': cfg['gpu_uuid'], 'reserve_mib': int(cfg['reserve_mib'])}   # the fields TG3.poll reads
        self.probe = probe or SA.Probe()
        self.observe_only = observe_only or bool(cfg.get('observe_only'))
        self.owner, self.owned, self.seen, self.breach = None, {}, {}, {'own_mem': 0, 'headroom': 0}
        self.fails = {'gpu': 0, 'ledger': 0, 'identity': 0, 'complete': 0}
        self.meas = {'n_samples': 0, 'own_used_peak_mib': None, 'own_used_last_mib': None, 'first_t': None, 'last_t': None, 'samples': []}
        pid = os.getpid() if me is None else int(me[0])
        self.me = {'pid': pid, 'start': osi.start(pid) if me is None else me[1]}
        self.completed, self.last_ok, self._serves_noted = [], False, None
        self._lock = None
        if lock:
            self._lock = take_lock(self.sd / 'guardian.lock')
            if self._lock is None:
                raise SystemExit('guardian lock held by a live guardian/watchdog (duplicate actor refused)')

    def close(self):
        if self._lock is not None and not self._lock.closed:
            self._lock.close()

    # ---------------- records ----------------
    def beat(self, phase, **kw):
        o = self.owner or {}
        write_json(self.sd / HB, {'guardian': self.me, 'boot_id': self.boot0, 'time': self.o.now(), 'phase': phase,
                                  'policy': self.c['foreign_policy'], 'observe_only': self.observe_only,
                                  'receipt': o.get('receipt'), 'launcher': list(o['launcher']) if o else None,
                                  'last_ok': self.last_ok, 'completed': self.completed[-20:], **kw})

    def entry_matches(self, it, o):
        return (it.get('session') == self.session and it.get('machine') == self.c['host'] and it.get('gpus') == [self.gpu]
                and it.get('launcher_pid') == o['launcher'][0] and it.get('consumer_pid') == o['consumer'][0])

    # ---------------- ownership ----------------
    def owner_now(self):
        """v2 proof; if it fails only because our launcher/consumer has exited (normal end in progress), the cached proof
        is kept while the SAME ledger entry (receipt, session, gpu, launcher_pid, consumer_pid) still exists."""
        try:
            return super().owner_now()
        except TG1.Unproven:
            o = self.owner
            if o:
                try:
                    it = self.o.harness('status').get(o['receipt'])
                except Exception:   # noqa: BLE001 - no ledger: no cached exception either
                    it = None
                if it is not None and self.entry_matches(it, o):
                    self.event('cached_owner_after_exit', receipt=o['receipt'],
                               launcher_alive=self.same(*o['launcher']), consumer_alive=self.same(*o['consumer']))
                    return o, 'cached_after_exit'
            raise

    def refresh(self):
        prev = self.owner
        super().refresh()
        if prev and self.owner is None and any(self.same(*k) for k in self.owned):
            self.unsafe('receipt released while owned processes still live (they were off the GPU at release)',
                        receipt=prev['receipt'], owned=[[p, s, r] for (p, s), r in self.owned.items() if self.same(p, s)])

    def own_watchdog(self, pid, start):
        """True only for THIS launch's watchdog, proven by everything at once (never by basename alone):
        our uid; argv = <SELF> watchdog --config <this guardian's config> --state-dir <this state dir> --launch-pid/
        --launch-start = the owner's launcher identity; and the armed file armed.<launcher pid>.<start>.json, written
        by that watchdog itself before the launch exec'd, names exactly this (pid, start) for this launcher on this boot.
        launch spawns the watchdog BEFORE exec, so it is a child of the pid that becomes the harness launcher (setsid
        does not change PPid); without this proof it would be adopted as an owned 'desc' and stopped / waited for."""
        o = self.owner
        if not o or self.cfg_path is None or start is None:
            return False
        lp, ls = o['launcher']
        a = argv_of(self.o, pid)
        if not (any(x.endswith(SELF) for x in a) and 'watchdog' in a) or self.o.uid(pid) != self.o.own_uid():
            return False
        if real(flag(a, '--config')) != real(self.cfg_path) or real(flag(a, '--state-dir')) != real(self.sd) \
                or flag(a, '--launch-pid') != str(lp) or flag(a, '--launch-start') != str(ls):
            return False
        arm = read_json(self.sd / 'watchdogs' / f'armed.{lp}.{ls}.json') or {}
        return (arm.get('watchdog') == {'pid': pid, 'start': start} and arm.get('launch') == {'pid': lp, 'start': ls}
                and arm.get('boot_id') == self.boot0)

    def guard_process(self, pid, start):
        """This actor itself, or this launch's proven watchdog: never owned, never signalled."""
        if (pid, start) == (self.me['pid'], self.me['start']) or self.own_watchdog(pid, start):
            if (pid, start) not in self.excluded:
                self.excluded.add((pid, start))
                self.event('own_guard_process_excluded', pid=pid, start=start, cmd=(self.o.cmdline(pid) or '')[:200])
            return True
        return False

    def _tree(self):
        """Owned = descendants of the live launcher (v2 rule) or, once the launcher is gone, of still-live proven
        identities (never walk a dead launcher pid: PID reuse). Our own guard processes and their subtrees are skipped."""
        o = self.owner
        if o and self.same(*o['launcher']):
            roots = [o['launcher'][0]]
        else:
            roots = [p for (p, s) in self.owned if self.same(p, s)]
        if roots:
            kids = {}
            for p in self.o.pids():
                pp = self.o.ppid(p)
                if pp:
                    kids.setdefault(pp, []).append(p)
            stack, done = list(roots), set()
            while stack:
                p = stack.pop()
                if p in done:
                    continue
                done.add(p)
                for k in kids.get(p, []):
                    st = self.o.start(k)
                    if st is None or self.guard_process(k, st):
                        continue                         # guard subtree (e.g. its nvidia-smi / harness calls) skipped
                    self.owned.setdefault((k, st), self.role(k))
                    stack.append(k)
        self.owned = {k: v for k, v in self.owned.items() if self.same(*k) and not self.guard_process(*k)}
        for k, r in list(self.owned.items()):            # a consumer that exec'd into the trainer keeps its identity
            if r == 'desc' and self.role(k[0]) != 'desc':
                self.owned[k] = self.role(k[0])

    def sig(self, pid, start, s, why):
        """v1 sig() (identity re-checked right before os.kill) with one extra refusal: our own guard processes."""
        if self.guard_process(pid, start):
            self.event('signal_refused_own_guard_process', pid=pid, start=start, sig=int(s), why=why); return False
        return super().sig(pid, start, s, why)

    def role(self, pid):
        return 'trainer' if self.has(pid, TG1.TRAINER_MARKS) else 'queue' if self.has(pid, TG1.QUEUE_MARKS) else 'desc'

    # ---------------- poll ----------------
    def poll(self):
        self.last_ok = False
        o = self.owner
        if o and not self.same(*o['launcher']):
            self._tree()
            if not any(self.same(*k) for k in self.owned):
                self.complete(o)                         # last_ok stays False: the next full poll re-verifies the GPU
                self.beat('watch'); return None
        r = super().poll()                               # TG3.poll: fail-closed refresh / NVML / UUID / identity paths
        self.beat('watch')
        return r

    def _evaluate(self, g):
        ours = [a for a in g['apps'] if self.owned_pid(a['pid'])]
        others = [a for a in g['apps'] if not self.owned_pid(a['pid'])]
        unknown, foreign = {}, []
        for a in others:
            st = self.o.start(a['pid'])                  # None = /proc unreadable = unverifiable = foreign (v3 rule)
            key = (a['pid'], st)
            unknown[key] = self.seen.get(key, 0) + 1
            foreign.append({'pid': a['pid'], 'start': st, 'uid': self.o.uid(a['pid']),
                            'cmd': (self.o.cmdline(a['pid']) or '')[:160], 'used_mib': a['used_mib']})
        self.seen = unknown
        need = int(self.c.get('confirm_polls', 2))
        if foreign:
            self.event('foreign_present', policy=self.c['foreign_policy'], pids=foreign, free_mib=g['free_mib'],
                       receipt=(self.owner or {}).get('receipt'))
        valid = self.probe.hostname() == self.c['hostname'] and self.o.boot() == self.boot0
        self.breach['host'] = 0 if valid else self.breach.get('host', 0) + 1
        if not valid:
            self.unsafe('host or boot differs from the guardian start (state unverifiable)')
        if not self.owner:
            if foreign:
                self.event('foreign_while_idle', pids=foreign)
            self.last_ok = valid and not foreign
            return None
        serves = int(self.owner['serves'])
        slots = min(serves, int(self.c['max_jobs']))
        if serves > int(self.c['max_jobs']) and self._serves_noted != self.owner['receipt']:
            self._serves_noted = self.owner['receipt']
            self.unsafe('receipt serves exceed max_jobs: own cap uses max_jobs', serves=serves, max_jobs=self.c['max_jobs'])
        cap = slots * int(self.c['per_job_mib'])
        own_used = sum(a['used_mib'] for a in ours)
        self.record(own_used, g, cap, self.c['per_job_basis'])
        trainers = len(self.alive('trainer'))
        if trainers > slots:
            self.event('trainers_over_slots', trainers=trainers, slots=slots)   # informational; memory is the stop rule
        self.breach['own_mem'] = self.breach['own_mem'] + 1 if own_used > cap else 0
        self.breach['headroom'] = self.breach['headroom'] + 1 if g['free_mib'] < int(self.c['reserve_mib']) else 0
        if self.breach['own_mem']:
            self.event('own_memory_over_bound', used_mib=own_used, cap_mib=cap)
        if self.breach['headroom']:
            self.event('headroom_low', free_mib=g['free_mib'], reserve_mib=self.c['reserve_mib'], foreign=bool(foreign))
        reasons = []
        if self.c['foreign_policy'] == 'stop':
            reasons += [f'foreign {k}' for k, n in unknown.items() if n >= need]
        reasons += [f'{k} breach' for k, n in self.breach.items() if n >= need]
        self.last_ok = valid and not foreign and not any(self.breach.values())
        return reasons or None

    # ---------------- normal end ----------------
    def complete(self, o):
        """Every owned identity is gone and the harness launcher has exited: return the receipt officially."""
        def defer(why, **kw):
            self.fails['complete'] += 1
            self.event('completion_deferred', why=why, attempts=self.fails['complete'], receipt=o['receipt'], **kw)
            if self.fails['complete'] >= int(self.c.get('confirm_polls', 2)):
                self.unsafe(f'launch ended but receipt return not provable: {why}', receipt=o['receipt'], **kw)
        try:
            g = self.probe.gpu(self.gpu)
        except Exception as e:   # noqa: BLE001 - NOT a pass
            return defer('GPU query failed', error=str(e)[:300])
        if g['uuid'] != self.c['gpu_uuid']:
            return defer('GPU UUID differs', uuid=g['uuid'])
        left = [a['pid'] for a in g['apps'] if self.owned_pid(a['pid'])
                or (self.o.uid(a['pid']) == self.o.own_uid() and self.has(a['pid'], TG1.TRAIN_MARKS))]
        if left:
            return defer('own-uid training process still on the GPU', pids=left)
        try:
            it = self.o.harness('status').get(o['receipt'])
        except Exception as e:   # noqa: BLE001
            return defer('harness ledger unreachable', error=str(e)[:300])
        if it is None:
            result = 'released_by_harness'
        elif not self.entry_matches(it, o):
            return defer('ledger entry no longer matches the proven owner', entry=it)
        elif self.observe_only:
            result = 'observe_only_receipt_left'
        else:
            try:
                self.o.harness('release', '--receipt', o['receipt'], '--session', self.session)
            except Exception as e:   # noqa: BLE001
                return defer('harness release failed', error=str(e)[:300])
            result = 'released_by_guardian'
        body = {'receipt': o['receipt'], 'result': result, 'time': self.o.now(), 'owner': o,
                'gpu_apps_at_return': g['apps'], 'free_mib': g['free_mib'], 'measure': {k: v for k, v in self.meas.items() if k != 'samples'}}
        write_json(self.sd / f'COMPLETED.{o["receipt"]}.json', body)
        self.event('launch_completed', receipt=o['receipt'], result=result, foreign_on_gpu=g['apps'])
        self.completed.append(o['receipt'])
        self.owner, self.owned, self.seen = None, {}, {}
        self.breach, self.fails['complete'] = {'own_mem': 0, 'headroom': 0}, 0

    # ---------------- action ----------------
    def act(self, reasons):
        self.beat('acting', reasons=reasons)
        r = super().act(reasons)                         # TrainGuardianV2.act (unchanged procedure)
        if not self.observe_only:
            write_json(self.sd / HOLD, {'time': self.o.now(), 'reasons': reasons, 'result': r, 'actor': self.me,
                                        'receipt': (self.owner or {}).get('receipt'),
                                        'note': 'no new admission until main inspects events/ACTION and removes HOLD.json'})
        self.beat('stopped', result=r)
        return r

    def run(self, once=False, max_polls=None):
        """TrainGuardian.run, except that a REFUSED action (no signal sent: ownership unproven at action time) does not
        end the guardian: our training may still run, so it keeps watching (HOLD already blocks new admissions)."""
        n, last = 0, 'no_action'
        while True:
            n += 1
            r = self.poll()
            if r:
                last = self.act(r)
                if last != 'refused':
                    return last
            if once or (max_polls and n >= max_polls):
                return last
            self.o.sleep(float(self.c.get('poll_s', 10)))


# ====================================================================================================== admission
def admit(cfg, cfg_path, sd, cmd, env, osi, probe):
    """-> (rc, why, evidence): 0 admitted, 2 wait (transient), 3 refused."""
    sd = Path(sd)
    try:
        validate(cfg)
    except SystemExit as e:
        return 3, str(e), {}
    gpu, mj = int(cfg['gpu']), int(cfg['max_jobs'])
    if env.get('GPUS') != str(gpu) or env.get('HARNESS_MACHINE') != cfg['host'] or env.get('HARNESS_SESSION') != cfg['session']:
        return 3, 'env GPUS / HARNESS_MACHINE / HARNESS_SESSION differ from the guardian config', {}
    if env.get('HARNESS_SHARING_AUTH') or env.get('HARNESS_RECEIPT'):
        return 3, 'HARNESS_SHARING_AUTH / HARNESS_RECEIPT must be unset (exclusive admission through a new receipt)', {}
    s = env.get('SERVES_PER_GPU', '')
    if not s.isdigit() or not 1 <= int(s) <= mj:
        return 3, f'SERVES_PER_GPU must be explicit in 1..{mj}', {}
    serves = int(s)
    if len(cmd) < 2 or not cmd[0].endswith('bash') or not any(cmd[1].endswith(e) for e in ENTRIES):
        return 3, f'command must be: bash <{"|".join(ENTRIES)}> ...', {}
    if cmd[1].endswith('train_queue_v2.sh'):
        k = flag(cmd, '--per-gpu')
        if '--gpus' in cmd or not (k or '').isdigit() or not 1 <= int(k) <= serves:
            return 3, 'train_queue_v2 needs explicit --per-gpu k <= SERVES_PER_GPU and no --gpus (harness GPUS)', {}
    for f in PENDING:
        if (sd / f).exists():
            return 3, f'pending {f} in the guardian state dir (main inspects first)', {}
    hb = read_json(sd / HB)
    if not hb or hb.get('boot_id') != osi.boot():
        return 3, 'no guardian heartbeat on this boot', {}
    gid = hb.get('guardian') or {}
    if gid.get('start') is None or osi.start(gid.get('pid')) != gid['start']:
        return 3, 'guardian identity (pid/start) not alive', {}
    if not guardian_argv_ok(osi, gid['pid'], cfg_path, sd) or hb.get('observe_only') or hb.get('policy') != cfg['foreign_policy']:
        return 3, 'heartbeat process is not an enforcing watch guardian on this config/state dir/policy', {}
    hang, _ = deadlines(cfg)
    age = osi.now() - float(hb.get('time', 0))
    if age > hang:
        return 3, f'guardian heartbeat stale ({age:.0f}s > {hang:.0f}s)', {}
    if hb.get('phase') != 'watch' or hb.get('receipt') is not None or not hb.get('last_ok'):
        return 2, f'guardian not idle-clean (phase {hb.get("phase")}, receipt {hb.get("receipt")}, last_ok {hb.get("last_ok")})', {}
    try:
        st = osi.harness('status')
    except Exception as e:   # noqa: BLE001
        return 2, f'harness ledger unreachable: {str(e)[:200]}', {}
    held = {r[:8]: it.get('session') for r, it in st.items() if it.get('machine') == cfg['host'] and gpu in (it.get('gpus') or [])}
    if held:
        return 2, f'GPU{gpu} has harness receipts {held}', {}
    try:
        g = probe.gpu(gpu)
    except Exception as e:   # noqa: BLE001
        return 2, f'GPU query failed: {str(e)[:200]}', {}
    if g['uuid'] != cfg['gpu_uuid']:
        return 3, f'GPU UUID {g["uuid"]} differs from config', {}
    if g['apps']:
        return 2, f'GPU{gpu} not empty: {g["apps"][:4]}', {}
    need = serves * int(cfg['per_job_mib']) + int(cfg['reserve_mib'])
    if g['free_mib'] < need:
        return 2, f'GPU{gpu} free {g["free_mib"]} < {serves} x {cfg["per_job_mib"]} + reserve {cfg["reserve_mib"]}', {}
    return 0, 'admitted', {'guardian': gid, 'heartbeat_age_s': age, 'serves': serves, 'free_mib': g['free_mib'],
                           'need_mib': need, 'uuid': g['uuid'], 'cmd': cmd}


def launch(cfg_path, sd, cmd, osi, probe, env=None, spawn=subprocess.Popen, execvpe=os.execvpe, me=None, python=None):
    env = dict(os.environ if env is None else env)
    sd, cmd = Path(sd), list(cmd[1:] if cmd and cmd[0] == '--' else cmd)
    cfg = json.loads(Path(cfg_path).read_text())

    def refuse(rc, why):
        print(f'{BLOCK} exclusive training admission: {why}' if rc == 2 else f'[exclusive-guard] REFUSED: {why}',
              file=sys.stderr, flush=True)
        return rc
    rc, why, ev = admit(cfg, cfg_path, sd, cmd, env, osi, probe)
    if rc:
        return refuse(rc, why)
    pid = os.getpid() if me is None else int(me[0])
    me = {'pid': pid, 'start': osi.start(pid) if me is None else me[1]}
    key = f'{me["pid"]}.{me["start"]}'
    wd = sd / 'watchdogs'; wd.mkdir(parents=True, exist_ok=True)
    with open(wd / f'ADMIT.{key}.json', 'x') as f:
        f.write(json.dumps({'time': osi.now(), 'launch': me, 'evidence': ev, 'env': {k: env.get(k) for k in
                            ('GPUS', 'HARNESS_MACHINE', 'HARNESS_SESSION', 'SERVES_PER_GPU', 'FOLD27_OUT')}}, indent=1, default=str))
    g = ev['guardian']
    wcmd = [python or sys.executable, str(Path(__file__).resolve()), 'watchdog', '--config', real(cfg_path), '--state-dir', real(sd),
            '--launch-pid', str(me['pid']), '--launch-start', str(me['start']),
            '--guardian-pid', str(g['pid']), '--guardian-start', str(g['start'])]
    with open(wd / f'watchdog.{key}.log', 'a') as log:
        w = spawn(wcmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    t_end = osi.now() + float(cfg.get('arm_wait_s', 60))
    while True:
        a = read_json(wd / f'armed.{key}.json')
        if a and a.get('launch') == me and (a.get('watchdog') or {}).get('pid') == w.pid and a.get('guardian') == g:
            break
        if osi.now() >= t_end:
            return refuse(3, 'watchdog not armed in time (it exits by itself when this launch identity is gone)')
        osi.sleep(1)
    rc, why, ev2 = admit(cfg, cfg_path, sd, cmd, env, osi, probe)   # re-check right before exec
    if rc or ev2['guardian'] != g:
        return refuse(rc or 3, why if rc else 'guardian identity changed during admission')
    print(json.dumps({'admitted': key, 'watchdog': w.pid, 'serves': ev2['serves'], 'free_mib': ev2['free_mib']}),
          file=sys.stderr, flush=True)
    execvpe(cmd[0], cmd, env)                         # this pid becomes the harness launcher (groot_launch_guard exec)
    return 3                                           # only reached with a fake exec


# ====================================================================================================== watchdog
class Watchdog:
    def __init__(self, cfg_path, sd, osi, probe, launch_ident, guardian_ident, me=None):
        self.cfg_path, self.sd, self.o, self.probe = cfg_path, Path(sd), osi, probe
        self.c = validate(json.loads(Path(cfg_path).read_text()))
        self.boot0 = osi.boot()
        self.launch = {'pid': int(launch_ident[0]), 'start': launch_ident[1]}
        self.gid = {'pid': int(guardian_ident[0]), 'start': guardian_ident[1]}
        pid = os.getpid() if me is None else int(me[0])
        self.me = {'pid': pid, 'start': osi.start(pid) if me is None else me[1]}
        if not self.same(self.launch):
            raise SystemExit('launch identity not alive')
        self.wd = self.sd / 'watchdogs'; self.wd.mkdir(parents=True, exist_ok=True)
        self.key = f'{self.launch["pid"]}.{self.launch["start"]}'
        self._lock = take_lock(self.wd / f'watchdog.{self.key}.lock')
        if self._lock is None:
            raise SystemExit('another watchdog guards this launch')
        self.log = self.wd / f'watchdog.{self.key}.jsonl'
        self.hang_s, self.act_s = deadlines(self.c)
        self.lost_n, self.lost_t0, self.hang_n, self.end_t = 0, None, 0, None
        write_json(self.wd / f'armed.{self.key}.json', {'launch': self.launch, 'watchdog': self.me, 'guardian': self.gid,
                                                         'boot_id': self.boot0, 'time': osi.now()})
        self.event('armed')

    def close(self):
        if self._lock is not None and not self._lock.closed:
            self._lock.close()

    def event(self, kind, **kw):
        with open(self.log, 'a') as f:
            f.write(json.dumps({'kind': kind, 'time': self.o.now(), **kw}, default=str) + '\n')

    def same(self, ident):
        return bool(ident) and ident.get('start') is not None and self.o.boot() == self.boot0 \
            and self.o.start(ident.get('pid')) == ident['start']

    def guardian_alive(self):
        return self.same(self.gid) and guardian_argv_ok(self.o, self.gid['pid'], self.cfg_path, self.sd)

    def check(self):
        """One poll -> None (keep watching) or a final result."""
        now = self.o.now()
        hb = read_json(self.sd / HB)
        if hb and hb.get('boot_id') != self.boot0:
            hb = None
        own_hb = bool(hb) and hb.get('guardian') == self.gid
        if not self.same(self.launch):                                   # launch ended
            self.end_t = self.end_t or now
            if (self.sd / HOLD).exists():
                self.event('ended_after_hold'); return 'ended_after_hold'
            if own_hb and float(hb.get('time', 0)) > self.end_t and (hb.get('launcher') or [None])[0] != self.launch['pid']:
                self.event('launch_completed', completed=hb.get('completed')); return 'launch_completed'
            if now - self.end_t > float(self.c.get('end_wait_s', 1800)) or not self.guardian_alive() and now - self.end_t > self.hang_s:
                self.event('ended_unconfirmed', guardian_alive=self.guardian_alive()); return 'ended_unconfirmed'
            return None
        if self.guardian_alive():
            self.lost_n, self.lost_t0 = 0, None
            if own_hb:
                age, phase = now - float(hb.get('time', 0)), hb.get('phase')
                limit = self.act_s if phase == 'acting' else self.hang_s
                stale = age > limit
            else:
                age, phase, limit, stale = None, None, self.hang_s, True
            if not stale:
                self.hang_n = 0; return None
            self.hang_n += 1
            self.event('guardian_hung', phase=phase, age=age, limit=limit, polls=self.hang_n)
            if self.hang_n >= int(self.c.get('confirm_polls', 2)) + 1:
                self.stop_guardian(f'heartbeat age {age} > {limit:.0f}s ({phase})')
            return None
        self.hang_n = 0
        self.lost_n += 1; self.lost_t0 = self.lost_t0 or now
        self.event('guardian_identity_missing', polls=self.lost_n, since=self.lost_t0)
        if self.lost_n >= int(self.c.get('confirm_polls', 2)) + 1 and now - self.lost_t0 >= float(self.c.get('confirm_s', 10)):
            return self.stop('guardian lost (identity gone / stopped as hung)')
        return None

    def stop_guardian(self, why):
        """TERM, then KILL, OUR hung guardian (identity + argv re-checked right before each signal)."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not self.guardian_alive():
                break
            try:
                self.o.kill(self.gid['pid'], sig)
            except (ProcessLookupError, PermissionError) as e:
                self.event('guardian_signal_failed', sig=int(sig), error=type(e).__name__); return
            self.event('guardian_signalled', sig=int(sig), why=why)
            t_end = self.o.now() + float(self.c.get('kill_wait_s', 60))
            while self.same(self.gid) and self.o.now() < t_end:
                self.o.sleep(1)
        self.hang_n = 0

    def stop(self, why):
        try:
            g = ExclusiveTrainGuardian(self.c, self.o, self.sd, probe=self.probe, me=(self.me['pid'], self.me['start']),
                                       cfg_path=self.cfg_path)
        except SystemExit as e:                                          # lock busy: a live actor exists
            self.event('stop_deferred_lock_busy', why=why, detail=str(e)); self.lost_n, self.lost_t0 = 0, None; return None
        try:
            try:
                g.refresh()
            except TG1.Unproven as e:
                g.unsafe('watchdog: ownership unproven, no signal', detail=str(e)); self.event('stop_unproven', detail=str(e))
                return None
            except Exception as e:   # noqa: BLE001
                self.event('stop_unavailable', error=str(e)[:300]); return None
            if not g.owner:
                self.event('stop_nothing_to_stop_yet', why=why); return None      # no receipt yet: keep watching
            if g.owner['launcher'] != (self.launch['pid'], self.launch['start']):
                g.unsafe('watchdog: the receipt belongs to another launch, no signal', receipt=g.owner['receipt'])
                self.event('stop_other_launch', receipt=g.owner['receipt']); return None
            g.event('watchdog_stop', why=why, watchdog=self.me, launch=self.launch)
            r = g.act([f'watchdog: {why}'])
            self.event('stopped', why=why, result=r)
            return None if r == 'refused' else 'watchdog_' + r          # refused = nothing signalled: keep watching
        finally:
            g.close()

    def run(self, max_polls=None):
        n = 0
        while True:
            n += 1
            r = self.check()
            if r:
                return r
            if max_polls and n >= max_polls:
                return None
            self.o.sleep(float(self.c.get('watchdog_poll_s', self.c.get('poll_s', 10))))


# ====================================================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    ps = {m: sub.add_parser(m) for m in ('watch', 'snapshot', 'launch', 'watchdog')}
    for p in ps.values():
        p.add_argument('--config', type=Path, required=True); p.add_argument('--state-dir', type=Path, required=True)
    ps['watch'].add_argument('--observe-only', action='store_true'); ps['watch'].add_argument('--once', action='store_true')
    for k in ('--launch-pid', '--launch-start', '--guardian-pid', '--guardian-start'):
        ps['watchdog'].add_argument(k, type=int, required=True)
    ps['launch'].add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cfg = validate(json.loads(a.config.read_text()))
    osi = TG1.HostOS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py']))
    if a.mode == 'launch':
        return launch(a.config, a.state_dir, a.command, osi, SA.Probe())
    if a.mode == 'watchdog':
        w = Watchdog(a.config, a.state_dir, osi, SA.Probe(), (a.launch_pid, a.launch_start), (a.guardian_pid, a.guardian_start))
        try:
            print(json.dumps({'result': w.run()}), flush=True)
        finally:
            w.close()
        return 0
    g = ExclusiveTrainGuardian(cfg, osi, a.state_dir, observe_only=getattr(a, 'observe_only', False), lock=a.mode == 'watch',
                               cfg_path=a.config)
    if a.mode == 'snapshot':
        try:
            g.refresh()
        except TG1.Unproven as e:
            print(json.dumps({'unproven': str(e)})); return 6
        print(json.dumps({'owner': g.owner, 'owned': [[p, s, r] for (p, s), r in g.owned.items()]}, default=str))
        return 0
    print(json.dumps({'mode': 'watch', 'guardian': g.me, 'policy': cfg['foreign_policy'], 'gpu': g.gpu,
                      'per_job_mib': cfg['per_job_mib'], 'max_jobs': cfg['max_jobs'], 'reserve_mib': cfg['reserve_mib']}), flush=True)
    try:
        r = g.run(once=a.once)
    finally:
        g.close()
    print(json.dumps({'result': r}), flush=True)
    return {'no_action': 0, 'observed': 0, 'released': 0, 'released_elsewhere': 0, 'kept': 5, 'refused': 6}.get(r, 1)


if __name__ == '__main__':
    sys.exit(main())
