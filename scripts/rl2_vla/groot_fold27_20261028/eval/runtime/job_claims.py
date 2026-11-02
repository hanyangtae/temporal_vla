"""Shared inter-process job claims + GPU fair-share rule for fold27 eval/train on one host (new; no live file changed).

Claims  <dir>/<key>.claim (O_EXCL) = {consumer pid/starttime, child pid/starttime, boot_id, gpu, port, receipt, session}.
        Every mutation runs under fcntl.flock(<dir>/claims.lock). A claim is LIVE while the boot is the same and the consumer
        OR its child (run_eval_arm) process still exists with the recorded start time (pid reuse safe). Stale claims are
        renamed to stale/ (never unlinked); finished claims are renamed to done/ (history kept). A partial arm left by a dead
        claimer is simply re-run by run_eval_arm, which resumes completed rows by hash (no row counted twice).
Share   eval_share(demand, n_gpus, serves, train_waiting): GPUs eval may hold. With a waiting trainer and eval demand,
        eval keeps ceil(demand/serves) GPUs but at most n-1 and at least 1; the trainer gets the rest (>=1). No waiting
        trainer -> eval may use all. Neither side can be starved: eval always keeps >=1 GPU while it has demand, and a
        consumer over its share stops taking jobs at its next completed ARM (never mid-episode).
Train request  <qdir>/TRAIN_REQUEST.json written only by the train coordinator (single writer); consumers only read it.
"""
import contextlib, fcntl, json, math, os, time
from pathlib import Path


def boot_id():
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return 'unknown'


def proc_start(pid):
    """Kernel start time (clock ticks since boot) of pid, or None if absent."""
    try:
        s = Path(f'/proc/{int(pid)}/stat').read_text()
        return int(s.rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def ident(pid):
    return {'pid': int(pid), 'start': proc_start(pid)}


def alive(idn, start_fn=proc_start):
    return bool(idn) and idn.get('start') is not None and start_fn(idn['pid']) == idn['start']


def key_name(key):
    return '.'.join(str(k) for k in key).replace('/', '_')


class ClaimStore:
    def __init__(self, d, start_fn=proc_start, boot_fn=boot_id, now=time.time):
        self.d = Path(d); self.d.mkdir(parents=True, exist_ok=True)
        self.start_fn, self.boot_fn, self.now = start_fn, boot_fn, now

    @contextlib.contextmanager
    def locked(self):
        with open(self.d / 'claims.lock', 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def path(self, key):
        return self.d / f'{key_name(key)}.claim'

    def read(self, key):
        p = self.path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def live(self, c):
        return c is not None and c.get('boot_id') == self.boot_fn() and (
            alive(c.get('consumer'), self.start_fn) or alive(c.get('child'), self.start_fn))

    def _aside(self, key, sub, extra=None):
        p = self.path(key)
        dst = self.d / sub / f'{key_name(key)}.{time.time_ns()}.json'
        dst.parent.mkdir(exist_ok=True)
        body = json.loads(p.read_text()); body.update(extra or {})
        dst.write_text(json.dumps(body, indent=1)); p.unlink()     # content preserved under sub/ before the claim file goes
        return dst

    # -- all public mutations hold the lock
    def try_claim(self, key, **info):
        """True if this caller now owns key. A stale claim is moved to stale/ first; a live one blocks."""
        with self.locked():
            return self._try_claim(key, **info)

    def _try_claim(self, key, **info):
        c = self.read(key)
        if c is not None:
            if self.live(c):
                return False
            self._aside(key, 'stale', {'reconciled_at': self.now()})
        body = {'key': list(key), 'boot_id': self.boot_fn(), 'consumer': ident(info.pop('pid', os.getpid())),
                'child': None, 'time': self.now(), **info}
        fd = os.open(self.path(key), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'w') as f:
            f.write(json.dumps(body, indent=1))
        return True

    def set_child(self, key, pid):
        with self.locked():
            c = self.read(key)
            if c is None:
                raise RuntimeError(f'claim {key} vanished while its job was starting')
            c['child'] = ident(pid)
            tmp = self.path(key).with_suffix('.tmp'); tmp.write_text(json.dumps(c, indent=1)); os.replace(tmp, self.path(key))

    def release(self, key, outcome):
        with self.locked():
            if self.path(key).exists():
                return self._aside(key, 'done', {'outcome': outcome, 'released_at': self.now()})

    def live_keys(self):
        with self.locked():
            out = set()
            for p in self.d.glob('*.claim'):
                c = json.loads(p.read_text())
                if self.live(c):
                    out.add(tuple(c['key']))
            return out


def eval_share(demand, n_gpus, serves, train_waiting):
    if demand <= 0:
        return 0
    if not train_waiting:
        return n_gpus
    return max(1, min(n_gpus - 1, math.ceil(demand / serves)))


def train_share(demand, n_gpus, serves):
    return n_gpus - eval_share(demand, n_gpus, serves, True)
