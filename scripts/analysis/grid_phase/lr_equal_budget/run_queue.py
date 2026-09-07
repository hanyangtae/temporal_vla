#!/usr/bin/env python3
"""Durable sequential experiment queue; stage commands own GPU lease handling.

Jobs specify id, command (argv), optional cwd/readiness checks, and outputs and/or
sentinel {path, text}. Sentinel text must match a complete line. Relative paths
are resolved against job cwd (default: configuration directory). A failed stage
is never retried automatically. Use a new reviewed state directory to retry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def argv_ok(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item) for item in value)


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate(config):
    if not isinstance(config, dict):
        raise ValueError('configuration must be an object')
    jobs = config.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('jobs must be a nonempty list')
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('each job must be an object')
        key = job.get('id', '')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', key) or key in seen:
            raise ValueError(f'invalid or duplicate job id: {key!r}')
        seen.add(key)
        if not argv_ok(job.get('command')):
            raise ValueError(f'{key}: command must be a nonempty argv array')
        if 'cwd' in job and (not isinstance(job['cwd'], str) or not job['cwd']):
            raise ValueError(f'{key}: invalid cwd')
        outputs = job.get('outputs', [])
        if not isinstance(outputs, list) or any(not isinstance(p, str) or not p for p in outputs):
            raise ValueError(f'{key}: outputs must be path strings')
        sentinel = job.get('sentinel')
        if sentinel is not None and (not isinstance(sentinel, dict) or
                not isinstance(sentinel.get('path'), str) or not sentinel['path'] or
                not isinstance(sentinel.get('text'), str) or not sentinel['text'] or
                '\n' in sentinel['text'] or '\r' in sentinel['text']):
            raise ValueError(f'{key}: sentinel requires path and single-line text')
        if not outputs and sentinel is None:
            raise ValueError(f'{key}: at least one output or sentinel is required')
        checks = job.get('readiness', [])
        if not isinstance(checks, list):
            raise ValueError(f'{key}: readiness must be a list')
        for check in checks:
            if not isinstance(check, dict) or not argv_ok(check.get('argv')) or not positive_number(check.get('timeout_s', 30)):
                raise ValueError(f'{key}: readiness requires argv and positive timeout_s')
        if job.get('timeout_s') is not None and not positive_number(job['timeout_s']):
            raise ValueError(f'{key}: timeout_s must be positive')
    return jobs


class Runner:
    def __init__(self, config_path, state_dir, poll_seconds=60, once=False):
        self.config_path = Path(config_path).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.poll_seconds = poll_seconds
        self.once = once
        self.child = None
        self.interrupted = None
        self.state = None

    def save(self):
        self.state['updated_at'] = now()
        atomic_json(self.state_dir / 'status.json', self.state)

    def update(self, key, status, reason, **details):
        self.state['jobs'][key].update(status=status, reason=reason, updated_at=now(), **details)
        self.save()
        print(f'{now()} {key} {status}: {reason}', flush=True)

    def on_signal(self, signum, _frame):
        self.interrupted = signum

    def stop_child(self):
        if self.child is None:
            return
        # start_new_session makes this PGID exclusively owned by this queue.
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Also stop descendants if the immediate child already exited.
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.child.wait()

    def process(self, argv, cwd, logfile, timeout_s):
        start = time.monotonic()
        with logfile.open('ab', buffering=0) as log:
            log.write((f'\n{now()} argv={json.dumps(argv)} cwd={cwd}\n').encode())
            try:
                self.child = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, start_new_session=True)
            except OSError as exc:
                return None, f'could not start: {exc}'
            try:
                while True:
                    if self.interrupted is not None:
                        self.stop_child()
                        return None, f'interrupted by signal {self.interrupted}'
                    if timeout_s is not None and time.monotonic() - start >= timeout_s:
                        self.stop_child()
                        return None, f'timed out after {timeout_s}s'
                    try:
                        code = self.child.wait(timeout=0.25)
                        return code, f'exit {code}'
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                self.child = None

    def cwd(self, job):
        value = Path(job.get('cwd', self.config_path.parent))
        if not value.is_absolute():
            value = self.config_path.parent / value
        return value.resolve()

    def verify_outputs(self, job):
        cwd = self.cwd(job)
        for output in job.get('outputs', []):
            if not (cwd / output).exists():
                return f'missing required output: {output}'
        sentinel = job.get('sentinel')
        if sentinel:
            path = cwd / sentinel['path']
            if not path.is_file():
                return f'missing sentinel file: {path}'
            try:
                with path.open(errors='replace') as source:
                    found = any(line.rstrip('\r\n') == sentinel['text'] for line in source)
            except OSError as exc:
                return f'cannot read sentinel: {exc}'
            if not found:
                return f'missing exact sentinel line: {sentinel["text"]}'
        return None

    def run_locked(self):
        raw = self.config_path.read_bytes()
        jobs = validate(json.loads(raw))
        digest = hashlib.sha256(raw).hexdigest()
        state_path = self.state_dir / 'status.json'
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            if self.state.get('config_sha256') != digest:
                raise ValueError('configuration SHA mismatch: refusing to resume this state directory')
            for job in jobs:
                key = job['id']
                if self.state['jobs'][key]['status'] == 'RUNNING':
                    self.update(key, 'FAILED', 'previous queue stopped while job was RUNNING; inspect before retry')
        else:
            self.state = {'schema_version': 1, 'config_sha256': digest,
                          'config_path': str(self.config_path), 'created_at': now(),
                          'jobs': {job['id']: {'status': 'PENDING', 'reason': 'not yet started'} for job in jobs}}
            self.save()
        self.state['queue_pid'] = os.getpid()
        self.save()
        for job in jobs:
            key = job['id']
            status = self.state['jobs'][key]['status']
            if status == 'FAILED':
                print(f'{key} is FAILED; automatic retry is disabled', file=sys.stderr)
                return 1
            if status == 'COMPLETED':
                error = self.verify_outputs(job)
                if error:
                    self.update(key, 'FAILED', f'completed artifact no longer valid: {error}')
                    return 1
                continue
            while True:
                if self.interrupted is not None:
                    self.update(key, 'PENDING', f'interrupted before start by signal {self.interrupted}')
                    return 128 + self.interrupted
                ready = True
                for index, check in enumerate(job.get('readiness', [])):
                    logfile = self.state_dir / 'logs' / f'{key}.readiness-{index}.log'
                    code, reason = self.process(check['argv'], self.cwd(job), logfile, check.get('timeout_s', 30))
                    if code != 0 or self.interrupted is not None:
                        self.update(key, 'PENDING', f'readiness {index}: {reason}', readiness_checked_at=now())
                        ready = False
                        break
                if ready:
                    break
                if self.interrupted is not None:
                    return 128 + self.interrupted
                if self.once:
                    return 75
                until = time.monotonic() + self.poll_seconds
                while self.interrupted is None and time.monotonic() < until:
                    time.sleep(min(0.25, max(0, until - time.monotonic())))
            if self.interrupted is not None:
                self.update(key, 'PENDING', 'interrupted before command start')
                return 128 + self.interrupted
            self.update(key, 'RUNNING', 'readiness passed; executing command', started_at=now())
            logfile = self.state_dir / 'logs' / f'{key}.command.log'
            code, reason = self.process(job['command'], self.cwd(job), logfile, job.get('timeout_s'))
            if code != 0 or self.interrupted is not None:
                self.update(key, 'FAILED', reason, exit_code=code, finished_at=now())
                return 128 + self.interrupted if self.interrupted else 1
            error = self.verify_outputs(job)
            if error:
                self.update(key, 'FAILED', f'command exited 0 but {error}', exit_code=0, finished_at=now())
                return 1
            self.update(key, 'COMPLETED', 'exit 0 and all required artifacts verified', exit_code=0, finished_at=now())
        return 0

    def run(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / 'queue.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('another queue owns this state directory')
            (self.state_dir / 'logs').mkdir(exist_ok=True)
            handlers = {sig: signal.signal(sig, self.on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                return self.run_locked()
            finally:
                self.stop_child()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--once', action='store_true', help='exit 75 if readiness is not met, instead of waiting')
    args = parser.parse_args()
    if not positive_number(args.poll_seconds):
        parser.error('--poll-seconds must be positive')
    try:
        return Runner(args.config, args.state_dir, args.poll_seconds, args.once).run()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'queue error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
