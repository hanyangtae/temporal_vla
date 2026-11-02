"""Queue a harness-guarded launch until resources free up (no forced claims, no reservation changes).

    python wait_launch.py --max-wait-h 24 --poll-s 300 -- env GPUS=5 HARNESS_MACHINE=kanu HARNESS_SESSION=s \
        bash scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue.sh --sources S --folds F

Retries ONLY when the harness refused the reservation ("[harness] BLOCKED" on stderr, rc 2 — groot_harness.py main;
precedent: common_qam/pipeline.py treats it as wait). Any other non-zero exit is a real failure and is returned at once.
State: <state>/WAIT_STATE.json (attempts, last block reason). Bounded by --max-wait-h.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

BLOCK_MARK = '[harness] BLOCKED'


def wait_launch(cmd, state_dir, max_wait_h, poll_s, run=subprocess.run, sleep=time.sleep, now=time.time):
    state_dir.mkdir(parents=True, exist_ok=True)
    deadline, attempt = now() + max_wait_h * 3600, 0
    while True:
        attempt += 1
        p = run(cmd, stdout=None, stderr=subprocess.PIPE)
        err = (p.stderr or b'').decode(errors='replace')
        sys.stderr.write(err)
        blocked = p.returncode != 0 and BLOCK_MARK in err
        st = {'attempt': attempt, 'rc': p.returncode, 'blocked': blocked, 'time': now(),
              'reason': next((l for l in err.splitlines() if BLOCK_MARK in l), None), 'cmd': cmd}
        tmp = state_dir / 'WAIT_STATE.tmp'; tmp.write_text(json.dumps(st, indent=1)); os.replace(tmp, state_dir / 'WAIT_STATE.json')
        if not blocked:
            return p.returncode
        if now() + poll_s > deadline:
            return 75                                   # EX_TEMPFAIL: still blocked at the deadline
        sleep(poll_s)


def main():
    if '--' not in sys.argv:
        sys.exit('usage: wait_launch.py [--max-wait-h H] [--poll-s S] [--state DIR] -- <command...>')
    i = sys.argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-wait-h', type=float, default=24); ap.add_argument('--poll-s', type=float, default=300)
    ap.add_argument('--state', type=Path, default=Path('outputs/rl2_vla/fold27_trqam_20261102/queue/wait'))
    a = ap.parse_args(sys.argv[1:i])
    sys.exit(wait_launch(sys.argv[i + 1:], a.state, a.max_wait_h, a.poll_s))


if __name__ == '__main__':
    main()
