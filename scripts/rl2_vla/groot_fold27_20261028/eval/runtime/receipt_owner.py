"""Classify harness receipts on the shared pool by owner (fix for eval_gpu_queue.legacy_eval_receipts, which treated EVERY
multi-GPU kind=eval receipt as the grouped eval queue — but fold27 training also reserves with kind=eval, e.g. the j3
receipt 1f82eca3… session fold27-trqam-v2 GPUs 5,6,7, 2026-11-02, which wedged handover_v5 at its drain step).

  grouped_eval   session == the eval session and > 1 GPU  -> BLOCKS per-GPU consumers / handover drain (genuine old queue)
  per_gpu_eval   session == the eval session and 1 GPU    -> not blocking (new consumers share via claims)
  train          session in the train sessions AND live process evidence on this host: launcher or consumer cmdline
                 names the fold27 trainer (train_queue_v2.sh / train_queue_v2.py / finetune_fold_v2.py / train_arm_v5.sh)
                 and NOT eval_pipeline.py                     -> not blocking (GPUs are simply not free; harness arbitrates)
  unknown        train session without that evidence (dead / reused pid, other launcher host, eval cmdline) -> BLOCKS
  foreign        any other session: multi-GPU kind=eval on the pool BLOCKS (fail closed); single-GPU does not
Receipts on other machines or entirely off the pool are ignored. Read-only (harness ledger + /proc cmdline).
"""
import socket
from pathlib import Path

EVAL_SESSION = 'fold27-v2-eval'
TRAIN_SESSIONS = ('fold27-trqam-v2',)
TRAIN_MARKERS = ('train_queue_v2.sh', 'train_queue_v2.py', 'finetune_fold_v2.py', 'train_arm_v5.sh')
EVAL_MARKERS = ('eval_pipeline.py', 'eval_queue.sh', 'eval_gpu_queue')


def cmdline(pid):
    try:
        return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, TypeError, ValueError):
        return None


def classify(it, eval_session=EVAL_SESSION, train_sessions=TRAIN_SESSIONS, cmd=cmdline, host=None):
    gpus = it.get('gpus', [])
    s = it.get('session')
    if s == eval_session:
        return ('grouped_eval' if len(gpus) > 1 else 'per_gpu_eval'), None
    if s in train_sessions:
        here = host or socket.gethostname().split('.')[0]
        if it.get('launcher_host') not in (None, here):
            return 'unknown', f'launcher_host {it.get("launcher_host")} != {here}: no local process evidence'
        ev = {k: cmd(it.get(k)) for k in ('launcher_pid', 'consumer_pid') if it.get(k)}
        text = ' '.join(v for v in ev.values() if v)
        if text and any(m in text for m in TRAIN_MARKERS) and not any(m in text for m in EVAL_MARKERS):
            return 'train', ev
        return 'unknown', ev
    return 'foreign', None


def blocking(state, host, pool, eval_session=EVAL_SESSION, train_sessions=TRAIN_SESSIONS, cmd=cmdline, here=None):
    """[(receipt, class, evidence)] that must keep per-GPU eval consumers / the handover drain waiting."""
    out = []
    for rid, it in state.items():
        if it.get('machine') != host or not set(it.get('gpus', [])) & set(pool):
            continue
        c, ev = classify(it, eval_session, train_sessions, cmd, here)
        if c in ('grouped_eval', 'unknown') or (c == 'foreign' and it.get('kind') == 'eval' and len(it.get('gpus', [])) > 1):
            out.append((rid, c, ev))
    return out
