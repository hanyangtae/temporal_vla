"""eval_gpu_queue with the owner-aware legacy guard (receipt_owner). eval_gpu_queue.py is unchanged: only its module-level
legacy_eval_receipts (looked up at call time by may_start() and main()) is replaced, so a live fold27 TRAINING receipt
(kind=eval, session fold27-trqam-v2, train process evidence) no longer blocks, while the grouped eval queue, unknown owners
and foreign multi-GPU eval receipts still do (fail closed). Same CLI as eval_gpu_queue.py."""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gpu_queue as EGQ   # noqa: E402
import receipt_owner as RO     # noqa: E402


def legacy_eval_receipts(state, host, pool, cmd=None):
    eval_session = os.environ.get('FOLD27_EVAL_SESSION', RO.EVAL_SESSION)
    train = tuple(x for x in os.environ.get('FOLD27_TRAIN_SESSIONS', ','.join(RO.TRAIN_SESSIONS)).split(',') if x)
    return [rid for rid, _, _ in RO.blocking(state, host, pool, eval_session, train, cmd or RO.cmdline)]


EGQ.legacy_eval_receipts = legacy_eval_receipts

if __name__ == '__main__':
    sys.exit(EGQ.main())
