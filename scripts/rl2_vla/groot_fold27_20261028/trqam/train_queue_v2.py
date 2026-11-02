"""train_queue v2: the frozen v1 queue logic unchanged, pointed at finetune_fold_v2 (full-state resume) and a NEW out root.

Only differences from v1: job script = finetune_fold_v2.py (+ optional --reuse-ae-from <v1 root>), fingerprint pins include
finetune_fold_v2.py + resume_state.py (== finetune_fold_v2.fingerprint_v2 'code'), freeze also pins this file, default
--out = outputs/rl2_vla/fold27_trqam_v2_20261102. v1 root (failed smoke, FAILED.json, QUEUE_FREEZE) is never touched.

    GPUS=5 HARNESS_MACHINE=kanu HARNESS_SESSION=fold27-trqam-v2 bash train_queue_v2.sh --sources S --folds F \
        [--reuse-ae-from outputs/rl2_vla/fold27_trqam_20261102]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import train_queue as TQ   # noqa: E402  (frozen v1)

V2_OUT = TQ.REPO / 'outputs/rl2_vla/fold27_trqam_v2_20261102'
TQ.FINGERPRINT_CODE = {**TQ.FINGERPRINT_CODE, 'finetune_fold_v2.py': HERE / 'finetune_fold_v2.py',
                       'resume_state.py': HERE / 'resume_state.py'}
TQ.EXTRA_CODE = {**TQ.EXTRA_CODE, 'train_queue_v2.py': Path(__file__).resolve()}


class TrainQueueV2(TQ.TrainQueue):
    def __init__(self, a, reuse_ae_from=None, **kw):
        if Path(a.out).resolve() == (TQ.REPO / 'outputs/rl2_vla/fold27_trqam_20261102').resolve():
            raise TQ.QueueError('v2 must not write into the v1 out root; use a new --out')
        self.reuse_ae_from = reuse_ae_from
        super().__init__(a, **kw)

    def cmd(self, f, arm, stage):
        c = super().cmd(f, arm, stage)
        c[1] = str(HERE / 'finetune_fold_v2.py')
        if self.reuse_ae_from is not None:
            c += ['--reuse-ae-from', str(self.reuse_ae_from)]
        return c


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    reuse = None
    if '--reuse-ae-from' in argv:
        i = argv.index('--reuse-ae-from'); reuse = Path(argv[i + 1]); del argv[i:i + 2]
    if '--out' not in argv:
        argv += ['--out', str(V2_OUT)]
    return TQ.parse(argv), reuse


def main():
    a, reuse = parse()
    try:
        print(json.dumps(TrainQueueV2(a, reuse_ae_from=reuse).run()))
    except TQ.QueueError as e:
        print('QUEUE_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()
