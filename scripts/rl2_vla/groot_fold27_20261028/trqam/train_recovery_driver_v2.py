"""train_recovery_driver v2 = v1 (train_recovery_driver.py, unchanged; a running v1 driver keeps its bytes) with ONE fix
in the eval-completion gate.

Bug (v1): Driver.eval_ended_cleanly accepted the eval consumer's exit record only as 'no runnable job'. When the old
training coordinator holds a live TRAIN_REQUEST while the consumer finishes its last arm, eval_gpu_queue records the
normal exit 'yielded to trainer' (failures_this_run == []), so v1 waits forever although every eval row is complete.
Fix (v2): the exit record may be 'no runnable job' OR 'yielded to trainer'. The exit record is evaluated LAST, i.e. only
after every v1 check below passed, all unchanged:
  failures_this_run non-empty                                   -> REFUSED
  (fold, arm) set != the pinned expected set                    -> REFUSED
  any load error / audit error / non-audited row                -> REFUSED
  planned rows != pinned rows_per_arm                           -> REFUSED
  complete arm whose RUN_COMPLETE does not bind fold/arm/rows/progress sha256 -> REFUSED
  any arm not run_complete or with rows != rows_per_arm         -> keep waiting (also when the exit says yielded)
  exit record missing or any other value                        -> keep waiting
The read-only boundary (eval consumer gone, no harness receipt on the GPU, GPU holds only the pinned foreign identity,
old coordinator idle without live claims, no replacement running) is evaluated BEFORE this gate by the unchanged v1 loop.
Everything else (identity re-verification, SIGTERM to the two pinned identities only, live contract derivation, guardian
v3 + coordinator g3 spawn with start capture, REFUSED/DONE records, O_EXCL INTENT) is v1's code.

    python3 train_recovery_driver_v2.py --config <driver.json whose state_dir is NEW (v1's INTENT.json is never reused)>
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_recovery_driver as RD1     # noqa: E402  (unchanged)

Refused = RD1.Refused
CLEAN_EXITS = ('no runnable job', 'yielded to trainer')


class DriverV2(RD1.Driver):
    def eval_ended_cleanly(self):
        """(complete, why). Same checks and order as v1; only the final exit-record test accepts CLEAN_EXITS."""
        st = self.h.read_json(self.c['eval_queue_state'])
        if st is not None and st.get('failures_this_run'):
            raise Refused(f'eval consumer recorded failures {st["failures_this_run"][:3]}')
        ev = self.c['eval_complete']                     # {'root','host','expected': {fold_id: [arms]}, 'rows_per_arm': 40}
        want = {(f, a) for f, arms in ev['expected'].items() for a in arms}
        snap = self.eval_snapshot(ev['root'], ev['host'])
        got = {(f['fold_id'], arm): (f, a) for f in snap['folds'] for arm, a in f['arms'].items()}
        if set(got) != want:
            raise Refused(f'eval root coordinates differ from the pinned set: missing {sorted(want - set(got))[:3]} '
                          f'extra {sorted(set(got) - want)[:3]}')
        bad, incomplete, rows = [], [], 0
        n = int(ev['rows_per_arm'])
        for (fid, arm), (f, a) in sorted(got.items()):
            if a['load_error'] or a['audit_errors'] or any(not r['audit_ok'] for r in a['rows']):
                bad.append(f'{fid}/{arm}: {a["load_error"] or a["audit_errors"][:2]}')
            if f['planned_rows'] != n:
                bad.append(f'{fid}: planned rows {f["planned_rows"]} != pinned {n}')
            if not a['run_complete'] or len(a['rows']) != n:
                incomplete.append(f'{fid}/{arm} {len(a["rows"])}/{n}')
            else:
                rc = self.h.read_json(Path(ev['root']) / fid / arm / 'RUN_COMPLETE.json')
                prog = Path(ev['root']) / fid / arm / 'progress.jsonl'
                if rc is None or rc.get('fold_id') != fid or rc.get('arm') != arm or rc.get('rows') != n \
                        or rc.get('progress_sha256') != self.file_sha(prog):
                    bad.append(f'{fid}/{arm}: RUN_COMPLETE does not bind fold/arm/rows/progress sha')
            rows += len(a['rows'])
        if bad:
            raise Refused(f'eval failed load/audit/completion binding: {bad[:3]}')
        if incomplete:
            return False, f'eval incomplete: {rows}/{len(want) * n} rows; {incomplete[:3]}'
        if st is None or st.get('exit') not in CLEAN_EXITS:
            return False, f'eval consumer exit record {st and st.get("exit")!r}'
        return True, f'{rows} audited complete rows over {len(want)} pinned (fold, arm); consumer exit {st["exit"]!r}'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        return DriverV2(json.loads(a.config.read_text()), RD1.HostOS()).run()
    except (Refused, FileExistsError) as e:
        print('TRAIN_RECOVERY_REFUSED', e, file=sys.stderr); return 2


if __name__ == '__main__':
    sys.exit(main())
