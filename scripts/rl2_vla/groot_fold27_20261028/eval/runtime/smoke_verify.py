"""Durable CPU watcher/verifier for the fold27 integration smoke s1 (read-only on the smoke outputs).

Waits (bounded) until every smoke arm has its rows, stops early if the systemd --user unit fails or ends without them,
then validates all current rows: progress re-hash (meta/traj), seeds, per-request decision rules (MIL every inference,
gate == mode rule, ungated action untouched, composed pass = same initial noise + QAM flow steps 0..3, RNG restored,
gate0 identity, finite), trace complete + one reset, HTTP sends == serve inferences, smoke_base success == archive.
Historical failure markers are preserved and allowed ONLY if they are exactly the recovered startup
OSError(98, 'Address already in use') that happened before any artifact of that arm existed (no episode executed).

PASS  -> outputs/analysis/groot_fold27_20261028/SMOKE_VERIFIED.json (write-once), exit 0
FAIL  -> outputs/analysis/groot_fold27_20261028/SMOKE_VERIFY_FAILED.<ns>.json, exit 1 (validation) / 3 (timeout / unit)

    python3 smoke_verify.py --unit fold27-gated-retry-20261102 [--timeout 10800] [--poll 30]
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

MANIFEST = C.SMOKE_ROOT + '/s1/OpenDrawer-left_s2_j2/manifest.json'
MANIFEST_SHA = '1de2df61d398183d6f7f1b51a96e7bacf8a1c6ff930513570796044f6a8896b4'
ARMS = ('smoke_base', 'smoke_always', 'smoke_gated')
OUT = 'outputs/analysis/groot_fold27_20261028'
ALLOWED_STARTUP_ERROR = "OSError(98, 'Address already in use')"


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def unit_state(unit):
    r = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'Result', '-p', 'LoadState'],
                       capture_output=True, text=True)
    return dict(x.split('=', 1) for x in r.stdout.split('\n') if '=' in x)


def rows_done(arm_dir):
    return len(lines(arm_dir / 'progress.jsonl'))


def check_failures(arm, d):
    """Return (allowed markers, errors). Allowed: exact startup EADDRINUSE before any artifact of this arm."""
    allowed, errs = [], []
    others = [p for p in d.iterdir() if p.name != 'failures']
    first_artifact = min((p.stat().st_mtime for p in others if p.name != 'progress.jsonl' or p.stat().st_size), default=None)
    prog_t = [x['time'] for x in lines(d / 'progress.jsonl')]
    for f in sorted((d / 'failures').glob('*.json')) if (d / 'failures').exists() else []:
        m = json.loads(f.read_text())
        t = float(m.get('time', 0))
        rows_dir = d / 'rows'
        attempts = [p for p in rows_dir.glob('n*/a*')] if rows_dir.exists() else []
        decs = [x['time'] for g in d.glob('*.decisions.jsonl') for x in lines(g)]
        no_episode = (all(a.stat().st_mtime > t for a in attempts) and all(x > t for x in decs + prog_t)
                      and (first_artifact is None or first_artifact > t))
        if m.get('error') == ALLOWED_STARTUP_ERROR and m.get('type') == 'OSError' and no_episode:
            allowed.append({'arm': arm, 'marker': C.rel(f), 'sha256': C.sha256_path(f), 'time': t,
                            'reason': 'startup port probe EADDRINUSE (TIME_WAIT) before any episode; recovered by retry'})
        else:
            errs.append(f'{arm}: failure marker {f.name} not an allowed recovered startup failure: {m.get("error")}')
    return allowed, errs


def validate(root=None, manifest=None, arms=ARMS, rows=(0, 6), check_files=True):
    import run_eval_arm as R
    mp = Path(manifest or C.R / MANIFEST)
    errs, table, allowed = [], {}, []
    if manifest is None and C.sha256_path(mp) != MANIFEST_SHA:
        return ['manifest sha differs from pinned smoke s1'], table, allowed
    m = json.loads(mp.read_text())
    if not m['smoke']:
        return ['not a smoke manifest'], table, allowed
    base = {r['n']: r for r in m['rows']}
    for arm in arms:
        d = mp.parent / arm
        try:
            R.verify_manifest(mp, arm, check_files=check_files)
            done = R.load_progress(d, m, arm)                      # re-hash every finished row
        except Exception as e:  # noqa: BLE001
            errs.append(f'{arm}: {e!r}'); continue
        if sorted(done) != sorted(rows):
            errs.append(f'{arm}: rows {sorted(done)} != {sorted(rows)}')
        a, e = check_failures(arm, d); allowed += a; errs += e
        mode = m['arms'][arm]['mode']
        dec = {}
        for f in glob.glob(str(d / '*.decisions.jsonl')):
            dec.update({x['request']: x for x in lines(f)})
        for n, it in sorted(done.items()):
            lo, hi = it['requests']
            seg = [dec.get(i) for i in range(lo, hi + 1)]
            if None in seg or len(seg) != it['http_sends'] or not seg:
                errs.append(f'{arm} n{n}: decisions incomplete'); continue
            if [x['mil']['t'] for x in seg] != list(range(len(seg))):
                errs.append(f'{arm} n{n}: MIL t not 0..T-1 (reset/every-inference)')
            for x in seg:
                tag = f'{arm} n{n} r{x["request"]}'
                want = mode == 'always' or (mode == 'gated' and x['mil']['fired'])
                if x['mode'] != mode or 'score' not in x['mil']:
                    errs.append(f'{tag}: mode/MIL score missing')
                elif x['mil']['fired'] != (x['mil']['score'] > 0.5):
                    errs.append(f'{tag}: fired != score>0.5')
                if x['gate'] != want:
                    errs.append(f'{tag}: gate rule')
                if not x['gate'] and (x['qam_delta_max_normalized'] != 0 or x['noise_sha256_composed'] is not None or x['qam_steps']):
                    errs.append(f'{tag}: ungated action modified')
                if x['gate'] and (x['noise_sha256_composed'] != x['noise_sha256_first_pass'] or x['qam_steps'] != [0, 1, 2, 3]):
                    errs.append(f'{tag}: composed initial noise / QAM steps')
                if not (x['action_finite'] and x['off_identity_verified'] and x['rng_restored_equal']):
                    errs.append(f'{tag}: finite/identity/RNG')
            if sum(x['gate'] for x in seg) != it['composed']:
                errs.append(f'{arm} n{n}: composed count')
            ev = [x['event'] for x in lines(d / it['trace'])]
            if not ev or ev[-1] != 'complete' or ev.count('reset') != 1 or ev.count('http_send') != it['http_sends']:
                errs.append(f'{arm} n{n}: collector trace incomplete / reset count / sends')
            if it['env_seed'] != base[n]['env_seed'] or it['inference_seed'] != base[n]['inference_seed']:
                errs.append(f'{arm} n{n}: seeds differ from archive row')
            if mode == 'base' and it['success'] != base[n]['base_success']:
                errs.append(f'{arm} n{n}: base success {it["success"]} != archive {base[n]["base_success"]}')
            table[f'{arm}/n{n}'] = {'success': it['success'], 'archive': base[n]['base_success'], 'inferences': it['http_sends'],
                                    'composed': it['composed'], 'fired': it['fired'], 'first_fire_t': it['first_fire_t'],
                                    'files_sha256': it['files_sha256']}
    return errs, table, allowed


def main(argv=None, unit_fn=unit_state, sleep=time.sleep):
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', help='systemd --user unit running the pending arm(s)')
    ap.add_argument('--timeout', type=float, default=3 * 3600); ap.add_argument('--poll', type=float, default=30)
    ap.add_argument('--manifest', type=Path); ap.add_argument('--out', type=Path, default=C.R / OUT)
    ap.add_argument('--rows', default='0,6')
    a = ap.parse_args(argv)
    rows = tuple(int(x) for x in a.rows.split(','))
    mp = Path(a.manifest or C.R / MANIFEST)
    t0 = time.time()

    def fail(code, errs, table=None):
        p = Path(a.out) / f'SMOKE_VERIFY_FAILED.{time.time_ns()}.json'
        C.write_json_new(p, {'status': 'FAIL', 'errors': errs, 'rows': table or {}, 'manifest': C.rel(mp), 'time': time.time()})
        print(json.dumps({'SMOKE': 'FAIL', 'marker': str(p), 'errors': errs[:20]}, indent=1)); return code

    ended_once = False
    while True:
        counts = {arm: rows_done(mp.parent / arm) for arm in ARMS}
        if all(c >= len(rows) for c in counts.values()):
            break
        if a.unit:
            st = unit_fn(a.unit)
            if st.get('ActiveState') == 'failed' or st.get('Result', 'success') != 'success':
                return fail(3, [f'unit {a.unit} failed: {st}; rows {counts}'])
            if st.get('ActiveState') in ('inactive', 'deactivating') or st.get('LoadState') == 'not-found':
                if ended_once:                                     # one extra poll for the last progress line
                    return fail(3, [f'unit {a.unit} ended without all rows: {st}; rows {counts}'])
                ended_once = True
        if time.time() - t0 > a.timeout:
            return fail(3, [f'timeout {a.timeout}s; rows {counts}'])
        sleep(a.poll)
    errs, table, allowed = validate(manifest=a.manifest, rows=rows)
    if errs:
        return fail(1, errs, table)
    rec = {'status': 'PASS', 'manifest': C.rel(mp), 'manifest_sha256': C.sha256_path(mp), 'arms': list(ARMS), 'rows': table,
           'allowed_historical_failures': allowed, 'validator_sha256': C.sha256_path(Path(__file__)),
           'scope': 'integration smoke (20-step diagnostic actor); runtime contract only, no performance claim'}
    p = Path(a.out) / 'SMOKE_VERIFIED.json'
    C.write_json_new(p, rec)
    print(json.dumps({'SMOKE': 'PASS', 'marker': str(p), 'rows': len(table), 'allowed_historical_failures': len(allowed)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
