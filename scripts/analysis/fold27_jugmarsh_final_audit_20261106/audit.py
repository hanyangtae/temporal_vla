#!/usr/bin/env python3
"""fold27 jug/marshmallow srv50 FINAL BINDING AUDIT + supplemental tables (read-only, stdlib, CPU, no GPU module).

Does NOT re-implement what existing tools already do -- it reads their outputs:
  aggregate_eval_report.py snapshot / merge  (row re-hash via run_eval_arm.load_progress, MIL decision audit, rates)
  scripts/collect/scan_video_integrity.py    (decode every video.mp4 under --grid-root: frames / freeze / noise flags)
It adds only the missing bindings and tables:
  B1 READY.json manifest sha; manifest sha == snapshot; production, rows n0..39
  B2 every RUN_COMPLETE.json binds fold/arm/rows=40/progress_sha256 == sha(progress.jsonl)        (24 expected)
  B3 every VERIFIED.json binds per_episode_tsv_sha256 == sha(per_episode.tsv); its verify log parses with errors []
     for all 4 arms; per_episode.tsv rows == snapshot rows (fold, arm, noise, base, success)        (6 expected)
  B4 snapshot: no load/pin/audit error, all rows audit_ok; merge report rows for this host == snapshot rows
  B5 exactly 6 folds x 4 arms x 40 n = 960 unique coordinates, 240 (fold, n) present in all 4 arms, and the row
     identity (j, env_seed, inference_seed, plan_id, base_success) identical across arms and equal to the manifest
  B6 attempts: each completed row's meta.json lies in rows/nNN/<progress attempt>/; the meta dir holds exactly one
     mp4, video.mp4, and no other mp4 exists in that attempt. Finished attempts not in progress (orphans) = error;
     unfinished attempts (no meta.json) = IGNORED PARTIAL, counted separately, never touched
  B7 video coverage: the 960 bound video dirs (relative to --root) are each present exactly once in the scan TSV with
     frames > 0 and flag OK; scan rows outside the bound set are classified (partial attempt / other) and listed
  B8 pins: FOLD_CONTRACT sha + fold/target_j/model sha + shared_checkpoint_reused False + heldout_proof flags, its
     val_episodes disjoint from the 40 eval ids; actor.npz/actor.json sha on disk == manifest; mil_qam65 = canonical
     common actor (eval_pipeline.COMMON_ACTOR_SHA, kind common, 500k); TRQAM kind/fold/50k/init == owner init sha;
     TRQAM DONE.json params/init; training QUEUE_FREEZE spec/sources/init == owner + --sources; every sources latents
     contract token_contract == 49 (native49)
Tables (rows = the audited snapshot rows): per arm overall / per task (instruction) / per fold SR; archived-baseline
rescue/damage with detected denominators (fired > 0, same rule as aggregate_eval_report.stats); paired vs mil_qam65 on
the exact same (fold, n): both, gained (qam 0 -> arm 1), lost (qam 1 -> arm 0), neither.
Status PASS only if every check passes with exact counts; otherwise FAIL (exit 1) and tables are marked PRELIMINARY.

    python3 audit.py --root outputs/eval/robocasa/groot_n15/fold27_jugmarsh_t49_20261105 --snapshot <srv50 snapshot.json> \
        --report <merge out>/report.json --scan <scan.tsv> --out <NEW dir> \
        [--owner outputs/analysis/groot_fold27_20261028/OFFLOAD_OWNER_JUGMARSH_20261105.json] \
        [--sources outputs/analysis/groot_fold27_20261028/trqam_sources_srv50_independent_t49_20261103.json] [--host srv50]
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/rl2_vla/groot_fold27_20261028/eval/runtime'))
import common as C                                     # noqa: E402  (stdlib only)
from eval_pipeline import ARM_ORDER, COMMON_ACTOR_SHA  # noqa: E402  (stdlib + common only)

ARMS = list(ARM_ORDER)
ROWS_PER_ARM = 40
IDENT = ('j', 'env_seed', 'inference_seed', 'plan_id', 'base_success')
TRQAM_STEPS = 50_000


def sha(p):
    return C.sha256_path(p)


def jload(p):
    return json.loads(Path(p).read_text())


def last_json(text):
    """Last top-level JSON object that runs to the end of the log (lines starting with '{')."""
    dec, starts = json.JSONDecoder(), [i for i in range(len(text)) if text[i] == '{' and (i == 0 or text[i - 1] == '\n')]
    for i in reversed(starts):
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            continue
        if not text[end:].strip():
            return obj
    raise ValueError('no trailing JSON result in verify log')


def jlines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


class Audit:
    def __init__(self, a):
        self.a, self.errors, self.notes = a, [], []
        self.root = Path(a.root).resolve()

    def err(self, where, msg):
        self.errors.append(f'{where}: {msg}')

    def check(self, ok, where, msg):
        if not ok:
            self.err(where, msg)
        return ok

    # ------------------------------------------------------------------ inputs
    def load(self):
        a = self.a
        self.owner = jload(a.owner)
        self.folds = list(self.owner['folds'])
        self.snap = jload(a.snapshot)
        self.rep = jload(a.report)
        self.inputs = {k: {'path': str(p), 'sha256': sha(p)} for k, p in
                       (('owner', a.owner), ('snapshot', a.snapshot), ('report', a.report), ('scan', a.scan), ('sources', a.sources)) if p}
        self.check(self.snap.get('host') == a.host and Path(self.snap['root']).resolve() == self.root, 'snapshot',
                   f'host/root {self.snap.get("host")}/{self.snap.get("root")} != {a.host}/{self.root}')
        self.check(any(s['host'] == a.host and Path(s['root']).resolve() == self.root and s['time'] == self.snap['time']
                       for s in self.rep.get('snapshots', [])), 'report', 'merge report was not built from this snapshot')
        self.sfold = {f['fold_id']: f for f in self.snap['folds']}
        self.check(sorted(self.sfold) == sorted(self.folds), 'snapshot', f'folds {sorted(self.sfold)} != owner {sorted(self.folds)}')

    # ------------------------------------------------------------------ B1-B4, B6, B8 per fold
    def fold(self, fid):
        d, f = self.root / fid, self.sfold.get(fid)
        res = {'fold_id': fid, 'arms': {}}
        if f is None:
            return res
        mp = d / 'manifest.json'
        m = jload(mp)
        res['manifest_sha256'] = sha(mp)
        self.check(res['manifest_sha256'] == f['manifest_sha256'], fid, 'manifest sha != snapshot')
        rd = d / 'READY.json'
        if self.check(rd.is_file(), fid, 'READY.json missing'):
            r = jload(rd)
            self.check(r.get('manifest_sha256') == res['manifest_sha256'] and r.get('fold_id') == fid and r.get('rows') == ROWS_PER_ARM
                       and sorted(r.get('arms', [])) == sorted(ARMS), fid, f'READY binding {r}')
        self.check(not m['smoke'] and [x['n'] for x in m['rows']] == list(range(ROWS_PER_ARM)) and sorted(m['arms']) == sorted(ARMS),
                   fid, 'manifest not production n0..39 x 4 arms')
        res['instruction'], res['target_j'], res['scene'] = m['instruction'], m['target_j'], m['scene']
        self.pins(fid, m, res)
        for arm in ARMS:
            res['arms'][arm] = self.arm(fid, arm, m, f['arms'].get(arm), d / arm)
        self.verified(fid, d, f)
        return res

    def arm(self, fid, arm, m, sa, out):
        w = f'{fid}/{arm}'
        r = {'run_complete_bound': False, 'videos': [], 'attempts_total': 0, 'ignored_partial': [], 'orphans': []}
        if not self.check(sa is not None, w, 'arm missing in snapshot'):
            return r
        for k in ('load_error', 'pin_error'):
            self.check(not sa.get(k), w, f'snapshot {k}: {sa.get(k)}')
        self.check(not sa.get('audit_errors'), w, f'snapshot audit_errors: {sa.get("audit_errors", [])[:3]}')
        self.check(len(sa['rows']) == ROWS_PER_ARM and all(x['audit_ok'] for x in sa['rows']), w,
                   f'snapshot rows {len(sa["rows"])} / audit_ok {sum(x["audit_ok"] for x in sa["rows"])}')
        prog = out / 'progress.jsonl'
        rc = out / 'RUN_COMPLETE.json'
        if self.check(rc.is_file() and prog.is_file(), w, 'RUN_COMPLETE.json or progress.jsonl missing'):
            b = jload(rc)
            ok = (b.get('fold_id') == fid and b.get('arm') == arm and b.get('rows') == ROWS_PER_ARM
                  and b.get('progress_sha256') == sha(prog))
            r['run_complete_bound'] = self.check(ok, w, f'RUN_COMPLETE does not bind fold/arm/rows/progress sha: {b}')
            r['progress_sha256'] = sha(prog)
        items = jlines(prog) if prog.is_file() else []
        bound = {}
        for it in items:
            n, att = it['n'], it['attempt']
            meta = out / it['meta']
            exp_dir = out / 'rows' / f'n{n:02d}' / att
            self.check(exp_dir in meta.parents and meta.is_file(), w, f'n{n}: meta {it["meta"]} not inside rows/n{n:02d}/{att}')
            mp4 = sorted(p.name for p in meta.parent.glob('*.mp4'))
            all_mp4 = sorted(exp_dir.rglob('*.mp4'))
            self.check(mp4 == ['video.mp4'] and all_mp4 == [meta.parent / 'video.mp4'], w,
                       f'n{n}: expected exactly one video.mp4 beside meta, found {mp4} / attempt {len(all_mp4)}')
            bound[n] = att
            r['videos'].append(str((meta.parent).relative_to(self.root)))
        self.check(sorted(bound) == list(range(ROWS_PER_ARM)), w, f'progress n set {sorted(bound)}')
        for adir in sorted((out / 'rows').glob('n*/a*')):
            r['attempts_total'] += 1
            n = int(adir.parent.name[1:])
            if bound.get(n) == adir.name:
                continue
            if any(adir.rglob('meta.json')):
                r['orphans'].append(str(adir.relative_to(self.root)))
                self.err(w, f'orphan finished attempt {adir.relative_to(self.root)}')
            else:
                r['ignored_partial'].append(str(adir.relative_to(self.root)))
        return r

    def verified(self, fid, d, f):
        p = d / 'VERIFIED.json'
        if not self.check(p.is_file(), fid, 'VERIFIED.json missing'):
            return
        v = jload(p)
        tsv = d / 'per_episode.tsv'
        self.check(v.get('fold_id') == fid and tsv.is_file() and v.get('per_episode_tsv_sha256') == sha(tsv), fid,
                   'VERIFIED per_episode_tsv_sha256 != sha(per_episode.tsv)')
        log = C.R / v.get('log', '')
        if self.check(v.get('log') and log.is_file(), fid, f'verify log missing: {v.get("log")}'):
            try:
                lr = last_json(log.read_text())   # verify_fold prints one indent=1 JSON; stderr is merged into the same log
                self.check(lr.get('errors') == [] and sorted(lr.get('arms', {})) == sorted(ARMS)
                           and all(x.get('errors') == [] and x.get('n') == ROWS_PER_ARM for x in lr['arms'].values()), fid,
                           f'verify log reports errors or incomplete arms: {lr.get("errors", [])[:3]}')
            except (ValueError, AttributeError) as e:
                self.err(fid, f'verify log not a clean verify_fold JSON result: {e}')
        if tsv.is_file():
            with tsv.open() as fh:
                t = {(r['arm'], int(r['noise'])): (int(r['base_success_archive']), int(r['success'])) for r in csv.DictReader(fh, delimiter='\t')}
            s = {(arm, x['n']): (x['base_success'], x['success']) for arm, a in f['arms'].items() for x in a['rows']}
            self.check(t == s, fid, f'per_episode.tsv rows ({len(t)}) differ from snapshot rows ({len(s)})')

    def pins(self, fid, m, res):
        mil = m['mil']
        fc = mil.get('fold_contract') or {}
        p = C.R / fc.get('path', '')
        ev_ids = {f'{x["plan_id"]}/{m["archive_machine"]}/{m["instruction"]}/s{m["scene"]}/j{x["j"]}/n{x["n"]}' for x in m['rows']}
        if self.check(fc.get('path') and p.is_file() and sha(p) == fc.get('sha256'), fid, 'FOLD_CONTRACT missing or sha mismatch'):
            c = jload(p)
            hp = c.get('heldout_proof') or {}
            self.check(c.get('fold_id') == fid and int(c.get('target_j', -1)) == int(m['target_j']) and c.get('model_sha256') == mil['sha256']
                       and c.get('shared_checkpoint_reused') is False, fid, 'FOLD_CONTRACT fold/target_j/model sha/shared mismatch')
            self.check(all(hp.get(k) is True for k in ('train_val_equals_fit160', 'train_val_disjoint_eval40', 'target_j_absent_from_train_val'))
                       and hp.get('train_episodes', 0) + hp.get('val_episodes', 0) == 160 and hp.get('test_episodes') == ROWS_PER_ARM
                       and int(m['target_j']) not in hp.get('train_val_jitters', [m['target_j']]), fid, f'heldout_proof incomplete: {hp}')
            self.check(len(ev_ids) == ROWS_PER_ARM and not set(c.get('val_episodes') or []) & ev_ids and c.get('val_episodes'), fid,
                       'FOLD_CONTRACT val_episodes missing or overlap the 40 eval ids')
            res['heldout'] = {k: hp.get(k) for k in ('train_episodes', 'val_episodes', 'test_episodes', 'train_val_jitters')}
        self.check((C.R / mil['model']).is_file() and sha(C.R / mil['model']) == mil['sha256'], fid, 'MIL model sha on disk')
        res['pins'] = {}
        for arm in ARMS:
            pin, w = m['arms'][arm], f'{fid}/{arm}'
            prov = pin.get('provenance') or {}
            ad = C.R / (pin.get('actor') or '')
            if self.check(pin.get('actor') and ad.is_dir(), w, 'actor dir missing'):
                self.check(sha(ad / 'actor.npz') == pin['actor_sha256'] == prov.get('actor_sha256')
                           and sha(ad / 'actor.json') == pin['actor_json_sha256'], w, 'actor.npz/actor.json sha != manifest pin')
            self.check(prov.get('production') is True and prov.get('smoke') is False and int(prov.get('common_steps', -1)) == C.COMMON_STEPS,
                       w, 'actor provenance not production / common 500k')
            if arm == 'mil_qam65':
                self.check(prov.get('kind') == 'common' and pin['actor_sha256'] == COMMON_ACTOR_SHA, w, 'common QAM actor != canonical')
            else:
                tag = arm.split('_')[-1]
                self.check(prov.get('kind') == f'trqam_{tag}' and prov.get('fold_id') == fid and int(prov.get('trqam_steps', -1)) == TRQAM_STEPS
                           and prov.get('init_params_sha256') == self.owner['init_sha256'], w, 'TRQAM provenance kind/fold/50k/init')
                run = C.R / prov.get('run', '')
                if not str(Path(prov.get('run', '')).as_posix()).startswith(self.owner['out'].rstrip('/') + '/'):
                    self.notes.append(f'{w}: TRQAM run {prov.get("run")} is outside owner out root (DONE/init still checked)')
                if self.check((run / 'DONE.json').is_file(), w, f'TRQAM DONE.json missing at {prov.get("run")}'):
                    dn = jload(run / 'DONE.json')
                    self.check(dn.get('params_sha256') == prov.get('params_sha256') and dn.get('smoke') is not True
                               and (dn['fingerprint'].get('init') or {}).get('params_sha256') == self.owner['init_sha256'], w, 'DONE params/init')
            res['pins'][arm] = {'actor_sha256': pin.get('actor_sha256'), 'kind': prov.get('kind'), 'run': prov.get('run')}

    def training_freeze(self):
        qf = C.R / self.owner['out'] / 'queue' / 'QUEUE_FREEZE.json'
        res = {'queue_freeze': str(qf.relative_to(C.R))}
        if not self.a.sources:
            res['status'] = 'NOT CHECKED (no --sources given; actor/DONE init pins are still checked)'
            self.notes.append('native49 token_contract / QUEUE_FREEZE not checked: pass --sources <training sources json>')
            return res
        if not self.check(qf.is_file(), 'native49', 'training QUEUE_FREEZE.json missing'):
            return res
        z = jload(qf)
        self.check(z.get('fold_spec_sha256') == self.owner['spec_sha256'] and z.get('init_params_sha256') == self.owner['init_sha256']
                   and z.get('sources_sha256') == sha(self.a.sources), 'native49', 'QUEUE_FREEZE spec/init/sources != owner / --sources')
        res['token_contract'] = {}
        for key, s in jload(self.a.sources).items():
            cp = C.R / s['latents'] / 'contract.json'
            tc = jload(cp).get('token_contract') if cp.is_file() else None
            res['token_contract'][key] = tc
            self.check(tc == 49 and (C.R / s['latents'] / 'COMPLETE.json').is_file(), 'native49', f'{key}: token_contract {tc} / COMPLETE')
        return res

    # ------------------------------------------------------------------ B5, B7, tables
    def coords(self):
        rows = [r for r in self.rep['rows'] if r['host'] == self.a.host and r['fold_id'] in self.folds]
        snap_rows = {(f['fold_id'], arm, x['n']): x for f in self.snap['folds'] for arm, a in f['arms'].items() for x in a['rows']}
        keys = [(r['fold_id'], r['policy'], r['n']) for r in rows]
        self.check(len(keys) == len(set(keys)) == len(self.folds) * len(ARMS) * ROWS_PER_ARM, 'coords',
                   f'{len(set(keys))} unique / {len(keys)} rows, expected {len(self.folds) * len(ARMS) * ROWS_PER_ARM}')
        self.check(set(keys) == set(snap_rows) and all(snap_rows[(r['fold_id'], r['policy'], r['n'])]['success'] == r['success'] for r in rows),
                   'coords', 'merge report rows differ from snapshot rows')
        by = defaultdict(dict)
        for r in rows:
            by[(r['fold_id'], r['n'])][r['policy']] = r
        common = [k for k, v in by.items() if sorted(v) == sorted(ARMS)]
        self.check(len(common) == len(self.folds) * ROWS_PER_ARM, 'coords', f'{len(common)} (fold, n) in all 4 arms, expected 240')
        for k, v in by.items():
            ids = {tuple(x.get(f) for f in IDENT) for x in v.values()}
            self.check(len(ids) == 1, 'coords', f'{k}: row identity differs across arms {ids}')
        for fid in self.folds:
            mrows = {x['n']: x for x in jload(self.root / fid / 'manifest.json')['rows']}
            for (f, n), v in by.items():
                if f == fid and n in mrows:
                    x = next(iter(v.values()))
                    self.check(all(x.get(k) == mrows[n].get(k) for k in IDENT), 'coords', f'{fid} n{n}: row != manifest')
        return rows, sorted(common)

    def videos(self, fold_res):
        bound = {v: (fid, arm) for fid, fr in fold_res.items() for arm, ar in fr['arms'].items() for v in ar['videos']}
        partial = {p for fr in fold_res.values() for ar in fr['arms'].values() for p in ar['ignored_partial']}
        with open(self.a.scan) as fh:
            scan = list(csv.DictReader(fh, delimiter='\t'))
        cnt = Counter(r['rel_path'] for r in scan)
        dup = [p for p, c in cnt.items() if c > 1]
        self.check(not dup, 'video', f'duplicate scan paths {dup[:3]}')
        sr = {r['rel_path']: r for r in scan}
        missing, bad = [], []
        for v in bound:
            r = sr.get(v)
            if r is None:
                missing.append(v); continue
            try:
                frames = int(r['frames'])
            except (TypeError, ValueError):
                frames = -1
            if frames <= 0 or r['flag'] != 'OK':
                bad.append({'rel_path': v, 'frames': r['frames'], 'flag': r['flag']})
        self.check(len(bound) == len(self.folds) * len(ARMS) * ROWS_PER_ARM, 'video', f'{len(bound)} bound video dirs, expected 960')
        self.check(not missing, 'video', f'{len(missing)} completed-row videos absent from scan: {missing[:3]}')
        self.check(not bad, 'video', f'{len(bad)} completed-row videos not OK / zero frames: {bad[:3]}')
        extra = [p for p in sr if p not in bound]
        ext_partial = [p for p in extra if any(p == q or p.startswith(q + '/') for q in partial)]
        ext_other = [p for p in extra if p not in ext_partial]
        return {'bound': len(bound), 'scanned_rows': len(scan), 'missing': missing, 'not_ok': bad,
                'flags_bound': dict(Counter(sr[v]['flag'] for v in bound if v in sr)),
                'frames_bound': sum(int(sr[v]['frames']) for v in bound if v in sr and str(sr[v]['frames']).lstrip('-').isdigit()),
                'scan_extra_partial_attempts': ext_partial, 'scan_extra_other': ext_other,
                'scan_extra_flags': dict(Counter(sr[p]['flag'] for p in extra))}

    @staticmethod
    def stat(rs):
        n, s = len(rs), sum(r['success'] for r in rs)
        f0 = [r for r in rs if not r['base_success']]; s0 = [r for r in rs if r['base_success']]
        return {'N': n, 'success': s, 'SR': round(s / n, 4) if n else None, 'base_success': len(s0),
                'rescue': sum(r['success'] for r in f0), 'rescue_den': len(f0), 'rescue_detected_den': sum(1 for r in f0 if (r['fired'] or 0) > 0),
                'damage': sum(1 for r in s0 if not r['success']), 'damage_den': len(s0),
                'damage_detected_den': sum(1 for r in s0 if (r['fired'] or 0) > 0)}

    def tables(self, rows, instr):
        t = {'overall': [], 'task': [], 'fold': [], 'paired_vs_mil_qam65': []}
        for arm in ARMS:
            ra = [r for r in rows if r['policy'] == arm]
            t['overall'].append({'policy': arm, **self.stat(ra)})
            for task in sorted({instr[r['fold_id']] for r in ra}):
                t['task'].append({'task': task, 'policy': arm, **self.stat([r for r in ra if instr[r['fold_id']] == task])})
            for fid in self.folds:
                t['fold'].append({'fold_id': fid, 'task': instr[fid], 'policy': arm, **self.stat([r for r in ra if r['fold_id'] == fid])})
        q = {(r['fold_id'], r['n']): r['success'] for r in rows if r['policy'] == 'mil_qam65'}
        for arm in ARMS[1:]:
            for scope in ['ALL'] + self.folds:
                pr = [(q[(r['fold_id'], r['n'])], r['success']) for r in rows
                      if r['policy'] == arm and (r['fold_id'], r['n']) in q and scope in ('ALL', r['fold_id'])]
                t['paired_vs_mil_qam65'].append({'policy': arm, 'scope': scope, 'pairs': len(pr), 'both': sum(1 for a, b in pr if a and b),
                                                 'gained': sum(1 for a, b in pr if not a and b), 'lost': sum(1 for a, b in pr if a and not b),
                                                 'neither': sum(1 for a, b in pr if not a and not b)})
        return t

    # ------------------------------------------------------------------ run
    def run(self):
        self.load()
        fold_res = {fid: self.fold(fid) for fid in self.folds}
        native = self.training_freeze()
        rows, common = self.coords()
        vid = self.videos(fold_res)
        instr = {fid: fr.get('instruction') for fid, fr in fold_res.items()}
        tabs = self.tables(rows, instr)
        counts = {'folds': len(fold_res), 'run_complete_bound': sum(ar['run_complete_bound'] for fr in fold_res.values() for ar in fr['arms'].values()),
                  'verified': sum((self.root / f / 'VERIFIED.json').is_file() for f in self.folds), 'rows': len(rows), 'common_fold_n': len(common),
                  'attempt_dirs': sum(ar['attempts_total'] for fr in fold_res.values() for ar in fr['arms'].values()),
                  'ignored_partial_attempts': sum(len(ar['ignored_partial']) for fr in fold_res.values() for ar in fr['arms'].values()),
                  'orphan_attempts': sum(len(ar['orphans']) for fr in fold_res.values() for ar in fr['arms'].values()),
                  'videos_bound': vid['bound']}
        exact = counts['run_complete_bound'] == 24 and counts['verified'] == 6 and counts['rows'] == 960 and counts['common_fold_n'] == 240
        status = 'PASS' if exact and not self.errors else 'FAIL'
        return {'kind': 'fold27_jugmarsh_final_binding_audit_v1', 'status': status, 'errors': self.errors, 'notes': self.notes, 'counts': counts,
                'inputs': self.inputs, 'audit_py_sha256': sha(Path(__file__)), 'root': str(self.root), 'host': self.a.host,
                'native49_and_training_freeze': native, 'video': vid, 'folds': fold_res, 'tables': tabs,
                'scope': ['episode-outcome aggregation over these fixed 960 coordinates (6 folds x 40 n x 4 policies), srv50 only',
                          'archived baseline = original collection outcome of the same rows; equivalence of its initialisation to the '
                          'current eval is NOT established -> rescue/damage are paired diagnostics, not causal effects',
                          'policy-vs-common-QAM pairs are interventional comparisons on identical coordinates/seeds/MIL gate; '
                          'counts only, no significance test, no generalisation beyond these folds',
                          'length / pooling / phase: N-A (episode outcome; no activation separation claimed)',
                          'video check = existing scan_video_integrity decode/freeze/noise flags; no manual content review',
                          'tables are PRELIMINARY unless status == PASS']}


def write(out, res):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'final_audit.json').write_text(json.dumps(res, indent=1, default=str))
    for name, rows in res['tables'].items():
        if rows:
            with (out / f'{name}.csv').open('x', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    c, tag = res['counts'], '' if res['status'] == 'PASS' else ' (PRELIMINARY: audit not PASS)'
    md = [f'# fold27 jug/marshmallow srv50 final binding audit: {res["status"]}', '',
          f'audit.py sha256 `{res["audit_py_sha256"]}`; root `{res["root"]}`', '',
          '| check | value | expected |', '|---|---|---|',
          f'| RUN_COMPLETE bound | {c["run_complete_bound"]} | 24 |', f'| VERIFIED bound | {c["verified"]} | 6 |',
          f'| unique rows | {c["rows"]} | 960 |', f'| (fold, n) in all 4 policies | {c["common_fold_n"]} | 240 |',
          f'| completed-row videos bound / missing / not OK | {res["video"]["bound"]} / {len(res["video"]["missing"])} / {len(res["video"]["not_ok"])} | 960 / 0 / 0 |',
          f'| orphan finished attempts | {c["orphan_attempts"]} | 0 |',
          f'| ignored partial attempts (unfinished, kept) | {c["ignored_partial_attempts"]} | reported |',
          f'| scan rows outside bound set: partial / other | {len(res["video"]["scan_extra_partial_attempts"])} / {len(res["video"]["scan_extra_other"])} | reported |',
          '', f'## Errors ({len(res["errors"])})', ''] + ([f'- {e}' for e in res['errors'][:300]] or ['- none'])
    for name in ('overall', 'task', 'fold', 'paired_vs_mil_qam65'):
        rows = res['tables'][name]
        if rows:
            md += ['', f'## {name}{tag}', '', '| ' + ' | '.join(rows[0]) + ' |', '|' + '---|' * len(rows[0])]
            md += ['| ' + ' | '.join('' if v is None else str(v) for v in r.values()) + ' |' for r in rows]
    md += ['', '## Scope / limitations', ''] + [f'- {x}' for x in res['scope']]
    (out / 'final_audit.md').write_text('\n'.join(md) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--snapshot', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--scan', required=True); ap.add_argument('--out', required=True); ap.add_argument('--host', default='srv50')
    ap.add_argument('--owner', default=str(REPO / 'outputs/analysis/groot_fold27_20261028/OFFLOAD_OWNER_JUGMARSH_20261105.json'))
    ap.add_argument('--sources', default=None, help='training sources JSON used by the jug/marsh QUEUE_FREEZE (enables native49 check)')
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        print(f'refusing existing --out {a.out}', file=sys.stderr); return 2
    res = Audit(a).run()
    write(a.out, res)
    print(json.dumps({'status': res['status'], 'errors': len(res['errors']), 'counts': res['counts'], 'out': a.out}))
    return 0 if res['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
