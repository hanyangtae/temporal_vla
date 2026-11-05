"""fold27 paired recovery timing: same-coordinate common-QAM failure -> TRQAM success, from the FIRST ACTUAL QAM
composition to the first success. READ-ONLY, stdlib only, no pickle, no new rollouts. Outputs are NEW files only.

  collect  on each host, one eval root:
             analyze.py collect --root <eval root> --host <label> --out <new collection.json>
           Every production manifest <root>/<fold>/manifest.json (smoke skipped+listed) x every PRODUCTION arm
           (common.PRODUCTION_ARMS, 4) x every manifest row -> exactly one record (data or explicit errors; never omitted).
           Per arm: run_eval_arm.verify_manifest(check_files=False), run_eval_arm.load_progress (re-hashes meta/traj) and
           verify_fold.audit_decisions (per-inference MIL gate audit) -- all imported UNCHANGED -- run BEFORE any timing.
  merge    anywhere:
             analyze.py merge --input a.json b.json ... --expected-rows 4320 --out-dir <new dir> [--reference report.json]
           rows.json/rows.csv (all records), selected_cases.json/csv, summary.json + summary_{policy,task,fold}.csv,
           report.json/report.md. Exit 0 = PASS, 1 = completed with issues (status lists them), 2 = refused.

DEFINITIONS (stated, not estimated):
  success_d  = meta.first_success_step: 1-based inference (decision) count at the first chunk-level collector success
               (success_timing.eval_d, unchanged: n_action_steps 5, max_episode_steps 720, asserts meta.steps == d, and
               first_success_step must be null for a failure). Chunk resolution; no seconds, no raw env-step index.
  t          = 0-based local inference index of the FIRST ACTUAL QAM COMPOSITION = request - lo, where [lo, hi] is the
               exact progress.requests range of the episode in its decisions log (decisions_log named by progress).
               Actual composition of a decision = gate is true AND noise_sha256_composed is present AND equal to
               noise_sha256_first_pass AND qam_steps == [0, 1, 2, 3]. mil.fired alone is NOT composition (kept separately
               as first_fired_t). A gate=true record that fails these checks, a gate=false record carrying a composed
               noise hash, or audited composed count != progress.composed make the row INVALID (error), never coerced.
               A composed pass whose qam_delta_max_normalized == 0 still counts as composition (flag first_composed_delta_zero).
  inclusive_count                 = d - t       (inferences from the first composed one through the success one, both incl.)
  exclusive_after_composed_chunk  = d - (t + 1) (inferences after the first composed chunk, through the success one)
  failure or never-composed -> both null (timing_status failure / no_composition).
  Quantiles: statistics.quantiles(method='inclusive') (linear interpolation between order statistics, = numpy 'linear');
  N=1 -> q1=median=q3=the value; N=0 -> {'N': 0}. Representative median case = lower-median element of the cases sorted
  by (inclusive_count, fold_id, n) -- an actual case, not an interpolated value.

LIMITATIONS: descriptive only. Same coordinate = same manifest row (fold, n: env/inference seed, plan, j); equivalence of
the full pre-intervention trajectory between the QAM and TRQAM runs is NOT verified. The archived baseline label
(manifest base_success) is the original collection outcome, not a re-run; 'archived_base_fail' marks cases that are also
rescues relative to the archive. Video content is not inspected (path + stat only). Data-file hashes are re-checked by the
existing load_progress (meta/traj) and recorded here as provenance (manifest/progress/meta/trace/decisions sha256).
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RT = ROOT / 'scripts/rl2_vla/groot_fold27_20261028/eval/runtime'
ST = ROOT / 'scripts/analysis/fold27_success_timing_20261104'
QAM, TRQAM = 'mil_qam65', ('mil_trqam_basic', 'mil_trqam_pbrs', 'mil_trqam_mix')
POLICIES = (QAM,) + TRQAM
EXPECTED_SELECTED = {'mil_trqam_basic': 65, 'mil_trqam_pbrs': 64, 'mil_trqam_mix': 82}
EXPECTED_FOLDS, ROWS_PER_FOLD = 27, 40
QAM_STEPS = [0, 1, 2, 3]
N_RE = re.compile(r'^n(\d+)\b')
KIND = 'fold27_paired_recovery_collection_v1'
IDENTITY = ('instruction', 'scene', 'j', 'env_seed', 'inference_seed', 'plan_id', 'base_success')
CONVENTION = ('success_d = meta.first_success_step (1-based inference count of first chunk-level success); t = 0-based local '
              'index (request - progress.requests[0]) of the first ACTUAL QAM composition; inclusive_count = d - t; '
              'exclusive_after_composed_chunk = d - (t + 1); failure / never-composed -> null; delta==0 composed passes count.')


class AnalysisError(ValueError):
    pass


def need(c, msg):
    if not c:
        raise AnalysisError(msg)


def _rt():
    for p in (RT, ST):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import common as C                          # noqa: E402  (unchanged)
    from run_eval_arm import load_progress, verify_manifest  # noqa: E402  (unchanged)
    from verify_fold import audit_decisions     # noqa: E402  (unchanged)
    import success_timing as S                  # noqa: E402  (unchanged: eval_d, summ)
    return C, load_progress, verify_manifest, audit_decisions, S


class FileCache:
    """Decision logs are shared by all episodes of a serve: parse + hash each file once."""

    def __init__(self, sha):
        self.sha, self._h, self._d = sha, {}, {}

    def hash(self, p):
        p = str(p)
        if p not in self._h:
            self._h[p] = self.sha(p)
        return self._h[p]

    def decisions(self, p):
        p = str(p)
        if p not in self._d:
            recs, dup = {}, set()
            for i, x in enumerate(Path(p).read_text().splitlines()):
                if not x.strip():
                    continue
                d = json.loads(x)
                need(isinstance(d, dict) and isinstance(d.get('request'), int), f'{p}:{i + 1}: decision record without int request')
                if d['request'] in recs:
                    dup.add(d['request'])
                recs[d['request']] = d
            self._d[p] = (recs, dup)
        return self._d[p]


# ------------------------------------------------------------------ pure timing core (unit-tested)
def actual_composed(d):
    """True / False for one decision record; raises on an invalid gate/composition record."""
    gate = d.get('gate')
    need(isinstance(gate, bool), f'r{d.get("request")}: gate {gate!r} not bool')
    comp, first = d.get('noise_sha256_composed'), d.get('noise_sha256_first_pass')
    if not gate:
        need(comp is None, f'r{d["request"]}: gate=false but noise_sha256_composed present')
        return False
    need(comp is not None and first is not None and comp == first,
         f'r{d["request"]}: gate=true but composed noise {comp!r} != first pass {first!r}')
    need(d.get('qam_steps') == QAM_STEPS, f'r{d["request"]}: gate=true but qam_steps {d.get("qam_steps")!r} != {QAM_STEPS}')
    return True


def episode_decisions(recs, dup, lo, hi):
    need(isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi, f'bad progress.requests [{lo}, {hi}]')
    need(not any(lo <= r <= hi for r in dup), f'duplicate request ids inside [{lo}, {hi}] of the decisions log')
    got = [recs.get(i) for i in range(lo, hi + 1)]
    miss = [lo + i for i, d in enumerate(got) if d is None]
    need(not miss, f'decisions log lacks requests {miss[:5]}{"..." if len(miss) > 5 else ""} of [{lo}, {hi}]')
    return got


def first_events(got, composed_expected):
    """(first_composed_t, first_fired_t, composed_count, first composed record). t = local 0-based offset."""
    flags = [actual_composed(d) for d in got]
    n = sum(flags)
    need(n == composed_expected, f'actual composed count {n} != progress.composed {composed_expected}')
    tc = next((i for i, f in enumerate(flags) if f), None)
    tf = next((i for i, d in enumerate(got) if bool((d.get('mil') or {}).get('fired'))), None)
    return tc, tf, n, (got[tc] if tc is not None else None)


def timing(success, d, tc):
    """(timing_status, inclusive_count, exclusive_after_composed_chunk)."""
    if not success:
        need(d is None, f'failure with success_d {d}')
        return 'failure', None, None
    need(isinstance(d, int) and d >= 1, f'success with success_d {d!r}')
    if tc is None:
        return 'no_composition', None, None
    need(0 <= tc < d, f'first composition t={tc} not before success d={d}')
    return 'valid', d - tc, d - (tc + 1)


# ------------------------------------------------------------------ collect (one eval root)
def _rel(p, root):
    return str(Path(p).relative_to(root))


def collect(root, host, rt=None):
    C, load_progress, verify_manifest, audit_decisions, S = rt or _rt()
    root = Path(root)
    fc = FileCache(C.sha256_path)
    out = {'kind': KIND, 'host': host, 'root': str(root), 'convention': CONVENTION, 'skipped_smoke': [],
           'unexpected_arms': [], 'fold_errors': [], 'rows': []}
    mps = sorted(root.glob('*/manifest.json'))
    if not mps:
        out['fold_errors'].append({'error': f'no */manifest.json under {root}'})
    for mp in mps:
        try:
            m = json.loads(mp.read_text())
            if m.get('smoke'):
                out['skipped_smoke'].append(_rel(mp, root)); continue
            for k in ('fold_id', 'rows', 'arms', 'instruction', 'scene'):
                need(k in m, f'manifest lacks {k}')
        except (OSError, ValueError, AnalysisError) as e:
            out['fold_errors'].append({'manifest': _rel(mp, root), 'error': f'{type(e).__name__}: {e}'}); continue
        fdir, msha = mp.parent, fc.hash(mp)
        out['unexpected_arms'] += [{'fold_id': m['fold_id'], 'arm': a} for a in sorted(m['arms']) if a not in POLICIES]
        for arm in POLICIES:
            adir = fdir / arm
            arm_err, done, audit_bad, audit_whole = [], {}, defaultdict(list), []
            try:
                need(arm in m['arms'], f'production arm {arm} not in manifest')
                verify_manifest(mp, arm, check_files=False)
                need((adir / 'progress.jsonl').is_file(), f'{_rel(adir, root)}/progress.jsonl missing')
                done = load_progress(adir, m, arm)
                for e in audit_decisions(done, adir):
                    mm = N_RE.match(e)
                    (audit_bad[int(mm.group(1))] if mm else audit_whole).append(e)
            except Exception as e:  # noqa: BLE001 - recorded on every row of this arm, never hidden
                arm_err.append(f'arm: {type(e).__name__}: {e}')
            psha = fc.hash(adir / 'progress.jsonl') if (adir / 'progress.jsonl').is_file() else None
            for r in m['rows']:
                out['rows'].append(_row(root, host, m, mp, msha, arm, adir, psha, r, done.get(r.get('n')),
                                        arm_err + [f'audit: {x}' for x in audit_whole + audit_bad.get(r.get('n'), [])], fc, S))
    return out


def _row(root, host, m, mp, msha, arm, adir, psha, r, it, errs, fc, S):
    rec = {'host': host, 'root': str(root), 'fold_id': m['fold_id'], 'instruction': m['instruction'], 'scene': m['scene'],
           'policy': arm, 'n': r.get('n'), 'j': r.get('j'), 'env_seed': r.get('env_seed'),
           'inference_seed': r.get('inference_seed'), 'plan_id': r.get('plan_id'), 'base_success': r.get('base_success'),
           'manifest': _rel(mp, root), 'manifest_sha256': msha,
           'progress': _rel(adir / 'progress.jsonl', root), 'progress_sha256': psha,
           'success': None, 'audit_ok': False, 'success_d': None, 'first_composed_t': None, 'first_fired_t': None,
           'progress_first_fire_t': None, 'composed': None, 'fired': None, 'http_sends': None, 'requests': None,
           'first_composed_request': None, 'first_composed_delta_max_normalized': None, 'first_composed_delta_zero': None,
           'first_composed_mil_t': None, 'inclusive_count': None, 'exclusive_after_composed_chunk': None,
           'timing_status': 'missing', 'attempt': None, 'meta': None, 'meta_sha256': None, 'trace': None,
           'trace_sha256': None, 'trace_http_sends': None, 'decisions': None, 'decisions_sha256': None,
           'video': None, 'video_bytes': None, 'video_mtime': None, 'errors': list(errs)}
    if it is None:
        if not errs:
            rec['errors'].append('no progress line for this manifest row (unfinished / missing)')
        return rec
    rec.update({'attempt': it.get('attempt'), 'progress_first_fire_t': it.get('first_fire_t'), 'composed': it.get('composed'),
                'fired': it.get('fired'), 'http_sends': it.get('http_sends'), 'requests': it.get('requests')})
    try:
        rec['success'] = int(it['success'])
        rec['audit_ok'] = not errs
        meta_p = adir / it['meta']
        rec['meta'] = _rel(meta_p, root)
        rec['meta_sha256'] = fc.hash(meta_p)
        vid = meta_p.parent / 'video.mp4'
        rec['video'] = _rel(vid, root)
        if vid.is_file():
            stt = vid.stat()
            rec['video_bytes'], rec['video_mtime'] = stt.st_size, stt.st_mtime
        else:
            rec['errors'].append(f'video missing: {rec["video"]}')    # non-blocking for timing, listed
        tr = adir / it['trace']
        rec['trace'] = _rel(tr, root)
        dl = adir / it['decisions_log']
        rec['decisions'] = _rel(dl, root)
    except (KeyError, OSError, ValueError, TypeError) as e:
        rec['errors'].append(f'paths: {type(e).__name__}: {e}')
        rec['timing_status'] = 'invalid'
        return rec
    if errs:                              # arm load / decision audit failed: no timing from unaudited data
        rec['timing_status'] = 'invalid'
        return rec
    try:
        meta = json.loads(meta_p.read_text())
        for mk, want in (('plan_id', r['plan_id']), ('scene_idx', m['scene']), ('jitter_idx', r['j']), ('noise_idx', r['n']),
                         ('scenario_seed', r['env_seed']), ('inference_seed', r['inference_seed'])):
            need(mk in meta, f'meta.json lacks {mk} ({rec["meta"]})')
            need(str(meta[mk]) == str(want), f'meta {mk} {meta[mk]!r} != manifest {want!r}')
        try:
            d = S.eval_d(meta)
        except S.TimingError as e:
            raise AnalysisError(f'meta timing schema ({rec["meta"]}): {e}') from None
        need(int(d is not None) == rec['success'], f'meta success {d is not None} != progress success {rec["success"]}')
        lo, hi = it['requests']
        need(meta.get('steps') == hi - lo + 1, f'meta.steps {meta.get("steps")} != progress request count {hi - lo + 1}')
        need(it['http_sends'] == hi - lo + 1, f'progress.http_sends {it["http_sends"]} != request count {hi - lo + 1}')
        rec['trace_sha256'] = fc.hash(tr)
        sends = sum(1 for x in tr.read_text().splitlines() if x.strip() and json.loads(x).get('event') == 'http_send')
        rec['trace_http_sends'] = sends
        need(sends == it['http_sends'], f'trace http_send {sends} != progress.http_sends {it["http_sends"]}')
        rec['decisions_sha256'] = fc.hash(dl)
        recs, dup = fc.decisions(dl)
        tc, tf, ncomp, dc = first_events(episode_decisions(recs, dup, lo, hi), it['composed'])
        rec.update({'success_d': d, 'first_composed_t': tc, 'first_fired_t': tf})
        if dc is not None:
            delta = dc.get('qam_delta_max_normalized')
            rec.update({'first_composed_request': dc['request'], 'first_composed_delta_max_normalized': delta,
                        'first_composed_delta_zero': delta == 0, 'first_composed_mil_t': (dc.get('mil') or {}).get('t')})
        rec['timing_status'], rec['inclusive_count'], rec['exclusive_after_composed_chunk'] = timing(rec['success'], d, tc)
    except (AnalysisError, KeyError, OSError, ValueError, TypeError) as e:
        rec['errors'].append(f'timing: {type(e).__name__}: {e}')
        rec['timing_status'] = 'invalid'
        for k in ('success_d', 'first_composed_t', 'inclusive_count', 'exclusive_after_composed_chunk'):
            rec[k] = None
    return rec


# ------------------------------------------------------------------ merge
def merge(cols, expected_rows, expected_selected=None, reference=None, summ=None):
    if summ is None:
        summ = _rt()[4].summ
    for c in cols:
        need(c.get('kind') == KIND, f'input kind {c.get("kind")!r} != {KIND}')
    roots = [(c['host'], c['root']) for c in cols]
    need(len(set(roots)) == len(roots), f'duplicate collection (host, root) {roots}')
    rows, seen, issues = [], {}, []
    for c in cols:
        issues += [f'{c["host"]}: fold error {e}' for e in c['fold_errors']]
        issues += [f'{c["host"]}: unexpected arm {e["fold_id"]}/{e["arm"]} excluded' for e in c['unexpected_arms']]
        for r in c['rows']:
            k = (r['fold_id'], r['policy'], r['n'])
            need(k not in seen, f'duplicate row fold={k[0]} policy={k[1]} n={k[2]} (hosts {seen.get(k)}, {r["host"]})')
            seen[k] = r['host']
            rows.append(r)
    if len(rows) != expected_rows:
        issues.append(f'rows {len(rows)} != expected {expected_rows}')
    by_coord = defaultdict(dict)
    for r in rows:
        by_coord[(r['fold_id'], r['n'])][r['policy']] = r
    folds = sorted({f for f, _ in by_coord})
    if len(folds) != EXPECTED_FOLDS:
        issues.append(f'folds {len(folds)} != {EXPECTED_FOLDS}')
    for f in folds:
        for p in POLICIES:
            k = sum(1 for (ff, _), v in by_coord.items() if ff == f and p in v)
            if k != ROWS_PER_FOLD:
                issues.append(f'{f}/{p}: {k} rows != {ROWS_PER_FOLD}')
    if len(by_coord) != EXPECTED_FOLDS * ROWS_PER_FOLD:
        issues.append(f'coordinates {len(by_coord)} != {EXPECTED_FOLDS * ROWS_PER_FOLD}')
    for (f, n), v in sorted(by_coord.items()):
        if set(v) != set(POLICIES):
            issues.append(f'{f}/n{n}: policies {sorted(v)} != {sorted(POLICIES)}')
        ids = {p: tuple(str(x[k]) for k in IDENTITY) for p, x in v.items()}
        if len(set(ids.values())) > 1:
            issues.append(f'{f}/n{n}: row identity differs across policies {ids}')
    row_errors = [{'fold_id': r['fold_id'], 'policy': r['policy'], 'n': r['n'], 'host': r['host'], 'error': e}
                  for r in rows for e in r['errors']]
    sel, unresolved = [], []
    for (f, n), v in sorted(by_coord.items()):
        q = v.get(QAM)
        for p in TRQAM:
            t = v.get(p)
            if q is None or t is None or not q['audit_ok'] or not t['audit_ok'] or q['success'] is None or t['success'] is None:
                unresolved.append({'fold_id': f, 'n': n, 'policy': p, 'reason': 'QAM or TRQAM row missing / failed audit'})
                continue
            if q['success'] == 0 and t['success'] == 1:
                sel.append(_case(q, t))
    counts = {p: sum(c['policy'] == p for c in sel) for p in TRQAM}
    if expected_selected:
        for p, k in expected_selected.items():
            if counts.get(p) != k:
                issues.append(f'selected {p}: {counts.get(p)} != expected {k}')
    ref = None
    if reference is not None:
        ref = reference_check(reference, sel)
        issues += ref['issues']
    summary = {'policy': _groups(sel, lambda c: (c['policy'],), summ),
               'policy_archived_base': _groups(sel, lambda c: (c['policy'], 'archived_base_fail' if c['archived_base_fail']
                                                               else 'archived_base_success'), summ),
               'task': _groups(sel, lambda c: (c['policy'], c['instruction']), summ),
               'fold': _groups(sel, lambda c: (c['policy'], c['fold_id']), summ)}
    for p in TRQAM:                     # empty groups stay visible
        summary['policy'].setdefault((p,), _gstats([], summ))
    reps = {p: representatives([c for c in sel if c['policy'] == p]) for p in TRQAM}
    timing_issue = sum(c['timing_status'] not in ('valid', 'no_composition') for c in sel)
    if timing_issue:
        issues.append(f'{timing_issue} selected cases with invalid/missing timing (listed, excluded from statistics)')
    status = 'PASS' if not issues and not row_errors else 'ISSUES'
    return {'kind': 'fold27_paired_recovery_report_v1', 'status': status, 'convention': CONVENTION,
            'rows_total': len(rows), 'coordinates': len(by_coord), 'folds': len(folds), 'selected_counts': counts,
            'expected_selected': expected_selected, 'reference': ref, 'issues': issues, 'row_errors': row_errors,
            'selection_unresolved': unresolved, 'summary': summary, 'representatives': reps,
            'inputs': [{'host': c['host'], 'root': c['root'], 'rows': len(c['rows']), 'skipped_smoke': c['skipped_smoke']}
                       for c in cols], 'rows': rows, 'selected': sel}


CASE_KEYS = ('host', 'root', 'fold_id', 'instruction', 'scene', 'j', 'n', 'env_seed', 'inference_seed', 'plan_id', 'policy',
             'base_success', 'success', 'success_d', 'first_composed_t', 'first_fired_t', 'progress_first_fire_t',
             'first_composed_request', 'first_composed_delta_max_normalized', 'first_composed_delta_zero', 'first_composed_mil_t',
             'inclusive_count', 'exclusive_after_composed_chunk', 'timing_status', 'composed', 'fired', 'http_sends', 'requests',
             'attempt', 'manifest', 'manifest_sha256', 'progress', 'progress_sha256', 'meta', 'meta_sha256', 'trace',
             'trace_sha256', 'decisions', 'decisions_sha256', 'video', 'video_bytes', 'errors')


def _case(q, t):
    c = {k: t.get(k) for k in CASE_KEYS}
    c['archived_base_fail'] = not t['base_success']
    c.update({'qam_success': q['success'], 'qam_meta': q.get('meta'), 'qam_video': q.get('video'), 'qam_trace': q.get('trace'),
              'qam_decisions': q.get('decisions'), 'qam_first_composed_t': q.get('first_composed_t'),
              'qam_timing_status': q.get('timing_status')})
    return c


def _gstats(cs, summ):
    v = [c for c in cs if c['timing_status'] == 'valid']
    return {'selected_N': len(cs), 'valid_timing_N': len(v),
            'no_composition_N': sum(c['timing_status'] == 'no_composition' for c in cs),
            'invalid_or_missing_N': sum(c['timing_status'] not in ('valid', 'no_composition') for c in cs),
            'archived_base_fail_N': sum(c['archived_base_fail'] for c in cs),
            'inclusive_count': summ([c['inclusive_count'] for c in v]),
            'exclusive_after_composed_chunk': summ([c['exclusive_after_composed_chunk'] for c in v]),
            'success_d': summ([c['success_d'] for c in v]), 'first_composed_t': summ([c['first_composed_t'] for c in v])}


def _groups(cs, keyf, summ):
    g = defaultdict(list)
    for c in cs:
        g[keyf(c)].append(c)
    return {k: _gstats(v, summ) for k, v in sorted(g.items())}


def representatives(cs):
    v = sorted((c for c in cs if c['timing_status'] == 'valid'), key=lambda c: (c['inclusive_count'], c['fold_id'], c['n']))
    if not v:
        return {}
    pick = lambda c: {k: c[k] for k in ('fold_id', 'n', 'host', 'root', 'inclusive_count', 'exclusive_after_composed_chunk',
                                        'success_d', 'first_composed_t', 'archived_base_fail', 'meta', 'video', 'trace',
                                        'decisions', 'qam_meta', 'qam_video')}  # noqa: E731
    return {'min': pick(v[0]), 'median': pick(v[(len(v) - 1) // 2]), 'max': pick(v[-1])}


def reference_check(ref, sel):
    """Independent recount from the canonical aggregate report rows (audit_ok only)."""
    issues, by = [], defaultdict(dict)
    for r in ref.get('rows', []):
        if r.get('audit_ok'):
            by[(r['fold_id'], r['n'])][r['policy']] = int(r['success'])
    exp = {p: {k for k, v in by.items() if v.get(QAM) == 0 and v.get(p) == 1} for p in TRQAM}
    got = {p: {(c['fold_id'], c['n']) for c in sel if c['policy'] == p} for p in TRQAM}
    for p in TRQAM:
        if exp[p] != got[p]:
            issues.append(f'reference mismatch {p}: reference {len(exp[p])}, here {len(got[p])}, '
                          f'only-reference {sorted(exp[p] - got[p])[:5]}, only-here {sorted(got[p] - exp[p])[:5]}')
    return {'counts': {p: len(exp[p]) for p in TRQAM}, 'issues': issues}


# ------------------------------------------------------------------ outputs
def _k(k):
    return ' / '.join(map(str, k))


def _cell(v):
    return json.dumps(v) if isinstance(v, (list, dict)) else ('' if v is None else v)


def _csv(path, rows, cols):
    with open(path, 'x', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([[_cell(r.get(c)) for c in cols] for r in rows])


def _flat(key, st):
    o = {'group': _k(key), **{k: st[k] for k in ('selected_N', 'valid_timing_N', 'no_composition_N', 'invalid_or_missing_N',
                                                 'archived_base_fail_N')}}
    for m in ('inclusive_count', 'exclusive_after_composed_chunk', 'success_d', 'first_composed_t'):
        for s in ('N', 'mean', 'median', 'q1', 'q3', 'min', 'max'):
            o[f'{m}_{s}'] = st[m].get(s)
    return o


def write(rep, out_dir, sources):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {**rep, 'sources': sources}
    summ_j = {g: {_k(k): v for k, v in d.items()} for g, d in rep['summary'].items()}

    def put(name, text):
        with open(out / name, 'x') as f:
            f.write(text)
    put('rows.json', json.dumps(rep['rows'], indent=1, default=str))
    _csv(out / 'rows.csv', rep['rows'], list(rep['rows'][0]) if rep['rows'] else ['fold_id'])
    put('selected_cases.json', json.dumps(rep['selected'], indent=1, default=str))
    _csv(out / 'selected_cases.csv', rep['selected'], list(CASE_KEYS) + ['archived_base_fail', 'qam_success', 'qam_meta',
                                                                          'qam_video', 'qam_trace', 'qam_decisions',
                                                                          'qam_first_composed_t', 'qam_timing_status'])
    put('summary.json', json.dumps(summ_j, indent=1))
    cols = list(_flat(('x',), _gstats([], lambda xs: {'N': 0})).keys())
    for g, name in (('policy', 'summary_policy.csv'), ('task', 'summary_task.csv'), ('fold', 'summary_fold.csv')):
        _csv(out / name, [_flat(k, v) for k, v in rep['summary'][g].items()], cols)
    slim = {k: v for k, v in rep.items() if k not in ('rows', 'selected', 'summary')}
    put('report.json', json.dumps({**slim, 'summary': summ_j}, indent=1, default=str))
    put('report.md', markdown(rep))


def _f(x):
    return '' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))


def markdown(rep):
    md = ['# fold27 paired recovery: common-QAM failure -> TRQAM success (same coordinate)', '',
          f'**Status: {rep["status"]}** — rows {rep["rows_total"]}, coordinates {rep["coordinates"]}, folds {rep["folds"]}.', '',
          f'Convention: {CONVENTION}', '',
          'Quantiles: statistics.quantiles(method="inclusive") (linear interpolation, numpy "linear"). Descriptive only: '
          'pre-intervention trajectory equivalence between the QAM and TRQAM runs is NOT verified; archived baseline = '
          'original collection label, not a re-run.', '',
          f'Selected counts: {rep["selected_counts"]} (expected {rep["expected_selected"]}; reference '
          f'{(rep["reference"] or {}).get("counts")}).', '',
          '| group | selected N | valid timing N | no composition | invalid/missing | archived base fail | inclusive mean / median [Q1, Q3] (min-max) | exclusive median | success_d median | first_composed_t median |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for g in ('policy', 'policy_archived_base'):
        for k, st in rep['summary'][g].items():
            i = st['inclusive_count']
            md.append(f'| {_k(k)} | {st["selected_N"]} | {st["valid_timing_N"]} | {st["no_composition_N"]} | '
                      f'{st["invalid_or_missing_N"]} | {st["archived_base_fail_N"]} | {_f(i.get("mean"))} / {_f(i.get("median"))} '
                      f'[{_f(i.get("q1"))}, {_f(i.get("q3"))}] ({_f(i.get("min"))}-{_f(i.get("max"))}) | '
                      f'{_f(st["exclusive_after_composed_chunk"].get("median"))} | {_f(st["success_d"].get("median"))} | '
                      f'{_f(st["first_composed_t"].get("median"))} |')
    md += ['', '## Representative cases (valid timing; sorted by inclusive_count, lower median)', '']
    for p, r in rep['representatives'].items():
        if not r:
            md.append(f'- {p}: no valid-timing case'); continue
        for w in ('min', 'median', 'max'):
            c = r[w]
            md.append(f'- {p} {w}: {c["fold_id"]} n{c["n"]} inclusive {c["inclusive_count"]} (t={c["first_composed_t"]}, '
                      f'd={c["success_d"]}, archived_base_fail={c["archived_base_fail"]}) [{c["host"]}:{c["root"]}] '
                      f'video `{c["video"]}` meta `{c["meta"]}` trace `{c["trace"]}` decisions `{c["decisions"]}`')
    md += ['', f'## Issues ({len(rep["issues"])}) / row errors ({len(rep["row_errors"])}) / unresolved selection '
               f'({len(rep["selection_unresolved"])})', '']
    md += [f'- {x}' for x in rep['issues'][:200]]
    md += [f'- {e["host"]} {e["fold_id"]}/{e["policy"]}/n{e["n"]}: {e["error"]}' for e in rep['row_errors'][:200]]
    if len(rep['row_errors']) > 200:
        md.append(f'- ... {len(rep["row_errors"]) - 200} more in report.json')
    return '\n'.join(md) + '\n'


# ------------------------------------------------------------------ CLI
def _kv(s):
    k, v = s.split('=')
    return k, int(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--root', type=Path, required=True); c.add_argument('--host', required=True)
    c.add_argument('--out', type=Path, required=True)
    m = sub.add_parser('merge')
    m.add_argument('--input', type=Path, nargs='+', required=True); m.add_argument('--expected-rows', type=int, required=True)
    m.add_argument('--out-dir', type=Path, required=True); m.add_argument('--reference', type=Path)
    m.add_argument('--expected-selected', type=_kv, nargs='+', help='policy=count (default basic 65, pbrs 64, mix 82)')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'collect':
            need(not a.out.exists(), f'{a.out} exists (new files only)')
            need(a.root.is_dir(), f'{a.root} is not a directory')
            rt = _rt()
            r = collect(a.root, a.host, rt)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, 'x') as f:
                f.write(json.dumps(r, indent=1, default=str))
            nerr = sum(bool(x['errors']) for x in r['rows'])
            print(json.dumps({'out': str(a.out), 'sha256': rt[0].sha256_path(a.out), 'rows': len(r['rows']),
                              'rows_with_errors': nerr, 'fold_errors': len(r['fold_errors'])}))
            return 1 if nerr or r['fold_errors'] else 0
        need(not a.out_dir.exists(), f'{a.out_dir} exists (new dir only)')
        rt = _rt()
        cols = [json.loads(p.read_text()) for p in a.input]
        ref = json.loads(a.reference.read_text()) if a.reference else None
        exp = dict(a.expected_selected) if a.expected_selected else dict(EXPECTED_SELECTED)
        rep = merge(cols, a.expected_rows, exp, ref, rt[4].summ)
        srcs = {'input': [[str(p), rt[0].sha256_path(p)] for p in a.input],
                'reference': [str(a.reference), rt[0].sha256_path(a.reference)] if a.reference else None}
        write(rep, a.out_dir, srcs)
        print(json.dumps({'out_dir': str(a.out_dir), 'status': rep['status'], 'selected': rep['selected_counts'],
                          'issues': len(rep['issues']), 'row_errors': len(rep['row_errors'])}))
        return 0 if rep['status'] == 'PASS' else 1
    except (AnalysisError, OSError, ValueError) as ex:
        print(f'[paired-recovery] REFUSED: {ex}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
