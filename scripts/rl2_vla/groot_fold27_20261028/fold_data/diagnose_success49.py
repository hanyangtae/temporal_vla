"""READ-ONLY diagnosis of the env-step success series of a scene's source rollouts (run where rollout.pkl lives; python
with torch for unpickling). Writes nothing except an optional NEW report file (--out, exclusive create). Never touches
the store, the derived exact-49 dir, the original 16-token extraction dir, or any frozen source.

Why: extract_scene49 calls the ORIGINAL extract_scene.convert unchanged; its success_lag (extract_scene 90e69e53 =
commit 01a528a3) requires the probe success to be continuous from the first success to the end and the stop lag to be
0 or one chunk. The historical 16-token extraction of PPCC candle/jug/marshmallow (remaining_b_v3, extractor
0c67c8cf = commit 60b65fe8) only required: success -> steps == end of the chunk containing the first probe success
(or the horizon); failure -> no probe success and steps == max_episode_steps. This tool evaluates BOTH rules on the
same bytes and dumps the raw evidence, so 'invalid data' can be separated from 'invalid validation assumption'.

Per episode: episode id, stem, pkl path, sha256(pkl) vs spec, spec success, payload episode_success, steps,
max_episode_steps, n_action_steps, n records, first probe success, probe-true runs [[start, end_inclusive], ...],
final probe value, window of the probe series around the first success / end, rule_60b65fe8 verdict, rule_current
verdict (the original success_lag message verbatim), continuity, lag, env_step_phases tail, every payload key whose
name mentions success/reward/done/term/trunc/stop (scalars or short summaries; hidden states / actions skipped), and
optionally (--compare-16 DIR) whether the original 16-token record stores the identical env_step_success series,
env_steps and first_success_step (read-only).

    <py with torch> diagnose_success49.py --spec <fold_spec.json> --store <store> --scene 'PPCC/candle|s0' \
        --missing-from <derived exact-49 scene dir> [--compare-16 <remaining_b_v3 dir>] [--out <new report.json>]
    (--missing-from reads that dir's PROGRESS.json 'missing' stems = the episodes not yet committed, in extraction
     order; the first one is the episode the extractor stopped on.)  Or --episode <episode_id> ... / --all.
"""
import argparse, io, json, pickle, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_scene as ES        # noqa: E402  (frozen; read-only use)
import extract_scene49 as E49     # noqa: E402  (frozen; scene_sources / source_path / stem_of only)

KEY_HINTS = ('succ', 'reward', 'done', 'term', 'trunc', 'stop')
SKIP_KEYS = ('hidden_states', 'actions')


def rule_60b65fe8(es, steps, k, max_steps, success):
    """Verbatim logic of extract_scene.episode_steps at commit 60b65fe8 (sha 0c67c8cf), the remaining_b_v3 extractor."""
    first = es.index(True) if any(es) else -1
    if success:
        want = min(-(-(first + 1) // k) * k, int(max_steps)) if first >= 0 else None
        if first < 0 or steps != want:
            return f'FAIL success episode: first success step {first}, steps {steps} != chunk-boundary end {want}'
        return 'OK'
    if first != -1 or steps != int(max_steps):
        return f'FAIL failure episode: first success {first}, steps {steps} vs max_episode_steps {max_steps}'
    return 'OK'


def rule_current(es, steps, k, max_steps, success):
    first = es.index(True) if any(es) else -1
    try:
        return f'OK lag={ES.success_lag(es, first, steps, k, int(max_steps), bool(success))}'
    except ES.ExtractError as e:
        return f'FAIL {e}'


def runs(es):
    out, start = [], None
    for i, v in enumerate(es + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i - 1]); start = None
    return out


def summarize(v):
    if isinstance(v, (bool, int, float, str, type(None), np.bool_, np.integer, np.floating)):
        return v.item() if hasattr(v, 'item') else v
    if isinstance(v, dict):
        return {str(k): summarize(x) for k, x in list(v.items())[:20]}
    try:
        a = np.asarray(v)
        if a.dtype != object and a.size <= 2000:
            flat = a.reshape(-1).tolist()
            return {'len': len(flat), 'head': flat[:10], 'tail': flat[-12:],
                    'nonzero_idx': [i for i, x in enumerate(flat) if x][:50] if a.dtype.kind in 'biuf' else None}
        return {'type': type(v).__name__, 'shape': list(a.shape)}
    except Exception:   # noqa: BLE001
        return {'type': type(v).__name__}


def diagnose_payload(payload, window=8):
    raw = payload.get('env_step_success')
    es = [bool(v) for v in raw] if raw is not None else []
    steps = len(es)
    k = int(payload.get('env_step_n_action_steps', payload.get('n_action_steps', -1)))
    mx = payload.get('max_episode_steps')
    succ = bool(payload.get('episode_success'))
    first = es.index(True) if any(es) else -1
    lo = max(0, (first if first >= 0 else steps) - window)
    ph = payload.get('env_step_phases')
    d = {'episode_success': succ, 'steps': steps, 'max_episode_steps': mx, 'n_action_steps': k,
         'n_records': len(payload['hidden_states']) if payload.get('hidden_states') is not None else None, 'first_probe_success': first, 'probe_true_runs': runs(es),
         'final_probe': es[-1] if es else None, 'probe_window': {'from': lo, 'bits': ''.join('1' if x else '0' for x in es[lo:steps])},
         'continuous_first_to_end': bool(first >= 0 and all(es[first:steps])),
         'first_success_chunk_end': (min(-(-(first + 1) // k) * k, int(mx)) if first >= 0 and mx is not None and k > 0 else None),
         'env_step_phases_tail': list(ph[-(window + 2):]) if ph is not None else None,
         'env_step_event_steps': summarize(payload.get('env_step_event_steps')),
         'payload_keys': sorted(map(str, payload)),
         'related_fields': {str(kk): summarize(v) for kk, v in payload.items()
                            if kk not in SKIP_KEYS and any(h in str(kk).lower() for h in KEY_HINTS)}}
    if mx is not None and k > 0:
        d['rule_60b65fe8'] = rule_60b65fe8(es, steps, k, mx, succ)
        d['rule_current_90e69e53'] = rule_current(es, steps, k, mx, succ)
    return d, es


def compare16(orig_dir, stem, es, steps, first):
    oj = Path(orig_dir) / f'{stem}.json'
    if not oj.exists():
        return {'present': False}
    o = json.loads(oj.read_text())
    with np.load(Path(orig_dir) / o['npz']) as z:
        oes = [bool(v) for v in z['env_step_success']] if 'env_step_success' in z.files else None
        return {'present': True, 'pkl_sha256': o.get('pkl_sha256'), 'env_steps': int(z['env_steps']) if 'env_steps' in z.files else None,
                'first_success_step': int(z['first_success_step']) if 'first_success_step' in z.files else None,
                'env_step_success_identical': oes == es, 'env_steps_equal': ('env_steps' in z.files and int(z['env_steps']) == steps),
                'first_equal': ('first_success_step' in z.files and int(z['first_success_step']) == first),
                'npz_keys': sorted(z.files)}


def select(spec, scene, episodes=None, missing_from=None, all_=False):
    src = E49.scene_sources(spec, scene)
    by_stem = {E49.stem_of(ES.episode_id(*k)): (k, r) for k, r in src.items()}
    if missing_from:
        prog = json.loads((Path(missing_from) / 'PROGRESS.json').read_text())
        order = [s for s in sorted(by_stem) if s in set(prog['missing'])]       # extract_scene49 iterates sorted stems
        return [(s, *by_stem[s]) for s in order], {'progress_done': prog.get('done'), 'progress_missing': len(prog['missing'])}
    if episodes:
        want = set(episodes)
        sel = [(s, k, r) for s, (k, r) in sorted(by_stem.items()) if ES.episode_id(*k) in want]
        if len(sel) != len(want):
            raise SystemExit(f'episodes not in scene {scene}: {sorted(want - {ES.episode_id(*k) for _, k, _ in sel})}')
        return sel, {}
    if all_:
        return [(s, *by_stem[s]) for s in sorted(by_stem)], {}
    raise SystemExit('choose --missing-from, --episode or --all')


def run(spec_path, store, scene, episodes=None, missing_from=None, all_=False, compare_16=None, window=8, load=None):
    spec = json.loads(Path(spec_path).read_text())
    sel, meta = select(spec, scene, episodes, missing_from, all_)
    load = load or (lambda p: Path(p).read_bytes())
    rows = []
    for s, k, r in sel:
        p = E49.source_path(store, k)
        row = {'episode_id': ES.episode_id(*k), 'stem': s, 'pkl': str(p), 'spec_success': int(r['success']),
               'spec_pkl_sha256': r['pkl_sha256']}
        try:
            blob = load(p)
        except OSError as e:
            row['error'] = f'read: {e}'; rows.append(row); continue
        row['pkl_sha256'] = ES.sha256_bytes(blob)
        row['sha_ok'] = row['pkl_sha256'] == r['pkl_sha256']
        payload = pickle.load(io.BytesIO(blob)); del blob
        d, es = diagnose_payload(payload, window)
        del payload
        row.update(d)
        if compare_16:
            row['compare_16'] = compare16(compare_16, s, es, d['steps'], d['first_probe_success'])
        rows.append(row)
    summ = {'episodes': len(rows), 'scene': scene, **meta,
            'sha_mismatch': [x['episode_id'] for x in rows if x.get('sha_ok') is False],
            'success_mismatch_vs_spec': [x['episode_id'] for x in rows if 'episode_success' in x and int(x['episode_success']) != x['spec_success']],
            'fail_rule_60b65fe8': [x['episode_id'] for x in rows if str(x.get('rule_60b65fe8', '')).startswith('FAIL')],
            'fail_rule_current': [x['episode_id'] for x in rows if str(x.get('rule_current_90e69e53', '')).startswith('FAIL')],
            'non_continuous_success': [x['episode_id'] for x in rows if x.get('episode_success') and not x.get('continuous_first_to_end')],
            'final_probe_false_on_success': [x['episode_id'] for x in rows if x.get('episode_success') and x.get('final_probe') is False],
            'first_in_extract_order': rows[0]['episode_id'] if rows and missing_from else None}
    return {'summary': summ, 'episodes': rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--scene', required=True); ap.add_argument('--episode', action='append')
    ap.add_argument('--missing-from', type=Path); ap.add_argument('--all', action='store_true')
    ap.add_argument('--compare-16', type=Path); ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--out', type=Path, help='NEW report file (exclusive create)')
    a = ap.parse_args(argv)
    r = run(a.spec, a.store, a.scene, a.episode, a.missing_from, a.all, a.compare_16, a.window)
    txt = json.dumps(r, indent=1, default=str)
    if a.out:
        with open(a.out, 'x') as f:
            f.write(txt)
    print(json.dumps(r['summary'], indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
