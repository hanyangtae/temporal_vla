"""Explicit 49-token re-extraction, VERSION 2 = collector-semantics success contract. NEW version, NEW output root only.
extract_scene.py (original, 90e69e53) and extract_scene49.py (v1, b6350355) are imported read-only and never modified;
v1 output dirs are refused as --out (path guard + different contract) and never read: no adoption, every record of
the scene is recomputed from its source pkl into the new root.

Why v2 (evidence: OPUS_CANDLE_CONTRACT_20261103.md + archive diagnostic fold27-candle-diag-missing-20261103.json):
  v1 reuses extract_scene.success_lag (commit 01a528a3), which added an UNSUPPORTED assumption: the per-raw-step probe
  success (env_step_success) must stay True from the first success to the end. The collector's actual outcome contract
  (http_feature_collect.py:1480-1486 step_success(reward, info) + Isaac-GR00T MultiStepWrapper terminate_on_success:
  any raw-step success inside a 5-step chunk -> episode ends at that chunk's end, episode_success sticky True; reward
  aggregated by max over the episode history) does not require persistence. Real case PPCC/candle s0 j0 n4
  (pkl 8c572d98…): probe True at 452, 453, False at 454, steps 455 (= end of the first-success chunk), max 720, 91
  records; historical 16-token extractor (60b65fe8, no continuity rule) accepted it with an identical series.
v2 success rule ('collector_v2'):
  success: first probe success exists; steps on a 5-step chunk boundary (or == max_episode_steps); steps >= end of the
           first-success chunk; lag = steps - that end in {0, 5} (unchanged from 01a528a3, OvenRack one-chunk lag).
           Probe persistence is NOT required; it is RECORDED (probe_true_runs, final_probe, transient_success).
  failure: unchanged (no probe success at all and steps == max_episode_steps).
Labels are never changed: success comes from the pkl's episode_success and must equal the fold spec (as v1).
Every array is still produced by the ORIGINAL extract_scene.convert code; only during that call, inside this process,
extract_scene.success_lag is bound to success_lag_v2 (restored afterwards, also on error). The file is not edited.
Raw termination evidence: the collector pkl stores no per-step terminated/truncated/done/reward; each record lists the
termination-related payload keys that DO exist and names the standard ones that are MISSING (nothing is inferred).
Everything else (strict pkl sha, spec success, 49-token block, dtype, compare-16 bitwise, resume/no-overwrite,
PROGRESS/COMPLETE, space guard) is v1's contract via v1's own helpers.

    <py with torch> extract_scene49_v2.py --spec SPEC --store STORE --scene 'PPCC/candle|s0' \
        --out <derived>/fold27_extract49_v2_20261103/srv50_candle_s0 --expect-records 200 --compare-16 <remaining_b_v3>
"""
import argparse, contextlib, io, json, os, pickle, shutil, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_scene as ES       # noqa: E402  (frozen original; read-only use)
import extract_scene49 as E49    # noqa: E402  (frozen v1; helpers only)

KIND = 'fold27_extract49_v2'
SUCCESS_RULE = 'collector_v2'
TERMINATION_STANDARD_KEYS = ('terminated', 'truncated', 'done', 'dones', 'reward', 'rewards', 'info_success', 'step_success')
TERMINATION_HINTS = ('term', 'trunc', 'done', 'reward', 'stop')


META_BYTES = 64 * 1024 ** 2         # json/PROGRESS/COMPLETE/contract allowance
DEFAULT_RESERVE_GIB = 3.0


class Extract49V2Error(RuntimeError):
    pass


def space_need(remaining, reserve_gib):
    """Bytes that must be free before writing the next record: raw upper bound of ALL remaining records + one extra
    record of temp (tmp npz coexists with the final npz until os.replace) + metadata allowance + reserve.
    200 records, 3 GiB: 4,446,761,600 + 22,233,808 + 67,108,864 + 3,221,225,472 = 7,757,329,744."""
    if remaining <= 0:
        return 0
    return E49.raw_upper_bound_bytes(remaining) + E49.raw_upper_bound_bytes(1) + META_BYTES + int(reserve_gib * 1024 ** 3)


def success_lag_v2(es, first, steps, k, max_steps, success):
    """Same signature/return as extract_scene.success_lag; continuity requirement removed (see module doc)."""
    if not success:
        if first != -1 or steps != max_steps:
            raise ES.ExtractError(f'failure episode: first success {first}, steps {steps} vs max_episode_steps {max_steps}')
        return -1
    if first < 0:
        raise ES.ExtractError('success episode without any probe success step')
    first_end = min(-(-(first + 1) // k) * k, max_steps)
    if steps < first_end:
        raise ES.ExtractError(f'success episode ended at {steps} before its first-success chunk end {first_end}')
    if steps % k and steps != max_steps:
        raise ES.ExtractError(f'success episode steps {steps} not on a {k}-step chunk boundary')
    lag = steps - first_end
    if lag not in (0, k):
        raise ES.ExtractError(f'success stop lag {lag} steps not in (0, {k}); outside observed collector behaviour')
    return lag


@contextlib.contextmanager
def collector_rule():
    orig = ES.success_lag
    ES.success_lag = success_lag_v2
    try:
        yield
    finally:
        ES.success_lag = orig


def probe_runs(es):
    out, start = [], None
    for i, v in enumerate(list(es) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i - 1]); start = None
    return out


def success_diagnostics(arrays):
    es = [bool(v) for v in arrays['env_step_success']]
    first, steps, succ = int(arrays['first_success_step']), int(arrays['env_steps']), bool(arrays['success'])
    cont = bool(first >= 0 and all(es[first:steps]))
    return {'success_rule': SUCCESS_RULE, 'probe_true_runs': probe_runs(es), 'final_probe': es[-1] if es else None,
            'probe_continuous_first_to_end': cont, 'transient_success': bool(succ and not cont)}


def termination_evidence(payload):
    present = {}
    for k, v in payload.items():
        if k in ('hidden_states', 'actions') or not any(h in str(k).lower() for h in TERMINATION_HINTS):
            continue
        if isinstance(v, (bool, int, float, str, type(None), np.bool_, np.integer, np.floating)):
            present[str(k)] = v.item() if hasattr(v, 'item') else v
        else:
            try:
                present[str(k)] = {'len': len(v)}
            except TypeError:
                present[str(k)] = {'type': type(v).__name__}
    return {'present': present, 'missing_standard': [k for k in TERMINATION_STANDARD_KEYS if k not in payload],
            'note': 'collector pkl termination fields as stored; absent fields are NOT inferred'}


def convert49_v2(payload):
    with collector_rule():
        arrays = ES.convert(payload, with_last_block=False)        # original code; only success_lag bound to v2
    lb, src_dtype = E49.last_block49(payload)
    if len(lb) != int(arrays['n_inferences']):
        raise Extract49V2Error('last block / record count mismatch')
    arrays['last_block_d0'] = lb
    return arrays, src_dtype, success_diagnostics(arrays), termination_evidence(payload)


def contract_of(spec_bytes, scene, expect, compare_16):
    return {'kind': KIND, 'scene': scene, 'expect_records': expect, 'spec_sha256': ES.sha256_bytes(spec_bytes),
            'extractor49_sha256': ES.sha256_bytes(Path(__file__).read_bytes()),
            'extract49_v1_helpers_sha256': ES.sha256_bytes(Path(E49.__file__).read_bytes()),
            'base_extractor_sha256': ES.sha256_bytes(Path(ES.__file__).read_bytes()), 'layers': ES.LAYERS,
            'last_block_layer': 15, 'last_block_denoise': 0, 'last_block_tokens': list(E49.TOKENS),
            'last_block_dtype': 'float16', 'with_last_block': True, 'success_rule': SUCCESS_RULE,
            'success_rule_doc': 'success: first probe success, chunk-boundary end, lag in {0,5}; probe persistence not '
                                'required (recorded); failure: no probe success and steps == max. Labels unchanged.',
            'compare_16_contract_sha256': (ES.sha256_bytes((Path(compare_16) / 'contract.json').read_bytes()) if compare_16 else None)}


def run_scene(spec_path, store, out, scene, expect, compare_16=None, reserve_gib=DEFAULT_RESERVE_GIB, estimate=False,
              disk=shutil.disk_usage, convert=convert49_v2, log=print):
    out = Path(out)
    if 'fold27_extract49_v1' in str(out):
        raise Extract49V2Error(f'{out}: v2 never writes under the v1 root')
    spec_bytes = Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    if not spec.get('production_ready'):
        raise Extract49V2Error('fold spec not production_ready')
    sources = E49.scene_sources(spec, scene)
    if len(sources) != expect:
        raise Extract49V2Error(f'{scene}: {len(sources)} spec sources != --expect-records {expect}')
    contract = contract_of(spec_bytes, scene, expect, compare_16)
    stems = {E49.stem_of(ES.episode_id(*k)): (k, r) for k, r in sources.items()}
    done = [s for s in stems if (out / f'{s}.json').exists()]
    bound = E49.raw_upper_bound_bytes(len(stems) - len(done))
    def fs():
        p = out
        while not p.exists():                                       # new v2 root may not exist yet
            p = p.parent
        return disk(p).free
    free = fs()
    need = space_need(len(stems) - len(done), reserve_gib)
    plan = {'scene': scene, 'records': len(stems), 'already_done': len(done), 'raw_upper_bound_bytes': bound,
            'temp_bytes': E49.raw_upper_bound_bytes(1), 'meta_bytes': META_BYTES, 'reserve_bytes': int(reserve_gib * 1024 ** 3),
            'free_bytes': free, 'required_bytes': need, 'headroom_bytes': free - need, 'fits': free >= need, 'kind': KIND}
    if estimate:
        return plan
    if free < need:
        raise Extract49V2Error(f'insufficient space on output filesystem: free {free} < required {need} (plan {plan})')
    out.mkdir(parents=True, exist_ok=True)
    cp = out / 'contract.json'
    if cp.exists():
        old = json.loads(cp.read_text())
        if old != contract:
            raise Extract49V2Error(f'{cp} is a different contract (kind {old.get("kind")}); use a new --out')
    else:
        with open(cp, 'x') as fh:
            fh.write(json.dumps(contract, indent=1))
    csha = ES.sha256_bytes(cp.read_bytes())
    for npz in out.glob('*.npz'):
        if not npz.name.endswith('.tmp.npz') and not (out / f'{npz.stem}.json').exists():
            raise Extract49V2Error(f'orphan npz without record {npz.name}; inspect before resuming')
    E49.write_progress(out, csha, stems, done, sorted(p.name for p in out.glob('*.tmp.npz')))
    computed, skipped = 0, 0
    for s, (k, r) in sorted(stems.items()):
        eid = ES.episode_id(*k)
        jp, npz = out / f'{s}.json', out / f'{s}.npz'
        if jp.exists():
            rec = json.loads(jp.read_text())
            if rec['episode_id'] != eid or rec['pkl_sha256'] != r['pkl_sha256'] or ES.sha256_bytes(npz.read_bytes()) != rec['npz_sha256'] \
                    or rec.get('success_rule') != SUCCESS_RULE:
                raise Extract49V2Error(f'committed record mismatch {eid}')
            E49.check_npz(npz, rec)
            skipped += 1
            continue
        remaining = len(stems) - len(done)
        free, need = fs(), space_need(remaining, reserve_gib)          # re-measured before EVERY record (disk may drop)
        if free < need:
            raise Extract49V2Error(f'space guard before {eid}: free {free} < required {need} for {remaining} remaining '
                                   f'records; committed records kept, resume after freeing space')
        t0 = time.time()
        blob = E49.source_path(store, k).read_bytes()
        if ES.sha256_bytes(blob) != r['pkl_sha256']:
            raise Extract49V2Error(f'pkl sha mismatch {eid}')
        payload = pickle.load(io.BytesIO(blob))
        del blob
        arrays, src_dtype, diag, term = convert(payload)
        del payload
        if int(arrays['success']) != int(r['success']):
            raise Extract49V2Error(f'pkl success {int(arrays["success"])} != spec {r["success"]} {eid}')
        if compare_16:
            E49.compare16(compare_16, s, r, arrays)
        tmp = out / f'{s}.{os.getpid()}.tmp.npz'
        np.savez(tmp, **arrays)
        if npz.exists():
            raise Extract49V2Error(f'{npz.name} appeared concurrently; refusing to overwrite')
        os.replace(tmp, npz)
        rec = {'episode_id': eid, 'plan_id': k[0], 'machine': k[1], 'instruction': k[2], 'scene': k[3], 'jitter': k[4],
               'noise': k[5], 'success': int(arrays['success']), 'n_inferences': int(arrays['n_inferences']),
               'env_steps': int(arrays['env_steps']), 'max_episode_steps': int(arrays['max_episode_steps']),
               'first_success_step': int(arrays['first_success_step']),
               'success_chunk_lag_steps': int(arrays['success_chunk_lag_steps']),
               'pkl_sha256': r['pkl_sha256'], 'npz': npz.name, 'npz_sha256': ES.sha256_bytes(npz.read_bytes()),
               'last_block_tokens': list(E49.TOKENS), 'source_hidden_dtype': src_dtype, 'compare_16': bool(compare_16),
               **diag, 'raw_termination': term, 'seconds': time.time() - t0}
        with open(jp, 'x') as fh:
            fh.write(json.dumps(rec, indent=1))
        computed += 1
        done.append(s)
        E49.write_progress(out, csha, stems, done, sorted(p.name for p in out.glob('*.tmp.npz')))
        log(json.dumps({'done': len(done), 'total': len(stems), 'episode': eid, 'transient_success': diag['transient_success']}))
    recs = {}
    for s, (k, r) in stems.items():
        jp = out / f'{s}.json'
        if not jp.exists():
            raise Extract49V2Error(f'missing record {s}')
        rec = json.loads(jp.read_text())
        if rec['episode_id'] != ES.episode_id(*k) or ES.sha256_bytes((out / rec['npz']).read_bytes()) != rec['npz_sha256'] \
                or rec.get('success_rule') != SUCCESS_RULE or int(rec['success']) != int(r['success']):
            raise Extract49V2Error(f'audit mismatch {s}')
        E49.check_npz(out / rec['npz'], rec)
        recs[s] = rec
    extra = sorted(p.name for p in out.glob('*.json')
                   if p.name not in ('contract.json', 'PROGRESS.json', 'COMPLETE.json') and p.stem not in stems)
    if extra:
        raise Extract49V2Error(f'unexpected records in output dir: {extra[:5]}')
    E49.write_progress(out, csha, stems, list(recs), sorted(p.name for p in out.glob('*.tmp.npz')))
    cpl = out / 'COMPLETE.json'
    if not cpl.exists():
        with open(cpl, 'x') as fh:
            fh.write(json.dumps({'contract': contract, 'contract_sha256': csha, 'episodes': len(recs),
                                 'decisions': sum(x['n_inferences'] for x in recs.values()), 'compare_16': bool(compare_16),
                                 'transient_success_episodes': sorted(x['episode_id'] for x in recs.values() if x['transient_success'])},
                                indent=1))
    return {'computed': computed, 'skipped': skipped, 'episodes': len(recs), 'plan': plan,
            'transient_success_episodes': sorted(x['episode_id'] for x in recs.values() if x['transient_success'])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--scene', required=True, help="exactly one '<instruction>|s<scene>'")
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--expect-records', type=int, required=True)
    ap.add_argument('--compare-16', type=Path, help='original 16-token extract dir (read-only) for bitwise cross-check')
    ap.add_argument('--reserve-gib', type=float, default=DEFAULT_RESERVE_GIB); ap.add_argument('--estimate', action='store_true')
    a = ap.parse_args(argv)
    try:
        r = run_scene(a.spec, a.store, a.out, a.scene, a.expect_records, a.compare_16, a.reserve_gib, a.estimate)
    except (Extract49V2Error, E49.Extract49Error, ES.ExtractError, FileNotFoundError) as e:
        print('EXTRACT49_V2_ERROR', e, file=sys.stderr); return 2
    print(json.dumps(r, indent=1)); return 0


if __name__ == '__main__':
    sys.exit(main())
