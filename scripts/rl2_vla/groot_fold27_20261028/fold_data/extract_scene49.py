"""Explicit 49-token re-extraction of ONE scene's verified rollouts (CPU, where rollout.pkl lives). New version; the
original extract_scene.py (16-token last block 33:49) is imported read-only and never modified, and original extraction
dirs / compact data are only READ (optional --compare-16).

Per episode npz = original extract_scene.convert(payload, with_last_block=False) arrays (tokmean_l12d3, actions_model12,
success / step contract, ...) UNCHANGED, plus
  last_block_d0  [n,49,1536] float16 = hidden_states[i][L15 (layer 15), denoise 0, tokens 0:49]  (all 49 tokens; no
                 padding, no slicing, f16 cast exactly as the original does for its 33:49 slice)
Strictness: pkl bytes sha256 == fold-spec pkl_sha256; capture_layers / record shape (7,4,49,1536) checked by the original
convert and again here; spec success == pkl success; record count of the scene == --expect-records (one scene per run).
--compare-16 <original extract dir>: for every episode the original record (same stem, same pkl sha, npz sha verified)
must satisfy new[:, 33:49] == original last_block_d0 bitwise AND every shared array equal bitwise; mismatch = no commit.
Output contract (kind fold27_extract49_v1, never the original's), exclusive-create records, unique temp names, refuses a
dir holding another contract or an npz without its json; PROGRESS.json (status incomplete, done/missing lists, stale temp
files) is rewritten atomically; COMPLETE.json is created exclusively only after a full audit of all expected records.
Space: before work, free space on the OUTPUT filesystem must cover the raw upper bound of the remaining records
(n_max = ceil(max_episode_steps / 5) decisions x 49 x 1536 x 2 B + small arrays) plus --reserve-gib. /tmp is not used.

    python extract_scene49.py --spec fold_spec.json --store /home/kimseungjun/datasets/temporal_vla_store \
        --scene 'PPCC/jug|s1' --out <datafs>/extract49/srv50_jug_s1 --expect-records 200 [--compare-16 <orig dir>] [--estimate]
"""
import argparse, hashlib, io, json, math, os, pickle, shutil, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_scene as ES   # noqa: E402  (frozen original; read-only use)

KIND = 'fold27_extract49_v1'
TOKENS = (0, 49)
N_ACTION_STEPS = 5
D = 1536


class Extract49Error(RuntimeError):
    pass


def stem_of(eid):
    return hashlib.sha256(eid.encode()).hexdigest()[:24]      # == extract_scene stems (compare-16 lookup)


def scene_sources(spec, scene):
    """{(plan, machine, instr, scene, j, n): spec row} for ONE '<instruction>|s<scene>' (rows of all its folds, deduped)."""
    out = {}
    for f in spec['folds']:
        if f'{f["instruction"]}|s{f["scene"]}' != scene:
            continue
        for r in f['fit'] + f['eval']:
            k = (r['plan_id'], f['archive_machine'], f['instruction'], f['scene'], r['j'], r['n'])
            if k in out and out[k]['pkl_sha256'] != r['pkl_sha256']:
                raise Extract49Error(f'spec rows disagree for {k}')
            out[k] = r
    return out


def source_path(store, k):
    return Path(store) / 'groot/n15/grid' / str(k[0]) / str(k[1]) / str(k[2]) / f's{k[3]}' / f'j{k[4]}' / f'n{k[5]}' / 'base' / 'rollout.pkl'


def last_block49(payload):
    hs = payload['hidden_states']
    lb = np.empty((len(hs), TOKENS[1] - TOKENS[0], D), np.float16)
    dtypes = set()
    for i, h in enumerate(hs):
        a = h.detach().cpu().numpy() if hasattr(h, 'detach') else np.asarray(h)
        if a.shape != (7, 4, 49, D):
            raise Extract49Error(f'record {i} shape {a.shape} != (7,4,49,{D})')
        dtypes.add(str(a.dtype))
        lb[i] = a[ES.L15, 0, TOKENS[0]:TOKENS[1]].astype(np.float16)
    if len(dtypes) != 1:
        raise Extract49Error(f'mixed source dtypes {sorted(dtypes)}')
    return lb, dtypes.pop()


def convert49(payload):
    arrays = ES.convert(payload, with_last_block=False)       # original arrays, original code
    lb, src_dtype = last_block49(payload)
    if len(lb) != int(arrays['n_inferences']):
        raise Extract49Error('last block / record count mismatch')
    arrays['last_block_d0'] = lb
    return arrays, src_dtype


def raw_upper_bound_bytes(n_records, max_episode_steps=720):
    n = math.ceil(max_episode_steps / N_ACTION_STEPS)
    per = n * (49 * D * 2 + D * 2 + 16 * 12 * 4) + max_episode_steps + 4096     # block f16 + tokmean f16 + actions f32 + step flags + zip/json
    return n_records * per


def check_npz(npz, rec):
    with np.load(npz) as z:
        lb = z['last_block_d0']
        if lb.dtype != np.float16 or lb.shape != (rec['n_inferences'], 49, D):
            raise Extract49Error(f'{npz.name}: last_block_d0 {lb.dtype} {lb.shape}')


def compare16(orig_dir, stem, r, arrays):
    oj = Path(orig_dir) / f'{stem}.json'
    if not oj.exists():
        raise Extract49Error(f'compare-16: original record {oj.name} missing')
    o = json.loads(oj.read_text())
    if o['pkl_sha256'] != r['pkl_sha256'] or ES.sha256_bytes((Path(orig_dir) / o['npz']).read_bytes()) != o['npz_sha256']:
        raise Extract49Error(f'compare-16: original record {oj.name} sha/linkage mismatch')
    with np.load(Path(orig_dir) / o['npz']) as z:
        if 'last_block_d0' not in z.files or z['last_block_d0'].shape[1] != 16:
            raise Extract49Error('compare-16: original has no 16-token last block')
        if not np.array_equal(z['last_block_d0'], arrays['last_block_d0'][:, 33:49]) or z['last_block_d0'].dtype != np.float16:
            raise Extract49Error(f'compare-16: new[:,33:49] != original 16-token block for {oj.name}')
        for k in z.files:
            if k != 'last_block_d0' and (k not in arrays or not np.array_equal(z[k], arrays[k])):
                raise Extract49Error(f'compare-16: shared array {k} differs for {oj.name}')
    return True


def write_progress(out, contract_sha, expected, done, stale):
    body = {'status': 'incomplete', 'contract_sha256': contract_sha, 'expected': len(expected), 'done': len(done),
            'missing': sorted(set(expected) - set(done)), 'stale_temp_files': stale, 'time': time.time()}
    tmp = out / f'PROGRESS.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(body, indent=1)); os.replace(tmp, out / 'PROGRESS.json')


def run_scene(spec_path, store, out, scene, expect, compare_16=None, reserve_gib=2.0, estimate=False,
              disk=shutil.disk_usage, convert=convert49, log=print):
    spec_bytes = Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    if not spec.get('production_ready'):
        raise Extract49Error('fold spec not production_ready')
    sources = scene_sources(spec, scene)
    if len(sources) != expect:
        raise Extract49Error(f'{scene}: {len(sources)} spec sources != --expect-records {expect}')
    out = Path(out)
    contract = {'kind': KIND, 'scene': scene, 'expect_records': expect, 'spec_sha256': ES.sha256_bytes(spec_bytes),
                'extractor49_sha256': ES.sha256_bytes(Path(__file__).read_bytes()),
                'base_extractor_sha256': ES.sha256_bytes(Path(ES.__file__).read_bytes()), 'layers': ES.LAYERS,
                'last_block_layer': 15, 'last_block_denoise': 0, 'last_block_tokens': list(TOKENS),
                'last_block_dtype': 'float16', 'with_last_block': True,
                'compare_16_contract_sha256': (ES.sha256_bytes((Path(compare_16) / 'contract.json').read_bytes())
                                               if compare_16 else None)}
    stems = {stem_of(ES.episode_id(*k)): (k, r) for k, r in sources.items()}
    done = [s for s in stems if (out / f'{s}.json').exists()]
    bound = raw_upper_bound_bytes(len(stems) - len(done))
    free = disk(out if out.exists() else out.parent).free
    need = bound + int(reserve_gib * 1024 ** 3)
    plan = {'scene': scene, 'records': len(stems), 'already_done': len(done), 'raw_upper_bound_bytes': bound,
            'free_bytes': free, 'required_bytes': need, 'fits': free >= need}
    if estimate:
        return plan
    if free < need:
        raise Extract49Error(f'insufficient space on output filesystem: free {free} < upper bound {bound} + reserve')
    out.mkdir(parents=True, exist_ok=True)
    cp = out / 'contract.json'
    if cp.exists():
        old = json.loads(cp.read_text())
        if old != contract:
            raise Extract49Error(f'{cp} is a different contract (kind {old.get("kind")}); use a new --out')
    else:
        with open(cp, 'x') as fh:
            fh.write(json.dumps(contract, indent=1))
    csha = ES.sha256_bytes(cp.read_bytes())
    for npz in out.glob('*.npz'):
        if not npz.name.endswith('.tmp.npz') and not (out / f'{npz.stem}.json').exists():
            raise Extract49Error(f'orphan npz without record {npz.name}; inspect before resuming')
    write_progress(out, csha, stems, done, sorted(p.name for p in out.glob('*.tmp.npz')))   # incomplete manifest up front
    computed, skipped = 0, 0
    for s, (k, r) in sorted(stems.items()):
        eid = ES.episode_id(*k)
        jp, npz = out / f'{s}.json', out / f'{s}.npz'
        if jp.exists():
            rec = json.loads(jp.read_text())
            if rec['episode_id'] != eid or rec['pkl_sha256'] != r['pkl_sha256'] or ES.sha256_bytes(npz.read_bytes()) != rec['npz_sha256']:
                raise Extract49Error(f'committed record mismatch {eid}')
            check_npz(npz, rec)
            skipped += 1
            continue
        t0 = time.time()
        blob = source_path(store, k).read_bytes()
        if ES.sha256_bytes(blob) != r['pkl_sha256']:
            raise Extract49Error(f'pkl sha mismatch {eid}')
        payload = pickle.load(io.BytesIO(blob))
        del blob
        arrays, src_dtype = convert(payload)
        if int(arrays['success']) != int(r['success']):
            raise Extract49Error(f'pkl success {int(arrays["success"])} != spec {r["success"]} {eid}')
        if compare_16:
            compare16(compare_16, s, r, arrays)
        tmp = out / f'{s}.{os.getpid()}.tmp.npz'
        np.savez(tmp, **arrays)
        if npz.exists():
            raise Extract49Error(f'{npz.name} appeared concurrently; refusing to overwrite')
        os.replace(tmp, npz)
        rec = {'episode_id': eid, 'plan_id': k[0], 'machine': k[1], 'instruction': k[2], 'scene': k[3], 'jitter': k[4],
               'noise': k[5], 'success': int(arrays['success']), 'n_inferences': int(arrays['n_inferences']),
               'env_steps': int(arrays['env_steps']), 'max_episode_steps': int(arrays['max_episode_steps']),
               'first_success_step': int(arrays['first_success_step']),
               'success_chunk_lag_steps': int(arrays['success_chunk_lag_steps']),
               'pkl_sha256': r['pkl_sha256'], 'npz': npz.name, 'npz_sha256': ES.sha256_bytes(npz.read_bytes()),
               'last_block_tokens': list(TOKENS), 'source_hidden_dtype': src_dtype, 'compare_16': bool(compare_16),
               'seconds': time.time() - t0}
        with open(jp, 'x') as fh:
            fh.write(json.dumps(rec, indent=1))
        computed += 1
        done.append(s)
        write_progress(out, csha, stems, done, sorted(p.name for p in out.glob('*.tmp.npz')))
        log(json.dumps({'done': len(done), 'total': len(stems), 'episode': eid}))
    # full audit before COMPLETE
    recs = {}
    for s, (k, r) in stems.items():
        jp = out / f'{s}.json'
        if not jp.exists():
            raise Extract49Error(f'missing record {s}')
        rec = json.loads(jp.read_text())
        if rec['episode_id'] != ES.episode_id(*k) or ES.sha256_bytes((out / rec['npz']).read_bytes()) != rec['npz_sha256']:
            raise Extract49Error(f'audit mismatch {s}')
        check_npz(out / rec['npz'], rec)
        recs[s] = rec
    extra = sorted(p.name for p in out.glob('*.json')
                   if p.name not in ('contract.json', 'PROGRESS.json', 'COMPLETE.json') and p.stem not in stems)
    if extra:
        raise Extract49Error(f'unexpected records in output dir: {extra[:5]}')
    write_progress(out, csha, stems, list(recs), sorted(p.name for p in out.glob('*.tmp.npz')))
    cpl = out / 'COMPLETE.json'
    if not cpl.exists():
        with open(cpl, 'x') as fh:
            fh.write(json.dumps({'contract': contract, 'contract_sha256': csha, 'episodes': len(recs),
                                 'decisions': sum(x['n_inferences'] for x in recs.values()),
                                 'compare_16': bool(compare_16)}, indent=1))
    return {'computed': computed, 'skipped': skipped, 'episodes': len(recs), 'plan': plan}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--scene', required=True, help="exactly one '<instruction>|s<scene>'")
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--expect-records', type=int, required=True)
    ap.add_argument('--compare-16', type=Path, help='original 16-token extract dir (read-only) for bitwise cross-check')
    ap.add_argument('--reserve-gib', type=float, default=2.0); ap.add_argument('--estimate', action='store_true')
    a = ap.parse_args(argv)
    try:
        r = run_scene(a.spec, a.store, a.out, a.scene, a.expect_records, a.compare_16, a.reserve_gib, a.estimate)
    except (Extract49Error, ES.ExtractError, FileNotFoundError) as e:
        print('EXTRACT49_ERROR', e, file=sys.stderr); return 2
    print(json.dumps(r, indent=1)); return 0


if __name__ == '__main__':
    sys.exit(main())
