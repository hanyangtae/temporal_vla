"""Run where rollout.pkl lives (승준, python with torch for unpickling; CPU only, no GPU).

One pass over each source rollout of the fold spec (1800 unique, shared by the 27 folds) -> per-episode npz:
  tokmean_l12d3   [n,1536] f16   MIL input: hidden[layer 12 (idx 5), denoise 3] f32 mean over 49 tokens -> f16
                                 (== task_classification MilPoolingTrainer._token_mean rounding; stored f16 so a
                                  [n,1,1536] DataShard re-applies the identical f32->f16->f32 path bit-exactly)
  last_block_d0   [n,49,1536] f16 (optional --with-last-block) layer 15 (idx 6), denoise 0: input of the exact
                                 decoder-input reconstruction (latent_reconstruction_v2) for TRQAM rollout latents
  actions_model12 [n,16,12] f32  GR00T predicted chunk per decision, env units, model order (contract.ACTION_KEYS)
  success, n_inferences
Integrity: sha256 of the pkl bytes read must equal fold-spec pkl_sha256 (archive meta); capture_layers and
record shape are checked. Resume: committed episodes (json + npz sha) are skipped; mismatch aborts.

usage: python extract_scene.py --spec fold_spec.json --store <store root> --out <dir> [--scene 'PPCC/bread|s4'] [--with-last-block]
"""
import argparse, hashlib, io, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

LAYERS = [0, 2, 4, 8, 10, 12, 15]
L12, L15 = LAYERS.index(12), LAYERS.index(15)
# Only the 16 action tokens of the last block are stored: the decoder-input tail (LayerNorm norm_out, timestep-only
# shift/scale, linear proj_out_2) is per-token, so latent = mean over tokens 33:49 needs exactly these (3x smaller).
# rollout_latents.py re-proves slice equivalence on the GPU before use.
LB_TOKENS = slice(33, 49)
ACTION_KEYS = (('end_effector_position', 3), ('end_effector_rotation', 3), ('gripper_close', 1),
               ('base_motion', 4), ('control_mode', 1))


class ExtractError(RuntimeError):
    pass


def episode_id(plan, machine, instr, s, j, n):
    return f'{plan}/{machine}/{instr}/s{s}/j{j}/n{n}'   # same id as task_classification fires/scores


def tokmean_f16(h):
    """[..., 49, D] -> [..., D] f16 via f32 mean (trainer _token_mean before its final f32 cast)."""
    return np.asarray(h).astype(np.float32).mean(-2).astype(np.float16)


def action_model12(chunk):
    """official chunk dict {'action.<key>': [16,d]} -> [16,12] f32 in model order."""
    parts = []
    for key, d in ACTION_KEYS:
        v = chunk.get(f'action.{key}', chunk.get(key))
        if v is None:
            raise ExtractError(f'action key {key} missing ({sorted(chunk)})')
        a = np.asarray(v, np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.shape != (16, d):
            raise ExtractError(f'action {key} shape {a.shape}')
        parts.append(a)
    return np.concatenate(parts, -1)


def episode_steps(payload, n):
    """Executed env steps from src/collect/robocasa/step_phase.py: env_step_success[k] = _check_success() right after
    env step k, so len = executed steps. Contract (not estimated): ceil(steps/n_action_steps) == n records;
    success/failure rules are in success_lag (collector stop may follow the probe's first-success chunk by one chunk);
    (read from the pkl, not hardcoded). Returns (steps, first_success_step or -1, max_episode_steps)."""
    es = payload.get('env_step_success')
    if es is None:
        raise ExtractError('pkl has no env_step_success; steps cannot be established')
    es = [bool(v) for v in es]
    steps, k = len(es), int(payload.get('env_step_n_action_steps', payload.get('n_action_steps')))
    max_steps = payload.get('max_episode_steps')
    if max_steps is None:
        raise ExtractError('pkl has no max_episode_steps')
    if k != 5 or -(-steps // k) != n:
        raise ExtractError(f'steps {steps} / n_action_steps {k} inconsistent with {n} records')
    ph = payload.get('env_step_phases')
    if ph is not None and len(ph) != steps + 1:      # step_phase.py:10-11: phases include s_0
        raise ExtractError(f'env_step_phases length {len(ph)} != steps+1 ({steps + 1})')
    first = es.index(True) if any(es) else -1
    lag = success_lag(es, first, steps, k, int(max_steps), bool(payload['episode_success']))
    return steps, first, int(max_steps), lag


def success_lag(es, first, steps, k, max_steps, success):
    """Two success signals exist (evidence: http_feature_collect.py:1480-1487 stops on the chunk-level
    step_success(reward, info) of MultiStepWrapper (multistep_wrapper.py:306, raw-env info['success'] / reward);
    env_step_success is a separate probe calling _check_success() after each raw step (step_phase.py:15)).
    The probe can fire one or more chunks before the collector's stop signal (real: first 134, steps 140), so post-
    success chunks are real executed data and are kept. Contract (fail-closed otherwise):
      success: steps on a chunk boundary (or == max), probe success continuous from the first success to the end,
               and the stop lag after the first-success chunk is 0 or exactly one chunk (observed).
      failure: no probe success at all and steps == max_episode_steps.
    Returns lag = steps - end_of_first_success_chunk (0 when both signals coincide; -1 for failures)."""
    if not success:
        if first != -1 or steps != max_steps:
            raise ExtractError(f'failure episode: first success {first}, steps {steps} vs max_episode_steps {max_steps}')
        return -1
    if first < 0:
        raise ExtractError('success episode without any probe success step')
    first_end = min(-(-(first + 1) // k) * k, max_steps)
    if steps < first_end:
        raise ExtractError(f'success episode ended at {steps} before its first-success chunk end {first_end}')
    if steps % k and steps != max_steps:
        raise ExtractError(f'success episode steps {steps} not on a {k}-step chunk boundary')
    if not all(es[first:steps]):
        raise ExtractError(f'probe success not continuous from first {first} to end {steps}; transient success unexplained')
    lag = steps - first_end
    # Observed (OvenRack/out-left s2 j0 n3: probe true 134..139, steps 140): the collector's chunk-level signal stops
    # exactly one chunk after the probe's first-success chunk. Only 0 or one-chunk lag is accepted; anything longer
    # is outside the evidence and fails closed.
    if lag not in (0, k):
        raise ExtractError(f'success stop lag {lag} steps not in (0, {k}); outside observed collector behaviour')
    return lag


def convert(payload, with_last_block):
    if list(payload.get('capture_layers') or []) != LAYERS:
        raise ExtractError(f'capture_layers {payload.get("capture_layers")}')
    hs = payload['hidden_states']
    n = len(hs)
    if n == 0 or len(payload['actions']) != n:
        raise ExtractError(f'records {n} vs actions {len(payload["actions"])}')
    tm = np.empty((n, 1536), np.float16)
    lb = np.empty((n, LB_TOKENS.stop - LB_TOKENS.start, 1536), np.float16) if with_last_block else None
    for i, h in enumerate(hs):
        a = h.detach().cpu().numpy() if hasattr(h, 'detach') else np.asarray(h)
        if a.shape != (7, 4, 49, 1536):
            raise ExtractError(f'record {i} shape {a.shape}')
        tm[i] = tokmean_f16(a[L12, 3])
        if lb is not None:
            lb[i] = a[L15, 0, LB_TOKENS].astype(np.float16)
    steps, first, max_steps, lag = episode_steps(payload, n)
    out = {'tokmean_l12d3': tm, 'actions_model12': np.stack([action_model12(c) for c in payload['actions']]),
           'success': np.int8(payload['episode_success']), 'n_inferences': np.int32(n),
           'env_steps': np.int32(steps), 'first_success_step': np.int32(first), 'max_episode_steps': np.int32(max_steps),
           'success_chunk_lag_steps': np.int32(lag),
           'env_step_success': np.asarray([bool(v) for v in payload['env_step_success']], np.int8),
           'env_step_phases_len': np.int32(len(payload['env_step_phases']) if payload.get('env_step_phases') is not None else -1)}
    if lb is not None:
        out['last_block_d0'] = lb
    return out


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


ADOPT_SAME = ('spec_sha256', 'with_last_block', 'layers')


def adopt_record(src_dir, dst_dir, src_json, spec_row, contract):
    """Bring one committed record of an older extractor version into this versioned dir WITHOUT duplicating bytes:
    the npz is hard-linked (same filesystem required; refuses to copy), its sha re-verified, its arrays re-validated
    with the current step contract (success_lag), and the json records the source contract sha + extractor version."""
    src_contract = json.loads((src_dir / 'contract.json').read_text())
    diff = [k for k in ADOPT_SAME if src_contract.get(k) != contract.get(k)]
    if diff:
        raise ExtractError(f'adopt source contract differs in {diff}')
    rec = json.loads(src_json.read_text())
    npz_src = src_dir / rec['npz']
    if sha256_bytes(npz_src.read_bytes()) != rec['npz_sha256'] or rec['pkl_sha256'] != spec_row['pkl_sha256']:
        raise ExtractError(f'adopt: sha/linkage mismatch {rec["episode_id"]}')
    with np.load(npz_src) as z:
        es = [bool(v) for v in z['env_step_success']]
        steps, first, mx, succ = int(z['env_steps']), int(z['first_success_step']), int(z['max_episode_steps']), bool(z['success'])
        lb = z['last_block_d0'].shape[1] if 'last_block_d0' in z.files else None
    if succ != bool(spec_row['success']) or len(es) != steps or (first if first >= 0 else None) != (es.index(True) if any(es) else None):
        raise ExtractError(f'adopt: stored arrays inconsistent {rec["episode_id"]}')
    lag = success_lag(es, first, steps, 5, mx, succ)
    dst_npz = dst_dir / rec['npz']
    os.link(npz_src, dst_npz)                         # OSError across filesystems -> no silent copy
    new = {**{k: v for k, v in rec.items() if k != 'seconds'}, 'success_chunk_lag_steps': lag, 'first_success_step': first,
           'adopted_from_contract_sha256': sha256_bytes((src_dir / 'contract.json').read_bytes()),
           'adopted_from_extractor_sha256': src_contract.get('extractor_sha256'),
           'last_block_tokens_in_npz': lb, 'npz_has_lag_key': False}
    with open(dst_dir / src_json.name, 'x') as fh:
        fh.write(json.dumps(new, indent=1))
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scene', action='append', help="'<instruction>|s<scene>' (default all 9)")
    ap.add_argument('--with-last-block', action='store_true')
    ap.add_argument('--adopt-from', type=Path, help='older-version extract dir: hard-link + re-validate its committed records')
    ap.add_argument('--shard', help='i/N: process only every N-th source starting at i (parallel, same --out)')
    ap.add_argument('--adopt-only', action='store_true', help='run adoption serially, then exit (before parallel shards)')
    a = ap.parse_args()
    if a.adopt_from and a.shard:
        sys.exit('--adopt-from must run serially (use --adopt-only first, then shards without --adopt-from)')
    if a.adopt_only and not a.adopt_from:
        sys.exit('--adopt-only needs --adopt-from')
    spec = json.loads(a.spec.read_text())
    if not spec.get('production_ready'):
        sys.exit('fold spec not production_ready (needs --archive-index confirmation)')
    sources = {}
    for f in spec['folds']:
        tag = f'{f["instruction"]}|s{f["scene"]}'
        if a.scene and tag not in a.scene:
            continue
        for r in f['fit'] + f['eval']:
            k = (r['plan_id'], f['archive_machine'], f['instruction'], f['scene'], r['j'], r['n'])
            sources[k] = r
    a.out.mkdir(parents=True, exist_ok=True)
    contract = {'spec_sha256': sha256_bytes(a.spec.read_bytes()), 'extractor_sha256': sha256_bytes(Path(__file__).read_bytes()),
                'with_last_block': a.with_last_block, 'layers': LAYERS,
                'last_block_tokens': [LB_TOKENS.start, LB_TOKENS.stop] if a.with_last_block else None}
    cp = a.out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        sys.exit('contract mismatch with existing output dir; use a new --out')
    if not cp.exists():
        cp.write_text(json.dumps(contract, indent=1))
    by_stem = {hashlib.sha256(episode_id(*k).encode()).hexdigest()[:24]: r for k, r in sources.items()}
    if a.adopt_from:
        adopted = 0
        for sj in sorted(a.adopt_from.glob('*.json')):
            if sj.name in ('contract.json', 'COMPLETE.json') or sj.stem not in by_stem or (a.out / sj.name).exists():
                continue
            adopt_record(a.adopt_from, a.out, sj, by_stem[sj.stem], contract); adopted += 1
        print(json.dumps({'adopted': adopted}), flush=True)
    if a.adopt_only:
        return
    items = sorted(sources.items())
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        if not 0 <= i < n:
            raise ExtractError(f'bad --shard {a.shard}')
        items = items[i::n]                          # disjoint subsets; parallel processes share the out dir safely
    done = 0
    for k, r in items:
        eid = episode_id(*k)
        stem = hashlib.sha256(eid.encode()).hexdigest()[:24]
        jp, npz = a.out / f'{stem}.json', a.out / f'{stem}.npz'
        if jp.exists():
            rec = json.loads(jp.read_text())
            if rec['episode_id'] != eid or sha256_bytes(npz.read_bytes()) != rec['npz_sha256']:
                raise ExtractError(f'committed record mismatch {eid}')
            done += 1
            continue
        t0 = time.time()
        rel = Path('groot/n15/grid', *map(str, k[:3]), f's{k[3]}', f'j{k[4]}', f'n{k[5]}', 'base', 'rollout.pkl')
        blob = (a.store / rel).read_bytes()
        if sha256_bytes(blob) != r['pkl_sha256']:
            raise ExtractError(f'pkl sha mismatch {eid}')
        payload = pickle.load(io.BytesIO(blob))
        del blob
        arrays = convert(payload, a.with_last_block)
        if int(arrays['success']) != int(r['success']):
            raise ExtractError(f'pkl success {int(arrays["success"])} != spec {r["success"]} {eid}')
        tmp = npz.with_suffix('.tmp.npz')
        np.savez(tmp, **arrays)
        os.replace(tmp, npz)
        rec = {'episode_id': eid, 'plan_id': k[0], 'machine': k[1], 'instruction': k[2], 'scene': k[3], 'jitter': k[4],
               'noise': k[5], 'success': int(arrays['success']), 'n_inferences': int(arrays['n_inferences']),
               'env_steps': int(arrays['env_steps']), 'max_episode_steps': int(arrays['max_episode_steps']),
               'first_success_step': int(arrays['first_success_step']), 'success_chunk_lag_steps': int(arrays['success_chunk_lag_steps']),
               'pkl_sha256': r['pkl_sha256'], 'npz': npz.name, 'npz_sha256': sha256_bytes(npz.read_bytes()),
               'seconds': time.time() - t0}
        with open(jp, 'x') as fh:
            fh.write(json.dumps(rec, indent=1))
        done += 1
        print(json.dumps({'done': done, 'total': len(items), 'episode': eid}), flush=True)
    recs = [json.loads(p.read_text()) for p in sorted(a.out.glob('*.json')) if p.name != 'contract.json' and p.name != 'COMPLETE.json']
    if len(recs) != len(sources):
        if a.shard:
            print(json.dumps({'shard_done': a.shard, 'records': len(recs), 'sources': len(sources)}))
            return                                   # another shard still running; last finisher writes COMPLETE
        raise ExtractError(f'{len(recs)} records != {len(sources)} sources')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'contract': contract, 'episodes': len(recs),
                                                     'decisions': sum(x['n_inferences'] for x in recs)}, indent=1))


if __name__ == '__main__':
    main()
