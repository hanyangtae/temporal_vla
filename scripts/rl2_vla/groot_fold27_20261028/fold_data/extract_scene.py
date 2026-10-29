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
    success -> the episode ends at the end of the chunk containing the first success step (<= 4 steps later);
    failure -> no success step and steps == max_episode_steps
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
    if bool(payload['episode_success']):
        # Collection finishes the 5-step chunk in which success first fires (real pkl: first 200, steps 205), so the
        # executed length ends at that chunk boundary (or the horizon). Both values are stored; neither is altered.
        want = min(-(-(first + 1) // k) * k, int(max_steps)) if first >= 0 else None
        if first < 0 or steps != want:
            raise ExtractError(f'success episode: first success step {first}, steps {steps} != chunk-boundary end {want}')
    elif first != -1 or steps != int(max_steps):
        raise ExtractError(f'failure episode: first success {first}, steps {steps} vs max_episode_steps {max_steps}')
    return steps, first, int(max_steps)


def convert(payload, with_last_block):
    if list(payload.get('capture_layers') or []) != LAYERS:
        raise ExtractError(f'capture_layers {payload.get("capture_layers")}')
    hs = payload['hidden_states']
    n = len(hs)
    if n == 0 or len(payload['actions']) != n:
        raise ExtractError(f'records {n} vs actions {len(payload["actions"])}')
    tm = np.empty((n, 1536), np.float16)
    lb = np.empty((n, 49, 1536), np.float16) if with_last_block else None
    for i, h in enumerate(hs):
        a = h.detach().cpu().numpy() if hasattr(h, 'detach') else np.asarray(h)
        if a.shape != (7, 4, 49, 1536):
            raise ExtractError(f'record {i} shape {a.shape}')
        tm[i] = tokmean_f16(a[L12, 3])
        if lb is not None:
            lb[i] = a[L15, 0].astype(np.float16)
    steps, first, max_steps = episode_steps(payload, n)
    out = {'tokmean_l12d3': tm, 'actions_model12': np.stack([action_model12(c) for c in payload['actions']]),
           'success': np.int8(payload['episode_success']), 'n_inferences': np.int32(n),
           'env_steps': np.int32(steps), 'first_success_step': np.int32(first), 'max_episode_steps': np.int32(max_steps),
           'env_step_success': np.asarray([bool(v) for v in payload['env_step_success']], np.int8),
           'env_step_phases_len': np.int32(len(payload['env_step_phases']) if payload.get('env_step_phases') is not None else -1)}
    if lb is not None:
        out['last_block_d0'] = lb
    return out


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scene', action='append', help="'<instruction>|s<scene>' (default all 9)")
    ap.add_argument('--with-last-block', action='store_true')
    a = ap.parse_args()
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
                'with_last_block': a.with_last_block, 'layers': LAYERS}
    cp = a.out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        sys.exit('contract mismatch with existing output dir; use a new --out')
    cp.write_text(json.dumps(contract, indent=1))
    done = 0
    for k, r in sorted(sources.items()):
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
               'pkl_sha256': r['pkl_sha256'], 'npz': npz.name, 'npz_sha256': sha256_bytes(npz.read_bytes()),
               'seconds': time.time() - t0}
        with open(jp, 'x') as fh:
            fh.write(json.dumps(rec, indent=1))
        done += 1
        print(json.dumps({'done': done, 'total': len(sources), 'episode': eid}), flush=True)
    recs = [json.loads(p.read_text()) for p in sorted(a.out.glob('*.json')) if p.name != 'contract.json' and p.name != 'COMPLETE.json']
    if len(recs) != len(sources):
        raise ExtractError(f'{len(recs)} records != {len(sources)} sources')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'contract': contract, 'episodes': len(recs),
                                                     'decisions': sum(x['n_inferences'] for x in recs)}, indent=1))


if __name__ == '__main__':
    main()
