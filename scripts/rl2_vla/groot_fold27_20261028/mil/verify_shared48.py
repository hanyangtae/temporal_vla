"""CPU verification of the 48 shared PR11 MIL checkpoints (output/share/grid_n10_n19/runs/*/attn/raw/s*).

Checks per run (no GPU, no data download):
  1. loads with MilPoolingModel.load_model(device='cpu') through MilGate.from_run (attn/raw contract, transform shapes)
  2. recipe = SGD lr .3 momentum .9 nesterov, fire_threshold .5; cell/target_jitter recorded
  3. stored scores.npz test w_F -> first crossing (> thr) reproduces fires.tsv fired/t_fire for every test episode
  4. online causal path: MilGate.step over a synthetic hidden sequence ([4,49,1536] f16 per step, layer-12 hook
     layout) == batch MilGate.scores(token_mean(hidden[:,3])) exactly; snapshot/restore round-trip
Feature-level reproduction of scores.npz needs the original rollouts (episode ids are archive coordinates);
run `--episodes <extract_scene out>` after extraction to compare w_F on the n0..19 episodes.

usage: python verify_shared48.py --out outputs/analysis/groot_fold27_20261028/mil/shared48_verification.json [--episodes DIR]
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mil_gate import MilGate, TC, token_mean, tc_commit   # noqa: E402

RUNS = TC / 'output/share/grid_n10_n19/runs'


def check_run(run, rng, episodes=None):
    g = MilGate.from_run(run)
    ck = g.ck
    tr = ck['config']['train']
    rec = {'run': str(run.relative_to(TC)), 'model_sha256': g.model_sha256, 'cells': ck['cell']['cells'],
           'target_jitter': ck['cell']['target_jitter'], 'train_jitter': ck['cell']['train_jitter'],
           'seed': ck.get('seed'), 'best_epoch': ck['best']['epoch'], 'threshold': g.threshold,
           'recipe_ok': tr['optimizer'] == 'sgd' and tr['lr'] == 0.3 and tr['momentum'] == 0.9 and bool(tr['nesterov'])}
    # 3. stored scores -> fires.tsv
    scores = np.load(run / 'scores.npz')
    with open(run / 'fires.tsv', newline='') as f:
        fires = list(csv.DictReader(f, delimiter='\t'))
    bad = []
    for row in fires:
        w = scores[f'{row["episode_id"]}|w_F']
        hit = np.flatnonzero(w > g.threshold)
        t = int(hit[0]) if len(hit) else None
        want_t = None if row['t_fire'] in ('', 'nan', 'None') else int(float(row['t_fire']))
        if (row['fired'] == 'True') != (t is not None) or t != want_t or len(w) != int(row['T']):
            bad.append(row['episode_id'])
    rec['fires_reproduced'] = f'{len(fires) - len(bad)}/{len(fires)}'
    # 4. online == batch on synthetic layer-12 hook output
    T = 12
    hidden = rng.normal(0, 1, size=(T, 4, 49, 1536)).astype(np.float16)
    batch = g.scores(token_mean(hidden[:, 3]))
    g.reset()
    online = []
    for t in range(T):
        snap = g.snapshot()
        r1 = g.step(g.feature_from_hidden(hidden[t][None], None))
        g.restore(snap)
        r2 = g.step(g.feature_from_hidden(hidden[t], None))
        if r1 != r2:
            raise AssertionError('snapshot/restore not idempotent')
        online.append(r2['score'])
    rec['online_equals_batch'] = bool(np.array_equal(np.float32(online), batch.astype(np.float32)))
    # optional: feature-level reproduction against stored w_F
    if episodes is not None:
        n_cmp, maxdiff = 0, 0.0
        for key in scores.files:
            eid = key.rsplit('|', 1)[0]
            jp = episodes / (hashlib.sha256(eid.encode()).hexdigest()[:24] + '.json')
            if not jp.exists():
                continue
            with np.load(episodes / json.loads(jp.read_text())['npz']) as z:
                w = g.scores(z['tokmean_l12d3'].astype(np.float32))
            if w.shape != scores[key].shape:
                raise AssertionError(f'length mismatch {eid}')
            maxdiff = max(maxdiff, float(np.abs(w - scores[key]).max())); n_cmp += 1
        rec['feature_reproduction'] = {'episodes_compared': n_cmp, 'max_abs_diff_w_F': maxdiff}
    rec['ok'] = rec['recipe_ok'] and not bad and rec['online_equals_batch']
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--episodes', type=Path)
    a = ap.parse_args()
    runs = sorted(p.parent for p in RUNS.glob('*/attn/raw/s*/model.pt'))
    rng = np.random.default_rng(20261029)
    recs = [check_run(r, rng, a.episodes) for r in runs]
    bread = [r for r in recs if r['run'].endswith('cell-n10-bread_s4/attn/raw/s0')]
    out = {'tc_commit': tc_commit(), 'runs': len(recs), 'ok': sum(r['ok'] for r in recs),
           'bread_s4_s0': bread[0] if bread else None, 'records': recs,
           'note': 'shared n10/n19 checkpoints validate loading/inference only; never reused as fold detectors'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('tc_commit', 'runs', 'ok')}))
    if len(recs) != 48 or out['ok'] != 48:
        sys.exit(2)


if __name__ == '__main__':
    main()
