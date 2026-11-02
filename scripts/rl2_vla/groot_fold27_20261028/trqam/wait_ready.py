"""CPU-only wait until the requested folds are READY for train_queue (run BEFORE the harness guard, so no GPU is held
while the latent conversion that needs the same GPU is still running — avoids a reservation deadlock).

READY = scene latents dir has COMPLETE.json whose episode count == --expect-records, and the compact dir has contract.json.
Exit 0 when all requested folds are ready; 75 when --max-wait-h passes first (nothing launched).

    .venvs/rl2_iid/bin/python wait_ready.py --sources <sources.json> --folds <fold_id> ... [--max-wait-h 24] [--poll-s 60]
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from folds import load_spec   # noqa: E402

SPEC = HERE.parents[3] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'


def not_ready(spec, sources, folds, expect):
    out = {}
    for f in spec['folds']:
        if f['fold_id'] not in folds:
            continue
        s = sources.get(f'{f["instruction"]}|s{f["scene"]}')
        if s is None:
            out[f['fold_id']] = 'scene missing from sources'; continue
        c = Path(s['latents']) / 'COMPLETE.json'
        if not (Path(s['episodes']) / 'contract.json').exists():
            out[f['fold_id']] = 'compact contract missing'
        elif not c.exists():
            out[f['fold_id']] = 'latents not COMPLETE'
        elif json.loads(c.read_text()).get('episodes') != expect:
            out[f['fold_id']] = f'latents COMPLETE episodes {json.loads(c.read_text()).get("episodes")} != {expect}'
    unknown = set(folds) - {f['fold_id'] for f in spec['folds']}
    out.update({u: 'unknown fold' for u in unknown})
    return out


def main(argv=None, sleep=time.sleep, now=time.time):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, default=SPEC); ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--folds', nargs='+', required=True); ap.add_argument('--expect-records', type=int, default=200)
    ap.add_argument('--max-wait-h', type=float, default=24); ap.add_argument('--poll-s', type=float, default=60)
    ap.add_argument('--require', type=Path, action='append', default=[],
                    help='extra gate file that must exist (e.g. the gated-smoke audit PASS marker); repeatable')
    a = ap.parse_args(argv)
    spec, sources = load_spec(a.spec), json.loads(a.sources.read_text())
    deadline = now() + a.max_wait_h * 3600
    while True:
        nr = not_ready(spec, sources, a.folds, a.expect_records)
        nr.update({str(p): 'required gate file missing' for p in a.require if not p.exists()})
        if any(v == 'unknown fold' or v.startswith('latents COMPLETE episodes') or v.startswith('scene missing') for v in nr.values()):
            print(json.dumps({'error': nr}), file=sys.stderr); return 2          # cannot become ready by waiting
        if not nr:
            print(json.dumps({'ready': a.folds})); return 0
        if now() + a.poll_s > deadline:
            print(json.dumps({'timeout': nr}), file=sys.stderr); return 75
        sleep(a.poll_s)


if __name__ == '__main__':
    sys.exit(main())
