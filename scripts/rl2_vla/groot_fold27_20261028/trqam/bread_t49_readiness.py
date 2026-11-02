"""Bread-only (PPCC/bread s4, srv50) TRQAM readiness on native 49-token latents — CPU, read-only. Run on the srv50 deploy
before any smoke/training. Checks exactly what finetune_fold / wait_ready will require, plus source-heldout separation:
  sources     the sources file holds ONLY 'PPCC/bread|s4' (so host coordinators train only the 3 bread folds, 9 arms)
  folds       spec folds for that scene = 3 (j1/j3/j4 targets); per fold: fit = 160 rows, all j != target_j;
              eval = 40 rows, all j == target_j; fit and eval episode ids disjoint (held-out target never trained on)
  linkage     every fit row (and eval row) has a compact record and a latent record with the same episode_id and the
              spec pkl_sha256 (finetune_fold.check_linkage formula); latents contract episodes_contract_sha256 == the
              compact contract's source_contract_sha256 (finetune_fold.check_latent_provenance)
  latents     COMPLETE.json episodes == 200; every latent record token_contract == 49; contract.json token_contract == 49
Exit 0 = ready, 3 = not ready (report lists every problem).

    python bread_t49_readiness.py --sources outputs/analysis/groot_fold27_20261028/trqam_sources_srv50_bread_t49.json
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import latents_v2_audit as AU   # noqa: E402

SCENE_KEY = 'PPCC/bread|s4'


def eid_of(fold, r):     # == finetune_fold.load_fit_episodes episode id
    return f'{r["plan_id"]}/{fold["archive_machine"]}/{fold["instruction"]}/s{fold["scene"]}/j{r["j"]}/n{r["n"]}'


def index(d):
    return {json.loads(p.read_text())['episode_id']: json.loads(p.read_text())
            for p in Path(d).glob('*.json') if p.name not in AU.NON_RECORD}


def check(spec_path, sources_path, repo=REPO, expect_scene_records=200):
    spec, src = json.loads(Path(spec_path).read_text()), json.loads(Path(sources_path).read_text())
    probs = []
    if set(src) != {SCENE_KEY}:
        probs.append(f'sources must hold only {SCENE_KEY}, got {sorted(src)}')
    s = src.get(SCENE_KEY, {})
    ep_dir, lat_dir = repo / s.get('episodes', '?'), repo / s.get('latents', '?')
    if not ep_dir.is_dir() or not lat_dir.is_dir():
        return {'ready': False, 'problems': probs + [f'missing dirs {ep_dir} / {lat_dir}'], 'folds': []}
    comp = lat_dir / 'COMPLETE.json'
    if not comp.exists() or json.loads(comp.read_text()).get('episodes') != expect_scene_records:
        probs.append(f'latents COMPLETE.json missing or episodes != {expect_scene_records}')
    lc, ec = json.loads((lat_dir / 'contract.json').read_text()), json.loads((ep_dir / 'contract.json').read_text())
    if lc.get('token_contract') != 49 or lc.get('version') != 'rollout_latents_v2':
        probs.append(f'latents contract {lc.get("version")} token_contract {lc.get("token_contract")} (need v2 / 49)')
    full_sha = ec['source_contract_sha256'] if ec.get('kind') == 'fold27_compact_v1' else AU.sha(ep_dir / 'contract.json')
    if lc.get('episodes_contract_sha256') != full_sha:
        probs.append('latents contract episodes_contract_sha256 != compact source contract (finetune_fold would refuse)')
    eps, lats = index(ep_dir), index(lat_dir)
    bad_tc = sorted(e for e, r in lats.items() if r.get('token_contract') != 49)
    if bad_tc:
        probs.append(f'{len(bad_tc)} latent records without token_contract 49')
    folds = [f for f in spec['folds'] if f'{f["instruction"]}|s{f["scene"]}' == SCENE_KEY]
    if len(folds) != 3:
        probs.append(f'{len(folds)} bread folds in spec (expected 3)')
    rep = []
    for f in folds:
        fp = []
        fit = [eid_of(f, r) for r in f['fit']]; ev = [eid_of(f, r) for r in f['eval']]
        if len(fit) != 160 or any(r['j'] == f['target_j'] for r in f['fit']):
            fp.append('fit is not 160 rows with j != target_j')
        if len(ev) != 40 or any(r['j'] != f['target_j'] for r in f['eval']):
            fp.append('eval is not 40 rows with j == target_j')
        if set(fit) & set(ev):
            fp.append('fit and eval share episodes')
        for role, rows in (('fit', f['fit']), ('eval', f['eval'])):
            for r in rows:
                e = eid_of(f, r)
                if e not in eps or e not in lats:
                    fp.append(f'{role} {e}: compact {e in eps} latent {e in lats}'); continue
                if not (r['pkl_sha256'] == eps[e].get('pkl_sha256') == lats[e].get('pkl_sha256')):
                    fp.append(f'{role} {e}: pkl sha linkage')
        rep.append({'fold': f['fold_id'], 'target_j': f['target_j'], 'fit': len(fit), 'eval': len(ev), 'problems': fp[:20],
                    'n_problems': len(fp)})
        probs += [f'{f["fold_id"]}: {p}' for p in fp[:5]]
    return {'ready': not probs, 'problems': probs, 'folds': rep, 'trainings': 3 * len(folds)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, default=REPO / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json')
    ap.add_argument('--sources', type=Path, required=True)
    a = ap.parse_args(argv)
    r = check(a.spec, a.sources)
    print(json.dumps(r, indent=1))
    return 0 if r['ready'] else 3


if __name__ == '__main__':
    sys.exit(main())
