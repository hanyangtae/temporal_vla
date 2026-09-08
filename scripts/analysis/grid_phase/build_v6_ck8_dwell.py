"""원격 자료 옆에서 scene별 ck8 detector → operator 순으로 만들고 작은 결과만 공개."""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import torch


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--episodes', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    scripts = repo/'scripts/analysis/grid_phase'
    a.out.mkdir(parents=True, exist_ok=True)
    released = a.out/'released'
    released.mkdir(exist_ok=True)
    rows = list(csv.DictReader(a.targets.open(), delimiter='\t'))
    index = list(csv.DictReader((repo/'configs/collect/n15_grid_v6_scene_jitter/index_v6_complete_cells.tsv').open(), delimiter='\t'))
    bundle = a.store/'ae_k8/ae_bundle_k8.npz'
    bundle_dest = released/'outputs/analysis/grid_phase/ae_k8/ae_bundle_k8.npz'
    bundle_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle, bundle_dest)
    ready = {'detectors': [], 'operators': [], 'failed': None, 'complete': False}
    ready_path = released/'ready.json'
    atomic_json(ready_path, ready)
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    def run(script, args):
        subprocess.run([sys.executable, str(scripts/script), *map(str, args)], check=True, env=env)
    try:
        for r in rows:
            slug, scene, j = r['slug'], int(r['scene_idx']), int(r['jitter_idx'])
            stem = f'{slug}__s{scene}'
            key = f'{slug}:{scene}:{j}'
            prepared = a.out/'prepared'/f'{stem}.npz'
            run('prepare_ck8_scene.py', ['--shard', a.store/'segA_scene'/f'{stem}.npz',
                '--bundle', bundle, '--slug', slug, '--out', prepared])
            with np.load(prepared, allow_pickle=False) as z:
                # 50 episodes and their original outcomes, not merely 50 record groups.
                ep = z['ep_id']; starts = [np.flatnonzero(ep == e)[0] for e in np.unique(ep)]
                observed = {(int(z['jitter'][i]), int(z['noise'][i])): int(z['succ'][i]) for i in starts}
                expected = {(int(v['jitter_idx']), int(v['noise_idx'])): int(v['success']) for v in index
                            if v['grid_instruction'] == r['instruction'] and int(v['scene_idx']) == scene}
                assert len(observed) == 50 and observed == expected, (key, 'shard/index mismatch')
            detout = a.out/'detector_build'/stem
            run('failure_detector_sim.py', ['--shard-dir', prepared.parent, '--shards', stem,
                '--out', detout, '--arm', 'loko-cell', '--loko-cells-tsv', a.targets,
                '--models', 'lstm', '--alphas', '0.1', '--truncate-train', 'phase-ck8',
                '--min-pool-fail', '1', '--min-calib-succ', '9', '--cp-folds', '0',
                '--loko-train-pool', 'other', '--seed', '0', '--threads', '8', '--quiet'])
            ckrel = Path('loko')/stem/f's{scene}'/f'j{j}'/f'detector_pertask_lstm_{stem}.pt'
            ck = torch.load(detout/ckrel, map_location='cpu', weights_only=False)
            assert ck['loko']['train_pool'] == 'other' and len(ck['loko']['train_ep_ids']) == 40
            assert ck['phase_source']['kind'] == 'ck8' and ck['truncate']['mode'] == 'phase-ck8'
            assert ck['feature']['denoise_idx'] == 3
            dest = released/'outputs/analysis/grid_phase/detector_v6_ck8_dwell'/ckrel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(detout/ckrel, dest)
            ready['detectors'].append(key)
            atomic_json(ready_path, ready)
            run('fit_ck8_dwell_setm.py', ['--shard', prepared, '--slug', slug, '--target', j,
                '--out', a.out/'operator_build', '--tag', 'v6_ck8dwell'])
            for variant in ('instr_setm_v6_ck8dwell_ck8', 'instr_setm_v6_ck8dwell_ck8_plain'):
                rel = Path(variant)/slug/f's{scene}'/f'j{j}'
                shutil.copytree(a.out/'operator_build'/rel, released/'outputs/steer/online_pipe_v4_pilot'/rel)
            diagnostic = a.out/'operator_build/diagnostics'/f'{slug}_s{scene}_j{j}.json'
            dd = released/'outputs/analysis/v6_preflight_audit_20260908/fit_diagnostics'
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(diagnostic, dd/diagnostic.name)
            ready['operators'].append(key)
            atomic_json(ready_path, ready)
        ready['complete'] = True
        atomic_json(ready_path, ready)
    except Exception as e:
        ready['failed'] = repr(e)
        atomic_json(ready_path, ready)
        raise


if __name__ == '__main__':
    main()
