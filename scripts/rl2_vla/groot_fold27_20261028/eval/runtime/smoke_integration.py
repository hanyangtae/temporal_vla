"""INTEGRATION SMOKE ONLY — never production. Prepares a smoke manifest + jobs file under SMOKE_ROOT.

Allows what production refuses: the 20-step smoke actor (diagnostic), any MIL checkpoint (e.g. shared48),
row subsets, and the smoke arms smoke_base (MIL scored, no QAM), smoke_always (QAM every inference),
smoke_gated (MIL-gated QAM). Production manifests/launches/verification reject every one of these.

    # 1) smoke actor (CPU, .venvs/rl2_iid, JAX_PLATFORMS=cpu)
    restore_actor.py smoke --run outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_smoke --step 20 \
        --out outputs/eval/robocasa/groot_n15/fold27_mil_qam_smoke_20261030/actors/smoke20
    # 2) manifest + jobs (stdlib)
    smoke_integration.py --fold OpenDrawer-left_s2_j2 --mil-run task_classification/output/share/grid_n10_n19/runs/cell-n10-drawer_left_s2/attn/raw/s0 \
        --smoke-actor outputs/eval/robocasa/groot_n15/fold27_mil_qam_smoke_20261030/actors/smoke20 --rows 0
    # 3) launch (prints the exact command)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from build_manifest import build  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True); ap.add_argument('--mil-run', type=Path, required=True)
    ap.add_argument('--smoke-actor', type=Path, required=True)
    ap.add_argument('--rows', default='0', help='comma list of noise indices (subset of n0..39)')
    ap.add_argument('--arms', nargs='+', default=['smoke_base', 'smoke_always', 'smoke_gated'])
    ap.add_argument('--tag', default='s1', help='smoke run tag (new tag = new manifest dir)')
    a = ap.parse_args(argv)
    rows = [int(x) for x in a.rows.split(',')]
    C.need(set(a.arms) <= set(C.SMOKE_ARMS), 'smoke arms only')
    actors = {arm: (a.smoke_actor if C.SMOKE_ARMS[arm][1] else None) for arm in a.arms}
    root = C.R / C.SMOKE_ROOT / a.tag
    path, m = build(a.fold, a.mil_run, actors, root, smoke=True, rows_subset=rows)
    jobs = root / f'{a.fold}.jobs.txt'
    text = ''.join(f'{C.rel(path)} {arm}\n' for arm in sorted(a.arms))
    C.need(not jobs.exists() or jobs.read_text() == text, f'{jobs} exists with different content')
    jobs.write_text(text)
    gpus = 'kanu: GPUS=<free g>' if m['host'] == 'kanu' else f'{m["host"]}: GPUS=<free g> + HARNESS_COORDINATOR_SSH/SCRIPT'
    print(json.dumps({'SMOKE_ONLY': True, 'manifest': str(path), 'jobs': str(jobs), 'host': m['host'], 'rows': rows,
                      'launch': f'HARNESS_MACHINE={m["host"]} HARNESS_SESSION=<owner> GPUS=<g> SERVES_PER_GPU={min(len(a.arms), C.caps(m["host"])[1])} '
                                f'bash scripts/rl2_vla/groot_fold27_20261028/eval/runtime/launch.sh --jobs {C.rel(jobs)} --rows {a.rows}',
                      'note': gpus}, indent=1))


if __name__ == '__main__':
    try:
        main()
    except C.RuntimeContractError as e:
        print('SMOKE_REFUSED', e, file=sys.stderr); sys.exit(2)
