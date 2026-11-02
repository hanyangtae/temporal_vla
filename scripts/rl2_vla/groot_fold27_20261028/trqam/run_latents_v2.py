"""Launcher for rollout_latents_v2.py (explicit token contract). The frozen run_latents.py is reused for its preflight
and per-host command builder (kanu docker model image / srv host conda lerobot_050_groot); only the target script is
swapped and the token contract is threaded through. Main runs it under the groot harness (run_latents_v2.sh).

  GPUS=<g> HARNESS_MACHINE=srv50 HARNESS_SESSION=<s> bash run_latents_v2.sh --token-contract 49 \
      --episodes <full_episodes/srv50_bread_s4> --out <new dir> [--smoke-only | --diagnose-only | --expect-records 200]
Steps: preflight (+ first records' last_block_d0 dtype/shape must match the contract) -> --diagnose (optional,
informational) -> --parity-smoke unless PARITY_SMOKE_V2.json passed for the same contract -> conversion.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_latents as V1R          # noqa: E402  (frozen; preflight / build_cmd reused)
import rollout_latents_v2 as R2    # noqa: E402

REL2 = 'scripts/rl2_vla/groot_fold27_20261028/trqam/rollout_latents_v2.py'


def peek_records(episodes, contract, k=3):
    """dtype/shape of last_block_d0 in the first k records must already match the contract (conversion re-checks all)."""
    recs = sorted(p for p in Path(episodes).glob('*.json') if p.name not in ('contract.json', 'COMPLETE.json'))[:k]
    seen = []
    for jp in recs:
        rec = json.loads(jp.read_text())
        with np.load(Path(episodes) / rec['npz']) as z:
            if 'last_block_d0' not in z.files:
                raise SystemExit(f'{jp.name}: no last_block_d0')
            lb = z['last_block_d0']
        R2.check_record(lb, contract)
        seen.append({'record': jp.name, 'shape': list(lb.shape), 'dtype': str(lb.dtype)})
    if not seen:
        raise SystemExit(f'no records in {episodes}')
    return seen


def smoke_passed(out, contract):
    p = Path(out) / 'PARITY_SMOKE_V2.json'
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    return (d.get('provenance') or {}).get('token_contract') == contract and (d.get('verdict') or {}).get('passed') is True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-contract', type=int, choices=(16, 49), required=True)
    ap.add_argument('--episodes', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke-only', action='store_true'); ap.add_argument('--diagnose-only', action='store_true')
    ap.add_argument('--diagnose', action='store_true', help='also write DIAG_V2.json before the smoke (informational)')
    ap.add_argument('--expect-records', type=int)
    a = ap.parse_args()
    if not (a.smoke_only or a.diagnose_only) and not a.expect_records:
        raise SystemExit('--expect-records required for conversion')
    if not os.environ.get('HARNESS_RECEIPT'):
        raise SystemExit('run through run_latents_v2.sh (groot harness guard)')
    machine, gpu = os.environ['HARNESS_MACHINE'], int(os.environ['GPUS'])
    episodes, out = a.episodes.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    R2.guard_out(out)
    sys.path[:0] = [str(V1R.REPO), str(V1R.ADAPTER)]
    from scripts.path_setup import CACHE_ROOT, DATA_ROOT
    data_root = Path('/cache/datasets') if machine == 'kanu' else Path(DATA_ROOT)
    n = V1R.preflight(episodes, out, Path(DATA_ROOT), machine)
    peek = peek_records(episodes, a.token_contract)
    if not (a.smoke_only or a.diagnose_only) and n != a.expect_records:
        raise SystemExit(f'{episodes}: {n} records staged != expected {a.expect_records}')
    image = None
    if machine == 'kanu':
        from paired_eval import MODEL_IMAGE as image
    tc = ['--token-contract', str(a.token_contract)]
    steps = []
    if a.diagnose or a.diagnose_only:
        steps.append(('diagnose', lambda ep: [*tc, '--diagnose']))
    if not a.diagnose_only and not smoke_passed(out, a.token_contract):
        steps.append(('parity_smoke_v2', lambda ep: [*tc, '--parity-smoke']))
    if not (a.smoke_only or a.diagnose_only):
        steps.append(('convert_v2', lambda ep: [*tc, '--episodes', str(ep), '--expect-records', str(a.expect_records)]))
    V1R.REL = REL2                                     # build_cmd reads the module global at call time
    for name, mode in steps:
        cmd, env = V1R.build_cmd(machine, gpu, episodes, out, mode, image=image, cache_root=CACHE_ROOT, data_root=data_root)
        k = 1 + sum(1 for _ in out.glob(f'{name}.*.log'))
        with open(out / f'{name}.{k}.log', 'x') as f:
            f.write('# ' + ' '.join(cmd) + '\n'); f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        if rc:
            raise SystemExit(f'{name} rc={rc} (log {out}/{name}.{k}.log)')
    print(json.dumps({'machine': machine, 'gpu': gpu, 'records': n, 'token_contract': a.token_contract, 'peek': peek,
                      'smoke': smoke_passed(out, a.token_contract), 'complete': (out / 'COMPLETE.json').exists()}))


if __name__ == '__main__':
    main()
