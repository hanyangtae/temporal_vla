"""Host-matched GPU rollout-latent job (parity smoke, then conversion) — called through run_latents.sh (harness guard).

env (set by run_latents.sh / groot_launch_guard): GPUS=<one gpu>, HARNESS_MACHINE=kanu|srv48|srv50, HARNESS_RECEIPT
args: --episodes <full extract dir for ONE scene, staged on this host> --out <latents dir> [--smoke-only]

kanu : docker model image (same image/mounts as common_qam/run_extract.py), repo ro at /temporal_vla, episodes ro at
       /episodes, out rw at /out, cache ro at /cache -> rollout_latents.py --data-root /cache/datasets
srv  : host conda python (docs/05: ~/miniconda3/envs/lerobot_050_groot, SERVE_PYTHONPATH=<repo>/lerobot/src),
       --data-root = scripts.path_setup.DATA_ROOT of that host
Steps: (1) --parity-smoke unless <out>/PARITY_SMOKE.json exists with a passing record (rollout_latents itself refuses a
provenance mismatch, i.e. other host/device/code); (2) conversion (resumable; device_gate = same collection host).
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ADAPTER = REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'
REL = 'scripts/rl2_vla/groot_fold27_20261028/trqam/rollout_latents.py'
HOST_PY = Path.home() / 'miniconda3/envs/lerobot_050_groot/bin/python'
CKPT_REL = 'huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/c484448aba1a9b60a04c9b0ca117241518ea69f3/gr00t_n1-5/multitask_learning/checkpoint-120000'
DEMO_REL = 'robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot'


class LatentLaunchError(RuntimeError):
    pass


def preflight(episodes, out, data_root, machine):
    missing = [str(p) for p in (episodes / 'contract.json', data_root / CKPT_REL, data_root / DEMO_REL,
                                REPO / 'outputs/analysis/groot_robocasa_lineage_20261021/latent_reconstruction_v2/result.json') if not p.exists()]
    if missing:
        raise LatentLaunchError(f'BLOCKED on {machine}: missing {missing}')
    recs = [p for p in episodes.glob('*.json') if p.name not in ('contract.json', 'COMPLETE.json')]
    if not recs:
        raise LatentLaunchError(f'no episode records in {episodes}')
    machines = {json.loads(p.read_text())['episode_id'].split('/')[1] for p in recs[:50]}
    want = {'kanu': 'kanu', 'srv48': 'worker1', 'srv50': 'worker2'}[machine]
    if machines != {want}:
        raise LatentLaunchError(f'{episodes} holds records collected on {machines}, this host is {want}; stage per collection host')
    if shutil.disk_usage(out.parent if not out.exists() else out).free < 5 * 1024 ** 3:
        raise LatentLaunchError('output filesystem reserve 5 GiB')
    return len(recs)


def build_cmd(machine, gpu, episodes, out, mode_args, *, image=None, cache_root=None, data_root=None):
    env_pass = ['HARNESS_RECEIPT', 'HARNESS_MACHINE']
    if machine == 'kanu':
        name = f'fold27-latents-{out.name}-{os.getpid()}'
        cmd = ['docker', 'run', '--rm', '--pull=never', '--log-driver', 'none', '--name', name, '--hostname', 'kanu',
               '--gpus', f'device={gpu}', '--network', 'none', '--shm-size', '1g', '--tmpfs', '/tmp:rw,size=1g',
               '--entrypoint', 'python', '-v', f'{REPO}:/temporal_vla:ro', '-v', f'{episodes}:/episodes:ro', '-v', f'{out}:/out:rw',
               '-v', f'{cache_root}:/cache:ro', '-v', '/home/dongkyu/.cache/huggingface:/home/dongkyu/.cache/huggingface:ro',
               '-e', 'HF_HOME=/cache/datasets/huggingface', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
               '-e', 'HF_MODULES_CACHE=/tmp/hf_modules', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=4',
               '-e', 'PDK_LOG_FILE=/out/module.log', '-e', 'PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla',
               *sum((['-e', k] for k in env_pass), []), '-w', '/temporal_vla', image, REL,
               '--out', '/out', '--data-root', '/cache/datasets', *mode_args(Path('/episodes'))]
        return cmd, None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
               PDK_LOG_FILE=str(out / 'module.log'),
               PYTHONPATH=f'{REPO}/lerobot/src:{REPO}/src/policies/Isaac-GR00T:{REPO}')
    return [str(HOST_PY), str(REPO / REL), '--out', str(out), '--data-root', str(data_root), *mode_args(episodes)], env


def smoke_passed(out):
    p = out / 'PARITY_SMOKE.json'
    return p.exists() and json.loads(p.read_text()).get('passed') is True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke-only', action='store_true')
    ap.add_argument('--expect-records', type=int, help='required for conversion (e.g. 200 per scene); smoke ignores it')
    a = ap.parse_args()
    if not a.smoke_only and not a.expect_records:
        raise LatentLaunchError('--expect-records required for conversion (no partial-scene latents)')
    if not os.environ.get('HARNESS_RECEIPT'):
        raise LatentLaunchError('run through run_latents.sh (groot harness guard)')
    machine, gpu = os.environ['HARNESS_MACHINE'], int(os.environ['GPUS'])
    episodes, out = a.episodes.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(REPO), str(ADAPTER)]
    from scripts.path_setup import CACHE_ROOT, DATA_ROOT
    data_root = Path('/cache/datasets') if machine == 'kanu' else Path(DATA_ROOT)
    n = preflight(episodes, out, Path(DATA_ROOT), machine)
    if not a.smoke_only and n != a.expect_records:
        raise LatentLaunchError(f'{episodes}: {n} records staged != expected {a.expect_records}; finish staging first')
    image = None
    if machine == 'kanu':
        from paired_eval import MODEL_IMAGE as image
    steps = [] if smoke_passed(out) else [('parity_smoke', lambda ep: ['--parity-smoke'])]
    if not a.smoke_only:
        steps.append(('convert', lambda ep: ['--episodes', str(ep), '--expect-records', str(a.expect_records)]))
    for name, mode in steps:
        cmd, env = build_cmd(machine, gpu, episodes, out, mode, image=image, cache_root=CACHE_ROOT, data_root=data_root)
        k = 1 + sum(1 for _ in out.glob(f'{name}.*.log'))
        with open(out / f'{name}.{k}.log', 'x') as f:
            f.write('# ' + ' '.join(cmd) + '\n'); f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        if rc:
            raise LatentLaunchError(f'{name} rc={rc} (log {out}/{name}.{k}.log)')
    if not a.smoke_only and not (out / 'COMPLETE.json').exists():
        raise LatentLaunchError('conversion returned 0 without COMPLETE.json')
    print(json.dumps({'machine': machine, 'gpu': gpu, 'records': n, 'smoke': smoke_passed(out),
                      'complete': (out / 'COMPLETE.json').exists()}))


if __name__ == '__main__':
    try:
        main()
    except LatentLaunchError as e:
        print('LATENT_LAUNCH_ERROR', e, file=sys.stderr); sys.exit(2)
