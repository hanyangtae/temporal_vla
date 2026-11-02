"""Isolated RoboCasa env container for srv48/srv50 eval (new version; common.py / run_eval_arm.py untouched).

The shared container `robocasa` mounts the ORIGINAL checkout at /temporal_vla and renders on its own fixed GPU. Fold27 eval
must exec into a container that mounts the isolated deploy and renders on the harness-reserved GPU, so:
  plan  (read-only) docker inspect the original container, print the exact `docker run` argv for fold27-iso-env-gpu<G>:
        same image ID, same Config.Env/User/WorkingDir/network/ipc/shm, same /cache bind, /temporal_vla <- <deploy> rw,
        NVIDIA_VISIBLE_DEVICES=<G> + --gpus device=<G>, entrypoint `sleep infinity` (the original's VNC entrypoint would bind
        the same host-network ports), labels fold27.deploy/fold27.gpu/fold27.source_image. Main runs the printed argv.
  gate  verify a container before/while it is used: running, image == expected, exactly one /temporal_vla mount whose
        resolved source == deploy and RW, NVIDIA_VISIBLE_DEVICES == GPU, labels match. Prints the image ID.

    python3 iso_env_container.py plan --from robocasa --gpu 2 --deploy /tmp/temporal-vla-fold27-production-20261102
    python3 iso_env_container.py gate --name fold27-iso-env-gpu2 --gpu 2 --deploy <deploy> [--image sha256:...]
"""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path

ISO_PREFIX = 'fold27-iso-env-gpu'
REPO_IN = '/temporal_vla'


class GateError(RuntimeError):
    pass


def need(ok, msg):
    if not ok:
        raise GateError(msg)


def docker_inspect(name, run=subprocess.run):
    r = run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
    need(r.returncode == 0, f'docker inspect {name} failed: {r.stderr.strip()}')
    return json.loads(r.stdout)[0]


def iso_name(gpu):
    return f'{ISO_PREFIX}{int(gpu)}'


def plan(src, gpu, deploy):
    """src = docker inspect dict of the original container. Returns the docker run argv (list)."""
    deploy = os.path.realpath(deploy)
    hc, cfg = src['HostConfig'], src['Config']
    binds = hc.get('Binds') or []
    repo = [b for b in binds if b.split(':')[1] == REPO_IN]
    need(len(repo) == 1, f'original container must bind exactly one {REPO_IN}: {binds}')
    need(os.path.realpath(repo[0].split(':')[0]) != deploy, 'original container already mounts the deploy; nothing to isolate')
    others = [b for b in binds if b.split(':')[1] != REPO_IN]
    env = [e for e in cfg.get('Env') or [] if not e.startswith('NVIDIA_VISIBLE_DEVICES=')] + [f'NVIDIA_VISIBLE_DEVICES={int(gpu)}']
    argv = ['docker', 'run', '-d', '--name', iso_name(gpu),
            '--label', f'fold27.deploy={deploy}', '--label', f'fold27.gpu={int(gpu)}', '--label', f'fold27.source_image={src["Image"]}',
            '--network', hc.get('NetworkMode') or 'default', '--ipc', hc.get('IpcMode') or 'private',
            '--shm-size', str(hc.get('ShmSize') or 67108864), '--gpus', f'device={int(gpu)}',
            '--user', cfg.get('User') or 'root', '--workdir', cfg.get('WorkingDir') or REPO_IN]
    for b in others:
        argv += ['-v', b]
    argv += ['-v', f'{deploy}:{REPO_IN}:rw']
    for e in env:
        argv += ['-e', e]
    argv += ['--entrypoint', 'sleep', src['Image'], 'infinity']
    return argv


def gate(info, gpu, deploy, image=None):
    deploy = os.path.realpath(deploy)
    need(info['State']['Running'], f'{info["Name"]} not running')
    if image:
        need(info['Image'] == image, f'image {info["Image"]} != expected {image}')
    mounts = [m for m in info['Mounts'] if m['Destination'] == REPO_IN]
    need(len(mounts) == 1, f'expected exactly one {REPO_IN} mount, got {len(mounts)}')
    need(os.path.realpath(mounts[0]['Source']) == deploy, f'{REPO_IN} <- {mounts[0]["Source"]} != deploy {deploy}')
    need(mounts[0].get('RW') is True, f'{REPO_IN} mount is not RW')
    env = dict(e.split('=', 1) for e in info['Config'].get('Env') or [])
    need(env.get('NVIDIA_VISIBLE_DEVICES') == str(int(gpu)), f'NVIDIA_VISIBLE_DEVICES={env.get("NVIDIA_VISIBLE_DEVICES")} != GPU {gpu}')
    labels = info['Config'].get('Labels') or {}
    need(labels.get('fold27.deploy') == deploy and labels.get('fold27.gpu') == str(int(gpu)), f'fold27 labels mismatch: {labels}')
    return info['Image']


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan'); p.add_argument('--from', dest='src', default='robocasa'); p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--deploy', required=True)
    g = sub.add_parser('gate'); g.add_argument('--name'); g.add_argument('--gpu', type=int, required=True)
    g.add_argument('--deploy', required=True); g.add_argument('--image')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'plan':
            argv = plan(docker_inspect(a.src), a.gpu, a.deploy)
            print(json.dumps({'name': iso_name(a.gpu), 'argv': argv})); print(' '.join(shlex.quote(x) for x in argv))
        else:
            print(gate(docker_inspect(a.name or iso_name(a.gpu)), a.gpu, a.deploy, a.image))
    except GateError as e:
        print('ISO_ENV_GATE_FAILED', e, file=sys.stderr); return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
