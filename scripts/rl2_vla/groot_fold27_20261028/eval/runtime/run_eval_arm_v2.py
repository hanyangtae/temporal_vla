"""run_eval_arm with an EXPLICIT env container (srv48/srv50 docker_exec hosts). run_eval_arm.py / common.py are unchanged:
this wrapper sets common.SRV_ENV_CONTAINER (read at call time by both srv_env_preflight and collect_command's docker exec)
and wraps srv_env_preflight with iso_env_container.gate, so the per-episode image-stability check (run_eval_arm: same image
before every episode) also re-checks mount/GPU/labels.

  --env-container NAME   required on docker_exec hosts (no silent fallback to the shared 'robocasa' container)
  --expect-repo PATH     must equal this checkout's root (common.R); the container's /temporal_vla must resolve to it, RW
  --expect-image ID      optional pin (else the image seen at the first preflight is held for the whole arm)
Kanu (docker_run per episode) refuses these flags: nothing to select there.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C               # noqa: E402
import run_eval_arm as RA        # noqa: E402  (unchanged; patched attributes only)
import iso_env_container as ISO  # noqa: E402


def pop(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def configure(argv, inspect=ISO.docker_inspect):
    """Returns the remaining run_eval_arm argv after installing the container selection + gate."""
    argv = list(argv)
    name, repo, image = pop(argv, '--env-container'), pop(argv, '--expect-repo'), pop(argv, '--expect-image')
    gpu = int(argv[argv.index('--gpu') + 1])
    manifest = Path(argv[argv.index('--manifest') + 1])
    import json
    host = json.loads(manifest.read_text())['host']
    if C.HOSTS[host]['env'] != 'docker_exec':
        C.need(name is None and repo is None and image is None, f'{host}: env runs per-episode docker_run; no env container to select')
        return argv
    C.need(name, '--env-container required on docker_exec hosts (no fallback to the shared container)')
    C.need(repo and Path(repo).resolve() == C.R, f'--expect-repo {repo} != this checkout {C.R}')
    C.need(name != 'robocasa', 'the shared original container is never used for fold27 isolated eval')
    C.SRV_ENV_CONTAINER = name
    orig = RA.srv_env_preflight

    def preflight(out):
        img = orig(out)                                    # original contract: running, /temporal_vla <- C.R rw
        try:
            got = ISO.gate(inspect(name), gpu, str(C.R), image)
        except ISO.GateError as e:
            raise RuntimeError(f'isolated env gate: {e}')
        C.need(got == img, 'image changed between checks')
        return img
    RA.srv_env_preflight = preflight
    return argv


def main(argv=None):
    rest = configure(sys.argv[1:] if argv is None else argv)
    return RA.main(rest)


if __name__ == '__main__':
    sys.exit(main())
