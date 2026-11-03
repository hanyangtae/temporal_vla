"""`eval_pipeline.py wait-resources` with the one-time sharing contract honoured (eval_pipeline.py unchanged).

Without HARNESS_SHARING_AUTH this is exactly eval_pipeline.wait_resources (ledger admissibility + gpu_busy: any compute
process = busy). With it, a GPU counts as free only if scripts/utils/gpu_sharing_authorization.authorized_processes
passes for (host, [gpu], session, kind=eval, serves): the pinned approved process(es) may be present, nothing else, and
memory headroom must hold. Any refusal (unapproved PID, PID reuse, expiry, UUID/boot/host mismatch, low memory, query
failure) keeps the GPU busy -> the follower keeps waiting (fail closed); the harness re-checks the same contract itself.

    python wait_resources_shared.py --host srv50 --gpus 3 --serves 2 --session fold27-v2-eval
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[4] / 'scripts/utils')]
import eval_pipeline as EP                 # noqa: E402  (unchanged)
import gpu_sharing_authorization as SA     # noqa: E402


def busy_fn_for(host, serves, session, auth_path, probe=None):
    if not auth_path:
        return EP.gpu_busy

    def busy(gpus):
        try:
            SA.authorized_processes(SA.load(auth_path), host, gpus, session, 'eval', serves, probe)
            return []
        except (SA.AuthError, OSError, ValueError, KeyError) as e:
            print(json.dumps({'time': time.time(), 'sharing_refused': str(e)[:300]}), flush=True)
            return list(gpus)
    return busy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True); ap.add_argument('--gpus', required=True); ap.add_argument('--serves', type=int, required=True)
    ap.add_argument('--session', required=True); ap.add_argument('--poll', type=float, default=120)
    a = ap.parse_args(argv)
    gpus = [int(g) for g in a.gpus.split(',')]
    EP.wait_resources(a.host, gpus, a.serves, a.session, a.poll,
                      busy_fn=busy_fn_for(a.host, a.serves, a.session, os.environ.get('HARNESS_SHARING_AUTH')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
