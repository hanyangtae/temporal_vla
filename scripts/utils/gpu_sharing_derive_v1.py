#!/usr/bin/env python3
"""Derive a SESSION-SPECIFIC kind-v1 sharing contract from an existing OFFICIAL kind-v1 contract (new; stdlib only;
gpu_sharing_authorization.py unchanged and used for every check).

Why: `gpu_sharing_authorization.py prepare` (v1) requires the GPU to hold EXACTLY the approved foreign PID. While our own
eval serves run on the same GPU under the official eval contract it refuses (correctly: it must never pin our processes
as foreign). The training contract needs the SAME pinned foreign identity, another session and the training budget.
This tool therefore never inspects, pins or approves any other process; it re-proves the pinned identity LIVE and copies
the source contract's pinned entry verbatim:
  source   SA.load OK, kind v1, unexpired, hostname/boot == live, exactly ONE pinned process, allowed_kind eval
  live     hostname, boot id and GPU UUID == source == main's expectations; the pinned PID is in NVML's compute apps with
           used <= its max_used_mib; /proc identity (uid, start ticks, cmdline sha256) == source == expectations
  budget   per_serve_mib >= --measured-peak-mib (a measured own peak; never below it), basis text required;
           reserve_mib >= source reserve_mib (never weaker); max_serves within policy
  expiry   min(now + --expires-in-h (<= 72), source expires_at): the derived contract never outlives its source
Writes the new contract with O_EXCL and re-validates it with SA.load. Other processes on the GPU (e.g. our eval serves)
are ignored here and are NOT approved: the harness/guardian/coordinator checks with this contract still require the GPU
to hold only the pinned process (plus our own proven training under the guardian).

    python3 gpu_sharing_derive_v1.py --source <official v1 json> --session S --max-serves 1 --per-serve-mib 7168 \
        --per-serve-basis measured_4584_x1.5_rounded_7168 --measured-peak-mib 4584 --reserve-mib 16384 --expires-in-h 24 \
        --expect-pid P --expect-uid U --expect-start-ticks T --expect-cmdline-sha256 H --expect-boot-id B \
        --expect-gpu-uuid GPU-... --approval '<user statement>' --out <new path>
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sharing_authorization as SA    # noqa: E402  (unchanged)

need = SA.need


def derive(a, probe=None):
    probe = probe or SA.Probe()
    src = SA.load(a.source)
    need(src.get('kind') == SA.KIND, 'source must be an official kind-v1 contract')
    need(probe.now() < float(src['expires_at']), 'source contract expired')
    need(src['allowed_kind'] == 'eval', 'source allowed_kind must be eval')
    need(len(src['foreign']) == 1, 'source must pin exactly one foreign process')
    f = src['foreign'][0]
    exp = {'pid': a.expect_pid, 'uid': a.expect_uid, 'start_ticks': a.expect_start_ticks, 'cmdline_sha256': a.expect_cmdline_sha256}
    need(all(f[k] == v for k, v in exp.items()), f'source pinned identity {[f[k] for k in exp]} != expected {list(exp.values())}')
    need(probe.hostname() == src['hostname'], 'not on the source contract host')
    need(probe.boot_id() == src['boot_id'] == a.expect_boot_id, 'boot id differs (live / source / expected)')
    g = probe.gpu(int(src['gpu']))
    need(g['uuid'] == src['gpu_uuid'] == a.expect_gpu_uuid, 'GPU UUID differs (live / source / expected)')
    app = [x for x in g['apps'] if x['pid'] == int(f['pid'])]
    need(len(app) == 1, f'pinned PID {f["pid"]} is not a compute process on GPU{src["gpu"]} now')
    need(app[0]['used_mib'] <= int(f['max_used_mib']), f'pinned PID uses {app[0]["used_mib"]} MiB > max_used_mib {f["max_used_mib"]}')
    live = probe.proc(int(f['pid']))
    need(live is not None and all(live[k] == f[k] for k in ('uid', 'start_ticks', 'cmdline_sha256')),
         'live /proc identity of the pinned PID differs (PID reuse / different process)')
    need(SA.matches_identity(src, int(f['pid']), probe), 'pinned identity check (matches_identity) failed')
    need(a.per_serve_basis.strip() != '', 'per_serve_basis (measurement statement) required')
    need(a.per_serve_mib >= a.measured_peak_mib > 0, f'per_serve_mib {a.per_serve_mib} below measured peak {a.measured_peak_mib}')
    need(a.reserve_mib >= int(src['reserve_mib']), f'reserve_mib {a.reserve_mib} weaker than source {src["reserve_mib"]}')
    need(0 < a.expires_in_h <= 72, 'expiry must be <= 72 h')
    need(a.session != src['session'], 'derive is for a different session; reuse the source contract for its own session')
    now = probe.now()
    body = {'kind': SA.KIND, 'machine': src['machine'], 'hostname': src['hostname'], 'gpu': src['gpu'], 'gpu_uuid': src['gpu_uuid'],
            'session': a.session, 'experiment': src['experiment'], 'allowed_kind': 'eval', 'max_serves': a.max_serves,
            'per_serve_mib': a.per_serve_mib, 'per_serve_basis': a.per_serve_basis, 'measured_peak_mib': a.measured_peak_mib,
            'reserve_mib': a.reserve_mib, 'expires_at': min(now + a.expires_in_h * 3600, float(src['expires_at'])),
            'boot_id': src['boot_id'], 'prepared_at': now, 'approval': a.approval, 'foreign': [dict(f)],
            'derived_from': {'path': str(Path(a.source).resolve()), 'sha256': hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
                             'session': src['session'], 'expires_at': src['expires_at']},
            'observed': {'free_mib': g['free_mib'], 'total_mib': g['total_mib'], 'approved_used_mib': app[0]['used_mib'],
                         'note': 'other compute processes present at derivation are NOT approved by this contract'}}
    pol = json.loads(SA.POLICY.read_text())['machines']
    need(1 <= a.max_serves <= pol[src['machine']]['max_serves_per_gpu'], 'max_serves outside the GPU policy')
    with open(a.out, 'x') as fh:
        fh.write(json.dumps(body, indent=1))
    SA.load(a.out)
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, t in (('source', Path), ('session', str), ('max-serves', int), ('per-serve-mib', int), ('per-serve-basis', str),
                 ('measured-peak-mib', int), ('reserve-mib', int), ('expires-in-h', float), ('expect-pid', int), ('expect-uid', int),
                 ('expect-start-ticks', int), ('expect-cmdline-sha256', str), ('expect-boot-id', str), ('expect-gpu-uuid', str),
                 ('approval', str), ('out', Path)):
        ap.add_argument('--' + k, type=t, required=True)
    a = ap.parse_args(argv)
    try:
        r = derive(a)
    except (SA.AuthError, OSError, KeyError, ValueError) as e:
        print(f'[sharing-derive] REFUSED: {e}', file=sys.stderr); return 2
    print(json.dumps({k: r[k] for k in ('session', 'max_serves', 'per_serve_mib', 'reserve_mib', 'expires_at', 'derived_from')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
