#!/usr/bin/env python3
"""Exact expiry extension of an EXISTING one-time GPU sharing contract (v1 or v2). New tool; gpu_sharing_authorization.py
is unchanged and its load()/Probe/classify are reused. Nothing but `expires_at` changes; everything else must stay
byte-for-byte equal in meaning (asserted), plus an appended `extensions` audit record.

    python3 scripts/utils/gpu_sharing_extend.py --auth <current contract> --add-seconds 86400 \
        --expect-expires-at 1793784546.7266493 --approval '<user statement, verbatim>' --out <NEW path> \
        [--install-over <current contract path>]

Refuses (rc 2, nothing written) unless, on the authorized host itself:
  * the current contract loads (gpu_sharing_authorization.load) and its expires_at == --expect-expires-at exactly
    (repr-equal float) and is still in the future (an expired contract is never revived -> prepare a new one);
  * boot id, hostname, GPU UUID equal the contract;
  * EVERY pinned approved process is live with identical uid + start_ticks + cmdline sha256, present on the GPU, and
    within its max_used_mib; for v2 every pinned container init is live with identical pid/start/cgroup/uid/gid/nested PID 1;
  * new expires_at = old + add_seconds (add_seconds > 0) and new expires_at - now <= 72 h (the prepare cap).
Other GPU processes (e.g. our own admitted serves) are recorded in the evidence, not judged: admission checks stay with
`gpu_sharing_authorization check` / the harness, which re-read the file at every check.
--install-over: compare-and-swap. The target must still hash to the bytes read as --auth; the original bytes are kept as
<target>.pre_extend.<sha16>.json (exclusive create) and the new contract replaces the target atomically (os.replace), so
processes that hold the PATH (harness, wait_resources_shared) see the extension. A RUNNING fold27 guardian v2/v3 holds
the contract IN MEMORY (loaded once at start): it keeps the old expiry until it is restarted.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sharing_authorization as SA   # noqa: E402

MAX_AHEAD_S = 72 * 3600


def verify_live(a, probe):
    """Evidence dict; raises SA.AuthError on any mismatch of the pinned identities."""
    SA.need(probe.hostname() == a['hostname'], f'host {probe.hostname()} != contract {a["hostname"]} (run on the host itself)')
    SA.need(probe.boot_id() == a['boot_id'], 'boot id changed since the contract (reboot): contract void')
    g = probe.gpu(int(a['gpu']))
    SA.need(g['uuid'] == a['gpu_uuid'], f'GPU{a["gpu"]} uuid {g["uuid"]} != contract {a["gpu_uuid"]}')
    used = {x['pid']: x['used_mib'] for x in g['apps']}
    pinned = []
    for f in a['foreign']:
        p = int(f['pid'])
        live = probe.proc(p)
        SA.need(live is not None, f'pinned PID {p} gone (approved task ended: nothing to extend)')
        bad = [k for k in ('uid', 'start_ticks', 'cmdline_sha256') if live[k] != f[k]]
        SA.need(not bad, f'pinned PID {p}: {bad} differ (PID reuse / different process)')
        SA.need(p in used, f'pinned PID {p} is not on GPU{a["gpu"]} (approved task ended: nothing to extend)')
        SA.need(used[p] <= int(f['max_used_mib']), f'pinned PID {p} uses {used[p]} MiB > authorized {f["max_used_mib"]}')
        pinned.append({'pid': p, 'start_ticks': live['start_ticks'], 'uid': live['uid'], 'used_mib': used[p]})
    containers = []
    for c in a.get('containers') or []:
        r = SA.container_member(c, int(c['init']['pid']), probe)
        SA.need(r is None, f'container {c["id"][:12]}: {r}')
        containers.append({'id': c['id'], 'init': c['init']})
    others = sorted(p for p in used if p not in {x['pid'] for x in pinned})
    if a.get('kind') == SA.KIND_V2:
        cl = SA.classify(a, g['apps'], probe)
        member = {p for ps in cl['containers'].values() for p in ps}
        others = [p for p in others if p not in member]
    return {'gpu_uuid': g['uuid'], 'free_mib': g['free_mib'], 'pinned_live': pinned, 'containers_live': containers,
            'other_gpu_pids_not_judged': others}


def extend(auth_path, add_s, expect_exp, approval, out, probe=None, install_over=None):
    probe = probe or SA.Probe()
    raw = Path(auth_path).read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()
    a = SA.load(auth_path)
    old = float(a['expires_at'])
    SA.need(repr(old) == repr(float(expect_exp)), f'expires_at {old!r} != expected {float(expect_exp)!r}')
    SA.need(add_s > 0, 'add-seconds must be > 0')
    now = probe.now()
    SA.need(now < old, 'contract already expired: it is never revived (prepare a new one)')
    new = old + float(add_s)
    SA.need(new - now <= MAX_AHEAD_S, f'new expiry {new - now:.0f}s ahead > {MAX_AHEAD_S}s cap')
    SA.need(approval.strip() != '', 'approval statement required')
    ev = verify_live(a, probe)
    body = json.loads(raw)
    nb = dict(body)
    nb['expires_at'] = new
    nb['extensions'] = list(body.get('extensions') or []) + [{
        'from_sha256': src_sha, 'from_path': str(auth_path), 'old_expires_at': old, 'add_seconds': float(add_s),
        'new_expires_at': new, 'extended_at': now, 'approval': approval, 'live': ev}]
    diff = sorted(k for k in set(body) | set(nb) if body.get(k) != nb.get(k))
    SA.need(diff == ['expires_at', 'extensions'], f'unexpected changed keys {diff}')
    out = Path(out)
    with open(out, 'x') as f:
        f.write(json.dumps(nb, indent=1))
    chk = SA.load(out)
    SA.need(repr(float(chk['expires_at'])) == repr(new), 'written expiry differs')
    res = {'out': str(out), 'out_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'from_sha256': src_sha,
           'old_expires_at': old, 'new_expires_at': new, 'changed_keys': diff, 'live': ev, 'installed': None}
    if install_over:
        t = Path(install_over)
        SA.need(hashlib.sha256(t.read_bytes()).hexdigest() == src_sha, 'install target changed since it was read (no swap)')
        keep = t.with_name(f'{t.name}.pre_extend.{src_sha[:16]}.json')
        with open(keep, 'xb') as f:
            f.write(raw)
        tmp = t.with_name(f'.{t.name}.extend.{os.getpid()}.tmp')
        tmp.write_bytes(out.read_bytes())
        SA.need(hashlib.sha256(t.read_bytes()).hexdigest() == src_sha, 'install target changed during install (no swap)')
        os.replace(tmp, t)
        res['installed'] = {'target': str(t), 'original_kept': str(keep), 'target_sha256': hashlib.sha256(t.read_bytes()).hexdigest()}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--auth', type=Path, required=True); ap.add_argument('--add-seconds', type=float, required=True)
    ap.add_argument('--expect-expires-at', type=float, required=True); ap.add_argument('--approval', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--install-over', type=Path)
    a = ap.parse_args(argv)
    try:
        r = extend(a.auth, a.add_seconds, a.expect_expires_at, a.approval, a.out, install_over=a.install_over)
    except (SA.AuthError, FileExistsError, OSError, ValueError, KeyError) as e:
        print(f'[sharing-extend] REFUSED: {e}', file=sys.stderr)
        return 2
    print(json.dumps(r, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
