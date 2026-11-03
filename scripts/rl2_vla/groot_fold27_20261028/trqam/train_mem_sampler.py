"""READ-ONLY GPU memory sampler for a bounded training-memory measurement (no signal, no launch, no GPU allocation).
Run it next to an existing harness-guarded smoke (smoke_v2b.sh, 200 steps, incl. the GPU resume check for basic) on a GPU
that the harness admitted EXCLUSIVELY (no other compute process), so the GPU total equals our usage. Samples NVML every
--interval s until --watch-pid exits (or --max-s), and writes a NEW json (exclusive create): samples, per-PID peak,
GPU-total peak, and the process set seen. The training session's per_serve_mib in the v2 sharing contract must be taken
from this measured peak (main decides any margin explicitly); nothing here invents a value.

    python train_mem_sampler.py --gpu <g> --watch-pid <smoke launcher pid> --out <new json> [--interval 0.5] [--max-s 3600]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def sample(gpu, run=subprocess.run):
    g = run(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
    a = run(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
    if g.returncode or a.returncode:
        return None
    uuid, used, total = [x.strip() for x in g.stdout.strip().splitlines()[0].split(',')]
    apps = {}
    for line in a.stdout.strip().splitlines():
        if line.strip():
            pid, mib = [x.strip() for x in line.split(',')]
            apps[int(pid)] = int(mib)
    return {'uuid': uuid, 'used_mib': int(used), 'total_mib': int(total), 'apps': apps}


def alive(pid):
    return Path(f'/proc/{int(pid)}').exists()


def measure(gpu, watch_pid, interval, max_s, sample_fn=sample, alive_fn=alive, now=time.time, sleep=time.sleep):
    t0, samples, peak_total, peak_pid, failed = now(), [], 0, {}, 0
    while alive_fn(watch_pid) and now() - t0 < max_s:
        s = sample_fn(gpu)
        if s is None:
            failed += 1
        else:
            samples.append({'t': now() - t0, 'used_mib': s['used_mib'], 'apps': s['apps']})
            peak_total = max(peak_total, s['used_mib'])
            for p, m in s['apps'].items():
                peak_pid[p] = max(peak_pid.get(p, 0), m)
        sleep(interval)
    return {'gpu': gpu, 'watch_pid': watch_pid, 'seconds': now() - t0, 'n_samples': len(samples), 'failed_samples': failed,
            'peak_gpu_used_mib': peak_total, 'peak_per_pid_mib': peak_pid, 'pids_seen': sorted(peak_pid),
            'complete': not alive_fn(watch_pid), 'samples': samples,
            'note': 'valid only if the harness admitted this GPU exclusively for the whole window (no foreign PID in pids_seen)'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--watch-pid', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--interval', type=float, default=0.5)
    ap.add_argument('--max-s', type=float, default=3600)
    a = ap.parse_args(argv)
    r = measure(a.gpu, a.watch_pid, a.interval, a.max_s)
    with open(a.out, 'x') as f:
        f.write(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ('peak_gpu_used_mib', 'n_samples', 'failed_samples', 'complete', 'pids_seen')}))
    return 0 if r['complete'] and r['n_samples'] else 3


if __name__ == '__main__':
    sys.exit(main())
