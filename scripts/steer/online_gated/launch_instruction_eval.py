"""Local coordinator: one machine queue admitted by the shared GR00T harness.

Run under main checkout with_gpu_lease.sh (same session). No remote ledger or
reverse SSH credential is needed: coordinator brokers the remote queue locally,
verifies the actual worker hostname/GPU processes, and retains reservation on error.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--machine', choices=['kanu','worker1','worker2'], required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--main-root', type=Path, required=True)
    p.add_argument('--port-base', type=int, required=True)
    a=p.parse_args()
    root=a.main_root.resolve();repo=Path(__file__).resolve().parents[3]
    host={'kanu':None,'worker1':'AISem_48_junhyeong','worker2':'AISem_50_junhyeong'}[a.machine]
    lease={'kanu':'kanu','worker1':'srv48','worker2':'srv50'}[a.machine]
    gpus=[int(g) for g in a.gpus.split(',')];serves=2 if host is None else 6
    harness=root/'scripts/utils/groot_harness.py'
    def broker(req):
        ret=subprocess.run(['python3',str(harness),'broker'],input=json.dumps(req),text=True,capture_output=True,check=True)
        return json.loads(ret.stdout)
    def read(cmd):
        if host:cmd=['ssh',host,shlex.join(cmd)]
        return subprocess.check_output(cmd,text=True).strip()
    def gpu_empty():
        for g in gpus:
            busy=read(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader'])
            if busy:raise RuntimeError(f'{lease} GPU{g} occupied: {busy}')
    assert read(['hostname','-s'])==a.machine
    gpu_empty()
    req=dict(action='reserve',machine=lease,gpus=gpus,serves=serves,session=a.session,kind='eval',
             launcher_host=socket.gethostname(),launcher_pid=os.getpid())
    receipt=broker(req)['receipt']
    state=root/'outputs/analysis/instruction_fallback_20260910'/f'{a.machine}_reservation.json'
    state.write_text(json.dumps(dict(req,receipt=receipt),indent=2)+'\n')
    child=None
    try:
        broker(dict(req,action='verify',receipt=receipt,consumer_pid=os.getpid()))
        gpu_empty()
        runtime='outputs/analysis/instruction_fallback_20260910/runtime'
        cfg='configs/experiments/instruction_fallback_20260910'
        out='outputs/eval/robocasa/groot_n15/og_instruction_fallback_20260910/'+a.machine
        args=['--machine',a.machine,'--gpus',a.gpus,'--lease-held','--port-base',str(a.port_base),
            '--phase-source','ck8','--artifact-tag','v6_ck8dwell','--reference-labels',
            '--arms','instruction_b08,cluster_instruction_b08','--manifest',runtime+'/episodes.tsv',
            '--plan-map',str(repo/cfg/'plans.json') if not host else cfg+'/plans.json',
            '--detector-root','outputs/analysis/grid_phase/detector_v6_ck8_dwell',
            '--cluster-bundle','outputs/analysis/grid_phase/ae_k8/ae_bundle_k8.npz','--out',out]
        if host:
            cmd=['ssh','-o','ServerAliveInterval=30',host,
                 'cd ~/pkt_ws/temporal_vla && exec setsid --wait '+shlex.join(['python3','scripts/steer/online_gated/run_v6_heldout_all.py',*args])]
        else:
            cmd=['python3',str(repo/'scripts/steer/online_gated/run_v6_heldout_all.py'),'--main-root',str(root),*args]
        print(f'[harness-admitted] {lease} GPUs={gpus} slots={serves} receipt={receipt}',flush=True)
        child=subprocess.Popen(cmd,cwd=root)
        rc=child.wait()
        print(f'[queue-finished] rc={rc}',flush=True)
        return rc
    finally:
        if child is None or child.poll() is not None:
            try:gpu_empty()
            except Exception as e:print(f'[reservation-retained] {receipt} {e}',flush=True)
            else:
                broker(dict(action='release',receipt=receipt,session=a.session))
                state.write_text(json.dumps(dict(receipt=receipt,released=True),indent=2)+'\n')

if __name__=='__main__':sys.exit(main())
