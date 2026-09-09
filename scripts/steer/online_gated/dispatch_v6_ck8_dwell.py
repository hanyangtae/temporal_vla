"""모델 준비 감시 → reseed 먼저 → 나머지 4 arm. baseline replay는 예약하지 않는다."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


class _ExternalProcess:
    """Small Popen-compatible view of a process owned by a prior dispatcher."""
    def __init__(self, pid, done_check):
        self.pid, self._done_check, self.returncode = int(pid), done_check, None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            os.kill(self.pid, 0)
            stat = Path(f'/proc/{self.pid}/stat').read_text().split()
            if len(stat) > 2 and stat[2] == 'Z':
                raise ProcessLookupError
            return None
        except (OSError, FileNotFoundError, ProcessLookupError):
            self.returncode = 0 if self._done_check() else 1
            return self.returncode


def atomic_copy(src, dst):
    """Do not expose a partially copied checkpoint to concurrent preflight."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + f'.copy-{os.getpid()}')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return str(dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-root', type=Path, required=True)
    p.add_argument('--analysis-repo', required=True)
    p.add_argument('--allow-busy-local', action='store_true')
    p.add_argument('--config-dir', type=Path, default=Path('configs/experiments/v6_ck8_dwell_20260908'),
                   help='experiment config directory (default: %(default)s)')
    p.add_argument('--build-dir', type=Path, default=Path('outputs/analysis/grid_phase/v6_ck8_dwell_build'),
                   help='released build directory (default: %(default)s)')
    p.add_argument('--state-dir', type=Path, default=Path('outputs/analysis/v6_ck8_dispatch_20260908'),
                   help='dispatcher state/log directory (default: %(default)s)')
    p.add_argument('--out-root', type=Path,
                   default=Path('outputs/eval/robocasa/groot_n15/og_v6_ck8_dwell_20260908'),
                   help='evaluation output root (default: %(default)s)')
    p.add_argument('--resume-existing', action='store_true',
                   help='restore active/done stages from state.json')
    a = p.parse_args()
    root = a.main_root.resolve()
    repo = Path(__file__).resolve().parents[3]
    def under_root(path):
        return path if path.is_absolute() else root / path
    config_dir = (a.config_dir if a.config_dir.is_absolute() else repo / a.config_dir).resolve()
    build_dir = under_root(a.build_dir).resolve()
    state_dir = under_root(a.state_dir).resolve()
    out_root = under_root(a.out_root).resolve()
    helper = root/'scripts/utils/remote_compute.sh'
    remote_env = dict(os.environ, REMOTE_REPO=a.analysis_repo)
    release = build_dir/'released'
    release_rel = str(release.relative_to(root)) if release.is_relative_to(root) else str(release)
    state_dir.mkdir(parents=True, exist_ok=True)
    pid_file = state_dir/'dispatcher.pid'
    if pid_file.exists():
        old = int(pid_file.read_text())
        if old != os.getpid() and Path(f'/proc/{old}/cmdline').exists():
            if b'dispatch_v6_ck8_dwell.py' in Path(f'/proc/{old}/cmdline').read_bytes():
                raise SystemExit(f'dispatcher already running: {old}')
    pid_file.write_text(str(os.getpid())+'\n')
    log = state_dir/'dispatcher.log'
    manifest = config_dir/'episodes.tsv'
    targets = list(csv.DictReader((config_dir/'targets.tsv').open(), delimiter='\t'))
    machines = {'kanu': dict(host=None, gpus='5,6,7', lease='kanu'),
                'worker2': dict(host='AISem_50_junhyeong', gpus='2', lease='srv50')}
    machine_file = config_dir/'machines.json'
    if machine_file.is_file():
        loaded = json.loads(machine_file.read_text())
        machines = loaded.get('machines', loaded)
    active, done = {}, []
    stage_specs = lambda cfg: ([('all', 'reseed,plain_b08,jfair_b09,reseed_plain_b08,reseed_jfair_b09', 'operators')]
                               if cfg.get('combined_arms') else
                               [('reseed', 'reseed', 'detectors'),
                                ('operators', 'plain_b08,jfair_b09,reseed_plain_b08,reseed_jfair_b09', 'operators')])
    def done_check(machine, stage):
        marker = out_root/stage/'DONE.json'
        if not machines[machine].get('host'):
            return marker.is_file()
        return subprocess.run(['ssh', machines[machine]['host'], 'test', '-f',
                               'pkt_ws/temporal_vla/' + str(marker.relative_to(root))]).returncode == 0
    if a.resume_existing:
        state_path = state_dir/'state.json'
        if state_path.is_file():
            prior = json.loads(state_path.read_text())
            done.extend(prior.get('done', []))
            for machine, cfg in machines.items():
                if machine+'_all' in done or machine+'_all' in (prior.get('active') or {}):
                    cfg['combined_arms'] = True
            for key, pid in (prior.get('active') or {}).items():
                machine, stage = key.rsplit('_', 1)
                if machine in machines:
                    active[key] = (_ExternalProcess(pid, lambda m=machine, s=stage: done_check(m, s)), None)
    def report(message):
        print(message, flush=True)
        (state_dir/'state.json').write_text(json.dumps(dict(active={k:v[0].pid for k,v in active.items()},
            done=done, status=message, updated=time.time()), indent=2)+'\n')
    def call(cmd, **kw):
        return subprocess.run(cmd, check=True, **kw)
    expected_stage_count = sum(len(stage_specs(cfg)) for cfg in machines.values())
    while len(done) < expected_stage_count:
        for key, (proc, fh) in list(active.items()):
            if proc.poll() is None: continue
            if fh is not None: fh.close()
            del active[key]
            if proc.returncode:
                report(f'FAILED {key} rc={proc.returncode}; no automatic retry')
                raise SystemExit(1)
            done.append(key)
            report(f'completed {key}')
        # Retrieve only the small released artifacts, never prepared/raw shards.
        with log.open('a') as fh:
            call(['bash', str(helper), 'pull-results', release_rel], env=remote_env, stdout=fh, stderr=fh)
        ready_path = release/'ready.json'
        if not ready_path.is_file():
            report(f'waiting build ready: {ready_path}')
            time.sleep(30)
            continue
        ready = json.loads(ready_path.read_text())
        if ready['failed']:
            report(f'BUILD FAILED {ready["failed"]}')
            raise SystemExit(1)
        for machine, cfg in machines.items():
            wanted = [f'{r["slug"]}:{r["scene_idx"]}:{r["jitter_idx"]}' for r in targets if r['machine'] == machine]
            started = any(k.startswith(machine+'_') for k in [*active, *done])
            if (cfg.get('prefer_combined_when_ready') and not started
                    and set(wanted) <= set(ready['operators'])):
                cfg['combined_arms'] = True
                expected_stage_count = sum(len(stage_specs(c)) for c in machines.values())
            for stage, arm_list, needed in stage_specs(cfg):
                key = machine+'_'+stage
                if key in active or key in done: continue
                if stage == 'operators' and not cfg.get('operators_overlap') and machine+'_reseed' not in done: continue
                if not set(wanted) <= set(ready[needed]): continue
                # Published files are immutable. Copy only this experiment's artifact roots.
                for rel in ['outputs/analysis/grid_phase/detector_v6_ck8_dwell',
                            'outputs/analysis/grid_phase/ae_k8',
                            'outputs/steer/online_pipe_v4_pilot/instr_setm_v6_ck8dwell_ck8',
                            'outputs/steer/online_pipe_v4_pilot/instr_setm_v6_ck8dwell_ck8_plain',
                            'outputs/analysis/v6_preflight_audit_20260908/fit_diagnostics']:
                    src = release/rel
                    if src.exists(): shutil.copytree(src, root/rel, dirs_exist_ok=True, copy_function=atomic_copy)
                proto = state_dir/key
                proto.mkdir(exist_ok=True)
                shutil.copy2(manifest, proto/'episodes.tsv')
                hashes = [dict(path=str(f.relative_to(release)), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                          for f in (release/'outputs').rglob('*') if f.is_file()]
                (proto/'artifacts.json').write_text(json.dumps(hashes, indent=2)+'\n')
                # GPU use is restricted to free GPUs and guarded by the local lease wrapper.
                gpu_cmd = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']
                busy_pids = set()
                stage_gpus = cfg.get('stage_gpus', {}).get(stage, cfg['gpus'])
                for gpu in stage_gpus.split(','):
                    cmd = gpu_cmd+['--id='+gpu]
                    if cfg['host']: cmd = ['ssh',cfg['host'], *cmd]
                    for line in subprocess.check_output(cmd, text=True).splitlines():
                        line = line.strip()
                        if line:
                            try: busy_pids.add(int(line.split()[0]))
                            except ValueError: busy_pids.add(-1)
                coexisting = {int(pid) for pid in cfg.get('coexisting_serve_pids', [])}
                overlap_ok = (stage == 'operators' and cfg.get('operators_overlap')
                              and coexisting and busy_pids <= coexisting)
                if busy_pids and not overlap_ok and not (machine == 'kanu' and a.allow_busy_local):
                    report(f'waiting free GPU: {key}'); continue
                if cfg['host']:
                    # Data only. Source code was already synced by git before dispatcher launch.
                    for rel in ['outputs/analysis/grid_phase/detector_v6_ck8_dwell',
                                'outputs/analysis/grid_phase/ae_k8',
                                'outputs/steer/online_pipe_v4_pilot/instr_setm_v6_ck8dwell_ck8',
                                'outputs/steer/online_pipe_v4_pilot/instr_setm_v6_ck8dwell_ck8_plain',
                                str(proto.relative_to(root))]:
                        if not (root/rel).exists(): continue
                        call(['rsync','-a','--mkpath',str(root/rel)+'/',cfg['host']+':pkt_ws/temporal_vla/'+rel+'/'])
                out_path = out_root/stage
                out = str(out_path.relative_to(root)) if out_path.is_relative_to(root) else str(out_path)
                args = ['--machine',machine,'--gpus',stage_gpus,'--lease-held','--port-base',str(cfg.get('port_base', 9400)),
                        '--phase-source','ck8','--artifact-tag','v6_ck8dwell','--reference-labels',
                        '--arms',arm_list,'--manifest',str((proto/'episodes.tsv').relative_to(root)),
                        '--detector-root','outputs/analysis/grid_phase/detector_v6_ck8_dwell',
                        '--cluster-bundle','outputs/analysis/grid_phase/ae_k8/ae_bundle_k8.npz', '--out',out]
                if (config_dir/'collection_plan.json').is_file():
                    assert not cfg['host'], 'custom plan remote deployment must be explicit'
                    args += ['--plan-json', str(config_dir/'collection_plan.json')]
                if stage == 'operators' and cfg.get('operators_overlap') and coexisting:
                    args += ['--coexisting-serve-pids', ','.join(str(pid) for pid in sorted(coexisting))]
                if machine == 'kanu' and a.allow_busy_local: args.append('--allow-busy-local')
                command = ['python',str(repo/'scripts/steer/online_gated/run_v6_heldout_all.py'),*args]
                if cfg['host']:
                    import shlex
                    command=['ssh','-o','ServerAliveInterval=30',cfg['host'],
                             'cd ~/pkt_ws/temporal_vla && exec setsid '+shlex.join(['python3','scripts/steer/online_gated/run_v6_heldout_all.py',*args])]
                launch=['bash',str(root/'scripts/utils/with_gpu_lease.sh'),cfg['lease'],stage_gpus.replace(',',' '),
                        'codex-ck8-dwell-'+key,key,'--',*command]
                fh=(state_dir/(key+'.log')).open('a')
                proc=subprocess.Popen(launch,cwd=root,stdout=fh,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                active[key]=(proc,fh)
                report(f'launched {key} pid={proc.pid}')
        if len(done) < expected_stage_count: time.sleep(30)
    report('ALL_EVAL_STAGES_COMPLETE')


if __name__ == '__main__': main()
