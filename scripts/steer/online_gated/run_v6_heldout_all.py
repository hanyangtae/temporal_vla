#!/usr/bin/env python3
"""대상 jitter 성공/실패 전판: heldout detector × 5 intervention arms + paired base.

stdlib orchestrator. 반드시 로컬 with_gpu_lease.sh 아래 실행한다. 원격은 그 래퍼
안에서 ssh로 실행해 전 머신 lease를 로컬 원장에 유지한다. 실패 작업은 자동 재시도하지 않는다.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ARMS = {
    'base': ('ps_base', '', '0'),
    'plain_b08': ('ps_setm', 'instr_setm_v6_gt_plain', '0.8'),
    'jfair_b09': ('ps_setm', 'instr_setm_v6_gt', '0.9'),
    'reseed': ('ps_reseed', '', '0'),
    'reseed_plain_b08': ('ps_reseed_setm', 'instr_setm_v6_gt_plain', '0.8'),
    'reseed_jfair_b09': ('ps_reseed_setm', 'instr_setm_v6_gt', '0.9'),
}

def build_arms(tag='v6', phase_source='gt'):
    """Return the selected artifact variant mapping for a phase source."""
    if phase_source not in ('gt', 'ck8'):
        raise ValueError(f'unknown phase source: {phase_source}')
    suffix = f'instr_setm_{tag}_{phase_source}'
    return {
        'instruction_b08': ('ps_setm', f'{suffix}_instruction_plain', '0.8'),
        'cluster_instruction_b08': ('ps_setm', f'{suffix}_cluster_instruction_plain', '0.8'),
        'base': ('ps_base', '', '0'),
        'plain_b08': ('ps_setm', f'{suffix}_plain', '0.8'),
        'jfair_b09': ('ps_setm', suffix, '0.9'),
        'reseed': ('ps_reseed', '', '0'),
        'reseed_plain_b08': ('ps_reseed_setm', f'{suffix}_plain', '0.8'),
        'reseed_jfair_b09': ('ps_reseed_setm', suffix, '0.9'),
    }

def read_tsv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def write_tsv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def key(r):
    return (r['grid_instruction'].replace('/', '_'), int(r['scene_idx']), int(r['jitter_idx']))

def result_rows(out, arm, cell, arm_defs=None):
    slug, s, j = cell
    runarm = (arm_defs or ARMS)[arm][0]
    path = out / arm / f'{slug}_s{s}_j{j}' / slug / runarm / 'per_episode.tsv'
    return read_tsv(path) if path.exists() else []

def check_rows(rows, expected):
    got = [int(r['noise_idx']) for r in rows]
    want = [int(r['noise_idx']) for r in expected]
    if len(got) != len(set(got)) or set(got) != set(want):
        raise ValueError(f'noise set mismatch: {got} != {want}')
    by_noise = {int(r['noise_idx']): r for r in expected}
    for r in rows:
        e = by_noise[int(r['noise_idx'])]
        for col in ('scene_idx', 'jitter_idx', 'env_seed', 'inference_seed'):
            if int(r[col]) != int(e[col]):
                raise ValueError(f'{col}: {r[col]} != {e[col]}')
        if int(r['collection_success']) != int(e['success']):
            raise ValueError('collection label mismatch')
        if r['success'] not in ('0', '1'):
            raise ValueError('invalid success')

def check_sidecars(out, arm, cell, rows, arm_defs=None):
    slug,s,j = cell
    defs = arm_defs or ARMS
    folder = out/arm/f'{slug}_s{s}_j{j}'/slug/defs[arm][0]/'raw_rollouts'
    want_op = {'ps_base': None, 'ps_setm': 'setm', 'ps_reseed': 'reseed',
               'ps_reseed_setm': 'reseed_setm'}[defs[arm][0]]
    for r in rows:
        paths = list(folder.rglob(f'task0--ep{r["ep"]}--succ*.json'))
        if len(paths) != 1:
            raise ValueError(f'sidecar missing/duplicate: ep {r["ep"]}')
        d = json.loads(paths[0].read_text())
        if d.get('perstep_op') != want_op or int(d['episode_success']) != int(r['success']):
            raise ValueError('sidecar op/success mismatch')
        if want_op in ('setm', 'reseed_setm'):
            spec = d.get('serve_steering') or {}
            if (spec.get('setpoint_application') != 'token_mean_common_shift_v2'
                    or spec.get('layers') != [12] or spec.get('token_select') != 'all'
                    or spec.get('denoise') != 'global'):
                raise ValueError('setpoint version/layer/token/denoise mismatch')
        if arm in ('instruction_b08', 'cluster_instruction_b08'):
            expected_mode = 'instruction_only' if arm == 'instruction_b08' else 'cluster_instruction_fallback'
            if (d.get('serve_steering') or {}).get('instruction_routing', {}).get('mode') != expected_mode:
                raise ValueError('instruction routing mismatch')
            if d.get('perstep_fallback_mode') != 'instruction':
                raise ValueError('instruction fallback mode mismatch')
            if any(str(v).startswith('reseed:') for v in d.get('gate_fallback', []) if v):
                raise ValueError('unexpected reseed in instruction arm')
        for i, seed2 in enumerate(d.get('perstep_seed2', [])):
            if seed2 is None:
                continue
            fallback = (d.get('gate_fallback') or [None]*len(d['perstep_seed2']))[i]
            offset = 900000 if want_op in ('reseed', 'reseed_setm') or str(fallback or '').startswith('reseed:') else 0
            if seed2 != int(r['inference_seed']) + i + offset:
                raise ValueError(f'seed2 mismatch at record {i}: {seed2}')

def summarize(out, cells, arms, strict=False, arm_defs=None, reference_labels=False):
    """같은 noise의 새 base 기준. 수집 라벨 불일치는 별도 기록해 해석을 보류한다."""
    episodes, totals, complete = [], [], True
    baseline_mismatch = 0
    for cell, expected in cells.items():
        base_rows = result_rows(out, 'base', cell, arm_defs) if 'base' in arms else []
        if 'base' not in arms and not reference_labels:
            raise ValueError('base arm absent; --reference-labels is required')
        base = {int(r['noise_idx']): int(r['success']) for r in base_rows}
        labels = {int(r['noise_idx']): int(r['success']) for r in expected}
        if 'base' not in arms:
            base = labels.copy()
        baseline_mismatch += sum(base[n] != labels[n] for n in base if n in labels)
        for arm in arms:
            rows = result_rows(out, arm, cell, arm_defs)
            try:
                check_rows(rows, expected)
            except (ValueError, KeyError):
                complete = False
                if strict:
                    raise
            for r in rows:
                n = int(r['noise_idx'])
                if n not in labels:
                    continue
                b = base.get(n)
                episodes.append(dict(r, eval_arm=arm, baseline_success='' if b is None else b,
                                     collection_label=labels[n]))
    for arm in arms:
        rs = [r for r in episodes if r['eval_arm'] == arm]
        paired = [r for r in rs if r['baseline_success'] != '']
        fails = [r for r in paired if int(r['baseline_success']) == 0]
        succs = [r for r in paired if int(r['baseline_success']) == 1]
        totals.append(dict(arm=arm, completed=len(rs), expected=sum(map(len, cells.values())),
                           paired=len(paired), baseline_fail=len(fails),
                           rescued=sum(int(r['success']) for r in fails),
                           baseline_success=len(succs),
                           destroyed=sum(1-int(r['success']) for r in succs),
                           success=sum(int(r['success']) for r in rs)))
    write_tsv(out / 'summary.tsv', totals, list(totals[0]) if totals else ['arm'])
    if episodes:
        write_tsv(out / 'episodes.tsv', episodes, list(episodes[0]))
    return {'complete': complete, 'reference_source': 'base' if 'base' in arms else 'collection_success',
            'baseline_collection_mismatches': baseline_mismatch,
            'arms': totals}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--machine', required=True, choices=['kanu', 'worker1', 'worker2'])
    p.add_argument('--gpus', required=True, help='comma separated, max 3 kanu / 1 server')
    p.add_argument('--main-root', type=Path, default=Path.home()/'pkt_ws/temporal_vla')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan-json', type=Path)
    p.add_argument('--plan-map', type=Path, help='plan_id to repository-relative original plan path')
    p.add_argument('--port-base', type=int, default=9100)
    p.add_argument('--cell', help='optional smoke cell slug:s:j')
    p.add_argument('--noises', help='optional smoke noise ids')
    p.add_argument('--maxep', type=int, default=720)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--lease-held', action='store_true')
    p.add_argument('--coexisting-serve-pids', default='', help='known existing GR00T serves on the single A100 GPU; reserve their slots')
    p.add_argument('--allow-busy-local', action='store_true', help='explicit one-run user exception for kanu GPU5/6/7')
    p.add_argument('--phase-source', choices=('gt', 'ck8'), default='gt')
    p.add_argument('--artifact-tag', default='v6')
    p.add_argument('--detector-root', type=Path)
    p.add_argument('--cluster-bundle', type=Path)
    p.add_argument('--reference-labels', action='store_true')
    a = p.parse_args()
    arm_defs = build_arms(a.artifact_tag, a.phase_source)
    repo = Path(__file__).resolve().parents[3]
    plan = (a.plan_json or repo/'configs/collect/n15_grid_v6_scene_jitter/collection_plan.json').resolve()
    a.main_root = a.main_root.resolve(); a.out = a.out.resolve(); a.manifest = a.manifest.resolve()
    gpus = [int(g) for g in a.gpus.split(',')]
    assert len(gpus) == len(set(gpus)) and 1 <= len(gpus) <= (3 if a.machine=='kanu' else 1)
    coexisting = {int(v) for v in a.coexisting_serve_pids.split(',') if v}
    if coexisting:
        assert a.lease_held and a.machine != 'kanu' and len(gpus) == 1 and len(coexisting) < 6
    arms = a.arms.split(','); assert set(arms) <= set(arm_defs) and len(set(arms)) == len(arms)
    if a.phase_source == 'ck8' and not a.cluster_bundle:
        p.error('--phase-source ck8 requires --cluster-bundle')
    if 'base' not in arms and not a.reference_labels:
        p.error('--arms without base requires --reference-labels')
    if not a.dry_run and not a.lease_held:
        p.error('with_gpu_lease.sh 아래에서 --lease-held 를 지정하세요')
    cells = collections.defaultdict(list)
    for r in read_tsv(a.manifest):
        if r['machine'] != a.machine:
            continue
        c = key(r)
        if a.cell and ':'.join(map(str,c)) != a.cell:
            continue
        if a.noises and r['noise_idx'] not in a.noises.split(','):
            continue
        cells[c].append(r)
    assert cells, 'empty eval set'
    for c, rows in cells.items():
        ns = [int(r['noise_idx']) for r in rows]
        assert len(ns) == len(set(ns)), (c, ns)
        if not a.noises:
            assert set(ns) == set(range(10)), (c, ns)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'DONE.json').exists():
        raise SystemExit('existing DONE.json; choose a new output or inspect completed run')
    bundle_sha = None
    if a.cluster_bundle:
        a.cluster_bundle = a.cluster_bundle.resolve()
        if not a.cluster_bundle.is_file():
            p.error(f'cluster bundle not found: {a.cluster_bundle}')
        bundle_sha = hashlib.sha256(a.cluster_bundle.read_bytes()).hexdigest()
    contract = {'machine':a.machine, 'gpus':gpus, 'arms':arms, 'maxep':a.maxep,
                'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                'cells':[list(c) for c in sorted(cells)], 'noises':a.noises,
                'detector':str((a.detector_root or Path('outputs/analysis/grid_phase/detector_v6_ho')).name),
                'phase_source':a.phase_source, 'artifact_tag':a.artifact_tag,
                'cluster_bundle_sha256':bundle_sha,
                'reference_source':'base' if 'base' in arms else 'collection_success',
                'fallback':('instruction' if set(arms) <= {'instruction_b08', 'cluster_instruction_b08'} else 'reseed'), 'phase':'GT' if a.phase_source == 'gt' else 'cluster',
                'setpoint_application':'token_mean_common_shift_v2',
                'fit_layer':12, 'hook_layers':[12], 'fit_denoise':3,
                'apply_denoise':'all_calls', 'fit_tokens':'all_49', 'apply_tokens':'all_49',
                'beta_selection':'prior 225 fail episodes; jfair .9 tie lower; plain .8 sole candidate'}
    if a.plan_json:
        contract['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest()
        contract['plan_id'] = json.loads(plan.read_text())['plan_id']
        assert all(r['plan_id'] == contract['plan_id'] for rs in cells.values() for r in rs)
    plans = {}
    if a.plan_map:
        for pid, rel in json.loads(a.plan_map.read_text()).items():
            pp = (repo/rel).resolve()
            assert json.loads(pp.read_text())['plan_id'] == pid
            plans[pid] = pp
        assert all(r['plan_id'] in plans for rs in cells.values() for r in rs)
        contract['plans'] = {pid: dict(path=str(pp.relative_to(repo)), sha256=hashlib.sha256(pp.read_bytes()).hexdigest()) for pid, pp in plans.items()}
    contract_file = a.out/'contract.json'
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise SystemExit('output contract mismatch')
    contract_file.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    write_tsv(a.out/'manifest.tsv', [r for rs in cells.values() for r in rs], list(next(iter(cells.values()))[0]))
    npz = a.main_root/'outputs/steer/online_pipe_v4_pilot'
    detector_root = (a.detector_root or a.main_root/'outputs/analysis/grid_phase/detector_v6_ho').resolve()
    hashes = {r['path']: r['sha256'] for r in json.loads(a.manifest.with_name('artifacts.json').read_text())}
    verified = set()
    def verify(path):
        if path in verified: return
        rel = str(path.relative_to(a.main_root))
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[rel]:
            raise ValueError(f'artifact hash mismatch: {path}')
        verified.add(path)
    if a.cluster_bundle: verify(a.cluster_bundle.resolve())
    jobs=[]
    for c, rows in sorted(cells.items()):
        slug,s,j = c; art = rows[0]['artifact_slug']; stem=f'{art}__s{s}'
        det=detector_root/'loko'/stem/f's{s}'/f'j{j}'/f'detector_pertask_lstm_{stem}.pt'
        assert det.is_file(), det
        verify(det)
        root=a.out/'npz_roots'/f'{slug}_s{s}_j{j}'
        (root/slug).mkdir(parents=True,exist_ok=True)
        variants = sorted({arm_defs[x][1] for x in arms if arm_defs[x][1]})
        for variant in variants:
            target=npz/variant/art/f's{s}'/f'j{j}'
            assert list(target.glob('*/dit_L12/conceptors.npz')) or (target/'fallback_only.json').is_file(), target
            if (target/'fallback_only.json').is_file(): verify(target/'fallback_only.json')
            if (target/'instruction_routing.json').is_file(): verify(target/'instruction_routing.json')
            for artifact in target.glob('*/dit_L12/*'):
                if artifact.name in ('conceptors.npz', 'metadata.json'):
                    verify(artifact)
            link=root/slug/variant
            if link.is_symlink():
                assert link.resolve()==target.resolve(), link
            else:
                link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        for arm in arms:
            runarm,variant,beta=arm_defs[arm]
            env=dict(os.environ, GPUS='', SERVES_PER_GPU='1', ALLOW_BUSY_GPU='1',
                     SLUGS=slug, ARMS=runarm, EP_MODE='replay', EVAL_SCENES=str(s),
                     EVAL_JITTERS=str(j), EVAL_NOISES=','.join(r['noise_idx'] for r in rows),
                     FIT_SCENES='0-4', FIT_NOISES='0-4', REPLAY_MACHINE=a.machine,
                     INDEX_TSV=str(a.out/'manifest.tsv'), PLAN_JSON=str(plans.get(rows[0]['plan_id'], plan)),
                     EP_META_DIR='', EP_META_LOAD_ENV_NAME='', DETECTOR_CKPT=str(det),
                     FAILURE_TASK=stem, FAILURE_ALPHA='0.1', PERSTEP_N='1',
                     DETECTOR_LAYERS='0,2,4,8,10,12,15', TOKEN_POOL='all_token_full',
                     NPZ_ROOT=str(root), NPZ_VARIANT=variant, STEER_OP='setpoint',
                     EXPECTED_STEER_LAYER='12',
                     STEER_ALPHA='0', STEER_BETA=beta, PERSTEP_FALLBACK=('instruction' if arm in ('instruction_b08', 'cluster_instruction_b08') else 'reseed'),
                     CLUSTER_BUNDLE=str(a.cluster_bundle) if a.cluster_bundle else '',
                     CLUSTER_TASK=art, OUT_ROOT=str(a.out/arm/f'{slug}_s{s}_j{j}'),
                     MAXEP=str(a.maxep), CAPTURE_FEATURES='0', SERVE_BOOT_TRIES='150',
                     DRY_RUN='1' if a.dry_run else '0')
            if a.machine!='kanu':
                env.update(SERVE_MODE='host',SERVE_PY=str(Path.home()/'miniconda3/envs/lerobot_050_groot/bin/python'),
                           SERVE_PYTHONPATH=str(a.main_root/'lerobot/src'))
            existing=result_rows(a.out,arm,c,arm_defs)
            if existing:
                try:
                    check_rows(existing,rows)
                    check_sidecars(a.out,arm,c,existing,arm_defs)
                except (ValueError, KeyError) as e:
                    raise SystemExit(f'partial existing job {arm} {c}; inspect before retry: {e}')
                print(f'[skip] {arm} {c}',flush=True); continue
            jobs.append((arm,c,rows,env))
    print(f'[preflight] verified {len(verified)} artifact hashes',flush=True)
    if not a.dry_run:
        for gpu in gpus:
            busy=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy and not (coexisting and set(map(int, busy.split())) <= coexisting) and not (a.allow_busy_local and a.machine == 'kanu' and set(gpus) <= {5,6,7}):
                raise SystemExit(f'GPU {gpu} already occupied: {busy}')
    slots=[g for g in gpus for _ in range(2 if a.machine=='kanu' else 6)]
    active={}; errors=[]; stopped=False; port=a.port_base
    def stop(sig,frame):
        nonlocal stopped
        stopped=True
        for proc,_,_,_ in active.values():
            proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    logdir=a.out/'logs'; logdir.mkdir(exist_ok=True)
    print(f'[start] pid={os.getpid()} machine={a.machine} jobs={len(jobs)} episodes={sum(len(x[2]) for x in jobs)} slots={slots}',flush=True)
    while jobs or active:
        for slot in list(active):
            proc,job,fh,lp=active[slot]
            rc=proc.poll()
            if rc is None: continue
            fh.close(); del active[slot]
            arm,c,rows,_=job
            error=None
            if rc: error=f'runner rc={rc}'
            elif not a.dry_run:
                try:
                    rr=result_rows(a.out,arm,c,arm_defs); check_rows(rr,rows)
                    check_sidecars(a.out,arm,c,rr,arm_defs)
                    if arm=='base' and a.maxep==720 and any(int(r['success'])!=int(r['collection_success']) for r in rr):
                        error='base != collection (replay mismatch); pending launches stopped'
                except (ValueError, KeyError) as ex:
                    error=str(ex)
            print(f'[finished] {arm} {c} rc={rc} error={error}',flush=True)
            if error:
                errors.append({'arm':arm,'cell':c,'error':error,'log':str(lp)})
                stopped=True
            if not a.dry_run:
                summarize(a.out,cells,arms,arm_defs=arm_defs,reference_labels=a.reference_labels)
        if stopped: jobs.clear()
        # Existing serves have no pending jobs: as they exit, refill their slots.
        # Query GPU residency, not PID existence (a zombie must not reserve a slot).
        slot_limit = len(slots)
        if coexisting and not a.dry_run:
            resident = subprocess.check_output(['nvidia-smi', f'--id={gpus[0]}',
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            slot_limit -= len(coexisting & {int(v) for v in resident.split()})
        for slot,gpu in enumerate(slots):
            if slot >= slot_limit: continue
            if not jobs: break
            if slot in active: continue
            job=jobs.pop(0); arm,c,rows,env=job
            while True:
                with socket.socket() as sock:
                    try: sock.bind(('127.0.0.1',port)); break
                    except OSError: port+=1
            env.update(GPUS=str(gpu),PORT_BASE=str(port))
            lp=logdir/f'{arm}_{c[0]}_s{c[1]}_j{c[2]}.log'; fh=lp.open('a')
            proc=subprocess.Popen(['bash',str(repo/'scripts/steer/online_gated/run_online_gated_eval.sh')],env=env,stdout=fh,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[slot]=(proc,job,fh,lp)
            print(f'[launch] {arm} {c} gpu={gpu} port={port} pid={proc.pid}',flush=True)
            port+=1
        if active: time.sleep(3)
    summary=summarize(a.out,cells,arms,arm_defs=arm_defs,reference_labels=a.reference_labels) if not a.dry_run else {'dry_run':True}
    summary.update(errors=errors,stopped=stopped)
    filename='DONE.json' if not errors and not stopped and (summary.get('complete') or a.dry_run) else 'INCOMPLETE.json'
    (a.out/filename).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(f'[{filename}] {summary}',flush=True)
    return 0 if filename=='DONE.json' else 1

if __name__=='__main__':
    sys.exit(main())
