#!/usr/bin/env python3
"""Target9 n40 수집 watcher (srv48·srv50 각 머신에서 실행, stdlib 전용).

정본 계약: configs/collect/n15_target9_n40_20261025/README.md (frozen, 이 스크립트는 수정하지 않는다).

한 사이클(기본 1800s):
  1. 다른 collect_grid.sh / http_feature_collect.py 프로세스가 있으면 대기 (자기 자신 제외).
  2. 승준 아카이브를 ssh 로 read-only 인벤토리 (requested 셀마다 meta/rollout/video/traj 4파일 비어있지 않음).
  3. 아카이브·로컬 여유 >= 20GiB 확인.
  4. phase 선택: srv48 = backfill(n20 계약, 별도 grid_n20_srv48_resume_20261026 루트) 먼저 → n40;
     srv50 = n40 만. remaining = requested − 아카이브 완료 − shipped − 로컬 meta.json.
  5. 빈 GPU(compute 프로세스 0 · memory.used < 100MiB) 1장 · 포트 6개 free · 전용 headless 컨테이너.
  6. 그 루트를 서비스하는 shipper 가 없으면 자기 shipper 1개 발사 (기존 프로세스는 절대 kill 하지 않음).
  7. collect_grid.sh 를 공통 harness 경유(groot_launch_guard → groot_harness run)로 blocking 실행.
     종료 후 완료 여부는 다음 사이클 아카이브 재감사로만 판정한다.

발사 전 검사(harness check-collection) 실패 = 설정 오류 → 재시도 없이 종료.
진척 없는 시도 3회 연속 → 발사 중단·종료. SIGTERM: 대기 중단, 수집기·shipper 는 살려두고 PID 기록 후 종료.

  python3 scripts/collect/watch_target9_n40.py --machine srv48 --dry-run --once   # 감사
  setsid nohup python3 scripts/collect/watch_target9_n40.py --machine srv48 \
      > /dev/null 2>&1 < /dev/null &                                              # 실행
"""
import argparse
import fcntl
import json
import logging
import logging.handlers
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
N40_DIR = REPO / 'configs/collect/n15_target9_n40_20261025'
N20_DIR = REPO / 'configs/collect/n15_target9_n20_20261013'
COLLECT_SH = REPO / 'scripts/safe/groot_n15/robocasa/collect/collect_grid.sh'
SHIP_SH = REPO / 'scripts/safe/groot_n15/robocasa/collect/ship_to_archive.sh'
CREATE_CONTAINER_SH = REPO / 'scripts/utils/create_collect_container.sh'
CHECK_CONTAINER_PY = REPO / 'scripts/utils/check_collect_container.py'
HARNESS_PY = REPO / 'scripts/utils/groot_harness.py'

KANU_COMMON = '/home/dongkyu/pkt_ws/temporal_vla'   # frozen 계약 안의 kanu 절대경로 접두
N40_PLAN_ID, N20_PLAN_ID = '2f26fb046fc4', '748863ce9233'
MACHINES = {  # machine → (아카이브 machine 층 = serve hostname, 참조 컨테이너)
    'srv48': ('worker1', 'robocasa-collect-gpu3-n20'),
    'srv50': ('worker2', 'robocasa-collect-gpu3'),
}
ARCHIVE_HOST, ARCHIVE_PORT = 'kimseungjun@166.104.146.37', '11112'
ARCHIVE_ROOT = '/home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid'
REQUIRED_FILES = ('meta.json', 'rollout.pkl', 'video.mp4', 'traj.csv')
MIN_FREE = 20 * 1024**3
GPU_MEM_MAX_MIB = 100
SERVES_PER_GPU, PORT_BASE = 6, 9860
MAX_FAILED_ATTEMPTS, KEEP_ATTEMPT_LOGS = 3, 3
PROC_PATTERNS = ('collect_grid.sh', 'http_feature_collect.py')
HARNESS_ENV = {
    'HARNESS_SESSION': 'collect-topup-fork',
    'HARNESS_COORDINATOR_SSH': 'kanu-groot-broker',
    'HARNESS_COORDINATOR_SCRIPT': '/home/dongkyu/pkt_ws/temporal_vla/scripts/utils/groot_harness.py',
}
SERVE_PY = '/home/junhyeong/miniconda3/envs/lerobot_050_groot/bin/python'
COLLECTOR_PY = '/temporal_vla/.claude/worktrees/collect-topup/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py'
PYPATH_PREFIX = '/temporal_vla/.claude/worktrees/collect-topup'

log = logging.getLogger('watch')
STOP = threading.Event()


# ── 순수 함수 (tests/test_watch_target9_n40.py) ──────────────────────────────────
def cell_key(instr, s, j, n):
    return f'{instr}|s{int(s)}|j{int(j)}|n{int(n)}'


def read_tsv(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split('\t')
    return [dict(zip(head, ln.split('\t'))) for ln in lines[1:] if ln.strip()]


def requested_n40(rows, machine, plan_id=N40_PLAN_ID):
    keys = [cell_key(r['instruction'], r['scene'], r['jitter'], r['noise'])
            for r in rows if r['machine'] == machine]
    bad = {r['plan_id'] for r in rows if r['machine'] == machine} - {plan_id}
    if bad:
        raise ValueError(f'requests.tsv plan_id mismatch: {sorted(bad)}')
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate requested n40 cells')
    return set(keys)


def requested_backfill(rows, archive_machine, source_plan=N20_PLAN_ID):
    keys = set()
    for r in rows:
        if r['machine'] != archive_machine:
            continue
        if r['source_plan'] != source_plan or not 10 <= int(r['noise']) <= 19:
            raise ValueError(f'backfill row outside n10..19/{source_plan}: {r}')
        keys.add(cell_key(r['instruction'], r['scene'], r['jitter'], r['noise']))
    return keys


def allowed_cells(contract, noise_lo, noise_hi):
    """contract 의 selected scene × j0..4 × n[lo,hi) — harness 가 수집 허용하는 범위."""
    return {cell_key(c['instruction'], c['scene'], j, n) for c in contract['selected_scenes']
            for j in range(contract['jitter_count']) for n in range(noise_lo, noise_hi)}


def cell_key_from_rel(rel):
    """<instr...>/s<s>/j<j>/n<n>/<arm> → 셀 키 (형식 이상이면 None)."""
    p = rel.strip('/').split('/')
    if len(p) < 5 or not (re.fullmatch(r's\d+', p[-4]) and re.fullmatch(r'j\d+', p[-3])
                          and re.fullmatch(r'n\d+', p[-2])):
        return None
    return f"{'/'.join(p[:-4])}|{p[-4]}|{p[-3]}|{p[-2]}"


def build_done_list(skip_lines, archive_complete, shipped, allowed, requested):
    """DONE_LIST = frozen skip ∪ 아카이브 완료 ∪ shipped ∪ (허용범위 − requested).
    마지막 항은 요청 밖 셀(이미 정상 수집분)을 절대 재수집하지 않게 한다."""
    return sorted(set(skip_lines) | set(archive_complete) | set(shipped) | (set(allowed) - set(requested)))


def phase_counts(requested, archive_complete, archive_incomplete, shipped, local_meta):
    req = set(requested)
    done = req & set(archive_complete)
    covered = done | (req & set(shipped)) | (req & set(local_meta))
    return {
        'requested': len(req), 'archive_complete': len(done),
        'archive_incomplete': len(req & set(archive_incomplete) - done),
        'shipped_not_archived': len(req & set(shipped) - done),
        'local_pending_ship': len(req & set(local_meta) - done),
        'remaining_to_launch': len(req - covered),
        'complete': done == req,
    }


def parse_gpu_csv(text):
    gpus = []
    for ln in text.strip().splitlines():
        idx, uuid, mem = [x.strip() for x in ln.split(',')]
        gpus.append({'index': int(idx), 'uuid': uuid, 'mem': int(mem) if mem.isdigit() else None})
    return gpus


def parse_apps_csv(text):
    apps = []
    for ln in text.strip().splitlines():
        if ln.strip() and 'No running' not in ln:
            uuid, pid = [x.strip() for x in ln.split(',')[:2]]
            apps.append({'uuid': uuid, 'pid': pid})
    return apps


def choose_gpu(gpus, apps, mem_max=GPU_MEM_MAX_MIB):
    """프로세스 0 · memory.used < mem_max 인 첫 GPU index (없으면 None)."""
    busy = {a['uuid'] for a in apps}
    for g in sorted(gpus, key=lambda g: g['index']):
        if g['uuid'] not in busy and g['mem'] is not None and g['mem'] < mem_max:
            return g['index']
    return None


def matching_procs(procs, patterns=PROC_PATTERNS, exclude=()):
    """procs = [(pid, argv)] → 패턴 파일명을 argv 토큰으로 가진 pid 목록."""
    out = []
    for pid, argv in procs:
        if pid in exclude:
            continue
        if any(os.path.basename(a) == p for a in argv for p in patterns):
            out.append(pid)
    return out


def shipper_root(argv, environ, cwd):
    """ship_to_archive.sh 프로세스면 (해석된 GRID_ROOT, 스크립트 경로), 아니면 None."""
    script = next((a for a in argv if os.path.basename(a) == 'ship_to_archive.sh'), None)
    if script is None or 'GRID_ROOT' not in environ:
        return None
    root = os.path.normpath(os.path.join(cwd, environ['GRID_ROOT']))
    return root, os.path.normpath(os.path.join(cwd, script))


def rebase(path, common):
    return common + path[len(KANU_COMMON):] if path.startswith(KANU_COMMON + '/') else path


def runtime_contract(contract, contract_dir, common, parent_grid_root):
    """frozen 계약 사본: parent_plan 을 이 머신의 절대경로로, parent_grid_root 를 실제 옛 grid 로."""
    out = dict(contract)
    pp = contract['parent_plan']
    pp = rebase(pp, common) if os.path.isabs(pp) else os.path.normpath(os.path.join(contract_dir, pp))
    out['parent_plan'] = pp
    out['parent_grid_root'] = parent_grid_root
    return out


def ports_free(base, count, host='0.0.0.0'):
    for p in range(base, base + count):
        with socket.socket() as s:
            try:
                s.bind((host, p))
            except OSError:
                return False
    return True


# ── 원격 인벤토리 (아카이브 read-only) ─────────────────────────────────────────
REMOTE_PY = r'''
import json, os, sys
req = json.loads(sys.argv[1]); root = req["root"]; out = {"available": os.path.isdir(root)}
if out["available"]:
    st = os.statvfs(root); out["free_bytes"] = st.f_bavail * st.f_frsize
for scan in req["scans"]:
    base = os.path.join(root, scan["plan"], scan["machine"]); comp, inc = [], []
    for d, _dirs, files in os.walk(base):
        if "meta.json" not in files and not set(files) & set(req["required"]):
            continue
        ok = all(f in files and os.path.getsize(os.path.join(d, f)) > 0 for f in req["required"])
        (comp if ok else inc).append(os.path.relpath(d, base))
    out[scan["plan"] + "/" + scan["machine"]] = {"complete": comp, "incomplete": inc}
print(json.dumps(out))
'''


def archive_inventory(scans):
    arg = json.dumps({'root': ARCHIVE_ROOT, 'scans': scans, 'required': list(REQUIRED_FILES)})
    cmd = ['bash', str(REPO / 'scripts/utils/remote_compute.sh'), 'run',
           'python3 -c ' + shlex.quote(REMOTE_PY) + ' ' + shlex.quote(arg)]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f'archive ssh rc={res.returncode}: {res.stderr.strip()[-300:]}')
    data = json.loads(res.stdout)
    for scan in scans:
        sub = data.get(f"{scan['plan']}/{scan['machine']}", {'complete': [], 'incomplete': []})
        comp = {k for k in map(cell_key_from_rel, sub['complete']) if k}
        inc = {k for k in map(cell_key_from_rel, sub['incomplete']) if k} - comp
        scan['complete'], scan['incomplete'] = comp, inc
    return data


# ── 로컬 상태 ─────────────────────────────────────────────────────────────────
def local_meta_keys(grid_root, plan_id, archive_machine):
    base = Path(grid_root) / plan_id / archive_machine
    keys = set()
    if base.is_dir():
        for meta in base.rglob('meta.json'):
            k = cell_key_from_rel(str(meta.parent.relative_to(base)))
            if k:
                keys.add(k)
    return keys


def read_lines(path):
    p = Path(path)
    return {x.strip() for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()


def free_bytes(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def list_procs():
    procs = []
    for d in Path('/proc').iterdir():
        if d.name.isdigit():
            try:
                argv = (d / 'cmdline').read_bytes().split(b'\0')
                procs.append((int(d.name), [a.decode(errors='replace') for a in argv if a]))
            except OSError:
                pass
    return procs


def proc_environ(pid):
    try:
        raw = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        return dict(x.decode(errors='replace').split('=', 1) for x in raw if b'=' in x)
    except OSError:
        return {}


def find_shippers(grid_root):
    """GRID_ROOT 가 grid_root 인 ship_to_archive.sh 프로세스 [(pid, script, sha_code)]."""
    target = os.path.realpath(grid_root)
    found = []
    for pid, argv in list_procs():
        if not any(os.path.basename(a) == 'ship_to_archive.sh' for a in argv):
            continue
        try:
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except OSError:
            cwd = '/'
        hit = shipper_root(argv, proc_environ(pid), cwd)
        if hit and os.path.realpath(hit[0]) == target:
            try:
                text = Path(hit[1]).read_text()
                sha = 'sha256sum' in text and 'HASH OK' in text
            except OSError:
                sha = False
            found.append({'pid': pid, 'script': hit[1], 'sha_verify_code': sha,
                          'own': proc_environ(pid).get('WATCH_TARGET9_OWNER')})
    return found


def query_gpus():
    q = ['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits']
    a = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']
    return (parse_gpu_csv(subprocess.check_output(q, text=True, timeout=60)),
            parse_apps_csv(subprocess.check_output(a, text=True, timeout=60)))


def docker_inspect(name):
    res = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=60)
    return json.loads(res.stdout)[0] if res.returncode == 0 else None


def mount_source(info, dest):
    return next((m['Source'] for m in info.get('Mounts', []) if m.get('Destination') == dest), None)


def container_gpu_ids(info):
    reqs = (info.get('HostConfig') or {}).get('DeviceRequests') or []
    return [d for r in reqs for d in (r.get('DeviceIDs') or [])]


# ── 로그 파이프 (분리 프로세스: watcher 가 죽어도 수집기 stdout 이 SIGPIPE 를 맞지 않게) ─
def log_pipe_main(path, max_bytes):
    h = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=1)
    h.setFormatter(logging.Formatter('%(message)s'))
    lg = logging.getLogger('pipe')
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for line in sys.stdin.buffer:
        lg.info(line.decode(errors='replace').rstrip('\n'))
    return 0


def spawn_logged(cmd, env, log_path, max_bytes, cwd):
    """cmd 를 새 세션으로 띄우고 출력은 분리된 rotating 로그 프로세스로 (총 ≤ 2×max_bytes)."""
    r, w = os.pipe()
    try:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--log-pipe', str(log_path),
                          '--log-max-bytes', str(max_bytes)], stdin=r, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        return subprocess.Popen(cmd, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=w,
                                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    finally:
        os.close(r)
        os.close(w)


# ── watcher ───────────────────────────────────────────────────────────────────
class Watcher:
    def __init__(self, args):
        self.a = args
        self.machine = args.machine
        self.arch_machine, self.ref_container = MACHINES[args.machine]
        self.common = args.common or str(Path(subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
            text=True).strip()).parent)
        out = Path(self.common) / 'outputs/collect'
        self.state = out / 'target9_n40_watch_20261026' / self.machine
        self.state.mkdir(parents=True, exist_ok=True)
        self.status_path = self.state / ('status_dryrun.json' if args.dry_run else 'status.json')
        self.failures = 0
        self.collector = None
        self.status = {'machine': self.machine, 'pid': os.getpid(), 'dry_run': args.dry_run,
                       'started': time.strftime('%F %T')}
        self.phases = self._build_phases(out)

    def _build_phases(self, out):
        phases = []
        n40_rows = read_tsv(N40_DIR / 'requests.tsv')
        if self.machine == 'srv48':
            c20 = json.loads((N20_DIR / 'srv48_contract.json').read_text())
            req = requested_backfill(read_tsv(N40_DIR / 'backfill_development.tsv'), self.arch_machine)
            v6 = rebase(c20['parent_grid_root'], self.common)
            phases.append(dict(
                name='backfill_n20', plan_id=N20_PLAN_ID, plan=N20_DIR / 'collection_plan.json',
                grid_root=str(out / 'grid_n20_srv48_resume_20261026'), noise_limit=20, requested=req,
                skip=read_lines(N20_DIR / 'srv48_skip.txt'), allowed=allowed_cells(c20, 10, 20),
                contract=runtime_contract(c20, str(N20_DIR), self.common, v6)))
        c40 = json.loads((N40_DIR / f'{self.machine}_contract.json').read_text())
        phases.append(dict(
            name='n40', plan_id=N40_PLAN_ID, plan=N40_DIR / 'collection_plan.json',
            grid_root=str(out / f'grid_n40_{self.machine}_20261026'), noise_limit=40,
            requested=requested_n40(n40_rows, self.machine),
            skip=read_lines(N40_DIR / f'{self.machine}_skip.txt'), allowed=allowed_cells(c40, 20, 40),
            contract=runtime_contract(c40, str(N40_DIR), self.common,
                                      str(out / f'grid_n20_{self.machine}_20261014'))))
        for ph in phases:
            stray = ph['requested'] - ph['allowed']
            if stray:
                raise ValueError(f"{ph['name']}: requested outside contract scope: {sorted(stray)[:3]}")
            ph['instructions'] = ','.join(ph['contract']['allowed_instructions'])
        return phases

    def write_status(self, **kw):
        self.status.update(kw, updated=time.strftime('%F %T'))
        tmp = self.status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.status, indent=2, ensure_ascii=False, default=sorted) + '\n')
        os.replace(tmp, self.status_path)

    # 한 사이클. 반환: 'done' | 'wait' | 'launched' | 'fatal'
    def cycle(self):
        active = matching_procs(list_procs(), exclude={os.getpid()})
        if active:
            self.write_status(state='wait_active_collector', active_pids=active)
            log.info('기존 수집 프로세스 %s — 대기', active)
            return 'wait'
        scans = [{'plan': ph['plan_id'], 'machine': self.arch_machine} for ph in self.phases]
        try:
            inv = archive_inventory(scans)
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
            self.write_status(state='wait_archive_unreachable', message=str(exc))
            log.warning('아카이브 인벤토리 실패: %s', exc)
            return 'wait'
        summary = {}
        for ph, scan in zip(self.phases, scans):
            ph['archive_complete'], ph['archive_incomplete'] = scan['complete'], scan['incomplete']
            ph['shipped'] = read_lines(Path(ph['grid_root']) / 'shipped_cells.txt')
            ph['local_meta'] = local_meta_keys(ph['grid_root'], ph['plan_id'], self.arch_machine)
            ph['counts'] = phase_counts(ph['requested'], ph['archive_complete'], ph['archive_incomplete'],
                                        ph['shipped'], ph['local_meta'])
            summary[ph['name']] = ph['counts']
        arch_free, loc_free = inv.get('free_bytes', 0), free_bytes(self.phases[-1]['grid_root'])
        self.write_status(phases=summary, archive_available=inv.get('available'),
                          archive_free_gib=round(arch_free / 1024**3, 1),
                          local_free_gib=round(loc_free / 1024**3, 1))
        log.info('인벤토리 %s', json.dumps(summary, ensure_ascii=False))
        for ph in self.phases:  # 로컬에 미이관 셀이 있으면 shipper 보장
            if ph['counts']['local_pending_ship']:
                self.ensure_shipper(ph)
        if all(ph['counts']['complete'] for ph in self.phases):
            self.stop_own_shippers()
            self.write_status(state='complete')
            return 'done'
        ph = next((p for p in self.phases if p['counts']['remaining_to_launch'] > 0), None)
        if ph is None:
            self.write_status(state='wait_shipping', message='발사할 셀 없음; 이관/아카이브 확인 대기')
            return 'wait'
        if not inv.get('available') or arch_free < MIN_FREE or loc_free < MIN_FREE:
            self.write_status(state='wait_disk', message='archive/local free < 20GiB or archive missing')
            return 'wait'
        return self.launch(ph)

    def prepare_phase_files(self, ph):
        tag = ph['name']
        contract_path = self.state / f'contract_{tag}.json'
        contract_path.write_text(json.dumps(ph['contract'], indent=2, ensure_ascii=False) + '\n')
        done = build_done_list(ph['skip'], ph['archive_complete'], ph['shipped'], ph['allowed'], ph['requested'])
        done_path = self.state / f'done_list_{tag}.txt'
        tmp = done_path.with_suffix('.tmp')
        tmp.write_text('\n'.join(done) + '\n')
        os.replace(tmp, done_path)
        return contract_path, done_path

    def harness_check(self, ph, contract_path, done_path):
        cmd = [sys.executable, str(HARNESS_PY), 'check-collection', '--contract', str(contract_path),
               '--plan', str(ph['plan']), '--instructions', ph['instructions'],
               '--noise-limit', str(ph['noise_limit']), '--done-list', str(done_path),
               '--grid-root', ph['grid_root'], '--machine', self.machine]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode, (res.stdout + res.stderr).strip()

    def ensure_container(self, gpu):
        name = f'robocasa-collect-n40-watch-gpu{gpu}'
        info = docker_inspect(name)
        if info is None:
            ref = docker_inspect(self.ref_container)
            if ref is None:
                raise ValueError(f'reference container {self.ref_container} missing (image/cache unknown)')
            image, cache = ref['Config']['Image'], mount_source(ref, '/cache')
            if not cache:
                raise ValueError(f'{self.ref_container}: no /cache mount')
            env = dict(os.environ, COLLECT_CONTAINER=name, COLLECT_GPU=str(gpu), MOUNT_ROOT=self.common,
                       CACHE_ROOT=cache, COLLECT_IMAGE=image)
            if self.a.dry_run:
                log.info('[dry] create container %s image=%s cache=%s', name, image, cache)
                return name
            subprocess.run(['bash', str(CREATE_CONTAINER_SH)], env=env, check=True, timeout=300,
                           stdout=subprocess.DEVNULL)
            info = docker_inspect(name)
        if info is None or container_gpu_ids(info) != [str(gpu)] \
                or mount_source(info, '/temporal_vla') != self.common:
            raise RuntimeError(f'{name}: existing container does not match GPU{gpu}/{self.common}; not replacing')
        if not self.a.dry_run:
            subprocess.run([sys.executable, str(CHECK_CONTAINER_PY), name], check=True, timeout=120)
        return name

    def ensure_shipper(self, ph):
        found = find_shippers(ph['grid_root'])
        self.status.setdefault('shippers', {})[ph['name']] = found
        if found:
            if not all(s['sha_verify_code'] for s in found):
                log.warning('%s shipper %s: 스크립트에 SHA 검증 코드 없음 — kill 하지 않음', ph['name'], found)
            return found[0]['pid']
        env = dict(os.environ, GRID_ROOT=ph['grid_root'], INTERVAL='30', QUIET_SEC='60', PARALLEL='2',
                   STAGING_CAP_GB='8', WATCH_TARGET9_OWNER=self.machine)
        if self.a.dry_run:
            log.info('[dry] start shipper GRID_ROOT=%s', ph['grid_root'])
            return None
        Path(ph['grid_root']).mkdir(parents=True, exist_ok=True)
        p = spawn_logged(['bash', str(SHIP_SH)], env, self.state / f'shipper_{ph["name"]}.log',
                         5 * 1024**2, str(REPO))
        log.info('shipper 발사 pid=%d root=%s', p.pid, ph['grid_root'])
        self.status['shippers'][ph['name']] = [{'pid': p.pid, 'own': True}]
        return p.pid

    def stop_own_shippers(self):
        """완료 시 이 watcher 가 띄운 shipper 만, 로컬 staging 에 meta.json 이 하나도 없을 때 정리."""
        for ph in self.phases:
            if self.a.dry_run or any(Path(ph['grid_root']).rglob('meta.json')):
                continue
            for s in find_shippers(ph['grid_root']):
                if s['own'] == self.machine:
                    log.info('완료: 자기 shipper pid=%d 정리', s['pid'])
                    try:
                        os.killpg(s['pid'], signal.SIGTERM)
                    except OSError as exc:
                        log.warning('shipper 정리 실패 %s', exc)

    def launch(self, ph):
        contract_path, done_path = self.prepare_phase_files(ph)
        rc, out = self.harness_check(ph, contract_path, done_path)
        if rc != 0:
            self.write_status(state='fatal_config', phase=ph['name'], message=out[-800:])
            log.error('harness check-collection 실패 (재시도 안 함): %s', out)
            return 'fatal'
        try:
            gpus, apps = query_gpus()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.write_status(state='wait_gpu_query', message=str(exc))
            return 'wait'
        gpu = choose_gpu(gpus, apps)
        if gpu is None:
            self.write_status(state='wait_no_empty_gpu', gpus=gpus)
            log.info('빈 GPU 없음')
            return 'wait'
        if not ports_free(PORT_BASE, SERVES_PER_GPU):
            self.write_status(state='wait_ports', message=f'ports {PORT_BASE}..+{SERVES_PER_GPU - 1} busy')
            return 'wait'
        try:
            container = self.ensure_container(gpu)
        except (ValueError, RuntimeError) as exc:
            self.write_status(state='fatal_container', message=str(exc))
            log.error('컨테이너: %s', exc)
            return 'fatal'
        except (OSError, subprocess.SubprocessError) as exc:
            self.write_status(state='wait_container', message=str(exc))
            return 'wait'
        self.ensure_shipper(ph)
        env = dict(os.environ, **HARNESS_ENV, HARNESS_MACHINE=self.machine,
                   HARNESS_COLLECTION_CONTRACT=str(contract_path), PLAN_JSON=str(ph['plan']),
                   GRID_ROOT=ph['grid_root'], INSTRUCTIONS=ph['instructions'], GPUS=str(gpu),
                   SERVES_PER_GPU=str(SERVES_PER_GPU), PORT_BASE=str(PORT_BASE),
                   NOISE_LIMIT=str(ph['noise_limit']), DONE_LIST=str(done_path), SERVE_MODE='host',
                   SERVE_PY=SERVE_PY, SERVE_PYTHONPATH=':'.join(
                       f'{self.common}/{p}' for p in ('lerobot/src', 'src/policies/Isaac-GR00T')) + ':' + self.common,
                   COLLECTOR_PY='/temporal_vla/' + str(REPO.relative_to(self.common)) + '/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py',
                   PYPATH_PREFIX='/temporal_vla/' + str(REPO.relative_to(self.common)), COLLECT_CONTAINER=container,
                   MOUNT_ROOT=self.common, SERVE_OMP_THREADS='4', STAGING_WAIT_GB='8',
                   CUDA_DEVICE_ORDER='PCI_BUS_ID')
        cmd = ['bash', str(COLLECT_SH)]
        n_todo = ph['counts']['remaining_to_launch']
        if self.a.dry_run:
            keys = ('GPUS', 'PLAN_JSON', 'GRID_ROOT', 'INSTRUCTIONS', 'NOISE_LIMIT', 'DONE_LIST',
                    'HARNESS_COLLECTION_CONTRACT', 'COLLECT_CONTAINER', 'PORT_BASE')
            log.info('[dry] %s todo=%d: %s %s', ph['name'], n_todo,
                     ' '.join(f'{k}={shlex.quote(env[k])}' for k in keys), shlex.join(cmd))
            self.write_status(state='dry_run_would_launch', phase=ph['name'], gpu=gpu, harness_check=out[-400:])
            return 'wait'
        attempts = self.state / 'attempts'
        attempts.mkdir(exist_ok=True)
        for old in sorted(attempts.glob('*.log'))[:-(KEEP_ATTEMPT_LOGS - 1) or None]:
            for f in (old, Path(str(old) + '.1')):
                f.unlink(missing_ok=True)
        alog = attempts / f'{time.strftime("%y%m%d%H%M%S")}_{ph["name"]}_gpu{gpu}.log'
        before = len(ph['requested'] & (ph['archive_complete'] | ph['shipped'] | ph['local_meta']))
        self.collector = spawn_logged(cmd, env, alog, 5 * 1024**2, str(REPO))
        info = {'pid': self.collector.pid, 'phase': ph['name'], 'gpu': gpu, 'log': str(alog),
                'started': time.strftime('%F %T'), 'todo': n_todo}
        self.write_status(state='collecting', collector=info)
        log.info('수집 발사 %s', info)
        while self.collector.poll() is None:
            cap_worker_logs(ph['grid_root'])
            if STOP.wait(10):
                self.write_status(state='stopped_collector_running', collector=info)
                log.info('SIGTERM: 수집기 pid=%d 유지한 채 종료', self.collector.pid)
                return 'stopped'
        rc = self.collector.returncode
        after = len(ph['requested'] & (read_lines(Path(ph['grid_root']) / 'shipped_cells.txt')
                                       | local_meta_keys(ph['grid_root'], ph['plan_id'], self.arch_machine)
                                       | ph['archive_complete']))
        progressed = after > before
        self.failures = 0 if progressed else self.failures + 1
        info.update(rc=rc, ended=time.strftime('%F %T'), progressed=after - before)
        self.write_status(state='collector_exited', collector=info, failed_attempts=self.failures)
        log.info('수집 종료 rc=%s 진척=%d (연속 무진척 %d)', rc, after - before, self.failures)
        if self.failures >= MAX_FAILED_ATTEMPTS:
            self.write_status(state='fatal_no_progress')
            return 'fatal'
        return 'launched'


def cap_worker_logs(grid_root):
    # Only this launch's logs; retain the latest 10MiB in place above 20MiB.
    for p in (Path(grid_root) / 'logs').glob('*.log'):
        try:
            if p.stat().st_size > 20 * 1024**2:
                with p.open('r+b') as f:
                    f.seek(-10 * 1024**2, 2)
                    tail = f.read()
                    f.seek(0)
                    f.write(tail)
                    f.truncate()
        except OSError as exc:
            log.warning('log cap failed %s: %s', p, exc)


def setup_logging(state_dir, console):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = logging.handlers.RotatingFileHandler(state_dir / 'watcher.log', maxBytes=10 * 1024**2, backupCount=3)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    ch = None
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return ch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--machine', choices=sorted(MACHINES))
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='read-only 감사: 컨테이너·shipper·수집 발사 안 함')
    ap.add_argument('--interval', type=int, default=1800)
    ap.add_argument('--common', help='공통 repo 루트 (기본: git common dir)')
    ap.add_argument('--log-pipe', help=argparse.SUPPRESS)
    ap.add_argument('--log-max-bytes', type=int, default=5 * 1024**2, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.log_pipe:
        return log_pipe_main(a.log_pipe, a.log_max_bytes)
    if not a.machine:
        ap.error('--machine required')
    w = Watcher(a)
    lock = None
    if not a.dry_run:  # dry-run 감사는 실행 중인 watcher 와 공존 가능 (발사 안 함)
        lock = (w.state / 'watcher.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f'[watch] {a.machine}: 다른 watcher 가 실행 중 ({w.state}/watcher.lock)')
    ch = setup_logging(w.state, console=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    log.info('start machine=%s dry=%s once=%s interval=%ds state=%s phases=%s', a.machine, a.dry_run,
             a.once, a.interval, w.state, [(p['name'], len(p['requested'])) for p in w.phases])
    rc = 0
    while not STOP.is_set():
        try:
            result = w.cycle()
        except Exception as exc:  # noqa: BLE001 — 사이클 하나의 예외로 watcher 가 죽지 않게
            log.exception('cycle error')
            w.write_status(state='cycle_error', message=repr(exc))
            result = 'wait'
        if ch and not (a.once or a.dry_run):  # 콘솔은 시작 직후 첫 사이클까지만
            log.removeHandler(ch)
            ch = None
        if result in ('done', 'fatal', 'stopped') or a.once:
            rc = 1 if result == 'fatal' else 0
            break
        STOP.wait(a.interval)
    if STOP.is_set() and w.status.get('state') != 'stopped_collector_running':
        w.write_status(state='stopped')
    log.info('exit rc=%d state=%s', rc, w.status.get('state'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
