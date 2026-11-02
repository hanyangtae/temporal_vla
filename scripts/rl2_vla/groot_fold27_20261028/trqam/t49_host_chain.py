"""Durable per-host chain for the 6 non-bread srv scenes on native exact-49 latents (srv48: DishL s3, DishR s1, OvenL s2;
srv50: candle s0, jug s1, marshmallow s4). Runs ON the collection host (cwd = deploy repo root), launched detached by
main. New orchestration only: every step is an EXISTING frozen tool invoked unchanged as a subprocess.

Per scene, in config order (one scene at a time, one GPU):
  1 archive   wait until <source49_root>/<name>/COMPLETE.json exists on the archive (read via stage_episodes'
              WrapperFetcher = scripts/utils/remote_compute.sh run); then contract.json must be the exact-49 contract
              (kind fold27_extract49_v1, this scene, 200, layer15/denoise0/tokens[0,49)/float16, with_last_block,
              spec_sha256 == local spec, extractor49_sha256 == pin) and COMPLETE == {contract, sha(contract bytes), 200}.
  2 stage     target <stage_root>/<name> (contract.json + 200 record json + npz ONLY — PROGRESS.json / COMPLETE.json /
              anything else in the staged dir is refused: latents_v2_audit and run_latents_v2.peek_records would read it
              as an episode record).
              - already verified -> adopt (no launch)
              - a live stage_episodes process targets the dir (/proc scan, any owner) -> wait; never a 2nd stager
              - stage 'adopt' scenes are never launched by this chain unless <state>/QUIESCENT_<name> exists (main writes
                it after confirming the original stager is gone) -> otherwise wait
              - own launch: space guard (remaining records x 23e6 B + reserve) BEFORE launch; any '.<npz>.part' left by
                a dead stager is MOVED (not deleted) to <stage_root>/.preserved_partials/<name>/<t>/ because
                stage_episodes unlinks its own partial on resume; then stage_episodes.py (sha-gated, json-last commit)
              - verify: every expected episode (spec scene rows) present; episode_id/pkl_sha256/success == spec;
                sha(npz) == record; last_block_d0 float16 [n_inferences,49,1536]; per fold fit 160 (j != target),
                eval 40 (j == target), disjoint, union == the scene's 200. Corruption -> stop; incomplete -> stop.
  3 latents   wait_launch.py (harness BLOCKED -> retry) -> run_latents_v2.sh --token-contract 49 --expect-records 200
              (its own parity smoke first). A PARITY_SMOKE_V2.json not passed -> stop (also before launch). After rc 0:
              COMPLETE episodes 200, contract rollout_latents_v2/49/episodes_contract_sha256 == sha(staged contract),
              parity passed for 49, latents_v2_audit.audit(..., 49, expect=200) ok. Any failure -> stop.
Then on the host:
  4 smokes    smoke fold: basic (incl. GPU exact-resume check), pbrs, mix via wait_launch -> smoke_v2b.sh with
              FOLD27_OUT=<out_root>; SMOKE_DONE.json required; RESUME_CHECK gates (coordinator v2's) required.
              Existing SMOKE_DONE -> skip; any FAILED.json -> stop.
  5 train     host_fold_coordinator_v2.py --sources <frozen host sources> --out <out_root> --gpus <g> (it waits for
              inputs, re-checks the smokes, launches train_queue_v2 per fold under the harness, verifies DONE).
State: <out_root>/t49_chain/<machine>/ CHAIN_STATE.json, CHAIN_FAILED.json (stop marker; --retry-failed renames it),
FREEZE.json (sha of config, sources, spec and every invoked tool; any change on restart -> refuse), chain.lock, logs.
Never deletes, never edits frozen tools, never touches bread / kanu outputs (refused by config validation).

    setsid nohup .fold27_env/bin/python scripts/rl2_vla/groot_fold27_20261028/trqam/t49_host_chain.py \
        --config outputs/analysis/groot_fold27_20261028/t49_chain_20261103/srv48.json > <log> 2>&1 &
"""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE), str(PKG), str(PKG / 'fold_data')]
import latents_v2_audit as AU      # noqa: E402  (frozen)
import stage_episodes as SE        # noqa: E402  (frozen: fetchers, needed(), stem())

KIND = 'fold27_t49_host_chain_v1'
EXTRACT_KIND = 'fold27_extract49_v1'
HOST_TO_ARCHIVE = {'srv48': 'worker1', 'srv50': 'worker2'}            # == host_fold_coordinator.HOST_TO_ARCHIVE (srv only)
FORBIDDEN_OUT = ('outputs/rl2_vla/fold27_trqam_20261102', 'outputs/rl2_vla/fold27_trqam_v2_20261102')
PER_RECORD_UPPER = 23_000_000      # B; extract_scene49 raw upper 4,446,761,600 / 200 = 22,233,808
TAGS = {'basic': 'basic', 'pbrs': 'pbrs_a0.1', 'mix': 'mix_a0.1_b0.0396'}   # == train_queue.TAGS (frozen)
RESUME_GATES = ('restore_exact', 'minibatch_indices_equal', 'np_rng_state_equal', 'trajectory_bitwise_equal')  # == coordinator v2
TOOLS = {'stage': PKG / 'fold_data/stage_episodes.py', 'wait': HERE / 'wait_launch.py', 'latents': HERE / 'run_latents_v2.sh',
         'latents_py': HERE / 'run_latents_v2.py', 'converter': HERE / 'rollout_latents_v2.py', 'audit': HERE / 'latents_v2_audit.py',
         'smoke': HERE / 'smoke_v2b.sh', 'coord': HERE / 'host_fold_coordinator_v2.py', 'chain': Path(__file__).resolve()}
EP_ALLOWED_META = ('contract.json',)


class ChainError(RuntimeError):
    pass


class Wait(Exception):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return SE.sha_file(p)


def scene_key(f):
    return f'{f["instruction"]}|s{f["scene"]}'


def load_config(path, repo=REPO):
    cfg = json.loads(Path(path).read_text())
    p = []
    if cfg.get('kind') != KIND:
        p.append(f'kind {cfg.get("kind")} != {KIND}')
    m = cfg.get('machine')
    if m not in HOST_TO_ARCHIVE:
        p.append(f'machine {m} not an srv host')
    if len(cfg.get('gpus', [])) != 1:
        p.append('exactly one GPU (one-GPU rule)')
    out = str(cfg.get('out_root', ''))
    if not out or out.rstrip('/') in FORBIDDEN_OUT or 'bread' in out:
        p.append(f'out_root {out!r} must be a NEW root separate from kanu/bread')
    scenes = cfg.get('scenes', [])
    keys, names = [s['key'] for s in scenes], [s['name'] for s in scenes]
    if not scenes or len(set(keys)) != len(keys) or len(set(names)) != len(names):
        p.append('scenes empty or duplicated')
    if any('bread' in k or 'bread' in n for k, n in zip(keys, names)):
        p.append('bread is out of scope (separate root, already running)')
    if any(not (isinstance(x, str) and '=' in x and x.split('=', 1)[0].isidentifier()
                and x.split('=', 1)[0] not in ('GPUS', 'HARNESS_MACHINE', 'HARNESS_SESSION', 'FOLD27_PY', 'FOLD27_OUT', 'CUDA_VISIBLE_DEVICES'))
           for x in cfg.get('extra_env', [])):
        p.append('extra_env must be NAME=value strings and may not override GPU/harness/output variables')
    if any(s.get('stage') not in ('adopt', 'own') for s in scenes):
        p.append("scene stage must be 'adopt' or 'own'")
    spec = json.loads((repo / cfg['spec']).read_text())
    want = HOST_TO_ARCHIVE.get(m)
    for k in keys:
        fs = [f for f in spec['folds'] if scene_key(f) == k]
        if not fs or any(f['archive_machine'] != want for f in fs):
            p.append(f'{k}: no folds or not collected on {m}')
    smoke = [f for f in spec['folds'] if f['fold_id'] == cfg.get('smoke_fold')]
    if not smoke or scene_key(smoke[0]) not in keys:
        p.append('smoke_fold must be a fold of a configured scene')
    src = json.loads((repo / cfg['sources']).read_text())
    exp_src = {s['key']: {'episodes': f'{cfg["stage_root"]}/{s["name"]}', 'latents': f'{cfg["latents_root"]}/{s["name"]}'}
               for s in scenes}
    if src != exp_src:
        p.append('sources file != {scene: {stage_root/name, latents_root/name}} for exactly the configured scenes')
    if p:
        raise ChainError('config refused: ' + '; '.join(p))
    return cfg, spec


def live_stagers(local_dir, proc_root=Path('/proc'), self_pid=None):
    """PIDs whose cmdline runs stage_episodes.py with --local-dir resolving to local_dir (relative args resolved against
    /proc/<pid>/cwd; unreadable cwd -> suffix match, i.e. conservative: treated as live)."""
    target, self_pid, hits = os.path.normpath(str(local_dir)), self_pid or os.getpid(), []
    for d in proc_root.iterdir():
        if not d.name.isdigit() or int(d.name) == self_pid:
            continue
        try:
            argv = (d / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        argv = [a.decode(errors='replace') for a in argv if a]
        if not any(a.endswith('stage_episodes.py') for a in argv) or '--local-dir' not in argv:
            continue
        i = argv.index('--local-dir')
        if i + 1 >= len(argv):
            continue
        arg = argv[i + 1]
        if os.path.isabs(arg):
            hit = os.path.normpath(arg) == target
        else:
            try:
                hit = os.path.normpath(os.path.join(os.readlink(d / 'cwd'), arg)) == target
            except OSError:
                hit = target.endswith(os.path.normpath(arg))
        if hit:
            hits.append(int(d.name))
    return hits


class Chain:
    def __init__(self, config, *, repo=REPO, fetcher=None, run=subprocess.run, sleep=time.sleep, now=time.time,
                 stagers=live_stagers, disk=shutil.disk_usage, retry_failed=False, poll_s=None):
        self.cfg_path, self.repo = Path(config), Path(repo)
        self.cfg, self.spec = load_config(self.cfg_path, self.repo)
        c = self.cfg
        self.fetcher = fetcher or SE.WrapperFetcher()
        self.run_, self.sleep, self.now, self.stagers, self.disk = run, sleep, now, stagers, disk
        self.retry_failed = retry_failed
        self.poll_s = poll_s if poll_s is not None else c.get('poll_s', 300)
        self.m, self.gpu = c['machine'], int(c['gpus'][0])
        self.expect = int(c.get('expect_records', 200))
        self.fit_n, self.eval_n = int(c.get('fold_fit', 160)), int(c.get('fold_eval', 40))
        self.out = self.repo / c['out_root']
        self.sd = self.out / 't49_chain' / self.m
        self.py = str(self.repo / c['python']) if not os.path.isabs(c['python']) else c['python']
        self.state = {'machine': self.m, 'kind': KIND, 'scenes': {}}
        self._lock = None

    # ---------- bookkeeping ----------
    def p(self, rel):
        return self.repo / rel

    def free(self, p):
        p = Path(p)
        while not p.exists():
            p = p.parent
        return self.disk(p).free

    def save(self, **kw):
        self.state.update(kw, time=self.now())
        tmp = self.sd / f'CHAIN_STATE.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(self.state, indent=1, default=str)); os.replace(tmp, self.sd / 'CHAIN_STATE.json')

    def fail(self, phase, why):
        body = {'phase': phase, 'reason': why, 'time': self.now(), 'state': self.state}
        with open(self.sd / 'CHAIN_FAILED.json', 'x') as f:
            f.write(json.dumps(body, indent=1, default=str))
        self.save(status='failed', phase=phase)
        raise ChainError(f'{phase}: {why}')

    def freeze_body(self):
        c = self.cfg
        return {'config_sha256': sha_file(self.cfg_path), 'sources_sha256': sha_file(self.p(c['sources'])),
                'spec_sha256': sha_file(self.p(c['spec'])), 'tools': {k: sha_file(v) for k, v in TOOLS.items()}}

    def freeze(self):
        fz, body = self.sd / 'FREEZE.json', self.freeze_body()
        if fz.exists():
            old = json.loads(fz.read_text())
            if old != body:
                diff = sorted(k for k in body if old.get(k) != body[k])
                raise ChainError(f'frozen inputs changed since first start ({diff}); use a new out_root/config')
        else:
            with open(fz, 'x') as f:
                f.write(json.dumps(body, indent=1))

    def acquire(self):
        self.sd.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.sd / 'chain.lock', 'a+')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ChainError(f'another chain holds {self.sd}/chain.lock')
        fp = self.sd / 'CHAIN_FAILED.json'
        if fp.exists():
            if not self.retry_failed:
                raise ChainError(f'{fp} present; inspect, then --retry-failed')
            os.replace(fp, fp.with_name(f'CHAIN_FAILED.prev.{int(self.now())}.json'))

    def wait_until(self, label, fn):
        deadline = self.now() + float(self.cfg.get('max_wait_h', 48)) * 3600
        while True:
            try:
                return fn()
            except Wait as w:
                self.save(status='waiting', phase=label, waiting_for=str(w))
                if self.now() + self.poll_s > deadline:
                    self.fail(label, f'deadline while waiting: {w}')
                self.sleep(self.poll_s)

    def logged(self, name, cmd, env=None):
        k = 1 + sum(1 for _ in self.sd.glob(f'{name}.*.log'))
        with open(self.sd / f'{name}.{k}.log', 'x') as f:
            f.write('# ' + ' '.join(map(str, cmd)) + '\n'); f.flush()
            return self.run_([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(self.repo), env=env).returncode

    def harness_cmd(self, label, tail, extra_env=()):
        c = self.cfg
        return [self.py, TOOLS['wait'], '--max-wait-h', str(c.get('max_wait_h', 48)), '--poll-s', str(c.get('harness_poll_s', 300)),
                '--state', self.sd / f'wait_{label}', '--', 'env', f'GPUS={self.gpu}', f'HARNESS_MACHINE={self.m}',
                f'HARNESS_SESSION={c["session"]}', f'FOLD27_PY={self.py}', *c.get('extra_env', []), *extra_env, 'bash', *tail]

    # ---------- 1 archive ----------
    def archive_gate(self, s):
        root = f'{self.cfg["source49_root"]}/{s["name"]}'
        try:
            comp = self.fetcher.read(f'{root}/COMPLETE.json')
            cb = self.fetcher.read(f'{root}/contract.json')
        except (SE.StageError, OSError, subprocess.TimeoutExpired) as e:
            raise Wait(f'archive {s["name"]} not COMPLETE/readable ({str(e)[:120]})')
        comp, con = json.loads(comp), json.loads(cb)
        why = []
        if comp.get('contract_sha256') != sha_bytes(cb) or comp.get('contract') != con:
            why.append('COMPLETE does not reference this contract.json')
        if comp.get('episodes') != self.expect:
            why.append(f'COMPLETE episodes {comp.get("episodes")} != {self.expect}')
        want = {'kind': EXTRACT_KIND, 'scene': s['key'], 'expect_records': self.expect, 'last_block_layer': 15,
                'last_block_denoise': 0, 'last_block_tokens': [0, 49], 'last_block_dtype': 'float16', 'with_last_block': True,
                'spec_sha256': sha_file(self.p(self.cfg['spec']))}
        if self.cfg.get('extractor49_sha256'):
            want['extractor49_sha256'] = self.cfg['extractor49_sha256']
        why += [f'contract {k}={con.get(k)!r} != {v!r}' for k, v in want.items() if con.get(k) != v]
        if why:
            self.fail(f'archive:{s["name"]}', '; '.join(why))
        return cb

    # ---------- 2 stage ----------
    def expected(self, key):
        return SE.needed(self.spec, scenes=[key])

    def verify_staged(self, s, contract_bytes):
        """-> ('ok'|'incomplete'|'corrupt', problems). Never writes."""
        import numpy as np
        d, key = self.p(f'{self.cfg["stage_root"]}/{s["name"]}'), s['key']
        exp = self.expected(key)
        corrupt, missing = [], []
        if len(exp) != self.expect:
            corrupt.append(f'spec scene rows {len(exp)} != {self.expect}')
        if not d.is_dir() or not (d / 'contract.json').exists():
            return 'incomplete', ['no staged contract']
        if (d / 'contract.json').read_bytes() != contract_bytes:
            corrupt.append('staged contract.json != archive contract')
        allowed = set(EP_ALLOWED_META)
        for eid, r in sorted(exp.items()):
            st = SE.stem(eid)
            jp = d / f'{st}.json'
            if not jp.exists():
                missing.append(eid); continue
            rec = json.loads(jp.read_text())
            allowed |= {jp.name, rec.get('npz', '')}
            if rec.get('episode_id') != eid or rec.get('pkl_sha256') != r['pkl_sha256'] or int(rec.get('success', -1)) != int(r['success']):
                corrupt.append(f'{eid}: identity/pkl/success'); continue
            npz = d / rec['npz']
            if not npz.exists() or sha_file(npz) != rec.get('npz_sha256'):
                corrupt.append(f'{eid}: npz sha'); continue
            with np.load(npz) as z:
                if 'last_block_d0' not in z.files or 'actions_model12' not in z.files:
                    corrupt.append(f'{eid}: arrays'); continue
                lb = z['last_block_d0']
            if lb.dtype != np.float16 or lb.shape != (int(rec.get('n_inferences', -1)), 49, 1536):
                corrupt.append(f'{eid}: last_block_d0 {lb.dtype} {lb.shape}')
        pending = {f'{SE.stem(e)}.npz' for e in missing}      # stage_episodes renames npz BEFORE its json commit marker
        extra = sorted(x.name for x in d.iterdir() if x.name not in allowed | pending and not x.name.endswith('.part'))
        if extra:
            corrupt.append(f'non-record files in staged dir (native input must hold contract+records only): {extra[:5]}')
        corrupt += self.fold_problems(key, exp)
        if corrupt:
            return 'corrupt', corrupt[:20]
        if missing or any(x.name.endswith('.part') or x.name in pending for x in d.iterdir()):
            return 'incomplete', [f'{len(missing)} missing'] + missing[:3]
        return 'ok', []

    def fold_problems(self, key, exp):
        out, union = [], set()
        for f in (f for f in self.spec['folds'] if scene_key(f) == key):
            eid = lambda r: SE.episode_id(r['plan_id'], f['archive_machine'], f['instruction'], f['scene'], r['j'], r['n'])
            fit, ev = {eid(r) for r in f['fit']}, {eid(r) for r in f['eval']}
            if len(f['fit']) != self.fit_n or any(r['j'] == f['target_j'] for r in f['fit']):
                out.append(f'{f["fold_id"]}: fit != {self.fit_n} rows with j != target')
            if len(f['eval']) != self.eval_n or any(r['j'] != f['target_j'] for r in f['eval']):
                out.append(f'{f["fold_id"]}: eval != {self.eval_n} rows with j == target')
            if fit & ev:
                out.append(f'{f["fold_id"]}: fit/eval overlap')
            union |= fit | ev
        if union != set(exp):
            out.append('fold union != scene episode set')
        return out

    def stage(self, s, contract_bytes):
        d = self.p(f'{self.cfg["stage_root"]}/{s["name"]}')
        st, probs = self.verify_staged(s, contract_bytes)
        if st == 'ok':
            return 'adopted'
        if st == 'corrupt':
            self.fail(f'stage:{s["name"]}', f'staged dir inconsistent: {probs}')
        live = self.stagers(d)
        if live:
            raise Wait(f'stage_episodes already running for {s["name"]} (pids {live}); not starting a duplicate')
        if s['stage'] == 'adopt' and not (self.sd / f'QUIESCENT_{s["name"]}').exists():
            raise Wait(f'adopt scene {s["name"]} incomplete ({probs[:1]}) and no live stager: main must confirm quiescence '
                       f'by creating {self.sd}/QUIESCENT_{s["name"]}')
        done = sum(1 for e in self.expected(s['key']) if (d / f'{SE.stem(e)}.json').exists())
        need = (self.expect - done) * PER_RECORD_UPPER + float(self.cfg.get('stage_reserve_gib', 20)) * 1024 ** 3
        free = self.free(d)
        if free < need:
            self.fail(f'stage:{s["name"]}', f'space guard: free {free} < need {int(need)} (before launch)')
        if d.exists():
            parts = sorted(d.glob('.*.part'))
            if parts:
                keep = self.p(self.cfg['stage_root']) / '.preserved_partials' / s['name'] / str(int(self.now()))
                keep.mkdir(parents=True, exist_ok=False)
                for x in parts:
                    os.replace(x, keep / x.name)      # preserved, not deleted (stage_episodes would unlink it)
        rc = self.logged(f'stage_{s["name"]}', [self.cfg.get('stage_python', 'python3'), TOOLS['stage'], '--spec', self.cfg['spec'],
                                                 '--remote-dir', f'{self.cfg["source49_root"]}/{s["name"]}',
                                                 '--local-dir', f'{self.cfg["stage_root"]}/{s["name"]}', '--scene', s['key'],
                                                 '--reserve-gib', str(self.cfg.get('stage_reserve_gib', 20))])
        if rc:
            self.fail(f'stage:{s["name"]}', f'stage_episodes rc={rc}')
        st, probs = self.verify_staged(s, contract_bytes)
        if st != 'ok':
            self.fail(f'stage:{s["name"]}', f'after staging: {st} {probs}')
        return 'staged'

    # ---------- 3 latents ----------
    def parity_state(self, lat):
        p = lat / 'PARITY_SMOKE_V2.json'
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return (d.get('provenance') or {}).get('token_contract') == 49 and (d.get('verdict') or {}).get('passed') is True

    def latents_problems(self, s):
        ep, lat = self.p(f'{self.cfg["stage_root"]}/{s["name"]}'), self.p(f'{self.cfg["latents_root"]}/{s["name"]}')
        pr = []
        if self.parity_state(lat) is not True:
            pr.append('parity smoke v2 (49) not passed')
        c = json.loads((lat / 'contract.json').read_text()) if (lat / 'contract.json').exists() else {}
        if c.get('version') != 'rollout_latents_v2' or c.get('token_contract') != 49:
            pr.append(f'contract {c.get("version")}/{c.get("token_contract")}')
        if c.get('episodes_contract_sha256') != sha_file(ep / 'contract.json'):
            pr.append('episodes_contract_sha256 != staged contract')
        comp = lat / 'COMPLETE.json'
        if not comp.exists() or json.loads(comp.read_text()).get('episodes') != self.expect:
            pr.append(f'COMPLETE episodes != {self.expect}')
        a = AU.audit(ep, lat, 49, expect=self.expect)
        if not a['ok']:
            pr.append(f'audit: { {k: v[:3] for k, v in a["problems"].items() if v} }')
        return pr

    def latents(self, s):
        lat = self.p(f'{self.cfg["latents_root"]}/{s["name"]}')
        if (lat / 'COMPLETE.json').exists():
            pr = self.latents_problems(s)
            if pr:
                self.fail(f'latents:{s["name"]}', f'existing COMPLETE but {pr}')
            return 'adopted'
        if self.parity_state(lat) is False:
            self.fail(f'latents:{s["name"]}', 'existing PARITY_SMOKE_V2.json not passed for 49')
        free = self.free(lat)
        if free < float(self.cfg.get('latents_reserve_gib', 10)) * 1024 ** 3:
            self.fail(f'latents:{s["name"]}', f'space guard: free {free} below latents reserve')
        rc = self.logged(f'latents_{s["name"]}', self.harness_cmd(
            f'latents_{s["name"]}', [TOOLS['latents'], '--token-contract', '49', '--episodes', f'{self.cfg["stage_root"]}/{s["name"]}',
                                     '--out', f'{self.cfg["latents_root"]}/{s["name"]}', '--expect-records', str(self.expect)]))
        if self.parity_state(lat) is False:
            self.fail(f'latents:{s["name"]}', 'parity smoke v2 failed')
        if rc == 75:
            self.fail(f'latents:{s["name"]}', 'harness still BLOCKED at the wait deadline')
        if rc:
            self.fail(f'latents:{s["name"]}', f'run_latents_v2 rc={rc}')
        pr = self.latents_problems(s)
        if pr:
            self.fail(f'latents:{s["name"]}', f'after conversion: {pr}')
        return 'converted'

    # ---------- 4 smokes ----------
    def resume_ok(self, fold):
        rc = self.out / fold / 'resume_check_v2' / 'RESUME_CHECK.json'
        if not rc.exists():
            return False, 'RESUME_CHECK.json missing'
        r = json.loads(rc.read_text())
        bad = [g for g in RESUME_GATES if r.get(g) is not True] + ([] if r.get('backend') == 'gpu' else ['backend'])
        return (not bad), bad

    def smokes(self):
        fold = self.cfg['smoke_fold']
        key = scene_key(next(f for f in self.spec['folds'] if f['fold_id'] == fold))
        s = next(x for x in self.cfg['scenes'] if x['key'] == key)
        done = []
        for arm, tag in TAGS.items():
            d = self.out / fold / f'{tag}_smoke'
            if (d / 'FAILED.json').exists():
                self.fail(f'smoke:{arm}', f'{d}/FAILED.json present')
            if not (d / 'SMOKE_DONE.json').exists():
                rc = self.logged(f'smoke_{arm}', self.harness_cmd(
                    f'smoke_{arm}', [TOOLS['smoke'], fold, arm, f'{self.cfg["stage_root"]}/{s["name"]}', f'{self.cfg["latents_root"]}/{s["name"]}'],
                    extra_env=(f'FOLD27_OUT={self.cfg["out_root"]}',)))
                if rc == 75:
                    self.fail(f'smoke:{arm}', 'harness still BLOCKED at the wait deadline')
                if rc or not (d / 'SMOKE_DONE.json').exists():
                    self.fail(f'smoke:{arm}', f'smoke_v2b rc={rc}, SMOKE_DONE {(d / "SMOKE_DONE.json").exists()}')
            if arm == 'basic':
                ok, bad = self.resume_ok(fold)
                if not ok:
                    self.fail('smoke:resume', f'GPU exact-resume check failed: {bad}')
            done.append(arm)
            self.save(status='smoking', smokes_done=done)
        return done

    # ---------- 5 train ----------
    def train(self):
        c = self.cfg
        cmd = [self.py, TOOLS['coord'], '--machine', self.m, '--spec', c['spec'], '--sources', c['sources'], '--out', c['out_root'],
               '--gpus', str(self.gpu), '--session', c['session'], '--python', self.py, '--smoke-fold', c['smoke_fold'],
               '--max-wait-h', str(c.get('max_wait_h', 48)), '--harness-poll-s', str(c.get('harness_poll_s', 300)),
               *(['--reuse-ae-from', c['reuse_ae_from']] if c.get('reuse_ae_from') else [])]
        rc = self.logged('coordinator_v2', cmd)
        if rc:
            self.fail('train', f'host_fold_coordinator_v2 rc={rc} (see its own state/FAILED under {c["out_root"]}/queue)')

    # ---------- main ----------
    def run(self):
        self.acquire()
        self.freeze()
        self.save(status='started', plan=[s['name'] for s in self.cfg['scenes']])
        for s in self.cfg['scenes']:
            st = self.state['scenes'].setdefault(s['name'], {})
            cb = self.wait_until(f'archive:{s["name"]}', lambda: self.archive_gate(s))
            st['archive_contract_sha256'] = sha_bytes(cb); self.save(status='archive_ok', phase=s['name'])
            st['stage'] = self.wait_until(f'stage:{s["name"]}', lambda: self.stage(s, cb)); self.save(status='staged', phase=s['name'])
            st['latents'] = self.latents(s); self.save(status='latents_ok', phase=s['name'])
        self.save(status='smoking', smokes_done=self.smokes())
        self.save(status='training', phase='coordinator_v2')
        self.train()
        self.save(status='done', phase='done')
        return self.state


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--check-config', action='store_true', help='validate config/sources/spec only, no state, no launch')
    a = ap.parse_args(argv)
    try:
        if a.check_config:
            cfg, _ = load_config(a.config)
            print(json.dumps({'config_ok': True, 'machine': cfg['machine'], 'scenes': [s['name'] for s in cfg['scenes']]}))
            return 0
        print(json.dumps(Chain(a.config, retry_failed=a.retry_failed).run(), default=str))
        return 0
    except ChainError as e:
        print('T49_CHAIN_ERROR', e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
