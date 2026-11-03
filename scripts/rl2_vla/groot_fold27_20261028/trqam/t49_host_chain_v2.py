"""Versioned host chain: t49_host_chain (v1) with PER-SCENE archive source (kind / root / extractor sha pin), so a
scene re-extracted under extract_scene49_v2 (collector-semantics success contract, kind fold27_extract49_v2) can run
next to scenes from extract_scene49 v1. Everything after the archive gate (single-owner staging, staged-dir contract,
native run_latents_v2.sh token 49, parity stop, smokes incl. exact resume, host_fold_coordinator_v2) is v1's code,
inherited unchanged. t49_host_chain.py itself is NOT modified (srv48's running chain pins its sha in FREEZE.json).

Config kind 'fold27_t49_host_chain_v2'. Differences from v1 config:
  - no top-level source49_root / extractor49_sha256;
  - every scene carries 'archive_kind' (fold27_extract49_v1 | fold27_extract49_v2), 'source49_root' (absolute archive
    root holding <name>/), 'extractor49_sha256' (64 hex, == archive contract.extractor49_sha256).
  - v2 archives must additionally carry success_rule == 'collector_v2'.
Same refusals as v1 (bread, kanu, >1 GPU, old roots, sources mismatch), plus out_root must differ from the v1 configs'.

    setsid nohup .fold27_env/bin/python scripts/rl2_vla/groot_fold27_20261028/trqam/t49_host_chain_v2.py \
        --config outputs/analysis/groot_fold27_20261028/t49_chain_20261103/srv50_candle_v2.json > <log> 2>&1 < /dev/null &
"""
import argparse, contextlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t49_host_chain as CH      # noqa: E402  (v1, unchanged)

KIND = 'fold27_t49_host_chain_v2'
ARCHIVE_KINDS = {'fold27_extract49_v1': None, 'fold27_extract49_v2': 'collector_v2'}   # kind -> required success_rule
EXTRACTORS = {'fold27_extract49_v1': CH.PKG / 'fold_data/extract_scene49.py', 'fold27_extract49_v2': CH.PKG / 'fold_data/extract_scene49_v2.py'}
TAKEN_OUT = ('outputs/rl2_vla/fold27_trqam_t49_hosts_20261103', 'outputs/rl2_vla/fold27_trqam_t49_srv50_independent_20261103')


def load_config(path, repo=CH.REPO):
    cfg = json.loads(Path(path).read_text())
    p = []
    if cfg.get('kind') != KIND:
        p.append(f'kind {cfg.get("kind")} != {KIND}')
    if 'source49_root' in cfg or 'extractor49_sha256' in cfg:
        p.append('v2 config: archive root / extractor pin are per scene only')
    m = cfg.get('machine')
    if m not in CH.HOST_TO_ARCHIVE:
        p.append(f'machine {m} not an srv host')
    if len(cfg.get('gpus', [])) != 1:
        p.append('exactly one GPU (one-GPU rule)')
    out = str(cfg.get('out_root', '')).rstrip('/')
    if not out or out in CH.FORBIDDEN_OUT or out in TAKEN_OUT or 'bread' in out:
        p.append(f'out_root {out!r} must be a NEW root (not kanu/bread/existing t49 chains)')
    if any(not (isinstance(x, str) and '=' in x and x.split('=', 1)[0].isidentifier()
                and x.split('=', 1)[0] not in ('GPUS', 'HARNESS_MACHINE', 'HARNESS_SESSION', 'FOLD27_PY', 'FOLD27_OUT', 'CUDA_VISIBLE_DEVICES'))
           for x in cfg.get('extra_env', [])):
        p.append('extra_env must be NAME=value strings and may not override GPU/harness/output variables')
    scenes = cfg.get('scenes', [])
    keys, names = [s.get('key') for s in scenes], [s.get('name') for s in scenes]
    if not scenes or len(set(keys)) != len(keys) or len(set(names)) != len(names):
        p.append('scenes empty or duplicated')
    if any('bread' in str(k) or 'bread' in str(n) for k, n in zip(keys, names)):
        p.append('bread is out of scope')
    for s in scenes:
        if s.get('stage') not in ('adopt', 'own'):
            p.append(f"{s.get('name')}: stage must be 'adopt' or 'own'")
        if s.get('archive_kind') not in ARCHIVE_KINDS:
            p.append(f"{s.get('name')}: archive_kind {s.get('archive_kind')!r} not in {sorted(ARCHIVE_KINDS)}")
        if not str(s.get('source49_root', '')).startswith('/'):
            p.append(f"{s.get('name')}: source49_root must be absolute")
        if s.get('archive_kind') == 'fold27_extract49_v2' and 'fold27_extract49_v1' in str(s.get('source49_root')):
            p.append(f"{s.get('name')}: v2 archive kind under a v1 root")
        if not re.fullmatch(r'[0-9a-f]{64}', str(s.get('extractor49_sha256', ''))):
            p.append(f"{s.get('name')}: extractor49_sha256 pin (64 hex) required")
    spec = json.loads((repo / cfg['spec']).read_text())
    want = CH.HOST_TO_ARCHIVE.get(m)
    for k in keys:
        fs = [f for f in spec['folds'] if CH.scene_key(f) == k]
        if not fs or any(f['archive_machine'] != want for f in fs):
            p.append(f'{k}: no folds or not collected on {m}')
    smoke = [f for f in spec['folds'] if f['fold_id'] == cfg.get('smoke_fold')]
    if not smoke or CH.scene_key(smoke[0]) not in keys:
        p.append('smoke_fold must be a fold of a configured scene')
    src = json.loads((repo / cfg['sources']).read_text())
    exp_src = {s['key']: {'episodes': f'{cfg["stage_root"]}/{s["name"]}', 'latents': f'{cfg["latents_root"]}/{s["name"]}'}
               for s in scenes}
    if src != exp_src:
        p.append('sources file != {scene: {stage_root/name, latents_root/name}} for exactly the configured scenes')
    if p:
        raise CH.ChainError('config refused: ' + '; '.join(p))
    return cfg, spec


class ChainV2(CH.Chain):
    def __init__(self, config, *, repo=CH.REPO, fetcher=None, run=subprocess.run, sleep=time.sleep, now=time.time,
                 stagers=CH.live_stagers, disk=shutil.disk_usage, retry_failed=False, poll_s=None):
        # same attributes as CH.Chain.__init__, but validated by the v2 loader (v1 __init__ hard-wires the v1 loader)
        self.cfg_path, self.repo = Path(config), Path(repo)
        self.cfg, self.spec = load_config(self.cfg_path, self.repo)
        c = self.cfg
        self.fetcher = fetcher or CH.SE.WrapperFetcher()
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

    def freeze_body(self):
        b = super().freeze_body()
        b['tools'].update(chain_v2=CH.sha_file(Path(__file__).resolve()),
                          **{f'extractor:{k}': CH.sha_file(v) for k, v in EXTRACTORS.items() if v.exists()})
        b['archives'] = {s['name']: [s['archive_kind'], s['source49_root'], s['extractor49_sha256']] for s in self.cfg['scenes']}
        return b

    @contextlib.contextmanager
    def scene_archive(self, s):
        """v1 methods read cfg['source49_root'] / cfg['extractor49_sha256']; bind them to this scene for the call."""
        keep = {k: self.cfg.get(k) for k in ('source49_root', 'extractor49_sha256')}
        self.cfg['source49_root'], self.cfg['extractor49_sha256'] = s['source49_root'], s['extractor49_sha256']
        try:
            yield
        finally:
            for k, v in keep.items():
                if v is None:
                    self.cfg.pop(k, None)
                else:
                    self.cfg[k] = v

    def archive_gate(self, s):
        kind = s['archive_kind']
        orig_kind = CH.EXTRACT_KIND
        CH.EXTRACT_KIND = kind                    # v1 gate compares contract kind against this module constant
        try:
            with self.scene_archive(s):
                cb = super().archive_gate(s)
        finally:
            CH.EXTRACT_KIND = orig_kind
        rule = ARCHIVE_KINDS[kind]
        if rule is not None:
            con = json.loads(cb)
            if con.get('success_rule') != rule:
                self.fail(f'archive:{s["name"]}', f'contract success_rule {con.get("success_rule")!r} != {rule!r}')
        return cb

    def stage(self, s, contract_bytes):
        with self.scene_archive(s):
            return super().stage(s, contract_bytes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--check-config', action='store_true', help='validate config/sources/spec only, no state, no launch')
    a = ap.parse_args(argv)
    try:
        if a.check_config:
            cfg, _ = load_config(a.config)
            print(json.dumps({'config_ok': True, 'kind': KIND, 'machine': cfg['machine'],
                              'scenes': [[s['name'], s['archive_kind']] for s in cfg['scenes']]}))
            return 0
        print(json.dumps(ChainV2(a.config, retry_failed=a.retry_failed).run(), default=str))
        return 0
    except CH.ChainError as e:
        print('T49_CHAIN_V2_ERROR', e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
