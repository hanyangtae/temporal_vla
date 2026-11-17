#!/usr/bin/env python3
"""RoboCasa grid store lossless zstd compression — frozen-manifest, verify-before-unlink.

Runs ON the archive node (stdlib only + `zstd` CLI; python>=3.8). Code reaches the node via git
(scripts/utils/remote_compute.sh), never by pasting source.

Subcommands
  build    walk <root> (no symlink following) and write a FROZEN allowlist manifest of exact regular
           files (stat + sha256) plus exclusions / symlinks / protected-coordinate reports.
           --no-hash = survey only (counts/bytes); `run` refuses such a manifest.
  run      process a frozen manifest. --mode verify: compress to a temp file, stream-decompress and
           compare sha256, then delete the temp file (sources untouched, no .zst kept).
           --mode execute: same, then recheck source stat+sha256 and open writers, atomically
           finalize <file>.zst (link, never overwrite), re-verify it, and ONLY THEN unlink the source.
  report   summarize a state dir journal.
  sample   compress+decompress named files into a scratch dir and report ratio/time/sha
           (sources are never modified; scratch outputs deleted unless --keep).

Eligibility (build)
  * regular file by lstat; symlinks are recorded, never followed for processing, and their resolved
    targets are PROTECTED together with everything under them; nlink == 1 (hardlinks preserved);
    suffix in --include-suffix (default .pkl); size >= --min-size; no existing sibling <name>.zst;
  * lives directly inside a rollout dir (dir holding meta.json) whose meta agrees with its path coord;
  * not under a path component in --exclude-dir (shard_robocasa, derived, checkpoints, videos);
  * not under a --protect-plan tree, and not the same WORLD as any protected rollout (physical or
    reached through a symlink): same task family (instruction key prefix; side/object suffix ignored —
    conservative) and same env_seed and compatible jitter (jitter_reset_idx, base_lat, base_back; a
    missing field matches anything — conservative). Index-only coincidences (same key/s/j, different
    env_seed = different kitchen) stay eligible and are counted in summary.json;
  * rollout.pkl sha256 must equal meta.json pkl_sha256 when present.

Restart safety: every step is journaled (fsync) in <state-dir>/journal.jsonl. Temp files are
'.<name>.zst.tmp-<runid>' next to the source; removed on failure/interrupt and on the next start.
A crash after finalize but before unlink is resumed by re-verifying the .zst and rehashing the source.
A .zst this state dir did not finalize is a conflict and is left alone.
"""
import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time

TOOL = 'zstd_store/1'
CHUNK = 8 << 20
DEFAULT_EXCLUDE_DIRS = ('shard_robocasa', 'derived', 'checkpoints', 'videos')
COORD_RE = re.compile(r'^(?P<plan>[0-9a-f]{12})/(?P<machine>[^/]+)/(?P<key>.+?)/s(?P<s>\d+)'
                      r'(?:/(?P<jk>[jk])(?P<j>\d+))?/n(?P<n>\d+)/(?P<arm>[^/]+)$')
MANIFEST_COLS = ['rel_path', 'size', 'mtime_ns', 'ctime_ns', 'dev', 'ino', 'nlink', 'mode', 'uid', 'gid',
                 'sha256', 'plan_id', 'machine', 'instruction', 'scene', 'jitter', 'noise', 'arm', 'env_seed']
STAT_KEYS = ('size', 'mtime_ns', 'ctime_ns', 'dev', 'ino', 'nlink', 'mode', 'uid', 'gid')
TERMINAL = {'done', 'done_resumed', 'skip_incompressible', 'skip_source_changed', 'skip_source_missing',
            'conflict_final_exists', 'conflict_final_bad', 'fail_verify'}
EXIT_OK, EXIT_ERR, EXIT_INTERRUPTED, EXIT_LOCKED, EXIT_LOW_DISK = 0, 1, 3, 4, 5

# test hooks: name -> callable(**info). Never set in production.
HOOKS = {}


class StoreError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- helpers

def _hook(name, **kw):
    fn = HOOKS.get(name)
    if fn is not None:
        fn(**kw)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stat_sig(st):
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns, 'dev': st.st_dev,
            'ino': st.st_ino, 'nlink': st.st_nlink, 'mode': st.st_mode, 'uid': st.st_uid, 'gid': st.st_gid}


def free_bytes(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize


def zstd_bin():
    for d in os.environ.get('PATH', '').split(os.pathsep):
        p = os.path.join(d, 'zstd')
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise StoreError('zstd CLI not found on PATH')


def compress_stream(src, dst, level, threads):
    """Read src in python (hashing), pipe into zstd, zstd writes dst (O_EXCL). Returns (sha256, nbytes)."""
    h = hashlib.sha256()
    n = 0
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as out:
        p = subprocess.Popen([zstd_bin(), '-q', '-c', '-%d' % level, '-T%d' % threads, '--no-progress'],
                             stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE)
        try:
            with open(src, 'rb') as f:
                while True:
                    b = f.read(CHUNK)
                    if not b:
                        break
                    h.update(b)
                    n += len(b)
                    p.stdin.write(b)
            p.stdin.close()
            err = p.stderr.read()
            rc = p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise
        if rc != 0:
            raise StoreError('zstd compress rc=%d: %s' % (rc, err.decode(errors='replace')[:300]))
        out.flush()
        os.fsync(out.fileno())
    return h.hexdigest(), n


def decompress_sha256(zpath):
    """Streaming `zstd -d -c` -> (sha256 of decompressed bytes, nbytes); raises StoreError on failure."""
    h = hashlib.sha256()
    n = 0
    p = subprocess.Popen([zstd_bin(), '-d', '-q', '-c', '--no-progress', zpath],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
        err = p.stderr.read()
        rc = p.wait()
    except BaseException:
        p.kill()
        p.wait()
        raise
    if rc != 0:
        raise StoreError('zstd decompress rc=%d: %s' % (rc, err.decode(errors='replace')[:300]))
    return h.hexdigest(), n


def open_by_any_process(st):
    """True if a visible process holds an fd on (dev, ino). Other users' /proc/<pid>/fd is unreadable
    without root — the min-age + stat recheck + sibling-partial checks cover that gap."""
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        fd_dir = '/proc/%s/fd' % pid
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                s = os.stat(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if s.st_dev == st.st_dev and s.st_ino == st.st_ino:
                return True
    return False


def sibling_partials(path, own_tmp_prefix):
    """Hidden siblings like rsync temp '.<name>.XXXXXX' signal an in-flight writer."""
    d, name = os.path.split(path)
    return [e for e in os.listdir(d) if e.startswith('.' + name) and not e.startswith(own_tmp_prefix)]


def has_symlink_component(path, stop_at):
    p, stop_at = os.path.abspath(path), os.path.abspath(stop_at)
    while True:
        if os.path.islink(p):
            return True
        if p == stop_at or len(p) <= len(stop_at):
            return False
        p = os.path.dirname(p)


# ----------------------------------------------------------------------------- coordinates / worlds

def parse_coord(rel_dir):
    m = COORD_RE.match(rel_dir)
    if not m:
        return None
    return {'plan_id': m.group('plan'), 'machine': m.group('machine'), 'instruction': m.group('key'),
            'scene': int(m.group('s')), 'jitter_kind': m.group('jk') or '',
            'jitter': int(m.group('j')) if m.group('j') is not None else None,
            'noise': int(m.group('n')), 'arm': m.group('arm')}


def _num(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def world_of(coord, meta, plan_json=None):
    """(family, env_seed, (reset_idx, lat, back)); raises StoreError if meta/path/plan disagree."""
    for k_meta, k_coord in (('grid_instruction', 'instruction'), ('scene_idx', 'scene'), ('noise_idx', 'noise')):
        if meta.get(k_meta) is not None and str(meta[k_meta]) != str(coord[k_coord]):
            raise StoreError('meta %s=%r != path %r' % (k_meta, meta[k_meta], coord[k_coord]))
    if coord['jitter_kind'] == 'j' and meta.get('jitter_idx') is not None \
            and str(meta['jitter_idx']) != str(coord['jitter']):
        raise StoreError('meta jitter_idx=%r != path j%r' % (meta['jitter_idx'], coord['jitter']))
    if meta.get('env_seed') in (None, ''):
        raise StoreError('meta has no env_seed')
    env_seed = int(meta['env_seed'])
    key = coord['instruction']
    if plan_json is not None:
        ins = (plan_json.get('instructions') or {}).get(key)
        if isinstance(ins, list) and coord['scene'] < len(ins) and isinstance(ins[coord['scene']], int) \
                and ins[coord['scene']] != env_seed:
            raise StoreError('plan instructions[%s][%d]=%r != meta env_seed %r'
                             % (key, coord['scene'], ins[coord['scene']], env_seed))
    reset = _num(meta.get('jitter_reset_idx'))
    if reset is None and coord['jitter_kind'] == 'k':
        reset = float(coord['jitter'])
    lat, back = _num(meta.get('base_lat')), _num(meta.get('base_back'))
    if plan_json is not None and coord['jitter_kind'] == 'j':
        try:
            jd = plan_json['jitters'][key][coord['scene']][coord['jitter']]
            if isinstance(jd, dict):
                reset = reset if reset is not None else _num(jd.get('reset_idx'))
                lat = lat if lat is not None else _num(jd.get('lat'))
                back = back if back is not None else _num(jd.get('back'))
        except (KeyError, IndexError, TypeError):
            pass
    return key.split('/')[0], env_seed, (reset, lat, back)


def jitter_compatible(a, b):
    for x, y in zip(a, b):
        if x is not None and y is not None and abs(x - y) > 1e-9:
            return False
    return True


# ----------------------------------------------------------------------------- build

def _read_json(path):
    with open(path) as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise StoreError('%s is not a JSON object' % path)
    return m


def _clean(x):
    return str(x).replace('\t', ' ').replace('\n', ' ')


def cmd_build(a):
    root = os.path.realpath(a.root)
    if os.path.abspath(a.root) != root:
        raise StoreError('root %s has a symlink component; pass the realpath %s' % (a.root, root))
    out = os.path.abspath(a.out)
    if out == root or out.startswith(root + os.sep):
        raise StoreError('out dir must be outside root')
    if os.path.exists(out) and os.listdir(out):
        raise StoreError('refuse: out dir %s not empty' % out)
    os.makedirs(out, exist_ok=True)
    excl_dirs = set(a.exclude_dir or DEFAULT_EXCLUDE_DIRS)
    suffixes = tuple(a.include_suffix or ['.pkl'])
    protect = set(a.protect_plan or [])
    plan_cache = {}
    t0 = time.time()

    def plan_json(pid):
        if pid not in plan_cache:
            p = os.path.join(root, pid, 'collection_plan.json')
            try:
                plan_cache[pid] = _read_json(p) if os.path.isfile(p) and not os.path.islink(p) else None
            except (OSError, ValueError, StoreError):
                plan_cache[pid] = None
        return plan_cache[pid]

    symlinks = []          # (rel, link_text, resolved, kind)
    excluded = []          # (rel, reason, size, detail)
    rollout_dirs = []      # (rel_dir, [regular file names], in_protect)
    protected_coords = {}  # (plan, machine, key, s, jkind, j, n, arm) -> 'physical'|'symlink'
    protected_worlds = []  # (family, env_seed|None, jit, rel_dir)
    link_targets = set()

    def note_link(rel, full):
        res = os.path.realpath(full)
        kind = 'dir' if os.path.isdir(full) else ('file' if os.path.exists(full) else 'dangling')
        symlinks.append((rel, os.readlink(full), res, kind))
        link_targets.add(res)
        return kind

    def add_protected(rel_dir, meta_path, how):
        c = parse_coord(rel_dir)
        if c is None:
            excluded.append((rel_dir + '/', 'protected_unparseable_coord', 0, how))
            return
        key = (c['plan_id'], c['machine'], c['instruction'], c['scene'], c['jitter_kind'], c['jitter'],
               c['noise'], c['arm'])
        protected_coords.setdefault(key, how)
        try:
            fam, es, jit = world_of(c, _read_json(meta_path), plan_json(c['plan_id']))
            protected_worlds.append((fam, es, jit, rel_dir))
        except (OSError, ValueError, StoreError) as e:
            # protected but unidentifiable: protect the whole task family conservatively
            protected_worlds.append((c['instruction'].split('/')[0], None, (None, None, None), rel_dir))
            excluded.append((rel_dir + '/meta.json', 'protected_meta_unreadable', 0, str(e)[:200]))

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = '' if rel_dir == '.' else rel_dir
        parts = rel_dir.split('/') if rel_dir else []
        in_protect = bool(parts) and parts[0] in protect
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            rel = os.path.join(rel_dir, d) if rel_dir else d
            if os.path.islink(full):
                kind = note_link(rel, full)
                if in_protect and kind == 'dir':
                    # read-only, only to learn which worlds the protected plan references
                    if os.path.isfile(os.path.join(full, 'meta.json')):
                        add_protected(rel, os.path.join(full, 'meta.json'), 'symlink')
                    else:
                        for sub, _, fs in os.walk(full, followlinks=False):
                            if 'meta.json' in fs:
                                srel = os.path.normpath(os.path.join(rel, os.path.relpath(sub, full)))
                                add_protected(srel, os.path.join(sub, 'meta.json'), 'symlink')
                continue
            if d in excl_dirs:
                excluded.append((rel + '/', 'excluded_dir_' + d, 0, ''))
                continue
            keep.append(d)
        dirnames[:] = keep
        regular, has_meta_link = [], False
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.join(rel_dir, fn) if rel_dir else fn
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                note_link(rel, full)
                has_meta_link = has_meta_link or fn == 'meta.json'
                continue
            if not stat.S_ISREG(st.st_mode):
                excluded.append((rel, 'not_regular', 0, oct(st.st_mode)))
                continue
            regular.append(fn)
        if 'meta.json' in regular or has_meta_link:
            if in_protect:
                add_protected(rel_dir, os.path.join(dirpath, 'meta.json'),
                              'physical' if 'meta.json' in regular else 'symlink')
            rollout_dirs.append((rel_dir, regular, in_protect, has_meta_link))
        else:
            for fn in regular:
                rel = os.path.join(rel_dir, fn) if rel_dir else fn
                excluded.append((rel, 'protected_plan' if in_protect else 'not_rollout_dir',
                                 os.lstat(os.path.join(dirpath, fn)).st_size, ''))

    # symlinks OUTSIDE root (workspaces, dataset aliases) that resolve INTO root: protect their targets.
    external_hits = 0
    for scan in (a.external_link_scan or []):
        scan = os.path.abspath(scan)
        for dp, dns, fns in os.walk(scan, followlinks=False, onerror=lambda e: None):
            if dp == root or dp.startswith(root + os.sep):
                dns[:] = []
                continue
            for nm in dns + fns:
                full = os.path.join(dp, nm)
                if not os.path.islink(full):
                    continue
                try:
                    res = os.path.realpath(full)
                except OSError:
                    continue
                if res == root or res.startswith(root + os.sep):
                    symlinks.append((full, os.readlink(full), res, 'external'))
                    link_targets.add(res)
                    external_hits += 1

    n_prot = len(protected_coords)
    if a.expect_protected is not None and n_prot != a.expect_protected:
        raise StoreError('protected coordinate count %d != expected %d (plans %s) — refusing to build'
                         % (n_prot, a.expect_protected, sorted(protect)))
    if protect and n_prot == 0:
        raise StoreError('protect plan(s) %s matched no rollout coordinates' % sorted(protect))
    n_worlds = len({(w[0], w[1], w[2]) for w in protected_worlds})
    if a.expect_protected_worlds is not None and n_worlds != a.expect_protected_worlds:
        raise StoreError('protected world count %d != expected %d — refusing to build'
                         % (n_worlds, a.expect_protected_worlds))

    def under_link_target(full):
        p = full
        while len(p) >= len(root):
            if p in link_targets:
                return True
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return False

    by_family = {}
    for fam, es, jit, rel in protected_worlds:
        by_family.setdefault(fam, []).append((es, jit, rel))

    def protected_match(fam, es, jit):
        for pe, pj, prel in by_family.get(fam, ()):
            if pe is None or (pe == es and jitter_compatible(pj, jit)):
                return prel
        return None

    prot_idx = {(k[2], k[3], k[5]) for k in protected_coords}
    index_only = 0
    cross_plan = {}  # 'plan|arm|n<noise>' -> rollout dirs protected by world match
    candidates = []
    for rel_dir, files, in_protect, meta_is_link in rollout_dirs:
        dfull = os.path.join(root, rel_dir)
        if in_protect:
            for fn in files:
                excluded.append((os.path.join(rel_dir, fn), 'protected_plan',
                                 os.lstat(os.path.join(dfull, fn)).st_size, ''))
            continue
        c = parse_coord(rel_dir)
        reason, detail, meta, world = None, '', None, None
        if c is None:
            reason = 'unparseable_coord'
        elif meta_is_link:
            reason = 'meta_is_symlink'
        else:
            try:
                meta = _read_json(os.path.join(dfull, 'meta.json'))
                if meta.get('plan_id') is not None and str(meta['plan_id']) != c['plan_id']:
                    raise StoreError('meta plan_id %r != path %r' % (meta['plan_id'], c['plan_id']))
                world = world_of(c, meta, plan_json(c['plan_id']))
                hit = protected_match(*world)
                if hit is not None:
                    reason, detail = 'protected_world_match', hit
                    k = '%s|%s|n%d' % (c['plan_id'], c['arm'], c['noise'])
                    cross_plan[k] = cross_plan.get(k, 0) + 1
                elif (c['instruction'], c['scene'], c['jitter']) in prot_idx:
                    index_only += 1
            except (OSError, ValueError, StoreError) as e:
                reason, detail = 'meta_unidentifiable', str(e)[:200]
        if under_link_target(dfull):
            reason, detail = 'symlink_target', ''
        for fn in files:
            full = os.path.join(dfull, fn)
            rel = os.path.join(rel_dir, fn)
            st = os.lstat(full)
            r = reason
            if r is None:
                if not fn.endswith(suffixes):
                    r = 'suffix_not_included'
                elif '\t' in rel or '\n' in rel:
                    r = 'unsafe_name'
                elif st.st_nlink != 1:
                    r = 'hardlink_preserve'
                elif st.st_size < a.min_size:
                    r = 'below_min_size'
                elif os.path.lexists(full + '.zst'):
                    r = 'zst_sibling_exists'
                elif under_link_target(full):
                    r = 'symlink_target'
            if r is not None:
                excluded.append((rel, r, st.st_size, detail))
            else:
                candidates.append((rel, st, c, meta, world))

    rows, changed = [], 0
    for i, (rel, st, c, meta, world) in enumerate(candidates):
        full = os.path.join(root, rel)
        sha = ''
        if not a.no_hash:
            sha = sha256_file(full)
            if stat_sig(os.lstat(full)) != stat_sig(st):
                excluded.append((rel, 'changed_during_hash', st.st_size, ''))
                changed += 1
                continue
            msha = meta.get('pkl_sha256') if os.path.basename(rel) == 'rollout.pkl' else None
            if msha and msha != sha:
                excluded.append((rel, 'meta_sha_mismatch', st.st_size, 'meta=%s' % msha))
                continue
            if a.progress_every and (i + 1) % a.progress_every == 0:
                print(json.dumps({'build_hashed': i + 1, 'of': len(candidates),
                                  'elapsed_s': round(time.time() - t0, 1)}), flush=True)
        s = stat_sig(st)
        rows.append([rel] + [s[k] for k in STAT_KEYS] +
                    [sha, c['plan_id'], c['machine'], c['instruction'], c['scene'],
                     '' if c['jitter'] is None else '%s%d' % (c['jitter_kind'], c['jitter']), c['noise'],
                     c['arm'], world[1]])

    header = {'tool': TOOL, 'root': root, 'hashed': not a.no_hash, 'protect_plan': sorted(protect),
              'expect_protected': a.expect_protected, 'expect_protected_worlds': a.expect_protected_worlds,
              'include_suffix': list(suffixes),
              'exclude_dir': sorted(excl_dirs), 'min_size': a.min_size, 'created_unix': int(time.time())}
    mpath = os.path.join(out, 'manifest.tsv')
    with open(mpath, 'x') as f:
        f.write('# ' + json.dumps(header, sort_keys=True) + '\n')
        f.write('\t'.join(MANIFEST_COLS) + '\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    msha = sha256_file(mpath)
    with open(mpath + '.sha256', 'x') as f:
        f.write('%s  manifest.tsv\n' % msha)
    with open(os.path.join(out, 'exclusions.tsv'), 'x') as f:
        f.write('rel_path\treason\tsize\tdetail\n')
        for r in excluded:
            f.write('\t'.join(_clean(x) for x in r) + '\n')
    with open(os.path.join(out, 'symlinks.tsv'), 'x') as f:
        f.write('rel_path\tlink_text\tresolved\tkind\n')
        for r in symlinks:
            f.write('\t'.join(_clean(x) for x in r) + '\n')
    with open(os.path.join(out, 'protected_coords.tsv'), 'x') as f:
        f.write('plan_id\tmachine\tinstruction\tscene\tjitter_kind\tjitter\tnoise\tarm\thow\n')
        for k in sorted(protected_coords, key=lambda x: tuple(str(v) for v in x)):
            f.write('\t'.join(map(str, k)) + '\t' + protected_coords[k] + '\n')
    reasons = {}
    for _, r, size, _ in excluded:
        d = reasons.setdefault(r, {'n': 0, 'bytes': 0})
        d['n'] += 1
        d['bytes'] += int(size)
    how = {}
    for v in protected_coords.values():
        how[v] = how.get(v, 0) + 1
    summary = dict(header, manifest_sha256=msha, manifest_files=len(rows),
                   manifest_bytes=sum(int(r[1]) for r in rows), protected_coords=n_prot,
                   protected_coords_by_how=how,
                   protected_worlds=n_worlds,
                   protected_worlds_list=sorted({'%s|%s|%s' % (w[0], w[1], w[2]) for w in protected_worlds}),
                   cross_plan_world_match_dirs=sum(cross_plan.values()),
                   cross_plan_world_match_by_plan_arm_noise=dict(sorted(cross_plan.items())),
                   protected_link_targets=len(link_targets), external_links_into_root=external_hits,
                   external_link_scan=[os.path.abspath(x) for x in (a.external_link_scan or [])],
                   index_only_coincidences_kept=index_only, symlinks=len(symlinks),
                   excluded_by_reason=reasons, changed_during_hash=changed,
                   elapsed_s=round(time.time() - t0, 1))
    with open(os.path.join(out, 'summary.json'), 'x') as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps({'manifest': mpath, 'manifest_sha256': msha, 'files': len(rows),
                      'bytes': summary['manifest_bytes'], 'protected_coords': n_prot}))
    return EXIT_OK


# ----------------------------------------------------------------------------- run

def load_manifest(path, want_sha):
    got = sha256_file(path)
    if got != want_sha:
        raise StoreError('manifest sha256 %s != --manifest-sha256 %s' % (got, want_sha))
    with open(path) as f:
        first = f.readline()
        if not first.startswith('# '):
            raise StoreError('manifest has no header line')
        header = json.loads(first[2:])
        cols = f.readline().rstrip('\n').split('\t')
        if cols != MANIFEST_COLS:
            raise StoreError('manifest columns mismatch')
        rows = []
        for line in f:
            r = dict(zip(cols, line.rstrip('\n').split('\t')))
            for k in STAT_KEYS:
                r[k] = int(r[k])
            rows.append(r)
    if header.get('tool') != TOOL:
        raise StoreError('manifest tool %r != %r' % (header.get('tool'), TOOL))
    if not header.get('hashed'):
        raise StoreError('manifest was built with --no-hash (survey only); rebuild with hashes to run')
    rels = [r['rel_path'] for r in rows]
    if len(set(rels)) != len(rels):
        raise StoreError('duplicate rel_path in manifest')
    for r in rows:
        if r['rel_path'].startswith('/') or '..' in r['rel_path'].split('/') \
                or not re.match(r'^[0-9a-f]{64}$', r['sha256']):
            raise StoreError('bad manifest row %r' % r['rel_path'])
    return header, rows


class Journal(object):
    def __init__(self, path):
        self.last, self.finalized, self.tmps = {}, set(), {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        self._apply(json.loads(line))
                    except ValueError:
                        continue  # torn last line after a crash
        self.f = open(path, 'a')

    def _apply(self, e):
        rel, ev = e.get('rel'), e.get('ev')
        if rel is None:
            return
        if ev == 'begin':
            self.tmps[rel] = e['tmp']
        elif ev == 'tmp_gone':
            self.tmps.pop(rel, None)
        elif ev == 'finalized':
            self.finalized.add(rel)
            self.tmps.pop(rel, None)
        elif ev == 'result':
            self.last[rel] = e

    def write(self, **e):
        e['t'] = round(time.time(), 3)
        self.f.write(json.dumps(e, sort_keys=True) + '\n')
        self.f.flush()
        os.fsync(self.f.fileno())
        self._apply(e)


class Runner(object):
    def __init__(self, a, header, rows):
        self.a, self.root, self.rows = a, header['root'], rows
        self.stop_reason = None
        self.runid = '%d.%d' % (os.getpid(), int(time.time()))
        self.journal = Journal(os.path.join(a.state_dir, 'journal.jsonl'))
        self.log = open(os.path.join(a.state_dir, 'progress.log'), 'a')
        self.saved_actual = self.unlinked = self.bytes_seen = 0
        self.t0 = time.time()
        self.min_free = int(a.min_free_gb * (1 << 30))

    def emit(self, **e):
        e['t'] = round(time.time(), 1)
        s = json.dumps(e, sort_keys=True)
        print(s, flush=True)
        self.log.write(s + '\n')
        self.log.flush()

    def stop_requested(self):
        if not self.stop_reason and os.path.exists(os.path.join(self.a.state_dir, 'STOP')):
            self.stop_reason = 'stop_file'
        return bool(self.stop_reason)

    def result(self, rel, status, **kw):
        self.journal.write(ev='result', rel=rel, status=status, **kw)
        return status

    def source_ok(self, r, full, rehash):
        try:
            st = os.lstat(full)
        except FileNotFoundError:
            return 'missing'
        if not stat.S_ISREG(st.st_mode):
            return 'not_regular'
        if has_symlink_component(os.path.dirname(full), self.root):
            return 'symlink_component'
        s = stat_sig(st)
        for k in STAT_KEYS:
            if s[k] != r[k]:
                return 'stat_changed:%s' % k
        now = time.time()
        if now - st.st_mtime < self.a.min_age_sec or now - st.st_ctime < self.a.min_age_sec:
            return 'too_recent'
        if open_by_any_process(st):
            return 'open_by_process'
        partials = sibling_partials(full, '.%s.zst.tmp-' % os.path.basename(full))
        if partials:
            return 'sibling_partial:%s' % partials[0]
        if rehash and sha256_file(full) != r['sha256']:
            return 'sha_changed'
        return None

    def cleanup_tmp(self, rel, tmp):
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        self.journal.write(ev='tmp_gone', rel=rel, tmp=tmp)

    def copy_meta(self, r, path):
        src = os.path.join(self.root, r['rel_path'])
        try:
            os.chown(path, r['uid'], r['gid'])
        except PermissionError:
            st = os.lstat(path)
            if (st.st_uid, st.st_gid) != (r['uid'], r['gid']):
                raise StoreError('cannot preserve owner %d:%d' % (r['uid'], r['gid']))
        os.chmod(path, stat.S_IMODE(r['mode']))
        os.utime(path, ns=(os.lstat(src).st_atime_ns, r['mtime_ns']))
        if hasattr(os, 'listxattr'):
            try:
                for name in os.listxattr(src, follow_symlinks=False):
                    os.setxattr(path, name, os.getxattr(src, name, follow_symlinks=False), follow_symlinks=False)
            except OSError as e:
                if e.errno not in (errno.ENOTSUP, errno.EPERM):
                    raise

    def startup_cleanup(self):
        for rel, tmp in list(self.journal.tmps.items()):
            if os.path.lexists(tmp):
                os.unlink(tmp)
            self.journal.write(ev='tmp_gone', rel=rel, tmp=tmp)
            self.emit(ev='startup_tmp_removed', rel=rel)

    def finish_unlink(self, r, full, final, resumed):
        rel = r['rel_path']
        zst = os.lstat(final)
        if not stat.S_ISREG(zst.st_mode):
            return self.result(rel, 'conflict_final_bad', why='final not regular')
        if self.a.final_verify or resumed:
            try:
                dsha, dn = decompress_sha256(final)
            except StoreError as e:
                dsha, dn = 'ERR:%s' % e, -1
            if dsha != r['sha256'] or dn != r['size']:
                return self.result(rel, 'conflict_final_bad', why='final verify failed')
        if not os.path.lexists(full):
            # crash after unlink but before its journal record
            return self.result(rel, 'done_resumed', saved=r['size'] - zst.st_size, zst_size=zst.st_size,
                               unlinked_by_this_run=False)
        why = self.source_ok(r, full, rehash=resumed)
        if why:
            return self.result(rel, 'skip_source_changed', why='pre_unlink:' + why, kept_zst=True)
        _hook('before_unlink', rel=rel, full=full, final=final)
        os.unlink(full)
        fsync_dir(os.path.dirname(full))
        saved = r['size'] - zst.st_size
        self.saved_actual += saved
        self.unlinked += 1
        return self.result(rel, 'done_resumed' if resumed else 'done', saved=saved, zst_size=zst.st_size,
                           unlinked_by_this_run=True)

    def process(self, r):
        a, rel = self.a, r['rel_path']
        full = os.path.join(self.root, rel)
        final, d = full + '.zst', os.path.dirname(full)
        if a.mode == 'execute' and rel in self.journal.finalized:
            if not os.path.lexists(final):
                return self.result(rel, 'conflict_final_bad', why='finalized zst vanished'), None
            return self.finish_unlink(r, full, final, resumed=True), None
        if os.path.lexists(final):
            return self.result(rel, 'conflict_final_exists'), None
        why = self.source_ok(r, full, rehash=False)
        if why == 'missing':
            return self.result(rel, 'skip_source_missing'), None
        if why in ('too_recent', 'open_by_process') or (why or '').startswith('sibling_partial'):
            return 'deferred', why
        if why:
            return self.result(rel, 'skip_source_changed', why=why), why
        free = free_bytes(d)
        if free < self.min_free + r['size']:
            self.stop_reason = 'low_disk free=%d need=%d' % (free, self.min_free + r['size'])
            return 'stopped', self.stop_reason
        tmp = os.path.join(d, '.%s.zst.tmp-%s' % (os.path.basename(full), self.runid))
        self.journal.write(ev='begin', rel=rel, tmp=tmp, mode=a.mode)
        try:
            src_sha, nread = compress_stream(full, tmp, a.level, a.threads)
            _hook('after_compress', rel=rel, full=full, tmp=tmp)
            zsize = os.lstat(tmp).st_size
            if src_sha != r['sha256'] or nread != r['size']:
                self.cleanup_tmp(rel, tmp)
                return self.result(rel, 'skip_source_changed', why='sha_at_compress'), 'sha_at_compress'
            if zsize > r['size'] * (1.0 - a.min_saving_ratio):
                self.cleanup_tmp(rel, tmp)
                return self.result(rel, 'skip_incompressible', zst_size=zsize), None
            if self.stop_requested():
                self.cleanup_tmp(rel, tmp)
                return 'stopped', self.stop_reason
            try:
                dsha, dn = decompress_sha256(tmp)
            except StoreError as e:
                dsha, dn = 'ERR:%s' % e, -1
            if dsha != r['sha256'] or dn != r['size']:
                self.cleanup_tmp(rel, tmp)
                return self.result(rel, 'fail_verify', why=dsha[:160]), dsha[:160]
            if a.mode == 'verify':
                self.cleanup_tmp(rel, tmp)
                return self.result(rel, 'verified', zst_size=zsize, would_save=r['size'] - zsize), None
            _hook('before_finalize', rel=rel, full=full, tmp=tmp)
            why = self.source_ok(r, full, rehash=True)
            if why:
                self.cleanup_tmp(rel, tmp)
                return self.result(rel, 'skip_source_changed', why='recheck:' + why), 'recheck:' + why
            if self.stop_requested():
                self.cleanup_tmp(rel, tmp)
                return 'stopped', self.stop_reason
            self.copy_meta(r, tmp)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                os.link(tmp, final)  # atomic, never overwrites
            except FileExistsError:
                self.cleanup_tmp(rel, tmp)
                return self.result(rel, 'conflict_final_exists', why='appeared_before_link'), None
            self.journal.write(ev='finalized', rel=rel, zst_size=zsize)
            os.unlink(tmp)
            fsync_dir(d)
        except BaseException:
            if rel not in self.journal.finalized and os.path.lexists(tmp):
                self.cleanup_tmp(rel, tmp)
            raise
        _hook('after_finalize', rel=rel, full=full, final=final)
        if self.stop_requested():
            return 'stopped', self.stop_reason  # resumed on next start
        return self.finish_unlink(r, full, final, resumed=False), None

    def link_targets_now(self):
        """Re-scan --external-link-scan dirs at run start: symlinks resolving into root (incl. inside it)."""
        targets = set()
        for scan in (self.a.external_link_scan or []) + [self.root]:
            for dp, dns, fns in os.walk(os.path.abspath(scan), followlinks=False, onerror=lambda e: None):
                for nm in dns + fns:
                    full = os.path.join(dp, nm)
                    if os.path.islink(full):
                        res = os.path.realpath(full)
                        if res == self.root or res.startswith(self.root + os.sep):
                            targets.add(res)
        return targets

    def run(self):
        self.startup_cleanup()
        targets = self.link_targets_now()
        self.emit(ev='link_scan', targets_into_root=len(targets))
        counts, todo = {}, []
        for r in self.rows:
            st = (self.journal.last.get(r['rel_path']) or {}).get('status')
            full = os.path.join(self.root, r['rel_path'])
            p, linked = full, False
            while len(p) >= len(self.root):
                if p in targets:
                    linked = True
                    break
                p = os.path.dirname(p)
            if linked and r['rel_path'] not in self.journal.finalized:
                counts['skip_link_target_now'] = counts.get('skip_link_target_now', 0) + 1
                continue
            if st in TERMINAL or (self.a.mode == 'verify' and st == 'verified'):
                counts['prior_' + st] = counts.get('prior_' + st, 0) + 1
            elif self.a.mode == 'execute' and r['rel_path'] in self.journal.finalized:
                todo.insert(0, r)  # resume half-finished first
            else:
                todo.append(r)
        if self.a.limit:
            todo = todo[:self.a.limit]
        total = sum(r['size'] for r in todo)
        self.emit(ev='start', mode=self.a.mode, todo=len(todo), todo_bytes=total, prior=counts, runid=self.runid)
        for i, r in enumerate(todo):
            if self.stop_requested():
                break
            try:
                status, why = self.process(r)
            except (StoreError, OSError) as e:
                status, why = 'error', '%s: %s' % (type(e).__name__, str(e)[:200])
            counts[status] = counts.get(status, 0) + 1
            if status != 'stopped':
                self.bytes_seen += r['size']
            el = time.time() - self.t0
            rate = self.bytes_seen / el if el > 0 else 0
            self.emit(ev='file', i=i + 1, n=len(todo), rel=r['rel_path'], status=status, why=why, size=r['size'],
                      saved_actual_total=self.saved_actual, unlinked_total=self.unlinked,
                      bytes_done=self.bytes_seen, mb_s=round(rate / 1e6, 1),
                      eta_s=int((total - self.bytes_seen) / rate) if rate > 0 else None)
            if status == 'stopped':
                break
            if self.a.max_bytes and self.bytes_seen >= self.a.max_bytes:
                self.stop_reason = 'max_bytes'
                break
        self.emit(ev='end', counts=counts, saved_actual_bytes=self.saved_actual, unlinked=self.unlinked,
                  stop_reason=self.stop_reason, elapsed_s=round(time.time() - self.t0, 1))
        if self.stop_reason and self.stop_reason.startswith('low_disk'):
            return EXIT_LOW_DISK
        if self.stop_reason in ('signal', 'stop_file'):
            return EXIT_INTERRUPTED
        return EXIT_ERR if counts.get('error') or counts.get('fail_verify') else EXIT_OK


def _flock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, ('%d\n' % os.getpid()).encode())
    return fd


def cmd_run(a):
    header, rows = load_manifest(a.manifest, a.manifest_sha256)
    root = header['root']
    if not os.path.isdir(root) or os.path.realpath(root) != root:
        raise StoreError('manifest root %s missing or not canonical' % root)
    sd = os.path.abspath(a.state_dir)
    if sd == root or sd.startswith(root + os.sep):
        raise StoreError('state dir must be outside the data root')
    os.makedirs(sd, exist_ok=True)
    lock_name = 'zstd_store_%s.lock' % hashlib.sha256(root.encode()).hexdigest()[:12]
    l1 = _flock(os.path.join(sd, 'LOCK'))
    l2 = _flock(os.path.join(a.lock_dir, lock_name)) if l1 is not None else None
    try:
        if l1 is None or l2 is None:
            print(json.dumps({'ev': 'locked', 'state_dir': sd}), flush=True)
            return EXIT_LOCKED
        return _run_locked(a, header, rows, sd)
    finally:
        for fd in (l1, l2):
            if fd is not None:
                os.close(fd)


def _run_locked(a, header, rows, sd):
    bind = os.path.join(sd, 'manifest.sha256')
    if os.path.exists(bind):
        with open(bind) as f:
            prev = f.read().split()
        if prev[:2] != [a.manifest_sha256, a.mode]:
            raise StoreError('state dir bound to %s; refusing manifest %s mode %s' % (prev, a.manifest_sha256, a.mode))
    else:
        with open(bind, 'x') as f:
            f.write('%s %s\n' % (a.manifest_sha256, a.mode))
    runner = Runner(a, header, rows)

    def on_signal(signum, frame):
        runner.stop_reason = 'signal'
    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        return runner.run()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        runner.journal.f.close()
        runner.log.close()


def cmd_report(a):
    j = Journal(os.path.join(a.state_dir, 'journal.jsonl'))
    j.f.close()
    by, saved, would = {}, 0, 0
    for e in j.last.values():
        st = e['status']
        by[st] = by.get(st, 0) + 1
        if st in ('done', 'done_resumed'):
            saved += int(e.get('saved', 0))
        would += int(e.get('would_save', 0))
    out = {'state_dir': os.path.abspath(a.state_dir), 'by_status': by, 'saved_actual_bytes': saved,
           'verify_would_save_bytes': would,
           'finalized_not_unlinked': sorted(r for r in j.finalized
                                            if (j.last.get(r) or {}).get('status') not in TERMINAL),
           'open_tmps': j.tmps}
    print(json.dumps(out, indent=1, sort_keys=True))
    return EXIT_OK


def cmd_sample(a):
    root = os.path.realpath(a.root)
    scratch = os.path.abspath(a.scratch)
    if scratch == root or scratch.startswith(root + os.sep):
        raise StoreError('scratch must be outside root')
    os.makedirs(scratch, exist_ok=True)
    res = []
    for rel in a.rel:
        full = os.path.join(root, rel)
        st = os.lstat(full)
        if not stat.S_ISREG(st.st_mode):
            raise StoreError('%s is not a regular file (lstat)' % rel)
        if free_bytes(scratch) < int(a.min_free_gb * (1 << 30)) + st.st_size:
            raise StoreError('low disk in scratch')
        z = os.path.join(scratch, '%d.%s.zst' % (len(res), os.path.basename(rel)))
        t = time.time()
        src_sha, n = compress_stream(full, z, a.level, a.threads)
        tc = time.time() - t
        t = time.time()
        dsha, dn = decompress_sha256(z)
        td = time.time() - t
        zs = os.lstat(z).st_size
        res.append({'rel': rel, 'size': st.st_size, 'zst_size': zs, 'ratio': round(zs / max(st.st_size, 1), 4),
                    'saved': st.st_size - zs, 'compress_s': round(tc, 1), 'decompress_s': round(td, 1),
                    'src_sha256': src_sha, 'decompressed_sha256': dsha,
                    'roundtrip_ok': src_sha == dsha and dn == n == st.st_size,
                    'source_untouched': stat_sig(os.lstat(full)) == stat_sig(st), 'nlink': st.st_nlink})
        if not a.keep:
            os.unlink(z)
    out = {'level': a.level, 'threads': a.threads, 'files': res,
           'total_size': sum(r['size'] for r in res), 'total_zst': sum(r['zst_size'] for r in res),
           'all_roundtrip_ok': all(r['roundtrip_ok'] and r['source_untouched'] for r in res)}
    print(json.dumps(out, indent=1))
    return EXIT_OK if out['all_roundtrip_ok'] else EXIT_ERR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd')
    b = sub.add_parser('build')
    b.add_argument('--root', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--protect-plan', action='append')
    b.add_argument('--expect-protected', type=int, help='gate: protected-plan rollout coordinate count')
    b.add_argument('--expect-protected-worlds', type=int, help='gate: distinct protected worlds')
    b.add_argument('--exclude-dir', action='append')
    b.add_argument('--external-link-scan', action='append',
                   help='dir to scan (no following) for symlinks resolving into root; their targets are protected')
    b.add_argument('--include-suffix', action='append')
    b.add_argument('--min-size', type=int, default=1 << 20)
    b.add_argument('--no-hash', action='store_true', help='survey only; run refuses such a manifest')
    b.add_argument('--progress-every', type=int, default=50)
    r = sub.add_parser('run')
    r.add_argument('--manifest', required=True)
    r.add_argument('--manifest-sha256', required=True)
    r.add_argument('--state-dir', required=True)
    r.add_argument('--mode', choices=('verify', 'execute'), required=True)
    r.add_argument('--level', type=int, default=3)
    r.add_argument('--threads', type=int, default=1)
    r.add_argument('--min-free-gb', type=float, default=20.0)
    r.add_argument('--min-saving-ratio', type=float, default=0.03)
    r.add_argument('--min-age-sec', type=float, default=3600.0)
    r.add_argument('--no-final-verify', dest='final_verify', action='store_false')
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--max-bytes', type=int, default=0)
    r.add_argument('--lock-dir', default='/tmp')
    r.add_argument('--external-link-scan', action='append',
                   help='re-scanned at start; files under a symlink target resolving into root are skipped')
    p = sub.add_parser('report')
    p.add_argument('--state-dir', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--root', required=True)
    s.add_argument('--scratch', required=True)
    s.add_argument('--rel', action='append', required=True)
    s.add_argument('--level', type=int, default=3)
    s.add_argument('--threads', type=int, default=1)
    s.add_argument('--min-free-gb', type=float, default=20.0)
    s.add_argument('--keep', action='store_true')
    a = ap.parse_args(argv)
    fn = {'build': cmd_build, 'run': cmd_run, 'report': cmd_report, 'sample': cmd_sample}.get(a.cmd)
    if fn is None:
        ap.print_help()
        return 2
    try:
        return fn(a)
    except StoreError as e:
        print(json.dumps({'ev': 'error', 'error': str(e)}), flush=True)
        return EXIT_ERR


if __name__ == '__main__':
    sys.exit(main())
