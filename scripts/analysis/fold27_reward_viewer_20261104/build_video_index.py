"""Build the viewer's video index from the archive (read-only; no copies, no pickle).

    python3 scripts/analysis/fold27_reward_viewer_20261104/build_video_index.py \
        --derived <root> [<root> ...] \
        --video-root /home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid \
        --out <NEW dir>          # writes video_index.json + video_index_receipt.json; refuses if it exists

Episodes = the unique episode_ids of the SHA-verified 18-fold arrays (viewer_data.load_all: receipt gates +
npz sha256, allow_pickle=False). For each id the ONLY candidate is <video-root>/<episode_id>/base/video.mp4
(episode_id used verbatim: plan/machine/instruction/s<scene>/j<j>/n<n>). Accepted iff:
  exists, regular file after resolving symlinks, resolved path inside --video-root, size > 0, ISO-BMFF 'ftyp' box,
  and identity is not contradicted: if JSON meta files exist next to it (META_NAMES, JSON only), their success
  flag must equal the fold arrays' outcome (and their episode coordinates, when present, must match).
Rejected ids go to 'missing' (no file) or 'mismatch' (file present but a check failed) — never into the index.
Accepted ids without any JSON meta are kept but flagged identity='path_only'. sha256 is computed by streaming the
video bytes; the viewer re-verifies it at startup (viewer_data.load_video_index).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer_data as VD  # noqa: E402

META_NAMES = ('meta.json', 'episode_meta.json', 'result.json', 'info.json')
SUCCESS_KEYS = ('success', 'is_success', 'episode_success')


def sha256_stream(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def episode_outcomes(folds):
    """episode_id -> success (must agree across every fold that contains it)."""
    out = {}
    for f in folds.values():
        a = f['arrays']
        for eid, s in zip(a['episode_id'], a['ep_success']):
            eid, s = str(eid), bool(s)
            if out.setdefault(eid, s) != s:
                raise VD.DataError('outcome differs across folds for ' + eid)
    return out


def check_meta(dirs, eid, success):
    """JSON meta next to the video: returns (identity, problems, files)."""
    _, scene, j, n = VD.split_episode_id(eid)
    problems, files = [], []
    for d in dirs:
        for name in META_NAMES:
            p = d / name
            if not p.is_file():
                continue
            try:
                m = json.loads(p.read_text())
            except (ValueError, UnicodeDecodeError) as e:
                problems.append('%s: unreadable json (%s)' % (name, type(e).__name__)); continue
            if not isinstance(m, dict):
                continue
            files.append(('base/' if d.name == 'base' else '') + name)
            for k in SUCCESS_KEYS:
                if k in m and bool(m[k]) != success:
                    problems.append('%s: %s=%r != arrays success %r' % (name, k, m[k], success))
            for k, want in (('scene', scene), ('jitter', j), ('j', j), ('noise', n), ('n', n)):
                if k in m and isinstance(m[k], int) and m[k] != want:
                    problems.append('%s: %s=%r != %r' % (name, k, m[k], want))
            if 'episode_id' in m and m['episode_id'] != eid:
                problems.append('%s: episode_id differs' % name)
    return ('meta_checked' if files else 'path_only'), problems, files


def check_one(root, eid, success):
    rel = eid + '/base/video.mp4'
    p = root / rel
    if not os.path.lexists(p):
        return 'missing', {'path': rel}
    rp = p.resolve()
    if os.path.commonpath([str(rp), str(root)]) != str(root):
        return 'mismatch', {'path': rel, 'reason': 'resolves outside video root'}
    if not rp.is_file():
        return 'mismatch', {'path': rel, 'reason': 'not a regular file'}
    size = rp.stat().st_size
    with open(rp, 'rb') as f:
        head = f.read(12)
    if size == 0 or head[4:8] != b'ftyp':
        return 'mismatch', {'path': rel, 'reason': 'not an mp4 (no ftyp box)', 'bytes': size}
    identity, problems, files = check_meta([p.parent, p.parent.parent], eid, success)
    if problems:
        return 'mismatch', {'path': rel, 'reason': '; '.join(problems), 'meta_files': files}
    return 'ok', {'path': rel, 'sha256': sha256_stream(rp), 'bytes': size, 'identity': identity,
                  'meta_files': files, 'success': success}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', type=Path, nargs='+', required=True)
    ap.add_argument('--video-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        sys.exit('out exists: %s' % a.out)
    root = a.video_root.resolve()
    if not root.is_dir():
        sys.exit('video root not a directory: %s' % a.video_root)
    t0 = time.time()
    folds, _ = VD.load_all(a.derived)
    outcomes = episode_outcomes(folds)
    ok, missing, mismatch = {}, {}, {}
    for eid in sorted(outcomes):
        status, rec = check_one(root, eid, outcomes[eid])
        {'ok': ok, 'missing': missing, 'mismatch': mismatch}[status][eid] = rec
    a.out.mkdir(parents=True)
    index = {eid: {'path': r['path'], 'sha256': r['sha256']} for eid, r in ok.items()}
    (a.out / 'video_index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False))
    receipt = {'video_root': str(root), 'rule': '<video-root>/<episode_id>/base/video.mp4',
               'folds': len(folds), 'unique_episodes': len(outcomes),
               'counts': {'ok': len(ok), 'missing': len(missing), 'mismatch': len(mismatch),
                          'ok_meta_checked': sum(r['identity'] == 'meta_checked' for r in ok.values()),
                          'ok_path_only': sum(r['identity'] == 'path_only' for r in ok.values())},
               'ok': ok, 'missing': missing, 'mismatch': mismatch,
               'video_index_sha256': sha256_stream(a.out / 'video_index.json'),
               'seconds': round(time.time() - t0, 1), 'copied_or_modified_originals': False, 'pickle_loaded': False}
    (a.out / 'video_index_receipt.json').write_text(json.dumps(receipt, indent=1, ensure_ascii=False))
    print(json.dumps({k: receipt[k] for k in ('unique_episodes', 'counts', 'video_index_sha256', 'seconds')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
