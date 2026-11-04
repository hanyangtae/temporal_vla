"""Read-only probe: verify the frame <-> env step <-> inference mapping of every indexed rollout video.

    python3 scripts/analysis/fold27_reward_viewer_20261104/probe_video_sync.py \
        --derived <root> [<root> ...] --video-index <video_index.json> --video-root <grid root> --out <NEW dir>

No ffprobe/opencv/pickle: a stdlib ISO-BMFF (MP4) box parser reads only the moov box (mvhd, mdhd, hdlr, stts, ctts,
stsz, elst) of each video; the media bytes are hashed (sha256 must equal the video index) but never decoded.

Code evidence for the hypothesis being tested (recorded in the receipt):
  collector  scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py -> src/policies/groot/robocasa/env_wrappers.py
             wrap_groot_robocasa_eval_env: VideoRecordingWrapper is INSIDE MultiStepWrapper (one wrapper step = one env step)
  upstream   gr00t/eval/sim/wrapper/video_recording_wrapper.py: reset() sets step_count=1 and writes NO frame;
             step(): step_count += 1, then writes one frame iff step_count % steps_per_render == 0;
             VideoRecorder (start_time None -> n_repeats 1) writes frames at a constant rate fps.
  => with s = steps_per_render and K executed env steps: frame f (0-based) is the observation right AFTER 0-based env
     step e_f = e0 + s*f, e0 = s*ceil(2/s) - 2; frame count = #multiples of s in [2, K+1]; frame f is presented at f/fps.
A video is 'verified' only if, from its own container: one video track; constant sample delta; sorted presentation
times (dts + ctts, edit list applied) == f * delta exactly; frame count == the prediction for EXACTLY ONE s in 1..5;
that s and fps agree with the collector defaults (s=2, fps=20) unless JSON meta next to the video states other values
(then they must agree with the meta). Anything else -> 'unverified' with the reason; the viewer then disables sync.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer_data as VD  # noqa: E402

CONTAINERS = {b'moov', b'trak', b'mdia', b'minf', b'stbl', b'edts'}
DEFAULT_S, DEFAULT_FPS = 2, 20
EVIDENCE = {
    'collector': 'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py (video_fps default 20, steps_per_render default 2)',
    'wrapper_order': 'src/policies/groot/robocasa/env_wrappers.py wrap_groot_robocasa_eval_env: VideoRecordingWrapper inside MultiStepWrapper',
    'frame_rule': 'src/policies/Isaac-GR00T/gr00t/eval/sim/wrapper/video_recording_wrapper.py: reset step_count=1 no frame; '
                  'step: step_count+=1, frame iff step_count % steps_per_render == 0; constant fps, n_repeats=1',
}


class ProbeError(ValueError):
    pass


def _boxes(buf, off, end):
    while off + 8 <= end:
        size, typ = struct.unpack('>I4s', buf[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack('>Q', buf[off + 8:off + 16])[0]; hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr or off + size > end:
            raise ProbeError('corrupt box %r' % typ)
        yield typ, off + hdr, off + size
        off += size


def read_moov(path):
    """Return the moov payload bytes by walking top-level boxes (only moov is read into memory)."""
    with open(path, 'rb') as f:
        f.seek(0, 2); total = f.tell(); off = 0
        while off + 8 <= total:
            f.seek(off); h = f.read(16)
            size, typ = struct.unpack('>I4s', h[:8]); hdr = 8
            if size == 1:
                size = struct.unpack('>Q', h[8:16])[0]; hdr = 16
            elif size == 0:
                size = total - off
            if size < hdr:
                raise ProbeError('corrupt top-level box')
            if typ == b'moov':
                f.seek(off)
                return f.read(size)
            off += size
    raise ProbeError('no moov box')


def parse_moov(moov):
    info = {'tracks': []}

    def walk(off, end, trak):
        for typ, p, e in _boxes(moov, off, end):
            if typ == b'trak':
                t = {}
                info['tracks'].append(t)
                walk(p, e, t)
            elif typ in CONTAINERS:
                walk(p, e, trak)
            elif typ == b'mvhd':
                v = moov[p]
                info['movie_timescale'] = struct.unpack('>I', moov[p + (20 if v == 1 else 12):p + (24 if v == 1 else 16)])[0]
            elif trak is not None and typ == b'mdhd':
                v = moov[p]
                if v == 1:
                    trak['timescale'], trak['duration'] = struct.unpack('>IQ', moov[p + 20:p + 32])
                else:
                    trak['timescale'], trak['duration'] = struct.unpack('>II', moov[p + 12:p + 20])
            elif trak is not None and typ == b'hdlr':
                trak['handler'] = moov[p + 8:p + 12].decode('latin-1')
            elif trak is not None and typ == b'stts':
                n = struct.unpack('>I', moov[p + 4:p + 8])[0]
                trak['stts'] = [struct.unpack('>II', moov[p + 8 + 8 * i:p + 16 + 8 * i]) for i in range(n)]
            elif trak is not None and typ == b'ctts':
                v = moov[p]; n = struct.unpack('>I', moov[p + 4:p + 8])[0]
                fmt = '>Ii' if v == 1 else '>II'
                trak['ctts'] = [struct.unpack(fmt, moov[p + 8 + 8 * i:p + 16 + 8 * i]) for i in range(n)]
            elif trak is not None and typ == b'stsz':
                trak['sample_size'], trak['sample_count'] = struct.unpack('>II', moov[p + 4:p + 12])
            elif trak is not None and typ == b'elst':
                v = moov[p]; n = struct.unpack('>I', moov[p + 4:p + 8])[0]
                ents, q = [], p + 8
                for _ in range(n):
                    if v == 1:
                        seg, mt = struct.unpack('>Qq', moov[q:q + 16]); q += 20
                    else:
                        seg, mt = struct.unpack('>Ii', moov[q:q + 8]); q += 12
                    ents.append((seg, mt))
                trak['elst'] = ents
    walk(0, len(moov), None)
    return info


def video_timing(path):
    info = parse_moov(read_moov(path))
    vids = [t for t in info['tracks'] if t.get('handler') == 'vide']
    if len(vids) != 1:
        raise ProbeError('expected exactly one video track, found %d' % len(vids))
    t = vids[0]
    n, ts = t.get('sample_count'), t.get('timescale')
    if not n or not ts or 'stts' not in t:
        raise ProbeError('missing stsz/mdhd/stts')
    deltas = {d for c, d in t['stts'] if c}
    if len(deltas) != 1:
        raise ProbeError('variable sample duration %s' % sorted(deltas))
    delta = deltas.pop()
    dts = []
    cur = 0
    for c, d in t['stts']:
        for _ in range(c):
            dts.append(cur); cur += d
    if len(dts) != n:
        raise ProbeError('stts count %d != stsz %d' % (len(dts), n))
    ct = [0] * n
    if 'ctts' in t:
        k = 0
        for c, o in t['ctts']:
            for _ in range(c):
                if k < n:
                    ct[k] = o
                k += 1
    empty_offset, media_time = 0.0, 0
    for seg, mt in t.get('elst', []):
        if mt == -1:
            empty_offset += seg / float(info.get('movie_timescale') or ts)
        else:
            media_time = mt
            break
    pts = sorted(dts[i] + ct[i] - media_time for i in range(n))
    regular = all(pts[f] == f * delta for f in range(n)) and empty_offset == 0
    return {'frames': n, 'timescale': ts, 'sample_delta': delta, 'fps': ts / float(delta),
            'elst': t.get('elst', []), 'has_ctts': 'ctts' in t, 'empty_edit_offset_s': empty_offset,
            'pts_regular_from_zero': regular, 'first_pts_ticks': pts[0], 'duration_s': n * delta / float(ts)}


def predicted_frames(K, s):
    m0 = s * math.ceil(2 / s)
    return 0 if K + 1 < m0 else (K + 1 - m0) // s + 1


def meta_video_params(d):
    for name in ('meta.json', 'episode_meta.json'):
        for p in (d / name, d.parent / name):
            if p.is_file():
                try:
                    m = json.loads(p.read_text())
                except ValueError:
                    continue
                if isinstance(m, dict) and ('video_fps' in m or 'steps_per_render' in m):
                    return {'video_fps': m.get('video_fps'), 'steps_per_render': m.get('steps_per_render'),
                            'file': ('base/' if p.parent == d else '') + name}
    return None


def probe_one(root, rec, K):
    p = (root / rec['path']).resolve()
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    if h.hexdigest() != rec['sha256']:
        return {'status': 'unverified', 'reason': 'sha256 != video index'}
    try:
        tm = video_timing(p)
    except (ProbeError, struct.error) as e:
        return {'status': 'unverified', 'reason': 'container: %s' % e}
    out = {'timing': tm, 'env_steps': K}
    match = [s for s in range(1, 6) if predicted_frames(K, s) == tm['frames']]
    out['steps_per_render_matches'] = match
    meta = meta_video_params(p.parent)
    out['meta'] = meta
    want_s = (meta or {}).get('steps_per_render') or DEFAULT_S
    want_fps = (meta or {}).get('video_fps') or DEFAULT_FPS
    reasons = []
    if not tm['pts_regular_from_zero']:
        reasons.append('presentation times not f*delta from 0')
    if match != [want_s]:
        reasons.append('frame count %d matches s=%s, expected exactly s=%d for K=%d' % (tm['frames'], match, want_s, K))
    if abs(tm['fps'] - want_fps) > 1e-9:
        reasons.append('fps %.6g != %s' % (tm['fps'], want_fps))
    if reasons:
        out.update(status='unverified', reason='; '.join(reasons))
        return out
    s = want_s
    out.update(status='verified', sync={'fps': tm['fps'], 'frames': tm['frames'], 'steps_per_render': s,
                                        'first_frame_env_step': s * math.ceil(2 / s) - 2,
                                        'env_steps': K, 'video_sha256': rec['sha256'],
                                        'semantics': 'frame f = observation right after 0-based env step '
                                                     'first_frame_env_step + steps_per_render*f; shown at t=f/fps'})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', type=Path, nargs='+', required=True)
    ap.add_argument('--video-index', type=Path, required=True); ap.add_argument('--video-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        sys.exit('out exists: %s' % a.out)
    t0 = time.time()
    folds, _ = VD.load_all(a.derived)
    steps = {}
    for f in folds.values():
        ar = f['arrays']
        for eid, k in zip(ar['episode_id'], ar['ep_steps']):
            if steps.setdefault(str(eid), int(k)) != int(k):
                raise VD.DataError('env_steps differ across folds for %s' % eid)
    index = json.loads(a.video_index.read_text())
    root = a.video_root.resolve()
    results = {eid: probe_one(root, rec, steps[eid]) if eid in steps else {'status': 'unverified',
                                                                            'reason': 'episode not in arrays'}
               for eid, rec in sorted(index.items())}
    a.out.mkdir(parents=True)
    sync = {eid: r['sync'] for eid, r in results.items() if r['status'] == 'verified'}
    (a.out / 'video_sync.json').write_text(json.dumps(sync, indent=1, ensure_ascii=False))
    from collections import Counter
    receipt = {'evidence': EVIDENCE, 'videos': len(index), 'verified': len(sync),
               'unverified': {eid: r.get('reason') for eid, r in results.items() if r['status'] != 'verified'},
               'frame_count_s_match_hist': dict(Counter(str(r.get('steps_per_render_matches')) for r in results.values())),
               'fps_hist': dict(Counter(str(r.get('timing', {}).get('fps')) for r in results.values())),
               'meta_with_video_params': sum(1 for r in results.values() if r.get('meta')),
               'results': results, 'seconds': round(time.time() - t0, 1), 'decoded_or_modified_videos': False,
               'pickle_loaded': False}
    (a.out / 'video_sync_receipt.json').write_text(json.dumps(receipt, indent=1, ensure_ascii=False))
    print(json.dumps({k: receipt[k] for k in ('videos', 'verified', 'frame_count_s_match_hist', 'fps_hist',
                                               'meta_with_video_params', 'seconds')}))
    print('unverified examples:', json.dumps(dict(list(receipt['unverified'].items())[:5]), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
