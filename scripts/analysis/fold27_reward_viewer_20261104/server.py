"""Fold27 reward viewer: stdlib HTTP server (no external deps, no CDN; Python >= 3.8, numpy >= 1.21).

    python3 scripts/analysis/fold27_reward_viewer_20261104/server.py \
        --derived <root> [<root> ...] [--video-index v.json --video-root DIR] --host 127.0.0.1 --port 8784

Routes (GET/HEAD only; every user input is matched against an in-memory whitelist, never joined into a path):
  /                      static/index.html
  /static/<name>         whitelisted static files only (app.js, app.css)
  /api/meta              tasks/instructions/folds, provenance, per-fold SHA panel (original record vs reconstructed)
  /api/fold?fold=F       episode list of fold F
  /api/fold_means?fold=F    success/failure means over rollouts that exist at inference index t (+n_t)
  /api/episode?fold=F&ep=I   per-chunk basic/MIX/alpha/-beta*C/delta (+phi when stored), cross-fold duplicates
  /video?id=K            validated video K (index into the startup-validated list) with Range support
Port 8782 (existing SIMPLER viewer) is refused. Binding fails loudly if the port is in use (no takeover).
"""
import argparse
import json
import os
import re
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer_data as VD  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / 'static'
STATIC = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript; charset=utf-8',
          'app.css': 'text/css; charset=utf-8'}
FORBIDDEN_PORTS = {8782}
INT_RE = re.compile(r'^\d{1,6}$')
SEC_HEADERS = {'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer',
               'Content-Security-Policy': "default-src 'self'; media-src 'self'; img-src 'self' data:; "
                                          "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'"}


class App:
    def __init__(self, roots, video_index=None, video_root=None):
        self.folds, self.failures = VD.load_all(roots)
        self.dup = VD.duplicates(self.folds)
        self.means = {k: VD.outcome_means(v) for k, v in self.folds.items()}
        videos, self.video_rejected = VD.load_video_index(video_index, video_root) if video_index else ({}, {})
        self.video_ids = sorted(videos)                    # K -> episode id
        self.videos = videos
        self.meta = VD.meta(self.folds, self.failures, videos)
        self.meta['videos_rejected'] = self.video_rejected
        self.static = {n: (STATIC_DIR / n).read_bytes() for n in STATIC}

    # pure routing: (status, headers, body bytes or ('file', path)); tested without sockets
    def route(self, raw_path):
        u = urlsplit(raw_path)
        path, q = u.path, parse_qs(u.query, keep_blank_values=True, max_num_fields=8)
        if any(len(v) != 1 for v in q.values()):
            return self.err(400, 'repeated query parameter')
        q = {k: v[0] for k, v in q.items()}
        if path == '/':
            return 200, {'Content-Type': STATIC['index.html']}, self.static['index.html']
        if path.startswith('/static/'):
            name = path[len('/static/'):]
            if name not in STATIC or name == 'index.html':
                return self.err(404, 'not found')
            return 200, {'Content-Type': STATIC[name]}, self.static[name]
        if path == '/api/meta':
            return self.json(self.meta)
        if path == '/api/fold_means':
            f = self.folds.get(q.get('fold', ''))
            if f is None:
                return self.err(404, 'unknown fold')
            return self.json({'fold': f['fold'], 'coefficients': VD.coefficients(f), 'groups': self.means[f['fold']]})
        if path in ('/api/fold', '/api/episode'):
            f = self.folds.get(q.get('fold', ''))
            if f is None:
                return self.err(404, 'unknown fold')
            if path == '/api/fold':
                return self.json({'fold': f['fold'], 'instruction': f['instruction'], 'episodes': VD.fold_episodes(f)})
            ep = q.get('ep', '')
            if not INT_RE.match(ep) or int(ep) >= len(f['arrays']['episode_id']):
                return self.err(400, 'bad ep')
            p = VD.episode_payload(f, int(ep), self.dup, self.videos)
            p['video_id'] = self.video_ids.index(p['episode_id']) if p['video'] else None
            return self.json(p)
        if path == '/video':
            k = q.get('id', '')
            if not INT_RE.match(k) or int(k) >= len(self.video_ids):
                return self.err(404, 'no such video')
            return 200, {'Content-Type': 'video/mp4'}, ('file', self.videos[self.video_ids[int(k)]])
        return self.err(404, 'not found')

    @staticmethod
    def json(obj):
        return 200, {'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store'}, \
            json.dumps(obj, ensure_ascii=False).encode()

    @staticmethod
    def err(code, msg):
        return code, {'Content-Type': 'application/json; charset=utf-8'}, json.dumps({'error': msg}).encode()


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        server_version = 'fold27-reward-viewer'
        sys_version = ''

        def _send(self, head_only):
            if len(self.path) > 2048:
                code, hdr, body = app.err(414, 'uri too long')
            else:
                code, hdr, body = app.route(self.path)
            rng = None
            if isinstance(body, tuple):
                size = os.path.getsize(body[1])
                start, end = 0, size - 1
                m = re.match(r'^bytes=(\d*)-(\d*)$', self.headers.get('Range', '') or '')
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1)); end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
                    else:
                        start = max(0, size - int(m.group(2)))
                    if start > end:
                        code, hdr, body = app.err(416, 'bad range')
                    else:
                        code, rng = 206, (start, end)
                if rng is None and not isinstance(body, bytes):
                    rng = (0, size - 1)
                if rng:
                    hdr['Accept-Ranges'] = 'bytes'
                    if code == 206:
                        hdr['Content-Range'] = 'bytes %d-%d/%d' % (rng[0], rng[1], size)
            self.send_response(code)
            for k, v in {**SEC_HEADERS, **hdr}.items():
                self.send_header(k, v)
            n = (rng[1] - rng[0] + 1) if (rng and isinstance(body, tuple)) else len(body)
            self.send_header('Content-Length', str(n))
            self.end_headers()
            if head_only:
                return
            if isinstance(body, tuple):
                with open(body[1], 'rb') as fh:
                    fh.seek(rng[0]); left = n
                    while left > 0:
                        chunk = fh.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk); left -= len(chunk)
            else:
                self.wfile.write(body)

        def do_GET(self):
            self._send(False)

        def do_HEAD(self):
            self._send(True)

        def _deny(self):
            self.send_response(405); self.send_header('Allow', 'GET, HEAD'); self.send_header('Content-Length', '0')
            self.end_headers()

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _deny

        def log_message(self, fmt, *args):
            sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', type=Path, nargs='+', required=True)
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8784)
    ap.add_argument('--video-index', type=Path); ap.add_argument('--video-root', type=Path)
    ap.add_argument('--pid-file', type=Path)
    ap.add_argument('--check-only', action='store_true', help='load + validate data, print summary, do not serve')
    a = ap.parse_args(argv)
    if a.port in FORBIDDEN_PORTS:
        sys.exit('port %d is reserved for an existing viewer' % a.port)
    if bool(a.video_index) != bool(a.video_root):
        sys.exit('--video-index and --video-root go together')
    app = App(a.derived, a.video_index, a.video_root)
    summary = {'folds': len(app.folds), 'rows': sum(len(f['arrays']['basic']) for f in app.folds.values()),
               'episodes_unique': len(app.dup), 'videos': len(app.videos), 'videos_rejected': len(app.video_rejected),
               'failed_attempts_listed': sorted(app.failures)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if a.check_only:
        return 0
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))     # raises if the port is taken
    srv.daemon_threads = True
    if a.pid_file:
        if a.pid_file.exists():
            sys.exit('pid file exists: %s' % a.pid_file)
        a.pid_file.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # kill <pid> -> finally removes the pid file
    print('serving http://%s:%d' % (a.host, a.port), flush=True)
    try:
        srv.serve_forever()
    finally:
        if a.pid_file and a.pid_file.exists() and a.pid_file.read_text() == str(os.getpid()):
            a.pid_file.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
