"""Browser smoke for a RUNNING viewer (stdlib only): headless Chromium --dump-dom checks + one screenshot.

    python3 scripts/analysis/fold27_reward_viewer_20261104/browser_smoke.py --url http://127.0.0.1:8784 \
        --chromium ~/.cache/ms-playwright/chromium_headless_shell-1181/chrome-linux/headless_shell --out <new dir>

Checks (all must pass, exit 0): API meta has 18 folds; rendered DOM has >= 3 chart SVGs (basic/MIX, decomposition,
phi) + outcome mean panels with N_t, alpha/beta from receipt, decomposition toggles
(#hide=), the SHA table shows only matches, the provenance/fit-not-eval banner, cluster ID "미저장". Writes DOM + PNG
into --out (must not exist). Does not start or stop the server.
"""
import argparse, json, re, subprocess, sys, urllib.request
from pathlib import Path


def run_chrome(chromium, args, timeout=60):
    base = [str(chromium), '--headless', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
            '--virtual-time-budget=8000', '--window-size=1300,3600']
    return subprocess.run(base + args, capture_output=True, timeout=timeout)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8784'); ap.add_argument('--chromium', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--expect-folds', type=int, default=18)
    a = ap.parse_args(argv)
    if a.out.exists():
        sys.exit('out exists: %s' % a.out)
    a.out.mkdir(parents=True)
    checks = {}
    meta = json.loads(urllib.request.urlopen(a.url + '/api/meta', timeout=20).read())
    checks['api_folds'] = len(meta['folds']) == a.expect_folds
    checks['api_sha_all_gates'] = all(all(f['gates'].values()) for f in meta['folds'].values())
    dom = run_chrome(a.chromium, ['--dump-dom', a.url + '/']).stdout.decode('utf-8', 'replace')
    (a.out / 'dom.html').write_text(dom)
    checks['charts_rendered'] = len(re.findall(r'<svg class="plot"', dom)) >= 3
    checks['sha_table_all_match'] = '✓ 일치' in dom and '✗ 불일치' not in dom
    checks['provenance_banner'] = '평가 아님' in dom and 'RoboCasa' in dom
    checks['cluster_not_stored'] = '미저장' in dom
    checks['coef_from_receipt'] = 'α = 0.1' in dom and 'β = 0.0396' in dom and 'receipt' in dom
    checks['outcome_means_with_n'] = '평균 · 집단 total N=' in dom and 'N_t' in dom and '0 채움' in dom
    checks['decomp_toggles'] = dom.count('class="tog') >= 3
    dom_h = run_chrome(a.chromium, ['--dump-dom', a.url + '/#hide=alpha_term']).stdout.decode('utf-8', 'replace')
    checks['toggle_hides_series'] = ('data-key="alpha_term" d=' in dom and 'data-key="alpha_term" d=' not in dom_h
                                     and 'data-key="delta" d=' in dom_h)
    checks['no_seconds_axis'] = not re.search(r'\d+(\.\d+)?\s*(sec|초)\b', dom.replace('시간(초) 추정', ''))
    shot = run_chrome(a.chromium, ['--screenshot=' + str(a.out / 'page.png'), a.url + '/'])
    checks['screenshot'] = (a.out / 'page.png').exists() and shot.returncode == 0
    (a.out / 'checks.json').write_text(json.dumps(checks, indent=1, ensure_ascii=False))
    print(json.dumps(checks, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
