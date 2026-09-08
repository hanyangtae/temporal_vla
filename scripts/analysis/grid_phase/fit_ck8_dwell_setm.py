"""Cluster별 dwell cap + episode 동일 기여 setM. 원본/GT 산출물은 수정하지 않는다."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

VERSION = 'ck8_dwell_cap_equal_episode_v1'


def capped_rows(ep, cluster, succ, allow):
    """학습 성공의 cluster dwell ceil(mean+std); 시간순 앞쪽. 없는 cluster는 보존."""
    caps, keep, contributions = {}, np.zeros(len(ep), bool), []
    for c in sorted(set(cluster[allow])):
        counts = [int(np.sum(allow & (ep == e) & (cluster == c)))
                  for e in np.unique(ep[allow & (succ == 1) & (cluster == c)])]
        if counts:
            caps[int(c)] = int(np.ceil(np.mean(counts) + np.std(counts)))
        for e in np.unique(ep[allow & (cluster == c)]):
            idx = np.flatnonzero(allow & (ep == e) & (cluster == c))
            selected = idx[:caps.get(int(c), len(idx))]
            keep[selected] = True
            contributions.append(dict(ep_id=int(e), cluster=int(c), success=int(succ[idx[0]]),
                                      raw=len(idx), used=len(selected), episode_weight=1.0))
    return keep, caps, contributions


def episode_mean(X, ep, mask):
    ids = np.unique(ep[mask])
    if not len(ids):
        return None
    return np.mean([X[mask & (ep == e)].mean(0, dtype=np.float64) for e in ids], axis=0)


def fit_one(X, ep, succ, jit, mask, target, form):
    ms, mf = mask & (succ == 1), mask & (succ == 0)
    if min(ms.sum(), mf.sum()) < 20:  # 기존 setM 등록 기준 유지
        return None, 'insufficient_records_existing_min20'
    mu_s, mu_f = episode_mean(X, ep, ms), episode_mean(X, ep, mf)
    if form == 'plain':
        delta, anchor = mu_s - mu_f, mu_s
    else:
        pairs = []
        for j in np.unique(jit[mask]):
            a, b = ms & (jit == j), mf & (jit == j)
            if min(a.sum(), b.sum()) >= 20:
                pairs.append(episode_mean(X, ep, a) - episode_mean(X, ep, b))
        tf = mf & (jit == target)
        if len(pairs) < 2 or tf.sum() < 20:
            return None, 'missing_mixed_jitter_or_target_cluster_existing_rule'
        delta = np.mean(pairs, axis=0)
        anchor = episode_mean(X, ep, tf) + delta
    norm = np.linalg.norm(delta)
    if not np.isfinite(norm) or norm < 1e-12:
        return None, 'undefined_direction'
    v = delta / norm
    return (v, float(anchor @ v)), ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=Path, required=True)
    p.add_argument('--slug', required=True)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tag', default='v6_ck8dwell')
    a = p.parse_args()
    with np.load(a.shard, allow_pickle=False) as f:
        meta = json.loads(str(f['meta_json']))
        assert meta['phase_source']['kind'] == 'ck8'
        X = f['X'][:, list(meta['capture_layers']).index(12), 3,
                   list(meta['segment_names']).index('all'), :].astype(np.float32)
        ep, c, su = f['ep_id'], f['phase_code'], f['succ']
        jit = f['jitter'] if 'jitter' in f else f['jitter_idx']
        scenes, rec = f['scene'], f['rec_idx']
    assert len(set(scenes)) == 1 and np.isfinite(X).all()
    order = np.lexsort((rec, ep))
    X, ep, c, su, jit = (z[order] for z in (X, ep, c, su, jit))
    scene = int(scenes[0])
    allow = (jit != a.target) | (su == 0)
    keep, caps, contrib = capped_rows(ep, c, su, allow)
    registry, diag = [], []
    for form, suffix, beta in [('jfair', '', .9), ('plain', '_plain', .8)]:
        dest = a.out / f'instr_setm_{a.tag}_ck8{suffix}' / a.slug / f's{scene}' / f'j{a.target}'
        if dest.exists():
            raise FileExistsError(dest)
        dest.mkdir(parents=True)
        for cl in range(8):
            mask = keep & (c == cl)
            fit, reason = fit_one(X, ep, su, jit, mask, a.target, form)
            ids_f = np.unique(ep[mask & (su == 0)])
            row = dict(cluster=cl, form=form, registered=fit is not None, reason=reason,
                       n_succ_ep=len(np.unique(ep[mask & (su == 1)])), n_fail_ep=len(ids_f))
            registry.append(row)
            if fit is None:
                continue
            v, s = fit
            d = dest / f'c{cl}' / 'dit_L12'
            d.mkdir(parents=True)
            np.savez_compressed(d/'conceptors.npz', alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(s))
            md = dict(row, op='setpoint', layer=12, denoise=3, token_read='all_49',
                      phase=f'c{cl}', phase_label_source='ck8', phase_source=meta['phase_source'],
                      target_scene=scene, target_jitter=a.target, setm_form=form,
                      length_control=VERSION, cap=caps.get(cl),
                      contributions=[z for z in contrib if z['cluster'] == cl],
                      n_ep_excluded_succ_tgt=len(np.unique(ep[(jit == a.target) & (su == 1)])))
            (d/'metadata.json').write_text(json.dumps(md, indent=2)+'\n')
            anchor = episode_mean(X, ep, mask & (su == 0) & (jit == a.target))
            for e in ids_f:
                other, why = fit_one(X, ep, su, jit, mask & (ep != e), a.target, form)
                r = dict(cluster=cl, form=form, omitted_fail_ep=int(e),
                         n_fail_ep=len(ids_f), defined=other is not None, reason=why)
                if other is not None:
                    vv, ss = other
                    r.update(cosine_signed=float(v @ vv), setpoint_delta=float(ss-s))
                    if anchor is not None:
                        shift = beta*(s-anchor@v)*v
                        shifted = beta*(ss-anchor@vv)*vv
                        r['shift_delta_l2_at_full_target_anchor'] = float(np.linalg.norm(shifted-shift))
                diag.append(r)
        if not any(r['registered'] for r in registry[-8:]):
            (dest/'fallback_only.json').write_text(json.dumps(dict(fallback_only=True,
                length_control=VERSION, phase_source=meta['phase_source'],
                reason='no_estimable_cluster_operator'), indent=2)+'\n')
        (dest/'registry.json').write_text(json.dumps(registry[-8:], indent=2)+'\n')
    report = a.out/'diagnostics'/f'{a.slug}_s{scene}_j{a.target}.json'
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(dict(length_control=VERSION, caps=caps, contributions=contrib,
                                      registry=registry, failure_loo=diag,
                                      claim='diagnostic only; no LOO cutoff used'), indent=2)+'\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
