#!/usr/bin/env python3
"""Reconstruct prior v6 plain operators from untruncated phase records."""
import argparse
import json
from pathlib import Path

import numpy as np
import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--references', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    shards = fit.Shards(json.loads((args.root/'shard_map.json').read_text()))
    references = json.loads(args.references.read_text())
    results = []
    for ref in references:
        # One manifest episode loads and validates the complete referenced shard.
        shards.load(ref['loader_row'])
        episodes = shards.cache[ref['mapkey']]
        groups = {0: [], 1: []}
        target_fail = []
        for (_, jitter, _), (x, phase, success, _) in episodes.items():
            if success and jitter == ref['target_jitter']:
                continue
            selected = x[phase == ref['phase']]
            if len(selected):
                groups[success].append(selected)
                if not success and jitter == ref['target_jitter']:
                    target_fail.append(selected)
        xs, xf = np.concatenate(groups[1]), np.concatenate(groups[0])
        # Match original float32 record pooling before float64 delta.
        ms, mf = xs.mean(0).astype(np.float64), xf.mean(0).astype(np.float64)
        d = ms-mf
        v = d/(np.linalg.norm(d)+1e-12)
        s = float(ms@v)
        vr = np.asarray(ref['v'])
        result = {'mapkey': ref['mapkey'], 'phase': ref['phase'], 'target_jitter': ref['target_jitter'],
                  'raw_counts': [len(xs),len(xf),sum(map(len,target_fail))],
                  'metadata_counts': ref['counts'], 'cosine': float(v@vr/np.linalg.norm(vr)),
                  'max_vector_error': float(np.max(np.abs(v-vr))), 'setpoint_error': s-ref['s'],
                  'source_sha256': next(iter(episodes.values()))[3]['sha256']}
        result['reproduced'] = (result['raw_counts']==result['metadata_counts'] and
                                result['max_vector_error'] < 2e-5 and abs(result['setpoint_error']) < .002)
        results.append(result)
        print(json.dumps(result),flush=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
