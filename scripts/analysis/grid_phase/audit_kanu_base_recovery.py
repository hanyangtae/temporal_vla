#!/usr/bin/env python3
"""Read-only reconstruction audit; never equates outcome agreement with replay identity."""
import json
import pickle
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/collect/kanu_base_recovery_20260916'
CONFIG = ROOT / 'configs/experiments/kanu_base_recovery_20260916'

def main():
    while True:
        state = json.loads((OUT / 'state.json').read_text())
        if state['phase'] in ('complete', 'failed'):
            break
        time.sleep(30)
    rows = json.loads((CONFIG / 'targets.json').read_text())
    results = []
    for row in rows:
        dest = OUT / 'grid' / row['original_rel_path']
        if not (dest / 'video.mp4').exists():
            results.append({'id': row['id'], 'status': 'missing'})
            continue
        with (dest / 'rollout.pkl').open('rb') as f:
            new = pickle.load(f)
        old = json.loads((ROOT / row['steering_trace']).read_text())
        old_flags = old.get('perstep_fired', [])
        boundary = next((i for i, fired in enumerate(old_flags) if fired), len(old_flags))
        comparisons = {}
        for field in ('speed', 'jerk'):
            a = old.get('action_kinematics', {}).get(field, [])[:boundary]
            b = new.get('action_kinematics', {}).get(field, [])[:boundary]
            comparisons[field] = {
                'expected_records': boundary, 'old_records': len(a), 'new_records': len(b),
                'max_abs_diff': max((abs(x-y) for x,y in zip(a,b)), default=None),
                'exact_summary_match': bool(a) and a == b,
            }
        old_meta, new_meta = old.get('ep_meta', {}), new.get('ep_meta', {})
        meta_diff = [k for k in sorted(set(old_meta)|set(new_meta)) if old_meta.get(k) != new_meta.get(k)]
        results.append({
            'id': row['id'], 'status': 'reconstructed',
            'original_success': row['original_success'], 'reproduced_success': int(new['episode_success']),
            'outcome_match': row['original_success'] == int(new['episode_success']),
            'steering_initial_metadata_differing_fields': meta_diff,
            'pre_intervention_action_summary_comparison': comparisons,
            'full_original_action_state_match': 'unverified: original action/state artifact unavailable',
            'usable_as_identical_historical_baseline': False,
        })
        del new
    report = {'execution_state': state, 'episodes': results,
              'limitation': 'Action norms and metadata are partial diagnostics, not full action/state identity.'}
    (OUT / 'reproduction_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'audited': sum(r['status']=='reconstructed' for r in results), 'total': len(rows)}), flush=True)

if __name__ == '__main__':
    main()
