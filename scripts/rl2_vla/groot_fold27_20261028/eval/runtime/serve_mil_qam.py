"""fold27 serve: scripts/serve/lerobot.py + per-inference fold MIL gate + QAM composition.

Launch form = serve_safe_qam.py (proven): kanu -> MODEL_IMAGE container, srv48/50 -> host conda lerobot_050_groot.
    serve_mil_qam.py --mode gated --qam-actor <actor dir> --actor-sha256 <sha> --mil-model <model.pt>
        --mil-sha256 <sha> --target-j <j> [--smoke] -- <lerobot.py args: --profile ... --host ... --port ...>
Production refuses: mode != gated, diagnostic/smoke actor, actor without production 500k provenance,
MIL model without a fold FOLD_CONTRACT.json (shared checkpoints) or wrong target_j, tc commit mismatch.
Decision log: $PDK_LOG_FILE -> .decisions.jsonl (one line per inference), .received.jsonl (payload hashes).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
sys.path.insert(0, str(C.ADAPTER))


def validate_actor_meta(meta, *, smoke):
    """Actor dir written by restore_actor.py (actor.json + runtime_provenance.json)."""
    prov = meta.get('runtime_provenance') or {}
    if smoke:
        return
    C.need(not meta.get('diagnostic_only'), 'production refuses a diagnostic-only actor')
    C.need(prov.get('production') is True and prov.get('smoke') is False, 'actor provenance is not production')
    C.need(int(prov.get('common_steps', -1)) == C.COMMON_STEPS, f'actor not derived from the {C.COMMON_STEPS} common QAM')
    C.need('smoke' not in str(meta.get('flags', {}).get('data_label', '')).lower(), 'smoke data_label in actor flags')


def validate_mil(model_path, sha, target_j, *, smoke):
    import mil_gate as G
    C.need(C.sha256_path(model_path) == sha, f'MIL model sha mismatch {model_path}')
    gate = G.MilGate.from_run(model_path, expected_sha256=sha, require_target_jitter=None if smoke else target_j)
    if not smoke:
        fc = Path(model_path).parent / 'FOLD_CONTRACT.json'
        C.need(fc.is_file(), 'production MIL needs the fold FOLD_CONTRACT.json (shared48 checkpoints are smoke-only)')
        d = json.loads(fc.read_text())
        C.need(d['model_sha256'] == sha and d['shared_checkpoint_reused'] is False and int(d['target_j']) == int(target_j),
               'FOLD_CONTRACT mismatch')
        C.need(d.get('tc_commit_expected') == C.TC_COMMIT, 'MIL tc commit mismatch')
    C.need(gate.threshold == 0.5, f'MIL threshold {gate.threshold} != 0.5')
    return gate


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--mode', choices=['gated', 'base', 'always'], required=True)
    p.add_argument('--qam-actor', type=Path); p.add_argument('--actor-sha256')
    p.add_argument('--mil-model', type=Path, required=True); p.add_argument('--mil-sha256', required=True)
    p.add_argument('--target-j', type=int, required=True); p.add_argument('--smoke', action='store_true')
    a, rest = p.parse_known_args()
    if rest and rest[0] == '--':
        rest = rest[1:]
    C.need(a.smoke or a.mode == 'gated', f'mode {a.mode} is smoke-only')
    actor_meta = None
    if a.mode != 'base':
        C.need(a.qam_actor is not None and a.actor_sha256, 'actor dir and sha required')
        C.need(C.sha256_path(a.qam_actor / 'actor.npz') == a.actor_sha256, 'actor.npz sha mismatch')
        actor_meta = json.loads((a.qam_actor / 'actor.json').read_text())
        prov = a.qam_actor / 'runtime_provenance.json'
        C.need(prov.is_file(), 'actor runtime_provenance.json missing (export with restore_actor.py)')
        actor_meta['runtime_provenance'] = json.loads(prov.read_text())
        validate_actor_meta(actor_meta, smoke=a.smoke)
    import mil_gate as G
    C.need(G.tc_commit() == C.TC_COMMIT, f'task_classification HEAD {G.tc_commit()} != {C.TC_COMMIT}')
    gate = validate_mil(a.mil_model, a.mil_sha256, a.target_j, smoke=a.smoke)

    import torch
    from extract_opendrawer_latents import import_serve_module
    from composition import ComposedFlowSampler, SPEC, check_head_contract
    from qam_torch_actor import TorchQAMVelocity
    from task_language_guard import canonical_hash
    from gated_action import GatedAction
    C.need(SPEC.signature() == C.FEATURE_SIGNATURE, 'feature signature drift')
    sv = import_serve_module()
    from lerobot_adapters.groot import GrootPolicyAdapter
    log = Path(os.environ['PDK_LOG_FILE'])
    stream = log.with_suffix('.decisions.jsonl').open('x', buffering=1)
    received = log.with_suffix('.received.jsonl').open('x', buffering=1)
    state = {'ready': False, 'mode': a.mode, 'smoke': a.smoke, 'mil': gate.spec(), 'target_j': a.target_j,
             'actor_sha256': a.actor_sha256, 'actor_provenance': None if actor_meta is None else actor_meta['runtime_provenance'],
             'qam_weight': C.QAM_WEIGHT, 'feature_signature': SPEC.signature()}
    seed0 = sv._apply_inference_seed

    def seed(payload):
        if 'task' in payload:
            received.write(json.dumps({'payload_sha256': canonical_hash(payload), 'task': payload['task'],
                                       'inference_seed': payload.get('inference_seed'), 'time': time.time()}) + '\n')
        return seed0(payload)
    sv._apply_inference_seed = seed
    original = GrootPolicyAdapter.load

    def load(self, *args, **kwargs):
        loaded = original(self, *args, **kwargs)
        loaded.policy.eval(); loaded.policy.requires_grad_(False)
        head = loaded.policy._groot_model.action_head
        check_head_contract(head)
        C.need(torch.cuda.is_available(), 'serve must run on CUDA')
        actor = None if a.mode == 'base' else TorchQAMVelocity(a.qam_actor, device='cpu')
        sampler = ComposedFlowSampler(SPEC, qam_fn=actor, qam_weight=C.QAM_WEIGHT)
        ga = GatedAction(head, head.get_action, gate, sampler, a.mode, use_cuda=True,
                         emit=lambda r: stream.write(json.dumps(r) + '\n'))
        reset0 = loaded.policy.reset

        def reset(*args, **kwargs):
            ga.reset()
            return reset0(*args, **kwargs)
        loaded.policy.reset = reset
        head.get_action = ga
        state['runtime'] = ga.state
        state['ready'] = True
        return loaded
    GrootPolicyAdapter.load = load

    @sv.app.get('/qam_contract')
    def contract():
        out = dict(state)
        if 'runtime' in state:
            out.update(state['runtime'])
        return out
    sys.argv = [str(Path(sv.__file__)), *rest]
    sv.main()


if __name__ == '__main__':
    main()
