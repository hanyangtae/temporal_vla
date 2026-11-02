"""rollout_latents v2: rollout decision latents with an EXPLICIT token-storage contract (v1 rollout_latents.py is frozen;
only its pinned helpers are imported). Main runs this on the collection host's GPU under the harness; Opus does not.

Why v2: v1 runs slice_proof unconditionally — a SYNTHETIC [3,49,1536] vs [3,16,1536] torch.equal of the decoder-input
tail — and its parity smoke requires exact_49 AND exact_16. That proof is only needed when the archive stores the 16-token
slice (tokens 33:49). It blocked srv50, whose bread run stores all 49 tokens. v2 separates the two contracts; neither is
loosened:
  --token-contract 49   archive last_block_d0 must be float16 [n,49,1536] (every record). No slicing happens:
                        latent = mean_{-16:}( tail(x[s:s+1]) ) with x the f16 archive -> model dtype, batch 1, the
                        exact call shape the LIVE parity smoke proves. Smoke gate: exact_49 on every smoke observation
                        (real GR00T forward, decoder-input pre-hook vs reconstruction from the f16-round-tripped block).
                        exact_16 is recorded for information only and never used to admit 16-token data.
  --token-contract 16   archive must be float16 [n,16,1536]. v1's slice_proof (unchanged) AND smoke exact_16 AND exact_49
                        are all required, exactly as v1.
Provenance (smoke and conversion must be byte-identical, else refuse): v1 fields + token_contract + kernel/precision
state (TF32 flags, float32 matmul precision, cuDNN deterministic/benchmark, deterministic algorithms, CUBLAS/TF32 env,
GPU name + capability, torch/CUDA/cuDNN versions, autocast dtype). Outputs: PARITY_SMOKE_V2.json / contract.json
(version rollout_latents_v2) in a NEW --out; a directory holding v1 artefacts (PARITY_SMOKE.json or a contract without
version v2) is refused, never overwritten.
--diagnose (GPU, informational): where does tail(x)[:,33:49] vs tail(x[:,33:49]) inequality arise on this device —
LayerNorm vs proj_out_2 GEMM, batch 1 vs 3, synthetic vs smoke block, TF32 state — written to DIAG_V2.json. It never
admits data.

  python rollout_latents_v2.py --token-contract 49 --parity-smoke --out /work/latents_v2_t49
  python rollout_latents_v2.py --token-contract 49 --episodes /work/episodes --out /work/latents_v2_t49 --expect-records 200
  python rollout_latents_v2.py --token-contract 49 --diagnose --out /work/diag_v2
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rollout_latents as V1   # noqa: E402  (frozen; helpers only)

VERSION = 'rollout_latents_v2'
SHAPES = {49: (49, 1536), 16: (16, 1536)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ pure gates (CPU-tested)
def check_record(lb, contract):
    """Archive block must be float16 with exactly the contract's token count; nothing else is admitted."""
    if lb.dtype != np.float16:
        raise SystemExit(f'last_block_d0 dtype {lb.dtype} != float16 (archive contract)')
    if lb.ndim != 3 or tuple(lb.shape[1:]) != SHAPES[contract]:
        raise SystemExit(f'last_block_d0 shape {tuple(lb.shape)} does not match --token-contract {contract} {SHAPES[contract]}')
    return True


def smoke_verdict(rows, contract):
    if not rows:
        raise SystemExit('parity smoke produced no rows')
    need = ('exact_49',) if contract == 49 else ('exact_49', 'exact_16')
    bad = [r for r in rows if not all(r.get(k) is True for k in need)]
    if bad:
        raise SystemExit(f'parity smoke failed for --token-contract {contract} (required {need}): {bad}')
    # the proof covers conversion only for the call shape it ran: batch 1, full 49-token block
    shape_bad = [r for r in rows if r.get('block_shape') != [1, 49, 1536] or (r.get('decoder_input_shape') or [None])[0] != 1]
    if shape_bad:
        raise SystemExit(f'parity smoke rows lack batch-1 [1,49,1536] block / batch-1 decoder input: {shape_bad}')
    return {'required': list(need), 'passed': True, 'call_shape': [1, 49, 1536]}


def kernel_settings(torch):
    cuda = torch.cuda.is_available()
    return {'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
            'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
            'env': {k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'device_name': torch.cuda.get_device_name(0) if cuda else 'cpu',
            'device_capability': list(torch.cuda.get_device_capability(0)) if cuda else None}


def guard_out(out):
    """Refuse directories carrying v1 artefacts; never overwrite them."""
    if (out / 'PARITY_SMOKE.json').exists():
        raise SystemExit(f'{out} holds a v1 PARITY_SMOKE.json; use a new --out for v2')
    cp = out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()).get('version') != VERSION:
        raise SystemExit(f'{out}/contract.json is not a {VERSION} contract; use a new --out')


def require_smoke(out, provenance):
    ps = out / 'PARITY_SMOKE_V2.json'
    if not ps.exists():
        raise SystemExit('run --parity-smoke first (PARITY_SMOKE_V2.json missing)')
    d = json.loads(ps.read_text())
    if d.get('provenance') != provenance:
        diff = sorted(k for k in set(provenance) | set(d.get('provenance') or {})
                      if (d.get('provenance') or {}).get(k) != provenance.get(k))
        raise SystemExit(f'PARITY_SMOKE_V2 provenance differs from this run in {diff}; rerun the smoke here')
    smoke_verdict(d.get('rows'), provenance['token_contract'])      # re-check rows; a hand-edited 'passed' is not enough
    return sha(ps)


# ------------------------------------------------------------------ device work (GPU, main)
def reconstruct(lb, tail_fn, torch, device, dtype):
    """latent per inference: mean over the last 16 tokens of tail(block), batch 1 (the smoke-proven call shape)."""
    outs = []
    for s in range(len(lb)):
        x = torch.from_numpy(lb[s:s + 1]).to(device, dtype=dtype)
        outs.append(tail_fn(x)[:, -16:].float().mean(1).detach().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def diagnose(dit, tail_fn, torch, device, dtype, smoke_block=None):
    """Informational: equality of tail(x)[:,33:49] vs tail(x[:,33:49]) and of its two stages, per batch size."""
    res = {}
    g = torch.Generator(device='cpu').manual_seed(20261029)
    sources = {'synthetic': torch.randn(3, 49, 1536, generator=g).to(torch.float16)}
    if smoke_block is not None:
        sources['smoke_block'] = smoke_block.to(torch.float16).cpu()
    for name, src in sources.items():
        for b in sorted({1, src.shape[0]}):
            x = src[:b].to(device, dtype=dtype)
            n_full, n_cut = dit.norm_out(x)[:, 33:49], dit.norm_out(x[:, 33:49].contiguous())
            t_full, t_cut = tail_fn(x)[:, 33:49], tail_fn(x[:, 33:49].contiguous())
            res[f'{name}_b{b}'] = {
                'layernorm_equal': bool(torch.equal(n_full, n_cut)),
                'tail_equal': bool(torch.equal(t_full, t_cut)),
                'tail_max_abs': float((t_full.float() - t_cut.float()).abs().max()),
                'gemm_rows': [int(b * 49), int(b * 16)]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-contract', type=int, choices=(16, 49), required=True)
    ap.add_argument('--episodes', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parity-smoke', action='store_true'); ap.add_argument('--diagnose', action='store_true')
    ap.add_argument('--cross-device-validated', type=Path)
    ap.add_argument('--data-root', type=Path, default=Path('/cache/datasets'))
    ap.add_argument('--audit', type=Path, default=V1.AUDIT)
    ap.add_argument('--expect-records', type=int)
    a = ap.parse_args()
    if not os.environ.get('HARNESS_RECEIPT'):
        raise SystemExit('GPU step: run under groot harness (HARNESS_RECEIPT)')
    a.out.mkdir(parents=True, exist_ok=True)
    guard_out(a.out)
    import torch
    sys.path[:0] = [str(V1.ADAPTER), str(V1.REPO)]
    from extract_opendrawer_latents import FrozenGroot, import_serve_module, checkpoint_dir, PROFILE_REL
    from contract import load_checkpoint_contract
    ck = checkpoint_dir(a.data_root)
    cc = load_checkpoint_contract(ck)
    serve = import_serve_module()
    model = FrozenGroot(serve, profile_path=V1.REPO / PROFILE_REL, pretrained=ck, device='cuda:0')
    dit = model.head.model
    use_bf16 = bool(model.policy.config.use_bf16)
    provenance = {'version': VERSION, 'token_contract': a.token_contract,
                  'audit_sha256': V1.audit_gate(a.audit), 'checkpoint_dir_name': ck.name, 'checkpoint_revision': V1.checkpoint_revision(ck),
                  'config_sha256': cc.config_sha256, 'use_bf16': use_bf16, 'autocast_dtype': 'bfloat16' if use_bf16 else None,
                  'script_sha256': sha(__file__), 'v1_helpers_sha256': sha(V1.__file__),
                  'extractor_helper_sha256': sha(V1.ADAPTER / 'extract_opendrawer_latents.py'),
                  'feature_signature': 'ccbfb51ca01724cb', 'latent_tokens': 'last 16 of the stored block', 'denoise': 0,
                  'conversion_batch': 1, 'archive_dtype': 'float16', 'archive_shape': list(SHAPES[a.token_contract]),
                  'harness_machine': os.environ.get('HARNESS_MACHINE'), 'kernel': kernel_settings(torch)}
    tail_fn = lambda x: dit.proj_out_2(dit.norm_out(x) * (1 + _ss[1][:, None]) + _ss[0][:, None])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        temb0 = dit.timestep_encoder(torch.tensor([0], device='cuda:0', dtype=torch.long))
        _ss = dit.proj_out_1(torch.nn.functional.silu(temb0)).chunk(2, dim=1)
        if a.diagnose:
            d = diagnose(dit, tail_fn, torch, 'cuda:0', torch.bfloat16)
            (a.out / 'DIAG_V2.json').write_text(json.dumps({'provenance': provenance, 'diag': d}, indent=1))
            print(json.dumps(d)); return
        if a.token_contract == 16:
            provenance['slice_equivalence'] = V1.slice_proof(dit, tail_fn)   # unchanged v1 gate for 16-token data
    if a.parity_smoke:
        rows = parity_smoke_rows(model, serve, dit, tail_fn, use_bf16, a.data_root)
        verdict = smoke_verdict(rows, a.token_contract)
        p = a.out / 'PARITY_SMOKE_V2.json'
        if p.exists():
            raise SystemExit(f'{p} exists; smoke results are never overwritten (new --out)')
        p.write_text(json.dumps({'provenance': provenance, 'rows': rows, 'verdict': verdict}, indent=1))
        print(json.dumps(rows)); return
    if a.episodes is None:
        raise SystemExit('--episodes required (or --parity-smoke / --diagnose)')
    smoke_sha = require_smoke(a.out, provenance)
    contract = {**provenance, 'parity_smoke_v2_sha256': smoke_sha, 'episodes_contract_sha256': sha(a.episodes / 'contract.json'),
                'cross_device_validated_sha256': sha(a.cross_device_validated) if a.cross_device_validated else None}
    cp = a.out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise SystemExit('contract mismatch; new --out')
    cp.write_text(json.dumps(contract, indent=1))
    recs = sorted(p for p in a.episodes.glob('*.json') if p.name not in ('contract.json', 'COMPLETE.json'))
    if a.expect_records is not None and len(recs) != a.expect_records:
        raise SystemExit(f'{len(recs)} episode records != expected {a.expect_records}')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i, jp in enumerate(recs):
            rec = json.loads(jp.read_text())
            V1.device_gate(rec['episode_id'], os.environ.get('HARNESS_MACHINE'), a.cross_device_validated is not None)
            out_j = a.out / jp.name
            if out_j.exists():
                lr = json.loads(out_j.read_text())
                if lr['episode_id'] != rec['episode_id'] or sha(a.out / lr['npz']) != lr['npz_sha256']:
                    raise SystemExit(f'committed latent mismatch {rec["episode_id"]}')
                continue
            if sha(a.episodes / rec['npz']) != rec['npz_sha256']:
                raise SystemExit(f'episode npz sha {rec["episode_id"]}')
            with np.load(a.episodes / rec['npz']) as z:
                if 'last_block_d0' not in z.files:
                    raise SystemExit('episodes extracted without --with-last-block')
                lb = z['last_block_d0']
            check_record(lb, a.token_contract)
            lat = reconstruct(lb, tail_fn, torch, 'cuda:0', torch.bfloat16)
            if lat.shape != (rec['n_inferences'], 1024) or not np.isfinite(lat).all():
                raise SystemExit(f'bad latent {rec["episode_id"]} {lat.shape}')
            npz = a.out / rec['npz']
            np.savez(npz.with_suffix('.tmp.npz'), latents=lat); os.replace(npz.with_suffix('.tmp.npz'), npz)
            out_j.write_text(json.dumps({'episode_id': rec['episode_id'], 'pkl_sha256': rec['pkl_sha256'],
                                         'source_npz_sha256': rec['npz_sha256'], 'npz': npz.name, 'npz_sha256': sha(npz),
                                         'n': int(len(lat)), 'token_contract': a.token_contract, 'time': time.time()}, indent=1))
            print(json.dumps({'done': i + 1, 'total': len(recs)}), flush=True)
    done = sum(1 for p in a.out.glob('*.json') if p.name not in ('contract.json', 'COMPLETE.json', 'PARITY_SMOKE_V2.json'))
    if done != len(recs) or (a.expect_records is not None and done != a.expect_records):
        raise SystemExit(f'latent records {done} != episodes {len(recs)} / expected {a.expect_records}; no COMPLETE')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'contract': contract, 'episodes': done}, indent=1))


def parity_smoke_rows(model, serve, dit, tail_fn, use_bf16, data_root, n_obs=2):
    """Same live forward as v1.parity_smoke (real GR00T call, hooks on the last block and the action_decoder input),
    but returns the rows with batch/shape/dtype provenance and leaves the pass decision to smoke_verdict()."""
    import torch, av
    from extract_opendrawer_latents import png_b64, state_payload_map, video_payload_map
    import pandas as pd
    D = Path(data_root) / 'robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot'
    mod = json.loads((D / 'meta/modality.json').read_text())
    sm, vm = state_payload_map(mod, model.profile), video_payload_map(mod)
    rows = []
    for ep in range(n_obs):
        df = pd.read_parquet(D / f'data/chunk-000/episode_{ep:06d}.parquet')
        state = np.asarray(df['observation.state'].iloc[0])
        lang = json.loads((D / f'extras/episode_{ep:06d}/ep_meta.json').read_text())['lang']
        frames = {}
        for video, key in vm.items():
            with av.open(str(D / 'videos/chunk-000' / video / f'episode_{ep:06d}.mp4')) as c:
                frames[key] = next(c.decode(video=0)).to_ndarray(format='rgb24')
        payload = {**{k: png_b64(v) for k, v in frames.items()}, **{k: state[lo:hi].tolist() for k, (lo, hi) in sm.items()}, 'task': lang}
        blocks, dec_in = [], []
        hooks = [dit.transformer_blocks[-1].register_forward_hook(lambda m, i, o: blocks.append(o.detach().clone())),
                 model.head.action_decoder.register_forward_pre_hook(lambda m, i: dec_in.append(i[0].detach().clone()))]
        try:
            batch = model.preprocessor(serve._apply_input_remap(serve.parse_payload(payload)))
            serve._apply_inference_seed({'inference_seed': 1400000 + ep})
            with torch.inference_mode():
                model.policy.predict_action_chunk(batch)
        finally:
            for hk in hooks:
                hk.remove()
        if len(blocks) != 4 or len(dec_in) != 4:
            raise SystemExit(f'parity smoke: expected 4 denoise calls, got {len(blocks)}/{len(dec_in)}')
        want = dec_in[0][:, -16:].float().mean(1)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            stored = blocks[0].to(torch.float16).to(blocks[0].device, dtype=blocks[0].dtype)
            got49 = tail_fn(stored)[:, -16:].float().mean(1)
            got16 = tail_fn(stored[:, 33:49].contiguous()).float().mean(1)
        rows.append({'episode': ep, 'block_shape': list(blocks[0].shape), 'block_dtype': str(blocks[0].dtype),
                     'decoder_input_shape': list(dec_in[0].shape), 'decoder_input_dtype': str(dec_in[0].dtype),
                     'device': str(blocks[0].device), 'exact_49': bool(torch.equal(got49, want)),
                     'exact_16': bool(torch.equal(got16, want)), 'max_abs_49': float((got49 - want).abs().max()),
                     'max_abs_16': float((got16 - want).abs().max())})
    return rows


if __name__ == '__main__':
    main()
