"""GPU (docker lerobot image, harness receipt): rollout decision latents for TRQAM from extract_scene last_block_d0.

latent = mean_{token 33:49}( dit.proj_out_2( dit.norm_out(x) * (1+scale) + shift ) ),
  shift, scale = dit.proj_out_1(silu(dit.timestep_encoder(0))).chunk(2),  x = last DiT block output at denoise 0
— the exact construction audited bit-exact (incl. fp16-quantized x) in
outputs/analysis/groot_robocasa_lineage_20261021/latent_reconstruction_v2/result.json (audit_latent_reconstruction_v2.py).
CONTRACT §6 forbids 1536->1024 *substitutes*; this is the same checkpoint's own tail recomputed bit-exactly (main-session
implementation decision 2026-10-29, recorded as such — not a user approval). Technical provenance contract instead:
audit v2 result pinned by sha and re-checked (denoise-0 native / fp16 exact), checkpoint revision + config/metadata sha
(contract.load_checkpoint_contract), use_bf16 flag, feature signature, source hashes, on-device slice-equivalence proof,
and a LIVE parity smoke (--parity-smoke): one real GR00T forward with hooks on the last block and action_decoder input;
the reconstructed latent from the f16-stored block (49- and 16-token paths) must torch.equal the captured decoder-input
latent. Rollout pkls hold no images, so the smoke observation is the first frame of OpenDrawer human demos in /cache.

usage (inside container, /work = workdir, under harness):
  python rollout_latents.py --parity-smoke --out /work/latents_v2
  python rollout_latents.py --episodes /work/episodes_v2 --out /work/latents_v2
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ADAPTER = REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'
AUDIT = REPO / 'outputs/analysis/groot_robocasa_lineage_20261021/latent_reconstruction_v2/result.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


AUDIT_SHA256 = 'b5a1f948fe7039b0870e4b53451934e066ef5c0d4ccbffb01a81789d87e01149'   # latent_reconstruction_v2/result.json


def audit_gate(path=AUDIT):
    """Audit file may be bundled (outputs/ is not git-tracked on srv hosts); its bytes are pinned by sha."""
    if sha(path) != AUDIT_SHA256:
        raise SystemExit(f'audit file {path} sha != pinned {AUDIT_SHA256}')
    r = json.loads(Path(path).read_text())
    ok = r.get('native_all_exact') and all(s['denoise'] != 0 or (s['native']['exact'] and s['fp16_tokens']['exact']
                                                                  and s['fp16_qam_latent']['exact'])
                                           for v in r['samples'].values() for s in v)
    if not ok:
        raise SystemExit('latent_reconstruction_v2 audit is not exact for denoise 0; refuse')
    return AUDIT_SHA256


EXPECT_REVISION = 'c484448aba1a9b60a04c9b0ca117241518ea69f3'


def checkpoint_revision(ck):
    """.../snapshots/<revision sha>/gr00t_n1-5/multitask_learning/checkpoint-120000 -> revision sha (pinned)."""
    ck = Path(ck)
    rel = ck.parts[-3:]
    if rel != ('gr00t_n1-5', 'multitask_learning', 'checkpoint-120000') or ck.parents[3].name != 'snapshots':
        raise SystemExit(f'unexpected checkpoint layout {ck}')
    rev = ck.parents[2].name
    if rev != EXPECT_REVISION:
        raise SystemExit(f'checkpoint revision {rev} != pinned {EXPECT_REVISION}')
    return rev


HOST_TO_ARCHIVE = {'kanu': 'kanu', 'srv48': 'worker1', 'srv50': 'worker2'}


def device_gate(episode_id, harness_machine, cross_device_ok):
    """Bit-exactness was shown on the device where the parity smoke ran (provenance pins the GPU name). Episodes are
    reconstructed on their own collection host unless a cross-device validation record is supplied."""
    ep_machine = episode_id.split('/')[1]
    here = HOST_TO_ARCHIVE.get(harness_machine)
    if here is None:
        raise SystemExit(f'HARNESS_MACHINE {harness_machine!r} unknown')
    if ep_machine != here and not cross_device_ok:
        raise SystemExit(f'{episode_id}: collected on {ep_machine}, running on {here}; cross-device equivalence not validated '
                         '(run on the collection host with its own --parity-smoke, or pass --cross-device-validated)')


def slice_proof(dit, tail_fn):
    """extract_scene may store only tokens 33:49 (16-token layout; the active bread v2 run stores 49 — both accepted).
    Valid only if the tail is per-token: tail(x)[:,33:49] == tail(x[:,33:49]) bit-exactly on this device/dtype."""
    import torch
    if not isinstance(dit.norm_out, torch.nn.LayerNorm):
        raise SystemExit(f'norm_out is {type(dit.norm_out).__name__}, not per-token LayerNorm; 16-token storage invalid')
    g = torch.Generator(device='cpu').manual_seed(20261029)
    probe = torch.randn(3, 49, 1536, generator=g).to(torch.float16).to('cuda:0', dtype=torch.bfloat16)
    if not torch.equal(tail_fn(probe)[:, 33:49], tail_fn(probe[:, 33:49].contiguous())):
        raise SystemExit('decoder-input tail not per-token exact on this device; only 49-token storage usable')
    return 'tail(x)[:,33:49]==tail(x[:,33:49]) torch.equal, 3 probes'


def parity_smoke(model, serve, dit, tail_fn, n_obs=2, data_root=Path('/cache/datasets')):
    """Live forward: decoder-input latent captured by hook == reconstruction from the f16-stored last block."""
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
            for h in hooks:
                h.remove()
        if len(blocks) != 4 or len(dec_in) != 4:
            raise SystemExit(f'parity smoke: expected 4 denoise calls, got {len(blocks)}/{len(dec_in)}')
        want = dec_in[0][:, -16:].float().mean(1)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.policy.config.use_bf16):
            stored = blocks[0].to(torch.float16).to(blocks[0].device, dtype=blocks[0].dtype)     # f16 archive round trip
            got49 = tail_fn(stored)[:, -16:].float().mean(1)
            got16 = tail_fn(stored[:, 33:49].contiguous()).float().mean(1)
        rows.append({'episode': ep, 'exact_49': bool(torch.equal(got49, want)), 'exact_16': bool(torch.equal(got16, want)),
                     'max_abs_16': float((got16 - want).abs().max())})
    if not all(r['exact_49'] and r['exact_16'] for r in rows):
        raise SystemExit(f'parity smoke failed: {rows}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parity-smoke', action='store_true')
    ap.add_argument('--cross-device-validated', type=Path,
                    help='json record (main session) showing reconstruction equality across the two GPU types')
    ap.add_argument('--data-root', type=Path, default=Path('/cache/datasets'),
                    help='holds the HF checkpoint snapshot and robocasa OpenDrawer demos (container: /cache/datasets)')
    ap.add_argument('--audit', type=Path, default=AUDIT, help='latent_reconstruction_v2/result.json (sha pinned)')
    ap.add_argument('--expect-records', type=int, help='conversion refuses unless exactly this many episode records exist')
    a = ap.parse_args()
    harness_machine = os.environ.get('HARNESS_MACHINE')
    if not os.environ.get('HARNESS_RECEIPT'):
        raise SystemExit('GPU step: run under groot harness (HARNESS_RECEIPT)')
    import torch
    sys.path[:0] = [str(ADAPTER), str(REPO)]
    from extract_opendrawer_latents import FrozenGroot, import_serve_module, checkpoint_dir, PROFILE_REL
    from contract import load_checkpoint_contract
    ck = checkpoint_dir(a.data_root)
    cc = load_checkpoint_contract(ck)            # config/metadata sha pinned (CONTRACT §1)
    serve = import_serve_module()
    model = FrozenGroot(serve, profile_path=REPO / PROFILE_REL, pretrained=ck, device='cuda:0')
    dit = model.head.model
    provenance = {'decision': 'main_session_implementation_decision_2026-10-29 (exact same-checkpoint tail; not a user approval)',
                  'audit_sha256': audit_gate(a.audit), 'checkpoint_dir_name': ck.name, 'checkpoint_revision': checkpoint_revision(ck),
                  'config_sha256': cc.config_sha256, 'use_bf16': bool(model.policy.config.use_bf16),
                  'script_sha256': sha(__file__), 'extractor_helper_sha256': sha(ADAPTER / 'extract_opendrawer_latents.py'),
                  'feature_signature': 'ccbfb51ca01724cb', 'tokens': [33, 49], 'denoise': 0, 'conversion_batch': 1,
                  'device_name': torch.cuda.get_device_name(0), 'harness_machine': harness_machine,
                  'torch': torch.__version__, 'cuda': torch.version.cuda}
    tail_fn = lambda x: dit.proj_out_2(dit.norm_out(x) * (1 + _ss[1][:, None]) + _ss[0][:, None])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.policy.config.use_bf16):
        temb0 = dit.timestep_encoder(torch.tensor([0], device='cuda:0', dtype=torch.long))
        _ss = dit.proj_out_1(torch.nn.functional.silu(temb0)).chunk(2, dim=1)
        provenance['slice_equivalence'] = slice_proof(dit, tail_fn)
    if a.parity_smoke:
        a.out.mkdir(parents=True, exist_ok=True)
        rows = parity_smoke(model, serve, dit, tail_fn, data_root=a.data_root)
        (a.out / 'PARITY_SMOKE.json').write_text(json.dumps({'provenance': provenance, 'rows': rows, 'passed': True}, indent=1))
        print(json.dumps(rows)); return
    if a.episodes is None:
        raise SystemExit('--episodes required (or --parity-smoke)')
    ps = a.out / 'PARITY_SMOKE.json'
    if not ps.exists() or json.loads(ps.read_text())['provenance'] != provenance:
        raise SystemExit('run --parity-smoke first with the same code/checkpoint (PARITY_SMOKE.json provenance must match)')
    contract = {**provenance, 'parity_smoke_sha256': sha(ps), 'episodes_contract_sha256': sha(a.episodes / 'contract.json'),
                'cross_device_validated_sha256': sha(a.cross_device_validated) if a.cross_device_validated else None}
    a.out.mkdir(parents=True, exist_ok=True)
    cp = a.out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise SystemExit('contract mismatch; new --out')
    cp.write_text(json.dumps(contract, indent=1))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.policy.config.use_bf16):
        tail = tail_fn
        recs = sorted(p for p in a.episodes.glob('*.json') if p.name not in ('contract.json', 'COMPLETE.json'))
        if a.expect_records is not None and len(recs) != a.expect_records:
            raise SystemExit(f'{len(recs)} episode records != expected {a.expect_records}; staging incomplete')
        for i, jp in enumerate(recs):
            rec = json.loads(jp.read_text())
            device_gate(rec['episode_id'], harness_machine, a.cross_device_validated is not None)
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
            if lb.shape[1:] not in ((49, 1536), (16, 1536)):
                raise SystemExit(f'last_block_d0 shape {lb.shape}')
            outs = []
            for s in range(len(lb)):
                # batch 1, exactly the shapes the live parity smoke proved bit-exact: [1,49,1536] (native call) or
                # [1,16,1536] (got16 path). Larger batches change the GEMM row count and are NOT covered by the proof.
                x = torch.from_numpy(lb[s:s + 1]).to('cuda:0', dtype=torch.bfloat16)   # f16 archive -> model dtype
                dec = tail(x)
                outs.append(dec[:, -16:].float().mean(1).cpu().numpy())        # 16-token input: all of it
            lat = np.concatenate(outs).astype(np.float32)
            if lat.shape != (rec['n_inferences'], 1024) or not np.isfinite(lat).all():
                raise SystemExit(f'bad latent {rec["episode_id"]} {lat.shape}')
            npz = a.out / rec['npz']
            np.savez(npz.with_suffix('.tmp.npz'), latents=lat); os.replace(npz.with_suffix('.tmp.npz'), npz)
            out_j.write_text(json.dumps({'episode_id': rec['episode_id'], 'pkl_sha256': rec['pkl_sha256'],
                                         'source_npz_sha256': rec['npz_sha256'], 'npz': npz.name,
                                         'npz_sha256': sha(npz), 'n': int(len(lat)), 'time': time.time()}, indent=1))
            print(json.dumps({'done': i + 1, 'total': len(recs)}), flush=True)
    done = sum(1 for p in a.out.glob('*.json') if p.name not in ('contract.json', 'COMPLETE.json', 'PARITY_SMOKE.json'))
    if done != len(recs) or (a.expect_records is not None and done != a.expect_records):
        raise SystemExit(f'latent records {done} != episodes {len(recs)} / expected {a.expect_records}; no COMPLETE')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'contract': contract, 'episodes': done}, indent=1))


if __name__ == '__main__':
    main()
