"""Per-inference MIL gate + QAM composition around the frozen GR00T action head (torch; serve process only).

Same structure as the proven groot_robocasa_adapter_20261021/serve_safe_qam.py (SAFE detector -> fold MIL):
  1. diagnostic first pass = native head.get_action (vanilla 4-step denoise) with a forward hook on
     transformer_blocks[12]; its LAST denoise call (k=3) hidden is the MIL input (known only after the pass).
  2. MIL w_F on that hidden, every inference (no latch). fired = w_F > checkpoint threshold (0.5).
  3. not gated -> the first-pass output is returned UNCHANGED (exact vanilla action, RNG already advanced once).
     gated     -> rewind CPU/CUDA RNG to the state before the first pass and run ComposedFlowSampler(gate=1)
                  so the composed denoise starts from the SAME initial noise; QAM feature = decoder input
                  last16 mean at flow step 0 of that pass (identical to the first pass at step 0).
  No reseed and no second random draw: after either path the RNG state equals the post-first-pass state.
Audits per request: initial-noise sha of every pass (action_encoder first input), RNG-state equality,
one-time gate0 composed == native identity, MIL score/first_fire_t, QAM delta.
"""
import copy
import hashlib
import time

import numpy as np
import torch

MIL_BLOCK = 12
MODES = ('gated', 'base', 'always')


class GateRuntimeError(RuntimeError):
    pass


def _sha_tensor(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _cloned(x):
    return type(x)(data={k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in x.items()})


class RngState:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.cpu = torch.random.get_rng_state()
        self.gpu = torch.cuda.get_rng_state() if use_cuda else None
        self.np = np.random.get_state()

    def restore(self):
        torch.random.set_rng_state(self.cpu)
        if self.use_cuda:
            torch.cuda.set_rng_state(self.gpu)
        np.random.set_state(self.np)

    def same_as_now(self):
        ok = torch.equal(self.cpu, torch.random.get_rng_state())
        if self.use_cuda:
            ok = ok and torch.equal(self.gpu, torch.cuda.get_rng_state())
        n = np.random.get_state()
        return bool(ok and n[2] == self.np[2] and np.array_equal(n[1], self.np[1]))


class GatedAction:
    """Replacement for head.get_action. `gate` has the MilGate surface (feature_from_hidden/step/snapshot/restore/reset)."""

    def __init__(self, head, native, gate, sampler, mode, *, use_cuda, emit=None):
        if mode not in MODES:
            raise GateRuntimeError(f'mode {mode}')
        if mode != 'base' and sampler.qam_fn is None:
            raise GateRuntimeError(f'mode {mode} needs a QAM actor')
        if not isinstance(head.action_encoder, torch.nn.Module):
            raise GateRuntimeError('action_encoder must be an nn.Module (initial-noise audit hook)')
        self.head, self.native, self.gate, self.sampler, self.mode = head, native, gate, sampler, mode
        self.use_cuda, self.emit = use_cuda, emit
        self.state = {'mode': mode, 'requests': 0, 'gate_calls': 0, 'composed_calls': 0, 'fired_requests': 0,
                      'off_identity_verified': False, 'episode_resets': 0, 'first_fire_t': None}

    def reset(self):
        self.gate.reset()
        self.state['episode_resets'] += 1
        self.state['first_fire_t'] = None

    def _pass(self, fn):
        hidden, noise = [], []

        def hook(_m, _a, out):
            if out.shape[0] != 1:
                raise GateRuntimeError(f'batch {out.shape[0]} != 1')
            hidden.append(out.detach().float().cpu().numpy().astype(np.float16)[0])

        def pre(_m, args):
            noise.append(args[0].detach().clone())
        h1 = self.head.model.transformer_blocks[MIL_BLOCK].register_forward_hook(hook)
        h2 = self.head.action_encoder.register_forward_pre_hook(pre)
        try:
            out = fn()
        finally:
            h1.remove(); h2.remove()
        if len(hidden) != 4 or len(noise) != 4:
            raise GateRuntimeError(f'expected 4 denoise calls, got hidden {len(hidden)} encoder {len(noise)}')
        return out, np.stack(hidden)[None], _sha_tensor(noise[0])

    def __call__(self, backbone_output, action_input):
        started = time.perf_counter()
        before = RngState(self.use_cuda)
        base, hidden, noise0 = self._pass(lambda: self.native(_cloned(backbone_output), action_input))
        base_action = base['action_pred']
        after = RngState(self.use_cuda)
        snap = self.gate.snapshot()
        pre = self.gate.step(self.gate.feature_from_hidden(hidden, 0))
        self.state['gate_calls'] += 1
        if pre['fired']:
            self.state['fired_requests'] += 1
            if self.state['first_fire_t'] is None:
                self.state['first_fire_t'] = pre['t']
        if not after.same_as_now():
            raise GateRuntimeError('MIL scoring changed the RNG state')
        if not self.state['off_identity_verified']:
            before.restore()
            off, _, noise_off = self._pass(lambda: self.sampler.sample(self.head, _cloned(backbone_output), action_input,
                                                                         decision_env_step=0, gate=0.)[0])
            if not torch.equal(off, base_action) or noise_off != noise0 or not after.same_as_now():
                raise GateRuntimeError('gate-off composed path is not identical to native (action/noise/RNG)')
            self.state['off_identity_verified'] = True
        fire = self.mode == 'always' or (self.mode == 'gated' and pre['fired'])
        result, post, delta, noise_c, qam_steps = base, pre, 0.0, None, []
        if fire:
            before.restore()
            self.gate.restore(snap)
            (actions, info), hidden_c, noise_c = self._pass(lambda: self.sampler.sample(
                self.head, _cloned(backbone_output), action_input, decision_env_step=pre['t'] * 5, gate=1.))
            if noise_c != noise0:
                raise GateRuntimeError('composed pass initial noise differs from the first pass')
            if not after.same_as_now():
                raise GateRuntimeError('composed pass left a different RNG state than the vanilla pass')
            post = self.gate.step(self.gate.feature_from_hidden(hidden_c, 0))   # bookkeeping only; t advances once
            qam_steps = list(info['qam_steps'])
            if qam_steps != [0, 1, 2, 3]:
                raise GateRuntimeError(f'QAM called at flow steps {qam_steps}')
            result = type(backbone_output)(data={'action_pred': actions})
            delta = float((actions[..., :12] - base_action[..., :12]).float().abs().max().item())
            self.state['composed_calls'] += 1
        self.state['requests'] += 1
        finite = bool(torch.isfinite(result['action_pred']).all())
        if not finite:
            raise GateRuntimeError('non-finite action')
        rec = {'time': time.time(), 'request': self.state['requests'], 'mode': self.mode, 'mil': pre, 'post_mil': post,
               'gate': bool(fire), 'qam_steps': qam_steps, 'qam_delta_max_normalized': delta,
               'noise_sha256_first_pass': noise0, 'noise_sha256_composed': noise_c, 'rng_restored_equal': True,
               'first_fire_t': self.state['first_fire_t'], 'latency_ms': (time.perf_counter() - started) * 1000,
               'action_finite': finite, 'off_identity_verified': self.state['off_identity_verified']}
        if self.emit is not None:
            self.emit(rec)
        return result
