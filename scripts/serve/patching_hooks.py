"""Donor-trajectory transplant hook (patchceil).

실패 episode 를 같은 (scenario_seed, inference_seed) 로 재실행하면서, 지정 record 창에서
GR00T N1.5 DiT hidden 을 donor(같은 cell 성공 episode) 저장 activation 으로 **통째 대입**
하는 forward hook. 설계 원안 `docs/steering/25a_patchceil_transplant_handoff.md`,
Gate 1 재설계 `docs/collab/2026-07-16-patching-transplant-gate1.md` (명칭: transplant probe,
"상한" 아님 — 구성적 하한).

hook 배선·denoise 카운터 규약은 ConceptorSteering(steering_hooks.py)을 따르되:
- 변환(h' = h·Mᵀ)이 아니라 **대입**(h' = donor[record, k]).
- rollout **record cursor** 상태가 추가된다 (아래).

상태 모델 (fail-loud):
- ``_k``: 요청 내 denoise call index. serve 가 요청 진입마다 ``reset_step_counter()``
  호출 → ``_k = 0`` 리셋 **및 ``_record_idx += 1``** (요청 1개 = record 1개 규약 —
  patch serve 는 /act_with_features(skip_features) 의 predict_action_chunk 경로만 허용,
  /act 큐-팝은 serve 쪽에서 409).
- ``_record_idx``: rollout record cursor (첫 요청에서 0). 에피소드 경계는
  ``reset_episode()`` (serve /reset 연동). **arm 상태는 /reset 에도 유지** — 러너가
  `/patch_arm` → collector 기동(내부에서 policy.reset() → /reset) 순서로 쓰기 때문.
- ``arm(...)``: donor 배열·창 파라미터·tag 를 원자적으로 설정 + 에피소드 카운터 리셋.
- ``fired``: 실제 패치가 적용된 (record, k) 로그. 러너가 rollout 종료 후
  `/patch_status` 로 기대 창과 대조해 불일치 시 해당 rollout 폐기 (무음 오적용 방지).

donor 고갈 규약 (Gate 1 §채택4): donor record 를 넘어서는 창은 **패치하지 않고**
``exhausted_at`` 에 기록만 한다 — freeze/loop/합성 금지. "donor horizon 내 성공" 채점은
러너 책임.

donor NPZ 규약 (pass B 추출기가 생산):
- key ``L{layer}``: float16 [R, K, T, D] (record, denoise step, token, hidden).
- key ``meta_json``: json bytes — cell, episode_idx, scenario_seed, inference_seed,
  feature_kind, n_records. 로더가 K·layer 존재를 대조 assert.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOKEN_SELECTS: tuple[str, ...] = ("all", "action")


def load_donor_npz(
    path: str | Path,
    layers: list[int] | tuple[int, ...],
    *,
    expected_k: int,
) -> tuple[dict[int, np.ndarray], dict, str]:
    """donor NPZ 에서 요청 layer 들의 [R,K,T,D] 배열을 로드.

    Returns:
        (arrays: {layer: [R,K,T,D] np}, meta: dict, sha12: npz 파일 sha256 앞 12자리)
    """
    p = Path(path)
    sha12 = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8")) if "meta_json" in z else {}
        arrays: dict[int, np.ndarray] = {}
        for layer in layers:
            key = f"L{int(layer)}"
            if key not in z:
                raise ValueError(f"donor npz {p} 에 {key} 없음 (keys={sorted(z.keys())})")
            arr = z[key]
            if arr.ndim != 4:
                raise ValueError(f"{key}: [R,K,T,D] 4차원 기대, got {arr.shape}")
            if arr.shape[1] != expected_k:
                raise ValueError(
                    f"{key}: K={arr.shape[1]} != 모델 num_inference_timesteps={expected_k}"
                )
            arrays[int(layer)] = arr
    r_set = {a.shape[0] for a in arrays.values()}
    if len(r_set) != 1:
        raise ValueError(f"layer 간 record 수 불일치: {r_set}")
    return arrays, meta, sha12


def load_vl_donor_npz(path: str | Path) -> tuple[np.ndarray, dict, str]:
    """VL donor NPZ 로드 (exp4-2 B1 — extract_vl_donor_npz.py 가 생산).

    규약: key ``VL`` = fp16 [R, T_vl, D] (record, VL token, hidden), key ``meta_json``.
    DiT donor 와 달리 K 축이 없다 (vl_self_attention 은 chunk 추론당 1회 호출).

    Returns:
        (arr: [R,T_vl,D] np, meta: dict, sha12: npz sha256 앞 12자리)
    """
    p = Path(path)
    sha12 = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8")) if "meta_json" in z else {}
        if "VL" not in z:
            raise ValueError(f"VL donor npz {p} 에 'VL' 키 없음 (keys={sorted(z.keys())})")
        arr = z["VL"]
    if arr.ndim != 3:
        raise ValueError(f"VL donor: [R,T_vl,D] 3차원 기대, got {arr.shape}")
    return arr, meta, sha12


class PatchSteering:
    """단일 주입 지점(DiT block residual 또는 DiT 최종)의 transplant hook.

    Args:
        groot_model: ``Gr00tN1d6`` (``.action_head.model``).
        layer: transformer_block 인덱스 (0..15). ``None`` 이면 DiT 최종 출력(D=1024)
            — v1 미사용 예약 (donor 캡처 kind 가 다름).
        expected_k: 모델 num_inference_timesteps (fire 수 검증).
        token_select: ``"all"`` (전 토큰 대입, 기본 — donor T == 출력 T 필수) |
            ``"action"`` (마지막 horizon 개 토큰만 — donor 의 같은 슬라이스 사용).
        horizon: token_select="action" 일 때 대입할 마지막 토큰 수 (기본
            action_head.action_horizon).
    """

    def __init__(
        self,
        groot_model: Any,
        *,
        layer: int | None,
        expected_k: int,
        token_select: str = "all",
        horizon: int | None = None,
    ):
        head = groot_model.action_head
        self.layer = layer
        if layer is None:
            self.module = head.model
        else:
            self.module = head.model.transformer_blocks[int(layer)]
        if token_select not in TOKEN_SELECTS:
            raise ValueError(f"Unsupported token_select: {token_select} (expected {TOKEN_SELECTS})")
        self.token_select = token_select
        self.horizon = int(horizon if horizon is not None else head.action_horizon)
        self.expected_k = int(expected_k)
        self._handle = None
        # --- arm 상태 (에피소드 간 유지, /patch_arm 으로만 변경) ---
        self._donor: np.ndarray | None = None  # [R,K,T,D]
        self._donor_gpu: torch.Tensor | None = None
        self._start_record = 0
        self._donor_start = 0
        self._patch_len = -1  # -1 = donor 고갈까지
        self._tag: str | None = None
        # --- 에피소드 상태 (reset_episode 로 초기화) ---
        self._record_idx = -1
        self._k = 0
        self._fired: list[tuple[int, int]] = []
        self._exhausted_at: list[int] = []

    # -- 상태 전이 ---------------------------------------------------------------
    def arm(
        self,
        donor: np.ndarray,
        *,
        start_record: int,
        donor_start: int = 0,
        patch_len: int = -1,
        tag: str | None = None,
    ) -> None:
        """donor 배열과 창 파라미터를 원자적으로 설정 (+에피소드 카운터 리셋)."""
        donor = np.asarray(donor)
        if donor.ndim != 4 or donor.shape[1] != self.expected_k:
            raise ValueError(
                f"donor [R,K,T,D] (K={self.expected_k}) 기대, got {donor.shape}"
            )
        if start_record < 0 or donor_start < 0 or donor_start >= donor.shape[0]:
            raise ValueError(
                f"창 파라미터 위반: start_record={start_record} donor_start={donor_start} "
                f"R={donor.shape[0]}"
            )
        self._donor = donor
        self._donor_gpu = None  # 첫 fire 에서 device/dtype 확정 후 통째 캐시
        self._start_record = int(start_record)
        self._donor_start = int(donor_start)
        self._patch_len = int(patch_len)
        self._tag = tag
        self.reset_episode()

    def disarm(self) -> None:
        self._donor = None
        self._donor_gpu = None
        self._tag = None
        self.reset_episode()

    def reset_episode(self) -> None:
        """에피소드 경계 (serve /reset). arm 상태는 유지, 카운터·로그만 초기화."""
        self._record_idx = -1
        self._k = 0
        self._fired = []
        self._exhausted_at = []

    def reset_step_counter(self) -> None:
        """요청(get_action) 진입마다 serve 가 호출 — k 리셋 + record cursor 전진."""
        self._k = 0
        self._record_idx += 1

    @property
    def armed(self) -> bool:
        return self._donor is not None

    def status(self) -> dict:
        fired_records = sorted({r for r, _ in self._fired})
        return {
            "layer": self.layer,
            "armed": self.armed,
            "tag": self._tag,
            "token_select": self.token_select,
            "start_record": self._start_record,
            "donor_start": self._donor_start,
            "patch_len": self._patch_len,
            "donor_records": None if self._donor is None else int(self._donor.shape[0]),
            "record_idx": self._record_idx,
            "fired_total": len(self._fired),
            "fired_records": fired_records,
            "exhausted_at": sorted(set(self._exhausted_at)),
        }

    # -- hook --------------------------------------------------------------------
    def _hook(self, _module: Any, _args: tuple, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        k = self._k
        if k >= self.expected_k:
            raise RuntimeError(
                f"patch hook over-fire: denoise call {k} >= K={self.expected_k} "
                "(요청 시작 reset_step_counter 누락 또는 K 불일치)"
            )
        self._k += 1
        if self._donor is None:
            return output
        off = self._record_idx - self._start_record
        if off < 0:
            return output
        if self._patch_len >= 0 and off >= self._patch_len:
            return output
        d_rec = self._donor_start + off
        if d_rec >= self._donor.shape[0]:
            # donor horizon 종료 — 합성/freeze 금지 (Gate 1 §채택4), 기록만.
            self._exhausted_at.append(self._record_idx)
            return output
        donor_gpu = self._donor_gpu
        if donor_gpu is None or donor_gpu.device != out.device or donor_gpu.dtype != out.dtype:
            donor_gpu = torch.as_tensor(self._donor, device=out.device, dtype=out.dtype)
            self._donor_gpu = donor_gpu
        donor_kt = donor_gpu[d_rec, k]  # [T, D]
        patched = out.clone()
        if self.token_select == "all":
            if donor_kt.shape[0] != out.shape[-2] or donor_kt.shape[1] != out.shape[-1]:
                raise RuntimeError(
                    f"donor [T,D]={tuple(donor_kt.shape)} != 출력 {tuple(out.shape[-2:])} "
                    "(token_select=all 은 full-token donor 필수)"
                )
            patched[..., :, :] = donor_kt  # batch 축 broadcast
        else:  # "action" — 마지막 horizon 토큰만
            h = self.horizon
            if donor_kt.shape[0] < h or donor_kt.shape[1] != out.shape[-1]:
                raise RuntimeError(
                    f"donor [T,D]={tuple(donor_kt.shape)} 로 마지막 {h}토큰 대입 불가 "
                    f"(출력 D={out.shape[-1]})"
                )
            patched[..., -h:, :] = donor_kt[-h:, :]
        self._fired.append((self._record_idx, k))
        if is_tuple:
            return (patched, *output[1:])
        return patched

    # -- 등록 (ConceptorSteering 과 동일 규약) --------------------------------------
    def register(self) -> "PatchSteering":
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "PatchSteering":
        return self.register()

    def __exit__(self, *_exc: Any) -> bool:
        self.unregister()
        return False


class PatchSteeringVL:
    """VL pathway transplant hook (exp4-2 B1) — ``action_head.vl_self_attention`` 출력 교체.

    DiT 판(PatchSteering)과의 차이:
    - ``vl_self_attention`` 은 ``process_backbone_output()`` 에서 **chunk 추론당 1회**
      호출된다 (denoise 루프 밖) → ``expected_fires = 1`` (요청당 1 fire, over-fire 는
      fail-loud). serve 의 fire-count 검증은 ``expected_fires`` 를 우선 사용.
    - donor 는 [R, T_vl, D] (K 축 없음). **T_vl 상이 허용** — 출력이 DiT cross-attention
      의 encoder_hidden_states 라 임의 길이가 동작하므로, clone-슬라이스 대입이 아니라
      donor 텐서로 **통째 교체**한다. D 불일치는 RuntimeError.
    - B==1 전용 (수집 serve). batch>1 이면 fail-loud.

    record cursor·arm/disarm/reset 규약은 PatchSteering 과 동일 (serve 배선 재사용):
    ``reset_step_counter()`` = k 리셋 + record 전진, ``reset_episode()`` = /reset 연동,
    arm 상태는 /reset 에도 유지. 창 밖/donor 고갈 → 원 출력 그대로 (closed-loop 복귀).
    """

    expected_fires = 1

    def __init__(self, groot_model: Any):
        self.module = groot_model.action_head.vl_self_attention
        self.layer = "VL"
        self.token_select = "all"
        self._handle = None
        # --- arm 상태 (에피소드 간 유지) ---
        self._donor: np.ndarray | None = None  # [R, T_vl, D]
        self._donor_gpu: torch.Tensor | None = None
        self._start_record = 0
        self._donor_start = 0
        self._patch_len = -1
        self._tag: str | None = None
        # --- 에피소드 상태 ---
        self._record_idx = -1
        self._k = 0
        self._fired: list[tuple[int, int]] = []
        self._exhausted_at: list[int] = []

    # -- 상태 전이 (PatchSteering 미러) -------------------------------------------
    def arm(
        self,
        donor: np.ndarray,
        *,
        start_record: int,
        donor_start: int = 0,
        patch_len: int = -1,
        tag: str | None = None,
    ) -> None:
        donor = np.asarray(donor)
        if donor.ndim != 3:
            raise ValueError(f"VL donor [R,T_vl,D] 기대, got {donor.shape}")
        if start_record < 0 or donor_start < 0 or donor_start >= donor.shape[0]:
            raise ValueError(
                f"창 파라미터 위반: start_record={start_record} donor_start={donor_start} "
                f"R={donor.shape[0]}"
            )
        self._donor = donor
        self._donor_gpu = None
        self._start_record = int(start_record)
        self._donor_start = int(donor_start)
        self._patch_len = int(patch_len)
        self._tag = tag
        self.reset_episode()

    def disarm(self) -> None:
        self._donor = None
        self._donor_gpu = None
        self._tag = None
        self.reset_episode()

    def reset_episode(self) -> None:
        self._record_idx = -1
        self._k = 0
        self._fired = []
        self._exhausted_at = []

    def reset_step_counter(self) -> None:
        self._k = 0
        self._record_idx += 1

    @property
    def armed(self) -> bool:
        return self._donor is not None

    def status(self) -> dict:
        fired_records = sorted({r for r, _ in self._fired})
        return {
            "layer": self.layer,
            "pathway": "vl",
            "armed": self.armed,
            "tag": self._tag,
            "token_select": self.token_select,
            "start_record": self._start_record,
            "donor_start": self._donor_start,
            "patch_len": self._patch_len,
            "donor_records": None if self._donor is None else int(self._donor.shape[0]),
            "donor_t_vl": None if self._donor is None else int(self._donor.shape[1]),
            "record_idx": self._record_idx,
            "fired_total": len(self._fired),
            "fired_records": fired_records,
            "exhausted_at": sorted(set(self._exhausted_at)),
        }

    # -- hook --------------------------------------------------------------------
    def _hook(self, _module: Any, _args: tuple, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if self._k >= 1:
            raise RuntimeError(
                "VL patch hook over-fire: vl_self_attention 이 요청당 2회 이상 호출됨 "
                "(reset_step_counter 누락 또는 모델 경로 변경)"
            )
        self._k += 1
        if self._donor is None:
            return output
        off = self._record_idx - self._start_record
        if off < 0:
            return output
        if self._patch_len >= 0 and off >= self._patch_len:
            return output
        d_rec = self._donor_start + off
        if d_rec >= self._donor.shape[0]:
            self._exhausted_at.append(self._record_idx)
            return output
        if out.shape[0] != 1:
            raise RuntimeError(f"VL patch 는 B==1 수집 serve 전용 (got B={out.shape[0]})")
        if self._donor.shape[2] != out.shape[-1]:
            raise RuntimeError(
                f"VL donor D={self._donor.shape[2]} != 출력 D={out.shape[-1]}"
            )
        donor_gpu = self._donor_gpu
        if donor_gpu is None or donor_gpu.device != out.device or donor_gpu.dtype != out.dtype:
            donor_gpu = torch.as_tensor(self._donor, device=out.device, dtype=out.dtype)
            self._donor_gpu = donor_gpu
        # T_vl 상이 허용 — clone-슬라이스가 아니라 donor 텐서로 통째 교체.
        patched = donor_gpu[d_rec].unsqueeze(0)  # [1, T_vl_donor, D]
        self._fired.append((self._record_idx, 0))
        if is_tuple:
            return (patched, *output[1:])
        return patched

    # -- 등록 ----------------------------------------------------------------------
    def register(self) -> "PatchSteeringVL":
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "PatchSteeringVL":
        return self.register()

    def __exit__(self, *_exc: Any) -> bool:
        self.unregister()
        return False
