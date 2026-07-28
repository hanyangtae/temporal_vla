"""SAE 학습 루프 — 미니배치 + val 손실 early stopping.

출처: task_classification@88543a2 `phase/train/_loop.py` (+ 엔트리 `phase/train/posthoc.py`
      의 학습 호출 관례). https://github.com/robots-oh/task_classification
이식 근거: docs/steering/29_sae_port_review.md §2.3, §5.A / §5.D(규모·미니배치)
이식 계획: docs/steering/30_sae_g1_port_handout.md §4 Phase A2

[원본 대비 변경]
- **hydra/omegaconf 제거**: `make_optimizer(params, cfg)`가 `cfg.name`/`cfg.lr` 속성 접근이었으나
  평범한 인자(name/lr/weight_decay)로 바꿨다. `fit`도 `cfg_train.epochs` 대신 명시 인자.
- **wandb 제거**: `wandb_init()`·`run.log()` 경로 삭제 → stdout `print`만 남긴다.
- **미니배치 필수**: 원본은 batch_size=None 이면 full-batch (데이터가 GPU 상주 전제).
  우리 G1 입력은 per-token 으로 펼친 수십만 행 × 1536-d 라 full-batch 불가 →
  `epoch()`가 batch_size 를 **요구**하고, 데이터는 CPU 에 두고 배치만 device 로 옮긴다.
- **seed 인자 추가**: 배치 순열까지 재현 가능하도록 torch/numpy seed 고정 + 전용 generator.
- **device 인자 추가**: GPU/CPU 어느 쪽이든 같은 코드로 (승준 원격 CPU / 로컬 GPU 1장).
"""
from __future__ import annotations

import random

import numpy as np
import torch

OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}


def set_seed(seed: int) -> None:
    """python/numpy/torch 전역 seed 고정 (재현 가능한 fit)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(params, name="adam", lr=1e-3, weight_decay=0.0):
    """옵티마이저 생성.

    기본값은 동료 SAE 설정(`conf/train/sae.yaml`)의 Adam·lr 1e-3·weight_decay 0.
    wd=0 인 이유 = 사전 수축이 `DecoderLinearDict` 의 단위노름 전제와 충돌하기 때문.
    """
    key = str(name).lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"optimizer name은 {tuple(OPTIMIZERS)} 중 하나여야 합니다: {name!r}")
    return OPTIMIZERS[key](params, lr=lr, weight_decay=weight_decay)


def epoch(model, tensors, opt=None, batch_size=4096, grad_clip=None,
          device=None, generator=None):
    """한 epoch. opt가 None이면 평가만. `model.loss(*batch)`를 부른다.

    tensors는 행이 정렬된 [N, ...] 텐서 튜플 — 비지도면 (x,), 지도면 (x, y).

    [원본과 다른 점] batch_size 가 필수다(full-batch 금지, 모듈 docstring 참고).
    tensors 는 CPU 에 둬도 되고, 배치 단위로 `device` 로 옮긴다.

    [반환] 행 가중 평균 손실 float.
    """
    if not tensors:
        raise ValueError("tensors 가 비어 있습니다")
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError(f"batch_size 는 양의 정수여야 합니다 (full-batch 금지): {batch_size!r}")

    train = opt is not None
    model.train(train)
    n = len(tensors[0])
    if train:
        idx = torch.randperm(n, generator=generator)
    else:
        idx = torch.arange(n)

    total = 0.0
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        batch = tuple(t[j].to(device) if device is not None else t[j] for t in tensors)
        with torch.set_grad_enabled(train):
            loss = model.loss(*batch)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        total += loss.item() * len(j)
    return total / n


def fit(model, train_epoch, val_epoch, epochs=800, patience=60, min_epochs=60,
        log_every=100, verbose=True):
    """val 손실 최소화 early stopping. train_epoch/val_epoch는 float 손실을 돌려주는 콜백.

    [반환] best = {"loss", "epoch", "state", "history"}.
           종료 시 model에 best state를 적재하고 eval().
           (history = [(ep, train_loss, val_loss), ...] — 우리 추가, 수렴 진단·테스트용.)
    """
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    history = []
    bad = 0
    for ep in range(int(epochs)):
        tr_loss = train_epoch()
        va_loss = val_epoch()
        history.append((ep, tr_loss, va_loss))
        if va_loss < best["loss"]:
            best = {"loss": va_loss, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience and ep >= min_epochs:
                break
        if verbose and log_every and ep % log_every == 0:
            print(f"  ep{ep:4d} train={tr_loss:10.4f} val={va_loss:10.4f}", flush=True)

    if best["state"] is None:
        raise RuntimeError("epochs=0 — 학습이 한 번도 실행되지 않았습니다")
    model.load_state_dict(best["state"])
    model.eval()
    best["history"] = history
    if verbose:
        print(f"  best ep{best['epoch']} val_loss={best['loss']:.4f}", flush=True)
    return best


def train_sae(model, x_train, x_val, *, epochs=800, patience=60, min_epochs=60,
              batch_size=4096, lr=1e-3, weight_decay=0.0, optimizer="adam",
              grad_clip=5.0, device="cpu", seed=0, log_every=100, verbose=True):
    """미니배치 SAE 학습 엔트리 (`phase/train/posthoc.py` 의 조립을 dict 설정 없이 재현).

    [인자]
        x_train/x_val  [N, D] float32 배열/텐서. **episode 단위 split** 이어야 한다
                       (record/token split 은 누수 — 핸드아웃 §4 B3, §6 함정 2).
        device         "cpu" | "cuda:N". 데이터는 CPU 에 두고 배치만 옮긴다.
        seed           배치 순열·초기화 재현용.

    [반환] fit()의 best dict (+ "config" 요약).
    """
    set_seed(seed)
    dev = torch.device(device)
    model.to(dev)

    xt = torch.as_tensor(np.asarray(x_train), dtype=torch.float32)
    xv = torch.as_tensor(np.asarray(x_val), dtype=torch.float32)
    if xt.ndim != 2 or xv.ndim != 2:
        raise ValueError(f"x_train/x_val 은 [N, D] 여야 합니다: {tuple(xt.shape)}, {tuple(xv.shape)}")
    if xt.shape[1] != xv.shape[1]:
        raise ValueError(f"train/val feature 차원 불일치: {xt.shape[1]} vs {xv.shape[1]}")

    gen = torch.Generator().manual_seed(int(seed))
    opt = make_optimizer(model.parameters(), optimizer, lr=lr, weight_decay=weight_decay)

    def _train():
        return epoch(model, (xt,), opt=opt, batch_size=batch_size,
                     grad_clip=grad_clip, device=dev, generator=gen)

    @torch.no_grad()
    def _val():
        return epoch(model, (xv,), opt=None, batch_size=batch_size, device=dev)

    if verbose:
        print(f"[sae.train] N_train={len(xt)} N_val={len(xv)} D={xt.shape[1]} "
              f"bs={batch_size} device={dev} seed={seed}", flush=True)
    best = fit(model, _train, _val, epochs=epochs, patience=patience,
               min_epochs=min_epochs, log_every=log_every, verbose=verbose)
    best["config"] = {"epochs": epochs, "patience": patience, "min_epochs": min_epochs,
                      "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                      "optimizer": optimizer, "grad_clip": grad_clip,
                      "device": str(dev), "seed": seed}
    return best


@torch.no_grad()
def encode_all(model, x, batch_size=4096, device="cpu"):
    """[N, D] → 잠재 [N, d] numpy (배치 단위, 메모리 안전). probe/클러스터 입력용."""
    dev = torch.device(device)
    model.to(dev).eval()
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    out = []
    for i in range(0, len(xt), int(batch_size)):
        out.append(model.latent(xt[i:i + int(batch_size)].to(dev)).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 0), np.float32)
