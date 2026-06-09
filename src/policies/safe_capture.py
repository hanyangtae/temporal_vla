"""Shared SAFE forward-hook capture lifecycle."""

from __future__ import annotations

from typing import Any, Callable

import torch


class SafeForwardCapture:
    """Register a temporary forward hook and collect feature tensors."""

    def __init__(
        self,
        module: torch.nn.Module,
        mode: str,
        slicer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        to_cpu: bool = False,
        dtype: torch.dtype | None = None,
    ):
        if mode not in {"pre", "post"}:
            raise ValueError(f"Unsupported hook mode: {mode}")
        self.module = module
        self.mode = mode
        self.slicer = slicer
        self.to_cpu = to_cpu
        self.dtype = dtype
        self.buf: list[torch.Tensor] = []
        self._handle: Any | None = None

    def _capture(self, tensor: torch.Tensor) -> None:
        captured = tensor
        if self.slicer is not None:
            captured = self.slicer(captured)
        captured = captured.detach()
        if self.to_cpu or self.dtype is not None:
            kwargs: dict[str, Any] = {}
            if self.to_cpu:
                kwargs["device"] = "cpu"
            if self.dtype is not None:
                kwargs["dtype"] = self.dtype
            captured = captured.to(**kwargs)
        self.buf.append(captured)

    def _pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        if args:
            self._capture(args[0])

    def _post_hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        self._capture(out)

    def __enter__(self) -> "SafeForwardCapture":
        self.buf = []
        if self.mode == "pre":
            self._handle = self.module.register_forward_pre_hook(self._pre_hook)
        else:
            self._handle = self.module.register_forward_hook(self._post_hook)
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
