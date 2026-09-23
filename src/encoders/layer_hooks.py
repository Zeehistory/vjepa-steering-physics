"""Forward-hook utilities for extracting intermediate activations from real encoders.

Used by the HuggingFace V-JEPA wrappers to capture per-block token outputs (and, when requested,
attention maps) without modifying the upstream model code. The wrapper registers hooks on the
transformer blocks, runs a forward pass, then reads the captured tensors.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


class ActivationCapture:
    """Register forward hooks on a list of modules and collect their outputs.

    Parameters
    ----------
    modules:
        Ordered iterable of submodules (e.g. transformer blocks) to capture, index = position.
    layers:
        Which indices to actually store.
    """

    def __init__(self, modules: list[nn.Module], layers: Iterable[int]) -> None:
        self.modules = modules
        self.layers = set(int(x) for x in layers)
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> ActivationCapture:
        for i, m in enumerate(self.modules):
            if i in self.layers:
                self._handles.append(m.register_forward_hook(self._make_hook(i)))
        return self

    def _make_hook(self, idx: int):
        def hook(_module, _inp, output):
            # HF blocks may return a tuple (hidden_states, ...) or a tensor.
            hs = output[0] if isinstance(output, tuple) else output
            self.activations[idx] = hs.detach()

        return hook

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
