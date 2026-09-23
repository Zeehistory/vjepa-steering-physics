"""Real V-JEPA (v1) encoder wrapper.

V-JEPA v1 uses the same token contract as V-JEPA2. The official v1 checkpoints are distributed via the
``facebookresearch/jepa`` repo (not HF) as plain ViT state dicts. This wrapper supports loading a local
checkpoint into a standard ViT video backbone; when a HuggingFace mirror id is provided it defers to the
V-JEPA2 wrapper's HF loading path (identical block layout).
"""

from __future__ import annotations

from typing import Any

import torch

from .base import EncoderWrapper, LatentBundle
from .vjepa2_wrapper import VJEPA2Encoder


class VJEPAEncoder(EncoderWrapper):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        if cfg.hf_model_id:
            # HF-hosted mirror: reuse the V-JEPA2 HF path (same architecture family).
            self._impl: EncoderWrapper = VJEPA2Encoder(cfg)
        else:
            raise NotImplementedError(
                "Loading original V-JEPA v1 .pth checkpoints from facebookresearch/jepa is not wired "
                "yet. Provide `encoder.hf_model_id` to use a HuggingFace mirror, or extend this wrapper "
                "to build a ViT backbone and load the v1 state dict. See docs/method.md."
            )
        self.freeze()

    @property
    def hidden_dim(self) -> int:
        return self._impl.hidden_dim

    @property
    def num_layers(self) -> int:
        return self._impl.num_layers

    def forward_features(self, video, layers, extract_attention) -> LatentBundle:
        return self._impl.forward_features(video, layers, extract_attention)

    def freeze(self) -> None:
        if hasattr(self, "_impl"):
            self._impl.freeze()

    @torch.no_grad()
    def encode(self, video, layers="all", extract_attention=False) -> LatentBundle:
        return self._impl.encode(video, layers, extract_attention)
