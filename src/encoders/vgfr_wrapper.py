"""VGFR-style video encoder wrapper (deferred stub).

VGFR (Video Generative Frame Representations / video-foundation-representation models) is a placeholder
for an alternative frozen video backbone family. The wrapper is a typed stub documenting the contract;
because it returns the same :class:`LatentBundle`, the rest of the pipeline needs no change once a
concrete VGFR checkpoint is wired in.
"""

from __future__ import annotations

from typing import Any

from .base import EncoderWrapper, LatentBundle


class VGFREncoder(EncoderWrapper):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "VGFR wrapper is a stub. Implement weight loading and `forward_features` returning a "
            "LatentBundle (see encoders/base.py and vjepa2_wrapper.py for the contract)."
        )

    @property
    def hidden_dim(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    @property
    def num_layers(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def forward_features(self, video, layers, extract_attention) -> LatentBundle:  # pragma: no cover
        raise NotImplementedError
