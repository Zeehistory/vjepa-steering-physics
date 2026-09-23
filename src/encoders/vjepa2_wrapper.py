"""Real V-JEPA 2 encoder wrapper (HuggingFace ``transformers``).

Loads ``facebook/vjepa2-*`` and exposes the same :class:`LatentBundle` contract as the mock encoder, so
swapping to real weights is a config change only. ``transformers`` is an optional dependency
(``pip install -e .[encoders]``) and is imported lazily so the offline mock pipeline never requires it.

The encoder is **frozen by default** (the whole research question is about what *already* exists in the
representation). Run on CUDA for non-trivial clip sizes.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import EncoderWrapper, LatentBundle, TokenGrid
from .layer_hooks import ActivationCapture


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Real V-JEPA2 weights need the `encoders` extra: pip install -e .[encoders]"
        ) from e
    return transformers


class VJEPA2Encoder(EncoderWrapper):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        transformers = _require_transformers()
        model_id = cfg.hf_model_id or "facebook/vjepa2-vitl-fpc64-256"
        # AutoModel returns the backbone; VJEPA2Model exposes hidden states + a blocks list.
        self.model = transformers.AutoModel.from_pretrained(model_id)
        self.model_id = model_id
        self._blocks = self._find_blocks(self.model)
        self._dim = int(getattr(self.model.config, "hidden_size", cfg.hidden_dim))
        self._num_layers = len(self._blocks)
        self.patch = int(getattr(self.model.config, "patch_size", cfg.patch_size))
        self.tubelet = int(getattr(self.model.config, "tubelet_size", cfg.tubelet_size))
        self.image_size = int(getattr(self.model.config, "image_size", cfg.image_size))
        self.freeze()

    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
        """Locate the ordered list of transformer blocks across HF V-JEPA2 layouts."""
        candidates = [
            "encoder.layer", "encoder.blocks", "blocks", "layers",
            "vjepa2.encoder.layer", "backbone.blocks",
        ]
        for path in candidates:
            obj: Any = model
            ok = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (list, torch.nn.ModuleList)):
                return list(obj)
        raise RuntimeError(
            "Could not locate transformer blocks on the V-JEPA2 model; "
            "inspect `model` and extend `_find_blocks` candidate paths."
        )

    def _hook_modules(self) -> list[torch.nn.Module]:
        """The modules to tap, one per block, per ``cfg.hook_site`` (see EncoderConfig.hook_site)."""
        site = str(getattr(self.cfg, "hook_site", "block") or "block")
        if site == "block":
            return self._blocks
        mods = []
        for b in self._blocks:
            m: Any = b
            for attr in site.split("."):
                if not hasattr(m, attr):
                    raise RuntimeError(
                        f"hook_site='{site}' not found on this block; missing '{attr}'. "
                        f"Available: {[n for n, _ in b.named_children()]}"
                    )
                m = getattr(m, attr)
            mods.append(m)
        return mods

    @property
    def hidden_dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def _infer_grid(self, num_tokens: int, t: int, h: int, w: int) -> TokenGrid:
        tp = (t // self.tubelet) or 1
        hp = h // self.patch
        wp = w // self.patch
        if tp * hp * wp != num_tokens:
            # Fall back to a square-ish factorization if the model adds/removes tokens.
            hp = wp = int(round((num_tokens / tp) ** 0.5))
        return TokenGrid(temporal=tp, height=hp, width=wp)

    def forward_features(
        self, video: torch.Tensor, layers: list[int], extract_attention: bool
    ) -> LatentBundle:
        b, t, c, h, w = video.shape
        # HF V-JEPA2 expects pixel_values_videos as (B, T, C, H, W).
        with ActivationCapture(self._hook_modules(), layers) as cap:
            outputs = self.model(
                pixel_values_videos=video,
                output_attentions=extract_attention,
                output_hidden_states=False,
            )
        acts = cap.activations
        if not acts:
            raise RuntimeError("No activations captured; check `_find_blocks` for this checkpoint.")
        num_tokens = next(iter(acts.values())).shape[1]
        grid = self._infer_grid(num_tokens, t, h, w)

        attn: dict[int, torch.Tensor] = {}
        if extract_attention and getattr(outputs, "attentions", None) is not None:
            for i in layers:
                if i < len(outputs.attentions):
                    attn[i] = outputs.attentions[i].detach()

        return LatentBundle(
            layers=acts,
            grid=grid,
            hidden_dim=self._dim,
            pooled={i: a.mean(dim=1) for i, a in acts.items()},
            attention=attn,
            meta={"encoder": "vjepa2", "model_id": self.model_id, "frozen": self.frozen},
        )
