"""Thin, optional Weights & Biases wrapper.

WandB is an optional dependency. If it is not installed or not enabled in the config, all calls become
no-ops so training never depends on it.
"""

from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(self, enabled: bool, project: str = "vjepa-physics-decoder", config: dict | None = None,
                 name: str | None = None, tags: list[str] | None = None) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(project=project, config=config, name=name, tags=tags)
        except Exception:  # pragma: no cover - optional path
            self.enabled = False
            self._run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._wandb.finish()
