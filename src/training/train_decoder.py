"""Decoder training loop (native PyTorch + Accelerate).

Trains a decoder on **cached frozen latents** (so the encoder is never rerun). Supports AMP,
gradient accumulation, gradient checkpointing, cosine/linear/constant LR with warmup, EMA, robust
checkpoint/resume, and full run provenance (resolved config + git commit + hardware).

Entry point: :func:`train_decoder(cfg)` returns a summary dict including the last/best checkpoint path.
The CLI wrapper is ``experiments/pipeline/03_train/train_decoder.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..decoders import DecoderLoss, build_decoder
from ..encoders.feature_extractor import LatentDataset, latent_collate
from ..utils.config import save_config
from ..utils.reproducibility import set_seed, write_run_metadata
from .checkpoints import EMA, load_checkpoint, save_checkpoint
from .distributed import build_accelerator
from .logging import JSONLLogger, setup_logging
from .schedulers import build_scheduler


def _infer_dims(dataset: LatentDataset) -> tuple[int, int]:
    rec = dataset.records[0]
    return int(rec["hidden_dim"]), int(rec["state_dim"])


def train_decoder(cfg: DictConfig) -> dict[str, Any]:
    logger = setup_logging()
    set_seed(cfg.train.seed, cfg.train.deterministic)
    if not cfg.latent_dir:
        raise ValueError("cfg.latent_dir must point at an extracted latent cache.")

    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bound the in-RAM shard cache. LatentDataset defaults to unbounded, which is fine for the small
    # splits but OOM-kills a full training split: a 4000-clip / 4-layer cache is ~172 GB and the
    # trainer touches every shard. `train.max_cached_shards` caps it (LRU); the default of 16 holds
    # ~half a fp16 split, and a miss only costs re-reading one shard tar.
    dataset = LatentDataset(cfg.latent_dir, layers=cfg.encoder.layers,
                            max_cached_shards=int(getattr(cfg.train, "max_cached_shards", 16)))
    enc_dim, state_dim = _infer_dims(dataset)

    # Fill decoder dims from the data / encoder so configs stay terse.
    cfg.decoder.state_dim = state_dim
    if "out_num_frames" not in cfg.decoder or cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames

    accelerator = build_accelerator(cfg.train.mixed_precision, cfg.optim.grad_accum)
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim)
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(dataset.available_layers())
    if accelerator.is_main_process:
        logger.info(f"decoder={cfg.decoder.name} params={decoder.num_parameters()/1e6:.2f}M "
                    f"enc_dim={enc_dim} state_dim={state_dim} mode={cfg.decoder.mode}")

    loss_fn = DecoderLoss(cfg.loss)
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=latent_collate, drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )
    scheduler = build_scheduler(
        optimizer, cfg.optim.scheduler, cfg.optim.warmup_steps, cfg.optim.max_steps
    )
    decoder, optimizer, loader, scheduler = accelerator.prepare(decoder, optimizer, loader, scheduler)

    ema = EMA(accelerator.unwrap_model(decoder), cfg.optim.ema_decay) if cfg.optim.ema_decay > 0 else None
    start_step = 0
    if cfg.train.resume:
        start_step = load_checkpoint(cfg.train.resume, accelerator.unwrap_model(decoder), optimizer,
                                     scheduler, ema)
        logger.info(f"resumed from {cfg.train.resume} @ step {start_step}")
    elif getattr(cfg.train, "init_from", None):
        # Weights only: fresh optimizer, fresh schedule, step 0. Deliberately NOT load_checkpoint --
        # that restores the stored step and would end the run before it starts.
        _ck = torch.load(cfg.train.init_from, map_location="cpu", weights_only=False)
        _missing, _unexpected = accelerator.unwrap_model(decoder).load_state_dict(_ck["model"],
                                                                                 strict=True)
        if ema is not None:
            ema.shadow.load_state_dict(accelerator.unwrap_model(decoder).state_dict())
        logger.info(f"init_from {cfg.train.init_from} (weights only, step 0; "
                    f"source step {int(_ck.get('step', -1))})")

    if accelerator.is_main_process:
        save_config(cfg, out_dir / "resolved_config.yaml")
        write_run_metadata(out_dir, {"experiment": "train_decoder", "decoder_params": accelerator.unwrap_model(decoder).num_parameters()})
    jsonl = JSONLLogger(out_dir / "metrics.jsonl") if accelerator.is_main_process else None

    decoder.train()
    step = start_step
    data_iter = _cycle(loader)
    while step < cfg.optim.max_steps:
        batch = next(data_iter)
        with accelerator.accumulate(decoder):
            grid = tuple(int(x) for x in batch["grid"])
            latents = {int(k): v for k, v in batch["layers"].items()}
            out = decoder(latents, grid)
            loss, logs = loss_fn(
                pred_frames=out.frames,
                target_frames=batch["frames"] if out.frames is not None else None,
                pred_state=out.state,
                target_state=batch["state"] if out.state is not None else None,
                state_mask=batch["state_mask"] if out.state is not None else None,
                state_keys=batch["state_keys"] if out.state is not None else None,
            )
            accelerator.backward(loss)
            if accelerator.sync_gradients and cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(decoder.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            if ema is not None:
                ema.update(accelerator.unwrap_model(decoder))
            step += 1
            if step % cfg.train.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                logger.info(f"step {step}/{cfg.optim.max_steps} loss={logs['total']:.4f} lr={lr:.2e}")
                if jsonl:
                    jsonl.log({"step": step, "lr": lr, **logs})
            if step % cfg.train.ckpt_every == 0 and accelerator.is_main_process:
                save_checkpoint(ckpt_dir / f"step_{step}.pt", accelerator.unwrap_model(decoder),
                                optimizer, scheduler, ema, step)

    last_ckpt = ckpt_dir / "last.pt"
    if accelerator.is_main_process:
        save_checkpoint(last_ckpt, accelerator.unwrap_model(decoder), optimizer, scheduler, ema, step)
        if jsonl:
            jsonl.close()
    accelerator.wait_for_everyone()
    return {"checkpoint": str(last_ckpt), "steps": step, "decoder_params": accelerator.unwrap_model(decoder).num_parameters()}


def _cycle(loader: DataLoader):
    while True:
        yield from loader
