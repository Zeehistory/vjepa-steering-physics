"""Dataset registry, sample schema, and collation."""

from __future__ import annotations

import torch

from src.data import build_dataset
from src.data.dataset_registry import collate


def test_synthetic_dataset_schema(tiny_cfg):
    ds = build_dataset(tiny_cfg.data, encoder_image_size=tiny_cfg.encoder.image_size,
                       encoder_frames=tiny_cfg.encoder.num_frames)
    assert len(ds) == tiny_cfg.data.num_clips
    s = ds[0]
    for key in ("id", "frames", "encoder_input", "state", "state_mask", "state_keys", "category"):
        assert key in s
    assert s["frames"].shape[0] == tiny_cfg.data.num_frames
    assert s["encoder_input"].shape[-1] == tiny_cfg.encoder.image_size
    assert s["state"].shape[-1] == len(s["state_keys"])


def test_state_padding_consistent_across_scenarios(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.data.scenarios = ["bouncing_ball", "collision"]  # 1- and 2-object scenarios
    ds = build_dataset(cfg.data, encoder_image_size=cfg.encoder.image_size,
                       encoder_frames=cfg.encoder.num_frames)
    dims = {ds[i]["state"].shape[-1] for i in range(len(ds))}
    assert len(dims) == 1  # padded to a common width


def test_collate(tiny_cfg):
    ds = build_dataset(tiny_cfg.data, encoder_image_size=tiny_cfg.encoder.image_size,
                       encoder_frames=tiny_cfg.encoder.num_frames)
    batch = collate([ds[0], ds[1]])
    assert batch["frames"].shape[0] == 2
    assert isinstance(batch["id"], list) and len(batch["id"]) == 2
    assert torch.is_tensor(batch["state"])


def test_unknown_dataset_raises(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.data.name = "does_not_exist"
    try:
        build_dataset(cfg.data)
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_latent_dataset_shard_cache_is_bounded():
    """``max_cached_shards`` evicts least-recently-used shards; the default stays unbounded.

    Regression for an OOM: the cache held every decoded shard, so a pass that streamed a whole split
    accumulated all of them (~136 GB on the 4096-clip x 4-layer spin split). Exercises the real
    ``_load_shard`` caching path with the tar read stubbed out, so it fails if the eviction is removed.
    """
    from src.encoders.feature_extractor import LatentDataset

    class _Stub(LatentDataset):
        def __init__(self, max_cached_shards):
            self._shard_cache = {}
            self.max_cached_shards = max_cached_shards
            self.layers = "all"
            self.reads = []

        def _decode_shard(self, shard):
            self.reads.append(shard)
            return {f"{shard}_sample": {"id": f"{shard}_sample"}}

    bounded = _Stub(2)
    for i in range(8):
        bounded._load_shard(f"s{i}")
    assert len(bounded._shard_cache) == 2
    assert set(bounded._shard_cache) == {"s6", "s7"}
    assert bounded.reads == [f"s{i}" for i in range(8)]

    # A cached shard must be served from RAM, and must be kept alive by that access (LRU, not FIFO).
    bounded._load_shard("s6")
    bounded._load_shard("s8")
    assert set(bounded._shard_cache) == {"s6", "s8"}, "recently used shard must survive eviction"
    assert bounded.reads.count("s6") == 1, "cache hit must not re-read the shard"

    unbounded = _Stub(None)
    for i in range(8):
        unbounded._load_shard(f"s{i}")
    assert len(unbounded._shard_cache) == 8, "default must stay unbounded for existing consumers"
