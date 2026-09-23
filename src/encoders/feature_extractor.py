"""Latent extraction, caching, and reading.

Runs a frozen encoder over a dataset and writes latents to disk so decoder/probe training never reruns
the encoder. Storage format (dependency-light, WebDataset-compatible layout):

    <out>/
      shard_00000.tar         # tar of per-sample `<id>.pt` tensors (torch.save)
      shard_00001.tar
      metadata.parquet        # one row per sample: id, shard, category, grid, layers, state info
      checksums.json          # sha256 per shard (integrity / provenance)
      extract_meta.json       # encoder id, config, commit, preprocessing, layer indices

Each per-sample `.pt` is a dict:
    {"id", "layers": {idx: (L, D) tensor}, "grid": (T', Hp, Wp), "state": (T, S), "state_mask": (S,),
     "state_keys": [...], "category": str}

This is read back by :class:`LatentDataset` for training and by the analysis modules.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.dataset_registry import collate
from .base import EncoderWrapper


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LatentShardWriter:
    """Accumulates per-sample tensors and flushes them into ``.tar`` shards."""

    def __init__(self, out_dir: str | Path, shard_size: int = 64) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._buf: list[tuple[str, dict[str, Any]]] = []
        self._shard_idx = 0
        self.records: list[dict[str, Any]] = []
        self.checksums: dict[str, str] = {}

    def add(self, sample_id: str, payload: dict[str, Any], record: dict[str, Any]) -> None:
        self._buf.append((sample_id, payload))
        record["shard"] = f"shard_{self._shard_idx:05d}.tar"
        self.records.append(record)
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        shard_path = self.out_dir / f"shard_{self._shard_idx:05d}.tar"
        with tarfile.open(shard_path, "w") as tar:
            for sample_id, payload in self._buf:
                buf = io.BytesIO()
                torch.save(payload, buf)
                data = buf.getvalue()
                info = tarfile.TarInfo(name=f"{sample_id}.pt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        self.checksums[shard_path.name] = _sha256(shard_path)
        self._buf.clear()
        self._shard_idx += 1

    def close(self, extract_meta: dict[str, Any]) -> None:
        self._flush()
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([_jsonify(r) for r in self.records])
        pq.write_table(table, self.out_dir / "metadata.parquet")
        (self.out_dir / "checksums.json").write_text(json.dumps(self.checksums, indent=2))
        (self.out_dir / "extract_meta.json").write_text(json.dumps(extract_meta, indent=2))


def _jsonify(rec: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in rec.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


_LATENT_DTYPES = {"float32": torch.float32, "float16": torch.float16}
_FRAME_DTYPES = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8}
_EMPTY_FRAMES = torch.zeros(0)


def _cast_frames(frames: torch.Tensor, frames_dtype: str) -> torch.Tensor:
    """Cast [0,1] float frames for storage. ``uint8`` round-trips losslessly for frames that
    originated as uint8; ``experiments/threads/model-size-sweep/06_probes/gate_dtype.py`` checks that before the sweep relies on it."""
    if frames_dtype == "uint8":
        return frames.mul(255.0).round().clamp_(0, 255).to(torch.uint8)
    return frames.to(_FRAME_DTYPES[frames_dtype])


def _restore_frames(frames: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_cast_frames`: always hand consumers [0,1] float32."""
    if frames.dtype == torch.uint8:
        return frames.to(torch.float32).div_(255.0)
    return frames.to(torch.float32)


@torch.no_grad()
def extract_latents(
    encoder: EncoderWrapper,
    dataset: Dataset,
    out_dir: str | Path,
    layers: Any = "all",
    extract_attention: bool = False,
    batch_size: int = 4,
    num_workers: int = 0,
    device: torch.device | str = "cpu",
    shard_size: int = 64,
    store_frames_size: int | None = None,
    extract_meta: dict[str, Any] | None = None,
    latent_dtype: str = "float32",
    frames_dtype: str = "float32",
    store_frames: bool = True,
) -> Path:
    """Run ``encoder`` over ``dataset`` and write latent shards to ``out_dir``. Returns ``out_dir``.

    The original (un-normalized) frames are stored alongside the latents so the latent cache is
    self-contained for reconstruction training. ``store_frames_size`` optionally downsamples the stored
    target frames to bound cache size; ``None`` keeps the dataset's native resolution.

    Cache-size levers, added for the multi-encoder sweep where a 4-layer float32 cache runs to ~207 GB
    per dataset at ViT-L (and ~1.4x that at ViT-g):

    ``latent_dtype``  ``"float32"`` or ``"float16"``. Halves the latent payload.
    ``frames_dtype``  ``"float32"``, ``"float16"``, or ``"uint8"``. Frames arrive as [0,1] floats
                      decoded from uint8 sources, so ``"uint8"`` is a lossless 4x saving -- but that
                      holds only if the dataset has not rescaled them, which
                      ``experiments/threads/model-size-sweep/06_probes/gate_dtype.py`` verifies before the sweep relies on it.
    ``store_frames``  ``False`` skips frames entirely. Splits used only to fit operators/probes never
                      read frames, so this drops ~27% of their footprint.

    :class:`LatentDataset` restores float32 on load, so every downstream consumer is unaffected.
    """
    import torch.nn.functional as F
    if latent_dtype not in _LATENT_DTYPES:
        raise ValueError(f"latent_dtype must be one of {sorted(_LATENT_DTYPES)}, got {latent_dtype!r}")
    if frames_dtype not in _FRAME_DTYPES:
        raise ValueError(f"frames_dtype must be one of {sorted(_FRAME_DTYPES)}, got {frames_dtype!r}")
    encoder = encoder.to(device)
    encoder.freeze()
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate, shuffle=False
    )
    writer = LatentShardWriter(out_dir, shard_size=shard_size)
    layer_indices: list[int] | None = None

    for batch in tqdm(loader, desc="extract latents"):
        video = batch["encoder_input"].to(device)
        bundle = encoder.encode(video, layers=layers, extract_attention=extract_attention)
        layer_indices = bundle.layer_indices
        b = video.shape[0]
        frames = batch["frames"]
        if store_frames_size is not None:
            t = frames.shape[1]
            frames = F.interpolate(
                frames.flatten(0, 1), size=(store_frames_size, store_frames_size),
                mode="bilinear", align_corners=False,
            ).reshape(frames.shape[0], t, frames.shape[2], store_frames_size, store_frames_size)
        frames = _cast_frames(frames, frames_dtype)
        lat_dtype = _LATENT_DTYPES[latent_dtype]
        for i in range(b):
            sample_layers = {
                li: bundle.layers[li][i].to(lat_dtype).cpu().clone() for li in bundle.layer_indices
            }
            payload = {
                "id": batch["id"][i],
                "layers": sample_layers,
                "frames": frames[i].cpu().clone() if store_frames else _EMPTY_FRAMES,
                "frames_dtype": frames_dtype if store_frames else None,
                "grid": (bundle.grid.temporal, bundle.grid.height, bundle.grid.width),
                "state": batch["state"][i].cpu().clone(),
                "state_mask": batch["state_mask"][i].cpu().clone(),
                "state_keys": batch["state_keys"],
                "category": batch["category"][i],
            }
            record = {
                "id": batch["id"][i],
                "category": batch["category"][i],
                "temporal": bundle.grid.temporal,
                "height": bundle.grid.height,
                "width": bundle.grid.width,
                "hidden_dim": bundle.hidden_dim,
                "layers": list(bundle.layer_indices),
                "state_dim": int(batch["state"][i].shape[-1]),
                "state_valid": bool(batch["state_mask"][i].sum() > 0),
                "has_frames": bool(store_frames),
            }
            writer.add(batch["id"][i], payload, record)

    meta = {
        "encoder": getattr(encoder, "model_id", encoder.__class__.__name__),
        "hidden_dim": encoder.hidden_dim,
        "num_layers": encoder.num_layers,
        "layer_indices": layer_indices,
        "extract_attention": extract_attention,
        "latent_dtype": latent_dtype,
        "frames_dtype": frames_dtype if store_frames else None,
        "store_frames": bool(store_frames),
    }
    if extract_meta:
        meta.update(extract_meta)
    writer.close(meta)
    return Path(out_dir)


class LatentDataset(Dataset):
    """Reads latent shards written by :func:`extract_latents`.

    Parameters
    ----------
    root: directory containing ``metadata.parquet`` and ``shard_*.tar``.
    layers: which layer indices to return (``"all"`` or a list). Selecting fewer layers saves memory
        for single-layer decoder/probe experiments.
    max_cached_shards: cap on how many decoded shards are held in RAM at once (least-recently-used
        evicted). ``None`` (the default) keeps the historical unbounded behaviour, so no existing
        consumer changes. Set it for any pass that streams a WHOLE split: the cache grows without
        bound otherwise, and on a 4096-clip / 4-layer split that is ~136 GB, which OOM-killed the
        operator fit. A cap of 2 is enough to keep each shard read exactly once when samples are
        visited in index order (a shard holds 128 clips = 8 consecutive scenes).
    """

    def __init__(self, root: str | Path, layers: Any = "all",
                 max_cached_shards: int | None = None) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        meta_path = self.root / "metadata.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.parquet under {self.root}; run extract_latents first.")
        self.records = pq.read_table(meta_path).to_pylist()
        self.layers = layers
        # Map sample id -> (shard, member) and cache opened shards lazily.
        self._index = {r["id"]: r["shard"] for r in self.records}
        self._ids = [r["id"] for r in self.records]
        # Category source of truth is metadata.parquet, so relabeling (e.g. fixing Physics-IQ
        # categories) only rewrites the small parquet and not every shard.
        self._category = {r["id"]: r.get("category") for r in self.records}
        self._shard_cache: dict[str, dict[str, Any]] = {}
        self.max_cached_shards = max_cached_shards

    def __len__(self) -> int:
        return len(self._ids)

    def available_layers(self) -> list[int]:
        return list(self.records[0]["layers"])

    def _load_shard(self, shard: str) -> dict[str, Any]:
        if shard in self._shard_cache:
            # Refresh recency so the LRU eviction below keeps the shard currently being streamed.
            if self.max_cached_shards is not None:
                self._shard_cache[shard] = self._shard_cache.pop(shard)
            return self._shard_cache[shard]
        samples = self._decode_shard(shard)
        self._shard_cache[shard] = samples
        if self.max_cached_shards is not None:
            while len(self._shard_cache) > self.max_cached_shards:
                self._shard_cache.pop(next(iter(self._shard_cache)))   # dicts are insertion-ordered
        return samples

    def _decode_shard(self, shard: str) -> dict[str, Any]:
        """Read and deserialize one shard tar. Split out from :meth:`_load_shard` so the caching
        policy above can be tested without a real latent cache on disk.

        When only a subset of layers is requested, the others are dropped before caching so the in-RAM
        shard cache holds e.g. 4/24 layers instead of the full payload. Without this, a consumer that
        streams every sample (decoder DataLoader, steering) accumulates all shards' full multi-layer
        tensors and OOMs (each shard is ~30GB on disk).
        """
        want = None
        if self.layers != "all" and self.layers is not None:
            want = {int(x) for x in self.layers}
        samples: dict[str, Any] = {}
        with tarfile.open(self.root / shard, "r") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                assert f is not None
                payload = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
                if want is not None:
                    payload["layers"] = {li: t for li, t in payload["layers"].items() if li in want}
                samples[payload["id"]] = payload
        return samples

    def _select_layers(self, payload: dict[str, Any]) -> dict[int, torch.Tensor]:
        if self.layers == "all" or self.layers is None:
            return payload["layers"]
        want = [int(x) for x in self.layers]
        return {li: payload["layers"][li] for li in want}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self._ids[idx]
        payload = self._load_shard(self._index[sid])[sid]
        layers = self._select_layers(payload)
        # Upcast on read so fp16/uint8 caches are invisible to every downstream consumer.
        return {
            "id": sid,
            "layers": {li: t.to(torch.float32) for li, t in layers.items()},
            "frames": _restore_frames(payload["frames"]),
            "grid": payload["grid"],
            "state": payload["state"],
            "state_mask": payload["state_mask"],
            "state_keys": payload["state_keys"],
            "category": self._category.get(sid) or payload["category"],
        }


def latent_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate latent samples: stack per-layer token tensors and state."""
    layer_keys = sorted(batch[0]["layers"].keys())
    out: dict[str, Any] = {
        "id": [b["id"] for b in batch],
        "category": [b["category"] for b in batch],
        "grid": batch[0]["grid"],
        "state_keys": batch[0]["state_keys"],
        "layers": {li: torch.stack([b["layers"][li] for b in batch], 0) for li in layer_keys},
        "frames": torch.stack([b["frames"] for b in batch], 0),
        "state": torch.stack([b["state"] for b in batch], 0),
        "state_mask": torch.stack([b["state_mask"] for b in batch], 0),
    }
    return out
