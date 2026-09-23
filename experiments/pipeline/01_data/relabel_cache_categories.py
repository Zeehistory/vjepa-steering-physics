#!/usr/bin/env python
"""Relabel a Physics-IQ latent cache with the official scenario->category mapping (no re-extraction).

The extractor stored ``category`` from the on-disk folder name, which for Physics-IQ is the *frame rate*
(16FPS/24FPS), not the physics category. This rewrites only ``metadata.parquet``'s ``category`` column
using the official mapping derived from each sample id's scenario (see data.physics_iq_categories).
``LatentDataset`` reads category from ``metadata.parquet``, so this is all that's needed — the latent
shards are untouched and no GPU re-extraction is required. A timestamped backup of the parquet is kept.

Example
-------
    python experiments/pipeline/01_data/relabel_cache_categories.py \
        --latent_dir /path/to/outputs/latents/physics_iq/vjepa2_large
"""

from __future__ import annotations

# --- repo-root shim: make ``src`` importable however this script is invoked ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

import argparse
import collections
import importlib.util
import shutil
from pathlib import Path

# Load the (pure-python, torch-free) category mapping directly by file path. Importing it via the
# `src.data` package would pull in torch and get the process killed on memory-capped login nodes.
_CAT_PATH = _REPO_ROOT / "src" / "data" / "physics_iq_categories.py"
_spec = importlib.util.spec_from_file_location("physics_iq_categories", _CAT_PATH)
_piq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_piq)  # type: ignore[union-attr]
category_for_id, scenario_for_id = _piq.category_for_id, _piq.scenario_for_id


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--dry_run", action="store_true", help="report the new distribution without writing")
    args = p.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    meta_path = Path(args.latent_dir) / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.parquet under {args.latent_dir}")
    records = pq.read_table(meta_path).to_pylist()

    new_counts: collections.Counter = collections.Counter()
    unmapped = []
    for r in records:
        cat = category_for_id(r["id"])
        if cat is None:
            unmapped.append((r["id"], scenario_for_id(r["id"])))
            cat = "misc"
        r["category"] = cat
        new_counts[cat] += 1

    print("[relabel] new category distribution:")
    for k, v in sorted(new_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {v:4d}  {k}")
    if unmapped:
        print(f"[relabel] WARNING: {len(unmapped)} samples had no official category (set to 'misc'); "
              f"e.g. {unmapped[:5]}")

    if args.dry_run:
        print("[relabel] dry run — metadata.parquet not modified.")
        return

    backup = meta_path.with_suffix(".parquet.bak")
    if not backup.exists():
        shutil.copy2(meta_path, backup)
        print(f"[relabel] backup -> {backup}")
    pq.write_table(pa.Table.from_pylist(records), meta_path)
    print(f"[relabel] rewrote {meta_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
