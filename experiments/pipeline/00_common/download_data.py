#!/usr/bin/env python
"""Dataset download / preparation helper.

* ``synthetic_physics`` — nothing to download; generated on the fly (this prints a note).
* ``physics_iq`` — prints the official source and expected on-disk layout. Automated download is
  intentionally not bundled (respect the dataset's license/terms); follow the printed instructions and
  point ``data.root`` at the result.
* ``droid`` — Stage-2; prints a pointer.
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


PHYSICS_IQ_INFO = """
Physics-IQ benchmark
  Project:  https://physics-iq.github.io/
  Code:     https://github.com/google-deepmind/physics-IQ-benchmark
Download the videos per the project's instructions, then arrange as either:
  <root>/manifest.json                 # list of {"path","category","split"} records, OR
  <root>/<split>/<category>/*.mp4      # directory tree
Then set `data.root: <root>` in configs/data/physics_iq.yaml.
"""

DROID_INFO = """
DROID robotics dataset (Stage-2 extension)
  Project: https://droid-dataset.github.io/
Support is a typed stub for now (src/data/droid.py). See docs/method.md roadmap.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["synthetic_physics", "physics_iq", "droid"])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.dataset == "synthetic_physics":
        print("[download_data] synthetic_physics is generated on the fly — nothing to download.")
        print("Generate a preview with notebooks/01_dataset_preview.ipynb or the smoke pipeline.")
    elif args.dataset == "physics_iq":
        print(PHYSICS_IQ_INFO)
    else:
        print(DROID_INFO)


if __name__ == "__main__":
    main()
