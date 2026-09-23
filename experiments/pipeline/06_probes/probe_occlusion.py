

#!/usr/bin/env python
"""Step 2 (velocity-first): occlusion probe — is velocity decodable during hidden frames?

The key question: if the ball disappears behind a wall for the middle frames, does the V-JEPA2 latent
still encode the ball's velocity during those hidden frames? If yes, the representation carries a
*physical state* (velocity of a hidden object) rather than just immediate visual content.

Method
------
For each clip in the occlusion latent cache, we split the temporal token sequence into:
* ``visible`` frames  — ball is visible (``obj0_visible == 1``),
* ``hidden`` frames   — ball is behind the wall (``obj0_visible == 0``).

We then fit a linear probe to decode velocity from the **visible-frame** tokens and evaluate it on the
**hidden-frame** tokens. If R² stays high on hidden frames, the model carries velocity through
the occlusion. A model that only encodes immediate pixel information would collapse to chance on hidden
frames even though the velocity is constant.

The temporal position of each token is preserved — we use the encoder's ``T'=8`` temporal patch
positions and assign them to visible/hidden based on the nearest ground-truth frame's ``visible`` flag.
This correctly handles the case where the encoder's temporal resolution is coarser than the video FPS.

Outputs: ``occlusion_probe.csv``, ``occlusion_probe.png`` (R² on visible vs hidden per layer).

Example
-------
    python experiments/pipeline/06_probes/probe_occlusion.py \
        --latent_dir outputs/latents/moving_ball_occlusion/vjepa2_large \
        --output_dir outputs/analysis/moving_ball_occlusion/probe
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
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.encoders.feature_extractor import LatentDataset


def _temporal_tokens(tokens: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """``(L, D)`` flat tokens -> ``(T', D)`` spatially-pooled temporal sequence."""
    tp, hp, wp = grid
    n = tp * hp * wp
    return tokens[:n].reshape(tp, hp * wp, -1).mean(axis=1)  # (T', D)


def _assign_visibility(state: np.ndarray, state_keys: list[str], tp: int) -> np.ndarray:
    """Per-token-position visibility: assign each of the T' token positions to visible/hidden.

    The encoder produces ``T'`` temporal positions (typically 8) from ``T=32`` frames. We map each
    token position to its nearest ground-truth frame and read the ``visible`` flag there. Returns a
    boolean array of shape ``(T',)``.
    """
    t = state.shape[0]
    vis_col = state_keys.index("obj0_visible")
    vis_per_frame = state[:, vis_col].astype(bool)  # (T,)
    # token position t' maps to frame round(t' * (T-1) / (T'-1))
    token_frames = np.round(np.linspace(0, t - 1, tp)).astype(int)
    return vis_per_frame[token_frames]  # (T',)


def _build_dataset(dataset: LatentDataset, layer: int):
    """Build (features, velocity-target, vis_mask) triples per token position across all clips."""
    xs_vis, ys_vis, xs_hid, ys_hid = [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        keys = s["state_keys"]
        st = s["state"].numpy()
        grid = tuple(s["grid"])
        tp = grid[0]
        seq = _temporal_tokens(s["layers"][layer].numpy(), grid)  # (T', D)
        vis_mask = _assign_visibility(st, keys, tp)               # (T',) bool
        # velocity target: constant across the clip, so we can use any frame
        vx_col, vy_col = keys.index("obj0_vel_x"), keys.index("obj0_vel_y")
        vel = st[:, [vx_col, vy_col]].mean(0)                     # (2,) constant velocity
        xs_vis.extend(seq[vis_mask].tolist())
        ys_vis.extend([vel.tolist()] * int(vis_mask.sum()))
        if (~vis_mask).any():
            xs_hid.extend(seq[~vis_mask].tolist())
            ys_hid.extend([vel.tolist()] * int((~vis_mask).sum()))
    return (np.array(xs_vis, np.float32), np.array(ys_vis, np.float32),
            np.array(xs_hid, np.float32) if xs_hid else None,
            np.array(ys_hid, np.float32) if ys_hid else None)


def _fit_r2(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    if len(Xtr) < 4 or len(Xte) < 2:
        return float("nan")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    model = Ridge(alpha=1.0).fit(Xtr_s, Ytr)
    pred = model.predict(Xte_s)
    return float(r2_score(Yte, pred))


def run_occlusion_probe(
    latent_dir: str | Path,
    layers: list[int] | str = "all",
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    dataset = LatentDataset(latent_dir, layers="all")
    available = dataset.available_layers()
    layer_list = available if layers == "all" else [int(x) for x in layers]

    records = []
    for layer in layer_list:
        Xv, Yv, Xh, Yh = _build_dataset(dataset, layer)
        # Evict the per-shard latent cache between layers (holds full multi-layer token tensors for
        # every shard); without this the cache accumulates across layers and OOMs the job.
        dataset._shard_cache.clear()
        # train on visible, test on visible (80/20 split)
        n = len(Xv)
        split = int(0.8 * n)
        r2_vis = _fit_r2(Xv[:split], Yv[:split], Xv[split:], Yv[split:])
        # train on visible, evaluate on hidden — the key experiment
        r2_hid = _fit_r2(Xv, Yv, Xh, Yh) if Xh is not None else float("nan")
        # shuffled-latent control: train on shuffled visible, test on hidden
        rng = np.random.default_rng(0)
        r2_ctrl = _fit_r2(Xv[rng.permutation(len(Xv))], Yv, Xh, Yh) if Xh is not None else float("nan")
        records.append({
            "layer": layer,
            "r2_visible": round(r2_vis, 4),
            "r2_hidden": round(r2_hid, 4),
            "r2_ctrl_shuffled": round(r2_ctrl, 4),
            "n_visible_tokens": int(len(Xv)),
            "n_hidden_tokens": int(len(Xh)) if Xh is not None else 0,
        })
        print(f"  layer {layer:2d}: visible R²={r2_vis:.3f}  hidden R²={r2_hid:.3f}  "
              f"ctrl={r2_ctrl:.3f}  (n_vis={len(Xv)}, n_hid={len(Xh) if Xh is not None else 0})")

    if output_csv and records:
        p = Path(output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
    return records


def _plot(records: list[dict], path: Path) -> None:
    layers = [r["layer"] for r in records]
    vis = [r["r2_visible"] for r in records]
    hid = [r["r2_hidden"] for r in records]
    ctrl = [r["r2_ctrl_shuffled"] for r in records]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(layers, vis, "o-", color="steelblue", label="visible frames (test)")
    ax.plot(layers, hid, "s-", color="tomato", label="hidden frames (test, trained on visible)")
    ax.plot(layers, ctrl, ":", color="grey", lw=0.9, label="ctrl (shuffled latent)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("R²  (velocity)")
    ax.set_title("Occlusion probe: velocity decodable during hidden frames?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True,
                   help="latent cache for the moving_ball_occlusion dataset")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="all")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = "all" if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    print("[probe_occlusion] probing velocity decodability on visible vs hidden frames...")
    records = run_occlusion_probe(args.latent_dir, layers=layers,
                                  output_csv=out / "occlusion_probe.csv")
    _plot(records, out / "occlusion_probe.png")
    best = max(records, key=lambda r: r["r2_hidden"] if not np.isnan(r["r2_hidden"]) else -1)
    print(f"[probe_occlusion] best: layer {best['layer']}  "
          f"visible R²={best['r2_visible']}  hidden R²={best['r2_hidden']}  "
          f"ctrl={best['r2_ctrl_shuffled']} -> {out}")


if __name__ == "__main__":
    main()
