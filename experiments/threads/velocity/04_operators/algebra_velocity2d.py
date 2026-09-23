

#!/usr/bin/env python
"""Do latent velocity interventions obey the algebra of physical transformations?

Four tests, each scored BOTH in the latent (cheap, but a latent can move without the physics moving)
and by decoding and independently pixel-tracking the result (the one that counts):

    identity        T_0(H)               ~ H
    inverse         T_{-dv}(T_{dv}(H))   ~ H
    composition     T_{dv2}(T_{dv1}(H))  ~ T_{dv1+dv2}(H)
    commutativity   T_x(T_y(H))          ~ T_y(T_x(H))

The operator is command-only: T is built from (v_a, v_b) through the 13-dimensional feature map and
the fitted low-rank write, never from the target latent.  Note this makes the tests non-trivial --
the feature map is nonlinear in the commands, so none of the four identities holds by construction.

Composition is applied SEQUENTIALLY: the second command is issued from the velocity the first one
asked for, which is what a planner would do.  The reference is the single direct command to the same
final velocity.

Baselines, so "small error" means something:
    noop        the unedited latent, i.e. how far apart two clips are anyway
    random      two independent norm-matched random edits, to check the decoder is sensitive
                at the magnitude the algebra violations live at

Usage:
    python experiments/threads/velocity/04_operators/algebra_velocity2d.py \
        --config configs/train/rolling_ball3d_decoder.yaml \
        --test_dir  .../rolling_ball3d/test/vjepa2_large \
        --artifacts_dir .../analysis/rolling_ball3d/subspace \
        --checkpoint .../runs/rolling_ball3d_decoder_fp/checkpoints/last.pt \
        --output_dir .../analysis/rolling_ball3d/algebra --num_scenes 40
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
import json
from pathlib import Path

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    return {int(L): sample["layers"][L].unsqueeze(0).to(device) for L in layers}


def _add(H, edit, device):
    """H (dict L->(1,Ltok,D) torch) + per-layer flat edit (numpy) -> new latent dict."""
    out = {}
    for L, t in H.items():
        e = torch.from_numpy(edit[L].reshape(t.shape[1], t.shape[2]).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


def _decode_vel(decoder, latents, grid):
    out = decoder(latents, grid)
    if out.frames is None:
        return np.array([np.nan, np.nan])
    m = measured_velocity(out.frames[0].cpu())
    return np.array([m["vel_x"], m["vel_y"]])


def _latent_diff(A, B):
    """Relative L2 distance between two latent dicts, pooled over layers."""
    num = sum(float(((A[L] - B[L]) ** 2).sum()) for L in A)
    den = sum(float((B[L] ** 2).sum()) for L in A)
    return float(np.sqrt(num / max(den, 1e-30)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_scenes", type=int, default=40)
    p.add_argument("--gain", type=float, default=2.0,
                   help="command gain; use the value selected on validation for this dataset")
    p.add_argument("--cmd_ku", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers, max_cached_shards=2)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]
    print(f"[algebra] {len(scenes)} test scenes, using {len(sids)}; layers={layers}; "
          f"gain={args.gain}")

    tag = "" if args.cmd_ku == 8 else f"_ku{args.cmd_ku}"
    Wu = {L: np.load(art / f"cmd_Wu{tag}_L{L}.npy").astype(np.float64) for L in layers}
    U = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}

    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), cfg.decoder.state_dim)
    decoder = decoder.to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    def T(va, vb):
        """Command-only edit taking a latent from velocity va to velocity vb."""
        phi = vo.command_features(va, vb)
        return {L: args.gain * ((phi @ Wu[L]) @ U[L][: Wu[L].shape[1]]) for L in layers}

    rng = np.random.default_rng(0)
    rows = []
    for n, sid in enumerate(sids):
        ranks = sorted(scenes[sid])
        sa = ds[scenes[sid][ranks[0]]]
        grid = tuple(int(x) for x in sa["grid"])
        va = vo.clip_velocity(sa)
        Ha = _to_dev(sa, layers, device)

        # two commands, taken from this scene's own swept velocities so they are in-distribution
        vb = vo.clip_velocity(ds[scenes[sid][ranks[len(ranks) // 2]]])
        vc = vo.clip_velocity(ds[scenes[sid][ranks[-1]]])

        with torch.no_grad():
            m_a = _decode_vel(decoder, Ha, grid)

            # ---- identity ------------------------------------------------------
            H_id = _add(Ha, T(va, va), device)
            m_id = _decode_vel(decoder, H_id, grid)

            # ---- inverse -------------------------------------------------------
            H_fwd = _add(Ha, T(va, vb), device)
            m_fwd = _decode_vel(decoder, H_fwd, grid)
            H_back = _add(H_fwd, T(vb, va), device)
            m_back = _decode_vel(decoder, H_back, grid)

            # ---- composition ---------------------------------------------------
            H_seq = _add(H_fwd, T(vb, vc), device)          # a -> b -> c
            m_seq = _decode_vel(decoder, H_seq, grid)
            H_dir = _add(Ha, T(va, vc), device)             # a -> c
            m_dir = _decode_vel(decoder, H_dir, grid)

            # ---- commutativity --------------------------------------------------
            # split the a->c command into an x-only and a y-only leg, both orders
            v_x = np.array([vc[0], va[1]])
            v_y = np.array([va[0], vc[1]])
            H_xy = _add(_add(Ha, T(va, v_x), device), T(v_x, vc), device)
            H_yx = _add(_add(Ha, T(va, v_y), device), T(v_y, vc), device)
            m_xy = _decode_vel(decoder, H_xy, grid)
            m_yx = _decode_vel(decoder, H_yx, grid)

            # ---- norm-matched random control ------------------------------------
            # NOT "compose two random edits": r1 then r2 is identically Ha + (r1 + r2), so that
            # comparison is a tautology that returns exactly zero and proves nothing.
            # The control that actually matters asks whether the decoder is even sensitive at the
            # scale of the algebra violations: decode two INDEPENDENT norm-matched random edits and
            # measure how far apart their decoded velocities land. If the algebra errors are no
            # bigger than this, the identities "hold" only because nothing at this magnitude moves
            # the decoded physics, and the test is vacuous.
            nrm = {L: float(np.linalg.norm(T(va, vc)[L])) for L in layers}
            def _rand():
                r = {L: rng.normal(size=U[L].shape[1]) for L in layers}
                return {L: r[L] * nrm[L] / np.linalg.norm(r[L]) for L in layers}
            H_r1 = _add(Ha, _rand(), device)
            H_r2 = _add(Ha, _rand(), device)
            m_r_seq = _decode_vel(decoder, H_r1, grid)
            m_r_dir = _decode_vel(decoder, H_r2, grid)

        rows.append({
            "scene": int(sid),
            "v_a": va.tolist(), "v_b": vb.tolist(), "v_c": vc.tolist(),
            "decoded": {
                "noop": m_a.tolist(), "identity": m_id.tolist(),
                "forward": m_fwd.tolist(), "inverse_return": m_back.tolist(),
                "sequential": m_seq.tolist(), "direct": m_dir.tolist(),
                "xy_order": m_xy.tolist(), "yx_order": m_yx.tolist(),
                "random_edit_a": m_r_seq.tolist(), "random_edit_b": m_r_dir.tolist(),
            },
            "latent_rel": {
                "identity_vs_noop": _latent_diff(H_id, Ha),
                "inverse_vs_noop": _latent_diff(H_back, Ha),
                "sequential_vs_direct": _latent_diff(H_seq, H_dir),
                "xy_vs_yx": _latent_diff(H_xy, H_yx),
                "random_a_vs_b": _latent_diff(H_r1, H_r2),
            },
        })
        if (n + 1) % 5 == 0:
            print(f"  ...{n + 1}/{len(sids)}")

    summary = {
        "test_dir": args.test_dir, "checkpoint": args.checkpoint, "artifacts_dir": str(art),
        "layers": layers, "gain": args.gain, "cmd_ku": args.cmd_ku,
        "n_scenes": len(rows),
        "tests": {
            "identity": "T_0(H) vs H",
            "inverse": "T_{b->a}(T_{a->b}(H)) vs H",
            "composition": "T_{b->c}(T_{a->b}(H)) vs T_{a->c}(H)  [sequential vs direct]",
            "commutativity": "x-leg then y-leg vs y-leg then x-leg, same endpoint",
        },
        "note": ("Scored on decoded, pixel-tracked velocity as well as latent distance. None of the "
                 "four identities holds by construction: the operator is linear in a nonlinear "
                 "13-dim feature map of (v_a, v_b), not in dv."),
        "per_scene": rows,
    }
    (out / "algebra_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[algebra] -> {out}/algebra_summary.json  ({len(rows)} scenes)")


if __name__ == "__main__":
    main()
