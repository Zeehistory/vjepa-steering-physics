

# --- repo-root shim: make ``src`` importable however this script is invoked ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

#!/usr/bin/env python
"""Phase-0 gate: does fp16 latent / uint8 frame storage change any downstream number?

The L/H/G sweep needs a ~5x cache-size reduction to fit on scratch, and half of that comes from
storing latents as float16 and target frames as uint8. That is only safe if it leaves the
quantities we actually report untouched. This measures that on REAL cached ViT-L latents rather
than assuming it.

Gates (from the sweep plan):
  fp16 latents  -- top-8 subspace principal angles < 1 deg, and |dR2| < 0.005 for a ridge probe
  uint8 frames  -- bit-exact round trip (frames originate as uint8; if the dataset rescaled them
                   this fails and the sweep falls back to float16 frames, a 2x saving not 4x)

Exit code 0 = all gates pass.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from src.encoders.feature_extractor import LatentDataset  # noqa: E402


def principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def pca_basis(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def ridge_r2(X: np.ndarray, y: np.ndarray, lam: float = 10.0, n_train: int | None = None) -> float:
    """Held-out R^2 of a ridge probe, fit on the first half and scored on the second."""
    n_train = n_train or len(X) // 2
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean(0)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ (ytr - ym))
    pred = Xte @ W + ym
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True, help="an existing float32 latent cache")
    ap.add_argument("--layer", type=int, default=None, help="layer index (default: deepest available)")
    ap.add_argument("--n", type=int, default=256, help="clips to load")
    ap.add_argument("--k", type=int, default=8, help="subspace rank for the principal-angle test")
    ap.add_argument("--max_angle_deg", type=float, default=1.0)
    ap.add_argument("--max_dr2", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = LatentDataset(args.latent_dir, layers="all")
    layer = args.layer if args.layer is not None else max(ds.available_layers())
    n = min(args.n, len(ds))
    print(f"[gate_dtype] {args.latent_dir}\n  {len(ds)} clips available, using {n}, layer {layer}")

    pooled32, pooled16, states = [], [], []
    frames_exact = True
    frames_checked = 0
    frames_max_abs_err = 0.0
    for i in range(n):
        s = ds[i]
        t = s["layers"][layer]                      # (tokens, dim) float32 as stored
        pooled32.append(t.mean(0).numpy())
        # Exactly what the sweep would write, then read back.
        pooled16.append(t.to(torch.float16).to(torch.float32).mean(0).numpy())
        states.append(s["state"].numpy())

        f = s["frames"]
        if f.numel() and frames_checked < 32:
            rt = f.mul(255.0).round().clamp_(0, 255).to(torch.uint8).to(torch.float32).div_(255.0)
            err = (rt - f).abs().max().item()
            frames_max_abs_err = max(frames_max_abs_err, err)
            frames_exact &= bool(torch.equal(rt, f))
            frames_checked += 1

    X32 = np.stack(pooled32).astype(np.float64)
    X16 = np.stack(pooled16).astype(np.float64)
    S = np.stack(states)                            # (n, T, state_dim)

    # --- gate 1: subspace stability -------------------------------------------------------
    ang = principal_angles_deg(pca_basis(X32, args.k), pca_basis(X16, args.k))
    max_ang = float(ang.max())

    # --- gate 2: probe R^2 stability ------------------------------------------------------
    # Regress the per-clip mean state (the physics target) off pooled tokens.
    y = np.nan_to_num(S.mean(1)).astype(np.float64)
    keep = y.std(0) > 1e-6                          # drop constant columns (e.g. unused state slots)
    y = y[:, keep]
    r2_32 = ridge_r2(X32, y)
    r2_16 = ridge_r2(X16, y)
    d_r2 = abs(r2_32 - r2_16)

    rel_err = float(np.abs(X16 - X32).max() / (np.abs(X32).max() + 1e-12))

    pass_fp16 = max_ang < args.max_angle_deg and d_r2 < args.max_dr2
    pass_u8 = frames_exact if frames_checked else None

    res = {
        "latent_dir": args.latent_dir, "layer": layer, "n_clips": n, "k": args.k,
        "fp16": {
            "principal_angles_deg": [round(a, 6) for a in ang.tolist()],
            "max_principal_angle_deg": max_ang,
            "ridge_r2_fp32": r2_32, "ridge_r2_fp16": r2_16, "abs_delta_r2": d_r2,
            "max_rel_elementwise_err": rel_err,
            "thresholds": {"max_angle_deg": args.max_angle_deg, "max_dr2": args.max_dr2},
            "pass": bool(pass_fp16),
        },
        "uint8_frames": {
            "clips_checked": frames_checked, "bit_exact": frames_exact,
            "max_abs_err": frames_max_abs_err, "pass": pass_u8,
        },
        "pass": bool(pass_fp16 and (pass_u8 is not False)),
    }

    print(f"\n  fp16 max principal angle : {max_ang:.6f} deg  (limit {args.max_angle_deg})")
    print(f"  fp16 max rel elem err    : {rel_err:.3e}")
    print(f"  ridge R2 fp32/fp16       : {r2_32:.6f} / {r2_16:.6f}   |d| = {d_r2:.2e} (limit {args.max_dr2})")
    print(f"  uint8 frames bit-exact   : {frames_exact} (max abs err {frames_max_abs_err:.3e}, "
          f"{frames_checked} clips)")
    print(f"\n  FP16 GATE : {'PASS' if pass_fp16 else 'FAIL'}")
    print(f"  UINT8 GATE: {'PASS' if pass_u8 else ('FAIL' if pass_u8 is False else 'N/A (no frames)')}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"  wrote {args.out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
