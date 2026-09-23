

#!/usr/bin/env python
"""Pixel-level proof for the ANGULAR-VELOCITY subspace / command operator (the rotational sibling of velocity).

The angular-velocity analog of ``experiments/threads/acceleration/05_steering/steer_accel2d.py``. Angular velocity is a SIGNED SCALAR ``omega``
(sign = CW/CCW, magnitude = spin rate), embedded as the 2-vector ``[omega, 0]`` so the accel subspace /
ridge / command machinery is reused unchanged (``velocity_ops.clip_angvel`` +
``fit_command_operators_accel.py --quantity angvel``). Loads the held-out TEST scene cache + the decoder
trained on angvel latents + the artifacts fit on TRAIN by ``accel_subspace.py`` /
``fit_command_operators_accel.py`` (``--quantity angvel``). For each test scene it forms the pair
(omega_a -> omega_b, Delta omega, Delta H = H_b - H_a) and steers H_a by several methods, then DECODES and
re-tracks the ball's ANGULAR velocity (marker orientation per frame -> slope):

  full_delta      H_a + Delta H                        per-pair on-manifold ceiling
  subspace_U[k]   H_a + P_U(Delta H)                   project true edit onto top-k global PCA basis
  random[k]       H_a + P_R(Delta H)                   random same-rank subspace control (should fail)
  ridge_global    H_a + B . Delta omega                steer straight from the command (bare ridge)
  cmd_U8_s{g}     H_a + g * (phi(w_a,w_b) Wu) U         COMMAND-ONLY synthesis in U (gain swept) -- the method
  ridge_rich      H_a + phi . B_rich                   richer-feature global ridge

Because the target is scalar, the metric is scalar too: Pearson correlation of decoded-vs-target omega,
sign accuracy, magnitude ratio, and mean absolute error. The gain is calibrated leakage-free (pick on a
val split of scenes minimizing MSE, report on a disjoint test split).

    python experiments/threads/angular-velocity/05_steering/steer_angvel2d.py --config configs/train/moving_ball_scene_angvel_decoder.yaml \
        --test_dir .../moving_ball_scene_angvel2d/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_angvel2d/subspace \
        --checkpoint .../moving_ball_scene_angvel2d_decoder_fp/checkpoints/last.pt \
        --output_dir outputs/analysis/moving_ball_angvel2d/steer --ks 2,4,8,16 --num_scenes 100 --device cuda
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
from src.analysis.ball_tracking import measured_angvel
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    from src.encoders.feature_extractor import latent_collate
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _apply_edit(Ha, edit_flat, device):
    out = {}
    for L, t in Ha.items():
        Ltok, Dd = t.shape[1], t.shape[2]
        e = torch.from_numpy(edit_flat[L].reshape(Ltok, Dd).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


@torch.no_grad()
def _decode_omega(decoder, latents, grid):
    out = decoder(latents, grid)
    if out.frames is None:
        return float("nan")
    return measured_angvel(out.frames[0].cpu())["omega"]


@torch.no_grad()
def _decode_frames(decoder, latents, grid):
    out = decoder(latents, grid)
    return None if out.frames is None else out.frames[0].cpu().numpy()  # (T, C, H, W)


def _save_viz(path, rows, titles, max_frames=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = rows[0].shape[0]
    idx = np.linspace(0, T - 1, min(max_frames, T)).round().astype(int)
    nr, nc = len(rows), len(idx)
    fig, axes = plt.subplots(nr, nc, figsize=(1.5 * nc, 1.7 * nr), squeeze=False)
    for r, (frames, title) in enumerate(zip(rows, titles)):
        for c, fi in enumerate(idx):
            img = np.transpose(frames[fi], (1, 2, 0)).clip(0, 1)
            ax = axes[r][c]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
            if r == 0:
                ax.set_title(f"t={fi}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _agg(decoded, target):
    """decoded/target: 1D arrays of scalar omega. Return rho, sign accuracy, magnitude ratio, MAE."""
    d = np.asarray(decoded, float); t = np.asarray(target, float)
    ok = np.isfinite(d) & np.isfinite(t)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "rho": float("nan"), "sign_acc": float("nan"),
                "mag_ratio": float("nan"), "mae": float("nan")}
    rho = float(np.corrcoef(t, d)[0, 1])
    sign_acc = float(np.mean(np.sign(d) == np.sign(t)))
    mag_ratio = float(np.median(np.abs(d) / (np.abs(t) + 1e-9)))
    mae = float(np.mean(np.abs(d - t)))
    return {"n": int(len(d)), "rho": round(rho, 4), "sign_acc": round(sign_acc, 3),
            "mag_ratio": round(mag_ratio, 3), "mae": round(mae, 5)}


def _calibrate(per_scene, gains, val_frac=0.5):
    """Leakage-free gain pick: minimize val MSE(decoded, target) on the first val_frac of scenes, report
    the held-out metrics on the disjoint rest. Scalar analog of calibrate_cmd_gain.py."""
    keys = sorted(per_scene)
    nval = max(1, int(round(len(keys) * val_frac)))
    val, test = keys[:nval], keys[nval:]

    def mse(scenes, method):
        e = [(per_scene[k][method] - per_scene[k]["omega_b"]) ** 2 for k in scenes
             if np.isfinite(per_scene[k][method])]
        return float(np.mean(e)) if e else float("inf")

    def metrics(scenes, method):
        return _agg([per_scene[k][method] for k in scenes], [per_scene[k]["omega_b"] for k in scenes])

    val_mse = {g: mse(val, f"cmd_U8_s{g:g}") for g in gains}
    best = min(val_mse, key=val_mse.get)
    refs = {m: metrics(test, m) for m in ("full_delta", "ridge_global", "subspace_U8") if m in
            per_scene[keys[0]]}
    return {
        "n_val": len(val), "n_test": len(test),
        "val_mse_by_gain": {f"{g:g}": round(v, 6) for g, v in val_mse.items()},
        "selected_gain": float(best),
        "HELDOUT_test": metrics(test, f"cmd_U8_s{best:g}"),
        "test_references": refs,
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ks", default="2,4,8,16")
    p.add_argument("--num_scenes", type=int, default=100)
    p.add_argument("--cmd_scales", default="1.0,1.5,2.0,2.5,3.0", help="gain sweep for cmd_U")
    p.add_argument("--cmd_ku", type=int, default=8, help="8 -> cmd_Wu_L*.npy; 16 -> cmd_Wu_ku16_L*.npy")
    p.add_argument("--viz_scenes", type=int, default=0,
                   help="save unsteered/cmd-U8-steered/target filmstrips for the first N scenes (0=off)")
    p.add_argument("--viz_gain", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device
    ks = [int(x) for x in args.ks.split(",")]
    cmd_scales = [float(x) for x in args.cmd_scales.split(",") if x.strip()]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[steer-w] {len(scenes)} test scenes, steering {len(scene_ids)}; layers={layers}")

    Bt = {L: np.load(art / f"ridge_Bt_L{L}.npy").astype(np.float64) for L in layers}
    Ubasis = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}
    rng = np.random.default_rng(0)
    Rbasis = {L: {k: vo.random_basis(Bt[L].shape[1], k, rng) for k in ks} for L in layers}
    wu_tag = "" if args.cmd_ku == 8 else f"_ku{args.cmd_ku}"
    Wu = {L: np.load(art / f"cmd_Wu{wu_tag}_L{L}.npy").astype(np.float64) for L in layers}
    Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
    print(f"[steer-w] loaded ridge B, U basis (rows={Ubasis[layers[0]].shape[0]}), cmd W_U (ku={args.cmd_ku})")

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    cmd_methods = [f"cmd_U8_s{g:g}" for g in cmd_scales] + ["ridge_rich"]
    methods = ["full_delta", "ridge_global"] + cmd_methods + \
              [f"subspace_U{k}" for k in ks] + [f"random{k}" for k in ks]
    decoded = {m: [] for m in methods}
    targets = []
    per_scene = {}

    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        wa, wb = vo.clip_angvel(sa), vo.clip_angvel(sb)   # [omega, 0]
        dw = wb - wa
        Ha = _to_dev(sa, layers, device)
        dH = {L: vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]) for L in layers}

        phi = vo.command_features(wa, wb)
        cU = {L: (phi @ Wu[L]) @ Ubasis[L][: Wu[L].shape[1]] for L in layers}
        edits = {
            "full_delta": dH,
            "ridge_global": {L: dw @ Bt[L] for L in layers},
            "ridge_rich": {L: phi @ Brich[L] for L in layers},
        }
        for g in cmd_scales:
            edits[f"cmd_U8_s{g:g}"] = {L: g * cU[L] for L in layers}
        for k in ks:
            edits[f"subspace_U{k}"] = {L: vo.project(dH[L], Ubasis[L][:k]) for L in layers}
            edits[f"random{k}"] = {L: vo.project(dH[L], Rbasis[L][k]) for L in layers}

        if n < args.viz_scenes:
            cU_edit = {L: args.viz_gain * cU[L] for L in layers}
            f_uns = _decode_frames(decoder, Ha, grid)
            f_steer = _decode_frames(decoder, _apply_edit(Ha, cU_edit, device), grid)
            f_tgt = _decode_frames(decoder, _to_dev(sb, layers, device), grid)
            if all(f is not None for f in (f_uns, f_steer, f_tgt)):
                _save_viz(out / f"viz_scene{s:05d}.png", [f_uns, f_steer, f_tgt],
                          [f"unsteered w_a={wa[0]:+.4f}", f"cmd-U8 x{args.viz_gain:g}",
                           f"target w_b={wb[0]:+.4f}"])

        targets.append(float(wb[0]))
        sc_row = {"omega_a": float(wa[0]), "omega_b": float(wb[0])}
        for m in methods:
            Hstar = _apply_edit(Ha, edits[m], device)
            w_dec = _decode_omega(decoder, Hstar, grid)
            decoded[m].append(w_dec)
            sc_row[m] = round(float(w_dec), 6)
        per_scene[f"scene{s:05d}"] = sc_row
        print(f"  scene{s:05d}: w_b={wb[0]:+.5f} full={decoded['full_delta'][-1]:+.5f} "
              f"cmdU8x2={per_scene[f'scene{s:05d}'].get('cmd_U8_s2', float('nan')):+.5f}")

    results = {m: _agg(decoded[m], targets) for m in methods}
    calib = _calibrate(per_scene, cmd_scales, val_frac=0.5)
    summary = {"test_dir": args.test_dir, "checkpoint": args.checkpoint, "layers": layers,
               "quantity": "angvel", "n_scenes": len(scene_ids), "ks": ks,
               "target": "omega_b (decoded vs target angular velocity)",
               "results": results, "calibration": calib, "per_scene": per_scene}
    (out / "steer2d_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "calib_cmd_gain.json").write_text(json.dumps(calib, indent=2))

    print("\n[steer-w] decoded-vs-target angular velocity (aggregate):")
    for m in methods:
        r = results[m]
        print(f"  {m:16s} rho={r['rho']!s:>7} sign_acc={r['sign_acc']!s:>5} "
              f"mag_ratio={r['mag_ratio']!s:>6} mae={r['mae']} n={r['n']}")
    h = calib["HELDOUT_test"]
    print(f"\n[steer-w] CALIBRATED gain={calib['selected_gain']:g} -> HELDOUT cmd_U8: "
          f"rho={h['rho']} sign_acc={h['sign_acc']} mag_ratio={h['mag_ratio']} mae={h['mae']} "
          f"(n_val={calib['n_val']}, n_test={calib['n_test']})")
    print(f"[steer-w] -> {out}/steer2d_summary.json")


if __name__ == "__main__":
    main()
