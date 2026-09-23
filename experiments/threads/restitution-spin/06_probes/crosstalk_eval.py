

#!/usr/bin/env python
"""Does steering ONE physical quantity move ANOTHER? The pixel-level crosstalk + commutativity test.

The PI's objection, made measurable. Every steering result in this project so far edits a latent along
a direction fitted for one quantity in a scene where nothing else varied, so none of them can say
whether the edit is *selective*. This script measures selectivity directly, on a scene where velocity
``v`` and spin ``omega`` vary independently by construction, using two pixel readouts that were
certified against ground truth before any latent was touched
(``experiments/threads/restitution-spin/01_data/validate_spin_ball3d.py``).

For each held-out commutation square -- base cell ``(vi_a, si_a)``, target cell ``(vi_b, si_b)``, whose
four corners are all REAL rendered clips -- it decodes these conditions:

  base        H_a                                    the reference every effect is measured against
  V           H_a + W_V . f_v(v_a, v_b)              velocity edit alone   -> should move v, not omega
  S           H_a + W_S . f_s(w_a, w_b)              spin edit alone       -> should move omega, not v
  V+S         H_a + W_V . f_v + W_S . f_s            composed edit         -> should move both
  joint       H_a + W_J . f_both                     operator fitted directly on the diagonal pairs
  rand_V      H_a + r,  ||r|| = ||W_V . f_v||        NORM-MATCHED RANDOM CONTROL
  rand_S      H_a + r,  ||r|| = ||W_S . f_s||        NORM-MATCHED RANDOM CONTROL
  gt_vel      H(vi_b, si_a)                          real clip: the ceiling a perfect V steer reaches
  gt_spin     H(vi_a, si_b)                          real clip: the ceiling a perfect S steer reaches
  gt_both     H(vi_b, si_b)                          real clip: the ceiling for the composed steer
  V_then_S    steer V, DECODE, RE-ENCODE, steer S, decode      } the order test
  S_then_V    steer S, DECODE, RE-ENCODE, steer V, decode      }

**Why the random control is not optional.** Some fraction of any large latent perturbation will disturb
omega simply because the decoder is not perfectly disentangled. Reporting "the velocity edit changed
omega by X%" means nothing without knowing what an arbitrary perturbation of the same size does. The
norm-matched random direction supplies that floor: crosstalk is only evidence of a *directional*
coupling if it is well below the random control -- and if it is comparable, the honest conclusion is
that the edit is no more selective than noise of the same magnitude.

**Why the order test round-trips through pixels.** As additive latent vectors, ``W_V`` and ``W_S``
commute trivially -- ``H + a + b`` is ``H + b + a`` by vector addition, and testing that would be
arithmetic, not science. Order can only matter if something non-linear happens between the two edits.
The version that a controller would actually run does exactly that: apply one edit, render, observe,
re-encode, apply the next. That round trip is the encoder-decoder composition, which is thoroughly
non-linear, so ``SV`` and ``VS`` genuinely may differ. Both are also compared against the one-shot
``V+S`` composition.

All effects are measured as differences from the DECODED ``base``, never from ground truth, so any
constant bias of the decoder or of the trackers cancels instead of being charged to the steer.

    PYTHONPATH=. python experiments/threads/restitution-spin/06_probes/crosstalk_eval.py \
        --config configs/train/spin_ball3d_decoder.yaml \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --operators_dir .../analysis/spin_ball3d/operators \
        --checkpoint .../runs/spin_ball3d_decoder/checkpoints/last.pt \
        --output_dir .../analysis/spin_ball3d/crosstalk --num_scenes 64
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

import sys
from pathlib import Path as _P

import numpy as np
import torch

from src.analysis import spin_ops as so
from src.analysis import spin_tracking as st
from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders import build_encoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _add_edit(H, edit, device):
    """``H`` (dict L->(1,Ltok,D) torch) + flat per-layer numpy edit -> new latent dict."""
    out = {}
    for L, t in H.items():
        if L not in edit:
            # Layers the operator was not fitted on pass through untouched. The conditioned fits were
            # run at layer 18 alone while the decoder consumes all four, so this is the normal case for
            # a single-layer steer -- not a missing edit.
            out[L] = t
            continue
        e = torch.from_numpy(edit[L].reshape(t.shape[1], t.shape[2]).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


def _op_edit(B: dict[int, np.ndarray], feats: np.ndarray, layers,
             grid=None, inv=None) -> dict[int, np.ndarray]:
    """Apply a fitted operator ``B[L]`` (in_dim, n_tok*D) to a command feature vector.

    ``inv`` undoes position canonicalization. A canon operator predicts its edit in the BALL-CENTRED
    frame, so adding it to a real latent without rolling it back lands the edit at the grid centre --
    for most scenes nowhere near the ball. That decodes as "the steer did nothing", which is
    indistinguishable from a genuine negative result and is precisely the confusion this measurement
    exists to avoid.
    """
    out = {}
    for L in layers:
        e = feats @ B[L]
        out[L] = vo.roll_layer(e, grid, inv) if inv is not None else e
    return out


def _edit_norm(edit: dict[int, np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.dot(e, e)) for e in edit.values())))


def _random_edit(edit: dict[int, np.ndarray], rng: np.random.Generator) -> dict[int, np.ndarray]:
    """An isotropic random edit with the SAME total norm and the same per-layer norm profile.

    Matching the per-layer profile as well as the total matters: the operators put most of their mass in
    particular layers, and a random edit that spread its norm uniformly would be a weaker perturbation
    where it counts, flattering the comparison.
    """
    out = {}
    for L, e in edit.items():
        r = rng.standard_normal(e.shape)
        n = np.linalg.norm(r)
        out[L] = r * (np.linalg.norm(e) / (n + 1e-12))
    return out


@torch.no_grad()
def _decode(decoder, latents, grid):
    out = decoder(latents, grid)
    return out.frames[0].cpu() if out.frames is not None else None


def _measure(frames) -> dict[str, float]:
    """Both physical quantities, read off the same decoded pixels."""
    if frames is None:
        return {"vel_x": np.nan, "vel_y": np.nan, "speed": np.nan, "omega": np.nan, "spin_resid": np.nan}
    mv = measured_velocity(frames)
    try:
        ms = st.measured_spin(frames)
    except Exception:
        # The spin readout locates the marker, and a decoder that renders velocity but erases the
        # marker has nothing for it to find. That is a known state of the current checkpoint, not a
        # bug here: the velocity columns are still valid and are what this run is for. Failing the
        # whole square on it would throw away the measurement that IS available.
        ms = {"omega": np.nan, "residual": np.nan}
    return {"vel_x": mv["vel_x"], "vel_y": mv["vel_y"], "speed": mv["speed"],
            "omega": ms["omega"], "spin_resid": ms["residual"]}


@torch.no_grad()
def _reencode(encoder, frames, layers, device):
    """Decoded frames (T,C,H,W in [0,1]) -> latent dict, the way the extractor encoded them.

    The same ImageNet normalization the extraction pass used; anything else would feed the encoder a
    distribution it never saw and the round trip would measure that mismatch instead of the steer.
    """
    from src.data.video_transforms import VideoTransform

    tf = VideoTransform(image_size=256, num_frames=None, do_normalize=True)
    x = tf(frames).unsqueeze(0).to(device)
    bundle = encoder.encode(x, layers=layers)
    return {int(L): bundle.layers[int(L)] for L in layers}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--operators_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_scenes", type=int, default=64)
    p.add_argument("--squares_per_scene", type=int, default=4)
    p.add_argument("--n_vel", type=int, default=4)
    p.add_argument("--n_spin", type=int, default=4)
    p.add_argument("--gains", default="1,2,4,8,16",
                   help="command-gain multipliers swept for the single-quantity edits; the latent "
                        "operators are ridge-shrunk (~16%% velocity, ~1%% spin at unit gain), so unit "
                        "gain alone would measure the shrinkage rather than the steering")
    p.add_argument("--skip_order", action="store_true", help="skip the re-encode order test (no encoder)")
    p.add_argument("--joint", action="store_true",
                   help="also load and decode the diagonal-fit `both` operator. Off by default: it is "
                        "a reference for the composition question, not part of the crosstalk "
                        "measurement, and it is the largest of the three operator files.")
    p.add_argument("--max_cached_shards", type=int, default=2,
                   help="LatentDataset shard cache. The default is unbounded, which grows to the whole "
                        "test split and gets OOM-killed alongside the operators.")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device
    opd = Path(args.operators_dir)

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers,
                       max_cached_shards=args.max_cached_shards)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]
    print(f"[crosstalk] {len(sids)} test scenes, layers={layers}", flush=True)

    # The operator's own metadata decides how it must be applied. Reading it from disk rather than from
    # a command-line flag means an operator cannot be misapplied by forgetting a switch -- and a
    # canon/conditioned operator applied as if it were neither produces a confident near-zero edit that
    # no downstream number could flag as wrong.
    meta_p = opd / "operators_meta.json"
    fit_meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    canon = bool(fit_meta.get("canon", False))
    cond_dim = int(fit_meta.get("cond_dim", 0))

    # `both` is the diagonal-fit reference operator, not part of the crosstalk measurement itself, and
    # at cond512 it is a 45 GB file against 29 + 21 GB for the two that ARE needed. Loading all three
    # OOM-killed a 96 GB job. It is loaded only when asked for.
    ops, projs, mus = {}, {}, {}
    for kind in ("vel", "spin") + (("both",) if args.joint else ()):
        z = np.load(opd / f"operator_{kind}.npz")
        # Only the layers this operator was actually FIT on. The conditioned fits were run at layer 18
        # alone, while the decoder consumes all four; indexing B_ by the decoder's layer list would
        # raise, and defaulting the missing ones to zero would silently steer nothing. Unedited layers
        # are passed through untouched, which is the correct semantics for a single-layer steer.
        op_layers = sorted(int(k[2:]) for k in z.files if k.startswith("B_"))
        ops[kind] = {L: z[f"B_{L}"].astype(np.float64) for L in op_layers}
        if cond_dim and kind == "vel":
            projs = {L: z[f"P_{L}"].astype(np.float64) for L in op_layers}
            mus = {L: z[f"mu_{L}"].astype(np.float64) for L in op_layers if f"mu_{L}" in z.files}
    edit_layers = sorted(ops["vel"])
    print(f"[crosstalk] operators {sorted(ops)} on layers {edit_layers} "
          f"(canon={canon} cond_dim={cond_dim}); decoder sees {layers}", flush=True)

    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    encoder = None
    if not args.skip_order:
        encoder = build_encoder(cfg.encoder).to(device)
        encoder.freeze()
        print("[crosstalk] encoder loaded for the order test", flush=True)

    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    print(f"[crosstalk] gain sweep: {gains}", flush=True)

    rng = np.random.default_rng(0)
    rows: list[dict] = []

    for n, s in enumerate(sids):
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        # Squares are drawn per scene with a fixed seed so the condition set is identical across runs.
        picks = [(rng.integers(args.n_vel), rng.integers(args.n_spin),
                  rng.integers(args.n_vel), rng.integers(args.n_spin))
                 for _ in range(args.squares_per_scene * 3)]
        picks = [(va, sa, vb, sb) for va, sa, vb, sb in picks if va != vb and sa != sb]
        picks = picks[: args.squares_per_scene]

        for (vi_a, si_a, vi_b, si_b) in picks:
            sq = so.commutation_square(cells, int(vi_a), int(si_a), int(vi_b), int(si_b))
            sam = {k: ds[i] for k, i in sq.items()}
            grid = tuple(int(x) for x in sam["base"]["grid"])
            Ha = _to_dev(sam["base"], layers, device)

            va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
            wa, wb = so.clip_spin(sam["base"]), so.clip_spin(sam["spin_only"])
            pos, phi0 = vo.clip_start_pos(sam["base"]), so.clip_phi0(sam["base"])
            fv = vo.command_features_pos(va, vb, pos)
            fs = so.spin_command_features(wa, wb, phi0)

            sh = vo.canon_shift(pos, grid)
            inv = (-sh[0], -sh[1]) if canon else None
            zc = None

            def _aug_v(f, _va=va, _vb=vb):
                """Re-augment a command feature vector with this square's scene code.

                The order test re-derives its second command from what the first steer ACHIEVED, so it
                builds fresh feature vectors mid-square. Those must carry the same conditioning block
                as the fit, or they arrive at the operator with the wrong input dimension.
                """
                if zc is None:
                    return f
                return np.concatenate([f, zc, np.outer(zc, np.asarray(_vb) - np.asarray(_va)).ravel()])

            def _aug_s(f, _wa=wa, _wb=wb):
                if zc is None:
                    return f
                return np.concatenate([f, zc, zc * (float(_wb) - float(_wa))])

            if cond_dim:
                # The scene code must be built in the SAME frame the fit built it in -- canonical if
                # the fit was canonical -- because the code is a function of the latent and the roll
                # changes it. A code computed in the wrong frame is not a smaller error than a wrong
                # projection matrix; both yield an operator confidently evaluated off its own basis.
                def _basefl(L):
                    f = vo.layer_flat(sam["base"]["layers"][L])
                    f = vo.roll_layer(f, grid, sh) if canon else f
                    return f - mus[L] if L in mus else f
                zc = np.mean([_basefl(L) @ projs[L] for L in sorted(projs)], axis=0)
                zc = zc / (np.linalg.norm(zc) + 1e-12)
                fv = np.concatenate([fv, zc, np.outer(zc, np.asarray(vb) - np.asarray(va)).ravel()])
                fs = np.concatenate([fs, zc, zc * (float(wb) - float(wa))])

            eV = _op_edit(ops["vel"], fv, edit_layers, grid, inv)
            eS = _op_edit(ops["spin"], fs, edit_layers, grid, inv)
            eJ = (_op_edit(ops["both"], np.concatenate([fv, fs]), edit_layers, grid, inv)
                  if args.joint else None)
            eVS = {L: eV[L] + eS[L] for L in edit_layers}

            conds: dict[str, dict] = {}
            conds["base"] = _measure(_decode(decoder, Ha, grid))
            conds["V"] = _measure(_decode(decoder, _add_edit(Ha, eV, device), grid))
            conds["S"] = _measure(_decode(decoder, _add_edit(Ha, eS, device), grid))
            conds["V+S"] = _measure(_decode(decoder, _add_edit(Ha, eVS, device), grid))

            # GAIN SWEEP. The latent measurement (experiments/threads/restitution-spin/06_probes/latent_crosstalk.py) showed both
            # operators are badly shrunk: at unit gain the velocity edit reproduces only ~16% of the
            # real velocity displacement and the spin edit ~1%. That is ridge shrinkage on the command
            # axis -- the same failure already diagnosed on the robotics loop, where affine calibration
            # on measured outcomes took paddle 54.8% -> 94.7%. Decoding only the unit-gain edit would
            # therefore measure the shrinkage and report it as "steering does not work", which is the
            # wrong conclusion. The existing velocity pipeline sweeps command gains for exactly this
            # reason (steer_velocity2d.py --cmd_scales). Crosstalk is scale-invariant for a linear
            # operator, so the sweep also lets the leak/gain RATIO be read at whatever gain actually
            # delivers the commanded change.
            for g in gains:
                conds[f"V_g{g:g}"] = _measure(_decode(
                    decoder, _add_edit(Ha, {L: g * eV[L] for L in edit_layers}, device), grid))
                conds[f"S_g{g:g}"] = _measure(_decode(
                    decoder, _add_edit(Ha, {L: g * eS[L] for L in edit_layers}, device), grid))
            if eJ is not None:
                conds["joint"] = _measure(_decode(decoder, _add_edit(Ha, eJ, device), grid))
            conds["rand_V"] = _measure(_decode(decoder, _add_edit(Ha, _random_edit(eV, rng), device), grid))
            conds["rand_S"] = _measure(_decode(decoder, _add_edit(Ha, _random_edit(eS, rng), device), grid))
            for tag, key in (("gt_vel", "vel_only"), ("gt_spin", "spin_only"), ("gt_both", "both")):
                conds[tag] = _measure(_decode(decoder, _to_dev(sam[key], layers, device), grid))

            if encoder is not None:
                # V then S: the second command is re-derived from what was actually ACHIEVED after the
                # first edit, which is what a controller would do -- not from the original plan.
                f1 = _decode(decoder, _add_edit(Ha, eV, device), grid)
                H1 = _reencode(encoder, f1, layers, device)
                m1 = _measure(f1)
                fs2 = so.spin_command_features(m1["omega"], wb, phi0)
                conds["V_then_S"] = _measure(_decode(decoder, _add_edit(H1, _op_edit(ops["spin"], _aug_s(fs2), edit_layers, grid, inv), device), grid))

                f2 = _decode(decoder, _add_edit(Ha, eS, device), grid)
                H2 = _reencode(encoder, f2, layers, device)
                m2 = _measure(f2)
                fv2 = vo.command_features_pos(np.array([m2["vel_x"], m2["vel_y"]]), vb, pos)
                conds["S_then_V"] = _measure(_decode(decoder, _add_edit(H2, _op_edit(ops["vel"], _aug_v(fv2), edit_layers, grid, inv), device), grid))

            row = {"scene": int(s), "vi_a": int(vi_a), "si_a": int(si_a),
                   "vi_b": int(vi_b), "si_b": int(si_b),
                   "gt_va": va.tolist(), "gt_vb": vb.tolist(), "gt_wa": wa, "gt_wb": wb,
                   "norm_V": _edit_norm(eV), "norm_S": _edit_norm(eS),
                   "cond": conds}
            rows.append(row)

        if (n + 1) % 5 == 0:
            print(f"[crosstalk] scene {n + 1}/{len(sids)}  rows={len(rows)}", flush=True)

    (out / "crosstalk_raw.json").write_text(json.dumps(rows, indent=1))
    print(f"[crosstalk] wrote {len(rows)} squares -> {out/'crosstalk_raw.json'}", flush=True)


if __name__ == "__main__":
    main()
