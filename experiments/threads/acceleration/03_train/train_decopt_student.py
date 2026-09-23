

#!/usr/bin/env python
"""AMORTIZE decoder-in-the-loop accel steering by SUPERVISED distillation of the optimized edits.

The prior amortization (`train_accel_decopt_amortized.py`) trained EditNet THROUGH the frozen decoder
(soft-tracker / pixel-distillation losses) and OVERFIT on all three variants -- the decoder-in-loop signal
lets the net game the objective. This script tries the one variant memory flagged as untried: distill from
many test-time-optimized edits as EXPLICIT SUPERVISED TARGETS. `steer_accel_decopt.py --dump_dir` saved the
winning free edit e*(scene) for a batch of TRAIN scenes; here we regress g(H_a, command) -> e* directly
(per-layer relative-MSE + cosine), no decoder in the loop. The frozen decoder is used ONLY for the honest
held-out eval (decode the steered latent, parabola-track the accel), same metric as the 5.07deg proof.

If a per-token conditional net can fit e* AND generalize (held-out decoded angle beats the 14.46 linear
wall, ideally toward 5.07), acceleration steering is AMORTIZED. If it fits train edits but held-out decode
stays ~random/14.5, the winning edit is scene-specific / not a function of (H_a, command) -- a real finding.

    python experiments/threads/acceleration/03_train/train_decopt_student.py --config configs/train/moving_ball_scene_decoder.yaml \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large --checkpoint .../last.pt \
        --edits_dir .../decopt_edits --output_dir .../decopt_student --epochs 60 --device cuda
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
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_accel_decopt_amortized import EditNet, soft_centroids_b, parabola_row  # noqa: E402

from src.analysis import velocity_ops as vo  # noqa: E402
from src.analysis.ball_tracking import measured_acceleration  # noqa: E402
from src.decoders import build_decoder  # noqa: E402
from src.encoders.feature_extractor import LatentDataset  # noqa: E402
from src.training.checkpoints import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def canon_edit_flat(sample, aa, ab, grid, Wu, U):
    """Command-only canon edit per layer (flat) = roll((cmd@Wu)@U, -canon_shift). Matches steer_accel2d."""
    cmd = vo.command_features(aa, ab)
    sh = vo.canon_shift(vo.clip_start_pos(sample), grid)
    return {L: vo.roll_layer((cmd @ Wu[L]) @ U[L], grid, (-sh[0], -sh[1])) for L in Wu}


def load_edits_and_Ha(edits_dir, ds, scenes, layers, grid, device, max_n=0, Wu=None, U=None):
    """Match dumped edits to their train scene's rank-0 H_a. Returns list of dicts (all CPU numpy).

    If Wu/U (canon operator) given, also store the canon edit and set the regression target to the RESIDUAL
    decopt - canon (the net then only learns the correction on top of the 14.5deg canon operator).
    """
    T, H, W = grid
    Ltok = T * H * W
    files = sorted(Path(edits_dir).glob("scene*.npz"))
    if max_n:
        files = files[:max_n]
    entries = []
    for k, f in enumerate(files):
        sid = int(f.stem.replace("scene", ""))
        if sid not in scenes:
            continue
        z = np.load(f)
        ranks = sorted(scenes[sid]); i0 = scenes[sid][ranks[0]]
        s0 = ds[i0]
        Ha = {L: np.asarray(s0["layers"][L], dtype=np.float32) for L in layers}   # (Ltok, D)
        edit = {L: z[f"L{L}"].reshape(Ltok, -1).astype(np.float32) for L in layers}
        aa, ab = z["a_a"].astype(np.float32), z["a_b"].astype(np.float32)
        ent = {"sid": sid, "Ha": Ha, "aa": aa, "ab": ab,
               "ref": vo.clip_positions(s0).astype(np.float32)}   # (F,2) for the decode-consistency term
        if Wu is not None:
            ce = canon_edit_flat(s0, aa, ab, grid, Wu, U)
            ent["canon"] = {L: ce[L].reshape(Ltok, -1).astype(np.float32) for L in layers}
            ent["edit"] = {L: edit[L] - ent["canon"][L] for L in layers}   # residual target
        else:
            ent["edit"] = edit
        entries.append(ent)
        if hasattr(ds, "_shard_cache") and len(ds._shard_cache) > 3:
            ds._shard_cache.clear()
        if (k + 1) % 100 == 0:
            print(f"  loaded {k + 1}/{len(files)} edits", flush=True)
    return entries


def batch_tensors(batch, layers, device, decode=False):
    Ha = {L: torch.tensor(np.stack([e["Ha"][L] for e in batch]), device=device) for L in layers}
    tgt = {L: torch.tensor(np.stack([e["edit"][L] for e in batch]), device=device) for L in layers}
    cmd = torch.tensor(np.stack([vo.command_features(e["aa"], e["ab"]) for e in batch]),
                       dtype=torch.float32, device=device)
    extra = {}
    if decode:
        ab = np.stack([e["ab"] for e in batch]); aa = np.stack([e["aa"] for e in batch])
        extra["ab"] = torch.tensor(ab, dtype=torch.float32, device=device)
        extra["da"] = torch.tensor(ab - aa, dtype=torch.float32, device=device)
        extra["ref"] = torch.tensor(np.stack([e["ref"] for e in batch]), dtype=torch.float32, device=device)
        if "canon" in batch[0]:
            extra["canon"] = {L: torch.tensor(np.stack([e["canon"][L] for e in batch]), device=device)
                              for L in layers}
    return Ha, cmd, tgt, extra


def edit_loss(pred, tgt, layers):
    """Per-layer relative MSE + (1 - cosine), averaged over layers."""
    mse = 0.0; cosl = 0.0
    for L in layers:
        p, t = pred[L], tgt[L]
        denom = (t ** 2).sum(dim=(1, 2)) + 1e-8
        mse = mse + (((p - t) ** 2).sum(dim=(1, 2)) / denom).mean()
        pf = p.flatten(1); tf = t.flatten(1)
        cosl = cosl + (1 - torch.nn.functional.cosine_similarity(pf, tf, dim=1)).mean()
    n = len(layers)
    return mse / n, cosl / n


@torch.no_grad()
def honest_eval_resid(net, decoder, ds, scenes, layers, grid, device, n, Wu=None, U=None):
    """Decode the steered latent (H_a + [canon] + net edit) and parabola-track the accel; held-out."""
    net.eval()
    dec, tgt = [], []
    for s in sorted(scenes)[:n]:
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        cmd = torch.tensor(vo.command_features(aa, ab)[None], dtype=torch.float32, device=device)
        Ha = {L: torch.tensor(np.asarray(sa["layers"][L], dtype=np.float32)[None], device=device) for L in layers}
        edit = net(Ha, cmd)
        ce = canon_edit_flat(sa, aa, ab, grid, Wu, U) if Wu is not None else None
        st = {}
        for L in layers:
            base = Ha[L] + edit[L]
            if ce is not None:
                base = base + torch.tensor(ce[L].reshape(1, Ha[L].shape[1], Ha[L].shape[2]),
                                           dtype=torch.float32, device=device)
            st[L] = base
        fr = decoder(st, grid).frames
        if fr is None:
            continue
        m = measured_acceleration(fr[0].cpu())
        dec.append([m["acc_x"], m["acc_y"]]); tgt.append(ab)
    net.train()
    d, t = np.asarray(dec), np.asarray(tgt)
    ok = np.isfinite(d).all(1); d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d))}
    cos = (d * t).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    md, mt = np.linalg.norm(d, axis=1), np.linalg.norm(t, axis=1)
    return {"n": int(len(d)), "angle_err_deg": round(float(ang.mean()), 2),
            "angle_median": round(float(np.median(ang)), 2),
            "mag_ratio_median": round(float(np.median(md / (mt + 1e-12))), 3),
            "mag_corr_r": round(float(np.corrcoef(md, mt)[0, 1]), 3)}


@torch.no_grad()
def edit_recon_val(net, entries, layers, device, batch=4):
    net.eval()
    tot = {L: 0.0 for L in layers}; nb = 0
    for i in range(0, len(entries), batch):
        b = entries[i:i + batch]
        Ha, cmd, tgt, _ = batch_tensors(b, layers, device)
        pred = net(Ha, cmd)
        for L in layers:
            pf = pred[L].flatten(1); tf = tgt[L].flatten(1)
            tot[L] += float(torch.nn.functional.cosine_similarity(pf, tf, dim=1).mean())
        nb += 1
    net.train()
    return {f"L{L}": round(tot[L] / max(1, nb), 3) for L in layers}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--edits_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--residual_canon", action="store_true",
                   help="learn the residual over the canon operator (target = decopt - canon; steer = "
                        "H_a + canon + net). Strictly improves on the 14.5deg canon baseline.")
    p.add_argument("--artifacts_dir", default="", help="canon operator dir (needed for --residual_canon)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--cos_weight", type=float, default=1.0)
    p.add_argument("--decode_weight", type=float, default=0.0,
                   help=">0 adds a decode-consistency term (decode the predicted edit, match target accel + "
                        "path anchor) on top of the supervised edit loss = the HYBRID that keeps the "
                        "predicted edit on the decoder's manifold. Slower (decodes each train batch).")
    p.add_argument("--anchor", type=float, default=1.0, help="trajectory-anchor weight in the decode term")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_n", type=int, default=30)
    p.add_argument("--max_edits", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    grid = tuple(int(x) for x in tr[0]["grid"])
    D = int(np.asarray(tr[0]["layers"][layers[0]]).shape[-1])
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)

    rec0 = tr.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in tr.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=dev)
    for pm in decoder.parameters():
        pm.requires_grad_(False)

    Wu = U = None
    if args.residual_canon:
        art = Path(args.artifacts_dir)
        Wu = {L: np.load(art / f"cmd_Wu_canon_L{L}.npy").astype(np.float64) for L in layers}
        U = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[student] RESIDUAL over canon operator (U rows={U[layers[0]].shape[0]})", flush=True)

    print(f"[student] loading dumped edits from {args.edits_dir}", flush=True)
    entries = load_edits_and_Ha(args.edits_dir, tr, tr_scenes, layers, grid, dev, args.max_edits, Wu, U)
    rng.shuffle(entries)
    nval = max(1, int(len(entries) * args.val_frac))
    val, train = entries[:nval], entries[nval:]
    print(f"[student] {len(entries)} edits -> {len(train)} train / {len(val)} val; layers={layers} D={D}",
          flush=True)

    net = EditNet(layers, D, grid, hidden=args.hidden, out_scale=1.0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"[student] EditNet params={nparam/1e6:.2f}M", flush=True)

    dw = args.decode_weight
    Mrow = None    # parabola row over the DECODED frame count (set on first decoded batch): accel = Mrow @ cen
    best = {"angle_err_deg": 1e9}
    hist = []
    order = list(range(len(train)))
    for ep in range(args.epochs):
        rng.shuffle(order); net.train()
        eml = ecl = edl = nb = 0.0
        for i in range(0, len(order), args.batch):
            b = [train[j] for j in order[i:i + args.batch]]
            Ha, cmd, tgt, extra = batch_tensors(b, layers, dev, decode=dw > 0)
            pred = net(Ha, cmd)
            mse, cosl = edit_loss(pred, tgt, layers)
            loss = mse + args.cos_weight * cosl
            dloss_v = 0.0
            if dw > 0:
                # decode-consistency: decode H_a + [canon] + pred, match target accel + stay on the path.
                st = {L: Ha[L] + pred[L] + (extra["canon"][L] if "canon" in extra else 0.0) for L in layers}
                fr = decoder(st, grid).frames                       # (B,T,C,H,W)
                cen = soft_centroids_b(fr)                          # (B,T,2)
                if Mrow is None or Mrow.shape[0] != cen.shape[1]:
                    Mrow = parabola_row(cen.shape[1], dev)          # (T_decoded,)
                accel = torch.stack([cen[:, :, 0] @ Mrow, cen[:, :, 1] @ Mrow], dim=1)   # (B,2)
                ab_sq = (extra["ab"] ** 2).sum(1) + 1e-10
                acc_err = (((accel - extra["ab"]) ** 2).sum(1) / ab_sq).mean()
                Tf = min(cen.shape[1], extra["ref"].shape[1])
                ffb = (torch.arange(Tf, device=dev, dtype=torch.float32) ** 2).view(1, Tf, 1)
                xb = extra["ref"][:, :Tf, :] + 0.5 * ffb * extra["da"].view(-1, 1, 2)
                traj = ((cen[:, :Tf] - xb) ** 2).sum(dim=(1, 2)).mean()
                dloss = acc_err + args.anchor * traj
                loss = loss + dw * dloss
                dloss_v = float(dloss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            eml += float(mse); ecl += float(cosl); edl += dloss_v; nb += 1
        msg = f"[student] ep{ep:02d} mse={eml/nb:.4f} cos_loss={ecl/nb:.4f} decode={edl/nb:.4f}"
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            vcos = edit_recon_val(net, val, layers, dev, args.batch)
            ev = honest_eval_resid(net, decoder, te, te_scenes, layers, grid, dev, args.eval_n, Wu, U)
            msg += f"  | val_editcos={vcos}  | HELDOUT-decode {ev}"
            hist.append({"epoch": ep, "val_editcos": vcos, **ev})
            if ev["angle_err_deg"] < best["angle_err_deg"]:
                best = {"epoch": ep, **ev}
                torch.save(net.state_dict(), out / "student_best.pt")
        print(msg, flush=True)

    torch.save(net.state_dict(), out / "student_last.pt")
    summary = {"best": best, "history": hist, "n_edits": len(entries),
               "config": vars(args), "references": {"decopt_testtime": 5.07, "canon": 14.46, "ceiling": 10.07}}
    (out / "student_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[student] BEST held-out decode: {best}")
    print(f"[student] -> {out}/student_summary.json", flush=True)


if __name__ == "__main__":
    main()
