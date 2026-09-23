

#!/usr/bin/env python
"""AMORTIZED decoder-in-the-loop acceleration operator: train g(H_a, command) -> edit.

The test-time optimizer (``steer_accel_decopt.py``) proved the target accel is REACHABLE from H_a + a
command-derived edit (5.07deg held-out, mag corr 0.94), but at ~1.5 min/scene. This amortizes it: a small
conditional per-token network ``g`` is trained ONCE, end-to-end through the FROZEN decoder, so at inference
the edit is a single forward pass ``e = g(H_a, command)`` -- fast and generalizing.

Architecture (``EditNet``): per layer, a shared per-token MLP maps
  [ H_a token (D) || pooled-H_a context (ctx) || command_features(a_a,a_b) (13) || pos-enc (t,h,w) (3) ] -> edit (D)
so the edit is content- AND position-conditioned (it can place the high-rank footprint), scaled small at
init (starts ~no-op). Trained to minimise the decoded-acceleration error via a differentiable soft-centroid
-> parabola-fit readout, plus a trajectory anchor (ball stays on the known path x_b(t)=x_a(t)+1/2 da t^2).
Frozen decoder: only ``g`` learns. NO H_b at train or test -- reference H_a + the command only.

Held-out eval every few epochs uses the HONEST (non-differentiable) tracker, same metric as the test-time
proof. Saves the best ``g`` checkpoint + an eval summary.

    python experiments/threads/acceleration/03_train/train_accel_decopt_amortized.py --config configs/train/moving_ball_scene_decoder.yaml \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large --checkpoint .../last.pt \
        --output_dir .../decopt_amortized --epochs 30 --batch 4 --lr 1e-3 --device cuda
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
import torch.nn as nn

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_acceleration
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


# ----- differentiable readouts (shared with the test-time optimizer) --------------------------------
def parabola_row(T: int, device):
    f = np.arange(T, dtype=np.float64)
    A = np.stack([np.ones_like(f), f, f * f], 1)
    M = np.linalg.inv(A.T @ A) @ A.T
    return torch.tensor(2.0 * M[2], dtype=torch.float32, device=device)  # (T,) : accel = row @ centroids


def soft_centroids_b(frames: torch.Tensor) -> torch.Tensor:
    """(B,T,C,H,W) -> (B,T,2)=(x,y) in [0,1]; ball = darker than the per-frame mean."""
    B, T, C, H, W = frames.shape
    bright = frames.mean(2)                                    # (B,T,H,W)
    m = bright.mean(dim=(2, 3), keepdim=True)
    w = torch.relu(m - bright) + 1e-8
    ys = torch.linspace(0, 1, H, device=frames.device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=frames.device).view(1, 1, 1, W)
    den = w.sum(dim=(2, 3))
    cx = (w * xs).sum(dim=(2, 3)) / den
    cy = (w * ys).sum(dim=(2, 3)) / den
    return torch.stack([cx, cy], dim=2)                        # (B,T,2)


# ----- the amortized edit network -------------------------------------------------------------------
class EditNet(nn.Module):
    def __init__(self, layers, D, grid, ctx=64, hidden=512, out_scale=0.05):
        super().__init__()
        self.layers = layers
        self.D = D
        self.grid = grid
        self.out_scale = out_scale
        T, H, W = grid
        pe = np.stack(np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 3)
        pe = pe / np.array([max(1, T - 1), max(1, H - 1), max(1, W - 1)])
        self.register_buffer("posenc", torch.tensor(pe, dtype=torch.float32))     # (Ltok,3)
        self.ctxproj = nn.ModuleDict({str(L): nn.Linear(D, ctx) for L in layers})
        self.mlp = nn.ModuleDict()
        for L in layers:
            m = nn.Sequential(nn.Linear(D + ctx + vo.COMMAND_FEATURE_DIM + 3, hidden), nn.GELU(),
                              nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, D))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)               # start ~no-op
            self.mlp[str(L)] = m

    def forward(self, Ha, cmd):
        """Ha: {L:(B,Ltok,D)}; cmd:(B,13) -> edit {L:(B,Ltok,D)}."""
        out = {}
        Ltok = self.posenc.shape[0]
        for L in self.layers:
            h = Ha[L]                                          # (B,Ltok,D)
            B = h.shape[0]
            ctx = self.ctxproj[str(L)](h.mean(1))              # (B,ctx)
            pe = self.posenc.unsqueeze(0).expand(B, -1, -1)    # (B,Ltok,3)
            ctx_b = ctx.unsqueeze(1).expand(-1, Ltok, -1)
            cmd_b = cmd.unsqueeze(1).expand(-1, Ltok, -1)
            feat = torch.cat([h, ctx_b, cmd_b, pe], dim=2)
            out[L] = self.mlp[str(L)](feat) * self.out_scale
        return out


# ----- data helpers ---------------------------------------------------------------------------------
@torch.no_grad()
def preload_train(ds, scenes, layers, decoder, grid, device, n_targets=3, max_scenes=0, seed=0):
    """Preload, per scene: rank-0 reference latent H_a + a few (command, DECODED-TARGET-FRAMES) pairs.

    The amortized net never needs H_b at inference, but at TRAIN we supervise with the decoded target CLIP
    (pixel distillation -- a dense, non-gameable objective, unlike the soft-centroid accel loss which the
    net exploits). So per scene we decode ``n_targets`` target ranks' H_b -> frames (stored on CPU) and keep
    their accel a_b; the target LATENTS are discarded. ~17GB H_a + ~6-19GB frames fits in node RAM, and every
    epoch trains from memory -- no 172GB/epoch disk stream. Uses H_b ONLY here (train), exactly like the
    velocity command operator regressed against the true Delta H.
    """
    rng = np.random.default_rng(seed)
    entries = []
    ids = sorted(scenes)[:max_scenes] if max_scenes else sorted(scenes)
    for n, s in enumerate(ids):
        ranks = sorted(scenes[s]); i0 = scenes[s][ranks[0]]
        s0 = ds[i0]
        Ha = {L: np.asarray(s0["layers"][L], dtype=np.float32) for L in layers}
        aa, ref = vo.clip_acceleration(s0), vo.clip_positions(s0)
        others = ranks[1:]
        pick = rng.choice(others, size=min(n_targets, len(others)), replace=False)
        targets = []
        for r in pick:
            sr = ds[scenes[s][int(r)]]
            ab = vo.clip_acceleration(sr)
            Hb = {L: torch.tensor(np.asarray(sr["layers"][L], dtype=np.float32)[None], device=device)
                  for L in layers}
            fr = decoder(Hb, grid).frames
            targets.append({"ab": ab, "frames": fr[0].cpu()})       # (T,C,H,W) on CPU
        entries.append({"Ha": Ha, "aa": aa, "ref": ref, "targets": targets})
        if hasattr(ds, "_shard_cache") and len(ds._shard_cache) > 3:
            ds._shard_cache.clear()
        if (n + 1) % 100 == 0:
            print(f"[amort]   preloaded {n + 1}/{len(ids)} train scenes", flush=True)
    return entries


def make_batch(batch, rng, layers, device):
    """Build tensors from preloaded entries; sample one (command, target-frames) pair per scene."""
    Ha = {L: torch.tensor(np.stack([e["Ha"][L] for e in batch]), device=device) for L in layers}
    aa = np.stack([e["aa"] for e in batch])
    picks = [e["targets"][int(rng.integers(len(e["targets"])))] for e in batch]
    ab = np.stack([p["ab"] for p in picks])
    cmd = torch.tensor(np.stack([vo.command_features(aa[i], ab[i]) for i in range(len(batch))]),
                       dtype=torch.float32, device=device)
    tgt_frames = torch.stack([p["frames"] for p in picks]).to(device)   # (B,T,C,H,W)
    ab_t = torch.tensor(ab, dtype=torch.float32, device=device)
    da_t = torch.tensor(ab - aa, dtype=torch.float32, device=device)
    ref = torch.tensor(np.stack([e["ref"] for e in batch]), dtype=torch.float32, device=device)  # (B,F,2)
    return Ha, cmd, ab_t, da_t, ref, tgt_frames


def ball_mask(centers, H, W, sigma, device):
    """(B,T,2) normalized (x,y) centers -> (B,T,H,W) Gaussian bumps (focus the pixel loss on the ball)."""
    ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
    cx = centers[..., 0:1].unsqueeze(-1)                       # (B,T,1,1)
    cy = centers[..., 1:2].unsqueeze(-1)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return torch.exp(-d2 / (2 * sigma * sigma))                # (B,T,H,W)


@torch.no_grad()
def honest_eval(net, decoder, ds, scenes, layers, grid, device, n=30):
    net.eval()
    dec, tgt = [], []
    for s in sorted(scenes)[:n]:
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        cmd = torch.tensor(vo.command_features(aa, ab)[None], dtype=torch.float32, device=device)
        Ha = {L: torch.tensor(np.asarray(sa["layers"][L], dtype=np.float32)[None], device=device) for L in layers}
        edit = net(Ha, cmd)
        st = {L: Ha[L] + edit[L] for L in layers}
        fr = decoder(st, grid).frames
        if fr is None:
            continue
        m = measured_acceleration(fr[0].cpu())
        dec.append([m["acc_x"], m["acc_y"]]); tgt.append(ab)
    d, t = np.asarray(dec), np.asarray(tgt)
    ok = np.isfinite(d).all(1); d, t = d[ok], t[ok]
    cos = (d * t).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    md, mt = np.linalg.norm(d, axis=1), np.linalg.norm(t, axis=1)
    net.train()
    return {"n": int(len(d)), "angle_err_deg": round(float(ang.mean()), 2),
            "angle_median": round(float(np.median(ang)), 2),
            "mag_ratio_median": round(float(np.median(md / (mt + 1e-12))), 3),
            "mag_corr_r": round(float(np.corrcoef(md, mt)[0, 1]), 3)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--anchor", type=float, default=1.0)
    p.add_argument("--out_scale", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--n_targets", type=int, default=3, help="decoded target clips cached per scene (command diversity)")
    p.add_argument("--sigma", type=float, default=0.06, help="ball-mask Gaussian width (normalized units)")
    p.add_argument("--max_train_scenes", type=int, default=0)
    p.add_argument("--eval_n", type=int, default=30)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    grid = tuple(int(x) for x in tr[0]["grid"])
    D = int(np.asarray(tr[0]["layers"][layers[0]]).shape[-1])
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    print(f"[amort] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers} D={D} grid={grid}; "
          f"epochs={args.epochs} batch={args.batch} lr={args.lr}", flush=True)

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

    print("[amort] preloading H_a + decoded target clips into RAM (once) ...", flush=True)
    entries = preload_train(tr, tr_scenes, layers, decoder, grid, dev,
                            n_targets=args.n_targets, max_scenes=args.max_train_scenes, seed=args.seed)
    print(f"[amort] preloaded {len(entries)} scenes; training in-memory (pixel distillation, no per-epoch disk)",
          flush=True)

    net = EditNet(layers, D, grid, out_scale=args.out_scale).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"angle_err_deg": 1e9}
    hist = []
    order = list(range(len(entries)))
    for ep in range(args.epochs):
        rng.shuffle(order)
        net.train()
        ep_loss = ep_acc = nb = 0
        for i in range(0, len(order), args.batch):
            batch = [entries[j] for j in order[i: i + args.batch]]
            Ha, cmd, ab, da, ref, tgt_frames = make_batch(batch, rng, layers, dev)
            edit = net(Ha, cmd)
            st = {L: Ha[L] + edit[L] for L in layers}
            fr = decoder(st, grid).frames                       # (B,T,C,H,W)
            B_, Tf, C_, Hf, Wf = fr.shape
            # ball-focused pixel distillation: weight the reconstruction to the reference + target ball path
            ffb = (torch.arange(Tf, device=dev, dtype=torch.float32) ** 2).view(1, Tf, 1)
            xa = ref[:, :Tf, :]
            xb = xa + 0.5 * ffb * da.view(-1, 1, 2)
            m = (ball_mask(xa, Hf, Wf, args.sigma, dev) + ball_mask(xb, Hf, Wf, args.sigma, dev)).clamp(0, 1)
            m = m.unsqueeze(2)                                  # (B,T,1,H,W)
            loss = (m * (fr - tgt_frames) ** 2).sum() / (m.sum() * C_ + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            ep_loss += float(loss); nb += 1
        msg = f"[amort] ep{ep:02d} pix_loss={ep_loss/max(1,nb):.5f}"
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            ev = honest_eval(net, decoder, te, te_scenes, layers, grid, dev, args.eval_n)
            msg += f"  | HELDOUT {ev}"
            hist.append({"epoch": ep, **ev})
            if ev["angle_err_deg"] < best["angle_err_deg"]:
                best = {"epoch": ep, **ev}
                torch.save(net.state_dict(), out / "editnet_best.pt")
        print(msg, flush=True)

    torch.save(net.state_dict(), out / "editnet_last.pt")
    summary = {"best": best, "history": hist, "config": {"epochs": args.epochs, "batch": args.batch,
               "lr": args.lr, "anchor": args.anchor, "out_scale": args.out_scale},
               "references": {"decopt_testtime": 5.07, "canon": 14.46, "ceiling": 10.07}}
    (out / "amortized_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[amort] BEST held-out: {best}")
    print(f"[amort] -> {out}/amortized_summary.json", flush=True)


if __name__ == "__main__":
    main()
