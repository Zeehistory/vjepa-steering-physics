#!/usr/bin/env python
"""Presentation-grade before/after steering GIFs.

Every panel is real decoder output (or real GT video) taken from the same dumps the paper figures
are built from -- nothing here is re-simulated. The script only does layout: it lays the clips
side by side, overlays the same darkness-centroid track that produces the reported numbers, and
renders labels/progress chrome so a viewer can read the claim without a caption.

    python experiments/pipeline/08_figures/make_slide_gifs.py --out scratchpad/slides_bundle/gifs

Sources
    restitution   viz_sceneXXXXX_rN.npz written by the decopt3 restitution steering run
    acceleration  viz_sceneXXXXX.gif written by experiments/threads/acceleration/05_steering/steer_accel_decopt.py (3 panels of 256 px)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palette (matches paper figures)
PAPER = (252, 252, 251)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
MUTED = (138, 137, 131)
GRID = (227, 226, 221)
BLUE = (42, 120, 214)
ORANGE = (235, 104, 52)
VIOLET = (74, 58, 167)

FONTDIR = None
try:
    import matplotlib

    FONTDIR = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
except Exception:  # noqa: BLE001
    pass


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    if FONTDIR is not None and (FONTDIR / name).exists():
        return ImageFont.truetype(str(FONTDIR / name), size)
    return ImageFont.load_default()


# ---------------------------------------------------------------- clip helpers
def to_uint8(clip: np.ndarray) -> np.ndarray:
    """(T,C,H,W) float in [0,1] or (T,H,W,3) uint8 -> (T,H,W,3) uint8."""
    a = np.asarray(clip)
    if a.ndim == 4 and a.shape[1] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a.astype(np.float32), (0, 2, 3, 1))
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, -1)
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a.astype(np.uint8)


def centroids(clip_u8: np.ndarray, band: float = 0.18) -> np.ndarray:
    """Dark-pixel centroid per frame in pixel coords, (T,2) as (x,y); NaN where nothing is found.

    Same family as the metric code -- the ball is the darkest blob on a lighter field -- but the
    threshold is adaptive per frame (darkest 0.5% of pixels, plus a `band`), because the four
    datasets here sit at very different contrasts: a black ball on white, a dark-red ball on a
    grey gradient, and a navy ball on a rendered cream floor.
    """
    g = clip_u8.astype(np.float32).mean(-1) / 255.0
    floor = np.quantile(g.reshape(g.shape[0], -1), 0.005, axis=1)[:, None, None]
    dark = np.clip((floor + band) - g, 0, None)
    T, H, W = dark.shape
    xs = np.arange(W)[None, None, :]
    ys = np.arange(H)[None, :, None]
    m = dark.sum((1, 2))
    cx = np.where(m > 1e-6, (dark * xs).sum((1, 2)) / np.maximum(m, 1e-6), np.nan)
    cy = np.where(m > 1e-6, (dark * ys).sum((1, 2)) / np.maximum(m, 1e-6), np.nan)
    return np.stack([cx, cy], 1)


# ---------------------------------------------------------------- drawing
def rounded_panel(im: Image.Image, box, radius=6, outline=GRID, width=1):
    ImageDraw.Draw(im).rounded_rectangle(box, radius=radius, outline=outline, width=width)


def draw_trail(im: Image.Image, origin, cen: np.ndarray, t: int, rgb, max_len=16):
    """Fading dots for the tracked centre over the last `max_len` frames."""
    d = ImageDraw.Draw(im, "RGBA")
    lo = max(0, t - max_len)
    for k in range(lo, t + 1):
        if not np.isfinite(cen[k]).all():
            continue
        age = (t - k) / max(1, max_len)
        alpha = int(200 * (1.0 - age) ** 1.6) + 12
        r = 2.0 + 2.6 * (1.0 - age)
        x = origin[0] + float(cen[k][0])
        y = origin[1] + float(cen[k][1])
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*rgb, alpha))


def dashed_hline(im: Image.Image, y, x0, x1, rgb, dash=7, gap=6, width=2):
    d = ImageDraw.Draw(im, "RGBA")
    x = x0
    while x < x1:
        d.line([x, y, min(x + dash, x1), y], fill=(*rgb, 190), width=width)
        x += dash + gap


def text(im: Image.Image, xy, s, f, fill, anchor="la"):
    ImageDraw.Draw(im).text(xy, s, font=f, fill=fill, anchor=anchor)


def fit_font(strings, max_px: int, start: int, bold=False, floor=9):
    """Largest size at which every string fits in max_px -- keeps captions inside their panel."""
    size = start
    while size > floor:
        f = font(size, bold)
        if max(f.getbbox(s)[2] for s in strings if s) <= max_px:
            return f
        size -= 1
    return font(floor, bold)


# ---------------------------------------------------------------- the composer
def build_gif(
    out_path: Path,
    panels,               # list of dicts: clip (T,H,W,3 uint8), label, sublabel, accent, emphasize
    title: str,
    subtitle: str,
    footer: str,
    mark_frame: int | None = None,
    mark_label: str = "contact",
    target_line: float | None = None,   # y in panel px: dashed reference across every panel
    target_line_label: str = "",
    scale: float = 1.0,
    ms_per_frame: int = 140,
    hold_frames: int = 8,
    max_colors: int = 200,
    mp4_repeats: int = 4,
):
    S = int(round(panels[0]["clip"].shape[1] * scale))
    n = len(panels)
    PAD, GAP = 28, 18
    TOP = 92           # title block
    LAB = 26           # panel label line
    LINE = 19          # one caption line under a panel
    BAR = 46           # progress block
    nsub = max(len(p["sublabel"]) for p in panels)
    SUB = 8 + LINE * nsub
    W = PAD * 2 + n * S + (n - 1) * GAP
    H = TOP + LAB + S + SUB + BAR + PAD

    f_title = fit_font([title], W - 2 * PAD, int(25 * scale), bold=True)
    f_sub = fit_font([subtitle], W - 2 * PAD, int(15 * scale))
    f_lab = fit_font([p["label"] for p in panels], S, int(16 * scale), bold=True)
    f_val = fit_font([s for p in panels for s in p["sublabel"]], S, int(13 * scale))
    f_foot = font(int(12 * scale))

    clips = []
    for p in panels:
        c = p["clip"]
        if scale != 1.0:
            c = np.stack([np.array(Image.fromarray(fr).resize((S, S), Image.LANCZOS)) for fr in c])
        clips.append(c)
    cens = [centroids(c) for c in clips]
    T = min(c.shape[0] for c in clips)

    x0s = [PAD + i * (S + GAP) for i in range(n)]
    ytop = TOP + LAB

    frames = []
    for t in range(T):
        im = Image.new("RGB", (W, H), PAPER)
        text(im, (PAD, 22), title, f_title, INK)
        text(im, (PAD, 22 + int(32 * scale)), subtitle, f_sub, INK2)

        for i, p in enumerate(panels):
            x0 = x0s[i]
            acc = p["accent"]
            text(im, (x0, TOP + 2), p["label"], f_lab, acc)
            im.paste(Image.fromarray(clips[i][t]), (x0, ytop))
            if target_line is not None:
                dashed_hline(im, ytop + target_line * scale, x0 + 4, x0 + S - 4, VIOLET)
            draw_trail(im, (x0, ytop), cens[i], t, acc)
            w = 3 if p.get("emphasize") else 1
            rounded_panel(im, [x0, ytop, x0 + S - 1, ytop + S - 1],
                          outline=acc if p.get("emphasize") else GRID, width=w)
            for k, s in enumerate(p["sublabel"]):
                text(im, (x0, ytop + S + 7 + k * LINE), s, f_val,
                     acc if (k == 0 and p.get("emphasize")) else INK2)

        # progress bar under the panels: contact tick labelled above, chrome below
        by = ytop + S + SUB + 20
        bx0, bx1 = PAD, W - PAD
        ImageDraw.Draw(im).line([bx0, by, bx1, by], fill=GRID, width=3)
        frac = t / max(1, T - 1)
        ImageDraw.Draw(im).line([bx0, by, bx0 + frac * (bx1 - bx0), by], fill=BLUE, width=3)
        if mark_frame is not None:
            mx = bx0 + (mark_frame / max(1, T - 1)) * (bx1 - bx0)
            ImageDraw.Draw(im).line([mx, by - 6, mx, by + 6], fill=ORANGE, width=3)
            text(im, (mx, by - 20), mark_label, f_foot, ORANGE, anchor="ma")
        text(im, (bx0, by + 11), f"frame {t:d} / {T - 1}", f_foot, MUTED)
        foot = footer + (("   ·   " + target_line_label) if target_line_label else "")
        text(im, (bx1, by + 11), foot, f_foot, MUTED, anchor="ra")

        frames.append(im)

    # one shared adaptive palette -> no inter-frame flicker and a much smaller file
    pal = frames[0].quantize(colors=max_colors, method=Image.MEDIANCUT)
    qs = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    # hold on the last frame instead of duplicating it (PIL would optimise duplicates away)
    durations = [ms_per_frame] * (T - 1) + [ms_per_frame * (1 + hold_frames)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qs[0].save(out_path, save_all=True, append_images=qs[1:], loop=0,
               duration=durations, optimize=True)
    kb = out_path.stat().st_size / 1024
    print(f"  wrote {out_path}  ({W}x{H}, {len(frames)} frames, {kb:.0f} KB)")

    if mp4_repeats:
        _write_mp4(out_path.with_suffix(".mp4"), frames, ms_per_frame, hold_frames, mp4_repeats)


def _write_mp4(path: Path, frames, ms_per_frame: int, hold_frames: int, repeats: int):
    """Same movie as an h264 .mp4 -- Keynote and older PowerPoint handle these more reliably."""
    try:
        import imageio.v2 as iio2
    except Exception:  # noqa: BLE001
        return
    fps = max(1, round(1000 / ms_per_frame))
    seq = (list(frames) + [frames[-1]] * hold_frames) * repeats
    # h264 needs even dimensions
    w, h = seq[0].size
    arr = [np.array(f)[: h - h % 2, : w - w % 2] for f in seq]
    try:
        iio2.mimwrite(path, arr, fps=fps, codec="libx264", quality=8,
                      pixelformat="yuv420p", macro_block_size=None)
        print(f"  wrote {path}  ({fps} fps, {len(arr)} frames, "
              f"{path.stat().st_size / 1024:.0f} KB)")
    except Exception as exc:  # noqa: BLE001
        print(f"  (mp4 skipped: {exc})")


# ---------------------------------------------------------------- the three stories
RES_DIR = Path("outputs/restitution")
ACC_GIF = Path("outputs/analysis/"
               "moving_ball_accel2d_mixed/steer_decopt_viz")


def peak_after(cen: np.ndarray, bf: int) -> float:
    """Highest point (smallest y) the tracked ball reaches after the contact frame, in px."""
    seg = cen[bf:, 1]
    seg = seg[np.isfinite(seg)]
    return float(seg.min()) if seg.size else np.nan


def restitution_gif(npz: Path, out: Path, direction: str, last_frame: int = 12):
    """`last_frame` trims the tail where the decoder's motion-streak artefact takes over; the
    rebound difference is fully resolved by then (contact is frame 8)."""
    z = np.load(npz)
    e_a, e_b, bf = float(z["e_a"]), float(z["e_b"]), int(z["bounce_frame"])
    cut = last_frame + 1
    uns, ste, gt = (to_uint8(z[k])[:cut] for k in ("unsteered", "steered", "gt_target"))
    H = uns.shape[1]
    pk_u, pk_s, pk_g = (peak_after(centroids(c), bf) for c in (uns, ste, gt))

    def h(p):  # rebound height as a fraction of the frame, measured up from the floor
        return "rebound %.2f" % (1.0 - p / H) if np.isfinite(p) else "rebound n/a"

    verb = "bouncier" if e_b > e_a else "deader"
    build_gif(
        out,
        panels=[
            dict(clip=uns, label="BEFORE   decode(H_a)",
                 sublabel=[f"unedited latent  ·  e_a = {e_a:.2f}", h(pk_u)],
                 accent=INK2, emphasize=False),
            dict(clip=ste, label="AFTER   decode(H_a + edit)",
                 sublabel=[f"command only  ·  asked for e = {e_b:.2f}", h(pk_s)],
                 accent=BLUE, emphasize=True),
            dict(clip=gt, label="TARGET   real video",
                 sublabel=[f"a real ball with e = {e_b:.2f}", h(pk_g)],
                 accent=VIOLET, emphasize=False),
        ],
        title=f"Steering a material property: restitution {direction}",
        subtitle=("The edit sees only H_a and the number e — never the target video. "
                  f"Same approach, {verb} rebound."),
        footer="held-out  MAE 0.009 · rho 0.996 (n=400)",
        mark_frame=bf,
        mark_label="wall contact",
        target_line=pk_g,
        target_line_label="dashed = target rebound height",
        ms_per_frame=170,
    )


def measured_accel_angle(clip_u8: np.ndarray, a_b) -> float:
    """Angle (deg) between the acceleration tracked out of THESE pixels and the commanded a_b.

    Measured from the displayed frames with the repo's own tracker, so a caption can never end up
    describing a clip that is not on screen. (It once did: the run's `init_canon` field is the
    canon-operator's clip, which the GIF does not show.)
    """
    import torch

    from src.analysis.ball_tracking import measured_acceleration

    t = torch.from_numpy(np.ascontiguousarray(clip_u8.transpose(0, 3, 1, 2))).float() / 255.0
    m = measured_acceleration(t)
    v = np.array([m["acc_x"], m["acc_y"]], dtype=float)
    b = np.asarray(a_b, dtype=float)
    c = v @ b / (np.linalg.norm(v) * np.linalg.norm(b) + 1e-30)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def accel_gif(scene: int, out: Path):
    import imageio.v3 as iio

    a = iio.imread(ACC_GIF / f"viz_scene{scene:05d}.gif")
    S = a.shape[1]
    uns, ste, tgt = (a[:, :, i * S:(i + 1) * S] for i in range(3))
    a_b = json.loads((ACC_GIF / "decopt_summary.json").read_text())["per_scene"][f"scene{scene:05d}"]["a_b"]
    e_uns, e_ste, e_tgt = (measured_accel_angle(c, a_b) for c in (uns, ste, tgt))
    print(f"  [accel scene{scene:05d}] measured off-command: before {e_uns:.1f}° "
          f"after {e_ste:.1f}° oracle {e_tgt:.1f}°")
    build_gif(
        out,
        panels=[
            dict(clip=uns, label="BEFORE   decode(H_a)",
                 sublabel=["unedited latent",
                           f"{e_uns:.0f}° off the commanded a"],
                 accent=INK2, emphasize=False),
            dict(clip=ste, label="AFTER   decode(H_a + edit)",
                 sublabel=["command only, decoder-in-the-loop",
                           f"{e_ste:.0f}° off the commanded a"],
                 accent=BLUE, emphasize=True),
            dict(clip=tgt, label="ORACLE   decode(H_b)",
                 sublabel=["the target latent itself, decoded —",
                           f"{e_tgt:.0f}° off: the render floor"],
                 accent=VIOLET, emphasize=False),
        ],
        title="Steering acceleration",
        subtitle=("Nine linear operators plateaued at 14.5°; optimising the edit through the "
                  "frozen decoder reaches 5.1°."),
        footer="held-out  5.07° · magnitude corr +0.94",
        target_line_label="dots = tracked ball centre",
        ms_per_frame=150,
    )


def rb3d_gif(npz: Path, out: Path):
    z = np.load(npz)
    uns, ste, gt = (to_uint8(z[k]) for k in ("dec_a", "dec_s", "gt_b"))
    e0, e1 = float(z["ang_deg_a"]), float(z["ang_deg"])
    build_gif(
        out,
        panels=[
            dict(clip=uns, label="BEFORE   decode(H_a)",
                 sublabel=["unedited latent",
                           f"{e0:.0f}° off the commanded direction"],
                 accent=INK2, emphasize=False),
            dict(clip=ste, label="AFTER   decode(H_a + edit)",
                 sublabel=["command only, no target latent",
                           f"{e1:.0f}° off the commanded direction"],
                 accent=BLUE, emphasize=True),
            dict(clip=gt, label="TARGET   real video",
                 sublabel=["the MuJoCo clip we asked for",
                           "(never shown to the operator)"],
                 accent=VIOLET, emphasize=False),
        ],
        title="Steering velocity in real 3D",
        subtitle=("MuJoCo rolling ball — perspective, shading, a textured floor. The same "
                  "command-only cmd-U8 edit as the 2D cartoon."),
        footer="held-out  10.9° command-only · 15.4° for the target-based method",
        target_line_label="dots = tracked ball centre",
        ms_per_frame=150,
    )


def angvel_gif(npz: Path, out: Path):
    z = np.load(npz)
    uns, ste, tgt = (to_uint8(z[k]) for k in ("unsteered", "steered", "target"))
    wa, wb = float(z["omega_a"]), float(z["omega_b"])
    ou, os_, ot = (float(z[k]) for k in ("om_unsteered", "om_steered", "om_target"))
    build_gif(
        out,
        panels=[
            dict(clip=uns, label="BEFORE   decode(H_a)",
                 sublabel=[f"unedited latent  ·  omega_a = {wa:+.2f}",
                           f"decoded spin {ou:+.2f}"],
                 accent=INK2, emphasize=False),
            dict(clip=ste, label="AFTER   decode(H_a + edit)",
                 sublabel=[f"command only  ·  asked for {wb:+.2f}",
                           f"decoded spin {os_:+.2f}"],
                 accent=BLUE, emphasize=True),
            dict(clip=tgt, label="ORACLE   decode(H_b)",
                 sublabel=["the target latent itself —",
                           f"decoded spin {ot:+.2f}"],
                 accent=VIOLET, emphasize=False),
        ],
        title="Steering rotation: angular velocity",
        subtitle=("A linear command axis gives rho 0.005. A Fourier-in-orientation basis gives "
                  "rho 0.94 from the command alone."),
        footer="held-out  rho 0.94 · sign 100% (interp ceiling 0.87, and it needs H_b)",
        target_line_label="dots = tracked ball centre",
        ms_per_frame=170,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="scratchpad/slides_bundle/gifs")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--angvel", action="store_true",
                    help="also build the angular-velocity clip -- inspection only, see the note below")
    a = ap.parse_args()
    out = Path(a.out)
    print("building slide GIFs ->", out)

    restitution_gif(RES_DIR / "decopt3_up_sc0/viz_scene00001_r7.npz",
                    out / "steer_restitution_up.gif", "UP")
    restitution_gif(RES_DIR / "decopt3_down_viz/viz_scene00000_r0.npz",
                    out / "steer_restitution_down.gif", "DOWN")
    accel_gif(3, out / "steer_acceleration.gif")

    rb3d = Path("scratchpad/viz_dumps/rb3d/viz_scene00004.npz")
    if rb3d.exists():
        rb3d_gif(rb3d, out / "steer_velocity_3d.gif")
    # Angular velocity is NOT built by default, and that is deliberate. The steer is real (held-out
    # rho 0.942, sign 100%), but this decoder renders the rotating bar as a featureless axis-aligned
    # rectangle at every frame -- the marker is never drawn, so all three panels look identical and
    # the clip reads as "nothing happened". See gifs/why_no_angvel_clip.png for the evidence.
    # --angvel forces it anyway, for inspection only.
    if a.angvel:
        picks = sorted(Path("scratchpad/viz_dumps/angvel").glob("viz_scene*.npz"))
        if picks:
            angvel_gif(picks[0], out / "steer_angular_velocity.gif")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
