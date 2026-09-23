"""Publication-quality visualizations, saved automatically to an organized output directory.

All plotting uses a non-interactive Matplotlib backend so it runs headless (CI / clusters). Functions
return the path written. Tensors are ``(T, C, H, W)`` in ``[0, 1]`` unless noted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..utils.video_io import save_video  # noqa: E402


def _to_hwc(frames: torch.Tensor) -> np.ndarray:
    return frames.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()


def reconstruction_grid(
    original: torch.Tensor, recon: torch.Tensor, path: str | Path, max_frames: int = 8
) -> Path:
    """Two-row grid: original (top) vs reconstruction (bottom)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o, r = _to_hwc(original), _to_hwc(recon)
    t = min(max_frames, o.shape[0], r.shape[0])
    fig, axes = plt.subplots(2, t, figsize=(1.6 * t, 3.4))
    if t == 1:
        axes = axes.reshape(2, 1)
    for i in range(t):
        axes[0, i].imshow(o[i]); axes[0, i].axis("off")
        axes[1, i].imshow(r[i]); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", rotation=0, ha="right", labelpad=30)
    axes[1, 0].set_ylabel("recon", rotation=0, ha="right", labelpad=30)
    fig.suptitle("Reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def error_map(original: torch.Tensor, recon: torch.Tensor, path: str | Path, max_frames: int = 8) -> Path:
    """Per-pixel absolute error heatmap over frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if recon.shape[-1] != original.shape[-1]:
        recon = torch.nn.functional.interpolate(recon, size=original.shape[-2:], mode="bilinear",
                                                align_corners=False)
    err = (original - recon).abs().mean(1).detach().cpu().numpy()  # (T, H, W)
    t = min(max_frames, err.shape[0])
    fig, axes = plt.subplots(1, t, figsize=(1.6 * t, 1.9))
    if t == 1:
        axes = [axes]
    for i in range(t):
        axes[i].imshow(err[i], cmap="magma", vmin=0, vmax=max(err.max(), 1e-3))
        axes[i].axis("off")
    fig.suptitle("Absolute error")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def trajectory_overlay(
    frames: torch.Tensor,
    state: torch.Tensor,
    state_keys: list[str],
    path: str | Path,
    draw_velocity: bool = True,
) -> Path:
    """Overlay object trajectories (and optional velocity arrows) on the last frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = _to_hwc(frames)[-1]
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    keys = state_keys
    n_obj = sum(1 for k in keys if k.endswith("pos_x"))
    s = state.detach().cpu().numpy()
    for o in range(n_obj):
        try:
            xi = keys.index(f"obj{o}_pos_x"); yi = keys.index(f"obj{o}_pos_y")
        except ValueError:
            continue
        xs, ys = s[:, xi] * w, s[:, yi] * h
        ax.plot(xs, ys, "-o", ms=3, lw=1.5, label=f"obj{o}")
        if draw_velocity:
            vxi, vyi = keys.index(f"obj{o}_vel_x"), keys.index(f"obj{o}_vel_y")
            ax.arrow(xs[-1], ys[-1], s[-1, vxi] * w * 4, s[-1, vyi] * h * 4,
                     color="cyan", head_width=3, length_includes_head=True)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Trajectory + velocity")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def layerwise_probe_plot(records: list[dict], path: str | Path) -> Path:
    """Line plot of probe R² by layer for each variable (linear vs MLP), with the control band."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    variables = sorted({r["variable"] for r in records})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for var in variables:
        for probe in ("linear", "mlp"):
            pts = sorted([r for r in records if r["variable"] == var and r["probe"] == probe],
                         key=lambda r: r["layer"])
            if not pts:
                continue
            ls = "-" if probe == "linear" else "--"
            ax.plot([p["layer"] for p in pts], [p["r2"] for p in pts], ls, marker="o", ms=3,
                    label=f"{var} ({probe})")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("probe R²")
    ax.set_title("Layerwise physical decodability")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def velocity_probe_plot(
    records: list[dict], path: str | Path, target: str = "vel"
) -> Path:
    """R² vs layer for one velocity target, one line per representation (clip_pool/temporal/...).

    Shows the headline Step-2 comparison: does keeping the temporal (8x1024) sequence decode velocity
    better than pooling to a single (1x1024) vector? Linear probe solid, MLP dashed; the shuffled-latent
    control is drawn as a thin grey line near zero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records if r["target"] == target]
    reps = sorted({r["representation"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cmap = plt.get_cmap("tab10")
    for ri, rep in enumerate(reps):
        for kind, ls in (("linear", "-"), ("mlp", "--")):
            ys = [next((r["r2"] for r in rows if r["layer"] == L and r["representation"] == rep
                        and r["probe"] == kind), np.nan) for L in layers]
            ax.plot(layers, ys, ls, color=cmap(ri), marker="o", ms=3,
                    label=f"{rep} ({kind})")
        # one shuffled-latent control line per representation (linear)
        cs = [next((r["ctrl_shuffled_latent_r2"] for r in rows if r["layer"] == L
                    and r["representation"] == rep and r["probe"] == "linear"), np.nan)
              for L in layers]
        ax.plot(layers, cs, ":", color="grey", lw=0.8)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("encoder layer"); ax.set_ylabel(f"R²  (target = {target})")
    ax.set_title(f"Velocity decodability by layer — target={target}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix_plot(
    matrix: np.ndarray, labels: list[str], path: str | Path, title: str = "Confusion (out-of-fold)"
) -> Path:
    """Row-normalised confusion-matrix heatmap (rows = true class, cols = predicted)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(matrix, dtype=float)
    m = m / (m.sum(1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(0.7 * len(labels) + 2, 0.7 * len(labels) + 1.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if m[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def similarity_heatmap(
    matrix: np.ndarray, labels: list[str], path: str | Path, title: str = "Cosine",
    vmin: float = -1.0, vmax: float = 1.0, cmap: str = "RdBu_r",
) -> Path:
    """Diverging heatmap for signed similarity / angle matrices (cosines, principal angles)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.7 * len(labels) + 2, 0.7 * len(labels) + 1.6))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def classification_by_layer_plot(records: list[dict], path: str | Path) -> Path:
    """Accuracy vs encoder depth for linear/MLP probes, with the shuffled-label control band.

    Physics (not appearance) structure should show accuracy that *rises with depth*; a flat line near
    the pixel baseline means the probe is reading appearance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for probe in ("linear", "mlp"):
        pts = sorted([r for r in records if r["probe"] == probe and r["layer"] >= 0],
                     key=lambda r: r["layer"])
        if pts:
            ls = "-" if probe == "linear" else "--"
            ax.plot([p["layer"] for p in pts], [p["accuracy"] for p in pts], ls, marker="o", ms=4,
                    label=f"{probe} accuracy")
            ax.plot([p["layer"] for p in pts], [p["ctrl_shuffled_label_accuracy"] for p in pts],
                    ls, color="gray", alpha=0.5, lw=1, label=f"{probe} shuffled ctrl")
    # pixel + control reference lines
    pix = [r for r in records if str(r["probe"]).startswith("pixel_")]
    for r in pix:
        ax.axhline(r["accuracy"], color="green", ls=":", lw=1, alpha=0.7,
                   label=f"{r['probe']} (appearance)")
    ax.set_xlabel("encoder layer"); ax.set_ylabel("category accuracy")
    ax.set_title("Category separability by depth")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def steering_sweep_plot(
    sweeps: dict[str, list[dict]], path: str | Path, xkey: str = "alpha", ykey: str = "readout",
    title: str = "Steering controllability", ylabel: str = "decoded readout",
) -> Path:
    """Plot one or more α-sweep curves (readout vs intervention strength). Monotonic = controllable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, rows in sweeps.items():
        rows = sorted(rows, key=lambda r: r[xkey])
        ax.plot([r[xkey] for r in rows], [r[ykey] for r in rows], marker="o", ms=4, label=name)
    ax.axvline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("intervention strength α"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def steering_filmstrip(
    frames_by_alpha: dict[float, "torch.Tensor"], path: str | Path, frame_idx: int = -1
) -> Path:
    """One row of decoded frames across α values (a single timestep), to eyeball the steered edit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alphas = sorted(frames_by_alpha)
    imgs = [_to_hwc(frames_by_alpha[a])[frame_idx] for a in alphas]
    fig, axes = plt.subplots(1, len(alphas), figsize=(1.6 * len(alphas), 1.9))
    if len(alphas) == 1:
        axes = [axes]
    for ax, a, im in zip(axes, alphas, imgs):
        ax.imshow(im); ax.axis("off"); ax.set_title(f"α={a:g}", fontsize=8)
    fig.suptitle("Steered decode (α sweep)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def transport_mask_overlay(frames, M_a: np.ndarray, M_b: np.ndarray, path: str | Path,
                           n_show: int = 8) -> Path:
    """Sanity figure: source/target trajectory masks overlaid on the (upsampled) ball frames.

    ``frames`` is ``(T_frames, C, H, W)`` (GT target clip); ``M_a``, ``M_b`` are ``(T_tok, Hp, Wp)`` soft
    masks. Each temporal token t spans frames 2t,2t+1, so we overlay token t on frame 2t. Red = source
    mask M_a, blue = target mask M_b — they should sit on the ball and its forward-simulated path.
    """
    import torch  # local: keep module import-light

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frames, torch.Tensor):
        fr = frames.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    else:
        fr = np.asarray(frames)
        if fr.ndim == 4 and fr.shape[1] in (1, 3):  # (T,C,H,W) -> (T,H,W,C)
            fr = np.transpose(fr, (0, 2, 3, 1))
    T_tok = M_a.shape[0]
    t = min(n_show, T_tok)
    fig, axes = plt.subplots(1, t, figsize=(1.7 * t, 1.9))
    if t == 1:
        axes = [axes]
    H, W = fr.shape[1], fr.shape[2]
    for i in range(t):
        ax = axes[i]
        ax.imshow(fr[min(2 * i, fr.shape[0] - 1)])
        for M, color in ((M_a, "Reds"), (M_b, "Blues")):
            up = np.kron(M[i], np.ones((H // M.shape[1], W // M.shape[2])))
            ax.imshow(up, cmap=color, alpha=0.45 * (up / (up.max() + 1e-9)), vmin=0, vmax=1)
        ax.axis("off"); ax.set_title(f"t={i}", fontsize=8)
    fig.suptitle("transport masks: source=red, target=blue", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def cka_heatmap(matrix: np.ndarray, labels: list[str], path: str | Path, title: str = "CKA") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def panel_video(
    original: torch.Tensor, recon: torch.Tensor, path: str | Path, fps: int = 8
) -> Path:
    """Side-by-side original|reconstruction|error mp4."""
    if recon.shape[-1] != original.shape[-1]:
        recon = torch.nn.functional.interpolate(recon, size=original.shape[-2:], mode="bilinear",
                                                align_corners=False)
    err = (original - recon).abs().clamp(0, 1)
    panel = torch.cat([original, recon, err], dim=-1)  # concat along width
    return save_video(panel, path, fps=fps)


def save_reconstruction_video(recon: torch.Tensor, path: str | Path, fps: int = 8) -> Path:
    return save_video(recon, path, fps=fps)
