"""Video and tensor I/O helpers.

Tensors follow the convention ``(T, C, H, W)`` with float values in ``[0, 1]`` unless noted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``(T, C, H, W)`` float [0,1] -> ``(T, H, W, C)`` uint8."""
    arr = frames.detach().cpu().float().clamp(0, 1).numpy()
    arr = np.transpose(arr, (0, 2, 3, 1))
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_video(frames: torch.Tensor, path: str | Path, fps: int = 8) -> Path:
    """Write a ``(T, C, H, W)`` float tensor to an mp4 (falls back to gif if ffmpeg absent)."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_uint8(frames)
    try:
        imageio.mimwrite(str(path), list(data), fps=fps, codec="libx264", quality=8)
    except Exception:
        gif_path = path.with_suffix(".gif")
        imageio.mimwrite(str(gif_path), list(data), fps=fps)
        return gif_path
    return path


def load_video(
    path: str | Path,
    num_frames: int | None = None,
    image_size: int | None = None,
) -> torch.Tensor:
    """Read a video file into a ``(T, C, H, W)`` float tensor in ``[0, 1]``.

    Uniformly samples ``num_frames`` if given; resizes to ``image_size`` if given.
    """
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    arr = np.stack(frames, axis=0)  # (T, H, W, C)
    if arr.ndim == 3:  # grayscale
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr[..., :3]
    tensor = torch.from_numpy(arr).float().div(255.0).permute(0, 3, 1, 2)  # (T, C, H, W)
    if num_frames is not None:
        tensor = sample_frames(tensor, num_frames)
    if image_size is not None:
        tensor = torch.nn.functional.interpolate(
            tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return tensor


def sample_frames(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Uniformly sample (or repeat-pad) ``num_frames`` along time from ``(T, C, H, W)``."""
    t = frames.shape[0]
    if t == num_frames:
        return frames
    idx = torch.linspace(0, max(t - 1, 0), num_frames).round().long().clamp(0, t - 1)
    return frames[idx]


def save_frames_grid(frames: torch.Tensor, path: str | Path, ncols: int = 8) -> Path:
    """Save a contact-sheet PNG of frames ``(T, C, H, W)``."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_uint8(frames)  # (T, H, W, C)
    t, h, w, c = data.shape
    ncols = min(ncols, t)
    nrows = (t + ncols - 1) // ncols
    canvas = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
    for i in range(t):
        r, col = divmod(i, ncols)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = data[i]
    imageio.imwrite(str(path), canvas)
    return path
