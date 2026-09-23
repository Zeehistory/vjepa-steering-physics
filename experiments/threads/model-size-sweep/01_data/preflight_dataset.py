

# --- repo-root shim: make ``src`` importable however this script is invoked ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

#!/usr/bin/env python
"""Phase-0 gate: is a generated dataset actually diverse enough to draw conclusions from?

This exists because a headline result in this project was invalidated after the fact when the
dataset behind it turned out to have ONE distinct geometry across 100 scenes. Everything
downstream was measuring a degenerate set. That must never cost GPU hours again, so this runs on
CPU, before any extraction is submitted, and `j_extract.sh` refuses to start without a passing
`preflight.json`.

Checks:
  scene diversity   n_distinct_scenes >= 0.9 * expected
  appearance        each of background / colour / radius / position is asserted against the
                    dataset's DECLARED contract: factors named in --fixed_appearance must be
                    exactly constant, all others must show >= 8 distinct values. Checking both
                    directions matters -- the base scenarios (scene_accel2d, scene_velocity2d)
                    deliberately pin the background so appearance is controlled, so demanding
                    diversity unconditionally flags the controls as broken, while demanding
                    nothing lets a silently-collapsed factor through.
  frame hashes      >= 95% distinct frame-0 hashes PER SCENE (frame 0 is identical across the
                    ranks of a scene by design -- that is the same-scene contract)
  label spread      std > 0.15 * range; no single label value > 20% of clips;
                    for 2-D labels both singular values > 0.2x the larger (rules out a
                    degenerate 1-D label manifold masquerading as 2-D)
  split disjoint    train and test share no frame-0 hash. Compared on CONTENT, not scene ids:
                    both splits number scenes from scene00000 and differ only by generator seed,
                    so an id-based test always reports a false collision.

Also writes a contact sheet of first frames so a human can eyeball the set in one glance.
Exit 0 = pass.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf  # noqa: E402

from src.data.dataset_registry import build_dataset  # noqa: E402


_RANK_RE = re.compile(r"_[vr]\d+$")


def scene_of(sample_id: str) -> str:
    """Scene key for a clip id. Ids look like `scene00007_v3` (velocity/accel families) or
    `..._r3`; the trailing rank index is what varies WITHIN a scene."""
    return _RANK_RE.sub("", str(sample_id))


def load_split(data_cfg: str, split: str, num_clips: int, seed: int, image_size: int, frames: int):
    cfg = OmegaConf.load(f"configs/data/{data_cfg}.yaml")
    cfg.split, cfg.num_clips, cfg.seed = split, num_clips, seed
    return build_dataset(cfg, encoder_image_size=image_size, encoder_frames=frames)


def collect(ds, n: int, label_keys: list[str] | None):
    """Pull per-clip label vectors, frame-0 hashes, scene ids and appearance proxies."""
    ids, hashes, labels, appear = [], [], [], []
    keys = None
    for i in range(min(n, len(ds))):
        s = ds[i]
        ids.append(s["id"])
        f0 = s["frames"][0]
        hashes.append(hashlib.md5(np.ascontiguousarray(f0.numpy()).tobytes()).hexdigest())
        keys = list(s["state_keys"])
        st = s["state"].numpy()
        if label_keys:
            idx = [keys.index(k) for k in label_keys if k in keys]
            labels.append(np.nan_to_num(st[:, idx]).mean(0) if idx else np.zeros(1))
        # Appearance proxies read straight off pixels, so this stays generator-agnostic:
        # corner patch ~ background shade, mean RGB over the ball's own pixels ~ ball colour,
        # dark-pixel count ~ radius. Colour must be an RGB triple -- a scalar like img.min()
        # is constant across a dark-blue..dark-red sweep and silently reports "1 colour".
        img = f0.numpy()
        dark = img.mean(0) < 0.5
        ball_rgb = tuple(np.round(img[:, dark].mean(1), 4)) if dark.any() else (0.0, 0.0, 0.0)
        ys, xs = np.nonzero(dark)
        start_pos = (round(float(xs.mean()), 2), round(float(ys.mean()), 2)) if dark.any() else (0.0, 0.0)
        appear.append((
            round(float(img[:, :8, :8].mean()), 5),
            ball_rgb,
            round(float(dark.sum()), 1),
            start_pos,
        ))
    return ids, hashes, np.asarray(labels, dtype=float) if labels else None, appear, keys


def check_label_spread(Y: np.ndarray) -> dict:
    out = {"n": int(len(Y)), "dims": int(Y.shape[1]) if Y.ndim > 1 else 1}
    Y = Y.reshape(len(Y), -1)
    rng = Y.max(0) - Y.min(0)
    std = Y.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rng > 0, std / np.maximum(rng, 1e-12), 0.0)
    out["std_over_range"] = [round(float(v), 4) for v in ratio]
    out["spread_ok"] = bool(np.all(ratio > 0.15))

    # No single label value may dominate.
    top = []
    for j in range(Y.shape[1]):
        c = Counter(np.round(Y[:, j], 6))
        top.append(c.most_common(1)[0][1] / len(Y))
    out["max_value_frequency"] = [round(float(v), 4) for v in top]
    out["no_dominant_value"] = bool(np.all(np.asarray(top) <= 0.20))

    # For >=2-D labels, both directions must carry real variance.
    if Y.shape[1] >= 2:
        sv = np.linalg.svd(Y - Y.mean(0), compute_uv=False)
        out["singular_values"] = [round(float(v), 4) for v in sv[:2]]
        out["rank_ok"] = bool(sv[1] > 0.2 * sv[0])
    else:
        out["rank_ok"] = True
    out["pass"] = bool(out["spread_ok"] and out["no_dominant_value"] and out["rank_ok"])
    return out


def contact_sheet(ds, path: Path, n: int = 16):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # a missing plot backend must not fail the gate
        print(f"  (contact sheet skipped: {e})")
        return
    k = min(n, len(ds))
    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < k:
            s = ds[i * max(1, len(ds) // k)]
            ax.imshow(s["frames"][0].permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(str(s["id"])[:22], fontsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  contact sheet -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", required=True, help="name under configs/data/")
    ap.add_argument("--label_keys", default="", help="comma-separated state keys that are the target")
    ap.add_argument("--n_train", type=int, default=256, help="clips to sample from train")
    ap.add_argument("--n_test", type=int, default=256)
    ap.add_argument("--train_clips", type=int, default=4000)
    ap.add_argument("--test_clips", type=int, default=800)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--test_seed", type=int, default=2)
    ap.add_argument("--clips_per_scene", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--fixed_appearance", default="",
                   help="comma-separated appearance factors this dataset INTENDS to hold constant "
                        "(background,color,radius,position). Asserted constant; the rest must vary.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    label_keys = [k for k in args.label_keys.split(",") if k]
    fixed_factors = {f.strip() for f in args.fixed_appearance.split(",") if f.strip()}
    res = {"data_cfg": args.data_cfg, "label_keys": label_keys,
           "fixed_appearance": sorted(fixed_factors), "checks": {}}
    print(f"[preflight] {args.data_cfg}  labels={label_keys or '(none)'}")

    splits = {}
    for split, n, clips, seed in [("train", args.n_train, args.train_clips, args.train_seed),
                                  ("test", args.n_test, args.test_clips, args.test_seed)]:
        ds = load_split(args.data_cfg, split, clips, seed, args.image_size, args.frames)
        ids, hashes, Y, appear, keys = collect(ds, n, label_keys)
        scenes = {scene_of(i) for i in ids}
        expected = max(1, len(ids) // args.clips_per_scene)
        c = {
            "n_clips_sampled": len(ids),
            "n_distinct_scenes": len(scenes),
            "expected_scenes": expected,
            "scene_diversity_ok": bool(len(scenes) >= 0.9 * expected),
            "n_distinct_frame0_hashes": len(set(hashes)),
            # Frame 0 is IDENTICAL across the ranks of a scene by design (the whole point of the
            # same-scene contract is that only the target quantity differs), so the ceiling here
            # is the scene count, not the clip count.
            "frame_hash_ok": bool(len(set(hashes)) >= 0.95 * len(scenes)),
            "distinct_background": len({a[0] for a in appear}),
            "distinct_ball_color": len({a[1] for a in appear}),
            "distinct_ball_area": len({a[2] for a in appear}),
            "distinct_start_pos": len({a[3] for a in appear}),
        }
        # Assert the dataset's DECLARED appearance contract in both directions. The base scenarios
        # (scene_accel2d, scene_velocity2d, ...) deliberately pin the background so appearance is
        # controlled and only the physics varies; the `_mixed` variants deliberately randomize it to
        # test transfer. Demanding diversity unconditionally flags the controls as broken, and
        # demanding nothing lets a silently-collapsed factor through -- so check both.
        c["appearance_contract"] = {}
        for factor, key in [("background", "distinct_background"), ("color", "distinct_ball_color"),
                            ("radius", "distinct_ball_area"), ("position", "distinct_start_pos")]:
            n_distinct = c[key]
            if factor in fixed_factors:
                ok = n_distinct == 1
                c["appearance_contract"][factor] = {"expect": "fixed", "n": n_distinct, "pass": ok}
            else:
                ok = n_distinct >= 8
                c["appearance_contract"][factor] = {"expect": "varied", "n": n_distinct, "pass": ok}
        c["appearance_ok"] = all(v["pass"] for v in c["appearance_contract"].values())
        if Y is not None and Y.size:
            c["label"] = check_label_spread(Y)
        c["pass"] = bool(c["scene_diversity_ok"] and c["frame_hash_ok"] and c["appearance_ok"]
                         and c.get("label", {"pass": True})["pass"])
        res["checks"][split] = c
        splits[split] = set(hashes)
        print(f"  {split:5s}: {len(ids)} clips, {len(scenes)}/{expected} scenes, "
              f"{len(set(hashes))} distinct frame0, appearance "
              + "/".join(f"{k}={v['n']}({v['expect'][:3]}{'' if v['pass'] else '!'})"
                         for k, v in c["appearance_contract"].items())
              + f" -> {'PASS' if c['pass'] else 'FAIL'}")
        if "label" in c:
            L = c["label"]
            print(f"         label std/range={L['std_over_range']} maxfreq={L['max_value_frequency']}"
                  f" sv={L.get('singular_values')} -> {'PASS' if L['pass'] else 'FAIL'}")
        if split == "test":
            contact_sheet(ds, Path(args.out).with_suffix(".contact.png"))

    # Disjointness must be tested on CONTENT, not scene ids: both splits number their scenes from
    # scene00000 and are separated only by the generator seed, so id sets always collide and would
    # report a false failure. Frame-0 hashes compare the actual rendered scenes.
    overlap = splits["train"] & splits["test"]
    res["checks"]["split_disjoint"] = {"basis": "frame0_hash",
                                       "n_overlapping_scenes": len(overlap),
                                       "examples": sorted(overlap)[:5], "pass": not overlap}
    print(f"  split disjoint (by frame0 content): {len(overlap)} overlapping scenes -> "
          f"{'PASS' if not overlap else 'FAIL'}")

    res["pass"] = all(v["pass"] for v in res["checks"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n  PREFLIGHT: {'PASS' if res['pass'] else 'FAIL'}  -> {args.out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
