"""Command features and pair enumeration for the velocity x spin CROSSTALK experiment.

The velocity half of this experiment reuses :mod:`~src.analysis.velocity_ops` unchanged
(:func:`~src.analysis.velocity_ops.command_features_pos`, the 27-dim position-aware feature set that
won on ``rolling_ball3d``). This module adds the two things that scene did not need: a command feature
set for the SPIN axis, and the pair enumeration that the factorial design makes possible.

**Why the pair enumeration is the whole point.** Every earlier operator in this project was fitted on
pairs that differed in the one quantity the scene varied, because that was the only quantity there was.
Here a scene contains a complete ``n_vel x n_spin`` grid, so pairs come in three kinds:

  * ``vel`` pairs   -- same spin cell, different velocity cell  => ``d_omega`` is EXACTLY zero
  * ``spin`` pairs  -- same velocity cell, different spin cell  => ``d_v`` is EXACTLY zero
  * ``both`` pairs  -- both cells differ                        => the diagonal of the square

Fitting ``W_V`` on ``vel`` pairs alone and ``W_S`` on ``spin`` pairs alone means neither operator has
ever seen the other quantity move. That is what makes the crosstalk measurement meaningful: if steering
with ``W_V`` perturbs omega at test time, it cannot be because omega was entangled into the fit — it has
to be a property of the latent geometry. Fitting on ``both`` pairs would quietly destroy that argument.

The ``both`` pairs are still generated, for the separate question of whether the composed edit
``W_V + W_S`` matches an operator fitted jointly on the diagonal.
"""

from __future__ import annotations

import numpy as np

SPIN_FEATURE_DIM = 12


def spin_command_features(wa: float, wb: float, phi0: float) -> np.ndarray:
    """Command feature vector ``(12,)`` for a spin edit ``omega_a -> omega_b``, without ``H_b``.

    Columns: ``[1, wb, wa, dw, |wb|, |wa|, sign(wb), sign(wa), cos(phi0), sin(phi0),
    dw*cos(phi0), dw*sin(phi0)]``.

    The design mirrors :func:`~src.analysis.velocity_ops.command_features`: a bias, the raw commands,
    their difference, and magnitude/direction split out separately so a linear map can express
    behaviour that scales with rate but flips with handedness. Two additions are specific to rotation:

    * ``sign(w)`` is carried explicitly because the dataset balances CW against CCW within every scene,
      so an operator that only sees ``|omega|`` scores zero. This is the same trap the robotics rig fell
      into when a fabricated ``sign(y)*|.|`` fallback made a lateral result look real.
    * ``phi0`` (the marker's frame-0 azimuth) enters as ``cos``/``sin`` and multiplied by ``dw``. Spin,
      unlike velocity, has a PHASE: the same ``d_omega`` applied to a clip whose marker starts at the
      near side and one starting at the far side must produce different pixels, so the edit cannot be a
      function of the rate alone. ``phi0`` is readable from clip ``a`` at test time (it is where the
      marker is in frame 0), so this stays a command-only operator with no access to ``H_b``.
    """
    wa = float(wa); wb = float(wb)
    dw = wb - wa
    c, s = float(np.cos(phi0)), float(np.sin(phi0))
    return np.array([1.0, wb, wa, dw, abs(wb), abs(wa), np.sign(wb), np.sign(wa),
                     c, s, dw * c, dw * s], dtype=np.float64)


# Everything below reads the STATE vector or the sample id, never ``meta``. The latent cache written by
# ``extract_latents`` persists id / layers / frames / grid / state / state_keys / category and DROPS the
# generator's ``meta`` dict, so any quantity an operator needs must live in a state column or be
# recoverable from the id. Both do: ``obj0_omega`` carries the spin rate, ``obj0_theta`` carries the
# marker azimuth per frame (so row 0 is phi0), and the id encodes ``scene`` and ``rank``.
def clip_spin(sample: dict) -> float:
    """Signed spin rate ``omega`` (rad/frame) of a clip, from its ``obj0_omega`` state column.

    The column is constant over the clip (constant angular velocity by construction), so the mean
    recovers it exactly. The scalar analogue of :func:`~src.analysis.velocity_ops.clip_velocity`.
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    return float(state[:, keys.index("obj0_omega")].mean())


def clip_phi0(sample: dict) -> float:
    """Marker azimuth at frame 0 (rad), from row 0 of the ``obj0_theta`` state column.

    Shared across a scene by construction, so within a scene this is a constant — but it is read
    per-clip rather than assumed, so a change to the scene's frame-0 convention cannot silently
    de-synchronise the spin operator's phase term.
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    return float(state[0, keys.index("obj0_theta")])


def cell_of(sample_id: str, n_spin: int) -> tuple[int, int]:
    """``(vel_index, spin_index)`` of a clip, decoded from its ``sceneNNNNN_vRANK`` id.

    The generator lays the factorial out as ``rank = vel_index * n_spin + spin_index``, so the cell is
    ``divmod(rank, n_spin)``. Returns ``(-1, -1)`` for an id that is not a scene clip.
    """
    from .velocity_ops import scene_rank

    sr = scene_rank(sample_id)
    if sr is None:
        return (-1, -1)
    return divmod(sr[1], int(n_spin))


def enumerate_pairs(cells: dict[tuple[int, int], int]) -> dict[str, list[tuple[int, int]]]:
    """Split a scene's factorial into ``vel`` / ``spin`` / ``both`` ordered pairs of dataset indices.

    ``cells`` maps ``(vel_index, spin_index) -> dataset index``. Pairs are ordered and both directions
    are emitted, so an operator is fitted on ``a->b`` and ``b->a`` alike and cannot acquire a sign
    preference from the enumeration order.
    """
    out: dict[str, list[tuple[int, int]]] = {"vel": [], "spin": [], "both": []}
    keys = sorted(cells)
    for ka in keys:
        for kb in keys:
            if ka == kb:
                continue
            same_v, same_s = ka[0] == kb[0], ka[1] == kb[1]
            kind = "spin" if same_v else "vel" if same_s else "both"
            out[kind].append((cells[ka], cells[kb]))
    return out


def commutation_square(cells: dict[tuple[int, int], int], vi_a: int, si_a: int,
                       vi_b: int, si_b: int) -> dict[str, int]:
    """The four REAL clips at the corners of a commutation square, as dataset indices.

    ``base`` = (vi_a, si_a), ``vel_only`` = (vi_b, si_a), ``spin_only`` = (vi_a, si_b),
    ``both`` = (vi_b, si_b). Because the factorial is complete, every corner is a clip that was actually
    simulated and rendered — the target of a two-quantity steer is a ground truth, not an extrapolation.
    """
    return {"base": cells[(vi_a, si_a)], "vel_only": cells[(vi_b, si_a)],
            "spin_only": cells[(vi_a, si_b)], "both": cells[(vi_b, si_b)]}


__all__ = ["SPIN_FEATURE_DIM", "spin_command_features", "clip_spin", "clip_phi0", "cell_of",
           "enumerate_pairs", "commutation_square"]
