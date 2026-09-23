"""Inverse model for the paddle strike: desired outcome speed -> paddle action.

This is the module the controllability claim rests on. Given a ball arriving at ``v_in`` and a
commanded post-contact velocity ``v*``, it returns the paddle speed ``v_p`` that achieves it -- and
the certificate then measures whether the simulator actually delivers ``v*`` within tolerance.

**Nothing here is derived from first principles.** The textbook two-body law

    v_out = (m_b - e*m_p)/(m_b + m_p) * v_in  +  m_p*(1+e)/(m_b + m_p) * v_p

is a good *description* -- the measured map is linear to ~0.3% of range -- but its coefficients are
off by a few percent because MuJoCo's compliant contact is not an ideal impulsive collision (the
predicted alpha is 1.851 against a measured 1.888). So both the forward coefficients and the inverse
are FIT from a simulated sweep. Two inverses are provided, and the certificate reports both:

* ``linear``    -- invert the fitted linear forward map. Two parameters, no capacity to overfit.
* ``quadratic`` -- least-squares fit of ``v_p`` directly as a quadratic in ``(v*, v_in)``. Absorbs the
  small contact-compliance curvature. Fit on training ``v_in`` values, scored on held-out ones.

The quadratic fits the inverse *directly* rather than inverting a fitted forward model, so no
root-finding or Newton step is needed at command time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def sweep_strikes(
    gen: Any,
    v_in_values: np.ndarray | list[float],
    v_p_values: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    """Roll out the full (v_in x v_p) grid and return MEASURED outcomes.

    Uses ``render=False``, so this never touches EGL or a GPU -- the whole controllability
    certificate is a CPU job.
    """
    keys = ("v_in_world", "v_out_world", "v_out_img", "contact_frame", "n_touches",
            "post_frames_visible", "v_out_world_std", "v_p_actual", "gap_frames", "post_start_x",
            "margin_after_pre", "margin_before_post")
    cols: dict[str, list] = {k: [] for k in keys}
    v_p_l = []
    for v_in in np.asarray(v_in_values, dtype=float):
        for v_p in np.asarray(v_p_values, dtype=float):
            r = gen.simulate(float(v_in), float(v_p), render=False)
            v_p_l.append(float(v_p))
            for k in keys:
                cols[k].append(r[k])
    return {
        "v_in": np.asarray(cols["v_in_world"]), "v_p": np.asarray(v_p_l),
        "v_out": np.asarray(cols["v_out_world"]), "v_out_img": np.asarray(cols["v_out_img"]),
        "contact_frame": np.asarray(cols["contact_frame"], dtype=float),
        "n_touches": np.asarray(cols["n_touches"], dtype=int),
        "post_frames_visible": np.asarray(cols["post_frames_visible"], dtype=int),
        "v_out_std": np.asarray(cols["v_out_world_std"]),
        "v_p_actual": np.asarray(cols["v_p_actual"]),
        "gap_frames": np.asarray(cols["gap_frames"]),
        "post_start_x": np.asarray(cols["post_start_x"]),
        "margin_after_pre": np.asarray(cols["margin_after_pre"]),
        "margin_before_post": np.asarray(cols["margin_before_post"]),
    }


def _design(v_out: np.ndarray, v_in: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(v_out), v_out, v_in]
    if order >= 2:
        cols += [v_out * v_out, v_out * v_in, v_in * v_in]
    return np.stack(cols, axis=1)


@dataclass
class StrikeInverse:
    """Fitted forward + inverse strike model.

    ``alpha``/``beta`` are the fitted forward coefficients ``v_out = alpha*v_p + beta*v_in + gamma``;
    ``coef_lin``/``coef_quad`` are the directly-fitted inverse maps.
    """

    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    coef_quad: np.ndarray | None = None
    forward_max_resid: float = float("nan")
    forward_range: float = float("nan")
    e_effective: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)

    # -- fitting -----------------------------------------------------------------------------------
    @classmethod
    def fit(cls, sweep: dict[str, np.ndarray], ball_mass: float, paddle_mass: float) -> StrikeInverse:
        v_in, v_p, v_out = sweep["v_in"], sweep["v_p"], sweep["v_out"]
        # forward: v_out = alpha*v_p + beta*v_in + gamma
        X = np.stack([v_p, v_in, np.ones_like(v_p)], axis=1)
        coef, *_ = np.linalg.lstsq(X, v_out, rcond=None)
        resid = np.abs(X @ coef - v_out)
        rng = float(v_out.max() - v_out.min())
        # Invert alpha = m_p(1+e)/(m_b+m_p) for the restitution the fitted map implies.
        #
        # DIAGNOSTIC ONLY -- do not treat this as the contact restitution. It assumes the two-body law,
        # and a servo-clamped striker obeys the moving-wall law instead (alpha = 1+e), in which case
        # this over-reports e: it returns 0.947 where the contact's true e is 0.902, and for a LIGHT
        # striker it returns values above 1, which no restitution can be. That impossible output is the
        # signal the assumed law is wrong, not that the fit failed. Measure e directly by striking a
        # clamped stationary striker (v_out = -e*v_in); see experiments/threads/paddle-robotics/01_data/hardware_specs.py (E7).
        e_eff = float(coef[0] * (ball_mass + paddle_mass) / paddle_mass - 1.0)
        # inverse: v_p = quadratic(v_out, v_in)
        A = _design(v_out, v_in, order=2)
        cq, *_ = np.linalg.lstsq(A, v_p, rcond=None)
        return cls(alpha=float(coef[0]), beta=float(coef[1]), gamma=float(coef[2]),
                   coef_quad=cq, forward_max_resid=float(resid.max()), forward_range=rng,
                   e_effective=e_eff,
                   meta={"n_samples": int(len(v_out)), "ball_mass": ball_mass,
                         "paddle_mass": paddle_mass})

    @classmethod
    def fit_constrained(cls, sweep: dict[str, np.ndarray], ball_mass: float,
                        paddle_mass: float) -> StrikeInverse:
        """Fit with ``alpha + beta = 1`` and ``gamma = 0`` IMPOSED, as the physics requires.

        A ball and striker moving at a common velocity ``u`` cannot collide, so any correct law must
        return ``v_out = u`` there: ``alpha*u + beta*u + gamma = u`` for every ``u``, forcing
        ``alpha + beta = 1`` and ``gamma = 0``. The whole law is therefore ONE parameter,

            v_out - v_in = alpha * (v_p - v_in)

        which is a regression through the origin on the relative velocity -- exactly the statement that
        a strike acts on the approach velocity and nothing else.

        Use this instead of :meth:`fit` whenever the sweep may contain rollouts the unconstrained fit
        would be dragged by. Three free parameters can absorb contaminated samples into a law that fits
        the data and is still physically impossible: the dynamically-actuated arm's export fitted
        ``alpha=1.5725, beta=-0.9132``, i.e. ``alpha+beta=0.659``, which predicts that two bodies moving
        together at 1 m/s emerge at 0.659 m/s. Constraining removes that failure mode by construction,
        and the residual then becomes an honest diagnostic: if it is large, the data are bad, and you
        find out instead of getting a confident wrong answer.
        """
        v_in, v_p, v_out = sweep["v_in"], sweep["v_p"], sweep["v_out"]
        x = np.asarray(v_p) - np.asarray(v_in)
        y = np.asarray(v_out) - np.asarray(v_in)
        alpha = float(x @ y / (x @ x))
        resid = np.abs(alpha * x - y)
        rng = float(np.max(v_out) - np.min(v_out))
        # e follows from the moving-wall law alpha = 1+e; with alpha+beta=1 this is the same as -beta
        e_eff = alpha - 1.0
        # quadratic inverse of the SAME constrained model, for API parity with fit()
        A = _design(np.asarray(v_out), np.asarray(v_in), order=2)
        cq, *_ = np.linalg.lstsq(A, np.asarray(v_p), rcond=None)
        return cls(alpha=alpha, beta=1.0 - alpha, gamma=0.0, coef_quad=cq,
                   forward_max_resid=float(resid.max()), forward_range=rng, e_effective=e_eff,
                   meta={"n_samples": int(len(y)), "ball_mass": ball_mass,
                         "paddle_mass": paddle_mass, "constrained": True})

    # -- use ---------------------------------------------------------------------------------------
    def forward(self, v_in: float | np.ndarray, v_p: float | np.ndarray) -> np.ndarray:
        return self.alpha * np.asarray(v_p) + self.beta * np.asarray(v_in) + self.gamma

    def action_for(self, v_in: float | np.ndarray, v_target: float | np.ndarray,
                   mode: str = "quadratic") -> np.ndarray:
        """Paddle speed that should produce ``v_target`` (signed; leftward is negative)."""
        v_in = np.asarray(v_in, dtype=float)
        v_target = np.asarray(v_target, dtype=float)
        if mode == "linear":
            return (v_target - self.beta * v_in - self.gamma) / self.alpha
        if mode == "quadratic":
            if self.coef_quad is None:
                raise RuntimeError("quadratic inverse not fitted")
            return _design(np.atleast_1d(v_target), np.atleast_1d(v_in), order=2) @ self.coef_quad
        raise ValueError(f"unknown mode '{mode}'")

    # -- io ----------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
                "coef_quad": None if self.coef_quad is None else list(map(float, self.coef_quad)),
                "forward_max_resid": self.forward_max_resid, "forward_range": self.forward_range,
                "forward_resid_pct_of_range": (100.0 * self.forward_max_resid / self.forward_range
                                               if self.forward_range else float("nan")),
                "e_effective": self.e_effective, "meta": self.meta}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> StrikeInverse:
        d = json.loads(Path(path).read_text())
        cq = d.get("coef_quad")
        return cls(alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"],
                   coef_quad=None if cq is None else np.asarray(cq, dtype=float),
                   forward_max_resid=d.get("forward_max_resid", float("nan")),
                   forward_range=d.get("forward_range", float("nan")),
                   e_effective=d.get("e_effective", float("nan")), meta=d.get("meta", {}))
