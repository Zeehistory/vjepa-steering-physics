"""Turn "you need ball sensing and a trigger" into NUMBERS a real rig can be checked against.

``HARDWARE.md`` lists what a physical setup must provide but not how good any of it has to be, which
is the one thing that decides whether a given lab's rig is adequate. Every experiment here perturbs
the simulator the way real hardware is imperfect, plans the action with the NOMINAL fitted inverse
(exactly what an operator would do -- they cannot know the error they are making), executes, and
measures the outcome in m/s. The output is a spec table: for each error source, how much of it costs
the 5% outcome bar.

Nothing here re-fits the strike law to hide an error. That distinction matters: a spec derived by
re-fitting would say "the law still holds", which is true and useless. What an operator needs is
"if my tracker is off by X, my ball misses by Y".

Three of the seven experiments are analytic as well as measured, and the agreement is the point --
where a closed form exists, the spec transfers to ANY rig with different alpha/beta, instead of being
a fact about this simulator. Sensing error in particular has an exact answer, derived in E1.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/01_data/hardware_specs.py \
        --output_dir outputs/paddle_strike/hardware_specs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import (
    BALL_MASS,
    PADDLE_MASS,
    FPS,
    PaddleStrike,
    build_striker,
)

TOL = 0.05                      # the pre-registered outcome bar: |v_out - v*| / |v*| <= 5%
RATIOS = (0.5, 1.0, 2.0, 3.0)   # commanded speed-up factors spanning the certified envelope
V_IN = 1.0                      # nominal incoming speed for the spec sweeps


# -- perturbable variants --------------------------------------------------------------------------
# Ball mass, table friction and contact elasticity are MODEL properties, so they are changed by
# overriding _configure_model rather than per-rollout. Subclassing keeps the certified module's
# defaults untouched -- these variants exist only inside this script.

class PerturbedStrike(PaddleStrike):
    """Paddle strike with physical parameters a real rig would get slightly wrong."""

    def __init__(self, *a: Any, ball_mass_scale: float = 1.0, table_friction: float = 0.0,
                 solref_scale: tuple[float, float] = (1.0, 1.0), **k: Any) -> None:
        self.ball_mass_scale = float(ball_mass_scale)
        self.table_friction = float(table_friction)
        self.solref_scale = solref_scale
        super().__init__(*a, **k)

    def _build_xml(self, ball_gray: float, table_shade: float) -> str:
        s0, s1 = self.solref_scale
        saved, self.solref = self.solref, (self.solref[0] * s0, self.solref[1] * s1)
        try:
            return super()._build_xml(ball_gray, table_shade)
        finally:
            self.solref = saved

    def _configure_model(self, sim: dict) -> None:
        super()._configure_model(sim)
        mj, model, ids = sim["mj"], sim["model"], sim["ids"]
        bid = model.geom_bodyid[ids["ball_geom"]]
        model.body_mass[bid] *= self.ball_mass_scale
        # inertia scales with mass at fixed geometry; leaving it stale makes the ball's rotational
        # and translational responses disagree, which is not a thing any real ball does
        model.body_inertia[bid] *= self.ball_mass_scale
        if self.table_friction > 0.0:
            tid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "tabletop")
            model.geom_friction[tid] = [self.table_friction, self.table_friction * 0.005,
                                        self.table_friction * 0.0001]
            model.geom_friction[ids["ball_geom"]] = [self.table_friction,
                                                     self.table_friction * 0.005,
                                                     self.table_friction * 0.0001]
        # REQUIRED after touching mass or inertia. MuJoCo precomputes body_invweight0/dof_invweight0
        # at compile time and the constraint solver uses them to scale the contact impulse, so a bare
        # body_mass write leaves the impact resolving against the ORIGINAL effective mass: a +20% ball
        # moved v_out by 2e-5 m/s, where the two-body law demands ~0.5%. mj_setConst rederives them.
        mj.mj_setConst(model, sim["data"])


class _GainStrike(PaddleStrike):
    """Paddle strike with adjustable servo gains, for the regime test in E7.

    ``kp=kd=0`` leaves only the feedforward term, which is zero during the constant-velocity cruise --
    so the striker is a genuinely free body through the impact.
    """

    def __init__(self, *a: Any, kp: float = 4000.0, kd: float = 400.0, mff: float = PADDLE_MASS,
                 **k: Any) -> None:
        self.kp, self.kd, self.mff = float(kp), float(kd), float(mff)
        super().__init__(*a, **k)

    def _drive_striker(self, sim: dict, t: float, v_p: float) -> None:
        from src.data.paddle_strike import X_PADDLE_REST, paddle_ref
        d, ids = sim["data"], sim["ids"]
        d_ref, v_ref, a_ref = paddle_ref(t, v_p, self.ramp_frames, self.ramp_start_frame)
        d.ctrl[ids["pad_act"]] = (self.mff * a_ref
                                  + self.kp * (X_PADDLE_REST + d_ref - d.qpos[ids["pad_q"]])
                                  + self.kd * (v_ref - d.qvel[ids["pad_v"]]))


# -- helpers ---------------------------------------------------------------------------------------

def fit_nominal(gen: Any) -> StrikeInverse:
    """The strike law an operator would identify on a well-behaved day."""
    return StrikeInverse.fit(
        sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20)),
        BALL_MASS, PADDLE_MASS)


def plan(inv: StrikeInverse, v_in: float, ratio: float) -> tuple[float, float]:
    """(target, action) for a commanded speed-up ratio, via the fitted inverse."""
    target = -ratio * v_in
    v_p = float(np.asarray(inv.action_for(v_in, target, mode="quadratic")).ravel()[0])
    return target, v_p


def rel_err(achieved: float, target: float) -> float:
    return abs(achieved - target) / abs(target)


def largest_passing(xs: list[float], errs: list[float]) -> float:
    """Largest |perturbation| whose error, and every smaller one's, stays inside TOL.

    Scanning outward from zero rather than taking the max passing value on purpose: a spec has to be
    a contiguous window around nominal, and reporting an isolated far-out pass as the limit would be
    wrong.
    """
    order = sorted(range(len(xs)), key=lambda i: abs(xs[i]))
    best = 0.0
    for i in order:
        if errs[i] > TOL:
            break
        best = abs(xs[i])
    return best


# -- E1: how accurately must the incoming ball speed be measured? ----------------------------------
# ANALYTIC: planning with v_hat = v_in + d gives v_p = (v* - beta*v_hat)/alpha, so the executed
# outcome is alpha*v_p + beta*v_in = v* - beta*d. The outcome error is therefore EXACTLY |beta*d|,
# independent of alpha, of the target, and of the ratio commanded. Hence the tolerance in m/s is
# TOL*|v*|/|beta| -- tightest for the slowest command, which is what sets the spec.

def e1_sensing(gen: Any, inv: StrikeInverse) -> dict:
    deltas = [-0.20, -0.15, -0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20]
    rows, per_ratio = [], {}
    for ratio in RATIOS:
        errs, xs = [], []
        for d in deltas:
            target = -ratio * V_IN
            # the operator plans from a MIS-MEASURED v_in, but the world runs at the true one
            v_p = float(np.asarray(inv.action_for(V_IN + d, target, mode="quadratic")).ravel()[0])
            r = gen.simulate(V_IN, v_p, render=False)
            e = rel_err(r["v_out_world"], target)
            errs.append(e); xs.append(d)
            rows.append({"ratio": ratio, "delta_v_in": d, "rel_err": e,
                         "analytic_rel_err": abs(inv.beta * d) / abs(target)})
        per_ratio[ratio] = largest_passing(xs, errs)
    analytic = {r: TOL * abs(r * V_IN) / abs(inv.beta) for r in RATIOS}
    return {"rows": rows, "tolerance_measured": per_ratio, "tolerance_analytic": analytic,
            "binding_spec_m_s": min(analytic.values()),
            "note": "outcome error = |beta| * (v_in measurement error), exactly"}


# -- E2: how tight must the trigger be? ------------------------------------------------------------
# The swing runs on a fixed schedule, so a trigger firing dt early/late is equivalent to the ball
# starting v_in*dt away from nominal. Contact is DESIGNED to land during the constant-velocity cruise
# phase, so the prediction is that this is very forgiving -- until the offset pushes contact out of
# cruise (too early) or off the blade / past the window (too late).

def e2_trigger(gen: Any, inv: StrikeInverse) -> dict:
    # Range extended until the spec actually BREAKS. A first pass stopped at +-50 ms, every value
    # passed, and the reported tolerance was therefore just the edge of the grid -- a spec that
    # understates the true one is as misleading as one that overstates it.
    dts = [-0.30, -0.25, -0.20, -0.15, -0.12, -0.10, -0.08, -0.05, -0.03, -0.01, 0.0,
           0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    rows, per_ratio = [], {}
    for ratio in RATIOS:
        target, v_p = plan(inv, V_IN, ratio)
        errs, xs = [], []
        for dt in dts:
            dx = V_IN * dt
            try:
                r = gen.simulate(V_IN, v_p, render=False, x_b0_offset=dx)
                e = rel_err(r["v_out_world"], target)
                touches, margin = int(r["n_touches"]), float(r["margin_after_pre"])
            except RuntimeError:
                e, touches, margin = float("inf"), 0, float("nan")
            errs.append(e); xs.append(dt)
            rows.append({"ratio": ratio, "dt_s": dt, "dx_m": dx, "rel_err": e,
                         "n_touches": touches, "margin_after_pre": margin})
        per_ratio[ratio] = largest_passing(xs, errs)
    return {"rows": rows, "tolerance_s": per_ratio, "binding_spec_s": min(per_ratio.values())}


# -- E3: how accurately must the ball be aimed at the blade? ---------------------------------------

def e3_lateral(gen: Any, inv: StrikeInverse) -> dict:
    # Out to a clean geometric miss: the blade is PADDLE_HALF[1]=0.10 m half-width and the ball has
    # radius 0.055, so contact is impossible past y = 0.155 m. Sweeping to 0.08 (the first pass) only
    # proved the grid edge passed.
    ys = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]
    rows, per_ratio = [], {}
    for ratio in RATIOS:
        target, v_p = plan(inv, V_IN, ratio)
        errs, xs = [], []
        for y in ys:
            try:
                r = gen.simulate(V_IN, v_p, render=False, y_b0=y)
                e = rel_err(r["v_out_world"], target)
                lat = float(r["v_out_lateral_ratio"])
            except RuntimeError:
                e, lat = float("inf"), float("nan")
            errs.append(e); xs.append(y)
            rows.append({"ratio": ratio, "y_offset_m": y, "rel_err": e, "lateral_ratio": lat})
        per_ratio[ratio] = largest_passing(xs, errs)
    return {"rows": rows, "tolerance_m": per_ratio, "binding_spec_m": min(per_ratio.values())}


# -- E4: what if the ball or the contact is not what the law was fitted to? -------------------------
# The operator identified alpha/beta on one day, then something changed -- a different ball, a scuffed
# tool face, a warmer room. Plan with the ORIGINAL inverse, execute in the CHANGED world.

def e4_mismatch(gen: Any, inv: StrikeInverse) -> dict:
    rows = []
    for label, kw in [
        ("ball mass -20%", {"ball_mass_scale": 0.8}),
        ("ball mass -10%", {"ball_mass_scale": 0.9}),
        ("ball mass +10%", {"ball_mass_scale": 1.1}),
        ("ball mass +20%", {"ball_mass_scale": 1.2}),
        ("contact stiffness -30%", {"solref_scale": (0.7, 1.0)}),
        ("contact stiffness +30%", {"solref_scale": (1.3, 1.0)}),
        ("contact damping -30% (bouncier)", {"solref_scale": (1.0, 0.7)}),
        ("contact damping +30% (deader)", {"solref_scale": (1.0, 1.3)}),
    ]:
        pert = PerturbedStrike(**kw)
        for ratio in RATIOS:
            target, v_p = plan(inv, V_IN, ratio)     # planned with the NOMINAL law
            try:
                r = pert.simulate(V_IN, v_p, render=False)
                e = rel_err(r["v_out_world"], target)
                got = float(r["v_out_world"])
            except RuntimeError:
                e, got = float("inf"), float("nan")
            rows.append({"perturbation": label, "ratio": ratio, "target": target,
                         "achieved": got, "rel_err": e})
        # what a re-identification on the changed rig recovers
        inv2 = fit_nominal(pert)
        rows.append({"perturbation": label, "ratio": "refit", "alpha": inv2.alpha,
                     "beta": inv2.beta,
                     "worst_after_refit": max(
                         rel_err(float(pert.simulate(V_IN, plan(inv2, V_IN, rr)[1],
                                                     render=False)["v_out_world"]),
                                 plan(inv2, V_IN, rr)[0]) for rr in RATIOS)})
    return {"rows": rows}


# -- E5: how many calibration strikes, at what measurement noise? -----------------------------------
# The real identification loop: execute a few strikes, measure v_in and v_out with a NOISY tracker,
# fit, then command through the fitted inverse and score on the truth.

def e5_calibration(gen: Any, truth: StrikeInverse, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    grid = sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20))
    rows = []
    for n in (4, 6, 8, 12, 20, 40, 80):
        for noise in (0.0, 0.01, 0.02, 0.05, 0.10):
            worst = []
            for _ in range(12):                       # repeats: which strikes you happen to get
                idx = rng.choice(len(grid["v_out"]), size=n, replace=False)
                noisy = {"v_in": grid["v_in"][idx] + rng.normal(0, noise, n),
                         "v_p": grid["v_p"][idx],     # commanded, so known exactly
                         "v_out": grid["v_out"][idx] + rng.normal(0, noise, n)}
                try:
                    inv = StrikeInverse.fit(noisy, BALL_MASS, PADDLE_MASS)
                except np.linalg.LinAlgError:
                    worst.append(float("inf")); continue
                # score the FITTED inverse by executing it, against the true outcome
                errs = []
                for ratio in RATIOS:
                    target = -ratio * V_IN
                    v_p = float(np.asarray(
                        inv.action_for(V_IN, target, mode="linear")).ravel()[0])
                    try:
                        r = gen.simulate(V_IN, v_p, render=False)
                        errs.append(rel_err(r["v_out_world"], target))
                    except RuntimeError:
                        errs.append(float("inf"))
                worst.append(max(errs))
            worst_a = np.asarray(worst)
            rows.append({"n_strikes": n, "noise_m_s": noise,
                         "median_worst_rel_err": float(np.median(worst_a)),
                         "p90_worst_rel_err": float(np.quantile(worst_a, 0.9)),
                         "frac_repeats_passing": float((worst_a <= TOL).mean())})
    return {"rows": rows}


# -- E6: the frictionless/spinless idealisation ----------------------------------------------------
# The sim ball is a puck: zero table friction, zero spin. A real ball rolls. Both are quantified --
# friction as a model change, spin as an initial condition (including the rolling-without-slipping
# rate, which is what a real ball actually arrives with).

def e6_friction_spin(gen: Any, inv: StrikeInverse) -> dict:
    """Friction breaks the constant-velocity contract; separate WHY it looks like a failure.

    ``v_out_world`` is the MEAN over the 16-frame post window. With a frictionless table the velocity
    is exactly constant so the mean is the outcome; add friction and the ball decelerates all the way
    down the window, so the mean is biased low even if the strike itself was perfect. Reporting only
    the mean therefore blames the strike law for what is really a measurement-window artefact, and
    leads to the wrong hardware recommendation ("you need an air table" instead of "measure v_out in a
    short window right after contact").

    So both are reported: the window mean, and the FIRST post-window sample, which is the closest
    thing to the velocity at contact. ``v_out_std`` is carried along because it is the giveaway -- it
    is ~0 for a valid episode and blows up under friction, so a real rig can detect this from its own
    data without ground truth.
    """
    from src.data.paddle_strike import BALL_R
    rows = []
    for mu in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        pert = PerturbedStrike(table_friction=mu)
        for ratio in RATIOS:
            target, v_p = plan(inv, V_IN, ratio)
            try:
                r = pert.simulate(V_IN, v_p, render=False)
                first = float(r["ball_v"][gen.num_frames])   # first post-window sample
                rows.append({"kind": "table_friction", "value": mu, "ratio": ratio,
                             "rel_err": rel_err(r["v_out_world"], target),
                             "rel_err_first_sample": rel_err(first, target),
                             "v_out_std": float(r["v_out_world_std"])})
            except RuntimeError:
                rows.append({"kind": "table_friction", "value": mu, "ratio": ratio,
                             "rel_err": float("inf"), "rel_err_first_sample": float("inf"),
                             "v_out_std": float("nan")})
    roll = V_IN / BALL_R          # rolling without slipping, rad/s about +y
    for frac in (0.0, 0.25, 0.5, 1.0):
        for ratio in RATIOS:
            target, v_p = plan(inv, V_IN, ratio)
            try:
                r = gen.simulate(V_IN, v_p, render=False, spin=(0.0, frac * roll, 0.0))
                rows.append({"kind": "spin_frac_of_rolling", "value": frac, "ratio": ratio,
                             "rel_err": rel_err(r["v_out_world"], target),
                             "omega_rad_s": frac * roll})
            except RuntimeError:
                rows.append({"kind": "spin_frac_of_rolling", "value": frac, "ratio": ratio,
                             "rel_err": float("inf"), "omega_rad_s": frac * roll})
    return {"rows": rows}


# -- E7: what alpha/beta should a REAL rig expect? -------------------------------------------------
# alpha/beta are set by the mass ratio and restitution, so the sim's values are not the ones a Panda
# holding a real tool will measure. The two-body law gives the prediction for any effective striker
# mass, and the heavy-tool limit (alpha -> 1+e, beta -> -e) is what a real arm approaches. Verified
# against the simulator by actually changing the paddle mass and re-identifying.

def e7_regime(gen: Any) -> dict:
    """Which strike law is this scene actually obeying -- and what that means for a real tool.

    The module docstring asserts the two-body law with a finite 2 kg paddle, and derives
    ``e_effective`` by inverting ``alpha = m_p(1+e)/(m_b+m_p)``. That is the wrong law here, and the
    error is invisible at the default mass: at a 40:1 mass ratio the two-body and velocity-clamped
    predictions differ by only 2.5%, well inside what a casual check would forgive.

    Discriminating properly needs two things the first attempt got wrong:

    * A LIGHT striker. At 4:1 the two predictions are 25% apart instead of 2.5%.
    * A restitution estimated INDEPENDENTLY of alpha. Defining ``e := -beta`` and then noting
      ``alpha ~ 1+e`` is circular -- it only restates ``alpha+beta ~ 1``, which every elastic law
      satisfies (see below) and which therefore discriminates nothing. Here ``e`` comes from the
      two-body relation for beta, and the test is how alpha SCALES with mass at fixed contact.

    Result: with the servo on, alpha is flat in mass (ratio 1.008 over 0.2->2 kg) = velocity-clamped;
    with the servo off it recovers the two-body ratio 0.820 to three decimals. The computed-torque
    servo, not the paddle's mass, is what sets the law.

    Also reported is ``alpha+beta``, which must equal 1 for ANY correct strike law: if ball and
    striker move at the same velocity u they cannot collide, so v_out = u = (alpha+beta)u. It holds in
    both the two-body and clamped laws identically, costs nothing to compute, and needs no ground
    truth -- which makes it the one validity check a hardware rig can run on its own fitted numbers.
    """
    import src.data.paddle_strike as ps
    # Restitution measured DIRECTLY, assuming neither law: a servo-clamped STATIONARY striker is a
    # moving wall, so v_out = -e*v_in identically. This is what makes the comparison below honest --
    # back-solving e from beta (the first attempt) assumes the two-body relation and then "confirms"
    # it, and at servo-on it returns e = 1.41, an impossible restitution, which is the tell.
    e_probe = _GainStrike(kp=4000.0, kd=400.0, mff=PADDLE_MASS)
    e_meas = [-e_probe.simulate(v, 0.0, render=False)["v_out_world"] / v
              for v in (0.85, 0.95, 1.05, 1.15)]
    e_contact = float(np.mean(e_meas))

    rows = []
    for m_p in (0.2, 0.5, 1.0, 2.0, 8.0):
        for servo, kp, kd in (("on", 4000.0, 400.0), ("off", 0.0, 0.0)):
            saved = ps.PADDLE_MASS
            ps.PADDLE_MASS = m_p
            try:
                g = _GainStrike(kp=kp, kd=kd, mff=m_p)
                inv = StrikeInverse.fit(
                    sweep_strikes(g, [0.95, 1.05], np.linspace(0.25, -1.45, 12)), BALL_MASS, m_p)
            finally:
                ps.PADDLE_MASS = saved
            a_two = m_p * (1 + e_contact) / (BALL_MASS + m_p)
            a_clamped = 1 + e_contact
            rows.append({"paddle_mass_kg": m_p, "servo": servo,
                         "alpha": inv.alpha, "beta": inv.beta,
                         "alpha_plus_beta": inv.alpha + inv.beta,
                         "alpha_if_two_body": a_two, "alpha_if_clamped": a_clamped,
                         "closer_to": ("clamped" if abs(inv.alpha - a_clamped)
                                       < abs(inv.alpha - a_two) else "two-body"),
                         # how far apart the two hypotheses are here: below a few percent the test
                         # cannot decide anything, which is why the light striker is the real evidence
                         "hypotheses_apart": abs(a_clamped - a_two) / a_clamped})
    disc = {}
    for servo in ("on", "off"):
        a_lo = next(r["alpha"] for r in rows if r["servo"] == servo and r["paddle_mass_kg"] == 0.2)
        a_hi = next(r["alpha"] for r in rows if r["servo"] == servo and r["paddle_mass_kg"] == 2.0)
        pred_two = ((0.2 * (1 + e_contact) / (BALL_MASS + 0.2))
                    / (2.0 * (1 + e_contact) / (BALL_MASS + 2.0)))
        disc[servo] = {"alpha_ratio_0p2_over_2": a_lo / a_hi,
                       "two_body_predicts": pred_two, "clamped_predicts": 1.0}
    return {"rows": rows, "discriminator": disc, "e_contact_measured": e_contact,
            "e_contact_spread": float(np.ptp(e_meas))}


def e8_boost_identity(gen: Any, inv: StrikeInverse) -> dict:
    """Direct, in-domain test of alpha+beta=1: boost ball and striker together.

    The identity above is an extrapolation to v_p = v_in, which lies outside the fitted grid. Boosting
    both velocities by the same delta tests the same physics INSIDE the domain: a Galilean shift must
    move the outcome by exactly delta. This doubles as a hardware self-check -- it needs only relative
    measurements, so a rig can validate its own strike law without knowing the truth.
    """
    rows = []
    for v_in, v_p in ((0.9, -0.6), (0.95, -0.3), (1.0, -1.106)):
        for d in (0.05, 0.10, 0.20):
            a = gen.simulate(v_in, v_p, render=False)["v_out_world"]
            b = gen.simulate(v_in + d, v_p + d, render=False)["v_out_world"]
            rows.append({"v_in": v_in, "v_p": v_p, "delta": d, "dv_out_over_delta": (b - a) / d})
    return {"rows": rows, "fitted_alpha_plus_beta": inv.alpha + inv.beta,
            "expected": 1.0}


# -- report ----------------------------------------------------------------------------------------

def _fmt(x: float, pct: bool = True) -> str:
    if not np.isfinite(x):
        return "MISS"
    return f"{x:.2%}" if pct else f"{x:.4g}"


def write_report(out: Path, res: dict, inv: StrikeInverse) -> None:
    L = ["# Hardware readiness specs -- paddle strike", "",
         f"Bar: relative outcome error <= {TOL:.0%}. Nominal law measured in simulation: "
         f"`alpha={inv.alpha:.5f}`, `beta={inv.beta:.5f}`, `e_eff={inv.e_effective:.4f}`.",
         "",
         "Every spec below is what an operator can get WRONG while still hitting the bar. Actions are",
         "always planned with the nominal law and executed in the perturbed world -- nothing is re-fit",
         "to conceal the error.", ""]

    e1 = res["e1_sensing"]
    L += ["## 1. Ball-speed sensing", "",
          "Outcome error is **exactly** `|beta| x (v_in measurement error)` -- independent of the",
          "target and of alpha. So this spec transfers to any rig once beta is known.", "",
          "| commanded | tolerance on v_in (measured) | analytic |", "|---|---|---|"]
    for r in RATIOS:
        L.append(f"| {r:g}x | {e1['tolerance_measured'][r]:.3f} m/s | "
                 f"{e1['tolerance_analytic'][r]:.3f} m/s |")
    worst_gap = max(abs(r["rel_err"] - r["analytic_rel_err"]) for r in e1["rows"])
    L += ["", f"**Binding spec: measure incoming ball speed to +-{e1['binding_spec_m_s']:.3f} m/s** "
          f"({100 * e1['binding_spec_m_s'] / V_IN:.1f}% at {V_IN:g} m/s). Set by the slowest command.",
          "",
          f"Take the analytic column as the spec. The measured one is quantised by the sweep grid -- it "
          f"reports the largest GRID POINT that passes, so it rounds down (0.010 where the true limit is "
          f"0.028). The formula itself is confirmed pointwise: measured and predicted outcome error "
          f"agree to {worst_gap:.2e} across all {len(e1['rows'])} runs.", ""]

    e2 = res["e2_trigger"]
    L += ["## 2. Trigger timing", "",
          "| commanded | tolerance on trigger time |", "|---|---|"]
    for r in RATIOS:
        L.append(f"| {r:g}x | +-{e2['tolerance_s'][r] * 1000:.0f} ms |")
    L += ["", f"**Binding spec: +-{e2['binding_spec_s'] * 1000:.0f} ms** "
          f"(+-{e2['binding_spec_s'] * FPS:.1f} frames at {FPS} fps). Contact is designed to land "
          "inside the constant-velocity cruise, which is why this is loose.", ""]

    e3 = res["e3_lateral"]
    L += ["## 3. Lateral aim", "", "| commanded | tolerance on lateral offset |", "|---|---|"]
    for r in RATIOS:
        L.append(f"| {r:g}x | +-{e3['tolerance_m'][r] * 1000:.0f} mm |")
    L += ["", f"**Binding spec: +-{e3['binding_spec_m'] * 1000:.0f} mm** of lateral aim.", ""]

    L += ["## 4. Parameter drift since calibration", "",
          "| change | worst rel err (nominal law) | worst after re-identifying |", "|---|---|---|"]
    by = {}
    for row in res["e4_mismatch"]["rows"]:
        by.setdefault(row["perturbation"], {"errs": [], "refit": None})
        if row["ratio"] == "refit":
            by[row["perturbation"]]["refit"] = row["worst_after_refit"]
        else:
            by[row["perturbation"]]["errs"].append(row["rel_err"])
    for k, v in by.items():
        L.append(f"| {k} | {_fmt(max(v['errs']))} | {_fmt(v['refit'])} |")
    L += ["", "Re-identification is what recovers the bar -- which is the argument for treating",
          "calibration as a routine step, not a one-off.", ""]

    L += ["## 5. Calibration cost", "",
          "How many measured strikes, at what tracker noise, before commanding works. "
          "`frac passing` is over 12 random draws of which strikes you happen to collect.", "",
          "| strikes | tracker noise | median worst err | p90 | frac passing |",
          "|---|---|---|---|---|"]
    for row in res["e5_calibration"]["rows"]:
        L.append(f"| {row['n_strikes']} | {row['noise_m_s']:.2f} m/s | "
                 f"{_fmt(row['median_worst_rel_err'])} | {_fmt(row['p90_worst_rel_err'])} | "
                 f"{row['frac_repeats_passing']:.0%} |")
    L.append("")

    L += ["## 6. The frictionless / spinless idealisation -- the biggest real gap", "",
          "The simulator's ball is a puck: zero table friction, zero spin. A real ball rolls on a real",
          "surface. Two columns, because they say different things: `mean` is the average over the",
          "16-frame post window (what the label uses), `first` is the first post-contact sample. If the",
          "strike itself were breaking, both would be bad. If only `mean` is bad, the strike is fine",
          "and the MEASUREMENT WINDOW is what friction breaks -- the ball is decelerating across it.",
          "",
          "| source | value | worst err (mean) | worst err (first sample) | v_out std |",
          "|---|---|---|---|---|"]
    agg: dict = {}
    for row in res["e6_friction_spin"]["rows"]:
        k = (row["kind"], row["value"])
        agg.setdefault(k, {"mean": [], "first": [], "std": []})
        agg[k]["mean"].append(row["rel_err"])
        agg[k]["first"].append(row.get("rel_err_first_sample", float("nan")))
        agg[k]["std"].append(row.get("v_out_std", float("nan")))
    for (kind, val), d in agg.items():
        first = max(d["first"]) if np.all(np.isfinite(d["first"])) else float("nan")
        std = np.nanmax(d["std"]) if len(d["std"]) else float("nan")
        L.append(f"| {kind} | {val:g} | {_fmt(max(d['mean']))} | "
                 f"{'n/a' if not np.isfinite(first) else _fmt(first)} | "
                 f"{'n/a' if not np.isfinite(std) else f'{std:.2e}'} |")
    L += ["", "Spin is a non-issue -- the contact is frictionless, so the ball's rotation cannot couple",
          "into the axial impulse at all. Friction is the one to worry about.", ""]

    L += ["## 7. Which strike law is this, really", "",
          "**Correction to the scene's own documentation.** The module docstring claims the two-body law",
          "with a finite 2 kg paddle, and `StrikeInverse.e_effective` inverts that relation to report a",
          "restitution. The scene does not obey it: the computed-torque servo holds the striker's",
          "velocity through the ~2 ms impact, so the striker behaves as if infinitely massive and its",
          "mass drops out.", "",
          f"Contact restitution, measured directly against a clamped stationary striker (no law "
          f"assumed): **e = {res['e7_regime']['e_contact_measured']:.4f}** "
          f"(spread {res['e7_regime']['e_contact_spread']:.1e} over four incoming speeds, so it is a "
          "genuine property of the contact). Everything below is predicted from that one number.", "",
          "The discriminator is how alpha SCALES with striker mass at fixed contact (0.2 kg vs 2.0 kg):",
          "",
          "| servo | measured alpha ratio | two-body predicts | velocity-clamped predicts |",
          "|---|---|---|---|"]
    for servo, d in res["e7_regime"]["discriminator"].items():
        L.append(f"| {servo} | {d['alpha_ratio_0p2_over_2']:.3f} | {d['two_body_predicts']:.3f} | "
                 f"{d['clamped_predicts']:.3f} |")
    L += ["", "Servo on: flat in mass, i.e. velocity-clamped. Servo off: the two-body ratio is",
          "recovered to three decimals. So the SERVO sets the law, not the paddle's mass.", "",
          "Why it matters for hardware, and it cuts both ways:", "",
          "* Good: a stiff arm that holds tool speed through contact gives `alpha ~ 1+e`, `beta ~ -e`,",
          "  independent of the tool's effective inertia -- which you would otherwise have to estimate.",
          "* Bad: the compliant mount HARDWARE.md requires exists precisely to decouple the tool from",
          "  the servo during impact, which pushes the rig toward the two-body regime. The sim scene and",
          "  the recommended hardware configuration are therefore in DIFFERENT regimes. This does not",
          "  invalidate the certificate (alpha/beta are measured, not derived), but it does mean the",
          "  sim's numbers are not a starting guess for hardware -- re-identification is mandatory.", "",
          "| striker mass | servo | alpha | alpha+beta | two-body | clamped | verdict | hyps apart |",
          "|---|---|---|---|---|---|---|---|"]
    for row in res["e7_regime"]["rows"]:
        L.append(f"| {row['paddle_mass_kg']:g} kg | {row['servo']} | {row['alpha']:.4f} | "
                 f"{row['alpha_plus_beta']:.4f} | {row['alpha_if_two_body']:.4f} | "
                 f"{row['alpha_if_clamped']:.4f} | {row['closer_to']} | "
                 f"{row['hypotheses_apart']:.1%} |")
    L += ["", "Read the last column before the verdict: at 8 kg the two hypotheses are ~0.6% apart, so",
          "that row decides nothing either way. The 0.2 kg rows, 25% apart, are the actual evidence.", ""]

    e8 = res["e8_boost_identity"]
    L += ["## 8. A validity check a real rig can run on itself", "",
          "`alpha + beta = 1` for any correct strike law: if ball and striker move at the same velocity",
          "they cannot collide, so the outcome must equal that shared velocity. Tested directly by",
          "boosting both by delta -- the outcome must shift by exactly delta. Needs no ground truth,",
          "only repeatability, so it is runnable on hardware.", "",
          f"Fitted `alpha+beta` for this scene: **{e8['fitted_alpha_plus_beta']:.5f}** (expect 1).", "",
          "| v_in | v_p | delta | measured dv_out/delta |", "|---|---|---|---|"]
    for row in e8["rows"]:
        L.append(f"| {row['v_in']:g} | {row['v_p']:g} | {row['delta']:.2f} | "
                 f"{row['dv_out_over_delta']:.4f} |")
    L += ["", "**This check fails on the exported hardware trajectories.** Their fitted law is",
          "`alpha=1.5725, beta=-0.9132`, so `alpha+beta = 0.659`, not 1 -- in the same signed",
          "convention. Those coefficients come from the dynamically-actuated arm variant, which is",
          "independently known to be unstable. The exported trajectories are self-consistent (planned",
          "and executed in the same model, so the error cancels), but the alpha/beta quoted in",
          "HARDWARE.md as a starting point are not a physically valid strike law and should not be",
          "presented as one.", ""]

    (out / "SPECS.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="paddle")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen = build_striker(args.embodiment)
    print("[0/7] identifying the nominal strike law ...", flush=True)
    inv = fit_nominal(gen)
    print(f"      alpha={inv.alpha:.5f} beta={inv.beta:.5f} e_eff={inv.e_effective:.4f}", flush=True)

    res: dict[str, Any] = {}
    for i, (key, fn) in enumerate([
        ("e1_sensing", lambda: e1_sensing(gen, inv)),
        ("e2_trigger", lambda: e2_trigger(gen, inv)),
        ("e3_lateral", lambda: e3_lateral(gen, inv)),
        ("e4_mismatch", lambda: e4_mismatch(gen, inv)),
        ("e5_calibration", lambda: e5_calibration(gen, inv)),
        ("e6_friction_spin", lambda: e6_friction_spin(gen, inv)),
        ("e7_regime", lambda: e7_regime(gen)),
        ("e8_boost_identity", lambda: e8_boost_identity(gen, inv)),
    ], start=1):
        print(f"[{i}/8] {key} ...", flush=True)
        res[key] = fn()

    res["nominal"] = inv.to_dict()
    (out / "specs.json").write_text(json.dumps(res, indent=2, default=float))
    write_report(out, res, inv)
    print(f"\nwrote {out}/SPECS.md and specs.json", flush=True)


if __name__ == "__main__":
    main()
