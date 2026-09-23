"""Direct inverse policy: (context belief, target belief) -> the action that gets you there.

This module implements *a separate predictor that predicts actions*. Given the V-JEPA
latent of the ball on its way in (``h_pre``) and a latent that encodes the outcome we want
(``z_target``, produced by steering), it returns the striker command ``v_p``.

**Why a separate predictor rather than inverting the forward model.** ``close_action_loop.py``
already closes the loop, but backwards: it fits a latent forward map ``H_hat(a)`` and picks the
action by ``argmin_a ||H_hat(a) - z_target||``. That works (94.7% pass@5% on paddle, with
calibration) but it has a diagnosed pathology -- ridge shrinks the command direction, the argmin
surface goes flat, and the returned action is an almost perfectly linear but *shrunk* copy of the
right one (slope 0.90 paddle, 0.68 franka transfer, both at R^2 > 0.99). Uncorrected it scores
54.8%. An affine calibration on measured outcomes repairs it after the fact.

Supervising the command axis *directly* removes that failure mode by construction: the action is
the regression target, so nothing shrinks it toward the mean except ordinary regularisation, which
is chosen against end-to-end error.

**The one subtlety that decides whether this works.** At training time the honest target latent is
the real post-strike clip ``z_post``. At test time no such clip exists -- that is the whole point,
we are asking for an outcome that has not happened -- so the target is *synthesised* by the steering
map ``W_tgt: [h_pre, v*] -> z``. Train on real and test on synthetic and the predictor is evaluated
off its training distribution, which is exactly the kind of gap that produces a good open-loop MAE
and a bad closed-loop pass rate. So ``target_source`` is a first-class knob:

* ``real``  -- train on ``z_post``. Honest, but not what test time looks like.
* ``synth`` -- train on ``W_tgt([h_pre, v_out_true])``, the same pathway test uses. Matched.
* ``both``  -- both, as augmentation. Matched *and* anchored to real perception.

It is an ablation, not a guess, and the driver reports all three.

The API deliberately mirrors :class:`~src.control.strike_inverse.StrikeInverse` (``fit`` /
``action_for`` / ``to_dict`` / ``save`` / ``load``) so the analytic inverse and the learned policy
are interchangeable downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FAMILIES = ("ridge", "mlp")


# -- features ----------------------------------------------------------------------------------------

def pair_features(h_pre: np.ndarray, z_target: np.ndarray, *,
                  use_pre: bool = True, use_diff: bool = True) -> np.ndarray:
    """Stack the (context, target) pair into one design matrix.

    ``h_pre`` and ``z_target`` live in the SAME PCA basis (the Reducer is fit on train post and
    applied to the pre clips too), so their difference is a meaningful vector and not a type error.
    The difference term is included because the strike law is a statement about *change* --
    ``v_out - v_in = alpha (v_p - v_in)`` -- so handing the head the change explicitly saves it from
    having to discover a subtraction that we already know is the right coordinate.

    Ablate with ``use_pre=False`` (target only: does the context matter at all?) and
    ``use_diff=False`` (does the explicit difference buy anything over the two vectors?).
    """
    h_pre = np.atleast_2d(np.asarray(h_pre, dtype=np.float64))
    z_target = np.atleast_2d(np.asarray(z_target, dtype=np.float64))
    if h_pre.shape != z_target.shape:
        raise ValueError(f"h_pre {h_pre.shape} and z_target {z_target.shape} must match")
    cols = [z_target]
    if use_pre:
        cols.append(h_pre)
    if use_diff:
        cols.append(z_target - h_pre)
    return np.concatenate(cols, axis=1)


def scene_holdout(scenes: list[int] | np.ndarray, frac: float = 0.15,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (fit, val) masks that are disjoint AT THE SCENE LEVEL.

    Clip-level splitting leaks here and it is not a subtle leak: the ranks within one scene are the
    same episode at different commanded ratios, so they share ``v_in`` and the entire pre-contact
    trajectory. A random clip split puts rank 3 of scene 7 in train and rank 5 of scene 7 in val,
    and the val error then measures interpolation between two commands of a memorised scene.
    """
    scenes = np.asarray(scenes)
    uniq = np.unique(scenes)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(uniq))))
    val_ids = set(rng.permutation(uniq)[:n_val].tolist())
    val = np.array([s in val_ids for s in scenes])
    return ~val, val


# -- heads -------------------------------------------------------------------------------------------

def _ridge_solve(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an UNPENALISED intercept, same convention as close_action_loop."""
    Fb = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
    R = lam * np.eye(Fb.shape[1])
    R[-1, -1] = 0.0
    return np.linalg.solve(Fb.T @ Fb + R, Fb.T @ y)


def _ridge_apply(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([F, np.ones((F.shape[0], 1))], axis=1) @ w


def _build_mlp(d_in: int, width: int, depth: int, drop: float) -> "Any":
    """Linear skip + residual MLP correction.

    Two choices here are load-bearing, and both were made after the first version measurably failed
    a linear-recovery check (MLP 3.08% of range where ridge got 0.195%).

    NO LayerNorm on the input. The design matrix is already standardised per COLUMN with statistics
    frozen on the fit rows, which is the normalisation that belongs here. LayerNorm would normalise
    per ROW instead, dividing every sample by its own norm -- and the overall magnitude of a latent
    is exactly what encodes how fast the ball is going, so that discards the signal we are asking
    about. Column-standardisation before, none after.

    A LINEAR SKIP straight from input to output. The true inverse is linear to ~0.3% of range (see
    strike_inverse); making the deep path a *correction* to a linear map rather than the whole map
    means the network starts near the right answer instead of having to rediscover it, and it can
    never do meaningfully worse than the ridge arm it is supposed to beat.
    """
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(width, width))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.net(x)

    class SkipMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, 1)
            self.deep = nn.Sequential(nn.Linear(d_in, width), *[Block() for _ in range(depth)],
                                      nn.LayerNorm(width), nn.Linear(width, 1))
            # Start the correction at exactly zero, so epoch 0 IS the linear model and every step
            # after it is a measured improvement on that baseline rather than a random walk away.
            nn.init.zeros_(self.deep[-1].weight)
            nn.init.zeros_(self.deep[-1].bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.linear(x) + self.deep(x)

    return SkipMLP()


def _fit_mlp(F: np.ndarray, y: np.ndarray, Fv: np.ndarray, yv: np.ndarray,
             cfg: dict[str, Any], seed: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """One MLP member. Returns its weights as numpy arrays plus a training trace.

    Pure CPU on purpose: the design is a few thousand rows of a few hundred columns, so a GPU would
    add queue time and nothing else. Everything the loop needs runs in one CPU allocation.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    d_in, width = F.shape[1], int(cfg["width"])
    depth, drop = int(cfg["depth"]), float(cfg["dropout"])

    model = _build_mlp(d_in, width, depth, drop)

    Ft = torch.tensor(F, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    Fvt = torch.tensor(Fv, dtype=torch.float32)
    yvt = torch.tensor(yv, dtype=torch.float32).view(-1, 1)

    epochs, bs = int(cfg["epochs"]), int(cfg["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Huber, not MSE. The action distribution has hard envelope ends and a handful of clips sit at
    # them; squared error lets those few dominate the gradient and tilts the whole map to serve the
    # extremes. delta is in units of the STANDARDISED action, so 1.0 is one sd.
    lossf = torch.nn.HuberLoss(delta=1.0)

    g = torch.Generator().manual_seed(seed)
    n = Ft.shape[0]
    best_val, best_state, best_epoch, patience = float("inf"), None, -1, int(cfg["patience"])
    trace = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Ft[idx]), yt[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(Fvt), yvt))
        trace.append(v)
        if v < best_val - 1e-6:
            best_val, best_epoch = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif ep - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    names = list(model.state_dict().keys())
    weights = [model.state_dict()[k].detach().numpy().astype(np.float64) for k in names]
    return weights, {"param_names": names, "best_val": best_val, "best_epoch": best_epoch,
                     "epochs_run": len(trace), "val_trace": trace}


def _mlp_forward(F: np.ndarray, weights: list[np.ndarray], names: list[str],
                 cfg: dict[str, Any]) -> np.ndarray:
    """Rebuild the torch module from saved weights and run it.

    Reconstructing rather than pickling keeps the saved artefact a plain ``.npz`` + json, which
    survives a torch version bump and can be inspected without importing anything.
    """
    import torch

    model = _build_mlp(F.shape[1], int(cfg["width"]), int(cfg["depth"]), float(cfg["dropout"]))
    sd = {k: torch.tensor(w, dtype=torch.float32) for k, w in zip(names, weights)}
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(F, dtype=torch.float32)).numpy().ravel().astype(np.float64)


# -- the predictor -----------------------------------------------------------------------------------

@dataclass
class ActionPredictor:
    """A fitted inverse policy. ``action_for(h_pre, z_target) -> v_p``."""

    family: str = "ridge"
    use_pre: bool = True
    use_diff: bool = True
    # standardisation, fit on the training design only
    x_mu: np.ndarray | None = None
    x_sd: np.ndarray | None = None
    y_mu: float = 0.0
    y_sd: float = 1.0
    # ridge
    w: np.ndarray | None = None
    lam: float = float("nan")
    # mlp ensemble
    members: list[list[np.ndarray]] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    cfg: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- fitting ---------------------------------------------------------------------------------
    @classmethod
    def fit(cls, h_pre: np.ndarray, z_target: np.ndarray, a: np.ndarray, *,
            scenes: np.ndarray | list[int], family: str = "mlp",
            use_pre: bool = True, use_diff: bool = True,
            lam_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4),
            n_members: int = 5, val_frac: float = 0.15, seed: int = 0,
            cfg: dict[str, Any] | None = None) -> ActionPredictor:
        """Fit on ``(h_pre, z_target) -> a`` pairs, selecting on a SCENE-DISJOINT holdout."""
        if family not in FAMILIES:
            raise ValueError(f"unknown family '{family}', expected one of {FAMILIES}")
        a = np.asarray(a, dtype=np.float64).ravel()
        X = pair_features(h_pre, z_target, use_pre=use_pre, use_diff=use_diff)
        if X.shape[0] != a.shape[0]:
            raise ValueError(f"{X.shape[0]} feature rows against {a.shape[0]} actions")
        fit_m, val_m = scene_holdout(scenes, val_frac, seed)
        if not fit_m.any() or not val_m.any():
            raise ValueError("scene holdout produced an empty side; too few distinct scenes")

        # Standardise on the FIT rows only. PCA scores fall off like the singular values, so the
        # last components are orders of magnitude smaller than the first; unstandardised, weight
        # decay and a shared learning rate both effectively ignore them.
        x_mu = X[fit_m].mean(axis=0)
        x_sd = X[fit_m].std(axis=0)
        x_sd[x_sd < 1e-12] = 1.0
        y_mu, y_sd = float(a[fit_m].mean()), float(a[fit_m].std())
        y_sd = y_sd if y_sd > 1e-12 else 1.0
        Xs = (X - x_mu) / x_sd
        ys = (a - y_mu) / y_sd

        self = cls(family=family, use_pre=use_pre, use_diff=use_diff,
                   x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd)
        n_scenes = int(len(np.unique(np.asarray(scenes))))

        if family == "ridge":
            best = (float("inf"), None, float("nan"))
            scores = {}
            for lam in lam_grid:
                w = _ridge_solve(Xs[fit_m], ys[fit_m], lam)
                mae = float(np.abs(_ridge_apply(Xs[val_m], w) - ys[val_m]).mean()) * y_sd
                scores[lam] = mae
                if mae < best[0]:
                    best = (mae, w, lam)
            # refit the chosen lam on ALL rows -- selection is done, and throwing away 15% of the
            # data after choosing a hyperparameter on it is a pure loss.
            self.lam = float(best[2])
            self.w = _ridge_solve(Xs, ys, self.lam)
            self.meta = {"val_action_mae": best[0], "lam_scores": {str(k): v for k, v in scores.items()},
                         "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()), "n_scenes": n_scenes}
            return self

        c = {"width": 256, "depth": 3, "dropout": 0.10, "lr": 2e-3, "weight_decay": 1e-4,
             "epochs": 300, "batch_size": 128, "patience": 40}
        c.update(cfg or {})
        self.cfg = c
        vals, traces = [], []
        for m in range(n_members):
            wts, tr = _fit_mlp(Xs[fit_m], ys[fit_m], Xs[val_m], ys[val_m], c, seed=seed + 1000 * m)
            self.members.append(wts)
            self.param_names = tr["param_names"]
            vals.append(tr["best_val"])
            traces.append({"best_val": tr["best_val"], "best_epoch": tr["best_epoch"],
                           "epochs_run": tr["epochs_run"]})
        pred_val = self._raw_predict(Xs[val_m])
        self.meta = {"val_action_mae": float(np.abs(pred_val * y_sd + y_mu - a[val_m]).mean()),
                     "member_val_huber": vals, "members": traces,
                     "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()), "n_scenes": n_scenes}
        return self

    # -- use -------------------------------------------------------------------------------------
    def _raw_predict(self, Xs: np.ndarray) -> np.ndarray:
        """Prediction in STANDARDISED action units, averaged over ensemble members."""
        if self.family == "ridge":
            assert self.w is not None
            return _ridge_apply(Xs, self.w)
        return np.mean([_mlp_forward(Xs, wts, self.param_names, self.cfg)
                        for wts in self.members], axis=0)

    def _design(self, h_pre: np.ndarray, z_target: np.ndarray) -> np.ndarray:
        X = pair_features(h_pre, z_target, use_pre=self.use_pre, use_diff=self.use_diff)
        return (X - self.x_mu) / self.x_sd

    def action_for(self, h_pre: np.ndarray, z_target: np.ndarray) -> np.ndarray:
        """The striker command predicted to turn ``h_pre`` into ``z_target`` (signed, m/s)."""
        return self._raw_predict(self._design(h_pre, z_target)) * self.y_sd + self.y_mu

    def action_spread(self, h_pre: np.ndarray, z_target: np.ndarray) -> np.ndarray:
        """Ensemble disagreement in m/s -- a free uncertainty estimate. Zero for ``ridge``."""
        if self.family == "ridge" or len(self.members) < 2:
            return np.zeros(np.atleast_2d(h_pre).shape[0])
        Xs = self._design(h_pre, z_target)
        P = np.stack([_mlp_forward(Xs, w, self.param_names, self.cfg) for w in self.members])
        return P.std(axis=0) * self.y_sd

    # -- io --------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "use_pre": self.use_pre, "use_diff": self.use_diff,
                "y_mu": self.y_mu, "y_sd": self.y_sd, "lam": self.lam,
                "param_names": self.param_names, "cfg": self.cfg, "meta": self.meta,
                "n_members": len(self.members)}

    def save(self, path: str | Path) -> None:
        """Write ``<path>.json`` (everything human-readable) + ``<path>.npz`` (the arrays)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".json").write_text(json.dumps(self.to_dict(), indent=2, default=float))
        arrs: dict[str, np.ndarray] = {"x_mu": self.x_mu, "x_sd": self.x_sd}
        if self.w is not None:
            arrs["w"] = self.w
        for mi, wts in enumerate(self.members):
            for wi, arr in enumerate(wts):
                arrs[f"m{mi}_{wi}"] = arr
        np.savez(p.with_suffix(".npz"), **arrs)

    @classmethod
    def load(cls, path: str | Path) -> ActionPredictor:
        p = Path(path)
        d = json.loads(p.with_suffix(".json").read_text())
        z = np.load(p.with_suffix(".npz"))
        members = []
        for mi in range(int(d["n_members"])):
            wts, wi = [], 0
            while f"m{mi}_{wi}" in z:
                wts.append(z[f"m{mi}_{wi}"])
                wi += 1
            members.append(wts)
        return cls(family=d["family"], use_pre=d["use_pre"], use_diff=d["use_diff"],
                   x_mu=z["x_mu"], x_sd=z["x_sd"], y_mu=d["y_mu"], y_sd=d["y_sd"],
                   w=z["w"] if "w" in z else None, lam=d.get("lam", float("nan")),
                   members=members, param_names=d.get("param_names", []),
                   cfg=d.get("cfg", {}), meta=d.get("meta", {}))
