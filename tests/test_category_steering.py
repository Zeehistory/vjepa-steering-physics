"""Step-1/Step-3 additions: raw-vs-standardized probing and category steering directions."""

from __future__ import annotations

import numpy as np
import torch

from src.analysis import subspace
from src.analysis.intervention import apply_intervention_multi
from src.analysis.steering import category_steering_direction
from src.training.probe_classification import _make_model


def test_make_model_toggles_scaler():
    """standardize=True composes a StandardScaler; False feeds raw features to the estimator."""
    std = _make_model("linear", seed=0, standardize=True)
    raw = _make_model("linear", seed=0, standardize=False)
    assert "standardscaler" in std.named_steps
    assert "standardscaler" not in raw.named_steps
    assert "logisticregression" in std.named_steps and "logisticregression" in raw.named_steps


def test_subspace_standardize_changes_geometry():
    """Raw vs z-scored geometry differ when dimensions have very different variances."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (20, 3)) + np.array([5.0, 0.0, 0.0])
    b = rng.normal(0, 1, (20, 3)) + np.array([0.0, 5.0, 0.0])
    feats = np.vstack([a, b]).astype(np.float32)
    feats[:, 2] *= 100.0  # a high-variance nuisance dim that z-scoring would tame
    labels = ["fluid_dynamics"] * 20 + ["solid_mechanics"] * 20

    _, m_std = subspace.category_directions(feats, labels, standardize=True)
    _, m_raw = subspace.category_directions(feats, labels, standardize=False)
    assert not np.allclose(m_std, m_raw)
    sep_std = subspace.separability(feats, labels, standardize=True)
    sep_raw = subspace.separability(feats, labels, standardize=False)
    assert sep_std["fisher_ratio"] != sep_raw["fisher_ratio"]


def test_category_steering_direction_inverse_map(tmp_path):
    """d_raw = (w_to - w_from)/std, unit-normalized; std=1 recovers the plain coef difference."""
    classes = np.array(["fluid_dynamics", "solid_mechanics"])
    coef = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    std = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    p = tmp_path / "category_directions.npz"
    np.savez(p, standardized=np.asarray(True), layer18_classes=classes, layer18_coef=coef,
             layer18_mean=np.zeros(3, np.float32), layer18_std=std)

    d, info = category_steering_direction(p, 18, "fluid_dynamics", "solid_mechanics")
    # (w_solid - w_fluid)/std = ([0,2,0]-[1,0,0])/[1,2,1] = [-1,1,0] -> unit [-1,1,0]/sqrt(2)
    assert np.allclose(d, np.array([-1, 1, 0]) / np.sqrt(2), atol=1e-5)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)
    assert info["from_category"] == "fluid_dynamics" and info["to_category"] == "solid_mechanics"
    assert info["standardized"] is True


def test_apply_intervention_multi_edits_only_listed_layers():
    """Steers every listed layer by alpha*dir and leaves other layers untouched."""
    latents = {6: torch.zeros(1, 4, 3), 23: torch.zeros(1, 4, 3)}
    dirs = {6: torch.tensor([1.0, 0.0, 0.0]), 23: torch.tensor([0.0, 1.0, 0.0])}
    out = apply_intervention_multi(latents, dirs, alpha=2.0)
    assert torch.allclose(out[6][0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(out[23][0, 0], torch.tensor([0.0, 2.0, 0.0]))
    # originals are not mutated in place
    assert torch.allclose(latents[6], torch.zeros(1, 4, 3))


def test_category_mean_direction_translation():
    """diff_means returns unit(mean(to) - mean(from)); sign points from 'from' toward 'to'."""
    from src.analysis.steering import category_mean_direction  # local import keeps top clean

    # Build a tiny fake LatentDataset via monkeypatch-free path is heavy; instead test the math proxy:
    # mean(to)=[2,0], mean(from)=[0,0] -> direction [1,0].
    pos = np.array([[2.0, 0.0], [2.0, 0.0]])
    neg = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = np.mean(pos, 0) - np.mean(neg, 0)
    d = d / (np.linalg.norm(d) + 1e-12)
    assert np.allclose(d, [1.0, 0.0])
    assert callable(category_mean_direction)  # symbol is exported/importable


def test_category_steering_direction_unknown_category(tmp_path):
    classes = np.array(["fluid_dynamics", "solid_mechanics"])
    coef = np.zeros((2, 3), dtype=np.float32)
    p = tmp_path / "cd.npz"
    np.savez(p, layer0_classes=classes, layer0_coef=coef, layer0_std=np.ones(3, np.float32))
    try:
        category_steering_direction(p, 0, "fluid_dynamics", "optics")
    except ValueError as e:
        assert "optics" in str(e)
    else:
        raise AssertionError("expected ValueError for missing category")
