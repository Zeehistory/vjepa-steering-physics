"""The marker loss must punish all THREE failure modes, not just the obvious one.

This test exists because a version that punished only omission cost 4000 GPU-steps. Masking on the
target alone makes a blank marker expensive and spurious marker paint FREE -- amber invented where the
target has none simply falls outside the mask. The decoder found that immediately and flooded the frame
with marker colour: ``marker_mass_decoded/rendered`` went 0.000 -> 467, speed correlation collapsed to
nan, heading error 127.5 deg. The loss was doing exactly what it said; it just did not say enough.

So the contract is stated as a test rather than as a docstring claim: omitting the marker, painting it
where it does not belong, and painting it everywhere must all be expensive relative to rendering it
correctly.
"""
from __future__ import annotations

import torch

from src.decoders.loss_functions import marker_weighted_charbonnier

GREY, MARKER = 0.7, (1.0, 0.75, 0.10)


def _scene(marker_at: tuple[int, int] | None, flood: bool = False) -> torch.Tensor:
    x = torch.full((1, 4, 3, 64, 64), GREY)
    if flood:
        for c, v in enumerate(MARKER):
            x[:, :, c] = v
        return x
    if marker_at is not None:
        i, j = marker_at
        for c, v in enumerate(MARKER):
            x[:, :, c, i:i + 4, j:j + 4] = v
    return x


def test_marker_loss_punishes_omission_misplacement_and_flooding() -> None:
    target = _scene((30, 30))
    perfect = marker_weighted_charbonnier(target.clone(), target)

    omitted = marker_weighted_charbonnier(_scene(None), target)
    misplaced = marker_weighted_charbonnier(_scene((10, 10)), target)
    flooded = marker_weighted_charbonnier(_scene(None, flood=True), target)

    # A correct render should be near-free; every failure mode should cost two orders more.
    assert perfect < 0.01, perfect
    for name, bad in (("omitted", omitted), ("misplaced", misplaced), ("flooded", flooded)):
        assert bad > 50 * perfect, f"{name} is too cheap: {bad} vs perfect {perfect}"

    # Flooding specifically: the regression that actually happened. Guard it by name so a future change
    # that reintroduces target-only masking fails here rather than 4000 steps into a training run.
    assert flooded > 0.1, f"over-painting the marker must not be cheap, got {flooded}"


def test_marker_loss_ignores_frames_with_no_marker_colour() -> None:
    """A scene with no marker anywhere must not produce NaN or a huge spurious penalty."""
    blank = _scene(None)
    loss = marker_weighted_charbonnier(blank.clone(), blank)
    assert torch.isfinite(torch.as_tensor(loss)), loss
    assert loss < 0.01, loss
