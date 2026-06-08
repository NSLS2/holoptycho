"""Tests for the ViT mosaic-canvas centering logic (holoptycho-specific)."""
import numpy as np

from holoptycho.mosaic_canvas import estimate_canvas_mid


def test_uses_observed_midpoint_when_range_well_covered():
    # observed range (10) >= 50% of commanded (12) -> plain observed midpoint
    finite = np.array([0.0, 5.0, 10.0])
    mid = estimate_canvas_mid(0.0, 10.0, 10.0, 12.0, finite)
    assert mid == 5.0


def test_uses_observed_midpoint_when_commanded_unknown():
    finite = np.array([0.0, 2.0])
    mid = estimate_canvas_mid(0.0, 2.0, 2.0, 0.0, finite)
    assert mid == 1.0


def test_infers_midpoint_from_positive_direction_when_underscanned():
    # only 1 µm of a commanded 10 µm seen so far, positions increasing
    # -> direction +1, start = obs_min, mid = obs_min + cmd/2
    finite = np.array([0.0, 0.5, 1.0])
    mid = estimate_canvas_mid(0.0, 1.0, 1.0, 10.0, finite)
    assert mid == 0.0 + 10.0 / 2.0


def test_infers_midpoint_from_negative_direction_when_underscanned():
    # positions decreasing -> direction -1, start = obs_max, mid = obs_max - cmd/2
    finite = np.array([1.0, 0.5, 0.0])
    mid = estimate_canvas_mid(0.0, 1.0, 1.0, 10.0, finite)
    assert mid == 1.0 - 10.0 / 2.0


def test_flat_direction_treated_as_negative():
    # no movement yet (all equal) -> sign 0 -> treated as -1 (HXN default sense)
    finite = np.array([2.0, 2.0, 2.0])
    mid = estimate_canvas_mid(2.0, 2.0, 0.0, 10.0, finite)
    assert mid == 2.0 - 10.0 / 2.0
