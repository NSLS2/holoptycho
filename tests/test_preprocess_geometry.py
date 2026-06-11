"""Tests for holoptycho detector cropping geometry.

The pipeline coordinate-corrects the WHOLE raw frame local->global via a D4
(`detector_orientation`), then crops a single global ROI to the model size with
the generic `ptychoml.crop_to_roi`. Source-dependent: live = 'rot180' (raw
Eiger), replay = 'identity' (Tiled already corrected). These tests cover the D4
correction and that both sources land on the same global pixels.
"""
import numpy as np
import pytest

# preprocess.py imports holoscan + cupy at module level.
pytest.importorskip("holoscan")
pytest.importorskip("cupy")

from ptychoml.preprocess import apply_d4, crop_to_roi


def _labelled_frame(h, w):
    """Frame where pixel (r, c) == r*100 + c, so provenance is readable."""
    r = np.arange(h)[:, None] * 100
    c = np.arange(w)[None, :]
    return (r + c).astype(np.int32)


def test_rot180_is_flip_both_axes():
    img = _labelled_frame(6, 8)
    np.testing.assert_array_equal(apply_d4(img, "rot180"), np.flip(np.flip(img, 0), 1))


def test_identity_is_noop():
    img = _labelled_frame(6, 8)
    np.testing.assert_array_equal(apply_d4(img, "identity"), img)


def test_correct_then_crop_is_plain_slice_in_global_frame():
    """After coordinate correction the ROI is a plain global crop (no mirroring)."""
    raw = _labelled_frame(10, 12)
    glob = apply_d4(raw, "rot180")
    roi = np.array([[2, 5], [3, 9]])  # rows 2:5, cols 3:9 in the GLOBAL frame
    out = crop_to_roi(glob, roi)
    np.testing.assert_array_equal(out, glob[2:5, 3:9])
    assert out.shape == (3, 6)


def test_live_and_replay_land_on_same_global_pixels():
    """Live (raw, 'rot180') and replay (Tiled == rot180(raw), 'identity') must
    crop the SAME global pixels with the SAME ROI — the source-dependent
    correction reconciles the two acquisition paths."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 1000, size=(16, 20)).astype(np.int32)   # live Eiger frame
    tiled = apply_d4(raw, "rot180")                                # what Tiled stores
    roi = np.array([[4, 12], [5, 17]])                            # one global ROI

    live = crop_to_roi(apply_d4(raw, "rot180"), roi)              # live applies rot180
    replay = crop_to_roi(apply_d4(tiled, "identity"), roi)        # replay skips it

    np.testing.assert_array_equal(live, replay)


def test_legacy_fliplr_left_live_and_replay_180_apart():
    """Regression rationale: the OLD uniform 'fliplr' on both sources left live
    and replay a rot180 apart — the bug this change fixes."""
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 1000, size=(16, 20)).astype(np.int32)
    tiled = apply_d4(raw, "rot180")
    old_live = apply_d4(raw, "fliplr")
    old_replay = apply_d4(tiled, "fliplr")
    # old_live vs old_replay differ by exactly rot180
    np.testing.assert_array_equal(apply_d4(old_live, "rot180"), old_replay)
    assert not np.array_equal(old_live, old_replay)
