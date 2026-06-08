"""Tests for holoptycho-specific preprocessing geometry.

Covers the Eiger2 horizontal-flip ROI mirroring in ImageBatchOp — detector
handling that lives in holoptycho (the generic crop is ptychoml.crop_to_roi).
"""
import numpy as np
import pytest

# preprocess.py imports holoscan + cupy at module level.
pytest.importorskip("holoscan")
pytest.importorskip("cupy")

from ptychoml.preprocess import crop_to_roi
from holoptycho.preprocess import crop_flipped_roi


def _labelled_frame(h, w):
    """Frame where pixel (r, c) == r*100 + c, so the crop's provenance is
    readable from the values."""
    r = np.arange(h)[:, None] * 100
    c = np.arange(w)[None, :]
    return (r + c).astype(np.int32)


def test_crop_flipped_roi_selects_mirrored_window():
    image = _labelled_frame(4, 10)
    roi = np.array([[1, 3], [2, 5]])  # rows 1:3, raw cols 2:5

    out = crop_flipped_roi(image, roi)

    assert out.shape == (2, 3)
    # rows 1,2 preserved; raw cols {2,3,4} selected but in reversed order
    # because the frame was flipped before cropping.
    assert out[0].tolist() == [104, 103, 102]
    assert out[1].tolist() == [204, 203, 202]


def test_crop_flipped_roi_equals_crop_then_flip():
    """Flipping the whole frame then taking the mirrored window is the same
    physical window as cropping first then reversing its columns."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 1000, size=(6, 12)).astype(np.int32)
    roi = np.array([[2, 5], [3, 9]])

    out = crop_flipped_roi(image, roi)
    expected = np.flip(crop_to_roi(image, roi), 1)

    np.testing.assert_array_equal(out, expected)
