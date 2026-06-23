"""Tests for lossless DP auto-centering (compute_center_box) and the
``--auto-center-dp`` config wiring.

The centering math lives in ``holoptycho.preprocess.compute_center_box`` (pure
numpy). ImageBatchOp segments an UN-flipped headroom window to find the beam,
builds a raw-left ``ny x nx`` crop box centered on it, and crops every batch to
that box — no ``np.roll``, no zero-fill. These tests cover the box math, the
flip-independence of the crop offset, clamping, the no-blob fallback, and the
flag → real-bool config round-trip.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# preprocess.py imports holoscan + cupy at module level; config_from_tiled
# imports tiled. All present in the default `pixi run test` env.
pytest.importorskip("holoscan")
pytest.importorskip("cupy")
pytest.importorskip("tiled")

from ptychoml.preprocess import crop_to_roi
from holoptycho.preprocess import compute_center_box


def _frame_with_blob(H, W, by, bx, r=4.0):
    """uint32 frame with a Gaussian blob centered at (by, bx)."""
    yy, xx = np.ogrid[:H, :W]
    g = np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * r * r)))
    return (g * 1000.0).astype(np.uint32)


def _configured(cy, cx, nx, ny):
    """Configured ny x nx ROI centered at (cy, cx), raw-left."""
    return np.array([[cy - ny // 2, cy + ny // 2], [cx - nx // 2, cx + nx // 2]])


def _headroom(roi, M, H, W):
    return np.array([
        [max(0, int(roi[0, 0]) - M), min(H, int(roi[0, 1]) + M)],
        [max(0, int(roi[1, 0]) - M), min(W, int(roi[1, 1]) + M)],
    ])


def _hbatch(frame, hroi, n=4):
    return np.stack([np.asarray(crop_to_roi(frame, hroi))] * n)


def test_box_centers_offcenter_blob_losslessly():
    H = W = 100
    nx = ny = 32
    by, bx = 60, 35  # beam off-center vs the configured ROI at the frame center
    roi = _configured(50, 50, nx, ny)
    hroi = _headroom(roi, 24, H, W)
    frame = _frame_with_blob(H, W, by, bx)

    box, clamped, *_ = compute_center_box(_hbatch(frame, hroi), hroi, nx, ny)

    assert not clamped
    assert box[0, 1] - box[0, 0] == ny and box[1, 1] - box[1, 0] == nx
    # box centered on the beam (within 1px rounding)
    assert abs((box[0, 0] + box[0, 1]) / 2 - by) <= 1
    assert abs((box[1, 0] + box[1, 1]) / 2 - bx) <= 1
    # lossless: cropping the box yields real pixels with the beam at the centre,
    # and no zero-fill band (every row/col has signal from the blob's vicinity).
    crop = np.asarray(crop_to_roi(frame, box))
    assert crop.shape == (ny, nx)
    pk = np.unravel_index(int(np.argmax(crop)), crop.shape)
    assert abs(pk[0] - ny // 2) <= 1 and abs(pk[1] - nx // 2) <= 1


def test_box_is_plain_global_crop():
    """The box is a plain global ROI: auto-centering runs on the already
    coordinate-corrected frame, so the box feeds a plain crop_to_roi and the
    beam lands at the centre (no flip/mirror logic)."""
    H = W = 100
    nx = ny = 32
    by, bx = 58, 38
    roi = _configured(50, 50, nx, ny)
    hroi = _headroom(roi, 24, H, W)
    frame = _frame_with_blob(H, W, by, bx)

    box, *_ = compute_center_box(_hbatch(frame, hroi), hroi, nx, ny)
    cropped = np.asarray(crop_to_roi(frame, box))

    assert cropped.shape == (ny, nx)
    pk = np.unravel_index(int(np.argmax(cropped)), cropped.shape)
    assert abs(pk[0] - ny // 2) <= 1 and abs(pk[1] - nx // 2) <= 1


def test_box_clamped_when_beam_beyond_headroom():
    H = W = 100
    nx = ny = 32
    roi = _configured(50, 50, nx, ny)
    hroi = _headroom(roi, 24, H, W)  # window rows/cols [10:90]
    frame = _frame_with_blob(H, W, 88, 88)  # near the window edge

    box, clamped, *_ = compute_center_box(_hbatch(frame, hroi), hroi, nx, ny)

    assert clamped
    assert box[0, 1] - box[0, 0] == ny and box[1, 1] - box[1, 0] == nx
    # stays inside the headroom window
    assert box[0, 0] >= hroi[0, 0] and box[0, 1] <= hroi[0, 1]
    assert box[1, 0] >= hroi[1, 0] and box[1, 1] <= hroi[1, 1]


def test_no_blob_returns_none():
    H = W = 100
    nx = ny = 32
    roi = _configured(50, 50, nx, ny)
    hroi = _headroom(roi, 24, H, W)
    frame = np.zeros((H, W), dtype=np.uint32)

    box, clamped, *_ = compute_center_box(_hbatch(frame, hroi), hroi, nx, ny)

    assert box is None and clamped is False


def test_centered_beam_box_equals_configured_roi():
    """OFF-equivalence proxy: when the beam already sits at the configured ROI
    centre, the computed box equals the configured ROI — i.e. auto-centering a
    centered beam is a no-op, reducing to the fixed-ROI crop the OFF path uses."""
    H = W = 100
    nx = ny = 32
    roi = _configured(50, 50, nx, ny)
    hroi = _headroom(roi, 24, H, W)
    frame = _frame_with_blob(H, W, 50, 50)

    box, clamped, *_ = compute_center_box(_hbatch(frame, hroi), hroi, nx, ny)

    assert not clamped
    assert box.tolist() == roi.tolist()


# --- config flag / real-bool round-trip --------------------------------------

_CFT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "config_from_tiled.py"
_spec = importlib.util.spec_from_file_location("config_from_tiled", _CFT_PATH)
_cft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cft)


def _parse(argv):
    p = argparse.ArgumentParser()
    _cft.add_reconstruction_arguments(p)
    return p.parse_args(argv)


def test_full_frame_search_finds_offcenter_beam():
    """headroom < 0 searches the WHOLE frame, so auto-centering finds a beam far
    from any configured offset — the full-detector (no-ROI) case."""
    H, W = 300, 320
    nx = ny = 64
    by, bx = 40, 280  # beam in a corner, nowhere near the frame centre
    frame = _frame_with_blob(H, W, by, bx, r=5.0)
    hroi = np.array([[0, H], [0, W]])  # whole-frame headroom (the M<0 result)

    box, clamped, *_ = compute_center_box(_hbatch(frame, hroi), hroi, nx, ny)

    assert not clamped
    assert abs((box[0, 0] + box[0, 1]) / 2 - by) <= 1
    assert abs((box[1, 0] + box[1, 1]) / 2 - bx) <= 1


def _auto_center_decision(args):
    """Mirror of build_full_config: auto-center on unless a crop ROI was passed."""
    roi_passed = args.batch_x0 is not None and args.batch_y0 is not None
    return bool(args.auto_center_dp or not roi_passed)


def test_auto_center_defaults_on_without_roi():
    assert _auto_center_decision(_parse([])) is True                       # no ROI
    assert _auto_center_decision(_parse(["--auto-center-dp"])) is True
    assert _auto_center_decision(
        _parse(["--batch-x0", "10", "--batch-y0", "20"])) is False         # manual ROI
    assert _auto_center_decision(
        _parse(["--batch-x0", "10", "--batch-y0", "20", "--auto-center-dp"])) is True


def test_detector_orientation_default_and_override():
    assert _parse([]).detector_orientation == "fliplr"          # LIVE default
    assert _parse(["--detector-orientation", "identity"]).detector_orientation == "identity"


def test_auto_center_flag_default_off():
    args = _parse([])
    assert args.auto_center_dp is False
    assert args.auto_center_headroom is None


def test_auto_center_flag_on():
    args = _parse(["--auto-center-dp", "--auto-center-headroom", "48"])
    assert args.auto_center_dp is True
    assert args.auto_center_headroom == 48


def test_auto_center_bool_survives_config_roundtrip():
    """Emitting a real bool (not the string 'false') round-trips through JSON and
    the pipeline's ast.literal_eval coercion correctly — lowercase 'false' would
    be mis-read as truthy by bool()."""
    for v in (True, False):
        back = json.loads(json.dumps({"auto_center_dp": v}))["auto_center_dp"]
        assert back is v
        # _coerce_config_value returns non-str values as-is; bool() is correct.
        assert bool(back) is v
    # demonstrate the footgun the real-bool choice avoids:
    assert bool("false") is True  # would wrongly enable if emitted as a string
    assert ast.literal_eval("False") is False  # capitalized would be fine too
