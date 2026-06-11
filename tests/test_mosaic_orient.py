"""Tests for the ViT mosaic display reorientation (issue #50).

MosaicWriterOp reorients the stitched canvas (phase + amplitude) with a D4
just before the Tiled upload so it matches the beamline view (top→bottom,
left→right). The default is `antitranspose`, which the issue describes as
"rotate 90° CCW, then horizontal flip".
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ptychoml")
pytest.importorskip("tiled")

from ptychoml.preprocess import D4_NAMES, apply_d4


def test_default_antitranspose_equals_rot90ccw_then_hflip():
    """The default `antitranspose` is exactly the issue's reorientation:
    rotate 90° CCW, then horizontal flip (fliplr)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 5))  # non-square, non-symmetric canvas
    rot90ccw_then_hflip = apply_d4(apply_d4(x, "rot90_ccw"), "fliplr")
    np.testing.assert_array_equal(apply_d4(x, "antitranspose"), rot90ccw_then_hflip)


def test_antitranspose_swaps_canvas_shape():
    x = np.zeros((7, 5), dtype=np.float32)
    assert apply_d4(x, "antitranspose").shape == (5, 7)


def test_identity_is_noop():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((7, 5)).astype(np.float32)
    np.testing.assert_array_equal(apply_d4(x, "identity"), x)


def test_reorientation_is_invertible_to_stitching_frame():
    """rot90_ccw∘hflip = antitranspose is an involution, so the un-reorient is
    the same op — useful for reasoning about a downstream consumer."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, 9)).astype(np.float32)
    once = apply_d4(x, "antitranspose")
    twice = apply_d4(once, "antitranspose")
    np.testing.assert_array_equal(twice, x)


# --- config flag --------------------------------------------------------------

_CFT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "config_from_tiled.py"
_spec = importlib.util.spec_from_file_location("config_from_tiled", _CFT_PATH)
_cft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cft)


def _parse(argv):
    p = argparse.ArgumentParser()
    _cft.add_reconstruction_arguments(p)
    return p.parse_args(argv)


def test_mosaic_orient_flag_default_and_override():
    assert _parse([]).mosaic_orient == "antitranspose"
    assert _parse(["--mosaic-orient", "identity"]).mosaic_orient == "identity"
    # default is a valid D4 name the pipeline will accept
    assert _parse([]).mosaic_orient in D4_NAMES


def test_mosaic_orient_survives_json_roundtrip():
    cfg = {"mosaic_orient": "antitranspose"}
    assert json.loads(json.dumps(cfg))["mosaic_orient"] == "antitranspose"
