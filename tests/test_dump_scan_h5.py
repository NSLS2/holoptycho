"""Tests for scripts/dump_scan_h5.py geometry helpers.

Covers the HXN-specific unit conversions (`_ratio_from_scale`,
`_energy_from_dcm_th`, `_lambda_from_energy`). The D4 orientation now comes
from `ptychoml.apply_d4` (tested in ptychoml); the Tiled fetch + HDF5 dump
path is integration-only.
"""
import importlib.util
import math
from pathlib import Path

import pytest

# dump_scan_h5 imports tiled + ptychoml at module load.
pytest.importorskip("tiled")

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dump_scan_h5.py"
_spec = importlib.util.spec_from_file_location("dump_scan_h5", _PATH)
dsh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsh)


def test_ratio_from_scale_none_and_zero_default_to_one():
    # v in (None, 0) -> treated as 1.0 -> -1/10000
    assert dsh._ratio_from_scale(None) == pytest.approx(-1e-4)
    assert dsh._ratio_from_scale(0) == pytest.approx(-1e-4)


def test_ratio_from_scale_sign_and_magnitude():
    assert dsh._ratio_from_scale(5000) == pytest.approx(-0.5)
    assert dsh._ratio_from_scale(-5000) == pytest.approx(0.5)


def test_lambda_from_energy():
    assert dsh._lambda_from_energy(12.0) == pytest.approx(1.2398 / 12.0)


def test_energy_from_dcm_th_matches_formula():
    deg = 11.0
    expected = 12.39842 / (2.0 * 3.1355893 * math.sin(math.radians(deg)))
    assert dsh._energy_from_dcm_th(deg) == pytest.approx(expected)


def test_energy_from_dcm_th_decreases_with_angle():
    # Bragg: larger DCM angle -> lower energy
    assert dsh._energy_from_dcm_th(8.0) > dsh._energy_from_dcm_th(12.0)
