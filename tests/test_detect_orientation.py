"""Tests for scripts/detect_orientation.py.

Covers `_read_positions` — the HDF5 position-shape handling specific to this
script. The spatially-diverse sampling and the orientation sweep are tested in
ptychoml (`spatially_diverse_sample`, `autodetect_orientation`); the full CLI
needs a TRT engine and runs in real model loads.
"""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

# scripts/ isn't a package; load the module by path. Its top-level imports are
# argparse/json/sys/pathlib/h5py/numpy — ptychoml is imported lazily in main(),
# so this import works without a GPU/TRT.
_PATH = Path(__file__).resolve().parent.parent / "scripts" / "detect_orientation.py"
_spec = importlib.util.spec_from_file_location("detect_orientation", _PATH)
det = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(det)


def _h5(tmp_path, arr):
    p = tmp_path / "pos.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("points", data=np.asarray(arr))
    return h5py.File(p, "r")


def test_read_positions_2byN(tmp_path):
    arr = np.array([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    with _h5(tmp_path, arr) as f:
        x, y = det._read_positions(f, "points")
    np.testing.assert_array_equal(x, [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(y, [20, 21, 22, 23, 24])


def test_read_positions_Nby2(tmp_path):
    arr = np.array([[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]])
    with _h5(tmp_path, arr) as f:
        x, y = det._read_positions(f, "points")
    np.testing.assert_array_equal(x, [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(y, [20, 21, 22, 23, 24])


def test_read_positions_2by2_defaults_to_2byN(tmp_path):
    # ambiguous (2, 2): HXN (2, N) convention -> rows are x, y
    arr = np.array([[1, 2], [3, 4]])
    with _h5(tmp_path, arr) as f:
        x, y = det._read_positions(f, "points")
    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_array_equal(y, [3, 4])


def test_read_positions_rejects_1d(tmp_path):
    with _h5(tmp_path, np.arange(5)) as f:
        with pytest.raises(ValueError):
            det._read_positions(f, "points")


def test_read_positions_rejects_bad_shape(tmp_path):
    # neither axis is length 2
    with _h5(tmp_path, np.zeros((3, 4))) as f:
        with pytest.raises(ValueError):
            det._read_positions(f, "points")
