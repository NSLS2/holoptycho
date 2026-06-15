"""Tests for the iterative-engine orientation/direction split.

Covers ``holoptycho.orientation`` (pure numpy, no holoscan/cupy):
- D4_INVERSE round-trips every element;
- ``reorient_d4`` converts between any two D4 frames (the relative transform
  ImageSendOp applies to the engine's copy of diff_amp);
- the production property: applying the relative D4 AFTER fftshift equals
  preprocessing with the target orientation directly, for even square frames;
- ``reduce_d4_sequence`` closure / validation;
- ``compute_pos_bases`` under scan-direction sign flips;
and the ``--dp-orient-iterative`` / ``--x|y-direction-iterative`` config flags
(emitted only when set).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ptychoml")

from ptychoml.preprocess import D4_NAMES, apply_d4

from holoptycho.orientation import (
    D4_INVERSE,
    compute_pos_bases,
    reduce_d4_sequence,
    reorient_d4,
)


def _x(seed=0, n=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, n, n)).astype(np.float32)  # batch of even square


def test_d4_inverse_round_trips_all_names():
    x = _x()
    for name in D4_NAMES:
        assert name in D4_INVERSE
        np.testing.assert_array_equal(
            apply_d4(apply_d4(x, name), D4_INVERSE[name]), x
        )


def test_reorient_d4_maps_between_any_two_frames():
    x = _x(1)
    for a in D4_NAMES:
        for b in D4_NAMES:
            np.testing.assert_array_equal(
                reorient_d4(apply_d4(x, a), a, b), apply_d4(x, b)
            )


def test_relative_d4_commutes_with_fftshift_even_square():
    """The production identity: ImageSendOp re-orients diff_amp AFTER the
    fftshift in preprocess_diffraction; for even square frames the result must
    equal having preprocessed with the target orientation in the first place."""
    x = _x(2)
    for a in D4_NAMES:
        produced = np.fft.fftshift(apply_d4(x, a), axes=(-2, -1))
        for b in D4_NAMES:
            want = np.fft.fftshift(apply_d4(x, b), axes=(-2, -1))
            np.testing.assert_array_equal(reorient_d4(produced, a, b), want)


def test_reorient_does_not_mutate_input():
    x = _x(3)
    produced = apply_d4(x, "rot90_cw").copy()
    before = produced.copy()
    out = reorient_d4(produced, "rot90_cw", "fliplr")
    np.testing.assert_array_equal(produced, before)  # input untouched
    assert out is not produced


def test_reduce_d4_sequence_closure_over_all_pairs():
    probe = np.arange(12).reshape(3, 4)
    for a in D4_NAMES:
        for b in D4_NAMES:
            reduced = reduce_d4_sequence(f"{a},{b}")
            assert reduced in D4_NAMES
            got = apply_d4(apply_d4(probe, a), b)
            want = apply_d4(probe, reduced)
            assert got.shape == want.shape
            np.testing.assert_array_equal(got, want)


def test_reduce_d4_sequence_single_and_identity():
    for name in D4_NAMES:
        assert reduce_d4_sequence(name) == name
    assert reduce_d4_sequence("") == "identity"
    assert reduce_d4_sequence("rot90_cw,rot90_ccw") == "identity"
    assert reduce_d4_sequence("rot90_cw,rot90_cw") == "rot180"


def test_reduce_d4_sequence_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown D4"):
        reduce_d4_sequence("rot45")
    with pytest.raises(ValueError, match="Unknown D4"):
        reduce_d4_sequence("rot90_cw,bogus")


def test_compute_pos_bases_ascending_and_descending():
    pos0 = np.array([5.0, 1.0, 3.0])
    up = np.array([0.0, 1.0, 2.0])
    x_base, y_base = compute_pos_bases(pos0, up, y_range_um=2.0)
    assert x_base == 1.0 and y_base == 0.0
    down = up[::-1]
    x_base, y_base = compute_pos_bases(pos0, down, y_range_um=2.0)
    assert x_base == 1.0 and y_base == 0.0  # 2.0 (first) - 2.0 (range)


def test_compute_pos_bases_sign_flip_mirrors_offsets():
    """Flipping the sign convention must keep positions mapping into the same
    [0, range] window — the offsets come out mirrored, not out of range."""
    rng = np.random.default_rng(4)
    pos1 = np.sort(rng.uniform(0.0, 2.0, 16))      # ascending rows
    pos0 = rng.uniform(-3.0, 3.0, 16)
    xb, yb = compute_pos_bases(pos0, pos1, 2.0)
    xb_f, yb_f = compute_pos_bases(-pos0, -pos1, 2.0)
    off = pos1 - yb
    off_f = (-pos1) - yb_f
    assert off.min() >= 0 and off_f.max() <= 2.0 + 1e-9
    # y offsets mirror within the scan range (descending base shifts by range)
    np.testing.assert_allclose(off_f, 2.0 - off, atol=1e-9, rtol=0)
    # x offsets mirror within the observed span (base = min either way)
    span = pos0.max() - pos0.min()
    np.testing.assert_allclose((-pos0) - xb_f, span - (pos0 - xb), atol=1e-9)


# --- config flag emission -----------------------------------------------------

_CFT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "config_from_tiled.py"
_spec = importlib.util.spec_from_file_location("config_from_tiled", _CFT_PATH)
_cft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cft)


def _parse(argv):
    p = argparse.ArgumentParser()
    _cft.add_reconstruction_arguments(p)
    return p.parse_args(argv)


def test_iterative_flags_default_unset():
    args = _parse([])
    # dp_orient_iterative defaults to 'rot90_cw' — holoscan-framework's hardcoded
    # engine-DP np.rot90(axes=(2,1)); pass 'identity' to follow the shared
    # dp_orient + autodetect. The direction overrides stay unset by default.
    assert args.dp_orient_iterative == "rot90_cw"
    assert args.x_direction_iterative is None
    assert args.y_direction_iterative is None


def test_iterative_flags_parse():
    args = _parse([
        "--dp-orient-iterative", "rot90_cw,fliplr",
        "--x-direction-iterative", "1.0",
        "--y-direction-iterative", "-1.0",
    ])
    assert args.dp_orient_iterative == "rot90_cw,fliplr"
    assert args.x_direction_iterative == 1.0
    assert args.y_direction_iterative == -1.0
    # the sequence reduces to a valid single element downstream
    assert reduce_d4_sequence(args.dp_orient_iterative) in D4_NAMES


def test_dp_orient_flag_default_unset_and_parse():
    """--dp-orient unset (None) = the pipeline default 'identity' with the
    autodetect OFF; 'auto' opts in to the sweep; a D4 name pins it."""
    assert _parse([]).dp_orient is None
    assert _parse(["--dp-orient", "auto"]).dp_orient == "auto"
    assert _parse(["--dp-orient", "rot90_cw"]).dp_orient == "rot90_cw"


def test_dp_orient_config_key_absent_unless_set():
    """build_full_config must omit dp_orient when the flag is unset — the
    pipeline then applies its deterministic default ('identity', no sweep)."""
    import inspect

    src = inspect.getsource(_cft.build_full_config)
    assert 'if args.dp_orient is not None' in src


def test_config_keys_absent_unless_set():
    """build_full_config must omit the iterative keys when flags are unset —
    an absent dp_orient_iterative key is what keeps the engine following the
    shared dp_orient (and the orientation autodetect)."""
    import inspect

    src = inspect.getsource(_cft.build_full_config)
    # emission is conditional on the args being set
    assert 'if args.dp_orient_iterative is not None' in src
    assert 'if args.x_direction_iterative is not None' in src
    assert 'if args.y_direction_iterative is not None' in src
    # and round-trips as plain strings through JSON
    cfg = {"dp_orient_iterative": "rot90_ccw", "x_direction_iterative": "1.0"}
    assert json.loads(json.dumps(cfg)) == cfg
