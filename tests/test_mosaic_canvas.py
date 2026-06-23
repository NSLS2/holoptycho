"""Tests for the ViT mosaic-canvas logic (holoptycho-specific)."""
import numpy as np

from holoptycho.mosaic_canvas import (
    estimate_canvas_mid,
    fit_slow_axis,
    partition_pending,
    slow_gate_mask,
)


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


# ----- partition_pending ----------------------------------------------------

def _frame(idx, val):
    # patch tagged with a value so we can confirm it travels with its index
    return (np.full((2, 2), float(val), dtype=np.float32), idx)


def test_partition_pending_all_ready():
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pending = [_frame(0, 10), _frame(2, 12)]

    ready, still = partition_pending(pending, positions)

    assert [i for _, i in ready] == [0, 2]
    assert still == []


def test_partition_pending_all_waiting_on_nan():
    positions = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    pending = [_frame(0, 10), _frame(1, 11)]

    ready, still = partition_pending(pending, positions)

    assert ready == []
    assert [i for _, i in still] == [0, 1]


def test_partition_pending_mixed_and_preserves_order():
    positions = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]])
    pending = [_frame(0, 10), _frame(1, 11), _frame(2, 12)]

    ready, still = partition_pending(pending, positions)

    assert [i for _, i in ready] == [0, 2]   # finite positions, in order
    assert [i for _, i in still] == [1]       # NaN position stays pending


def test_partition_pending_out_of_range_index_stays_pending():
    # index 5 is beyond the loaded positions -> not yet known
    positions = np.array([[1.0, 1.0], [2.0, 2.0]])
    pending = [_frame(0, 10), _frame(5, 99)]

    ready, still = partition_pending(pending, positions)

    assert [i for _, i in ready] == [0]
    assert [i for _, i in still] == [5]


def test_partition_pending_partial_nan_row_not_ready():
    # one axis finite, the other NaN -> not ready (needs both)
    positions = np.array([[1.0, np.nan]])
    pending = [_frame(0, 10)]

    ready, still = partition_pending(pending, positions)

    assert ready == []
    assert [i for _, i in still] == [0]


def test_partition_pending_carries_patch_with_index():
    positions = np.array([[1.0, 1.0], [np.nan, np.nan]])
    pending = [_frame(0, 10), _frame(1, 11)]

    ready, still = partition_pending(pending, positions)

    # the ready patch is the one tagged 10 (frame index 0)
    assert ready[0][0][0, 0] == 10.0
    assert still[0][0][0, 0] == 11.0


def test_partition_pending_empty():
    assert partition_pending([], np.zeros((3, 2))) == ([], [])


# ----- slow-axis gate (mid-scan-join settling rejection) --------------------

def _raster_slow(n_bogus=200, n_lines=210, period=280, rng_um=21.0, seed=0):
    """Build (frame_idx, slow) for a raster with a leading bogus settling line.

    The first ``n_bogus`` frames sit at slow≈0 (stale encoder reading on a
    mid-scan join); the real scan then ramps slow from +rng/2 down to -rng/2,
    one step per fast-line.
    """
    rng = np.random.default_rng(seed)
    idx, slow = [], []
    for i in range(n_bogus):
        idx.append(i)
        slow.append(0.0 + rng.normal(0, 0.05))
    f = n_bogus
    half = rng_um / 2.0
    for ln in range(n_lines):
        s = half - ln * (rng_um / (n_lines - 1))
        for _ in range(period):
            idx.append(f)
            slow.append(s + rng.normal(0, 0.02))
            f += 1
    return np.array(idx), np.array(slow)


def test_fit_slow_axis_rejects_leading_bogus_line():
    idx, slow = _raster_slow()
    m = idx < 1500  # warm-up window: bogus line + a few clean lines
    tol = 0.05 * 28.0
    fit = fit_slow_axis(idx[m], slow[m], tol=tol, min_frames=512)
    assert fit is not None
    a, b = fit
    assert a > 9.0          # extrapolated start ≈ +10.5
    assert b < 0            # decreasing scan
    keep = slow_gate_mask(idx[m], slow[m], a, b, tol)
    bogus = idx[m] < 200
    assert keep[bogus].sum() == 0        # all bogus dropped
    assert keep[~bogus].all()            # all clean kept


def test_fit_slow_axis_returns_none_when_too_few():
    idx = np.arange(100)
    slow = np.linspace(10, -10, 100)
    assert fit_slow_axis(idx, slow, tol=1.0, min_frames=512) is None


def test_slow_axis_better_than_fast_axis():
    # The fast axis sweeps its full range each line -> not linear in frame idx,
    # so it yields far fewer inliers than the genuinely-linear slow axis.
    idx, slow = _raster_slow()
    rng = np.random.default_rng(1)
    fast = []
    for _ in range(200):
        fast.append(rng.uniform(-14, 14))
    for _ln in range(210):
        for k in range(280):
            fast.append(14 - 28 * (k / 280) + rng.normal(0, 0.02))
    fast = np.array(fast)
    m = idx < 1500
    tol = 0.05 * 28.0
    fs = fit_slow_axis(idx[m], slow[m], tol=tol, min_frames=512)
    ff = fit_slow_axis(idx[m], fast[m], tol=tol, min_frames=512)
    frac_slow = slow_gate_mask(idx[m], slow[m], *fs, tol).mean()
    frac_fast = (
        slow_gate_mask(idx[m], fast[m], *ff, tol).mean() if ff else 0.0
    )
    assert frac_slow > 0.6
    assert frac_slow > frac_fast

