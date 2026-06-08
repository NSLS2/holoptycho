"""Tests for the ViT mosaic-canvas logic (holoptycho-specific)."""
import numpy as np

from holoptycho.mosaic_canvas import estimate_canvas_mid, partition_pending


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
