"""Orientation / scan-direction helpers for the iterative-vs-ViT split.

Pure numpy (no holoscan/cupy imports) so unit tests can exercise the math
without the GPU stack — same extraction pattern as ``mosaic_canvas.py``.

The pipeline applies one ``dp_orient`` in ``ImagePreprocessorOp`` and fans the
result out to both engines. ``dp_orient_iterative`` re-orients only the
iterative engine's copy by composing the inverse of the orientation a batch was
produced with and the desired target. D4 (the 8 axis-aligned rotations/flips)
is a closed group, so any such composition — and any user-supplied sequence of
D4 names — reduces to a single D4 element.
"""

import numpy as np

from ptychoml.preprocess import D4_NAMES, apply_d4

# Inverse of each D4 element: the quarter-turns invert each other; the six
# remaining elements (identity, the flips, rot180, the transposes) are
# involutions.
D4_INVERSE = {
    'identity': 'identity',
    'fliplr': 'fliplr',
    'flipud': 'flipud',
    'rot180': 'rot180',
    'transpose': 'transpose',
    'antitranspose': 'antitranspose',
    'rot90_cw': 'rot90_ccw',
    'rot90_ccw': 'rot90_cw',
}

# Non-square labeled probe: distinct entries + distinct dims disambiguate all
# 8 D4 elements by (shape, values).
_PROBE = np.arange(6, dtype=np.int64).reshape(2, 3)


def reduce_d4_sequence(names) -> str:
    """Reduce a D4 name or comma-separated sequence to one D4 element.

    Accepts ``"rot90_cw"`` or ``"rot90_cw,fliplr"`` (applied left to right) or
    an iterable of names. Because D4 is closed, the composition always equals
    exactly one of the 8 named elements; we find it by applying the sequence to
    a labeled non-square probe and matching the result (shape-aware) against
    each single-element transform. Raises ``ValueError`` on unknown names.
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(',') if n.strip()]
    names = list(names)
    if not names:
        return 'identity'
    for n in names:
        if n not in D4_NAMES:
            raise ValueError(
                f"Unknown D4 transform: {n!r}; valid names are {D4_NAMES!r}"
            )
    composed = _PROBE
    for n in names:
        composed = apply_d4(composed, n)
    for candidate in D4_NAMES:
        ref = apply_d4(_PROBE, candidate)
        if composed.shape == ref.shape and np.array_equal(composed, ref):
            return candidate
    raise AssertionError("D4 composition did not reduce to a D4 element")


def reorient_d4(batch: np.ndarray, from_name: str, to_name: str) -> np.ndarray:
    """Re-orient ``batch`` (last two axes) from one D4 frame to another.

    Equivalent to undoing ``from_name`` and applying ``to_name``. Returns a new
    array view chain — NEVER mutates ``batch`` in place (in the pipeline the
    input object is shared with the ViT branch).
    """
    return apply_d4(apply_d4(batch, D4_INVERSE[from_name]), to_name)


def compute_pos_bases(pos0, pos1, y_range_um: float):
    """Base offsets for converting scan positions (um) to engine pixel coords.

    Mirrors the historical ``PointProcessorOp`` logic, computed on the first
    position chunk: the x base is the minimum fast-axis position; the y base is
    the first slow-axis position, shifted down by the scan range when the rows
    run in descending order (so positions map into [0, y_range] either way).
    """
    x_base = float(np.min(pos0))
    y_base = float(pos1[0])
    if pos1[-1] < pos1[0]:
        y_base -= float(y_range_um)
    return x_base, y_base
