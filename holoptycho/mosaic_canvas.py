"""Pure geometry/bookkeeping helpers for the streaming ViT mosaic canvas.

Kept free of Holoscan / Tiled imports so the canvas logic can be unit tested
without the full pipeline (``vit_inference`` performs import-time I/O setup
that needs a live Tiled endpoint).
"""
from __future__ import annotations

import numpy as np


def fit_slow_axis(frame_idx, slow_vals, *, tol, min_frames=512, max_iter=3):
    """Fit the monotonic slow-axis trajectory ``slow ≈ a + b·frame_idx``.

    A clean fly raster steps the slow axis once per fast-line, so the slow
    position is ~linear in absolute frame index. When the pipeline joins a
    scan mid-way, the first ~1-3 fast-lines arrive with a *stale* slow-axis
    encoder reading (≈ mid-range) before the position stream syncs to the true
    scan start — those frames sit at a slow value wildly inconsistent with
    their (very low) frame index, so they are large-residual outliers against
    the line. We fit by least squares, reject points whose residual exceeds
    ``tol``, and refit a few times so the small leading bogus run can't drag
    the line. The leading frames then fail the residual test and get dropped
    from the mosaic.

    Returns ``(a, b)`` once at least ``min_frames`` samples spanning a usable
    index range are available, else ``None`` (not enough to lock yet — caller
    keeps buffering). ``a`` is the slow value extrapolated to ``frame_idx==0``,
    ``b`` the per-frame slope (sign = scan direction).
    """
    fi = np.asarray(frame_idx, dtype=np.float64)
    sv = np.asarray(slow_vals, dtype=np.float64)
    if len(fi) < min_frames or fi.ptp() < 1:
        return None
    keep = np.ones(len(fi), dtype=bool)
    a = b = 0.0
    for _ in range(max_iter):
        b, a = np.polyfit(fi[keep], sv[keep], 1)
        resid = np.abs(sv - (a + b * fi))
        new_keep = resid <= tol
        if new_keep.sum() < min_frames // 2:
            # Refit would throw away too much — keep the previous fit.
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(a), float(b)


def slow_gate_mask(frame_idx, slow_vals, a, b, tol):
    """Boolean mask of frames consistent with the locked slow-axis line.

    ``True`` where ``|slow - (a + b·frame_idx)| <= tol`` — i.e. the frame's
    slow position matches where the monotonic raster says that frame index
    should be. Leading mid-scan-join settling frames (stale slow at the wrong
    index) fall outside ``tol`` and are excluded.
    """
    fi = np.asarray(frame_idx, dtype=np.float64)
    sv = np.asarray(slow_vals, dtype=np.float64)
    return np.abs(sv - (a + b * fi)) <= tol


def partition_pending(pending_frames, positions_um):
    """Split buffered ``(patch, frame_index)`` pairs by position availability.

    ViT batches can arrive before ``PointProcessorOp`` has filled the matching
    scan positions, so frames whose position is still NaN (or whose index is
    beyond the currently-loaded positions) are buffered and retried on later
    batches. A frame is "ready" once ``positions_um[frame_index]`` is in range
    and fully finite.

    Returns ``(now_ready, still_pending)``, each a list of ``(patch, index)``
    in the original order.
    """
    n = len(positions_um)
    now_ready, still_pending = [], []
    for patch, idx in pending_frames:
        if idx < n and np.isfinite(positions_um[idx]).all():
            now_ready.append((patch, idx))
        else:
            still_pending.append((patch, idx))
    return now_ready, still_pending


def estimate_canvas_mid(obs_min, obs_max, obs_range, cmd_range, finite_vals):
    """Estimate the mosaic-canvas centre along one axis.

    When the canvas is allocated before the slow axis has traversed more than
    half its commanded range (e.g. the slow/y axis barely moves during the
    first ``PointProcessorOp`` batch of a raster scan), the naive midpoint of
    the observed positions is biased toward the scan start, which pushes the
    far end of the scan off the fixed-size canvas.

    If the observed range covers at least half the commanded range (or the
    commanded range is unknown), use the plain observed midpoint. Otherwise
    infer the scan direction from the sign of ``finite_vals[-1] -
    finite_vals[0]`` and project the centre to the midpoint of the *commanded*
    range from the appropriate end (a flat/zero direction is treated as
    negative, matching the HXN default scan sense).
    """
    if obs_range >= cmd_range * 0.5 or cmd_range <= 0:
        return 0.5 * (obs_min + obs_max)
    direction = float(np.sign(finite_vals[-1] - finite_vals[0])) or -1.0
    start = obs_max if direction < 0 else obs_min
    return start + direction * cmd_range / 2.0
