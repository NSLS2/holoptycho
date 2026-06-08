"""Pure geometry helpers for the streaming ViT mosaic canvas.

Kept free of Holoscan / Tiled imports so the canvas-sizing logic can be unit
tested without the full pipeline (``vit_inference`` performs import-time I/O
setup that needs a live Tiled endpoint).
"""
from __future__ import annotations

import numpy as np


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
