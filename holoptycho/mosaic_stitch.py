"""Fourier-shift sub-pixel patch placement for ViT mosaic stitching.

Algorithm adapted from the ptycho-vit reference implementation
(``utils/ptychi_utils.py:place_patches_fourier_shift`` on the holostitching
branch: https://github.com/SYNAPS-I/ptycho-vit), in turn adapted from Ming
Du's pty-chi (https://github.com/AdvancedPhotonSource/pty-chi). The trained
ViT outputs are reassembled this way at evaluation time
(``run_inference.py``); doing anything different recon-side produces an
image inconsistent with what the network was supervised against.

Implemented in numpy (not torch) — ``vit_inference.py`` deliberately avoids
torch to stay container-light. The algorithm is just FFT + scatter-add +
divide, no autograd needed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _fourier_shift(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Sub-pixel shift each (H, W) plane of ``images`` by ``shifts[i] = (dy, dx)``.

    Parameters
    ----------
    images : (N, H, W) float32 array.
    shifts : (N, 2) float array, in pixels (y, x).

    Returns
    -------
    (N, H, W) float32 array, same dtype as input.
    """
    h, w = images.shape[-2:]
    ft = np.fft.fft2(images.astype(np.complex128, copy=False))
    freq_y = np.fft.fftfreq(h)[:, None]   # (H, 1)
    freq_x = np.fft.fftfreq(w)[None, :]   # (1, W)
    # Phase ramp per image, vectorised: shape (N, H, W).
    phase = -2j * np.pi * (
        shifts[:, 0, None, None] * freq_y[None, :, :]
        + shifts[:, 1, None, None] * freq_x[None, :, :]
    )
    ft = ft * np.exp(phase)
    out = np.fft.ifft2(ft)
    return out.real.astype(images.dtype, copy=False)


def place_patches_fourier_shift(
    image: np.ndarray,
    positions: np.ndarray,
    patches: np.ndarray,
    pad: int = 1,
) -> np.ndarray:
    """Add patches into ``image`` with sub-pixel Fourier shifts.

    Mirrors ``ptycho-vit:place_patches_fourier_shift`` with ``op="add"`` and
    ``adjoint_mode=False``: each patch is over-extracted by ``pad`` pixels,
    Fourier-shifted by its fractional position, then center-cropped back to
    its original size before scatter-add. The over-extract+crop discards
    Fourier wrap-around artifacts at the patch boundary.

    Parameters
    ----------
    image : (H, W) float32 array (mutated and returned).
    positions : (N, 2) float array of (y, x) center positions in pixels.
        Origin at image (0, 0).
    patches : (N, h, w) float32 array.
    pad : int
        Pixels to over-extract per side before the FFT shift; cropped after.
        Default 1 matches the reference. Stitcher uses 32 to match training.

    Returns
    -------
    Updated image (same shape as input). The input is mutated in place when
    no out-of-bounds padding is required; otherwise a new array is returned
    and the caller should reassign.
    """
    ph, pw = patches.shape[-2:]

    sys_float = positions[:, 0] - (ph - 1.0) / 2.0
    sxs_float = positions[:, 1] - (pw - 1.0) / 2.0

    # Over-extract by `pad` per side (negative patch_padding in reference).
    sys = np.floor(sys_float).astype(np.int64) + pad
    sxs = np.floor(sxs_float).astype(np.int64) + pad
    eys = sys + ph - 2 * pad
    exs = sxs + pw - 2 * pad

    fractional = np.stack(
        [sys_float - sys + pad, sxs_float - sxs + pad], axis=-1
    ).astype(np.float64)

    pad_lengths = [
        max(int(-sys.min()), 0),
        max(int(eys.max() - image.shape[0]), 0),
        max(int(-sxs.min()), 0),
        max(int(exs.max() - image.shape[1]), 0),
    ]
    if any(pad_lengths):
        image = np.pad(
            image,
            ((pad_lengths[0], pad_lengths[1]), (pad_lengths[2], pad_lengths[3])),
            mode="constant",
        )
        sys = sys + pad_lengths[0]
        sxs = sxs + pad_lengths[2]

    if not np.allclose(fractional, 0.0, atol=1e-7):
        patches = _fourier_shift(patches, fractional)

    if pad > 0:
        patches = patches[:, pad:ph - pad, pad:pw - pad]
        ph_eff = ph - 2 * pad
        pw_eff = pw - 2 * pad
    else:
        ph_eff = ph
        pw_eff = pw

    # Add each patch into the canvas. Within a single patch the destination
    # indices are unique (a contiguous (ph_eff, pw_eff) slice), so a plain
    # ``+=`` accumulates correctly. Overlaps between patches are handled by
    # the sequential loop. This is much faster than ``np.add.at`` over a
    # huge flat-index array.
    for i in range(len(patches)):
        image[sys[i]:sys[i] + ph_eff, sxs[i]:sxs[i] + pw_eff] += patches[i]

    if any(pad_lengths):
        # Crop the working canvas back to the original shape.
        image = image[
            pad_lengths[0]: image.shape[0] - pad_lengths[1],
            pad_lengths[2]: image.shape[1] - pad_lengths[3],
        ]
    return image


def stitch_batch_into(
    canvas: np.ndarray,
    counts: np.ndarray,
    patches: np.ndarray,
    positions_px: np.ndarray,
    *,
    pad: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate one batch of cropped patches into (canvas, counts).

    ``patches`` should already be center-cropped (the caller decides
    ``inner_crop``). ``positions_px`` is (N, 2) in canvas pixel coordinates,
    (y, x), pointing at the patch centers.

    Scatter-add is associative, so per-batch accumulation gives the same
    result (up to FFT noise) as one-shot stitching of all patches.

    Returns the (possibly reassigned) canvas and counts. When the
    pre-allocated canvas already covers all positions, the input arrays are
    mutated in place and returned unchanged.
    """
    canvas = place_patches_fourier_shift(canvas, positions_px, patches, pad=pad)
    counts = place_patches_fourier_shift(
        counts, positions_px, np.ones_like(patches), pad=pad
    )
    return canvas, counts
