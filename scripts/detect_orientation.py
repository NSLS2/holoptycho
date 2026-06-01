"""Auto-detect ``dp_orient`` for the holoptycho scan config.

Reads a recorded scan from HDF5, runs ``ptychoml.autodetect_orientation``
against the live model's TRT engine (which carries the probe baked in
via ``convert_pt_to_onnx.py --probe``), and prints / writes a JSON snippet
with the winning ``dp_orient`` to merge into the scan config.

The other orientation knobs — ``patch_flip``, ``x_direction``,
``y_direction``, ``position_swap_xy`` — are not auto-detected: forward
physics doesn't constrain them (patch_flip mis-aligns the probe;
positions don't enter the per-frame forward model), so they stay as
manual operator preferences in the scan JSON.

Workflow:

    1. Export the ONNX with the matching probe baked in:

        python ptycho-vit/scripts/convert_pt_to_onnx.py \\
            --config config.yaml --checkpoint best.pth \\
            --output model.onnx --output-kind amp_phase \\
            --probe probe.npy

    2. Build the .engine from that ONNX.

    3. Record a scan (or use a saved replay HDF5).

    4. Run this script once per detector/optics change:

        python scripts/detect_orientation.py \\
            --data /path/to/dp.hdf5 \\
            --engine /path/to/model.engine \\
            --normalization 1.5e5 \\
            --hot-pixel-count-threshold 50000 \\
            --output orientation.json

    5. Merge ``dp_orient`` from ``orientation.json`` into your scan config
       and restart holoptycho.

If the engine has no baked probe and no ``--probe`` override is given,
the script errors out: this tool only does forward-physics scoring.

Source dataset conventions (override with ``--intensity-key`` /
``--positions-key``):

    /dp       — (N, H, W) detector counts (intensity in detector frame)
    /points   — (2, N) or (N, 2) scan positions in microns; column 0 = x

The ``--input-kind`` switch lets you point at an amplitude dataset (e.g.
HXN ``/diffamp``); the script squares it back to intensity before
preprocessing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _read_positions(f, key):
    arr = np.asarray(f[key][()])
    if arr.ndim != 2:
        raise ValueError(f"{key!r} must be 2D; got shape {arr.shape}")
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        return arr[0], arr[1]
    if arr.shape[1] == 2 and arr.shape[0] != 2:
        return arr[:, 0], arr[:, 1]
    if arr.shape == (2, 2):
        # Ambiguous (2, 2): default to the (2, N) convention used by HXN.
        return arr[0], arr[1]
    raise ValueError(
        f"{key!r} must be (2, N) or (N, 2); got {arr.shape}"
    )


def _spatially_diverse_sample(x, y, n_target, rng):
    """Spatially-spread subset selector. Avoids the failure mode where a
    uniform random sample clusters in one part of the scan area —
    coincidentally symmetric structure there could make a wrong
    orientation score well. Bucketing the scan bounding box and picking
    one position per occupied cell guarantees coverage.
    """
    n = len(x)
    if n_target >= n:
        return np.arange(n)
    grid_n = max(1, int(np.ceil(np.sqrt(n_target))))
    x_edges = np.linspace(x.min(), x.max() + 1e-12, grid_n + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-12, grid_n + 1)
    bx = np.clip(np.searchsorted(x_edges, x, side='right') - 1, 0, grid_n - 1)
    by = np.clip(np.searchsorted(y_edges, y, side='right') - 1, 0, grid_n - 1)
    bucket = by * grid_n + bx
    chosen = [rng.choice(np.where(bucket == b)[0]) for b in np.unique(bucket)]
    return np.sort(np.array(chosen))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Auto-detect orientation parameters for a holoptycho scan "
            "config by sweeping (dp_orient × patch_flip × position_map)."
        ),
    )
    parser.add_argument("--data", required=True, help="HDF5 with frames + positions.")
    parser.add_argument("--engine", required=True, help="TensorRT .engine path.")
    parser.add_argument("--output", required=True, help="JSON output path for the winning candidate.")
    parser.add_argument(
        "--intensity-key", default="dp",
        help="HDF5 key for the frame stack (default: 'dp').",
    )
    parser.add_argument(
        "--positions-key", default="points",
        help="HDF5 key for scan positions (default: 'points').",
    )
    parser.add_argument(
        "--input-kind", choices=("intensity", "amplitude"), default="intensity",
        help="Whether the source frames are counts or sqrt(counts).",
    )
    parser.add_argument(
        "--n-eval", type=int, default=64,
        help="Subsampled frame count for scoring (default: 64).",
    )
    parser.add_argument(
        "--normalization", type=float, default=None,
        help="Per-scan max intensity (hot pixels excluded). Computed from "
             "the input if omitted.",
    )
    parser.add_argument(
        "--scale", type=float, default=10000.0,
        help="Global scale factor (ptycho-vit default: 10000.0).",
    )
    parser.add_argument(
        "--hot-pixel-count-threshold", type=float, default=None,
        help="Photon-count threshold for hot-pixel zeroing. Omit to disable.",
    )
    parser.add_argument(
        "--fftshift", choices=("auto", "on", "off"), default="auto",
        help="DC-convention control for preprocess_diffraction. 'auto' "
             "(default) detects whether the central beam is at the corners "
             "and fftshifts only when it is; 'on' forces a shift; 'off' "
             "skips it. Pass 'off' to reproduce the historical bool=False "
             "behaviour of this flag.",
    )
    parser.add_argument(
        "--probe", default=None,
        help="Override probe .npy (single mode complex). When the engine was "
             "exported with --probe via convert_pt_to_onnx, the baked probe "
             "is used automatically and this flag is unnecessary. Pass this "
             "only to test an engine without a baked probe, or to override "
             "the baked one for debugging.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA device ordinal for the TRT engine.",
    )
    parser.add_argument(
        "--phase-channel-index", type=int, default=1,
        help="Which model output channel is phase (other = amplitude).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the spatially-diverse sample.",
    )
    args = parser.parse_args(argv)

    from ptychoml import (
        PtychoViTInference,
        autodetect_orientation,
        compute_intensity_normalization,
    )

    with h5py.File(args.data, "r") as f:
        if args.intensity_key not in f:
            print(
                f"Dataset {args.intensity_key!r} not found in {args.data}. "
                f"Available: {list(f.keys())}",
                file=sys.stderr,
            )
            return 1
        if args.positions_key not in f:
            print(
                f"Dataset {args.positions_key!r} not found in {args.data}. "
                f"Available: {list(f.keys())}",
                file=sys.stderr,
            )
            return 1
        x_um_all, y_um_all = _read_positions(f, args.positions_key)
        n_total = f[args.intensity_key].shape[0]

        rng = np.random.default_rng(args.seed)
        sel = _spatially_diverse_sample(
            np.asarray(x_um_all), np.asarray(y_um_all), args.n_eval, rng,
        )
        print(f"Loading {len(sel)} spatially-diverse frames from {n_total}...")
        intensity_subset = np.asarray(f[args.intensity_key][np.sort(sel)])
    positions_um = np.stack([x_um_all[sel], y_um_all[sel]], axis=1)

    if args.input_kind == "amplitude":
        intensity_subset = (intensity_subset.astype(np.float64)) ** 2

    if args.normalization is None:
        norm = compute_intensity_normalization(
            intensity_subset,
            hot_pixel_count_threshold=args.hot_pixel_count_threshold,
        )
        print(
            f"Computed normalization from subset (hot-pixel cutoff="
            f"{args.hot_pixel_count_threshold}): {norm:g}"
        )
    else:
        norm = args.normalization

    # Override probe (rare path — typically the baked probe wins).
    override_probe = None
    if args.probe is not None:
        override_probe = np.load(args.probe)
        if override_probe.ndim == 3:
            override_probe = override_probe[0]
        elif override_probe.ndim != 2:
            print(
                f"Probe .npy must be 2D or 3D; got ndim={override_probe.ndim}",
                file=sys.stderr,
            )
            return 1
        override_probe = override_probe.astype(np.complex64)
        print(f"Loaded override probe from --probe: shape={override_probe.shape}")

    fftshift_choice = {'auto': None, 'on': True, 'off': False}[args.fftshift]
    preprocess_kwargs = dict(
        normalization=float(norm),
        scale=float(args.scale),
        hot_pixel_count_threshold=args.hot_pixel_count_threshold,
        fftshift=fftshift_choice,
    )

    with PtychoViTInference(
        engine_path=args.engine, gpu=args.gpu,
        fftshift=False,  # we control the shift via preprocess_kwargs above
    ) as session:
        session._init_engine()
        # Probe priority: explicit --probe override > probe baked into the
        # engine by ``convert_pt_to_onnx.py --probe`` > error.
        if override_probe is not None:
            probe = override_probe
            print("Using override probe (from --probe); ignoring any baked probe.")
        elif session.baked_probe is not None:
            probe = session.baked_probe
            print(
                f"Using baked probe from engine: shape={probe.shape}"
            )
        else:
            print(
                "ERROR: engine has no baked probe and no --probe override was "
                "supplied. Re-export the ONNX with "
                "``convert_pt_to_onnx.py --probe path/to/probe.npy``, then "
                "rebuild the .engine; or pass --probe explicitly.",
                file=sys.stderr,
            )
            return 1
        report = autodetect_orientation(
            intensity_subset,
            positions_um,
            session=session,
            probe=probe,
            preprocess_kwargs=preprocess_kwargs,
            phase_channel_index=args.phase_channel_index,
        )

    best = report.best
    config_snippet = {
        "dp_orient": best.candidate.dp_orient,
        # Provenance — keep alongside the value so an operator inspecting
        # the JSON later can see what produced it.
        "_detect_orientation": {
            "score": float(best.score),
            "n_eval_frames": int(intensity_subset.shape[0]),
            "n_candidates_swept": len(report.ranked),
            "engine": str(Path(args.engine).resolve()),
            "data": str(Path(args.data).resolve()),
        },
    }

    print()
    print("Winning candidate:")
    print(f"  score:     {best.score:.6g}")
    print(f"  dp_orient: {best.candidate.dp_orient}")
    print()
    print("Full ranking (lower = better match to forward physics):")
    for i, r in enumerate(report.ranked):
        marker = "  ← winner" if i == 0 else ""
        print(f"  {r.candidate.dp_orient:<14s}  {r.score:.6g}{marker}")

    Path(args.output).write_text(json.dumps(config_snippet, indent=2) + "\n")
    print(f"\nWrote config snippet to {args.output}")
    print(
        "Merge ``dp_orient`` into your scan JSON and restart holoptycho. "
        "The other orientation knobs (``patch_flip``, ``x_direction``, "
        "``y_direction``, ``position_swap_xy``) are manual operator "
        "preferences and aren't auto-detected — set them in the scan "
        "JSON to match your dashboard / scan-grid conventions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
