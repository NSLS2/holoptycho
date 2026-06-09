"""Dump a Tiled scan to HDF5 in the hxn_to_vit output format.

Applies exactly the same preprocessing pipeline as holoptycho (flip, crop,
bad-pixel removal, hot-pixel zeroing, D4 orientation, fftshift) and the same
encoder-count → µm position conversion as PointProcessorOp.

The output dp.hdf5 has the same structure as hxn_to_vit.py's {scan_id}_dp.hdf5:

    dp        (N, H, W) float32  — intensity counts, DC at center, dp_orient applied
    points    (2, N)   float64  — scan positions in µm: [x_um, y_um]

Scalar attributes on the root group record all processing parameters so the
output is self-describing and can be fed directly to ptychoml-predict:

    normalization, dp_orient, lambda_nm, z_m, ccd_pixel_um, angle,
    hot_pixel_count_threshold, batch_x0, batch_y0, nx, ny, x_ratio,
    y_ratio, x_direction, y_direction

Usage:

    pixi run -e replay python scripts/dump_scan_h5.py \\
        --scan-id 405667 \\
        --nx 256 --ny 256 \\
        --batch-x0 75 --batch-y0 138 \\
        --dp-orient rot90_cw \\
        -o /tmp/405667_dp.hdf5

Then feed to ptychoml-predict:

    pixi run -e client ptychoml-predict \\
        --data /tmp/405667_dp.hdf5 --dataset dp --input-kind intensity \\
        --engine /path/to/model.engine --output /tmp/405667_pred.npy \\
        --dp-orient identity --normalization <value from attrs>
"""

import argparse
import sys
import math

import numpy as np
import h5py

from tiled.client import from_uri

# D4 orientation transforms come from ptychoml so this offline dump applies
# the exact same set the live ImagePreprocessorOp uses (no inline copy).
from ptychoml import apply_d4, D4_NAMES


# ---------------------------------------------------------------------------
# Tiled helpers (mirrors config_from_tiled.py logic)
# ---------------------------------------------------------------------------

_TILED_URL = "https://tiled.nsls2.bnl.gov"
_CATALOG_CANDIDATES = [
    ("hxn", "processed", "holoptycho"),
    ("hxn", "migration"),
    ("hxn", "raw"),
]


def _open_run(run_uid, tiled_url=_TILED_URL):
    root = from_uri(tiled_url, timeout=30)
    for path in _CATALOG_CANDIDATES:
        node = root
        try:
            for part in path:
                node = node[part]
            run = node[run_uid]
            if hasattr(run, "primary"):
                return run
        except (KeyError, Exception):
            continue
    raise RuntimeError(f"Run {run_uid!r} not found in {tiled_url}")


def _lookup_by_scan_id(scan_id, tiled_url=_TILED_URL):
    """Return the UID of the most recent run matching scan_id at HXN."""
    from tiled.queries import Eq
    root = from_uri(tiled_url, timeout=30)
    for path in (("hxn", "migration"), ("hxn", "raw")):
        node = root
        try:
            for part in path:
                node = node[part]
            results = list(node.search(Eq("scan_id", int(scan_id))))
            if results:
                # newest first
                uid = results[-1]
                print(f"  Found scan_id={scan_id} → UID={uid}", file=sys.stderr)
                return uid
        except Exception:
            continue
    raise RuntimeError(f"scan_id={scan_id} not found in Tiled")


def _run_start(run):
    md = getattr(run, "metadata", {})
    return md.get("start") or getattr(run, "start", {})


def _get_stream(run, name="primary"):
    try:
        return run[name]
    except Exception:
        return getattr(run, name, None)


# ---------------------------------------------------------------------------
# Geometry helpers (mirrors config_from_tiled.py)
# ---------------------------------------------------------------------------

def _ratio_from_scale(v):
    if v in (None, 0):
        v = 1.0
    return -float(v) / 10000.0


def _energy_from_dcm_th(deg):
    return 12.39842 / (2.0 * 3.1355893 * math.sin(math.radians(deg)))


def _lambda_from_energy(kev):
    return 1.2398 / kev


# ---------------------------------------------------------------------------
# Main dump routine
# ---------------------------------------------------------------------------

def dump(
    run_uid,
    output_path,
    nx=256,
    ny=256,
    batch_x0=0,
    batch_y0=0,
    dp_orient="rot90_cw",
    hot_pixel_count_threshold=50000.0,
    flip_image=True,
    max_frames=None,
    tiled_url=_TILED_URL,
    chunk_size=256,
):
    print(f"Opening run {run_uid} ...", file=sys.stderr)
    root = from_uri(tiled_url, timeout=30)

    run = None
    for path in (("hxn", "migration"), ("hxn", "raw")):
        node = root
        try:
            for part in path:
                node = node[part]
            run = node[run_uid]
            break
        except (KeyError, Exception):
            continue
    if run is None:
        raise RuntimeError(f"Run {run_uid!r} not found")

    start = _run_start(run)
    scan_id = start.get("scan_id", run_uid)
    primary = _get_stream(run, "primary")
    primary_keys = list(primary)

    # --- Energy / wavelength ---
    baseline = _get_stream(run, "baseline")
    energy_kev = 0.0
    lambda_nm = 0.0
    try:
        dcm_th = float(np.asarray(baseline["dcm_th"].read()).ravel()[0])
        energy_kev = _energy_from_dcm_th(dcm_th)
        lambda_nm = _lambda_from_energy(energy_kev)
    except Exception as exc:
        print(f"  WARNING: could not read energy: {exc}", file=sys.stderr)

    # --- Angle ---
    angle = 0.0
    try:
        scan_motors = start.get("motors", [])
        if len(scan_motors) > 1 and scan_motors[1] == "zpssy":
            angle = float(np.asarray(baseline["zpsth"].read()).ravel()[0])
        elif len(scan_motors) > 1 and scan_motors[1] == "ssy":
            angle = 0.0
        else:
            angle = float(np.asarray(baseline["dsth"].read()).ravel()[0])
    except Exception:
        pass

    # --- Pixel size + z_m ---
    _DETECTOR_PIXEL_UM = {"eiger2": 75.0, "eiger1": 75.0}
    _DEFAULT_PIXEL_UM = 75.0
    detector_keys = ("eiger2_image", "eiger1_image", "eiger2", "eiger1")
    detectorkind = next((k for k in detector_keys if k in primary_keys), "")
    detector_base = detectorkind.replace("_image", "") if detectorkind else ""
    ccd_pixel_um = _DETECTOR_PIXEL_UM.get(detector_base, _DEFAULT_PIXEL_UM)

    scan_md = start.get("scan", {})
    z_m = scan_md.get("detector_distance") or 1.0

    # --- Position conversion factors ---
    x_ratio = _ratio_from_scale(start.get("x_scale_factor"))
    y_ratio = _ratio_from_scale(start.get("z_scale_factor"))
    x_direction = -1.0
    y_direction = -1.0

    # --- Detector frames ---
    if not detectorkind:
        raise RuntimeError("Could not identify detector key in primary stream")
    det_node = primary[detectorkind]
    det_shape = det_node.structure().shape
    # det_shape: (1, n_frames, H, W) or (n_frames, H, W)
    has_leading_one = (len(det_shape) == 4)
    n_total = int(det_shape[1] if has_leading_one else det_shape[0])
    frame_h, frame_w = int(det_shape[-2]), int(det_shape[-1])

    n_frames = n_total if max_frames is None else min(max_frames, n_total)
    print(
        f"  detector={detectorkind}, total={n_total} frames ({frame_h}x{frame_w}), "
        f"will dump {n_frames}",
        file=sys.stderr,
    )

    # ROI for crop (matches ptycho_holo.py roi construction)
    y0, y1 = batch_y0, batch_y0 + ny
    x0, x1 = batch_x0, batch_x0 + nx
    if y1 > frame_h or x1 > frame_w:
        raise ValueError(
            f"ROI ({y0}:{y1}, {x0}:{x1}) exceeds frame size ({frame_h}x{frame_w})"
        )

    # --- Encoder positions ---
    x_candidates = ("inenc2_val", "zpssx")
    y_candidates = ("inenc3_val", "zpssy")
    enc_x_key = next((k for k in x_candidates if k in primary_keys), None)
    enc_y_key = next((k for k in y_candidates if k in primary_keys), None)
    if enc_x_key is None or enc_y_key is None:
        raise RuntimeError(
            f"Could not find encoder keys; available: {primary_keys[:20]}"
        )
    print(
        f"  encoder keys: x={enc_x_key}, y={enc_y_key}",
        file=sys.stderr,
    )
    raw_x = np.asarray(primary[enc_x_key].read()).ravel().astype(np.float64)
    raw_y = np.asarray(primary[enc_y_key].read()).ravel().astype(np.float64)

    # Detect upsample ratio (PandA sends N samples per frame)
    upsample = max(1, len(raw_x) // n_total)
    print(f"  panda_upsample={upsample}", file=sys.stderr)

    # Average N samples → 1 position per frame (mirrors PointProcessorOp)
    n_pos = len(raw_x) // upsample
    pos_x_counts = raw_x[: n_pos * upsample].reshape(n_pos, upsample).mean(axis=1)
    pos_y_counts = raw_y[: n_pos * upsample].reshape(n_pos, upsample).mean(axis=1)

    # Unit conversion: encoder counts → µm (matches PointProcessorOp.process_point_info)
    # Detect if values look like motor readback (µm) vs raw counts, and convert if needed
    if np.abs(pos_x_counts).max() < 1e3 or np.abs(pos_y_counts).max() < 1e3:
        print(
            "  WARNING: encoder values look like motor readback in µm "
            f"(max |x|={np.abs(pos_x_counts).max():.3f}); converting to count-space",
            file=sys.stderr,
        )
        x_scale = x_ratio * x_direction
        y_scale = y_ratio * y_direction
        if x_scale != 0:
            pos_x_counts = pos_x_counts / x_scale
        if y_scale != 0:
            pos_y_counts = pos_y_counts / y_scale

    x_um = pos_x_counts * x_ratio * x_direction
    y_um = pos_y_counts * y_ratio * y_direction

    # Angle correction (matches PointProcessorOp, angle_correction_flag=True)
    if abs(angle) > 0:
        if abs(angle) <= 45.0:
            x_um = x_um * abs(math.cos(math.radians(angle)))
        else:
            x_um = x_um * abs(math.sin(math.radians(angle)))
        if angle <= -45.0:
            x_um = -x_um

    # Trim to n_frames and pair as (2, N)
    n_pos_use = min(n_frames, len(x_um))
    # hxn_to_vit convention (pos_map=(-y,-x)): points[0] = slow axis (INENC3/y_um),
    # points[1] = fast axis (INENC2/x_um). Span: points[0]~y_range, points[1]~x_range.
    points_um = np.stack([y_um[:n_pos_use], x_um[:n_pos_use]], axis=0)  # (2, N) [slow, fast]

    # Pad to n_frames with NaN if positions are shorter
    if n_pos_use < n_frames:
        pad = np.full((2, n_frames - n_pos_use), np.nan)
        points_um = np.concatenate([points_um, pad], axis=1)

    # --- Compute normalization over all frames (hot pixels excluded) ---
    print(
        f"  Computing normalization (scanning {n_frames} frames in chunks of {chunk_size}) ...",
        file=sys.stderr,
    )
    normalization = 0.0
    for i in range(0, n_frames, chunk_size):
        j = min(i + chunk_size, n_frames)
        if has_leading_one:
            chunk = np.asarray(det_node.read(
                slice=[slice(None), slice(i, j), slice(None), slice(None)]
            )).squeeze(0).astype(np.float32)
        else:
            chunk = np.asarray(det_node.read(
                slice=[slice(i, j), slice(None), slice(None)]
            )).astype(np.float32)
        # flip (Eiger2 vertical flip)
        if flip_image:
            chunk = np.flip(chunk, axis=1)
        # crop
        chunk = chunk[:, y0:y1, x0:x1]
        # remove detector saturation pixels
        chunk[chunk == np.iinfo(np.uint16).max] = 0
        # hot pixel exclusion for normalization
        mask = chunk <= hot_pixel_count_threshold
        if mask.any():
            v = float(chunk[mask].max())
            if v > normalization:
                normalization = v
        print(f"    {j}/{n_frames} normalization so far: {normalization:.1f}", file=sys.stderr, end="\r")
    print(f"\n  normalization = {normalization:.1f}", file=sys.stderr)

    # --- Write dp.hdf5 ---
    print(f"  Writing {output_path} ...", file=sys.stderr)
    with h5py.File(output_path, "w") as f:
        dp_dset = f.create_dataset(
            "dp",
            shape=(n_frames, ny, nx),
            dtype=np.float32,
            chunks=(min(chunk_size, n_frames), ny, nx),
        )

        for i in range(0, n_frames, chunk_size):
            j = min(i + chunk_size, n_frames)
            if has_leading_one:
                chunk = np.asarray(det_node.read(
                    slice=[slice(None), slice(i, j), slice(None), slice(None)]
                )).squeeze(0).astype(np.float32)
            else:
                chunk = np.asarray(det_node.read(
                    slice=[slice(i, j), slice(None), slice(None)]
                )).astype(np.float32)

            # 1. Vertical flip (Eiger2 readout — matches ImageBatchOp.compute)
            if flip_image:
                chunk = np.flip(chunk, axis=1)
            # 2. Crop to ROI
            chunk = chunk[:, y0:y1, x0:x1]
            # 3. Zero detector saturation / overflow pixels
            chunk[chunk == np.iinfo(np.uint16).max] = 0
            # 4. Hot-pixel count threshold (intensity domain, before sqrt)
            chunk[chunk > hot_pixel_count_threshold] = 0.0
            # 5. D4 orientation (same as ImagePreprocessorOp → preprocess_diffraction dp_orient)
            chunk = np.ascontiguousarray(apply_d4(chunk, dp_orient))
            # 6. No fftshift: raw Eiger data already has the direct beam at the
            #    physical detector center (DC-at-center). hxn_to_vit write_outputs
            #    also stores dp with DC-at-center for raw_data sources. Applying
            #    fftshift would move DC to corner, which is wrong.

            dp_dset[i:j] = chunk
            print(f"    wrote frames {i}:{j}", file=sys.stderr, end="\r")

        print(f"\n  dp dataset written: shape={dp_dset.shape}", file=sys.stderr)

        # positions
        f.create_dataset("points", data=points_um)

        # scalar metadata (mirrors live_compare_viewer.py + ptychoml-predict expectations)
        f.create_dataset("lambda_nm", data=lambda_nm)
        f.create_dataset("z_m", data=z_m)
        f.create_dataset("ccd_pixel_um", data=ccd_pixel_um)
        f.create_dataset("angle", data=angle)

        # processing-parameter attributes (self-describing for audit)
        f.attrs["scan_id"] = str(scan_id)
        f.attrs["run_uid"] = str(run_uid)
        f.attrs["normalization"] = float(normalization)
        f.attrs["dp_orient"] = dp_orient
        f.attrs["hot_pixel_count_threshold"] = float(hot_pixel_count_threshold)
        f.attrs["flip_image"] = bool(flip_image)
        f.attrs["batch_x0"] = batch_x0
        f.attrs["batch_y0"] = batch_y0
        f.attrs["nx"] = nx
        f.attrs["ny"] = ny
        f.attrs["x_ratio"] = x_ratio
        f.attrs["y_ratio"] = y_ratio
        f.attrs["x_direction"] = x_direction
        f.attrs["y_direction"] = y_direction
        f.attrs["panda_upsample"] = upsample

    print(f"Done → {output_path}", file=sys.stderr)
    print(f"  dp: shape={n_frames}x{ny}x{nx}, normalization={normalization:.1f}, "
          f"dp_orient={dp_orient!r}", file=sys.stderr)
    print(f"  points: shape=(2, {n_frames}), "
          f"x_um=[{float(np.nanmin(points_um[0])):.3f} .. {float(np.nanmax(points_um[0])):.3f}], "
          f"y_um=[{float(np.nanmin(points_um[1])):.3f} .. {float(np.nanmax(points_um[1])):.3f}]",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dump a Tiled scan to HDF5 in hxn_to_vit output format for comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    id_grp = parser.add_mutually_exclusive_group(required=True)
    id_grp.add_argument("--scan-id", type=int, help="Bluesky scan_id (integer)")
    id_grp.add_argument("--uid", help="Bluesky run UID (UUID4 string)")

    parser.add_argument(
        "-o", "--output",
        help="Output HDF5 path (default: <scan_id>_dp.hdf5 in cwd)",
    )
    parser.add_argument("--nx", type=int, default=256, help="Crop width (pixels)")
    parser.add_argument("--ny", type=int, default=256, help="Crop height (pixels)")
    parser.add_argument("--batch-x0", type=int, default=0, help="Crop x offset (pixels)")
    parser.add_argument("--batch-y0", type=int, default=0, help="Crop y offset (pixels)")
    parser.add_argument(
        "--dp-orient",
        default="rot90_cw",
        choices=list(D4_NAMES),
        help="D4 transform applied to each frame before writing (matches holoptycho default)",
    )
    parser.add_argument(
        "--hot-pixel-count-threshold",
        type=float,
        default=50000.0,
        help="Photon-count threshold for hot-pixel zeroing",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit to first N frames (for quick tests)",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable the Eiger2 vertical flip (flip_image=True by default)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of frames per Tiled fetch / write chunk",
    )
    parser.add_argument(
        "--tiled-url",
        default=_TILED_URL,
        help="Tiled server URL",
    )
    args = parser.parse_args()

    # Resolve UID from scan_id if needed
    if args.uid:
        run_uid = args.uid
        scan_label = args.uid[:8]
    else:
        run_uid = _lookup_by_scan_id(args.scan_id, tiled_url=args.tiled_url)
        scan_label = str(args.scan_id)

    output = args.output or f"{scan_label}_dp.hdf5"

    dump(
        run_uid=run_uid,
        output_path=output,
        nx=args.nx,
        ny=args.ny,
        batch_x0=args.batch_x0,
        batch_y0=args.batch_y0,
        dp_orient=args.dp_orient,
        hot_pixel_count_threshold=args.hot_pixel_count_threshold,
        flip_image=not args.no_flip,
        max_frames=args.max_frames,
        tiled_url=args.tiled_url,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
