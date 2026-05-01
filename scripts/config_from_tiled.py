"""Build a holoptycho config JSON from a Tiled run document.

Reads the Bluesky start document for a given HXN run UID from Tiled and maps
the beamline metadata to holoptycho config parameters. Reconstruction
parameters (nx, ny, alg_flag, etc.) must be supplied via CLI flags or
edited in the output JSON before passing to hp start.

Usage
-----
    # Authenticate once
    tiled login https://tiled.nsls2.bnl.gov

    # Print config JSON for a specific run UID
    python scripts/config_from_tiled.py --uid 67e77251-cbe4-444c-8a8c-36491b0b9100

    # Pipe directly into hp start
    hp start "$(python scripts/config_from_tiled.py --uid 67e77251-cbe4-444c-8a8c-36491b0b9100)"

    # Override reconstruction parameters
    python scripts/config_from_tiled.py --uid 67e77251-cbe4-444c-8a8c-36491b0b9100 \\
        --nx 256 --ny 256 --n-iterations 1000 --alg-flag DM
"""

import argparse
import json
import math
import sys
from urllib.parse import urlsplit, urlunsplit

from tiled.client import from_uri
from tiled.client.utils import ClientError

TILED_URL = "https://tiled.nsls2.bnl.gov"

# Per-detector pixel size (µm). Dectris Eigers are 75 µm; Merlin/Maxipix are
# 55 µm. The legacy ptycho_gui defaulted to 55 µm regardless, which was
# silently wrong for eiger2/eiger1 scans — fix that by looking up by detector.
DETECTOR_PIXEL_UM = {
    "eiger1": 75.0,
    "eiger2": 75.0,
    "merlin": 55.0,
    "merlin1": 55.0,
    "maxipix": 55.0,
}
DEFAULT_CCD_PIXEL_UM = 75.0  # HXN's primary detectors are eigers

LEGACY_PTYCHO_DEFAULTS = {
    "mode_flag": "False",
    "multislice_flag": "False",
    "init_obj_dpc_flag": "False",
    "prb_center_flag": "False",
    "mask_prb_flag": "False",
    "mask_obj_flag": "False",
    "norm_prb_amp_flag": "False",
    "mesh_flag": "True",
    "cal_scan_pattern_flag": "False",
    "bragg_flag": "False",
    "pc_flag": "False",
    "save_tmp_pic_flag": "False",
    "position_correction_flag": "False",
    "angle_correction_flag": "False",
    "sf_flag": "False",
    "ms_pie_flag": "False",
    "weak_obj_flag": "False",
    "preview_flag": "True",
    "save_config_history": "True",
    "cal_error_flag": "True",
    "refine_data_flag": "False",
    "profiler_flag": "False",
    "postprocessing_flag": "True",
    "use_NCCL": "False",
    "use_CUDA_MPI": "False",
    "frame_num": "0",
    "slice_num": "2",
    "dm_version": "2",
    "processes": "0",
    "pc_kernel_n": "32",
    "position_correction_start": "50",
    "position_correction_step": "10",
    "start_update_probe": "0",
    "start_update_object": "0",
    "refine_data_start_it": "10",
    "refine_data_interval": "5",
    "z_m": "1.0",
    # Object amplitude / phase clipping bounds. The defaults match what
    # ptycho_gui writes for HXN scans; without tight bounds the DM updates
    # can run away to NaN within a few iterations.
    "amp_max": "1.0",
    "amp_min": "0.5",
    "pha_max": "0.01",
    "pha_min": "-1.0",
    "slice_spacing_m": "5e-06",
    "start_ave": "0.8",
    "sigma2": "5e-05",
    "bragg_theta": "0.0",
    "bragg_gamma": "0.0",
    "bragg_delta": "0.0",
    "pc_sigma": "2.0",
    "refine_data_step": "0.05",
    "prb_filename": "",
    "prb_dir": "",
    "obj_filename": "",
    "obj_dir": "",
    "obj_path": "",
    "mpi_file_path": "",
    "pc_alg": "lucy",
    "asso_scan_numbers": "[]",
}


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def _descend_tiled_path(node, path_parts: list[str], tiled_url: str):
    for path_part in path_parts:
        try:
            node = node[path_part]
        except KeyError:
            print(
                f"ERROR: tiled path {'/'.join(path_parts)!r} not found in catalog at {tiled_url}",
                file=sys.stderr,
            )
            sys.exit(1)
    return node


def open_tiled_node(tiled_url: str, *, timeout_s: float = 300.0):
    """Open a Tiled server root or a catalog path URL.

    ``tiled.client.from_uri`` expects a server URL, not necessarily a catalog
    path like ``https://tiled.nsls2.bnl.gov/hxn/raw``. Try the URL directly
    first, then peel off trailing path segments until a server root is found and
    descend into the catalog path manually.

    ``timeout_s`` controls the httpx read timeout. The migration server is
    sometimes very slow for HXN raw eiger frames, so we default well above
    the httpx 30s default to avoid spurious ReadTimeout failures.
    """
    import httpx

    timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=10.0, pool=10.0)

    try:
        return from_uri(tiled_url, timeout=timeout)
    except ClientError as exc:
        if not _is_not_found(exc):
            raise

    parsed = urlsplit(tiled_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        raise

    for split_index in range(len(path_parts) - 1, -1, -1):
        base_path = "/" + "/".join(path_parts[:split_index]) if split_index else ""
        base_url = urlunsplit((parsed.scheme, parsed.netloc, base_path, parsed.query, parsed.fragment))
        try:
            root = from_uri(base_url, timeout=timeout)
        except ClientError as exc:
            if _is_not_found(exc):
                continue
            raise
        return _descend_tiled_path(root, path_parts[split_index:], tiled_url)

    return from_uri(tiled_url)


def _has_streams(run) -> bool:
    """Check whether a run exposes both primary and baseline streams.

    HXN tiled has three known layouts:
    * legacy migration: ``run/primary`` and ``run/baseline``
    * newer migration:  ``run/streams/primary`` and ``run/streams/baseline``
    * newer raw:        ``run/primary/data`` and ``run/baseline/data``

    Stub entries (e.g. migration redirect placeholders) only have a subset of
    these and 404 on actual reads.
    """
    try:
        keys = set(list(run))
    except Exception:
        return False
    if {"primary", "baseline"}.issubset(keys):
        return True
    if "streams" in keys:
        try:
            sub = set(list(run["streams"]))
        except Exception:
            return False
        return {"primary", "baseline"}.issubset(sub)
    return False


def get_stream(run, stream_name: str):
    """Return the array-level node for a stream, handling all known layouts."""
    keys = list(run)
    node = run["streams"][stream_name] if "streams" in keys else run[stream_name]
    return node["data"] if "data" in list(node) else node


def lookup_run(client, run_uid: str, tiled_url: str):
    candidates = [client]

    # Also try descending from the tiled root, so a user passing
    # --tiled-url .../hxn/migration can still find runs that only live in raw.
    parsed = urlsplit(tiled_url)
    root_url = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    try:
        candidates.append(from_uri(root_url))
    except Exception:
        pass

    for candidate in list(candidates):
        for path_parts in (("hxn", "migration"), ("hxn", "raw")):
            node = candidate
            try:
                for path_part in path_parts:
                    node = node[path_part]
            except KeyError:
                continue
            candidates.append(node)

    # Prefer entries that actually expose primary + baseline data (any layout).
    # Stub entries are deferred to fallback so we still return something usable
    # if no fully-populated entry exists.
    fallback = None
    for candidate in candidates:
        try:
            run = candidate[run_uid]
        except KeyError:
            continue
        if _has_streams(run):
            return run
        if fallback is None:
            fallback = run

    if fallback is not None:
        return fallback

    print(
        f"ERROR: run UID {run_uid!r} not found in tiled catalog at {tiled_url}",
        file=sys.stderr,
    )
    sys.exit(1)


def _run_start(run) -> dict:
    metadata = getattr(run, "metadata", {})
    start = metadata.get("start")
    if start is not None:
        return start
    return getattr(run, "start", {})


def _energy_from_dcm_th(dcm_th_deg: float) -> float:
    """Convert DCM angle (degrees) to X-ray energy (keV).

    Uses the Si(111) d-spacing: d = 3.1355893 Å
    E = 12.39842 / (2 * d * sin(θ))
    """
    return 12.39842 / (2.0 * 3.1355893 * math.sin(math.radians(dcm_th_deg)))


def _lambda_from_energy(energy_kev: float) -> float:
    """Convert energy (keV) to wavelength (nm) using the legacy ptycho_gui
    convention ``λ = 1.2398 / E``.  This matches the value
    ``streaming_recon.gpu_setup`` falls back to when ``lambda_nm`` is missing
    from the config, so the two stay in lockstep.
    """
    return 1.2398 / energy_kev


def _ratio_from_scale(scale_factor: float | int | None) -> float:
    if scale_factor in (None, 0):
        scale_factor = 1.0
    return -float(scale_factor) / 10000.0


def load_config_from_tiled(
    run_uid: str,
    tiled_url: str = TILED_URL,
) -> dict:
    """Load run metadata from Tiled and return a partial holoptycho config dict.

    Parameters
    ----------
    run_uid:
        Bluesky run UID.
    tiled_url:
        Tiled server URL. Uses cached credentials from ``tiled login``.

    Returns
    -------
    dict
        Partial config with all beamline-derived parameters set as strings.
        Reconstruction parameters (nx, ny, alg_flag, etc.) are not included
        and must be added before passing to ``hp start``.
    """
    client = open_tiled_node(tiled_url)

    run = lookup_run(client, run_uid, tiled_url)

    start = _run_start(run)
    scan_num = start.get("scan_id", run_uid)
    plan_name = start.get("plan_name", "")
    plan_args = start.get("plan_args", {})
    scan_md = start.get("scan", {})
    # Encoder counts → microns. The scale factors are positive but the legacy
    # ptycho code stores them as a negative ratio.
    x_scale = float(start.get("x_scale_factor") or 0.0)
    y_scale = float(start.get("z_scale_factor") or 0.0)
    x_ratio = _ratio_from_scale(start.get("x_scale_factor"))
    y_ratio = _ratio_from_scale(start.get("z_scale_factor"))

    baseline = get_stream(run, "baseline")
    primary_keys = list(get_stream(run, "primary"))

    # --- Energy ---
    try:
        dcm_th = float(baseline["dcm_th"].read()[0])
        energy_kev = _energy_from_dcm_th(dcm_th)
    except Exception as exc:
        print(f"WARNING: could not read DCM angle from baseline: {exc}", file=sys.stderr)
        energy_kev = 0.0

    # --- Scan geometry ---
    # The legacy ptycho_gui config records:
    #   * x_range / y_range as the *raw* span between scan endpoints
    #   * dr_x / dr_y as the per-step distance scaled by the encoder
    #     calibration (scale_factor) so they reflect the physical motor step
    #     in microns
    # The 2D_FLY_PANDA branch was previously subtracting one dr from the range
    # and storing the *unscaled* dr — both wrong relative to ptycho_gui output.
    try:
        if scan_md.get("type") == "2D_FLY_PANDA" and len(scan_md.get("scan_input", [])) >= 6:
            scan_input = scan_md["scan_input"]
            x_range = scan_input[1] - scan_input[0]
            y_range = scan_input[4] - scan_input[3]
            x_num = int(scan_input[2])
            y_num = int(scan_input[5])
            dr_x = (x_range / x_num) * (x_scale or 1.0)
            dr_y = (y_range / y_num) * (y_scale or 1.0)
        elif plan_name == "FlyPlan2D":
            x_range = plan_args["scan_end1"] - plan_args["scan_start1"]
            y_range = plan_args["scan_end2"] - plan_args["scan_start2"]
            x_num = int(plan_args["num1"])
            y_num = int(plan_args["num2"])
            dr_x = x_range / x_num
            dr_y = y_range / y_num
            x_range -= dr_x
            y_range -= dr_y
        elif plan_name in ("rel_spiral_fermat", "fermat"):
            x_range = plan_args["x_range"]
            y_range = plan_args["y_range"]
            dr_x = plan_args["dr"]
            dr_y = plan_args["dr"]
            x_num = 0
            y_num = 0
        else:
            # Generic mesh scan
            args = plan_args["args"]
            x_range = args[2] - args[1]
            y_range = args[6] - args[5]
            x_num = int(args[3])
            y_num = int(args[7])
            dr_x = x_range / x_num
            dr_y = y_range / y_num
            x_range -= dr_x
            y_range -= dr_y
    except Exception as exc:
        print(f"WARNING: could not parse scan geometry from plan_args: {exc}", file=sys.stderr)
        x_range = y_range = dr_x = dr_y = 0.0
        x_num = y_num = 0

    # --- Stage angle ---
    try:
        scan_motors = start.get("motors", [])
        if len(scan_motors) > 1 and scan_motors[1] == "zpssy":
            angle = float(baseline["zpsth"].read()[0])
        elif len(scan_motors) > 1 and scan_motors[1] == "ssy":
            angle = 0.0
        else:
            angle = float(baseline["dsth"].read()[0])
    except Exception as exc:
        print(f"WARNING: could not read stage angle from baseline: {exc}", file=sys.stderr)
        angle = 0.0

    lambda_nm = _lambda_from_energy(energy_kev) if energy_kev > 0 else 0.0

    # --- Detector kind + pixel size ---
    # The image stream key in primary tells us which detector recorded the
    # diffraction frames. Fall back to ``scan["detectors"]`` for older runs
    # that store the array under a non-standard key.
    detector_keys = ("eiger2_image", "eiger1_image", "eiger2", "eiger1")
    detectorkind = next((k for k in detector_keys if k in primary_keys), "")
    if not detectorkind:
        for d in scan_md.get("detectors") or []:
            if d.startswith(("eiger", "merlin", "maxipix")):
                detectorkind = f"{d}_image"
                break
    detector_base = detectorkind.replace("_image", "") if detectorkind else ""
    ccd_pixel_um = DETECTOR_PIXEL_UM.get(detector_base, DEFAULT_CCD_PIXEL_UM)

    # --- Sample-to-detector distance ---
    # 2D_FLY_PANDA scans record this in ``scan["detector_distance"]`` (in m).
    # Older scans don't expose it; the LEGACY_PTYCHO_DEFAULTS z_m=1.0 fallback
    # remains in build_full_config() for those.
    z_m = scan_md.get("detector_distance")

    config = {
        "scan_num": str(scan_num),
        "scan_type": plan_name,
        "xray_energy_kev": str(round(energy_kev, 6)),
        "lambda_nm": str(round(lambda_nm, 12)),
        "ccd_pixel_um": str(ccd_pixel_um),
        "dr_x": str(round(dr_x, 6)),
        "dr_y": str(round(dr_y, 6)),
        "x_range": str(round(x_range, 6)),
        "y_range": str(round(y_range, 6)),
        "x_num": str(x_num),
        "y_num": str(y_num),
        "angle": str(round(angle, 4)),
        # streaming_recon defaults both to -1.0; matches legacy ptycho_gui
        # output for HXN scans.
        "x_direction": "-1.0",
        "y_direction": "-1.0",
        "x_ratio": str(round(x_ratio, 8)),
        "y_ratio": str(round(y_ratio, 8)),
    }
    if detectorkind:
        config["detectorkind"] = detectorkind
    if z_m is not None:
        config["z_m"] = str(z_m)

    return config


def add_reconstruction_arguments(parser: argparse.ArgumentParser):
    """Add reconstruction override flags used by hp start configs."""
    recon = parser.add_argument_group(
        "reconstruction parameters",
        "These are not in the scan metadata and must be set explicitly.",
    )
    recon.add_argument("--working-directory", default="/ptycho_gui_holoscan")
    recon.add_argument("--nx", type=int, default=128)
    recon.add_argument("--ny", type=int, default=128)
    # batch-width/height default to nx/ny (substituted in build_full_config
    # when None) — they're almost always equal to the recon frame size.
    recon.add_argument("--batch-width", type=int, default=None,
                       help="Detector crop width in pixels (default: --nx).")
    recon.add_argument("--batch-height", type=int, default=None,
                       help="Detector crop height in pixels (default: --ny).")
    # batch-x0/y0 default to None and are auto-computed from the diffraction
    # center of the first frames if the replay script has the data on hand.
    recon.add_argument("--batch-x0", type=int, default=None,
                       help="Detector crop column offset (default: auto from "
                            "diffraction center of mass).")
    recon.add_argument("--batch-y0", type=int, default=None,
                       help="Detector crop row offset (default: auto from "
                            "diffraction center of mass).")
    recon.add_argument("--det-roix0", type=int, default=0)
    recon.add_argument("--det-roiy0", type=int, default=0)
    recon.add_argument("--gpu-batch-size", type=int, default=256)
    recon.add_argument("--distance", type=float, default=0.5,
                       help="Sample-to-detector distance in m")
    recon.add_argument("--alg-flag", default="ML_grad")
    recon.add_argument("--n-iterations", type=int, default=500)
    recon.add_argument("--gpus", default="[0]")
    recon.add_argument("--sign", default="t1")
    recon.add_argument("--display-interval", type=int, default=10)
    recon.add_argument(
        "--recon-mode",
        choices=["iterative", "vit", "both"],
        default="both",
        help=(
            "Which reconstruction branches to wire in the pipeline. "
            "'iterative' = DM/ML solver only, 'vit' = ViT inference only, "
            "'both' = parallel (default). Useful for isolating GPU-contention "
            "issues on single-GPU nodes."
        ),
    )


def build_full_config(run_uid: str, tiled_url: str, args: argparse.Namespace) -> dict:
    """Build a full hp start config from scan metadata plus CLI overrides."""
    config = load_config_from_tiled(run_uid, tiled_url=tiled_url)
    scan_num = config["scan_num"]

    # Defaults only fill in keys that ``load_config_from_tiled`` didn't set —
    # otherwise ``LEGACY_PTYCHO_DEFAULTS["z_m"] = "1.0"`` would overwrite the
    # ``detector_distance`` we just extracted from the run metadata, silently
    # halving the pixel size and distorting the reconstruction.
    for k, v in LEGACY_PTYCHO_DEFAULTS.items():
        config.setdefault(k, v)
    # Default batch box size to nx/ny; default ROI offset to 0 if the caller
    # didn't run auto-centering (the replay script fills these in by computing
    # the diffraction center of mass before calling build_full_config).
    batch_width = args.batch_width if args.batch_width is not None else args.nx
    batch_height = args.batch_height if args.batch_height is not None else args.ny
    batch_x0 = args.batch_x0 if args.batch_x0 is not None else 0
    batch_y0 = args.batch_y0 if args.batch_y0 is not None else 0

    config.update({
        # Provenance for the per-run Tiled container metadata.
        "raw_uid": run_uid,
        "scan_id": str(scan_num),
        "working_directory": args.working_directory,
        "shm_name": f"ptycho_{scan_num}",
        "nx": str(args.nx),
        "ny": str(args.ny),
        "batch_width": str(batch_width),
        "batch_height": str(batch_height),
        "batch_x0": str(batch_x0),
        "batch_y0": str(batch_y0),
        "det_roix0": str(args.det_roix0),
        "det_roiy0": str(args.det_roiy0),
        "gpu_batch_size": str(args.gpu_batch_size),
        "distance": str(args.distance),
        "nz": str(int(config["x_num"]) * int(config["y_num"])),
        "x_arr_size": config["x_num"],
        "y_arr_size": config["y_num"],
        "alg_flag": args.alg_flag,
        "alg2_flag": args.alg_flag,
        "alg_percentage": "0.5",
        "n_iterations": str(args.n_iterations),
        "ml_mode": "Poisson",
        # Match ptycho_gui's tuned values for HXN scans. ml_weight=5.0 was
        # too aggressive and caused NaN divergence on scans with tight
        # clipping bounds.
        "ml_weight": "0.1",
        "beta": "0.9",
        "init_obj_flag": "True",
        "init_prb_flag": "True",
        "prb_path": "",
        "prb_mode_num": "1",
        "obj_mode_num": "1",
        "gpu_flag": "True",
        "gpus": args.gpus,
        "precision": "single",
        "nth": "5",
        "sign": args.sign,
        "display_interval": str(args.display_interval),
        "recon_mode": args.recon_mode,
    })

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Build a holoptycho config JSON from a Tiled HXN scan.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--uid",
        required=True,
        help="Bluesky run UID",
    )
    parser.add_argument(
        "--tiled-url",
        default=TILED_URL,
        help=f"Tiled server URL (default: {TILED_URL})",
    )

    add_reconstruction_arguments(parser)

    args = parser.parse_args()
    config = build_full_config(args.uid, tiled_url=args.tiled_url, args=args)

    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
