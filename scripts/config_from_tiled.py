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
CCD_PIXEL_UM = 55.0  # HXN Eiger pixel size (µm) — hard-coded, matches ptycho_gui

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
    "amp_max": "1.0",
    "amp_min": "0.01",
    "pha_max": "1.0",
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


def open_tiled_node(tiled_url: str):
    """Open a Tiled server root or a catalog path URL.

    ``tiled.client.from_uri`` expects a server URL, not necessarily a catalog
    path like ``https://tiled.nsls2.bnl.gov/hxn/raw``. Try the URL directly
    first, then peel off trailing path segments until a server root is found and
    descend into the catalog path manually.
    """
    try:
        return from_uri(tiled_url)
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
            root = from_uri(base_url)
        except ClientError as exc:
            if _is_not_found(exc):
                continue
            raise
        return _descend_tiled_path(root, path_parts[split_index:], tiled_url)

    return from_uri(tiled_url)


def lookup_run(client, run_uid: str, tiled_url: str):
    candidates = [client]

    for path_parts in (("hxn", "migration"), ("hxn", "raw")):
        node = client
        try:
            for path_part in path_parts:
                node = node[path_part]
        except KeyError:
            continue
        candidates.append(node)

    for candidate in candidates:
        try:
            return candidate[run_uid]
        except KeyError:
            continue

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
    """Convert energy (keV) to wavelength (nm)."""
    return (6.62607e-34 * 2.99792e8) / (energy_kev * 1e3 * 1.60218e-19) * 1e9


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
    x_ratio = _ratio_from_scale(start.get("x_scale_factor"))
    y_ratio = _ratio_from_scale(start.get("z_scale_factor"))

    # --- Energy ---
    try:
        baseline = run["baseline"]
        dcm_th = float(baseline["dcm_th"].read()[0])
        energy_kev = _energy_from_dcm_th(dcm_th)
    except Exception as exc:
        print(f"WARNING: could not read DCM angle from baseline: {exc}", file=sys.stderr)
        energy_kev = 0.0

    # --- Scan geometry ---
    try:
        if scan_md.get("type") == "2D_FLY_PANDA" and len(scan_md.get("scan_input", [])) >= 6:
            scan_input = scan_md["scan_input"]
            x_range = scan_input[1] - scan_input[0]
            y_range = scan_input[4] - scan_input[3]
            x_num = int(scan_input[2])
            y_num = int(scan_input[5])
            dr_x = x_range / x_num
            dr_y = y_range / y_num
            x_range -= dr_x
            y_range -= dr_y
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

    config = {
        "scan_num": str(scan_num),
        "scan_type": plan_name,
        "xray_energy_kev": str(round(energy_kev, 6)),
        "lambda_nm": str(round(lambda_nm, 12)),
        "ccd_pixel_um": str(CCD_PIXEL_UM),
        "dr_x": str(round(dr_x, 6)),
        "dr_y": str(round(dr_y, 6)),
        "x_range": str(round(x_range, 6)),
        "y_range": str(round(y_range, 6)),
        "x_num": str(x_num),
        "y_num": str(y_num),
        "angle": str(round(angle, 4)),
        "x_direction": "1.0",
        "y_direction": "-1.0",
        "x_ratio": str(round(x_ratio, 8)),
        "y_ratio": str(round(y_ratio, 8)),
    }

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
    recon.add_argument("--batch-width", type=int, default=128)
    recon.add_argument("--batch-height", type=int, default=128)
    recon.add_argument("--batch-x0", type=int, default=0)
    recon.add_argument("--batch-y0", type=int, default=0)
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


def build_full_config(run_uid: str, tiled_url: str, args: argparse.Namespace) -> dict:
    """Build a full hp start config from scan metadata plus CLI overrides."""
    config = load_config_from_tiled(run_uid, tiled_url=tiled_url)
    scan_num = config["scan_num"]

    config.update(LEGACY_PTYCHO_DEFAULTS)
    config.update({
        # Provenance for the per-run Tiled container metadata.
        "raw_uid": run_uid,
        "scan_id": str(scan_num),
        "working_directory": args.working_directory,
        "shm_name": f"ptycho_{scan_num}",
        "nx": str(args.nx),
        "ny": str(args.ny),
        "batch_width": str(args.batch_width),
        "batch_height": str(args.batch_height),
        "batch_x0": str(args.batch_x0),
        "batch_y0": str(args.batch_y0),
        "det_roix0": str(args.det_roix0),
        "det_roiy0": str(args.det_roiy0),
        "gpu_batch_size": str(args.gpu_batch_size),
        "distance": str(args.distance),
        "nz": str(int(config["x_num"]) * int(config["y_num"])),
        "x_arr_size": config["x_num"],
        "y_arr_size": config["y_num"],
        "alg_flag": args.alg_flag,
        "alg2_flag": args.alg_flag,
        "alg_percentage": "0.3",
        "n_iterations": str(args.n_iterations),
        "ml_mode": "Poisson",
        "ml_weight": "5.0",
        "beta": "0.9",
        "init_obj_flag": "True",
        "init_prb_flag": "False",
        "prb_path": "",
        "prb_mode_num": "1",
        "obj_mode_num": "1",
        "gpu_flag": "True",
        "gpus": args.gpus,
        "precision": "single",
        "nth": "5",
        "sign": args.sign,
        "display_interval": str(args.display_interval),
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
