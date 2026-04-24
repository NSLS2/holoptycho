"""Build a holoptycho config JSON from a Tiled scan start document.

Reads the Bluesky start document for a given HXN scan from Tiled and maps
the beamline metadata to holoptycho config parameters. Reconstruction
parameters (nx, ny, alg_flag, etc.) must be supplied via CLI flags or
edited in the output JSON before passing to hp start.

Usage
-----
    # Authenticate once
    tiled login https://tiled.nsls2.bnl.gov

    # Print config JSON for scan 320045
    python scripts/config_from_tiled.py --scan-num 320045

    # Pipe directly into hp start
    hp start "$(python scripts/config_from_tiled.py --scan-num 320045)"

    # Override reconstruction parameters
    python scripts/config_from_tiled.py --scan-num 320045 \\
        --nx 256 --ny 256 --n-iterations 1000 --alg-flag DM
"""

import argparse
import json
import math
import sys

from tiled.client import from_uri

TILED_URL = "https://tiled.nsls2.bnl.gov"
CCD_PIXEL_UM = 55.0  # HXN Eiger pixel size (µm) — hard-coded, matches ptycho_gui


def _energy_from_dcm_th(dcm_th_deg: float) -> float:
    """Convert DCM angle (degrees) to X-ray energy (keV).

    Uses the Si(111) d-spacing: d = 3.1355893 Å
    E = 12.39842 / (2 * d * sin(θ))
    """
    return 12.39842 / (2.0 * 3.1355893 * math.sin(math.radians(dcm_th_deg)))


def _lambda_from_energy(energy_kev: float) -> float:
    """Convert energy (keV) to wavelength (nm)."""
    return (6.62607e-34 * 2.99792e8) / (energy_kev * 1e3 * 1.60218e-19) * 1e9


def load_config_from_tiled(
    scan_num: int,
    tiled_url: str = TILED_URL,
) -> dict:
    """Load scan metadata from Tiled and return a partial holoptycho config dict.

    Parameters
    ----------
    scan_num:
        HXN scan number.
    tiled_url:
        Tiled server URL. Uses cached credentials from ``tiled login``.

    Returns
    -------
    dict
        Partial config with all beamline-derived parameters set as strings.
        Reconstruction parameters (nx, ny, alg_flag, etc.) are not included
        and must be added before passing to ``hp start``.
    """
    client = from_uri(tiled_url)

    try:
        scan = client["hxn"]["raw"][str(scan_num)]
    except KeyError:
        print(
            f"ERROR: scan {scan_num} not found at {tiled_url}/hxn/raw",
            file=sys.stderr,
        )
        sys.exit(1)

    start = scan.start
    plan_name = start.get("plan_name", "")
    plan_args = start.get("plan_args", {})

    # --- Energy ---
    try:
        baseline = scan["baseline"].read()
        dcm_th = float(baseline["dcm_th"].iloc[0])
        energy_kev = _energy_from_dcm_th(dcm_th)
    except Exception as exc:
        print(f"WARNING: could not read DCM angle from baseline: {exc}", file=sys.stderr)
        energy_kev = 0.0

    # --- Scan geometry ---
    try:
        if plan_name == "FlyPlan2D":
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
            angle = float(baseline["zpsth"].iloc[0])
        elif len(scan_motors) > 1 and scan_motors[1] == "ssy":
            angle = 0.0
        else:
            angle = float(baseline["dsth"].iloc[0])
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
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Build a holoptycho config JSON from a Tiled HXN scan.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan-num",
        required=True,
        type=int,
        help="HXN scan number",
    )
    parser.add_argument(
        "--tiled-url",
        default=TILED_URL,
        help=f"Tiled server URL (default: {TILED_URL})",
    )

    # Reconstruction parameters — optional overrides
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

    args = parser.parse_args()

    config = load_config_from_tiled(args.scan_num, tiled_url=args.tiled_url)

    # Merge in reconstruction parameters
    config.update({
        "working_directory": args.working_directory,
        "shm_name": f"ptycho_{args.scan_num}",
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

    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
