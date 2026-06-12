"""Run the ACTUAL StreamingPtychoRecon port offline on dumped engine inputs.

Loads the pipeline's own HOLOPTYCHO_DUMP_ENGINE_INPUTS dumps (diff_d.npy +
point_info_d.npy — byte-identical to what the engine consumed in the run) and
the warm-start probe, instantiates StreamingPtychoRecon directly, activates
ALL points from iteration 0, and runs the DM loop.

Discriminator: if this converges cleanly, the port's DM math is fine and the
pipeline problem is streaming-activation dynamics; if it produces the same
fringed object as the pipeline, the DM port bug is reproducible here in
seconds per experiment.

    pixi run python scripts/offline_engine_test.py \
        --probe ~/.cache/holoptycho/recon_411993_t4_mADMM_t3_probe.npy
"""
import argparse
import importlib.util
import os
import types

import numpy as np


def load_streaming_recon():
    """File-based import: skips the holoptycho package __init__ (holoscan)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "holoptycho", "streaming_recon.py")
    spec = importlib.util.spec_from_file_location("streaming_recon", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--dump-dir",
                   default=os.path.expanduser("~/.cache/holoptycho/offline"))
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--update-probe", action="store_true",
                   help="enable probe updates (default frozen, like the runs)")
    args = p.parse_args()

    import cupy as cp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff = np.load(os.path.join(args.dump_dir, "diff_d.npy"))
    pinf = np.load(os.path.join(args.dump_dir, "point_info_d.npy"))
    N = diff.shape[0]
    print(f"{N} points from dumps; windows x {pinf[:,0].min()}..{pinf[:,1].max()} "
          f"y {pinf[:,2].min()}..{pinf[:,3].max()}")

    # Mirror the replay run's engine config (see GUI config + replay overrides).
    config = types.SimpleNamespace(
        nx=256, ny=256,
        gpu_batch_size=1024,
        gpus=[0],
        n_iterations=args.iterations + 1,   # ring depth >= iterations (no wrap)
        init_prb_flag=False,
        prb_path=args.probe,
        xray_energy_kev=9.04,
        lambda_nm=0.137146014993,
        z_m=2.0500008,
        ccd_pixel_um=75.0,
        distance=0.0,
        alpha=1e-3,
        beta=0.9,
        sigma2=5e-05,
        amp_max=1.0, amp_min=0.1, pha_max=0.01, pha_min=-2.0,
        mask_obj_flag=False, mask_prb_flag=False,
        obj_mode_num=1, prb_mode_num=1,
        obj_pad=4,
        precision="single",
        start_update_object=0,
        start_update_probe=(0 if args.update_probe else 10**9),
        x_direction=-1.0, y_direction=-1.0,
        scan_num="411993",
    )

    sr = load_streaming_recon()
    recon = sr.StreamingPtychoRecon(config=config)
    recon.gpu_setup(num_points_max=N)
    # Object canvas: same ranges the replay used (13 um = 8 * 1.625 headroom)
    recon.reset_for_scan("411993", 13.0, 13.0, N)

    # Load the engine's exact inputs.
    recon.diff_d[...] = cp.asarray(diff)
    recon.point_info_d[...] = cp.asarray(pinf.astype(np.int32))
    recon.num_points_recon = N
    recon.initial_probe(N)
    print("probe warm-started; running DM ...", flush=True)

    for it in range(args.iterations):
        recon.iter_once(it)
        if it % 25 == 0:
            obj = recon.obj_mode[0]
            fin = np.isfinite(obj).all()
            print(f"it {it}: obj finite={fin}", flush=True)

    obj = recon.obj_mode[0].copy()
    prb = recon.prb_mode[0].copy()
    amp = np.abs(obj)
    mask = np.abs(amp - np.median(amp)) > 0.02
    rows = np.where(mask.any(axis=1))[0]; cols = np.where(mask.any(axis=0))[0]
    r0, r1 = (rows.min(), rows.max()) if len(rows) else (0, amp.shape[0])
    c0, c1 = (cols.min(), cols.max()) if len(cols) else (0, amp.shape[1])
    print(f"covered region rows {r0}..{r1} cols {c0}..{c1}")

    np.save(os.path.join(args.dump_dir, "engine_port_obj.npy"), obj)
    a = np.abs(obj[r0:r1, c0:c1])
    lo, hi = np.percentile(a, [2, 98])   # robust clim: edge pixels where
    # prb_norm ~ alpha divide into huge values and crush the colormap
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    ax[0, 0].imshow(a, cmap="gray", vmin=lo, vmax=hi)
    ax[0, 0].set_title(f"ENGINE-PORT object amp ({args.iterations} it, clim p2..p98)")
    ax[0, 1].imshow(np.angle(obj[r0:r1, c0:c1]), cmap="twilight")
    ax[0, 1].set_title("object phase")
    ax[1, 0].imshow(np.abs(prb), cmap="viridis"); ax[1, 0].set_title("probe amp")
    ax[1, 1].imshow(np.angle(prb), cmap="twilight"); ax[1, 1].set_title("probe phase")
    out = os.path.join(args.dump_dir, "engine_port_test.png")
    fig.tight_layout(); fig.savefig(out, dpi=90)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
