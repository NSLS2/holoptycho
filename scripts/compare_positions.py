"""Compare PandA-derived per-frame positions (what holoptycho writes to
``<run>/positions_um``) against an offline conversion of the same encoder
channels from the raw HXN scan in tiled. The offline conversion mirrors what
``preprocess.py::PointProcessorOp.process_point_info`` does — averages the
10x-upsampled encoder reads and multiplies by ``x_ratio*x_direction`` /
``y_ratio*y_direction`` — but operates on the full encoder array at once.

If streaming and offline match, the streaming math is faithful and any recon
divergence sits elsewhere. If they differ, the streaming path has a bug.

Run inside the replay env:

    pixi run -e replay python scripts/compare_positions.py \\
        --run-uid 96f4149c62c9472a8c7d432b3a031dd1 \\
        --raw-uid 7fcf8d25-f609-4f2c-8710-44793341455f \\
        --x-ratio -9.542e-05 --y-ratio -0.00010309 \\
        --x-direction -1.0 --y-direction -1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tiled.client import from_uri

# Reuse the proven raw-scan loaders that handle HXN's migration/raw layout
# differences and the long httpx timeouts the migration server needs. Direct
# `from_uri('https://tiled.nsls2.bnl.gov')['hxn']['raw'][...]` 500s on encoder
# arrays for this scan; `open_tiled_node('.../hxn/migration')` succeeds.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_from_tiled import open_tiled_node, lookup_run, get_stream  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-uid", required=True,
                   help="UID of a holoptycho run with positions_um written.")
    p.add_argument("--raw-uid", required=True,
                   help="UID of the raw HXN scan.")
    p.add_argument("--x-key", default="inenc2_val")
    p.add_argument("--y-key", default="inenc3_val")
    p.add_argument("--upsample", type=int, default=10,
                   help="Encoder samples per frame (PandA upsample factor).")
    p.add_argument("--x-ratio", type=float, required=True)
    p.add_argument("--y-ratio", type=float, required=True)
    p.add_argument("--x-direction", type=float, default=-1.0)
    p.add_argument("--y-direction", type=float, default=-1.0)
    p.add_argument("--tiled-url", default="https://tiled.nsls2.bnl.gov")
    p.add_argument("--raw-tiled-url", default="https://tiled.nsls2.bnl.gov/hxn/migration",
                   help="Tiled URL the *raw* scan loaders should use (migration layout).")
    p.add_argument("--n-show", type=int, default=10)
    args = p.parse_args()

    # Streaming positions: read from the processed catalog (different server path).
    proc_root = from_uri(args.tiled_url)
    proc = proc_root["hxn"]["processed"]["holoptycho"][args.run_uid]
    panda = np.asarray(proc["positions_um"][:])
    finite = np.isfinite(panda).all(axis=1)
    print(f"streaming positions_um: shape={panda.shape}, finite={finite.sum()}/{len(panda)}")

    # Raw scan via the migration loader the replay script uses successfully.
    raw_client = open_tiled_node(args.raw_tiled_url)
    run = lookup_run(raw_client, args.raw_uid, args.raw_tiled_url)
    stream = get_stream(run, "primary")
    enc_x_full = np.asarray(stream[args.x_key].read()).ravel()
    enc_y_full = np.asarray(stream[args.y_key].read()).ravel()
    n_total = enc_x_full.size // args.upsample
    print(f"raw {args.x_key}/{args.y_key}: {enc_x_full.size} samples → {n_total} frames at {args.upsample}x upsample")

    enc_x = enc_x_full[: n_total * args.upsample].reshape(n_total, args.upsample).mean(axis=1)
    enc_y = enc_y_full[: n_total * args.upsample].reshape(n_total, args.upsample).mean(axis=1)
    offline_x = enc_x * args.x_ratio * args.x_direction
    offline_y = enc_y * args.y_ratio * args.y_direction
    offline = np.stack([offline_x, offline_y], axis=1)
    print(f"offline conversion: shape={offline.shape}")

    # Stats per side
    print(f"\nstreaming x: range {panda[finite,0].min():+.4f} .. {panda[finite,0].max():+.4f}, "
          f"span {np.ptp(panda[finite,0]):.4f} um")
    print(f"streaming y: range {panda[finite,1].min():+.4f} .. {panda[finite,1].max():+.4f}, "
          f"span {np.ptp(panda[finite,1]):.4f} um")
    print(f"offline   x: range {offline_x.min():+.4f} .. {offline_x.max():+.4f}, "
          f"span {np.ptp(offline_x):.4f} um")
    print(f"offline   y: range {offline_y.min():+.4f} .. {offline_y.max():+.4f}, "
          f"span {np.ptp(offline_y):.4f} um")

    # Frame-by-frame comparison on overlap
    n = min(len(panda), len(offline))
    mask = np.isfinite(panda[:n]).all(axis=1)
    if not mask.any():
        print("No finite streaming frames; cannot diff.")
        return
    dx = panda[:n, 0][mask] - offline[:n, 0][mask]
    dy = panda[:n, 1][mask] - offline[:n, 1][mask]
    print(f"\nDiff (streaming - offline) over {mask.sum()} frames:")
    print(f"  dx: mean={dx.mean():+.4e} um, std={dx.std():.4e}, max|d|={np.abs(dx).max():.4e}")
    print(f"  dy: mean={dy.mean():+.4e} um, std={dy.std():.4e}, max|d|={np.abs(dy).max():.4e}")

    # Linear fits (slope=1, intercept=0 means perfect match)
    if np.std(offline[:n, 0][mask]) > 0:
        ax, bx = np.polyfit(offline[:n, 0][mask], panda[:n, 0][mask], 1)
        print(f"  fit x: streaming = {ax:+.6f} * offline {bx:+.4e}")
    if np.std(offline[:n, 1][mask]) > 0:
        ay, by = np.polyfit(offline[:n, 1][mask], panda[:n, 1][mask], 1)
        print(f"  fit y: streaming = {ay:+.6f} * offline {by:+.4e}")

    # Side-by-side head/tail dump
    nshow = args.n_show
    finite_idx = np.where(mask)[0]
    print(f"\nFirst {nshow} finite frames:")
    print(f"  {'idx':>5}  {'stream_x':>10} {'stream_y':>10}  {'offl_x':>10} {'offl_y':>10}  {'dx':>8} {'dy':>8}")
    for i in finite_idx[:nshow]:
        print(f"  [{i:5d}] {panda[i,0]:+10.4f} {panda[i,1]:+10.4f}  {offline[i,0]:+10.4f} {offline[i,1]:+10.4f}  {panda[i,0]-offline[i,0]:+8.4f} {panda[i,1]-offline[i,1]:+8.4f}")
    print(f"\nLast {nshow} finite frames:")
    for i in finite_idx[-nshow:]:
        print(f"  [{i:5d}] {panda[i,0]:+10.4f} {panda[i,1]:+10.4f}  {offline[i,0]:+10.4f} {offline[i,1]:+10.4f}  {panda[i,0]-offline[i,0]:+8.4f} {panda[i,1]-offline[i,1]:+8.4f}")


if __name__ == "__main__":
    main()
