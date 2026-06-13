"""Standalone offline ePIE reconstruction from Tiled — no pipeline involved.

A debugging/validation harness for the iterative engine's data conventions:
fetches raw frames + encoder positions straight from Tiled, applies the same
preprocessing the pipeline does (crop, saturate-zero, hot-pixel, sqrt), then
the ENGINE conventions (D4 orientation + fftshift to DC-at-corner), and runs
a frozen-probe (or full) ePIE on the GPU.

This validated the iterative recipe for scan 411993: engine feed =
rot90_cw(global crop) + fftshift, warm mADMM probe frozen, encoder positions
upsample-averaged. See AGENTS.md "Iterative engine conventions".

Usage (default pixi env — needs cupy + tiled):
    pixi run python scripts/offline_epie.py --probe /path/to/probe.npy
    pixi run python scripts/offline_epie.py --probe ... --n-frames 8000 --epochs 50

Fetched+preprocessed amplitudes are cached under ~/.cache/holoptycho/offline/
(memmap, written incrementally — an interrupted fetch keeps its progress).
Snapshot PNGs land in the cache dir every --snapshot-every epochs.
"""
import argparse
import os

import numpy as np

D4 = {
    "identity":  lambda a: a,
    "rot90_cw":  lambda a: a.swapaxes(-1, -2)[..., :, ::-1],
    "rot90_ccw": lambda a: a.swapaxes(-1, -2)[..., ::-1, :],
    "rot180":    lambda a: a[..., ::-1, ::-1],
    "fliplr":    lambda a: a[..., :, ::-1],
    "flipud":    lambda a: a[..., ::-1, :],
    "transpose": lambda a: a.swapaxes(-1, -2),
    "antitranspose": lambda a: a.swapaxes(-1, -2)[..., ::-1, ::-1],
}


def get_stream(run, name):
    """Stream access across known Tiled layouts (mirrors config_from_tiled)."""
    try:
        node = run["streams"]["primary"]
    except KeyError:
        node = run["primary"]
    try:
        node = node["data"]
    except KeyError:
        pass
    return node[name]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uid", default="48a107cb-34a5-4a6a-8633-f931db468d35",
                   help="raw run uid in hxn/migration (default: scan 411993)")
    p.add_argument("--probe", default=None,
                   help=".npy probe from a prior reconstruction ((modes,n,n) or "
                   "(n,n) complex). Omit for a cold start: the probe is then "
                   "initialised from the mean diffraction amplitude (the "
                   "engine's probe-from-diff formula) and --update-probe is "
                   "implied.")
    p.add_argument("--n-frames", type=int, default=40000)
    p.add_argument("--crop", default="141,397,80,336",
                   help="row0,row1,col0,col1 of the global-frame crop box "
                        "(default: the segmentation auto-center box for 411993)")
    p.add_argument("--dp-orient", default="rot90_cw", choices=sorted(D4),
                   help="engine D4 orientation relative to the global frame")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.8, help="ePIE object step")
    p.add_argument("--update-probe", action="store_true",
                   help="enable ePIE probe updates (default: frozen probe)")
    p.add_argument("--beta-prb", type=float, default=0.8, help="ePIE probe step")
    p.add_argument("--hot-pixel-threshold", type=float, default=50000.0)
    p.add_argument("--lambda-nm", type=float, default=0.137146014993)
    p.add_argument("--z-m", type=float, default=2.0500008)
    p.add_argument("--ccd-pixel-um", type=float, default=75.0)
    p.add_argument("--x-ratio", type=float, default=-9.542e-05)
    p.add_argument("--y-ratio", type=float, default=-0.00010309)
    p.add_argument("--x-direction", type=float, default=-1.0)
    p.add_argument("--y-direction", type=float, default=-1.0)
    p.add_argument("--swap-pos", action="store_true",
                   help="swap the encoder position channels (INENC2<->INENC3) "
                   "after calibration — pairs each channel with the other "
                   "object axis. Equivalent in effect to a transpose-family "
                   "dp-orient, but expressed deterministically on the "
                   "position side.")
    p.add_argument("--flip-pos-x", action="store_true",
                   help="negate the x positions (on top of --x-direction)")
    p.add_argument("--flip-pos-y", action="store_true",
                   help="negate the y positions (on top of --y-direction)")
    p.add_argument("--cache-dir",
                   default=os.path.expanduser("~/.cache/holoptycho/offline"))
    return p.parse_args()


def fetch_cached(args, npix):
    """Fetch + preprocess frames/positions, cached incrementally on disk."""
    os.makedirs(args.cache_dir, exist_ok=True)
    tag = f"{args.uid[:8]}_{args.n_frames}"
    amp_path = os.path.join(args.cache_dir, f"amp_{tag}.npy")
    pos_path = os.path.join(args.cache_dir, f"pos_{tag}.npz")
    done = amp_path + ".done"
    if os.path.exists(done) and os.path.exists(pos_path):
        print(f"loaded cache {amp_path}", flush=True)
        z = np.load(pos_path)
        return np.load(amp_path, mmap_mode="r"), z["posx_um"], z["posy_um"]

    # Reuse a larger cache of the same uid by slicing its first n frames.
    import glob
    for cand in sorted(glob.glob(os.path.join(args.cache_dir, f"amp_{args.uid[:8]}_*.npy"))):
        try:
            m = int(cand.rsplit("_", 1)[1].split(".")[0])
        except ValueError:
            continue
        cpos = os.path.join(args.cache_dir, f"pos_{args.uid[:8]}_{m}.npz")
        if m >= args.n_frames and os.path.exists(cand + ".done") and os.path.exists(cpos):
            print(f"slicing first {args.n_frames} frames from cache {cand}", flush=True)
            z = np.load(cpos)
            return (np.load(cand, mmap_mode="r")[: args.n_frames],
                    z["posx_um"][: args.n_frames], z["posy_um"][: args.n_frames])

    from tiled.client import from_profile

    run = from_profile("nsls2")["hxn"]["migration"][args.uid]
    det = get_stream(run, "eiger2_image")
    print("detector node:", det.shape, flush=True)
    n_total = det.shape[1] if len(det.shape) == 4 else det.shape[0]
    posx_node = get_stream(run, "inenc2_val")
    upsample = int(round(posx_node.shape[0] / n_total))
    print(f"panda upsample = {upsample}", flush=True)
    n = args.n_frames
    posx = np.asarray(posx_node[: n * upsample]).astype(np.float64)
    posy = np.asarray(get_stream(run, "inenc3_val")[: n * upsample]).astype(np.float64)
    posx = posx.reshape(n, upsample).mean(axis=1) * args.x_ratio * args.x_direction
    posy = posy.reshape(n, upsample).mean(axis=1) * args.y_ratio * args.y_direction
    np.savez(pos_path, posx_um=posx, posy_um=posy)

    r0, r1, c0, c1 = (int(v) for v in args.crop.split(","))
    amp = np.lib.format.open_memmap(amp_path, mode="w+", dtype=np.float16,
                                    shape=(n, npix, npix))
    chunk = 1000
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        arr = np.asarray(det[0, i:j]) if len(det.shape) == 4 else np.asarray(det[i:j])
        dp = arr[:, r0:r1, c0:c1].astype(np.float32)
        dp[dp >= np.iinfo(np.uint16).max] = 0
        dp[dp >= args.hot_pixel_threshold] = 0
        amp[i:j] = np.sqrt(dp).astype(np.float16)
        print(f"fetched+preprocessed {j}/{n}", flush=True)
    amp.flush()
    open(done, "w").close()
    print(f"cached {amp_path}", flush=True)
    return amp, posx, posy


def main():
    args = parse_args()
    import cupy as cp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r0_, r1_, c0_, c1_ = (int(v) for v in args.crop.split(","))
    npix = r1_ - r0_
    assert npix == c1_ - c0_, "crop must be square"
    px_m = args.lambda_nm * 1e-9 * args.z_m / (npix * args.ccd_pixel_um * 1e-6)
    print(f"pixel size {px_m*1e9:.2f} nm", flush=True)

    amp_global, pos_x_um, pos_y_um = fetch_cached(args, npix)
    n = args.n_frames

    # Position-side geometry experiments (cache stores calibrated channels,
    # so these apply cleanly on top).
    if args.swap_pos:
        pos_x_um, pos_y_um = pos_y_um, pos_x_um
        print("positions: channels SWAPPED", flush=True)
    if args.flip_pos_x:
        pos_x_um = -pos_x_um
        print("positions: x negated", flush=True)
    if args.flip_pos_y:
        pos_y_um = -pos_y_um
        print("positions: y negated", flush=True)

    # Engine conventions: D4 orientation, then fftshift to DC-at-corner.
    fn = D4[args.dp_orient]
    work = np.empty(amp_global.shape, dtype=np.float16)
    for i in range(0, n, 2000):
        j = min(i + 2000, n)
        work[i:j] = np.fft.ifftshift(
            np.ascontiguousarray(fn(amp_global[i:j])), axes=(-2, -1))
    print("oriented + DC-shifted", flush=True)

    px_x = np.round((pos_x_um - pos_x_um.min()) * 1e-6 / px_m).astype(int)
    px_y = np.round((pos_y_um - pos_y_um.min()) * 1e-6 / px_m).astype(int)
    H, W = px_x.max() + npix + 64, px_y.max() + npix + 64
    rr, cc = px_x + 32, px_y + 32
    print(f"object {H}x{W}, x extent {px_x.max()}px, y extent {px_y.max()}px", flush=True)

    if args.probe:
        prb0 = np.load(args.probe).squeeze().astype(np.complex64)
        if prb0.ndim == 3:
            prb0 = prb0[0]
        prb = cp.asarray(np.ascontiguousarray(fn(prb0)))
    else:
        # Cold start — the engine's probe-from-diff formula: mean diffraction
        # amplitude back-transformed (work is corner-DC, so plain ifft2 +
        # fftshift centers it). Requires probe updates to converge.
        args.update_probe = True
        mean_amp = cp.asarray(work[:1024].astype(np.float32)).mean(axis=0)
        prb = (cp.fft.fftshift(cp.fft.ifft2(mean_amp))
               * cp.sqrt(cp.float32(npix * npix))).astype(cp.complex64)
    print(f"probe {prb.shape} ({'file' if args.probe else 'from-diff'}), "
          f"update={args.update_probe}", flush=True)

    def recenter_probe(p):
        """Wrap-aware re-centering: the Fourier AMPLITUDE constraint is
        invariant to circular shifts of the probe, so cold-start updates can
        drift it into a corner-wrapped degenerate state. Roll the circular
        intensity centroid back to the array center each epoch (gauge fix)."""
        w = cp.abs(p) ** 2
        ii = cp.arange(npix, dtype=cp.float64)
        ang = 2 * np.pi * ii / npix
        def cent(waxis):
            zc = cp.sum(waxis * cp.exp(1j * ang))
            return (float(cp.angle(zc)) % (2 * np.pi)) * npix / (2 * np.pi)
        cy = cent(w.sum(axis=1))
        cx = cent(w.sum(axis=0))
        return cp.roll(p, (int(round(npix / 2 - cy)), int(round(npix / 2 - cx))),
                       axis=(0, 1))

    dp_show = work[n // 2].astype(np.float32)

    def save_png(o, p, tag):
        fig, ax = plt.subplots(2, 3, figsize=(21, 13))
        ax[0, 0].imshow(np.abs(o), cmap="gray")
        ax[0, 0].set_title(f"object AMP ({args.dp_orient}, {n} frames, {tag})")
        ax[0, 1].imshow(np.angle(o), cmap="twilight"); ax[0, 1].set_title("object PHASE")
        ax[1, 0].imshow(np.abs(p), cmap="viridis"); ax[1, 0].set_title("probe AMP")
        ax[1, 1].imshow(np.angle(p), cmap="twilight"); ax[1, 1].set_title("probe PHASE")
        ax[0, 2].imshow(np.log1p(dp_show), cmap="viridis")
        ax[0, 2].set_title("DP as the ENGINE sees it (log, DC at corners)")
        ax[1, 2].imshow(np.log1p(np.fft.fftshift(dp_show)), cmap="viridis")
        ax[1, 2].set_title("same DP, re-centered for viewing")
        out = os.path.join(args.cache_dir, f"epie_{tag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
        print(f"saved {out}", flush=True)

    obj = cp.ones((H, W), dtype=cp.complex64)
    prb_max2 = float(cp.max(cp.abs(prb) ** 2))
    for ep in range(args.epochs):
        err = 0.0
        for i in range(n):
            dk = cp.asarray(work[i]).astype(cp.float32)
            a, b = int(rr[i]), int(rr[i]) + npix
            c, d = int(cc[i]), int(cc[i]) + npix
            patch = obj[a:b, c:d].copy() if args.update_probe else obj[a:b, c:d]
            psi = prb * patch
            PSI = cp.fft.fft2(psi, norm="ortho")
            aPSI = cp.abs(PSI) + 1e-8
            err += float(cp.sum((aPSI - dk) ** 2))
            dpsi = cp.fft.ifft2(dk * PSI / aPSI, norm="ortho") - psi
            obj[a:b, c:d] = patch + args.alpha * cp.conj(prb) / prb_max2 * dpsi
            if args.update_probe:
                obj_max2 = float(cp.max(cp.abs(patch) ** 2)) + 1e-8
                prb = prb + args.beta_prb * cp.conj(patch) / obj_max2 * dpsi
                prb_max2 = float(cp.max(cp.abs(prb) ** 2))
        if args.update_probe:
            prb = recenter_probe(prb)
            prb_max2 = float(cp.max(cp.abs(prb) ** 2))
        print(f"epoch {ep}: err/frame {err/n:.0f}", flush=True)
        if (ep + 1) % args.snapshot_every == 0:
            save_png(cp.asnumpy(obj), cp.asnumpy(prb), f"ep{ep+1:03d}")

    o = cp.asnumpy(obj)
    np.save(os.path.join(args.cache_dir, "epie_object.npy"), o)
    save_png(o, cp.asnumpy(prb), "final")


if __name__ == "__main__":
    main()
