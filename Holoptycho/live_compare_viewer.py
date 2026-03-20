#!/usr/bin/env python3
"""
Live side-by-side viewer: PtychoRecon (iterative) vs PtychoViT (single-shot).

Run this in a separate terminal WHILE simulate mode is running in the GUI.
It polls both outputs and updates a matplotlib figure every ~1 second.

Usage:
    python live_compare_viewer.py /path/to/scan.h5

    # Custom save directory:
    python live_compare_viewer.py /path/to/scan.h5 --save-dir /data/users/Holoscan
"""

import argparse
import os
import time
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load_scan_info(h5_path):
    """Load scan positions (microns) and pixel size from H5 file."""
    with h5py.File(h5_path, "r") as f:
        points = f["points"][:]  # [2, nz] — positions in microns
        lambda_nm = float(f["lambda_nm"][()])
        z_m = float(f["z_m"][()])
        nx = f["diffamp"].shape[1]
        ccd_pixel_um = float(f["ccd_pixel_um"][()])
    pixel_size_m = lambda_nm * 1e-9 * z_m / (nx * ccd_pixel_um * 1e-6)
    return points, pixel_size_m


class IncrementalStitcher:
    """Incrementally stitches VIT patches into a mosaic.

    Follows the same approach as edgePtychoViT/stitch_viewer.py:
    - Scan positions (microns) are converted to meters
    - Centered local patch coordinates are computed from pixel_size_m
    - Patches are cropped (inner_crop) to reduce edge artifacts
    - RegularGridInterpolator places each patch onto a common meter-scale grid
    - Overlapping regions are averaged
    """

    def __init__(self, positions_um, pixel_size_m, inner_crop=32, pad_m=0.5e-6):
        # Convert scan positions from microns to meters
        self.pos_x = positions_um[0].astype(np.float64) * 1e-6
        self.pos_y = positions_um[1].astype(np.float64) * 1e-6
        self.pixel_size_m = pixel_size_m
        self.inner_crop = inner_crop
        self.pad_m = pad_m

        # Deferred until first batch (need actual patch size)
        self.xx = None
        self.yy = None
        self.mosaic = None
        self.counts = None
        self.x_grid = None
        self.y_grid = None
        self.batches_stitched = 0
        self._initialized = False

    def _init_from_patch(self, patch_size):
        """Build coordinate arrays once we know the actual patch dimensions."""
        ps = self.pixel_size_m

        # Centered local coordinates for one patch (same as stitch_viewer.py)
        coords = (np.arange(patch_size, dtype=np.float64) - (patch_size - 1) / 2.0) * ps
        self.xx = coords
        self.yy = coords.copy()

        # Global canvas grid covering all scan positions + padding
        self.x_grid = np.arange(
            self.pos_x.min() - self.pad_m,
            self.pos_x.max() + self.pad_m,
            ps,
        )
        self.y_grid = np.arange(
            self.pos_y.min() - self.pad_m,
            self.pos_y.max() + self.pad_m,
            ps,
        )
        self.mosaic = np.zeros((len(self.y_grid), len(self.x_grid)), dtype=np.float32)
        self.counts = np.zeros_like(self.mosaic, dtype=np.int32)
        self._initialized = True

        print(f"  Stitcher: patch={patch_size}, crop={self.inner_crop}, "
              f"canvas={self.mosaic.shape}, pixel={ps*1e9:.2f} nm")

    def add_batch(self, pred, indices):
        """Stitch a batch of VIT predictions into the mosaic.

        Args:
            pred: [B, 2, H, W] float32 -- channel 0=amplitude, 1=phase
            indices: [B] int -- frame indices into the positions array
        """
        if not self._initialized:
            self._init_from_patch(pred.shape[-1])

        crop = self.inner_crop

        for i in range(len(pred)):
            idx = int(indices[i])
            if idx >= len(self.pos_x):
                continue

            raw_patch = pred[i, 1]  # phase channel

            # Crop border pixels to reduce edge artifacts (like stitch_viewer.py)
            if crop > 0 and raw_patch.shape[0] > 2 * crop:
                data = raw_patch[crop:-crop, crop:-crop]
                x_local = (self.xx + self.pos_x[idx])[crop:-crop]
                y_local = (self.yy + self.pos_y[idx])[crop:-crop]
            else:
                data = raw_patch
                x_local = self.xx + self.pos_x[idx]
                y_local = self.yy + self.pos_y[idx]

            # Find canvas region this patch covers
            ix0 = max(int(np.searchsorted(self.x_grid, x_local[0], side="left")), 0)
            ix1 = min(int(np.searchsorted(self.x_grid, x_local[-1], side="right")), len(self.x_grid))
            iy0 = max(int(np.searchsorted(self.y_grid, y_local[0], side="left")), 0)
            iy1 = min(int(np.searchsorted(self.y_grid, y_local[-1], side="right")), len(self.y_grid))
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            # Interpolate patch onto canvas grid
            interp = RegularGridInterpolator(
                (y_local, x_local), data, bounds_error=False, fill_value=0.0
            )
            yy, xx = np.meshgrid(self.y_grid[iy0:iy1], self.x_grid[ix0:ix1], indexing="ij")
            vals = interp(np.stack([yy.ravel(), xx.ravel()], axis=-1)).reshape(yy.shape)

            mask = vals != 0.0
            self.mosaic[iy0:iy1, ix0:ix1] += vals
            self.counts[iy0:iy1, ix0:ix1] += mask.astype(np.int32)

        self.batches_stitched += 1

    def get_stitched(self):
        if self.mosaic is None:
            return np.zeros((2, 2), dtype=np.float32)
        return (self.mosaic / np.maximum(self.counts, 1)).astype(np.float32)


def _clear_output_dir(save_dir):
    """Remove stale VIT batch files and live results from a previous run."""
    for pattern in ["vit_batch_*_pred.npy", "vit_batch_*_indices.npy",
                     "vit_pred_latest.npy", "obj_live.npy", "prb_live.npy"]:
        for f in glob(os.path.join(save_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass
    print(f"Cleared old results from {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Live side-by-side: iterative recon vs VIT"
    )
    parser.add_argument("h5_path", help="Path to the scan H5 file")
    parser.add_argument(
        "--save-dir",
        default="/data/users/Holoscan",
        help="Directory where VIT batches and obj_live.npy are saved",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Poll interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--inner-crop",
        type=int,
        default=32,
        help="Crop border pixels from each VIT patch before stitching (default: 32)",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    h5_path = args.h5_path

    # Load scan positions (in microns) and pixel size
    print(f"Loading positions from {h5_path}...")
    positions_um, pixel_size_m = load_scan_info(h5_path)
    nz = positions_um.shape[1]
    print(f"  {nz} scan points, pixel size: {pixel_size_m*1e9:.2f} nm")
    print(f"  scan span: {(positions_um[0].max()-positions_um[0].min()):.2f} x "
          f"{(positions_um[1].max()-positions_um[1].min()):.2f} um")

    # Open H5 for reading diffraction patterns on demand
    h5f = h5py.File(h5_path, "r", locking=False)
    diffamp_dset = h5f["diffamp"]  # [nz, nx, ny]

    stitcher = IncrementalStitcher(positions_um, pixel_size_m, inner_crop=args.inner_crop)

    # Clear stale results from previous runs so we start fresh
    _clear_output_dir(save_dir)

    # Set up matplotlib — 3 panels: diffraction pattern, iterative recon, VIT
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Waiting for data...", fontsize=13)
    for ax in axes:
        ax.set_axis_off()
    axes[0].set_title("Diffraction pattern")
    axes[1].set_title("PtychoRecon (iterative)")
    axes[2].set_title("PtychoViT (single-shot)")

    im_diff = axes[0].imshow(
        np.zeros((2, 2)), cmap="gray", interpolation="nearest"
    )
    im_recon = axes[1].imshow(
        np.zeros((2, 2)), cmap="twilight", interpolation="nearest"
    )
    im_vit = axes[2].imshow(
        np.zeros((2, 2)), cmap="twilight", interpolation="nearest"
    )
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    last_batch_num = 0
    latest_frame_idx = 0
    batch0_mtime = 0.0  # track batch 0 mtime to detect scan restarts

    print(f"Polling {save_dir} for results... (Ctrl+C to stop)")
    try:
        while True:
            updated = False

            # --- Poll iterative recon ---
            obj_path = os.path.join(save_dir, "obj_live.npy")
            if os.path.exists(obj_path) and os.path.getsize(obj_path) > 0:
                try:
                    obj_live = np.load(obj_path)
                    recon_phase = np.rot90(np.angle(obj_live[0]))
                    im_recon.set_data(recon_phase)
                    im_recon.autoscale()
                    updated = True
                except Exception as e:
                    print(f"  obj_live load error: {e}")

            # --- Detect new scan (batch 0 reappears with newer mtime) ---
            batch0_path = os.path.join(save_dir, "vit_batch_000000_pred.npy")
            if os.path.exists(batch0_path):
                mtime = os.path.getmtime(batch0_path)
                if mtime > batch0_mtime and last_batch_num > 0:
                    print("  New scan detected — resetting viewer")
                    last_batch_num = 0
                    latest_frame_idx = 0
                    stitcher = IncrementalStitcher(
                        positions_um, pixel_size_m, inner_crop=args.inner_crop
                    )
                    im_vit.set_data(np.zeros((2, 2)))
                    im_recon.set_data(np.zeros((2, 2)))
                batch0_mtime = mtime

            # --- Poll VIT batches (drain all available) ---
            while True:
                pred_path = os.path.join(
                    save_dir, f"vit_batch_{last_batch_num:06d}_pred.npy"
                )
                idx_path = os.path.join(
                    save_dir, f"vit_batch_{last_batch_num:06d}_indices.npy"
                )
                if not (os.path.exists(pred_path) and os.path.exists(idx_path)):
                    break
                try:
                    pred = np.load(pred_path)
                    indices = np.load(idx_path)
                    stitcher.add_batch(pred, indices)
                    latest_frame_idx = int(indices[-1])
                    if last_batch_num % 10 == 0:
                        print(f"  VIT batch {last_batch_num}: "
                              f"indices {indices[0]}-{indices[-1]}")
                    last_batch_num += 1
                except Exception as e:
                    print(f"  VIT batch {last_batch_num} error: {e}")
                    break

            # --- Update diffraction pattern display ---
            if latest_frame_idx > 0 and latest_frame_idx < nz:
                try:
                    diff_frame = diffamp_dset[latest_frame_idx]
                    im_diff.set_data(np.log1p(np.fft.fftshift(diff_frame)))
                    im_diff.autoscale()
                    updated = True
                except Exception:
                    pass

            # --- Update VIT stitched display ---
            vit_stitched = stitcher.get_stitched()
            if stitcher.batches_stitched > 0:
                # Mosaic has rows=pos_y, cols=pos_x.
                # Iterative recon obj uses rot90 to match display convention.
                # VIT mosaic is already in (row=slow_axis, col=fast_axis) —
                # no rotation needed.
                im_vit.set_data(vit_stitched)
                im_vit.autoscale()
                updated = True

            if updated:
                patches_done = last_batch_num * 64
                fig.suptitle(
                    f"PtychoRecon vs PtychoViT -- "
                    f"~{min(patches_done, nz)}/{nz} frames",
                    fontsize=13,
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            plt.pause(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")
        out_path = os.path.join(save_dir, "recon_vs_vit_comparison.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    finally:
        h5f.close()


if __name__ == "__main__":
    main()
