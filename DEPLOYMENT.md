# Deployment Guide: Holoscan Ptychography Pipeline with PtychoViT Inference

## 1. Overview

This pipeline performs **live ptychographic reconstruction** and **PtychoViT neural-network inference in parallel** on streaming X-ray diffraction data. It is built on [NVIDIA Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) and runs inside a Podman/Docker container with GPU acceleration.

**Two parallel branches process every batch of diffraction frames:**

| Branch | Method | GPU | Output |
|--------|--------|-----|--------|
| **PtychoRecon** | Iterative phase retrieval (CuPy + Numba CUDA) | GPU 0 | `obj_live.npy`, `prb_live.npy` |
| **PtychoViTInferenceOp** | Single-shot TensorRT inference | GPU 1 | `vit_batch_*_pred.npy` |

**Two application modes:**

- **`PtychoApp`** (live) — receives ZMQ streams from Eiger detector + PandaBox encoders at NSLS-II HXN beamline.
- **`PtychoSimulApp`** (simulate) — replays an HDF5 scan file from disk. Used for testing and development.

---

## 2. Prerequisites

### Required repositories (sibling directories under a common parent)

```
parent_dir/
├── holoscan-framework/        # This repo (Holoscan pipeline)
├── src/                       # nsls2ptycho package (branch: holoscan)
│   └── nsls2ptycho/
│       └── core/
│           ├── ptycho/        # Iterative reconstruction engine
│           └── Holoptycho/    # ← symlink or copy of holoscan-framework/Holoptycho/
└── (optional) edgePtychoViT/  # ONNX export / TRT engine build scripts
```

> **Important:** `Holoptycho/` lives inside `nsls2ptycho.core` because it uses relative imports (`from ..ptycho.utils import parse_config`). The entry point is:
> ```
> python -m nsls2ptycho.core.Holoptycho <config_file> [simulate]
> ```

### Required hardware

- **2 NVIDIA GPUs** (e.g. A100, H100) — GPU 0 for iterative recon, GPU 1 for TRT inference.
  Single-GPU is possible but risks CUDA context crashes (CuPy + PyCUDA on the same device from different threads).
- NVIDIA Container Toolkit (CDI mode) or `nvidia-container-runtime`

### Required data (simulate mode)

- An HDF5 scan file (e.g. `scan_320045.h5`) containing:
  - `diffamp` — `[nz, nx, ny]` float32 preprocessed diffraction amplitudes
  - `points` — `[2, nz]` float64 scan positions in microns
  - `lambda_nm`, `z_m`, `ccd_pixel_um` — optical parameters
- A `ptycho_config` text file matching the scan (see `eiger_test/ptycho_holo/ptycho_config_320045.txt`)

### Required models (VIT inference)

- A TensorRT `.engine` file built for the **target GPU architecture** (see Section 4).
  Engine files are **not portable** between GPU architectures (A100 vs H100 vs A6000 etc.).

---

## 3. Building the Container

### Dockerfile

The Dockerfile lives at `podman_dir/Dockerfile`. It:

1. Starts from `nvidia/cuda:12.6.0-devel-ubuntu22.04`
2. Installs pixi (package manager) at `/usr/local`
3. Installs nsight-systems for profiling
4. **Bakes the pixi environment at `/pixi_env/`** — this is critical; the old path `/podman_dir/` would get clobbered by bind-mounts at runtime

### Build command

Build on a machine with internet access (e.g. `axinite`):

```bash
cd holoscan-framework
podman build ./podman_dir -t hxn-ptycho-holoscan --network host
```

> **Note:** Some machines (e.g. `jade`) may fail with permission errors on `/etc/hosts`. Build on `axinite` instead, or use `--no-hosts` and `TMPDIR=/local/pmyint/tmp`.

### Transfer container between machines

```bash
# Save on build machine
podman save hxn-ptycho-holoscan | gzip > /local/pmyint/hxn-ptycho-holoscan.tar.gz

# Copy to target machine
scp /local/pmyint/hxn-ptycho-holoscan.tar.gz target-host:/local/pmyint/

# Load on target machine
gunzip -c /local/pmyint/hxn-ptycho-holoscan.tar.gz | podman load
```

---

## 4. Preparing Models (ONNX Export + TRT Engine Build)

### Step 1: Export ONNX (needs PyTorch — run outside the container)

```bash
python export_edge_onnx.py \
  --config ../ptycho-vit/configs/9id/config_finetune.yaml \
  --checkpoint ../ptycho-vit/reference_resources/model/best_model.pth \
  --output /local/pmyint/models/ptycho_vit_amp_phase_b64.onnx \
  --batch-size 64 \
  --output-kind amp_phase
```

### Step 2: Build TRT engine (MUST run on the target GPU)

```bash
podman run --rm --device nvidia.com/gpu=all \
    -v /local/pmyint/models:/models \
    hxn-ptycho-holoscan \
    pixi run python /edgePtychoViT/build_trt_engine.py \
      --onnx /models/ptycho_vit_amp_phase_b64.onnx \
      --engine /models/ptycho_vit_amp_phase_b64.engine \
      --fp16
```

> **TRT engines are GPU-architecture-specific.** An engine built on A100 will NOT work on H100 or A6000. Always build on the exact GPU where inference will run.

---

## 5. Running the Pipeline

### 5a. Launch the container

```bash
podman run --rm -it --userns=keep-id \
    --device nvidia.com/gpu=all \
    --shm-size=32g \
    -v /path/to/ptycho_gui_holoscan:/ptycho_gui_holoscan:ro \
    -v /path/to/edgePtychoViT:/edgePtychoViT:ro \
    -v /local/pmyint/holoscan-test/models:/models:ro \
    -v /local/pmyint/holoscan-test/data:/data \
    -v /local/pmyint/holoscan-test/output:/data/users/Holoscan \
    -e DISPLAY=:0 \
    hxn-ptycho-holoscan \
    /bin/bash
```

Or use the provided `run_container` script (edit the `/path/to/models` line first):

```bash
./run_container
```

### 5b. Inside the container — activate pixi environment

```bash
cd /pixi_env
pixi shell
```

### 5c. Run simulate mode (H5 replay)

```bash
python -m nsls2ptycho.core.Holoptycho \
    /ptycho_gui_holoscan/holoscan-framework/eiger_test/ptycho_holo/ptycho_config_320045.txt \
    simulate
```

This will:
- Read the H5 scan file specified in the config (`working_directory/scan_<scan_num>.h5`)
- Stream batches of 64 frames through both PtychoRecon and PtychoViT
- Save iterative results to `/data/users/Holoscan/obj_live.npy` (updated every 10 iterations)
- Save VIT predictions to `/data/users/Holoscan/vit_batch_NNNNNN_pred.npy`

### 5d. Run live mode (NSLS-II beamline)

Requires environment variables for ZMQ encryption (see `podman_dir/bashrc_template`):

```bash
export CLIENT_SECRET_KEY=<key>
export CLIENT_PUBLIC_KEY=<key>
export SERVER_STREAM_SOURCE="tcp://10.66.16.45:5559"  # Eiger2
export SERVER_PUBLIC_KEY=<key>
export PANDA_STREAM_SOURCE="tcp://10.66.16.45:6666"   # PandaBox

python -m nsls2ptycho.core.Holoptycho \
    /ptycho_gui_holoscan/holoscan-framework/eiger_test/ptycho_holo/ptycho_config.txt
```

---

## 6. Container Management

### Check running containers

```bash
podman ps
```

### Open a second shell in a running container

```bash
podman exec -it <container_id> /bin/bash
cd /pixi_env && pixi shell
```

### Stop a running container

```bash
podman stop <container_id>
```

---

## 7. Live Comparison Viewer

The viewer displays 3 panels side-by-side: diffraction pattern | iterative recon | VIT stitched mosaic.

**Run from the host** (not inside the container) with X11 forwarding:

```bash
# SSH with X11 forwarding
ssh -Y user@host

# Run the viewer
python /path/to/holoscan-framework/Holoptycho/live_compare_viewer.py \
    /path/to/scan_320045.h5 \
    --save-dir /local/pmyint/holoscan-test/output \
    --interval 0.5
```

The viewer polls the output directory for `obj_live.npy` (iterative) and `vit_batch_*_pred.npy` (VIT), updating the display as new batches arrive. On Ctrl+C it saves a comparison PNG.

### Viewer features

- Diffraction pattern panel: shows `log1p(diffamp)` from the H5 file
- Iterative recon panel: reads `obj_live.npy`, displays phase via `np.angle()`
- VIT panel: incrementally stitches 128x128 patches into a full mosaic using `IncrementalStitcher`
  - Crops inner 32 pixels from each patch edge to reduce artifacts
  - Uses `RegularGridInterpolator` to place patches on a common meter-scale grid
  - Overlapping regions are averaged

---

## 8. Architecture

### PtychoSimulApp (simulate mode) data flow

```
                         ┌─────────────────────┐
                         │     InitSimul        │
                         │  (H5 file replay)    │
                         └────┬───────────┬─────┘
                              │           │
                    diff_amp  │           │  diff_amp
                  + indices   │           │  + indices
                              ▼           ▼
                    ┌─────────────┐  ┌────────────────────┐
                    │ ImageSendOp │  │ PtychoViTInference  │
                    │ (→ GPU 0)   │  │   (TRT on GPU 1)   │
                    └──────┬──────┘  └─────────┬──────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐    ┌──────────────┐
                    │ PointProc   │    │ SaveViTResult │
                    │ → PtychoRec │    │ (per-batch)   │
                    └──────┬──────┘    └──────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ SaveResult  │
                    │ SaveLiveRes │
                    └─────────────┘
```

### PtychoApp (live mode) data flow

```
  ┌──────────────┐   ┌──────────────┐
  │ EigerZmqRxOp │   │ PositionRxOp │
  │ (ZMQ stream) │   │ (PandaBox)   │
  └──────┬───────┘   └──────┬───────┘
         │                  │
         ▼                  │
  ┌──────────────┐          │
  │ Decompress   │          │
  └──────┬───────┘          │
         │                  │
         ▼                  │
  ┌──────────────┐          │
  │ ImageBatchOp │          │
  └──────┬───────┘          │
         │                  │
         ▼                  │
  ┌──────────────────┐      │
  │ImagePreprocessor │      │
  │ (bad pixel, rot, │      │
  │  fftshift, sqrt) │      │
  └──┬───────────┬───┘      │
     │           │           │
     ▼           ▼           │
  ┌────────┐ ┌────────────┐  │
  │ImgSend │ │PtychoViT   │  │
  │(GPU 0) │ │(TRT GPU 1) │  │
  └───┬────┘ └─────┬──────┘  │
      │            │          │
      ▼            ▼          │
  ┌─────────┐ ┌──────────┐   │
  │PointProc│ │SaveViTRes│   │
  │→PtyRecon│ └──────────┘   │
  └───┬─────┘                 │
      │◄──────────────────────┘
      ▼
  ┌──────────────┐
  │ SaveResult   │
  │ SaveLiveRes  │
  └──────────────┘
```

### Key differences

| Aspect | PtychoSimulApp | PtychoApp |
|--------|---------------|-----------|
| Data source | InitSimul (H5 file) | EigerZmqRxOp (ZMQ) + PositionRxOp |
| Preprocessing | None (H5 data is already preprocessed) | ImageBatchOp → ImagePreprocessorOp (bad pixel, rot90, fftshift, sqrt) |
| VIT input tapped from | InitSimul `diff_amp` output | ImagePreprocessorOp `diff_amp` output |
| Required env vars | None | `SERVER_STREAM_SOURCE`, `SERVER_PUBLIC_KEY`, `CLIENT_*`, `PANDA_STREAM_SOURCE` |

---

## 9. Known Issues & Workarounds

### Pixi environment clobbered by bind-mount

**Problem:** If the Dockerfile installs pixi at `/podman_dir/` and `run_container` mounts `./podman_dir:/podman_dir`, the pre-installed `.pixi/` directory gets overwritten.

**Solution:** The Dockerfile installs pixi at `/pixi_env/` (never bind-mounted). Do NOT change the `WORKDIR` or add a mount that shadows `/pixi_env/`.

### hxntools import error in simulate mode

**Problem:** `from hxntools.motor_info import motor_table` fails if hxntools is not installed.

**Solution:** Already handled — the import is wrapped in a `try/except` that sets `motor_table = None`. Simulate mode does not use it; live mode requires hxntools to be mounted (via `run_container`'s `-v .../hxntools:/hxntools`).

### edgePtychoViT symlink / path

**Problem:** `vit_inference.py` imports from `.edgePtychoViT.helper_trt`, which lives at `Holoptycho/edgePtychoViT/`. This subdirectory must exist inside the `Holoptycho` package.

**Solution:** Either copy or symlink `edgePtychoViT/` into `Holoptycho/`. Verify `__init__.py` exists in the directory.

### Spatial padding for mismatched model/data sizes

**Problem:** If the diffraction data is 128x128 but the TRT engine expects 256x256 input, inference will fail.

**Solution:** `PtychoViTInferenceOp` automatically center-pads the input to the engine's expected spatial dimensions and crops the output back. No manual intervention needed.

### TRT engines are GPU-architecture-specific

**Problem:** A `.engine` built on one GPU model (e.g. A100) will not load on a different model (e.g. H100).

**Solution:** Always build the TRT engine on the same GPU architecture where it will be used (see Section 4).

### Podman build fails on certain machines

**Problem:** `/etc/hosts` permission errors during `podman build` on some machines.

**Solution:** Build on a machine with internet access and correct permissions (e.g. `axinite`), then transfer via `podman save/load`.

---

## 10. File Inventory

### Core pipeline (`Holoptycho/`)

| File | Purpose |
|------|---------|
| `__main__.py` | Entry point — calls `ptycho_holo.main()` |
| `ptycho_holo.py` | `PtychoSimulApp`, `PtychoApp`, `PtychoRecon`, `InitRecon`, `SaveResult`, `SaveLiveResult` |
| `datasource.py` | `EigerZmqRxOp` (ZMQ receiver), `EigerDecompressOp`, `PositionRxOp` |
| `preprocess.py` | `ImageBatchOp`, `ImagePreprocessorOp`, `PointProcessorOp`, `ImageSendOp` |
| `vit_inference.py` | `PtychoViTInferenceOp` (TRT inference), `SaveViTResult` |
| `live_simulation.py` | `InitSimul` — H5 file replay for simulate mode |
| `live_compare_viewer.py` | Live side-by-side viewer (run from host with X11) |
| `liverecon_utils.py` | `parse_scan_header()` utility |
| `edgePtychoViT/` | `helper_trt.py` (TRT engine loading/inference), `build_trt_engine.py` (ONNX → TRT) |

### Container build (`podman_dir/`)

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image definition (CUDA 12.6 + pixi + nsight) |
| `pixi.toml` | Dependency specification (holoscan, cupy, tensorrt, pycuda, etc.) |
| `pixi.lock` | Locked dependency versions for reproducible builds |
| `bashrc_template` | Template for ZMQ encryption keys and stream endpoints |

### Scripts (repo root)

| File | Purpose |
|------|---------|
| `build_container` | One-liner to build the container image |
| `run_container` | Podman run with all volume mounts and env vars |

### Test pipelines (`eiger_test/`)

| File | Purpose |
|------|---------|
| `holoscan_config.yaml` | Eiger/PandaBox connection settings |
| `ptycho_holo/ptycho_config*.txt` | Reconstruction parameter files per scan |
| `pipeline_source.py` | Test: ZMQ receiver only |
| `pipeline_preprocess.py` | Test: receiver + preprocessing |
| `pipeline_ptycho.py` | Test: full pipeline (without VIT) |

### Eiger simulation (`eiger_simulation/`)

| File | Purpose |
|------|---------|
| `Dockerfile` | Simulated Eiger detector API (SimplonAPI) |
| `trigger_detector.py` | REST client to trigger simulated acquisition |

---

## Appendix: Environment Variables Reference

| Variable | Required for | Description |
|----------|-------------|-------------|
| `PYTHONPATH` | Both modes | Must include: `nsls2ptycho/src`, `holoscan-framework`, `hxntools/src` |
| `SERVER_STREAM_SOURCE` | Live only | ZMQ endpoint for Eiger stream (e.g. `tcp://10.66.16.45:5559`) |
| `SERVER_PUBLIC_KEY` | Live only | ZMQ CURVE public key for Eiger |
| `CLIENT_SECRET_KEY` | Live only | ZMQ CURVE secret key |
| `CLIENT_PUBLIC_KEY` | Live only | ZMQ CURVE public key |
| `PANDA_STREAM_SOURCE` | Live only | ZMQ endpoint for PandaBox positions |
| `OMPI_ALLOW_RUN_AS_ROOT` | Container | Set to `1` for MPI inside container |
| `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM` | Container | Set to `1` for MPI inside container |
| `HOLOSCAN_ENABLE_PROFILE` | Optional | Set to `1` to enable Holoscan profiling |
| `DISPLAY` | Viewer | X11 display for live_compare_viewer |
