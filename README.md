# Holoptycho

Real-time streaming ptychographic reconstruction using the [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk) framework. Developed for the HXN beamline at NSLS-II.

## Architecture

**What this repo is**: streaming ptychographic reconstruction pipeline. Consumes detector data via ZMQ, emits reconstruction results to filesystem and/or Tiled.

**What this repo is not**:
- Batch/offline reconstruction → use [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho)
- Model training → use `ptycho-vit` (PyTorch training code maintained by ANL, produces ONNX checkpoints)

### Related repos

| Repo | Role | Used by holoptycho as |
|---|---|---|
| [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) | Iterative reconstruction algorithms + GPU kernels | Kernel library |
| `NSLS2/ptychoml` (planned) | Neural network inference (PtychoViT TRT) | Inference library |
| `ptycho-vit` | PyTorch training for PtychoViT models | Not imported — produces ONNX files for `ptychoml` to convert to TensorRT |

**Design principle**: `ptycho` and `ptychoml` are pure computation libraries (no I/O). Holoptycho handles all I/O — ZMQ streams, filesystem writes, Tiled publishing — and pipeline orchestration.

### Pipeline operators

- **`EigerZmqRxOp` / `PositionRxOp`** — receive diffraction data and motor positions via ZMQ
- **`ImagePreprocessorOp` / `PointProcessorOp`** — preprocess frames and compute scan coordinates
- **`PtychoRecon`** — iterative DM reconstruction via `StreamingPtychoRecon` (uses `ptycho` kernels)
- **`PtychoViTInferenceOp`** — optional neural network inference for fast estimates
- **Output sinks** — filesystem today; Tiled and ZMQ-publish planned

## Prerequisites

- Linux (x86_64)
- NVIDIA GPU with CUDA support
- [pixi](https://pixi.sh) package manager

## Install

```bash
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install
```

This creates a conda environment with all dependencies (CUDA, cupy, holoscan, ptycho, etc.).

## Run tests

```bash
pixi run test
```

Tests include:
- **Smoke tests** — verify all modules import cleanly
- **Unit tests** — `liverecon_utils` config parsing
- **GPU tests** — `StreamingPtychoRecon` buffer allocation, probe initialization, scan reset, save/load

GPU tests require a CUDA-capable GPU and are automatically skipped if cupy is not available.

## Container deployment

A Docker image is built and pushed to Azure Container Registry on every merge to main. See [`.github/workflows/build-container.yml`](.github/workflows/build-container.yml) and the root [`Dockerfile`](Dockerfile).

## Simulating a data stream

To test without a live detector, use the simulated Eiger data stream:

```bash
# Build the simulator container
docker build ./eiger_simulation -t eiger_sim:test --network host

# Run it
docker run -d -p 8000:8000 -p 5555:5555 eiger_sim:test

# Trigger frames
docker exec -it <container_id> python trigger_detector.py -n 10000 -dt 0.001
```

## Profiling

```bash
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 \
    pixi run python -m holoptycho <config_file>
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```
