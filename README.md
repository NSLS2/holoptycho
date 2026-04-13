# Holoptycho

Real-time streaming ptychographic reconstruction using the [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk) framework. Developed for the HXN beamline at NSLS-II.

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
- **Smoke tests** -- verify all modules import cleanly
- **Unit tests** -- `liverecon_utils` config parsing
- **GPU tests** -- `StreamingPtychoRecon` buffer allocation, probe initialization, scan reset, save/load

GPU tests require a CUDA-capable GPU and are automatically skipped if cupy is not available.

## Container deployment

The `podman_dir/` directory contains the Dockerfile and container-specific pixi environment for production deployment. See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions on building and running the container.

> **Note:** The container scripts (`build_container`, `run_container`) and `DEPLOYMENT.md` still reference the old `ptycho_gui` layout and will be updated in a follow-up PR.

## Architecture

Holoptycho is a Holoscan pipeline with these key operators:

- **EigerZmqRxOp / PositionRxOp** -- receive diffraction data and motor positions via ZMQ
- **ImagePreprocessorOp / PointProcessorOp** -- preprocess frames and compute scan coordinates
- **PtychoRecon** -- streaming DM ptychographic reconstruction via `StreamingPtychoRecon`
- **PtychoViTInferenceOp** -- optional TensorRT-accelerated neural network inference
- **SaveLiveResult / SaveResult** -- live visualization and final output

`StreamingPtychoRecon` (in `holoptycho/streaming_recon.py`) owns all GPU reconstruction state and uses the `ptycho` package purely as a kernel library for GPU dispatch (cupy_util, cupy_collection, numba_collection, prop_class_asm).

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
