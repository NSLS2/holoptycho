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
| [`NSLS2/ptychoml`](https://github.com/NSLS2/ptychoml) | Neural network inference (PtychoViT TRT) | Inference library |
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

## API server and CLI (`hp`)

Holoptycho includes a FastAPI control server and a CLI client (`hp`) for starting, stopping, and monitoring the pipeline remotely — for example over an SSH tunnel.

### Starting the API server

The API server must be running before any CLI commands can be used.

```bash
pixi run start-api
```

This starts a [uvicorn](https://www.uvicorn.org) server on `http://127.0.0.1:8000` (localhost only). To access it from a remote machine, set up an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<host>
```

### Starting the Holoscan application via the API

Once the server is running, use `hp start` to launch the pipeline:

```bash
# Simulate mode (replay from H5 file on disk)
hp start --mode simulate --config /path/to/ptycho_config.txt

# Live mode (ZMQ streams from detector)
hp start --mode live --config /path/to/ptycho_config.txt
```

If the application is already running, `hp start` will return an error. Run `hp stop` first.

### CLI reference

The `hp` command connects to the API server at `http://localhost:8000` by default. Override with `--url` or the `HOLOPTYCHO_URL` environment variable.

```bash
export HOLOPTYCHO_URL=http://localhost:8000   # optional, this is the default
```

| Command | Description |
|---|---|
| `hp start --mode <mode> --config <path>` | Start the Holoscan application |
| `hp stop` | Stop the running application |
| `hp restart` | Stop and restart with the same mode and config |
| `hp status` | Show application status (`stopped` / `starting` / `running` / `finished` / `error`) |
| `hp logs [--lines N]` | Tail the last N lines of `holoptycho.log` (default 100) |
| `hp model set <name> --version <ver>` | Swap the ViT model (pulls ONNX from Azure ML, recompiles, hot-swaps) |
| `hp model status` | Poll the current model swap progress |
| `hp model list` | List available models in Azure ML |

#### Examples

```bash
# Check what's running
hp status

# Stream the last 50 log lines
hp logs --lines 50

# Swap to a new model version (returns immediately; poll hp model status)
hp model set ptycho_vit --version 4
hp model status
```

### API endpoints

The server exposes the following REST endpoints (useful for scripting or building tooling on top):

| Method | Path | Description |
|---|---|---|
| `POST` | `/run` | Start app — body: `{"mode": "simulate"\|"live", "config_path": "..."}` |
| `POST` | `/stop` | Stop the running app |
| `GET` | `/status` | App status, mode, uptime, current model |
| `GET` | `/logs?lines=N` | Tail of `holoptycho.log` |
| `POST` | `/model` | Trigger model swap — body: `{"name": "...", "version": "..."}` |
| `GET` | `/model/status` | Poll model swap progress |
| `GET` | `/model/list` | List available models from Azure ML |

### Model selection

`hp model set` selects the ViT engine to use on the **next** `hp start` or `hp restart`:

1. If the compiled `.engine` is already in the model folder — `state` is updated immediately, no network access needed.
2. If not cached locally — the ONNX is pulled from Azure ML, compiled to TensorRT via `trtexec`, cached, then state is updated.

The selected engine takes effect the next time the pipeline starts.

#### Using a local engine (no Azure ML)

Drop a compiled `.engine` file into the model folder (default `/models`), then:

```bash
hp model list          # shows the file under "Local cache"
hp model set ptycho_vit --version 3   # selects it by name/version
```

#### Pulling from Azure ML

Set these environment variables, then `hp model set` will download and compile automatically if the engine is not already cached:

```bash
export AZURE_SUBSCRIPTION_ID=<subscription-id>
export AZURE_RESOURCE_GROUP=<resource-group>
export AZURE_ML_WORKSPACE=<workspace-name>
```

The compiled `.engine` files are cached in `/models` by default. Override with `ENGINE_CACHE_DIR=/path/to/cache`.

---

## Simulating a data stream

Two simulation modes are available for testing without a live detector.

### In-process H5 replay (simplest)

Reads diffraction patterns + scan positions from a pre-recorded HDF5 file and feeds them into the pipeline, bypassing ZMQ entirely:

```bash
pixi run python -m holoptycho <config_file> simulate
```

The config's `working_directory` + `scan_num` together determine the H5 path: `{working_directory}/scan_{scan_num}.h5`. The file must contain:

| Dataset | Shape | Description |
|---|---|---|
| `diffamp` | `[N, H, W]` | Diffraction amplitudes (or use the `raw_data` subgroup for Eiger-style raw files) |
| `ic` | `[N]` | Intensity normalization vector |
| `points` | `[2, N]` | Scan positions (x, y) in microns |

Outputs land at `/data/users/Holoscan/`:
- `prb_live.npy` / `obj_live.npy` — updated every 10 iterations
- `probe.npy` / `object.npy` — final reconstruction (in a timestamped directory)
- `vit_batch_*_pred.npy` — per-batch ViT predictions

### Eiger simulator container (end-to-end with ZMQ)

For testing the full ZMQ path, including the detector network protocol:

```bash
# Build the simulator container
docker build ./eiger_simulation -t eiger_sim:test --network host

# Run it
docker run -d -p 8000:8000 -p 5555:5555 eiger_sim:test

# Trigger frames
docker exec -it <container_id> python trigger_detector.py -n 10000 -dt 0.001
```

Then point holoptycho's `SERVER_STREAM_SOURCE` env var at `tcp://<host>:5555` and run without the `simulate` argument.

## Profiling

```bash
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 \
    pixi run python -m holoptycho <config_file>
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```
