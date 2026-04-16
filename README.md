# Holoptycho

Real-time streaming ptychographic reconstruction using [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), developed for the HXN beamline at NSLS-II.

## Architecture

Holoptycho is a streaming reconstruction pipeline: it consumes detector data via ZMQ and emits results to the filesystem. [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) and [`NSLS2/ptychoml`](https://github.com/NSLS2/ptychoml) are used as pure computation libraries; holoptycho handles all I/O and orchestration.

**Pipeline operators:**
- **`EigerZmqRxOp` / `PositionRxOp`** — receive diffraction data and motor positions via ZMQ
- **`ImagePreprocessorOp` / `PointProcessorOp`** — preprocess frames and compute scan coordinates
- **`PtychoRecon`** — iterative DM reconstruction
- **`PtychoViTInferenceOp`** — optional neural network inference for fast estimates

## Install

Requires Linux (x86_64), an NVIDIA GPU, and [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install
```

## Run tests

```bash
pixi run test
```

## Container deployment

A Docker image is built and pushed to Azure Container Registry on every merge to main. See [`.github/workflows/build-container.yml`](.github/workflows/build-container.yml).

## API server and CLI (`hp`)

Holoptycho includes a control server and `hp` CLI for starting, stopping, and monitoring the pipeline — for example over an SSH tunnel.

### Starting the holoptycho server

```bash
pixi run start-api
```

Starts a server on `http://127.0.0.1:8000` (localhost only). For remote access, SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<host>
```

### Starting the Holoscan pipeline

```bash
hp start --mode simulate --config /path/to/ptycho_config.txt  # replay from H5
hp start --mode live --config /path/to/ptycho_config.txt      # live ZMQ streams
```

If already running, `hp start` returns an error — run `hp stop` first.

### CLI reference

Connects to `http://localhost:8000` by default. Override with `--url` or `HOLOPTYCHO_URL`.

| Command | Description |
|---|---|
| `hp start --mode <mode> --config <path>` | Start the pipeline |
| `hp stop` | Stop the pipeline |
| `hp restart` | Stop and restart with the same mode and config |
| `hp status` | Pipeline status (`stopped` / `starting` / `running` / `finished` / `error`) |
| `hp logs [--lines N]` | Tail `holoptycho.log` (default 100 lines) |
| `hp model list` | List local and Azure ML models |
| `hp model set <name> --version <ver>` | Select model for next start |
| `hp model status` | Show current model selection status |

### Model selection

`hp model list` shows two sections:
- **Local cache** — `.engine` files in `/models` (default), ready to use immediately
- **Azure ML** — registered models, with a `cached` column showing what's already local

`hp model set` selects the engine for the next `hp start` or `hp restart`. If the engine is already cached locally it's selected immediately; otherwise it's pulled from Azure ML and compiled via `trtexec` first.

To use Azure ML, set:
```bash
export AZURE_SUBSCRIPTION_ID=<id>
export AZURE_RESOURCE_GROUP=<group>
export AZURE_ML_WORKSPACE=<workspace>
```

Override the default model folder with `ENGINE_CACHE_DIR=/path/to/cache`.

---

## Simulating a data stream

Use `hp start --mode simulate` to replay from a pre-recorded HDF5 file without a live detector. The H5 path is `{working_directory}/scan_{scan_num}.h5` (from config). Outputs land at `/data/users/Holoscan/`.

For end-to-end ZMQ testing with the Eiger simulator container:

```bash
docker build ./eiger_simulation -t eiger_sim:test --network host
docker run -d -p 8000:8000 -p 5555:5555 eiger_sim:test
docker exec -it <container_id> python trigger_detector.py -n 10000 -dt 0.001
```

Point `SERVER_STREAM_SOURCE` at `tcp://<host>:5555` and use `hp start --mode live`.

## Profiling

```bash
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 \
    pixi run python -m holoptycho <config_file>
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```
