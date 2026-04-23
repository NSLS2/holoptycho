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

### Pulling and running the container

```bash
# Log in to Azure — use device code flow for headless/Slurm environments
az login --use-device-code

# If you have multiple subscriptions, list them and select the right one
az account list --query "[].{name:name, id:id}" -o table
az account set --subscription <AZURE_SUBSCRIPTION_ID>

# Log in to ACR
az acr login --name genesisdemosacr

# Pull and run
docker pull genesisdemosacr.azurecr.io/holoptycho:latest
docker run --gpus all -p 127.0.0.1:8000:8000 \
  --shm-size=32g \
  genesisdemosacr.azurecr.io/holoptycho:latest
```

The container starts the API server by default. Use the `hp` CLI (or any HTTP client) to start, stop, and configure the pipeline from outside the container — typically via an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<host>
hp status
```

## API server and CLI (`hp`)

Holoptycho includes a control server and `hp` CLI for starting, stopping, and monitoring the pipeline — for example over an SSH tunnel.

When running via the container the API starts automatically. Run the container with port 8000 mapped to the host:

```bash
docker run --gpus all -p 127.0.0.1:8000:8000 holoptycho-image
```

For a local (non-container) install:

```bash
pixi run start-api
```

In both cases the server listens on `http://127.0.0.1:8000` (localhost only). For remote access, open an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<host>
```

### Starting the Holoscan pipeline

Before starting the pipeline, select a config:

```bash
hp config select <name>
```

Then start:

```bash
hp start --mode simulate   # replay from H5
hp start --mode live       # live ZMQ streams
```

If already running, `hp start` returns an error — run `hp stop` first.

### CLI reference

Connects to `http://localhost:8000` by default. Override with `--url` or `HOLOPTYCHO_URL`.

```
Usage: hp [OPTIONS] COMMAND [ARGS]...

Options:
  --url TEXT  API base URL [env var: HOLOPTYCHO_URL]

Commands:
  start    Start the pipeline
  stop     Stop the pipeline
  restart  Stop and restart with the same mode
  status   Show pipeline status
  logs     Tail holoptycho.log
  config   Config management commands
  model    Model management commands
```

Use `hp <command> --help` for options on any command, e.g. `hp config --help`.

### Config management

Configs are stored on the server as JSON key/value pairs (saved as INI files internally). The selected config is persisted across server restarts.

```bash
hp config list                              # list configs, show which is selected
hp config set <name> '<json>'               # create or overwrite a config
hp config show <name>                       # print config as JSON
hp config select <name>                     # select config for next run
hp config rename <old_name> <new_name>      # rename a config
hp config delete <name>                     # delete a config
```

Example:

```bash
hp config set hxn_sim '{"scan_num": "339015", "x_range": "2.0", "working_directory": "/nsls2/data/..."}'
hp config select hxn_sim
hp start --mode simulate
```

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

For container deployments, authenticate via a service principal certificate instead of `az login`. All values are resolved at runtime via `az cli` — no IDs hardcoded in the command:

```bash
docker run --gpus all -p 127.0.0.1:8000:8000 --shm-size=32g \
  -e AZURE_CERTIFICATE_B64="$(az keyvault secret show \
    --vault-name genesisdemoskv \
    --name holoptycho-sp-cert \
    --query value -o tsv | base64 -w 0)" \
  -e AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)" \
  -e AZURE_CLIENT_ID="$(az ad app show --display-name 'NSLS2-Genesis-Holoptycho' --query appId -o tsv)" \
  -e AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
  genesisdemosacr.azurecr.io/holoptycho:latest
```

The private key is never written to disk or stored in a shell variable.

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
    pixi run start-api
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```
