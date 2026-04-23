# Holoptycho

Real-time streaming ptychographic reconstruction using [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), developed for the HXN beamline at NSLS-II.

## Architecture

Holoptycho is a streaming reconstruction pipeline: it consumes detector data via ZMQ and emits results to the filesystem. [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) and [`NSLS2/ptychoml`](https://github.com/NSLS2/ptychoml) are used as pure computation libraries; holoptycho handles all I/O and orchestration.

**Pipeline operators:**
- **`EigerZmqRxOp` / `PositionRxOp`** — receive diffraction data and motor positions via ZMQ
- **`ImagePreprocessorOp` / `PointProcessorOp`** — preprocess frames and compute scan coordinates
- **`PtychoRecon`** — iterative DM reconstruction
- **`PtychoViTInferenceOp`** — optional neural network inference for fast estimates

---

## Container deployment

A Docker image is built and pushed to Azure Container Registry on every merge to main. See [`.github/workflows/build-container.yml`](.github/workflows/build-container.yml).

### 1. Log in to Azure and ACR

```bash
az login
podman login genesisdemosacr.azurecr.io \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$(az acr login --name genesisdemosacr --expose-token --query accessToken -o tsv)"
```

The ACR token is cached in `~/.config/containers/auth.json` and lasts 3 hours. Re-run the `podman login` step when it expires.

### 2. Run the container

```bash
docker run --gpus all -p 127.0.0.1:8000:8000 --shm-size=32g \
  -e AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)" \
  -e AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)" \
  -e AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)" \
  -e AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv | base64 | tr -d '\n')" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
  genesisdemosacr.azurecr.io/holoptycho:latest
```

The private key is never written to disk. The server binds to `0.0.0.0:8000` inside the container, exposed only on `127.0.0.1:8000` of the host.

> **On Slurm nodes (rootless podman):** run this once per session before the `docker run`:
> ```bash
> export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u)
> mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
> ```

### 3. Connect via SSH tunnel

The API server binds to `127.0.0.1:8000` (localhost only). For remote access, open an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<host>
```

---

## Controlling the pipeline

Use the `hp` CLI to start, stop, and configure the pipeline. It connects to `http://localhost:8000` by default — override with `--url` or `HOLOPTYCHO_URL`.

### Installing the CLI

The `client` pixi environment installs only the CLI and its dependencies — no GPU or Holoscan deps. It works on Linux and macOS:

```bash
pixi install -e client
pixi run -e client hp --help
```

### Starting and stopping

Select a config, then start:

```bash
hp config select <name>
hp start --mode simulate   # replay from H5
hp start --mode live       # live ZMQ streams
hp stop
hp restart
hp status
hp logs
```

### Config management

Configs are stored on the server as JSON key/value pairs, serialised to INI files at pipeline start. The selected config persists across server restarts.

```bash
hp config list                          # list configs, show which is selected
hp config set <name> '<json>'           # create or overwrite a config
hp config show <name>                   # print config as JSON
hp config select <name>                 # select config for next run
hp config rename <old_name> <new_name>  # rename a config
hp config delete <name>                 # delete a config
```

Example:

```bash
hp config set hxn_sim '{"scan_num": "339015", "x_range": "2.0", "working_directory": "/nsls2/data/..."}'
hp config select hxn_sim
hp start --mode simulate
```

### Model selection

`hp model list` shows two sections:
- **Local cache** — `.engine` files in `ENGINE_CACHE_DIR` (default `/models`), ready to use immediately
- **Azure ML** — registered models, with a `cached` column showing what's already local

`hp model set` selects the engine for the next `hp start` or `hp restart`. If the engine is not cached locally it is pulled from Azure ML and compiled via `trtexec` first.

```bash
hp model list
hp model set <model-name> --version <version>
hp model status
```

---

## Local development

Requires Linux (x86_64), an NVIDIA GPU, and [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install
pixi run test
pixi run start-api   # starts the API server locally on port 8000
```

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

---

## Profiling

```bash
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 \
    pixi run start-api
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```
