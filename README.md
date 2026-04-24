# Holoptycho

Real-time streaming ptychographic reconstruction using [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), developed for the HXN beamline at NSLS-II.

## Scope

**Holoptycho is for real-time streaming reconstruction only.** It consumes live detector data via ZMQ and emits results to a Tiled catalog as the scan runs.

For batch/offline reconstruction of completed scans, use [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) or [`NSLS2/ptychoml`](https://github.com/NSLS2/ptychoml) directly.

---

## Architecture

Holoptycho is a streaming pipeline: it receives diffraction patterns from the Eiger detector and motor positions from the PandA box over two independent ZMQ streams, reconstructs the ptychographic object iteratively on GPU, and writes results to Tiled in real time.

**Pipeline operators:**
- **`EigerZmqRxOp`** — receives diffraction frames from the Eiger detector (encrypted CurveZMQ, bslz4 compressed)
- **`PositionRxOp`** — receives motor positions from the PandA box (plain ZMQ JSON)
- **`ImageBatchOp` / `ImagePreprocessorOp`** — batch and preprocess diffraction frames
- **`PointProcessorOp`** — maps encoder values to scan coordinates
- **`PtychoRecon`** — iterative DM/ML reconstruction on GPU 0
- **`PtychoViTInferenceOp`** — parallel neural network inference on GPU 1

Results are written to Tiled under `hxn/processed/holoptycho/{scan_num}/` (overrideable via `TILED_CATALOG_PATH`), tagged with the `synaps_project` spec.

---

## Required environment variables

| Variable | Description |
|---|---|
| `SERVER_STREAM_SOURCE` | ZMQ endpoint of the Eiger detector, e.g. `tcp://<host>:5555` |
| `PANDA_STREAM_SOURCE` | ZMQ endpoint of the PandA box, e.g. `tcp://<host>:5556` |
| `SERVER_PUBLIC_KEY` | CurveZMQ server (Eiger) public key |
| `CLIENT_PUBLIC_KEY` | CurveZMQ client public key |
| `CLIENT_SECRET_KEY` | CurveZMQ client secret key |
| `TILED_BASE_URL` | URL of the Tiled server |
| `TILED_API_KEY` | Tiled API key (store in Azure Key Vault — see below) |
| `TILED_CATALOG_PATH` | *(optional)* Tiled catalog path (default: `hxn/processed/holoptycho`) |

The pipeline will refuse to start if `SERVER_STREAM_SOURCE` or `PANDA_STREAM_SOURCE` are not set. If `TILED_BASE_URL` or `TILED_API_KEY` are absent, results fall back to `.npy` files under `/data/users/Holoscan/` with a warning.

---

## Container deployment

A Docker image is built and pushed to Azure Container Registry on every merge to main. See [`.github/workflows/build-container.yml`](.github/workflows/build-container.yml).

### 1. Log in to Azure and ACR

```bash
az login

# az acr login normally hands a token to the Docker daemon, but this cluster
# uses rootless podman (no daemon). --expose-token prints the token instead
# so we can pass it directly to podman login.
podman login genesisdemosacr.azurecr.io \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$(az acr login --name genesisdemosacr --expose-token --query accessToken -o tsv)"
```

### 2. Run the container

```bash
docker run --pull=always --gpus all -p 127.0.0.1:8000:8000 --shm-size=32g \
  -e AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)" \
  -e AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)" \
  -e AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)" \
  -e AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv)" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
  -e TILED_BASE_URL="https://tiled.nsls2.bnl.gov" \
  -e TILED_API_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)" \
  -e SERVER_STREAM_SOURCE="tcp://<eiger-host>:5555" \
  -e PANDA_STREAM_SOURCE="tcp://<panda-host>:5556" \
  -e SERVER_PUBLIC_KEY="<eiger-server-public-key>" \
  -e CLIENT_PUBLIC_KEY="<client-public-key>" \
  -e CLIENT_SECRET_KEY="<client-secret-key>" \
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
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install -e client
pixi run -e client hp --help
```

To avoid typing `pixi run -e client` each time, add a shell alias. Use `--manifest-path` so it works from any directory:

```bash
# bash
echo 'alias hp="pixi run --manifest-path ~/code/holoptycho/pixi.toml -e client hp"' >> ~/.bashrc && source ~/.bashrc

# zsh
echo 'alias hp="pixi run --manifest-path ~/code/holoptycho/pixi.toml -e client hp"' >> ~/.zshrc && source ~/.zshrc
```

### Updating the CLI

```bash
cd ~/code/holoptycho && git pull
```

If `pixi.lock` changed, also run:

```bash
pixi install -e client
```

### Starting and stopping

Select a config, then start:

```bash
hp config select <name>
hp start
hp stop
hp restart   # stop + restart with the same config; use after updating config
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
hp config set hxn_scan '{"scan_num": "339015", "x_range": "2.0", "working_directory": "/nsls2/data/..."}'
hp config select hxn_scan
hp start
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

## Testing with the replay script

To test holoptycho end-to-end without a live beamline, use `scripts/replay_from_tiled.py`. This reads a real scan from Tiled and publishes it over ZMQ in the exact Eiger and PandA wire formats.

### Install the replay environment

```bash
pixi install -e replay
```

### Run the replay script

```bash
pixi run -e replay python scripts/replay_from_tiled.py \
    --scan-num 320045 \
    --tiled-url https://tiled.nsls2.bnl.gov \
    --tiled-api-key <key> \
    --eiger-endpoint tcp://0.0.0.0:5555 \
    --panda-endpoint tcp://0.0.0.0:5556 \
    --rate 200
```

Then start holoptycho pointing at the same ports:

```bash
SERVER_STREAM_SOURCE=tcp://localhost:5555 \
PANDA_STREAM_SOURCE=tcp://localhost:5556 \
hp start
```

### Running the replay script from a remote machine

If holoptycho is running on a Slurm node, open an SSH tunnel that forwards both ZMQ ports in addition to the API port:

```bash
ssh -L 8000:localhost:8000 \
    -L 5555:localhost:5555 \
    -L 5556:localhost:5556 \
    <user>@<slurm-login-node>
```

Then run the replay script locally — it will bind to `tcp://0.0.0.0:5555` and `tcp://0.0.0.0:5556` and the tunnel delivers the traffic to the Slurm node. Point holoptycho at `tcp://localhost:5555` and `tcp://localhost:5556`.

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

## Profiling

```bash
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 \
    pixi run start-api
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```

---

## Deprecated (planned for removal)

The following are no longer used and remain in the repo for reference only. They will be removed in a future release:

- **`PtychoSimulApp`**, **`InitSimul`**, **`live_simulation.py`** — simulate mode that replayed H5 files directly, bypassing ZMQ. Use `scripts/replay_from_tiled.py` instead.
- **`InitRecon`**, **`liverecon_utils.py`** — scan header file watcher for detecting new scans from a beamline-written text file. Scan parameters now come from the API config.
- **`--mode simulate`** CLI option — removed; `hp start` always runs the live ZMQ pipeline.
- **`eiger_simulation/`** — bespoke Eiger simulator container. Use `scripts/replay_from_tiled.py` with a plain Python environment instead.
