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

```bash
hp start                  # start using current config
hp start '<json>'         # start with a new config (becomes current config)
hp stop
hp restart                # stop + restart with current config
hp restart '<json>'       # stop + restart with a new config
hp config show            # print the current config as JSON
hp status
hp logs
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

## Config parameters

The config is a flat JSON dict passed to `hp start` or `hp restart`. All values are strings (matching the INI format the reconstructor reads). See `AGENTS.md` for a full example.

### Parameter reference

| Parameter | Type | Description |
|---|---|---|
| `scan_num` | int (str) | Scan number — tags all Tiled output for this run |
| `working_directory` | path | Root directory for input/output data |
| `shm_name` | str | Shared-memory segment name for ZMQ live data |
| `scan_type` | str | Scan pattern, e.g. `pt_fly2dcontpd` |
| `nx`, `ny` | int (str) | Reconstruction array size (pixels) |
| `batch_width`, `batch_height` | int (str) | Diffraction pattern tile size |
| `batch_x0`, `batch_y0` | int (str) | Top-left crop offset in the detector frame |
| `det_roix0`, `det_roiy0` | int (str) | Detector ROI origin (pixels) |
| `gpu_batch_size` | int (str) | Number of patterns per GPU batch |
| `xray_energy_kev` | float (str) | X-ray energy in keV |
| `lambda_nm` | float (str) | X-ray wavelength in nm — derive from energy (see below) |
| `ccd_pixel_um` | float (str) | Detector pixel size in µm |
| `distance` | float (str) | Sample-to-detector distance in mm |
| `dr_x`, `dr_y` | float (str) | Scan step size in µm |
| `x_num`, `y_num` | int (str) | Number of scan positions (fast/slow axis) |
| `x_range`, `y_range` | float (str) | Total scan range in µm |
| `x_direction`, `y_direction` | float (str) | Sign convention for scan axes (`1.0` or `-1.0`) |
| `x_ratio`, `y_ratio` | float (str) | Encoder-to-µm scale factor for each axis |
| `pos_x_channel`, `pos_y_channel` | str | ZMQ field names for X/Y encoder values from PandA |
| `alg_flag` | str | Primary algorithm: `ML_grad`, `DM`, `ePIE`, etc. |
| `alg2_flag` | str | Secondary algorithm (used after `alg_percentage` of iterations) |
| `alg_percentage` | float (str) | Fraction of iterations using `alg_flag` |
| `n_iterations` | int (str) | Total reconstruction iterations |
| `ml_mode` | str | Noise model: `Poisson` or `Gaussian` |
| `ml_weight` | float (str) | ML regularisation weight |
| `beta` | float (str) | Momentum parameter for ML gradient |
| `init_obj_flag` | bool (str) | Initialise object from DPC (`True`/`False`) |
| `init_prb_flag` | bool (str) | Load probe from file (`True`/`False`) |
| `prb_path` | path | Full path to probe `.npy` file — empty to generate synthetically |
| `prb_mode_num` | int (str) | Number of probe modes |
| `obj_mode_num` | int (str) | Number of object modes |
| `gpu_flag` | bool (str) | Use GPU (`True`/`False`) |
| `gpus` | list (str) | JSON list of GPU indices, e.g. `"[0]"` |
| `precision` | str | Float precision: `single` or `double` |
| `nth` | int (str) | Number of threads for CPU operations |
| `sign` | str | Arbitrary run label used to tag output |
| `display_interval` | int (str) | Iterations between live Tiled updates |

**Wavelength from energy:**

```python
lambda_nm = (6.62607e-34 * 2.99792e8) / (energy_kev * 1e3 * 1.60218e-19) * 1e9
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
