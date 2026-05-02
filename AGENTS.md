# holoptycho Agent Skill

This document teaches an AI agent how to operate the `hp` CLI to control the
holoptycho Holoscan pipeline: start/stop runs, manage configs, and manage
TensorRT engine models.

## Scope

**Holoptycho is for real-time streaming reconstruction only.** It always
connects to live ZMQ streams from the Eiger detector and PandA box.
For batch/offline reconstruction use `NSLS2/ptycho` or `NSLS2/ptychoml`.

## Self-improvement protocol

This file is a living document. Whenever you (the agent) discover any of the
following, **update this file before finishing the task**:

- A CLI flag, command, or behaviour that is missing or wrong in this document
- A config parameter that is undocumented, mis-described, or has an incorrect
  type/default
- A workflow step that failed and required a workaround
- An error message and its resolution that would save future agents time
- Any environment variable, path convention, or server behaviour not yet
  recorded here

**How to update:**

1. Make the edit to `AGENTS.md` using whatever file-editing tool is available.
2. Do not remove existing content unless it is factually wrong — prefer
   appending or correcting in place.

Treat every task as an opportunity to leave this document better than you
found it.

## Prerequisites

- The holoptycho API server must already be running on the target machine.
  It binds to `127.0.0.1:8000` and is reached via SSH tunnel.
- The `hp` CLI is installed as a pyproject entry point (`pixi run hp …` or
  just `hp …` if the venv is active).
- By default all commands talk to `http://localhost:8000`.
  Override with `--url <URL>` or `HOLOPTYCHO_URL=<URL>`.
- `SERVER_STREAM_SOURCE` and `PANDA_STREAM_SOURCE` **must** be set in the
  container environment before `hp start` will succeed.

---

## Setting up the hp CLI

If the user doesn't have `hp` working locally, walk them through:

### 1. Install pixi

If not already installed:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Clone the repo and install the client environment

```bash
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install -e client
```

### 3. Run hp

```bash
pixi run -e client hp --help
```

### 4. (Optional) Add a shell alias

To avoid typing `pixi run -e client` each time, add an alias to the user's shell config. Ask the user which shell they use, then:

**bash** (`~/.bashrc`):
```bash
echo 'alias hp="pixi run -e client hp"' >> ~/.bashrc
source ~/.bashrc
```

**zsh** (`~/.zshrc`):
```bash
echo 'alias hp="pixi run -e client hp"' >> ~/.zshrc
source ~/.zshrc
```

The alias assumes the user runs `hp` from the `holoptycho` repo directory, since pixi needs the `pixi.toml` to resolve the environment. If they want to run it from anywhere, use an absolute path:

```bash
echo 'alias hp="pixi run --manifest-path ~/code/holoptycho/pixi.toml -e client hp"' >> ~/.zshrc
```

### 5. Updating the CLI

Since the package is an editable install, a `git pull` is all that's needed to pick up new versions:

```bash
cd ~/code/holoptycho
git pull
```

If `pixi.toml` or `pixi.lock` changed (i.e. new dependencies were added), also run:

```bash
pixi install -e client
```

To check: `git diff HEAD@{1} pixi.lock` — if it has changes, re-run `pixi install -e client`.

---

## Starting the server on a Slurm node

If the server is not already running, walk the user through the following steps. Ask for the Slurm login node hostname if you don't have it.

### 1. Allocate a GPU node

Ask the user to run on the Slurm login node:

```bash
salloc --gpus=1
```

Ask them to note the allocated node name from the output.

### 2. Set up podman runtime

Ask the user to run once per session on the allocated node:

```bash
export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u)
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
```

### 3. Log in to Azure and ACR

```bash
az login
podman login genesisdemosacr.azurecr.io \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$(az acr login --name genesisdemosacr --expose-token --query accessToken -o tsv)"
```

### 4. Start the container

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
  -e HOLOPTYCHO_LOG_LEVEL="DEBUG" \
  -e SERVER_STREAM_SOURCE="tcp://<eiger-host>:5555" \
  -e PANDA_STREAM_SOURCE="tcp://<panda-host>:5556" \
  -e SERVER_PUBLIC_KEY="<eiger-server-public-key>" \
  -e CLIENT_PUBLIC_KEY="<client-public-key>" \
  -e CLIENT_SECRET_KEY="<client-secret-key>" \
  genesisdemosacr.azurecr.io/holoptycho:latest
```

### 5. Open SSH tunnel

Ask the user to run on their local machine:

```bash
ssh -L 8000:localhost:8000 <slurm-login-node>
```

The `hp` CLI can now reach the server at `http://localhost:8000`.

For testing with the replay script, also forward the ZMQ ports:

```bash
ssh -L 8000:localhost:8000 \
    -L 5555:localhost:5555 \
    -L 5556:localhost:5556 \
    <slurm-login-node>
```

---

## Starting the server natively (without container)

For local development on a host with an NVIDIA GPU, the API server can be run
directly from the pixi env — no podman/docker required. This is the path used
when iterating on the pipeline code or testing against `scripts/replay_from_tiled.py`
on the same machine.

### 1. Build the default pixi env

Requires the system CUDA toolkit (`cuda.h` under `/usr/local/cuda/include`)
and the NVIDIA driver lib (`libcuda.so`). On WSL2 the driver lib is at
`/usr/lib/wsl/lib/`; on bare-metal Linux it is under `/usr/lib/x86_64-linux-gnu/`.

Conda-forge ships `libcurand.so.10` without the unversioned dev symlink that
the linker requires. Create it once:

```bash
ln -sf libcurand.so.10 .pixi/envs/default/lib/libcurand.so
```

Then build the env with the toolchain pointed at both the system CUDA headers
and the driver lib path:

```bash
CUDA_ROOT=/usr/local/cuda CUDA_HOME=/usr/local/cuda CPATH=/usr/local/cuda/include \
  LIBRARY_PATH=/usr/lib/wsl/lib:$PWD/.pixi/envs/default/lib \
  pixi install
```

Drop `/usr/lib/wsl/lib` from `LIBRARY_PATH` on non-WSL hosts.

### 2. Resolve Azure + Tiled credentials

The API server reads the same env vars as the container. Pull them once per
shell with `az` (after running `az login`):

```bash
export AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
export AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)"
export AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
export AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv)"
export AZURE_RESOURCE_GROUP=rg-genesis-demos
export AZURE_ML_WORKSPACE=genesis-mlw

export TILED_BASE_URL="https://tiled.nsls2.bnl.gov"
export TILED_API_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)"
```

### 3. Set ZMQ sources and engine cache

```bash
export SERVER_STREAM_SOURCE="tcp://localhost:5555"
export PANDA_STREAM_SOURCE="tcp://localhost:5556"

# /models (the container default) is not writable outside the container.
export ENGINE_CACHE_DIR="$HOME/.cache/holoptycho/models"
mkdir -p "$ENGINE_CACHE_DIR"
```

### 4. Start the server

```bash
pixi run start-api   # listens on 127.0.0.1:8000
```

Server reads env vars at startup, so changing any of them requires a restart.
Verify with `hp status` once it is up; the `hp` CLI from the `client` env still
talks to `http://localhost:8000` as normal.

---

## CLI reference

### Pipeline lifecycle

```bash
# Show current status (state, last config summary, current model)
hp status

# Start the pipeline (always live ZMQ — no mode parameter)
# Pass a JSON config string to use for this run; uses current config if omitted.
hp start
hp start '<json>'

# Stop the pipeline
hp stop

# Restart with the same config (use after a scan completes)
# Optionally pass a new config JSON string.
hp restart
hp restart '<json>'

# Print the current config as JSON
hp config show

# Tail the log
hp logs
hp logs --lines 50
```

### Model management

Models are TensorRT `.engine` files stored locally or pulled from Azure ML.

```bash
# List local cache and (optionally) Azure ML models
hp model list

# Show current model status
hp model status

# Select a model (downloads + compiles from Azure ML if not cached locally)
# The new engine takes effect on the NEXT pipeline start, not the current run.
# --version is optional; omitting it selects the latest version from Azure ML.
hp model set <azure-model-name>
hp model set <azure-model-name> --version <version>
```

---

## Config file structure

Configs are stored as **flat JSON dicts** (no nesting).  Every key maps
directly to a parameter in the ptycho reconstructor.  When the pipeline
starts, the JSON is serialised to an INI file with a single `[GUI]` section.

### Minimal example

```json
{
  "scan_num": "320045",
  "working_directory": "/ptycho_gui_holoscan",
  "shm_name": "ptycho_320045",
  "scan_type": "pt_fly2dcontpd",

  "nx": "128",
  "ny": "128",
  "batch_width": "128",
  "batch_height": "128",
  "batch_x0": "0",
  "batch_y0": "0",
  "gpu_batch_size": "256",

  "xray_energy_kev": "15.093",
  "lambda_nm": "0.08216037112357172",
  "ccd_pixel_um": "75.0",
  "distance": "30.0",
  "dr_x": "0.02",
  "dr_y": "0.02",
  "x_arr_size": "303.0",
  "y_arr_size": "336.0",
  "x_range": "2.0",
  "y_range": "2.0",
  "x_direction": "1.0",
  "y_direction": "-1.0",
  "z_m": "1.0",

  "alg_flag": "ML_grad",
  "alg2_flag": "ML_grad",
  "alg_percentage": "0.3",
  "n_iterations": "500",
  "ml_mode": "Poisson",
  "ml_weight": "5.0",
  "beta": "0.9",

  "init_obj_flag": "True",
  "init_prb_flag": "True",
  "prb_dir": "",
  "prb_filename": "",
  "prb_path": "",
  "prb_mode_num": "1",
  "obj_mode_num": "1",

  "gpu_flag": "True",
  "gpus": "[0]",
  "precision": "single",
  "nth": "5",

  "sign": "t1",
  "display_interval": "10",
  "save_config_history": "True"
}
```

### Key parameters explained

| Parameter | Type | Description |
|---|---|---|
| `scan_num` | int (str) | Scan number used to tag output in Tiled |
| `working_directory` | path | Root directory for input/output data |
| `shm_name` | str | Shared-memory segment name for ZMQ live data |
| `scan_type` | str | Scan pattern, e.g. `pt_fly2dcontpd` |
| `nx`, `ny` | int (str) | Reconstruction array size (pixels) |
| `batch_width`, `batch_height` | int (str) | Diffraction pattern tile size |
| `batch_x0`, `batch_y0` | int (str) | Top-left crop offset in the detector frame |
| `gpu_batch_size` | int (str) | Number of patterns per GPU batch |
| `recon_mode` | str | Which reconstruction branches to wire: `iterative`, `vit`, or `both`. Default `both`. Use `iterative` to skip the ViT op entirely (no engine load); use `vit` to skip the iterative DM/ML solver (no `live/`/`final/` Tiled writes). |
| `vit_batch_writes` | bool | (Optional) Enable per-batch `pred` + `indices` writes to `<run>/vit/batches/NNNNNN/...` via `BatchWriterOp`. Default `false`. Each batch's `pred` is `(64, 2, 256, 256)` float32 (~33 MB) and a tiled HTTPS PUT runs at ~1 MB/s, so enabling this gates the whole ViT branch at ~28 s/batch. Leave off for live mosaic viewing; turn on only when offline analysts need the raw per-batch arrays. |
| `raw_uid` | str | (Optional) UID of the raw Bluesky run being reconstructed. Stored as metadata on the per-run Tiled container. The `replay_from_tiled.py` and `config_from_tiled.py` config builders fill it in automatically from `--uid`. |
| `scan_id` | str | (Optional) Scan id of the raw run. Defaults to `scan_num` if omitted. Stored as metadata on the per-run Tiled container. |
| `xray_energy_kev` | float (str) | X-ray energy in keV |
| `lambda_nm` | float (str) | X-ray wavelength in nm (derived from energy) |
| `ccd_pixel_um` | float (str) | Detector pixel size in µm |
| `distance` | float (str) | Sample-to-detector distance in mm |
| `dr_x`, `dr_y` | float (str) | Scan step size in µm |
| `x_arr_size`, `y_arr_size` | float (str) | Number of scan positions (fast/slow axis) |
| `x_range`, `y_range` | float (str) | Total scan range in µm |
| `x_direction`, `y_direction` | float (str) | Sign convention for scan axes (`1.0` or `-1.0`) |
| `z_m` | float (str) | Sample z position in m |
| `alg_flag` | str | Primary algorithm: `ML_grad`, `DM`, `ePIE`, etc. |
| `alg2_flag` | str | Secondary algorithm (after `alg_percentage` fraction) |
| `alg_percentage` | float (str) | Fraction of iterations using `alg_flag` |
| `n_iterations` | int (str) | Total number of reconstruction iterations |
| `ml_mode` | str | Noise model: `Poisson` or `Gaussian` |
| `ml_weight` | float (str) | ML regularisation weight |
| `beta` | float (str) | Momentum parameter for ML gradient |
| `init_obj_flag` | bool (str) | Initialise object from DPC (`True`/`False`) |
| `init_prb_flag` | bool (str) | Load probe from file (`True`/`False`) |
| `prb_path` | path | Full path to probe `.npy` file (empty = generate) |
| `gpu_flag` | bool (str) | Use GPU (`True`/`False`) |
| `gpus` | list (str) | JSON list of GPU indices, e.g. `"[0]"` |
| `precision` | str | Float precision: `single` or `double` |
| `sign` | str | Run label / tag (arbitrary string) |
| `display_interval` | int (str) | How often (iterations) to update display |

> **Note**: All values are stored and transmitted as **strings** in the JSON
> dict, matching the INI file format that `configparser` reads.  Pass integers
> and floats as quoted strings: `"nx": "256"`, not `"nx": 256`.

### Wavelength from energy

```python
lambda_nm = (6.62607e-34 * 2.99792e8) / (energy_kev * 1e3 * 1.60218e-19) * 1e9
```

---

## Typical workflow

```bash
# 1. Pull beamline metadata from Tiled and start the pipeline
tiled login https://tiled.nsls2.bnl.gov
hp start "$(pixi run -e client config-from-tiled --scan-num 320045)"

# 2. (Optional) Override reconstruction parameters
hp start "$(pixi run -e client config-from-tiled --scan-num 320045 --nx 256 --n-iterations 1000)"

# 2b. (Optional) Pick which reconstruction branches to run
hp start "$(pixi run -e client config-from-tiled --scan-num 320045 --recon-mode iterative)"  # or vit | both (default)

# 3. (Optional) Switch to a different model
hp model set my_vit_model --version 3

# 4. Watch the log
hp logs --lines 200

# 5. Stop when done
hp stop

# 6. For the next scan: restart with a new config
hp restart "$(pixi run -e client config-from-tiled --scan-num 320046)"
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HOLOPTYCHO_URL` | `http://localhost:8000` | API server URL |
| `HOLOPTYCHO_DB_PATH` | `holoptycho.db` | SQLite DB path (server-side) |
| `HOLOPTYCHO_CONFIG_DIR` | `configs/` | Directory for generated INI files (server-side) |
| `ENGINE_CACHE_DIR` | `/models` | Directory for cached `.engine` files (server-side). Outside the container this default is not writable — point it at a user-writable path before starting the server (e.g. `$HOME/.cache/holoptycho/models`). |
| `SERVER_STREAM_SOURCE` | — | **Required.** ZMQ endpoint of the Eiger detector |
| `PANDA_STREAM_SOURCE` | — | **Required.** ZMQ endpoint of the PandA box |
| `SERVER_PUBLIC_KEY` | — | CurveZMQ server (Eiger) public key |
| `CLIENT_PUBLIC_KEY` | — | CurveZMQ client public key |
| `CLIENT_SECRET_KEY` | — | CurveZMQ client secret key |
| `TILED_BASE_URL` | — | Tiled server URL. If unset, falls back to .npy writes |
| `TILED_API_KEY` | — | Tiled API key (store in Key Vault as `holoptycho-tiled-api-key`) |
| `TILED_CATALOG_PATH` | `hxn/processed/holoptycho` | Tiled catalog path for output |
| `HOLOPTYCHO_LOG_LEVEL` | `INFO` | Root log level for API + pipeline logs. Set to `DEBUG` to surface `TiledWriter.write_live` / `write_vit` debug logs in `hp logs`. |
| `AZURE_SUBSCRIPTION_ID` | — | Azure subscription (for Azure ML model pull) |
| `AZURE_RESOURCE_GROUP` | — | Azure resource group |
| `AZURE_ML_WORKSPACE` | — | Azure ML workspace name |
| `AZURE_CERTIFICATE_B64` | — | Base64-encoded PEM (private key + cert) from Key Vault secret. If set, uses `CertificateCredential`; otherwise falls back to `AzureCliCredential`. |
| `AZURE_TENANT_ID` | — | Entra ID tenant ID. Required when `AZURE_CERTIFICATE_B64` is set. Resolve via `az account show --query tenantId -o tsv`. |
| `AZURE_CLIENT_ID` | — | App registration client ID (not object ID). Required when `AZURE_CERTIFICATE_B64` is set. Resolve via `az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv`. |

### Fetching the certificate for container launch

All values are resolved at runtime via `az cli` — no IDs hardcoded:

```bash
docker run --pull=always --gpus all -p 127.0.0.1:8000:8000 --shm-size=32g \
  -e AZURE_CERTIFICATE_B64="$(az keyvault secret show \
    --vault-name genesisdemoskv \
    --name holoptycho-sp-cert \
    --query value -o tsv)" \
  -e AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)" \
  -e AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)" \
  -e AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
  -e TILED_BASE_URL="https://tiled.nsls2.bnl.gov" \
  -e TILED_API_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)" \
  -e SERVER_STREAM_SOURCE="tcp://<eiger-host>:5555" \
  -e PANDA_STREAM_SOURCE="tcp://<panda-host>:5556" \
  -e SERVER_PUBLIC_KEY="<eiger-server-public-key>" \
  -e CLIENT_PUBLIC_KEY="<client-public-key>" \
  -e CLIENT_SECRET_KEY="<client-secret-key>" \
  <image> <command>
```

The private key is never written to disk — it lives only in the container's environment for the lifetime of the process.

**Note:** Key Vault exports certificates in PKCS12 format (binary), not PEM. `CertificateCredential` requires `password=b""` to deserialize it:

```python
CertificateCredential(
    tenant_id=...,
    client_id=...,
    certificate_data=base64.b64decode(cert_b64),
    password=b"",
)
```

---

## Testing with the replay script

To test end-to-end without a live beamline, use `scripts/replay_from_tiled.py`. The replay script and holoptycho must run on the **same machine** — ZMQ traffic stays local. Run both on the compute node and control holoptycho from your local machine via the `8000` SSH tunnel as normal.

> **`--uid` is a Bluesky UUID4, not a scan number.** Passing the scan
> number (e.g. `404611`) fails with `Run not found` because the catalog
> is keyed by UUID. Always look up the UID from the scan id first (see
> step 2 below).

> **`TILED_BASE_URL` must be set in the shell that runs the replay
> script** (separate from the value the API server reads). Either
> `export TILED_BASE_URL=https://tiled.nsls2.bnl.gov/hxn/migration` or
> pass `--tiled-url` on every invocation; the script aborts otherwise.

#### Setup (once per shell)

```bash
# On the compute node — authenticate and install the replay env
tiled login https://tiled.nsls2.bnl.gov
pixi install -e replay

# If holoptycho has no selected engine yet, choose one before using --hp-start
hp model set nsls0408_bs1
hp model status
```

#### 1. Look up the run UID from a scan id

```bash
pixi run -e replay python - <<'PY'
from tiled.client import from_uri
from tiled.queries import Eq

catalog = from_uri("https://tiled.nsls2.bnl.gov")["hxn"]["raw"]
results = catalog.search(Eq("scan_id", 404611))   # ← scan id
uid = next(iter(results))
print(uid)                                         # ← UUID4 to pass as --uid
PY
```

#### 2. Canonical replay command (with `--hp-start`)

`--hp-start` is the standard pattern: the replay script builds the run
config from the same run metadata and `/run`s or `/restart`s holoptycho
before it begins publishing, so the pipeline always sees the right
`scan_num`, geometry, and pixel size.

```bash
# 256x256 detector ROI, ViT branch only, full HXN scan at 1 kHz publish rate.
# Replace --uid with the UUID from step 1.
pixi run -e replay replay \
    --uid 7fcf8d25-f609-4f2c-8710-44793341455f \
    --tiled-url https://tiled.nsls2.bnl.gov/hxn/migration \
    --hp-start --hp-url http://localhost:8000 \
    --eiger-endpoint tcp://0.0.0.0:5555 \
    --panda-endpoint tcp://0.0.0.0:5556 \
    --nx 256 --ny 256 \
    --rate 1000 --chunk-size 1024 --skip-frames 64 \
    --recon-mode vit --no-compress
```

Tune for your scan:
* `--nx` / `--ny`: must match the detector ROI (256 for HXN scans, 128 for many older scans). See "Best practices for replay" below — these *must* be passed together.
* `--rate`: publish frequency in Hz. 1000 matches the production HXN rate; lower values smoke out throughput regressions; higher values (e.g. 4000) stress-test the pipeline ceiling and surface frame drops.
* `--skip-frames`: drops the first N Eiger frames + matching encoder samples. Required for scans with settling/ramp-up rows.
* `--recon-mode`: `vit` is the fastest path when iterating on `mosaic_stitch.py` / `SaveViTResult`; `iterative` exercises only DM/ML; `both` runs both branches in parallel.

#### Variant: replay without restarting holoptycho

Use this only when holoptycho is already running with a config that exactly matches the scan being replayed (same `nx`/`ny`/geometry). Most of the time you want `--hp-start`.

```bash
pixi run -e replay replay \
    --uid 7fcf8d25-f609-4f2c-8710-44793341455f \
    --tiled-url https://tiled.nsls2.bnl.gov/hxn/migration \
    --eiger-endpoint tcp://0.0.0.0:5555 \
    --panda-endpoint tcp://0.0.0.0:5556 \
    --rate 1000 --no-compress
```

By default the replay script publishes plain ZMQ. To test CurveZMQ, also
pass the full Eiger key set: `--eiger-server-public-key`,
`--eiger-server-secret-key`, and `--eiger-client-public-key`.

`--tiled-url` may be either the Tiled server root
(`https://tiled.nsls2.bnl.gov`) or an exact catalog path such as
`https://tiled.nsls2.bnl.gov/hxn/migration`. The replay/config loaders resolve
both forms and still fall back to `hxn/migration` when given the server root.

If `--tiled-api-key` is provided together with a catalog-path URL, the current
implementation still relies on cached `tiled login` credentials for that path
resolution logic.

By default, leave `SERVER_PUBLIC_KEY`, `CLIENT_PUBLIC_KEY`, and
`CLIENT_SECRET_KEY` unset in the holoptycho container so it subscribes without
CurveZMQ. To test CurveZMQ, set all three in the container and pass the
matching Eiger publisher keys to `scripts/replay_from_tiled.py`. Partial auth
configuration is rejected on both sides.

When `--hp-start` is used, the replay script builds the run config from the
same run metadata and chooses `/run` or `/restart` automatically based on the
current holoptycho server state before publishing. If `hp model status` shows
no selected engine, run `hp model set <model-name>` once first.

On single-GPU nodes, older builds may log repeated `pycuda._driver.LogicError:
cuDeviceGet failed: invalid device ordinal` from `PtychoViTInferenceOp` during
replay because the ViT branch was hard-coded to `gpu=1`. Fixed builds fall back
to GPU 0 when only one configured GPU is available.

If replay publishes successfully but `hp status` reports an error like
`New scan dimensions (...x...) exceed pre-allocated maximum (...x...)`, the
failure is in `holoptycho.streaming_recon.StreamingPtychoRecon.gpu_setup()`.
That limit is about reconstruction object-buffer preallocation for the scan
geometry, not the TensorRT model input size.

The streaming-pipeline frame batch size is fixed at 64 (in
`PtychoApp.compose()`). The ViT op reads its engine's compiled batch dim via
`read_engine_batch_size()` in `_init_session()` and chunks each incoming
pipeline batch into engine-sized sub-batches before running TRT inference, so
small-batch engines (e.g. `nsls0408_bs1`) coexist with the throughput-oriented
streaming batch without backpressuring the pipeline.

If `hp logs` ever shows `ValueError: Batch too large: input X vs engine Y`
from `PtychoViTInferenceOp`, the chunking loop is misbehaving — check that
`engine_batch_size` was set correctly on the op (logged at INFO during
`ptychoml.PtychoViTInference created`).

### Best practices for replay

* **Pass `--nx` and `--ny` together.** `--hp-start` builds a fresh config from
  `config-from-tiled`, which defaults `nx`/`ny` to 128. Detector frames for most
  HXN scans are 256×256, so a 128×anything ROI causes
  `ValueError: could not broadcast input array from shape (256,128) into shape (128,256)`
  in `preprocess.py::ImagePreprocessorOp.compute` at startup. For 256-pixel
  scans always pass `--nx 256 --ny 256` (or whatever matches the actual
  detector ROI).

* **Run only one replay at a time.** Concurrent `--hp-start` replays mid-stream
  the pipeline: the second run's `/restart` interrupts the first while it's
  publishing, leaving PandA frame counters and Eiger frame indices out of
  sync. The pipeline then sits with `positions_um` stuck at NaN and the
  dashboard hangs. Kill any running replay (`pkill -f replay_from_tiled`)
  before launching a new one.

* **Use `--skip-frames` for scans with settling/ramp-up rows.** Some HXN
  scans (e.g. 404611) have the first ~10 rows where encoder readings
  overshoot the commanded scan range by 2–3×. The iterative recon's
  pre-allocated object grid can't fit those positions and crashes. The ViT
  branch tolerates them but stitches them into the wrong canvas region.
  `--skip-frames N` drops the first `N` Eiger frames *and* the corresponding
  upsampled encoder samples so the published stream is consistent.

* **`--chunk-size` controls fetch granularity.** The replay script streams
  frames from tiled in chunks (default 256 frames) instead of loading the
  whole 5 GB scan up front — replay starts publishing within seconds rather
  than waiting for a full transfer. As long as a chunk's fetch (~hundreds
  of ms) is shorter than the publish interval `chunk_size / rate_hz`, the
  stream stays smooth. Drop `--chunk-size` if you see jitter; raise it for
  fewer round-trips.

* **`--max-frames` for quick smoke tests.** Cap the publish to the first N
  frames. Combine with `--n-iterations 50–100` to get a full end-to-end
  cycle (config write → stream → recon → final write) in under a minute.

* **`--recon-mode {iterative,vit,both}` for branch-isolation.** Use
  `iterative` to test the DM/ML solver without TRT competition, `vit` to
  test the ViT branch (and the server-side mosaic stitching) without
  iterative, or `both` to run them in parallel for comparison. `vit`-only
  is the fastest path for verifying changes to `mosaic_stitch.py` /
  `SaveViTResult`.

* **Default to `--no-compress`.** `dectris-compression 0.3.1` removed the
  C `compress` entrypoint, so without this flag the replay script falls
  back to a Python `bitshuffle.compress_lz4` per frame which gates publish
  throughput at ~15 frames/sec — a 50-second scan takes ~12 minutes to
  replay and the pipeline can fall behind by tens of batches. `--no-compress`
  publishes raw frame bytes with a `"raw"` encoding header; the receiver
  in `holoptycho/datasource.py::decode_json_message` recognises this and
  reshapes the bytes directly, skipping decompression. Localhost ZMQ
  handles the ~10× larger wire size easily, so replay runs at the
  requested `--rate`. Live mode is unaffected — the real Eiger detector
  never sets `encoding=raw`.

Then start holoptycho with:

```bash
hp start '{"scan_num": "320045", ...}'
```

The container must be started with `SERVER_STREAM_SOURCE=tcp://localhost:5555` and `PANDA_STREAM_SOURCE=tcp://localhost:5556`.

For same-node testing with the replay script, `localhost` only works if the
container is started with `--network host`. With bridge networking, `localhost`
inside the container refers to the container itself, not the Slurm node host.

---

## Tiled output structure

Each pipeline start (`hp start` / `hp restart`) creates a new container under
`TILED_CATALOG_PATH/{run_uid}/` with a freshly-generated UUID, tagged with the
`synaps_project` spec. The container's metadata records the raw run that was
reconstructed, so multiple runs of the same `scan_num` never collide:

```
hxn/processed/holoptycho/
  {run_uid}/                    ← uuid4 hex; metadata: {run_uid, raw_uid,
                                                         scan_id, scan_num,
                                                         started_at, recon_mode}
    live/
      probe      ← overwritten every display_interval iterations
      object     ← overwritten every display_interval iterations
    final/
      probe      ← written once at scan completion
      object
      timestamps
      num_points
    positions_um   ← (nz, 2) per-frame scan positions in microns,
                     filled in by PointProcessorOp as PandA data arrives
    vit/
      pred_latest    ← overwritten each ViT batch (B, 2, H, W)
      indices_latest ← overwritten each ViT batch (B,)
      mosaic         ← server-side stitched phase mosaic, overwritten each
                       ViT batch (counts-normalised, Fourier-shift placed via
                       holoptycho.mosaic_stitch.place_patches_fourier_shift).
                       Unfilled pixels are filled with the median of the valid
                       region — tiled's PNG renderer treats NaN as 0, which
                       wrecks contrast scaling.
      batches/
        000000/      ← append-only per-batch history
          pred
          indices
        000001/
          pred
          indices
        ...
```

`run_uid` is generated in `PtychoApp.compose()` and surfaced via
`TiledWriter.start_run(run_uid, metadata)`. `raw_uid` and `raw_scan_id` come
from the run config and are populated automatically by
`scripts/config_from_tiled.py::build_full_config()` (and therefore by
`scripts/replay_from_tiled.py --hp-start`).

To list runs for a particular raw scan, query the catalog with
`tiled.queries.Eq("scan_id", "<scan_id>")` (or `Eq("raw_uid", ...)`).

If `TILED_BASE_URL` or `TILED_API_KEY` are not set, the pipeline falls back to
writing `.npy` files under `/data/users/Holoscan/{run_uid}/` (one directory per
run) with a warning. A `metadata.json` sidecar mirrors the Tiled container
metadata for offline matching.

---

## Deprecated (planned for removal)

The following remain in the repo for reference and will be removed in a future release:

- **`PtychoSimulApp`**, **`InitSimul`**, **`live_simulation.py`** — simulate mode that replayed H5 files directly, bypassing ZMQ.
- **`InitRecon`**, **`liverecon_utils.py`** — scan header file watcher.
- **`eiger_simulation/`** — bespoke Eiger simulator container. Use `scripts/replay_from_tiled.py` instead.
