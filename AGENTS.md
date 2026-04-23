# holoptycho Agent Skill

This document teaches an AI agent how to operate the `hp` CLI to control the
holoptycho Holoscan pipeline: start/stop runs, manage configs, and manage
TensorRT engine models.

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

Or to avoid typing `pixi run -e client` each time, activate the environment:

```bash
pixi shell -e client
hp --help
```

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
  -e AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv | base64 | tr -d '\n')" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
  genesisdemosacr.azurecr.io/holoptycho:latest
```

### 5. Open SSH tunnel

Ask the user to run on their local machine:

```bash
ssh -L 8000:localhost:8000 <slurm-login-node>
```

The `hp` CLI can now reach the server at `http://localhost:8000`.

---

## CLI reference

### Pipeline lifecycle

```bash
# Show current status (state, selected config, current model)
hp status

# Start the pipeline  (mode = "live" or "simulate")
hp start --mode simulate
hp start --mode live

# Stop the pipeline
hp stop

# Restart with the same mode
hp restart

# Tail the log
hp logs
hp logs --lines 50
```

### Config management

Configs are stored in SQLite as flat JSON dicts.  They are converted to INI
format (a `[GUI]` section) and written to disk when the pipeline starts.

```bash
# List all configs; the selected one is marked
hp config list

# Print a config as JSON
hp config show <name>

# Create or overwrite a config from a JSON string
hp config set <name> '<json>'

# Select a config for the next run (persists across server restarts)
hp config select <name>

# Rename a config
hp config rename <old_name> <new_name>

# Delete a config
hp config delete <name>
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
hp model set <azure-model-name> --version <version>
```

---

## Config file structure

Configs are stored as **flat JSON dicts** (no nesting).  Every key maps
directly to a parameter in the ptycho reconstructor.  When the pipeline
starts, the JSON is serialised to an INI file with a single `[GUI]` section.

### Minimal example (simulate mode)

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

  "simulate_live_recon": "True",
  "sign": "t1",
  "display_interval": "10",
  "save_config_history": "True"
}
```

### Key parameters explained

| Parameter | Type | Description |
|---|---|---|
| `scan_num` | int (str) | Scan number used to locate raw data |
| `working_directory` | path | Root directory for input/output data |
| `shm_name` | str | Shared-memory segment name for ZMQ live data |
| `scan_type` | str | Scan pattern, e.g. `pt_fly2dcontpd` |
| `nx`, `ny` | int (str) | Reconstruction array size (pixels) |
| `batch_width`, `batch_height` | int (str) | Diffraction pattern tile size |
| `batch_x0`, `batch_y0` | int (str) | Top-left crop offset in the detector frame |
| `gpu_batch_size` | int (str) | Number of patterns per GPU batch |
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
| `simulate_live_recon` | bool (str) | Replay H5 file instead of live ZMQ stream |
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
# 1. Create a config for a new scan
hp config set scan320045 '{
  "scan_num": "320045",
  "working_directory": "/ptycho_gui_holoscan",
  "shm_name": "ptycho_320045",
  "nx": "128", "ny": "128",
  "batch_width": "128", "batch_height": "128",
  "gpu_batch_size": "256",
  "xray_energy_kev": "15.093",
  "lambda_nm": "0.08216037112357172",
  "ccd_pixel_um": "75.0",
  "distance": "30.0",
  "dr_x": "0.02", "dr_y": "0.02",
  "x_arr_size": "303.0", "y_arr_size": "336.0",
  "x_range": "2.0", "y_range": "2.0",
  "x_direction": "1.0", "y_direction": "-1.0",
  "alg_flag": "ML_grad", "n_iterations": "500",
  "gpu_flag": "True", "gpus": "[0]",
  "simulate_live_recon": "True", "sign": "t1"
}'

# 2. Select it
hp config select scan320045

# 3. (Optional) Switch to a different model
hp model set my_vit_model --version 3

# 4. Start a simulation run
hp start --mode simulate

# 5. Watch the log
hp logs --lines 200

# 6. Stop when done
hp stop
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HOLOPTYCHO_URL` | `http://localhost:8000` | API server URL |
| `HOLOPTYCHO_DB_PATH` | `holoptycho.db` | SQLite DB path (server-side) |
| `HOLOPTYCHO_CONFIG_DIR` | `configs/` | Directory for generated INI files (server-side) |
| `ENGINE_CACHE_DIR` | `models/` | Directory for cached `.engine` files (server-side) |
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
    --query value -o tsv | base64 | tr -d '\n')" \
  -e AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)" \
  -e AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)" \
  -e AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
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

