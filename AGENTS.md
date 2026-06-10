# holoptycho Agent Skill

This document teaches an AI agent how to operate the `hp` CLI to control the
holoptycho Holoscan pipeline: start/stop runs, manage configs, and manage
TensorRT engine models.

## Scope

**Holoptycho is for real-time streaming reconstruction only.** It always
connects to live ZMQ streams from the Eiger detector and PandA box.
For batch/offline reconstruction use `NSLS2/ptycho` or `NSLS2/ptychoml`.

Generic, array-in/array-out ptychography code lives in `NSLS2/ptychoml` and
is imported rather than duplicated here: mosaic patch placement
(`ptychoml.stitch`, e.g. `stitch_batch_into`) and the far-field sample
pixel size (`ptychoml.compute_sample_pixel_size`). Keep that boundary —
new generic numpy helpers belong in ptychoml; holoptycho holds the
Holoscan operators and pipeline wiring.

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

The alias assumes the user runs `hp` from the `holoptycho` repo directory, since pixi needs the `pixi.toml` to resolve the environment. If they want to run it from anywhere, use `$PWD` while inside the repo so the path is captured correctly:

```bash
echo "alias hp=\"pixi run --manifest-path $PWD/pixi.toml -e client hp\"" >> ~/.zshrc
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

Use `start.sh` rather than a raw `docker run` — it handles ACR login, secret
fetching, and ZMQ source selection in one step:

```bash
# Live beamline (Eiger + PandA on xf03idc-eiger2-ioc.nsls2.bnl.local)
./start.sh --live

# Replay / testing (listens on localhost:5555/5556 for replay_from_tiled.py)
./start.sh

# Run detached and use shared Tiled API key from Key Vault
./start.sh --live -d --api-key
```

`--live` fetches the Eiger CurveZMQ server public key from Key Vault secret
`holoptycho-eiger-server-public-key` and sets:
- `SERVER_STREAM_SOURCE=tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:5559`
- `PANDA_STREAM_SOURCE=tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:6666`

Without `--live` those are set to `tcp://host.docker.internal:5555/5556` for
the replay script.

If you need to run the container manually (e.g. with extra debug flags):

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
  -e SERVER_STREAM_SOURCE="tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:5559" \
  -e PANDA_STREAM_SOURCE="tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:6666" \
  -e SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-eiger-server-public-key --query value -o tsv)" \
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
pixi run api   # listens on 127.0.0.1:8000
```

Server reads env vars at startup, so changing any of them requires a restart.
Verify with `hp status` once it is up; the `hp` CLI from the `client` env still
talks to `http://localhost:8000` as normal.

---

## Verifying ZMQ connectivity

Before starting a live run, use `scripts/check_zmq.py` to confirm both streams
are reachable and the CurveZMQ key is accepted. It uses the `replay` pixi env
(no GPU or Holoscan required).

```bash
pixi install -e replay  # once

# Run the check with all three keys (recommended — tests the exact keys the container will use)
SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-eiger-server-public-key --query value -o tsv)" \
CLIENT_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-client-public-key --query value -o tsv)" \
CLIENT_SECRET_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-client-secret-key --query value -o tsv)" \
pixi run -e replay python scripts/check_zmq.py

# Longer timeout during an active scan to confirm data flows
SERVER_PUBLIC_KEY="..." pixi run -e replay python scripts/check_zmq.py --timeout 30
```

Outcomes per stream:

| Result | Meaning |
|---|---|
| `OK` | Connected and received data — detector armed / scan running |
| `TIMEOUT` | TCP reachable, no data — no scan running (expected when idle) or CurveZMQ handshake failed |
| `ERROR` | TCP connection failed — host unreachable or port closed |

The script now does a TCP pre-check before attempting ZMQ. `ERROR` means the
port is not reachable at all (firewall or wrong host). `TIMEOUT` with a scan
running likely means a CurveZMQ key mismatch — the handshake fails silently
and the SUB socket receives nothing.

`TIMEOUT` with no scan running is the expected result when confirming a new
deployment. `ERROR` means something is actually wrong.

`CLIENT_PUBLIC_KEY` / `CLIENT_SECRET_KEY` are optional for this script — it
generates a throwaway keypair automatically when only `SERVER_PUBLIC_KEY` is
set. The production pipeline (`EigerZmqRxOp` in `datasource.py`) requires all
three keys and will raise `RuntimeError` if only some are set.

### CurveZMQ key format

Keys must be **Z85-encoded 40-character ASCII strings** as produced by
`zmq.curve_keypair()`. If a key stored in Key Vault contains non-ASCII
characters, it was stored in the wrong format and will cause
`UnicodeEncodeError: 'ascii' codec can't encode characters` in `datasource.py`.

To generate a valid keypair and store it:

```bash
pixi run -e client python3 -c '
import zmq
pub, sec = zmq.curve_keypair()
print("PUBLIC:", pub.decode("ascii"))
print("SECRET:", sec.decode("ascii"))
'
az keyvault secret set --vault-name genesisdemoskv --name holoptycho-client-public-key --value "<pub>"
az keyvault secret set --vault-name genesisdemoskv --name holoptycho-client-secret-key --value "<sec>"
```

Note: use **single quotes** around the Python snippet in bash — double quotes
cause the shell to interpret the inner quotes and produce a `SyntaxError`.

The Eiger server does not allowlist client keys — it uses them only for
encryption — so regenerated client keys work without any beamline-side changes.

**After storing new keys in KV, the container must be restarted** — keys are
fetched from KV at `start.sh` time and baked into the container environment.
`hp restart` alone is not enough; you must run `./start.sh --live` again (and
then `hp model set` + `hp start` since the engine cache is lost on restart).

### Diagnosing CurveZMQ TIMEOUT

If `check_zmq.py` returns `TIMEOUT` with a scan running (TCP reachable, PandA
OK), use the ephemeral keypair to narrow down which key is wrong:

```bash
# Step 1: try with only SERVER_PUBLIC_KEY (ephemeral client pair)
SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-eiger-server-public-key --query value -o tsv)" \
pixi run -e replay python scripts/check_zmq.py --timeout 30
```

| Step 1 result | Meaning | Fix |
|---|---|---|
| `OK` | SERVER_PUBLIC_KEY correct, client keys wrong | Regenerate client keypair (see above) |
| `TIMEOUT` | SERVER_PUBLIC_KEY wrong | Contact beamline team for current Eiger server key |

### Eiger nftables firewall

The Eiger IOC machine (`xf03idc-eiger2-ioc.nsls2.bnl.local`) runs nftables
with a `policy drop` chain (`inet firewall inbound`). Connections from compute
nodes outside the beamline subnet (e.g. mars5 at `10.65.3.205`) are blocked by
default.

To allow a machine to connect to the ZMQ ports, add a rule **before** the
catch-all drop rules. The correct approach:

```bash
# 1. Find the handle of the first catch-all drop (look for "Connection Denied")
nft -a list chain inet firewall inbound | grep "Connection Denied"

# 2. Find the handle of the existing allowed-source rules (e.g. 10.65.15.35)
#    and insert at the same position
nft -a list chain inet firewall inbound | grep "10.65.15.35"

# 3. Insert BEFORE those rules (use the handle of the first existing accept rule
#    for ports 5559/6666 as the position anchor)
nft insert rule inet firewall inbound position <handle> ip saddr <mars5-ip> tcp dport { 5559, 6666 } accept
```

**Important:** `nft add rule` appends to the end of the chain — after the
catch-all drop — so the rule is never evaluated. Always use
`nft insert rule ... position <handle>` to place it before the drop rules.
`nft insert rule position N` inserts **before** handle N.

Verify the rule landed in the right place:
```bash
nft -a list chain inet firewall inbound | grep -A2 -B2 "<mars5-ip>"
```

The accepted source IPs for ports 5559/6666 as of 2026-05-18:
- `10.65.15.35` — pre-existing (unknown machine)
- `10.65.3.205` — mars5

---

## CLI reference

### Remote management

By default `hp` connects to `http://localhost:8000`. Use `hp remote` to
switch to a different server and persist the choice across sessions:

```bash
# List all remotes (* marks the active one)
hp remote list

# Show current active remote
hp remote status

# Switch to mars5 (persisted to ~/.config/holoptycho/remote)
hp remote set mars5

# Switch back to localhost
hp remote set localhost
```

Named remotes: `localhost` (`http://localhost:8000`), `mars5` (`http://mars5.nsls2.bnl.gov:8000`).

`--url` / `HOLOPTYCHO_URL` always override the persisted remote for a single invocation:
```bash
HOLOPTYCHO_URL=http://mars5.nsls2.bnl.gov:8000 hp status
```

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
| `recon_mode` | str | Which reconstruction branches to wire: `iterative`, `vit`, or `both`. Default `both`. Use `iterative` to skip the ViT op entirely (no engine load); use `vit` to skip the iterative DM/ML solver — the `PtychoRecon`/`StreamingPtychoRecon` engine is **not created at all** (no CuPy context on the ViT GPU, avoiding the PyCUDA+CuPy SIGABRT), and the sample-plane geometry the engine would supply (pixel size, ranges) is derived from config via `ptychoml.compute_sample_pixel_size`. No `live/`/`final/` Tiled writes. |
| `vit_batch_writes` | bool | (Optional) Enable per-batch `pred` + `indices` writes to `<run>/vit/batches/NNNNNN/...` via `BatchWriterOp`. Default `false`. Each batch's `pred` is `(64, 2, 256, 256)` float32 (~33 MB) and a tiled HTTPS PUT runs at ~1 MB/s, so enabling this gates the whole ViT branch at ~28 s/batch. Leave off for live mosaic viewing; turn on only when offline analysts need the raw per-batch arrays. |
| `auto_center_dp` | bool | (Optional) Lossless one-shot DP auto-centering in `ImageBatchOp`. Default `false`. CLI: `--auto-center-dp`. See the `auto_center_dp` note below. |
| `auto_center_headroom` | int | (Optional) Search margin (px/side) for `auto_center_dp`. Default `nx//4`. CLI: `--auto-center-headroom`. |
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
tiled profile create https://tiled.nsls2.bnl.gov --name nsls2  # once
tiled login --profile nsls2
hp start "$(pixi run -e client config-from-tiled --scan-id 320045)"

# 2. (Optional) Override reconstruction parameters
hp start "$(pixi run -e client config-from-tiled --scan-id 320045 --nx 256 --n-iterations 1000)"

# 2b. (Optional) Pick which reconstruction branches to run
hp start "$(pixi run -e client config-from-tiled --scan-id 320045 --mode iterative)"  # or vit | both (default)

# 2c. (Optional) Enable lossless DP auto-centering (off by default)
hp start "$(pixi run -e client config-from-tiled --scan-id 320045 --auto-center-dp)"

# 3. (Optional) Switch to a different model
hp model set my_vit_model --version 3

# 4. Watch the log
hp logs --lines 200

# 5. Stop when done
hp stop

# 6. For the next scan: restart with a new config
hp restart "$(pixi run -e client config-from-tiled --scan-id 320046)"
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HOLOPTYCHO_URL` | `http://localhost:8000` | API server URL |
| `HOLOPTYCHO_DB_PATH` | `holoptycho.db` | SQLite DB path (server-side) |
| `HOLOPTYCHO_CONFIG_DIR` | `configs/` | Directory for generated INI files (server-side) |
| `ENGINE_CACHE_DIR` | `/models` | Directory for cached `.engine` files (server-side). Outside the container this default is not writable — point it at a user-writable path before starting the server (e.g. `$HOME/.cache/holoptycho/models`). |
| `SERVER_STREAM_SOURCE` | — | **Required.** ZMQ endpoint of the Eiger detector. Live beamline: `tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:5559`. For replay use `tcp://host.docker.internal:5555`. Set automatically by `start.sh --live`. |
| `PANDA_STREAM_SOURCE` | — | **Required.** ZMQ endpoint of the PandA box. Live beamline: `tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:6666`. For replay use `tcp://host.docker.internal:5556`. Set automatically by `start.sh --live`. |
| `SERVER_PUBLIC_KEY` | — | CurveZMQ server (Eiger) public key. Fetched from Key Vault secret `holoptycho-eiger-server-public-key` by `start.sh --live`. |
| `CLIENT_PUBLIC_KEY` | — | CurveZMQ client public key. Fetched from Key Vault secret `holoptycho-client-public-key` by `start.sh --live`. |
| `CLIENT_SECRET_KEY` | — | CurveZMQ client secret key. Fetched from Key Vault secret `holoptycho-client-secret-key` by `start.sh --live`. |
| `TILED_BASE_URL` | — | **Required.** Tiled server URL |
| `TILED_API_KEY` | — | Tiled API key (optional — falls back to cached `tiled login` token; store in Key Vault as `holoptycho-tiled-api-key` for production) |
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

> **If `hp model list` / `hp model set` returns `Authentication failed: AADSTS700027`
> (certificate not registered on application)**, the public key in the app
> registration is out of sync with the private key in Key Vault. Fix it by
> re-uploading the public key:
>
> ```bash
> az keyvault certificate download \
>   --vault-name genesisdemoskv \
>   --name holoptycho-sp-cert \
>   --encoding PEM \
>   --file /tmp/holoptycho-sp-cert.pem
>
> az ad app credential reset \
>   --id "$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)" \
>   --cert @/tmp/holoptycho-sp-cert.pem \
>   --append
>
> rm /tmp/holoptycho-sp-cert.pem
> ```
>
> This happens when the cert in Key Vault is rotated/replaced without
> re-registering the new public key on the app registration.

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
  -e SERVER_STREAM_SOURCE="tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:5559" \
  -e PANDA_STREAM_SOURCE="tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:6666" \
  -e SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-eiger-server-public-key --query value -o tsv)" \
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
tiled profile create https://tiled.nsls2.bnl.gov --name nsls2  # once
tiled login --profile nsls2
pixi install -e replay

# If holoptycho has no selected engine yet, choose one before replaying
hp model set run042901
hp model status
```

#### 1. Canonical replay command

By default the replay script `/run`s or `/restart`s holoptycho with a config
built from the same run metadata before publishing, so the pipeline always
sees the right `scan_num`, geometry, and pixel size. Pass `--no-hp-start`
to skip that step when holoptycho is already running with a matching config.

Use `--scan-id <int>` to look up the run automatically (newest match wins
— scan_id is not unique), or pass `--uid <UUID4>` directly if you already
have the UUID.

```bash
# ViT branch only, full HXN scan.
# --tiled-url, --hp-url, --eiger/panda-endpoint, and --nx/--ny default to
# HXN-typical values (256×256 crop, the current engine input size).
pixi run -e replay replay --scan-id 404611 --mode vit
```

Tune for your scan:
* `--nx` / `--ny`: must match the selected engine's input dimensions (default 256, matching current HXN engines). See "Best practices for replay" below — they *must* be passed together if you override.
* `--rate`: informational only — the publisher dumps each chunk as fast as ZMQ drains and ignores this value. Set it for the startup banner if you want; it doesn't gate timing.
* `--skip-frames`: drops the first N Eiger frames + matching encoder samples. Required for scans with settling/ramp-up rows.
* `--mode`: `vit` is the fastest path when iterating on `ptychoml.stitch` / `SaveViTResult`; `iterative` exercises only DM/ML; `both` runs both branches in parallel.

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

* **`--nx`/`--ny` must match the selected engine's input dimensions.**
  These set the detector-frame crop size fed into the pipeline; the default
  256×256 matches current HXN engines (`ptycho_vit_amp_phase_b64`,
  `run042901`). A mismatch raises
  `ValueError: could not broadcast input array from shape (256,128) into shape (128,256)`
  in `preprocess.py::ImagePreprocessorOp.compute` at startup. The detector
  frame can be larger — the pipeline crops down — but it must be at least
  `nx × ny`. If you override, pass both together.

* **Run only one replay at a time.** Concurrent `--hp-start` replays mid-stream
  the pipeline: the second run's `/restart` interrupts the first while it's
  publishing, leaving PandA frame counters and Eiger frame indices out of
  sync. The pipeline then sits with `positions_um` stuck at NaN and the
  dashboard hangs. Kill any running replay (`pkill -f replay_from_tiled`)
  before launching a new one.

* **`panda_upsample` (config field, default 1) — raw encoder samples per
  detector frame.** `PointProcessorOp` averages each group of this many raw
  PandA samples down to one position. Replays of pre-averaged tiled data
  use 1 (no averaging); the replay script auto-detects the ratio from
  `len(encoder_array) // n_frames` and forwards it via the hp config. The
  beamline's prod config is currently set to 10, matching a historical
  assumption that HXN PandA oversamples 10×. **Open question:** the 10×
  story hasn't been verified against current PandA firmware — if the real
  beamline emits 1:1 or some other ratio today, position averaging is
  either redundant or wrong. Worth confirming with the beamline team and
  potentially reducing pipeline complexity.

* **`auto_center_dp` (config field, default `false`) — one-shot **lossless**
  diffraction centering via scipy segmentation.** Toggle with the
  `--auto-center-dp` flag on `config_from_tiled` / `replay`, or set the config
  bool directly. It lives in **`ImageBatchOp`** (`preprocess.py::compute_center_box`):
  on the first batch it buffers a `±auto_center_headroom` window (default
  `nx//4`) around the configured ROI, averages it, masks saturated pixels,
  thresholds at 5% of peak, runs `scipy.ndimage.label`, takes the centroid of
  the largest component, and crops **every** batch (including the first) to an
  `nx × ny` box centered on it — real detector pixels, **no `np.roll`, no
  zero-fill** (the old lossy roll in `ImagePreprocessorOp` was removed). The
  beam's raw column is flip-independent, so segmentation runs on the un-flipped
  window and the box still feeds `crop_flipped_roi` correctly. It **composes**
  with `batch_x0/batch_y0` (which position the search window) and refines within
  `±headroom`. **Default off:** centering moves the beam away from the position
  the model was trained on. Enable only when the detector ROI is too far
  off-centre to fix with `batch_x0`/`batch_y0` alone. If no blob is found on the
  first batch it falls back to the configured ROI (warns); if the beam sits
  beyond the headroom the box is clamped to the window edge (warns).

* **Diffraction geometry + normalization knobs (config fields).** The model
  input branch of `ImagePreprocessorOp` delegates to
  `ptychoml.preprocess_diffraction` (normalize → hot-pixel mask → sqrt →
  D4 → fftshift); the intensity tap is oriented with `ptychoml.apply_d4`.
  All are settable from the scan JSON:
  * `tap_orient` (default `antitranspose`) — D4 element applied to the saved
    intensity tap (`/dp`). The default reproduces the historical HXN
    anti-diagonal flip; use `identity` to save raw detector frames.
  * `dp_orient` (default `rot90_cw`) — D4 element on the model-input branch.
    The default reproduces the prior hardcoded chain (verified equivalent).
    Determine the right value offline with `scripts/detect_orientation.py`
    (reads a recorded scan HDF5 + the model engine, sweeps all 8 D4
    candidates via `ptychoml.autodetect_orientation`, and writes a JSON
    snippet with the winning `dp_orient` to merge into the scan config).
    **Live auto-detect:** if the engine carries a baked probe (exported with
    `--probe`), `PtychoViTInferenceOp` runs the same sweep once on the first
    batch with ≥64 finite scan positions and overrides `dp_orient` in place —
    `ImagePreprocessorOp` buffers pre-D4 frames for it (capped 256), and the
    config `dp_orient` is the pre-sweep default. Engines without a baked probe
    keep the configured value.
  * `fftshift_dp` (default unset → `None`) — DC convention for the model
    input. `None` lets ptychoml auto-detect via `detect_dc_at_corner` and
    shift only when the central beam sits at the corners. Override with
    `true`/`false` only when a specific dataset misbehaves.
  * `vit_normalization` (default `1e5`) — per-scan max intensity (hot pixels
    excluded). In live mode the full DP stack isn't available, so this must
    come from the scan JSON (precomputed offline, or via
    `ptychoml.compute_intensity_normalization` on a representative subset).
    Without the right value the amplitude scale drifts from training.
  * `vit_scale` (default `1e4`) — model-input amplitude scale factor.
  * `hot_pixel_count_threshold` (default unset → disabled) — photon-count
    threshold for hot-pixel zeroing; matches `hxn_to_vit.py` when enabled.
  * `position_swap_xy` (default `false`) — `PointProcessorOp` writes scan
    positions as col 0 = slow axis (INENC3/y_range), col 1 = fast axis
    (INENC2/x_range), matching the `hxn_to_vit` convention and the
    `SaveViTResult` canvas mapping (width ← col 0 ← y_range). Set `true` to
    revert to the legacy (fast→col 0) assignment. Affects only the ViT/mosaic
    path, not the iterative engine's `point_info`.

* **`mosaic_overshoot_factor` (config field, default 1.2) — canvas safety
  margin for the ViT mosaic.** Sized as `max(observed_range, commanded_range
  × overshoot)`. 1.2 means the canvas is 20% bigger than the commanded
  scan extent — fine for scans where encoders stay near commanded. HXN
  scans with settling-row overshoot (e.g. 404611: commanded 2 µm → observed
  6 µm) need a larger value (3.0). Off-canvas frames are dropped with a
  warning, so under-sizing degrades the mosaic but doesn't crash the run.
  `mosaic_overshoot_factor < 1.0` switches to *crop mode* — the canvas is
  intentionally smaller than commanded to trim the low-coverage border.

* **ViT mosaic stitching uses `ptychoml.stitch.stitch_batch_livestitch_into`
  (nearest-integer placement) — `SaveViTResult` + `MosaicWriterOp`.**
  `SaveViTResult` keeps a running `(canvas, counts)` sum; `MosaicWriterOp`
  averages it each batch via `ptychoml.stitch.normalize_mosaic` (covered
  pixels → `canvas / counts`, under-covered → NaN, plus a median fill value).
  Three related behaviours worth knowing when debugging the live mosaic:
  * **Staircase artifact → full-canvas writes.** `MosaicWriterOp` writes the
    *entire* cumulative canvas every batch (latest-wins, capacity-1 + drop
    policy), not just the bbox touched this batch. Writing only the bbox left
    earlier rows stale on unidirectional scans (a staircase). The cumulative
    canvas already contains all patches `0..N`, so a full write is always a
    complete snapshot.
  * **Centring bias → direction inference.** When the canvas is allocated
    before the slow axis has moved (first `PointProcessorOp` batch), the
    observed midpoint is biased toward the scan start. `estimate_canvas_mid`
    (in `holoptycho/mosaic_canvas.py`) detects when the observed range is
    < 50% of commanded and projects the centre to the midpoint of the
    *commanded* range using the inferred scan direction. Tested in
    `tests/test_mosaic_canvas.py`.
  * **NaN-position race → pending-frame buffer.** ViT batches can arrive
    before `PointProcessorOp` has filled the matching scan positions, so a
    frame's position is NaN. Rather than drop it, `SaveViTResult` buffers it
    in `_pending_frames` and re-stitches it on a later batch once the position
    is known (capped at 500 frames). This prevents missing rows when the
    position stream lags the ViT stream.

* **`patch_flip` (config field, default `identity`) — D4 transform applied to
  each ViT output patch before stitching** (via `ptychoml.apply_d4`). Use when
  the model's patch orientation doesn't match the scan-position frame; the
  orientation auto-detector will set it once wired up. **`mosaic_min_overlap`**
  (default 0.5) is the minimum fractional coverage for a mosaic pixel to count
  as filled; below it the pixel takes the valid-region median instead.

* **`inner_crop` (config field, default unset) — pixels trimmed from each edge
  of a ViT output patch before stitching**, to drop the FFT-leakage border
  outside the probe support. Resolution precedence (in `compose()`):
  explicit config `inner_crop` > value derived from the ONNX probe geometry
  (`holoptycho.engine_probe.find_onnx_for_engine` →
  `inner_crop_from_onnx` → `ptychoml.inner_crop_from_probe`) > `None`, in which
  case `SaveViTResult` auto-derives `min(patch_h, patch_w) // 4`. The ONNX
  derivation needs the `onnx` package and a model ONNX reachable via the
  `{name}_v{ver}.engine` → `onnx/{name}/{ver}/*.onnx` cache convention; it
  silently falls back to auto-derive otherwise.

* **`frame_write_stride` (config field, default 1000 for `recon_mode='vit'`,
  else 1) — detector-frame downsampling for tiled writes.** Persisting every
  detector frame is ~1 MB per 64-frame patch over WAN — fine for fine-tuning
  runs (`recon_mode='iterative'` or `'both'`), wasteful for ViT-only spot
  checks where the operator only needs to confirm preprocessing looks right.
  `<run>/diffraction/dp` is allocated at `(n_keep, H, W)` where `n_keep =
  (nz - 1) // stride + 1`; only frames where `frame_idx % stride == 0` are
  kept, and they map to compact rows via `row = frame_idx // stride`. The
  stride is stamped in run metadata as `dp_stride` so the dashboard can
  label its detector tile ("frame 39000 · 1 of every 1000 frames") and
  consumers can recover the scan-frame number from a row index. Set
  `frame_write_stride=1` in the config to capture every frame regardless of
  recon_mode (required if the run will be used as a ptycho-vit fine-tuning
  sample).

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

* **`--mode {iterative,vit,both}` for branch-isolation.** Use
  `iterative` to test the DM/ML solver without TRT competition, `vit` to
  test the ViT branch (and the server-side mosaic stitching) without
  iterative, or `both` to run them in parallel for comparison. `vit`-only
  is the fastest path for verifying changes to `ptychoml.stitch` /
  `SaveViTResult`.

* **Leave compression off** (now the default). `dectris-compression 0.3.1`
  removed the C `compress` entrypoint, so passing `--compress` falls back
  to Python `bitshuffle.compress_lz4` per frame, gating publish throughput
  at ~15 frames/sec — a 50-second scan takes ~12 minutes to replay and the
  pipeline can fall behind by tens of batches. The default raw-bytes path
  publishes frames with a `"raw"` encoding header; the receiver in
  `holoptycho/datasource.py::decode_json_message` recognises this and
  reshapes the bytes directly, skipping decompression. Localhost ZMQ
  handles the ~10× larger wire size easily, so replay runs at the
  ZMQ-bound rate. Live mode is unaffected — the real Eiger detector
  never sets `encoding=raw`. Enable `--compress` only when explicitly
  verifying the decompression code path.

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
                       ptychoml.stitch.place_patches_fourier_shift).
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
    diffraction/                ← always written (every run)
      dp                         ← (nz, H, W) uint8 amplitude (= sqrt of
                                   detector intensity, rounded). Captured post
                                   bad-pixel inpaint, pre rot90/fftshift. uint8
                                   instead of raw uint16 because Tiled does not
                                   accept compressed write_block payloads;
                                   sqrt-then-uint8 is lossless for ML training
                                   (1-count quantization < Poisson noise) and
                                   halves the wire volume. ptycho-vit's Tiled
                                   loader skips its own sqrt step on this path.
                                   Structure registered at start_run with chunks
                                   of 64 frames; chunks landed via write_block
                                   in scan order as frames arrive.
      probe_position_x_m         ← (nz,) float64 meters; sibling to positions_um
      probe_position_y_m         ← (nz,) float64 meters
```

Run metadata also includes `xray_energy_kev`, `wavelength_m`, `distance_m`,
`fine_tunable: bool`, and `complete: bool`.

- `xray_energy_kev`, `wavelength_m`, `distance_m` — needed by physics-aware
  loaders.
- `fine_tunable` — `True` iff `recon_mode` is `iterative` or `both` (i.e.
  the iterative branch will populate `final/probe` and `final/object`,
  which ptycho-vit's training loader requires). Filter for fine-tuning
  candidates via `Eq("fine_tunable", True)`.
- `complete` — starts `False`; flipped to `True` when the holoscan pipeline
  finishes processing this scan (at iterative end-of-run for `iterative` /
  `both` modes, or at clean subprocess exit for `vit`-only). Filter for
  finalised runs via `Eq("complete", True)`.

`run_uid` is generated in `PtychoApp.compose()` and surfaced via
`TiledWriter.start_run(run_uid, metadata)`. `raw_uid` and `raw_scan_id` come
from the run config and are populated automatically by
`scripts/config_from_tiled.py::build_full_config()` (and therefore by
`scripts/replay_from_tiled.py --hp-start`).

To list runs for a particular raw scan, query the catalog with
`tiled.queries.Eq("scan_id", "<scan_id>")` (or `Eq("raw_uid", ...)`).

`TILED_BASE_URL` is required; the pipeline raises `RuntimeError` and refuses
to start if it is unset. `TILED_API_KEY` is optional — when omitted, the
writer reuses the cached token from `tiled login` for the same server, or
falls back to anonymous access.

---

## Deprecated (planned for removal)

The following remain in the repo for reference and will be removed in a future release:

- **`InitRecon`**, **`liverecon_utils.py`** — scan header file watcher.
