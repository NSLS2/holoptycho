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

Each pipeline run produces a fresh container under `hxn/processed/holoptycho/{run_uid}/` (a per-run UUID; the catalog root is overrideable via `TILED_CATALOG_PATH`), tagged with the `synaps_project` spec. Container metadata records the raw scan it was reconstructed from (`raw_uid`, `scan_id`, `scan_num`, `started_at`, `recon_mode`).

---

## Required environment variables

| Variable | Description |
|---|---|
| `SERVER_STREAM_SOURCE` | ZMQ endpoint of the Eiger detector, e.g. `tcp://<host>:5555` |
| `PANDA_STREAM_SOURCE` | ZMQ endpoint of the PandA box, e.g. `tcp://<host>:5556` |
| `TILED_BASE_URL` | URL of the Tiled server |
| `TILED_API_KEY` | Tiled API key (store in Azure Key Vault — see below) |

The pipeline will refuse to start if `SERVER_STREAM_SOURCE` or `PANDA_STREAM_SOURCE` are not set. If `TILED_BASE_URL` or `TILED_API_KEY` are absent, results fall back to `.npy` files under `/data/users/Holoscan/` with a warning.

## Optional environment variables

| Variable | Description |
|---|---|
| `TILED_CATALOG_PATH` | Tiled catalog path (default: `hxn/processed/holoptycho`) |
| `HOLOPTYCHO_LOG_LEVEL` | Root log level for the API and pipeline logs (default: `INFO`; set to `DEBUG` for per-write Tiled debug logs) |
| `SERVER_PUBLIC_KEY` | CurveZMQ public key of the [holoscan-proxy](https://github.com/NSLS2/holoscan-proxy). Required only if the proxy is configured with `encrypt: true`. |
| `CLIENT_PUBLIC_KEY` | CurveZMQ public key of this client. Required if `SERVER_PUBLIC_KEY` is set. |
| `CLIENT_SECRET_KEY` | CurveZMQ secret key of this client. Required if `SERVER_PUBLIC_KEY` is set. |

---

## Container deployment

A Docker image is built and pushed to Azure Container Registry on every merge to main. See [`.github/workflows/build-container.yml`](.github/workflows/build-container.yml).

### 1. Log in to Azure and ACR

```bash
az login
```

### 2. Extra step on slurm.

> ```bash
> export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u)
> mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
> ```

# az acr login normally hands a token to the Docker daemon, but this cluster
# uses rootless podman (no daemon). --expose-token prints the token instead
# so we can pass it directly to podman login.
```bash
podman login genesisdemosacr.azurecr.io \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$(az acr login --name genesisdemosacr --expose-token --query accessToken -o tsv)"
```

### 3. Run the container

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
  -e SERVER_STREAM_SOURCE="tcp://localhost:5555" \
  -e PANDA_STREAM_SOURCE="tcp://localhost:5556" \
  genesisdemosacr.azurecr.io/holoptycho:latest
```

The private key is never written to disk. The server binds to `0.0.0.0:8000` inside the container, exposed only on `127.0.0.1:8000` of the host.


### 4. Connect via SSH tunnel

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

Beamline metadata (energy, scan geometry, pixel size) can be pulled directly from Tiled and piped into `hp start`:

```bash
tiled login https://tiled.nsls2.bnl.gov
hp start "$(pixi run -e client config-from-tiled --scan-num 320045)"
```

Override reconstruction parameters as needed:

```bash
hp start "$(pixi run -e client config-from-tiled --scan-num 320045 --nx 256 --ny 256 --n-iterations 1000)"

# Run only the iterative solver or only the ViT branch (default is both):
hp start "$(pixi run -e client config-from-tiled --scan-num 320045 --recon-mode iterative)"
```

### Model selection

`hp model list` shows two sections:
- **Local cache** — `.engine` files in `ENGINE_CACHE_DIR` (default `/models`), ready to use immediately
- **Azure ML** — registered models, with a `cached` column showing what's already local

`hp model set` selects the engine for the next `hp start` or `hp restart`. If the engine is not cached locally it is pulled from Azure ML and compiled via the TensorRT Python API first.

```bash
hp model list
hp model set <model-name>                  # uses latest version
hp model set <model-name> --version <ver>  # pin to a specific version
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
| `recon_mode` | str | Which reconstruction branches to run: `iterative`, `vit`, or `both`. Default `both`. |
| `raw_uid` | str | (Optional) UID of the raw Bluesky run this reconstruction came from; stored on the per-run Tiled container as metadata. |
| `scan_id` | str | (Optional) Scan id of the raw run; stored on the per-run Tiled container as metadata. Defaults to `scan_num` if omitted. |
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

## Running on Slurm (sbatch)

For persistent operation independent of your SSH session, use the provided sbatch script. The job survives disconnects — only a Slurm job cancellation or walltime expiry will stop it.

```bash
sbatch scripts/slurm_start_holoptycho.sh
```

Once the job is running, check which node it landed on and open an SSH tunnel:

```bash
squeue -u $USER          # note the node name (e.g. mars5)
ssh -L 8000:localhost:8000 -J <login-node> <compute-node>
```

The `hp` CLI will now reach the server at `http://localhost:8000` as normal.

### Checking running jobs

```bash
squeue -u $USER          # show your running jobs and their node
squeue -u $USER -l       # verbose — includes time limit and reason
```

### Cancelling a job

```bash
scancel <jobid>
```

### Updating to the latest container image

The script uses `--pull=always`, so to pick up a new image just cancel the job and resubmit:

```bash
scancel <jobid>
sbatch scripts/slurm_start_holoptycho.sh
```

> **Note:** The script resolves Azure credentials at job start time using `az` CLI. Make sure you have run `az login` on the cluster before submitting — credentials are stored in `~/.azure/` which is available on compute nodes via the shared home directory.

---

## Testing with the replay script

To test holoptycho end-to-end without a live beamline, use `scripts/replay_from_tiled.py`. It reads a real scan from Tiled and publishes it over ZMQ on the same node as holoptycho, in the exact Eiger and PandA wire formats. Both the replay script and holoptycho must run on the **same machine** — ZMQ traffic stays local.

### On the compute node

```bash
# 1. Authenticate with Tiled
tiled login https://tiled.nsls2.bnl.gov

# If you need the run UID for a scan ID first:
pixi run -e replay python - <<'PY'
from tiled.client import from_uri
from tiled.queries import Eq

catalog = from_uri("https://tiled.nsls2.bnl.gov")["hxn"]["raw"]
results = catalog.search(Eq("scan_id", 320045))
uid = next(iter(results))
print(uid)
PY

# 2. Install the replay environment (once)
pixi install -e replay

# 3. If holoptycho has no selected engine yet, choose one before using --hp-start
hp model set nsls0408_bs1
hp model status

# 4. Start the replay script — it binds :5555 (Eiger) and :5556 (PandA)
# By default it publishes plain ZMQ. To test CurveZMQ, also pass the full
# Eiger key set: --eiger-server-public-key, --eiger-server-secret-key,
# and --eiger-client-public-key.
pixi run -e replay replay \
    --uid 67e77251-cbe4-444c-8a8c-36491b0b9100 \
    --tiled-url https://tiled.nsls2.bnl.gov/hxn/migration \
    --eiger-endpoint tcp://0.0.0.0:5555 \
    --panda-endpoint tcp://0.0.0.0:5556 \
    --rate 200 \
    --no-compress

# Or let the replay script build the config from the same run and start or
# restart holoptycho before it publishes anything. This requires a selected
# model/engine, so run `hp model set ...` first if needed.
pixi run -e replay replay \
    --uid 67e77251-cbe4-444c-8a8c-36491b0b9100 \
    --tiled-url https://tiled.nsls2.bnl.gov/hxn/migration \
    --hp-start \
    --hp-url http://localhost:8000 \
    --eiger-endpoint tcp://0.0.0.0:5555 \
    --panda-endpoint tcp://0.0.0.0:5556 \
    --rate 200 \
    --no-compress

# 5. If you did not use --hp-start, start holoptycho in another terminal
hp start '{"scan_num": "320045", ...}'
```

The container must be started with `SERVER_STREAM_SOURCE=tcp://localhost:5555` and `PANDA_STREAM_SOURCE=tcp://localhost:5556`. By default, leave `SERVER_PUBLIC_KEY`, `CLIENT_PUBLIC_KEY`, and `CLIENT_SECRET_KEY` unset so holoptycho subscribes without CurveZMQ. To test CurveZMQ, set all three in the container and pass the matching Eiger publisher keys to `scripts/replay_from_tiled.py`. Partial auth configuration is rejected on both sides. Control holoptycho from your local machine as normal via the `8000` SSH tunnel.

`--tiled-url` may be either the Tiled server root (`https://tiled.nsls2.bnl.gov`) or a catalog path (`https://tiled.nsls2.bnl.gov/hxn/migration`). The replay and config loaders resolve either form.

When `--hp-start` is used, the replay script builds the run config from the
same run metadata and chooses `/run` or `/restart` automatically based on the
current holoptycho server state before publishing. If `hp model status` shows
no selected engine, run `hp model set <model-name>` once first.

For same-node testing with the replay script, `localhost` only works if the
container is started with `--network host`. With bridge networking, `localhost`
inside the container refers to the container itself, not the Slurm node host.

### Useful replay flags

These flags only take effect when `--hp-start` is used (they're written into
the config the replay script POSTs to holoptycho):

- **`--recon-mode {iterative,vit,both}`** — which reconstruction branches the
  pipeline wires up. `iterative` runs only the DM/ML solver, `vit` runs only
  the ViT inference network, and `both` (default) runs them in parallel.
  Useful for isolating GPU-contention issues on single-GPU nodes or for
  comparing the two outputs side by side in Tiled (`live/` + `final/` come
  from iterative; `vit/` comes from the ViT branch).
- **`--n-iterations N`** — caps the iterative solver at `N` ticks
  (default 500). Once the iteration counter hits the cap, holoptycho trips
  the natural-termination path: `SaveResult` writes the final
  probe/object/timestamps to Tiled, then `fragment.stop_execution()`
  releases the run loop and the pipeline subprocess exits. Use a small
  value (50–100) for end-to-end smoke tests; use the production value
  (~500) for real reconstructions.
- **`--max-frames N`** — only publishes the first `N` frames of the scan,
  trimming positions to match. Handy for quick tests on big scans where
  downloading and replaying every frame would take too long.
- **`--skip-frames N`** — drops the first `N` Eiger frames (and aligned
  encoder samples) before publishing. Useful when a scan's initial rows
  overshoot the commanded extent during settling/ramp-up and crash the
  iterative recon, or when the first row of ViT predictions ends up in the
  wrong canvas region.
- **`--chunk-size N`** — number of frames per tiled fetch during streaming
  (default 256). The replay script pulls frames from tiled lazily rather
  than loading the whole scan up front, so replay starts publishing within
  seconds even for multi-GB scans. Smaller chunks = lower startup latency
  and lower peak memory; larger = fewer round-trips.
- **`--no-compress`** — skip bslz4 compression and publish raw frame bytes
  with a `"raw"` encoding header. The receiver inside holoptycho recognises
  this and reshapes the bytes directly without invoking the dectris
  decompressor. Avoids the ~30s/batch Python `bitshuffle` bottleneck that
  otherwise gates publish throughput (`dectris-compression 0.3.1` removed
  the C `compress` entrypoint, so the script falls back to a pure-Python
  implementation). Localhost ZMQ handles the ~10× larger wire size easily,
  so a 50s scan replays in 50s instead of ~12 min. Live mode is unaffected
  — the real Eiger detector never uses the `"raw"` encoding.

### Best practices

- **Pass `--nx` and `--ny` together.** `--hp-start` builds a fresh config
  from `config-from-tiled`, which defaults `nx`/`ny` to 128. Most HXN
  detector frames are 256×256, so a 128×anything ROI causes
  `ValueError: could not broadcast input array from shape (256,128) into shape (128,256)`
  at pipeline startup. For 256-pixel scans always pass `--nx 256 --ny 256`.
- **Run only one replay at a time.** Concurrent `--hp-start` replays
  mid-stream the pipeline: the second run's `/restart` interrupts the first
  while it's publishing, leaving PandA and Eiger out of sync. Positions
  stay NaN and the dashboard hangs. Kill any running replay
  (`pkill -f replay_from_tiled`) before launching a new one.
- **Use `--skip-frames` for scans with settling/ramp-up rows.** Some scans
  (e.g. 404611) have the first ~10 rows where encoder readings overshoot
  the commanded scan range by several × and crash the iterative recon's
  pre-allocated object grid. The ViT branch tolerates them but stitches
  them into the wrong canvas region. Drop those rows.
- **Default to `--recon-mode vit`** when iterating on ViT/mosaic code —
  fastest cycle and the iterative branch can't crash the run.
- **`--max-frames N` plus `--n-iterations 50–100`** gets you a full
  end-to-end cycle (config → stream → recon → final write) in under a
  minute for quick smoke tests on big scans.
- **Pass `--no-compress`** for replay throughput. With the current
  `dectris-compression` package the C `compress` is missing, so the
  default path falls back to Python `bitshuffle` which gates the
  pipeline at ~15 frames/sec. `--no-compress` removes that bottleneck
  entirely.

---

## Local development

Requires Linux (x86_64), an NVIDIA GPU, the system CUDA toolkit (for `cuda.h` and the matching driver lib), and [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install
pixi run test
```

### Building the default (GPU) env

The default pixi env builds `pycuda` from source against the system CUDA toolkit. If `pixi install` fails with `cuda.h: No such file or directory` or `cannot find -lcuda` / `-lcurand`, you need to:

1. Make sure `cuda.h` is reachable via `/usr/local/cuda/include` (system CUDA toolkit, e.g. installed via the NVIDIA `.run` installer or `nvidia-cuda-toolkit` apt package).
2. Make sure `libcuda.so` is reachable. On WSL2 it lives at `/usr/lib/wsl/lib/libcuda.so` (provided by the Windows NVIDIA driver). On bare-metal Linux the driver places it under `/usr/lib/x86_64-linux-gnu/`.
3. Conda-forge ships `libcurand.so.10` without the unversioned dev symlink that the linker needs. Create it once:

   ```bash
   ln -sf libcurand.so.10 .pixi/envs/default/lib/libcurand.so
   ```

Then run `pixi install` with the toolchain pointed at both the system CUDA headers and the WSL/driver lib path:

```bash
CUDA_ROOT=/usr/local/cuda CUDA_HOME=/usr/local/cuda CPATH=/usr/local/cuda/include \
  LIBRARY_PATH=/usr/lib/wsl/lib:$PWD/.pixi/envs/default/lib \
  pixi install
```

Drop `/usr/lib/wsl/lib` from `LIBRARY_PATH` on non-WSL hosts.

### Running the API server natively

The API server reads the same environment variables as the container. Pull them from Azure once per shell, then start the server:

```bash
az login   # one time

export AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
export AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)"
export AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
export AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv)"
export AZURE_RESOURCE_GROUP=rg-genesis-demos
export AZURE_ML_WORKSPACE=genesis-mlw

export TILED_BASE_URL="https://tiled.nsls2.bnl.gov"
export TILED_API_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)"

export SERVER_STREAM_SOURCE="tcp://localhost:5555"
export PANDA_STREAM_SOURCE="tcp://localhost:5556"

# ENGINE_CACHE_DIR defaults to /models, which is not writable outside the container.
# Point it at a user-writable path before starting the server.
export ENGINE_CACHE_DIR="$HOME/.cache/holoptycho/models"
mkdir -p "$ENGINE_CACHE_DIR"

pixi run start-api   # listens on 127.0.0.1:8000
```

`SERVER_STREAM_SOURCE` and `PANDA_STREAM_SOURCE` are required — the pipeline refuses to start without them. Use `tcp://localhost:5555` / `tcp://localhost:5556` when pairing with `scripts/replay_from_tiled.py` on the same host.

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
