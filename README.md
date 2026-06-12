# Holoptycho

Real-time streaming ptychographic reconstruction using [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), developed for the HXN beamline at NSLS-II. **Holoptycho is for real-time streaming reconstruction only** — it consumes live detector data via ZMQ and emits results to a Tiled catalog as the scan runs.

For batch/offline reconstruction of completed scans, use [`NSLS2/ptycho`](https://github.com/NSLS2/ptycho) or [`NSLS2/ptychoml`](https://github.com/NSLS2/ptychoml) directly.

---

## Quick start

### 1. Install the CLI

```bash
git clone git@github.com:NSLS2/holoptycho.git
cd holoptycho
pixi install -e client
```

Add a shell alias so you can type `hp` from anywhere. Set `HOLOPTYCHO_REPO` to wherever you cloned the repo:

```bash
# bash
echo 'alias hp="pixi run --manifest-path /path/to/holoptycho/pixi.toml -e client hp"' >> ~/.bashrc && source ~/.bashrc

# zsh
echo 'alias hp="pixi run --manifest-path /path/to/holoptycho/pixi.toml -e client hp"' >> ~/.zshrc && source ~/.zshrc
```

### 2. Open an SSH tunnel to mars5

The API port is firewalled. The tunnel must be opened as your **personal NSLS-II account** (not the beamline account) since compute nodes require key-based SSH auth. Once the tunnel is open, any account on the machine can use it.

Add an alias to your personal shell config:

```bash
# bash (~/.bashrc)
echo 'alias proxy_mars="ssh -L 8000:localhost:8000 -N mars5.nsls2.bnl.gov"' >> ~/.bashrc && source ~/.bashrc

# zsh (~/.zshrc)
echo 'alias proxy_mars="ssh -L 8000:localhost:8000 -N mars5.nsls2.bnl.gov"' >> ~/.zshrc && source ~/.zshrc
```

SSH will use your current username by default. Run `proxy_mars` in a terminal and keep it open — `hp` commands from any account on the machine will reach the server through the tunnel.

### 3. Point hp at the server

The tunnel forwards `localhost:8000` to mars5, so the default remote works as-is — no configuration needed.

### 4. Start a reconstruction

```bash
hp start "$(pixi run -e client config-from-tiled --scan-id <scan_id>)"
```

That's it. Use `hp status`, `hp logs`, and `hp stop` to monitor and control the pipeline.

---

## Server deployment

The server runs on a Slurm GPU node (`mars5`). To (re)start it:

1. Allocate a GPU node:
   ```bash
   salloc --gres=gpu:1 --mem=64G --cpus-per-gpu=2 --account=staff
   ```

2. Log in to Azure:
   ```bash
   az login
   ```

3. Cache a personal Tiled token (skip if using `--api-key`):
   ```bash
   pixi run -e client tiled profile create https://tiled.nsls2.bnl.gov --name nsls2
   pixi run -e client tiled login --profile nsls2
   ```

4. Start the container:
   ```bash
   ./start.sh --live --expose -d           # live beamline, detached, network-accessible
   ./start.sh --live --expose -d --api-key # same but with shared Tiled API key from Key Vault
   ./start.sh                              # replay/testing mode (localhost ZMQ only)
   ```

   `--live` connects to the Eiger and PandA ZMQ streams on the beamline.
   `--expose` binds the API to `0.0.0.0` so other machines on the network can reach it at `http://mars5.nsls2.bnl.gov:8000`.

---

## Verifying ZMQ connectivity

Before starting a live run, use `scripts/check_zmq.py` to confirm both streams
are reachable and the CurveZMQ key is accepted. The script connects to the
Eiger and PandA endpoints, waits up to `--timeout` seconds for any message,
and reports one of three outcomes per stream:

| Result | Meaning |
|---|---|
| `OK` | Connected and received data — detector is armed / scan is running |
| `TIMEOUT` | Connected fine but no data — no scan is running (expected when idle) |
| `ERROR` | Could not connect — wrong key, host unreachable, or port closed |

A `TIMEOUT` with no scan running is the expected result and confirms the
connection is healthy.

```bash
# Fetch the Eiger server key and run the check (uses the replay env — no GPU needed)
pixi install -e replay   # once

SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-eiger-server-public-key --query value -o tsv)" pixi run -e replay python scripts/check_zmq.py

# Longer timeout during a running scan to confirm data flows
SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-eiger-server-public-key --query value -o tsv)" pixi run -e replay python scripts/check_zmq.py --timeout 30
```

`CLIENT_PUBLIC_KEY` and `CLIENT_SECRET_KEY` are optional for this script — if
absent it generates a throwaway keypair automatically. The Eiger server uses
the client key only for encryption, not for allowlisting, so an ephemeral pair
is sufficient for connectivity checks. The production pipeline (`datasource.py`)
requires real client keys and will refuse to start with only the server key set.

---

## Testing with the replay script

To test holoptycho end-to-end without a live beamline, use `scripts/replay_from_tiled.py`. It reads a real scan from Tiled and publishes it over ZMQ on the same node as holoptycho, in the exact Eiger and PandA wire formats. Both the replay script and holoptycho must run on the **same machine** — ZMQ traffic stays local.

### On the compute node

```bash
# 1. Authenticate with Tiled and install the replay env (once)
tiled profile create https://tiled.nsls2.bnl.gov --name nsls2
tiled login --profile nsls2
pixi install -e replay

# 2. If holoptycho has no selected engine yet, choose one before --hp-start
hp model set run042901
hp model status

# 3. Run the replay. Use --scan-id to look up the run automatically (newest
#    run with that scan_id wins — scan_id is not unique), or pass --uid
#    directly if you already have a UUID. The --tiled-url, --hp-url, and
#    --eiger/panda-endpoint flags all default to the HXN-typical values.
pixi run -e replay replay --scan-id 404611 --mode vit
```

By default the replay script publishes plain ZMQ. To test CurveZMQ, also
pass the full Eiger key set: `--eiger-server-public-key`,
`--eiger-server-secret-key`, and `--eiger-client-public-key`.

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

- **`--mode {iterative,vit,both}`** — which reconstruction branches the
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
- **`--compress`** — opt in to bslz4 compression and publish frames in the
  same wire format the live Eiger uses. **Off by default**, because
  `dectris-compression 0.3.1` removed the C `compress` entrypoint and the
  pure-Python `bitshuffle` fallback gates publish throughput at ~15 fps. The
  default raw-bytes path uses a `"raw"` encoding header that holoptycho's
  receiver recognises; localhost ZMQ handles the ~10× larger wire size
  easily. Enable only when explicitly testing the decompression code path.
- **Diffraction centering — `--batch-x0/--batch-y0` vs auto-centering.** The crop
  is a single `nx × ny` window in the coordinate-corrected (global) frame.
  **If you don't pass `--batch-x0/--batch-y0`, the pipeline auto-centers** —
  `ImageBatchOp` segments the first batch (largest connected component of the
  thresholded blob), searching the **whole frame**, and crops every batch to an
  `nx × ny` box on the beam (lossless: real pixels, no `np.roll`, no zero-fill).
  This is the same mechanism used live. Pass `--batch-x0/--batch-y0` to crop at a
  fixed global offset instead, or `--auto-center-dp` to force auto-centering
  alongside an explicit offset (then bounded by `--auto-center-headroom`, default
  `nx//4`). (Replay no longer COM-fills the offset — that bypassed auto-centering.)
- **`--detector-orientation`** — the local→global coordinate correction (a D4
  name) applied to the whole incoming frame *before* cropping. **Default
  `fliplr`** (raw live Eiger — a flip over the vertical axis). **Replay forces
  `identity`** because Tiled data is already corrected — *unless* `--simulate-live`
  is given. This replaces the old hardcoded `flip_image=True`, which applied
  `fliplr` to *both* sources: that was correct for live but double-flipped replay
  back into local coords.
- **`--simulate-live`** — make replay behave like the **live** Eiger to exercise
  the coordinate-correction path: un-rotate the (already-corrected) Tiled frames
  by `detector_orientation⁻¹` so they're raw-like, and set the config's
  `detector_orientation` to the live value (`--detector-orientation`, default
  `fliplr`) so the pipeline re-applies it. The recon is unchanged (round-trip is
  exact), but the live `fliplr` code runs — use it to validate live before
  deploying.
- **`--dp-orient-iterative`, `--x-direction-iterative`, `--y-direction-iterative`**
  — give the **iterative engine** its own diffraction orientation (one D4 name
  or a comma-separated sequence, e.g. `rot90_cw,fliplr`) and scan-axis sign
  conventions, independent of the ViT/AI branch (which keeps `dp_orient` /
  `x_direction` / `y_direction`). **Unset by default = identical behavior to
  before.** Demo flexibility for when the two recons disagree about input
  orientation on live data; once the right values are known, bake them into the
  config defaults. Note: setting `--dp-orient-iterative` opts the engine out of
  the live orientation auto-detect (it follows the shared `dp_orient` when
  unset).

### Best practices

- **`--nx` / `--ny` must match the selected engine's input dimensions.**
  These set the detector-frame crop size fed into the pipeline; the
  default of 256×256 matches the current HXN engines
  (`ptycho_vit_amp_phase_b64`, `run042901`). A mismatch with the engine
  input raises `ValueError: could not broadcast input array from shape
  (256,128) into shape (128,256)` at pipeline startup. The detector frame
  can be larger — the pipeline crops down — but it must be at least
  `nx × ny`.
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
- **Default to `--mode vit`** when iterating on ViT/mosaic code —
  fastest cycle and the iterative branch can't crash the run.
- **`--max-frames N` plus `--n-iterations 50–100`** gets you a full
  end-to-end cycle (config → stream → recon → final write) in under a
  minute for quick smoke tests on big scans.
- **Leave compression off** (the default). With the current
  `dectris-compression` package the C `compress` is missing, so enabling
  `--compress` falls back to Python `bitshuffle` and gates the pipeline at
  ~15 frames/sec. Enable only when you specifically need to test the
  decompression path.

---

## Development container

On hosts with a glibc too old to run the pixi env directly (e.g. older RHEL), use [`start_editable.sh`](start_editable.sh) to drop into a minimal CUDA+pixi container with the repo bind-mounted. Edit, commit, and push from the host as normal; only run code inside the container.

```bash
./start_editable.sh
```

The first run builds a small `cuda-dev` image (nvidia/cuda runtime + pixi) — about a minute. Subsequent runs reuse it. Inside the shell:

```bash
pixi install                                                              # first time / after pixi.lock changes
pixi run tiled profile create https://tiled.nsls2.bnl.gov --name nsls2    # once per dev shell
pixi run tiled login --profile nsls2
export ENGINE_CACHE_DIR=/tmp/models
pixi run api
```

Why this works:
- `--network host` so the holoscan app reaches host services (Azure ML / MLflow, Tiled, ZMQ streams) as if it were running on the host.
- The whole repo (incl. `.pixi/`) is bind-mounted at `/app`, so host-side edits show up inside immediately.
- `HOME=/tmp` keeps caches and tiled tokens out of the mounted repo; they die with `--rm`.
- Azure secrets are piped via `--env-file <(...)` — an in-kernel FIFO — so they never touch disk and don't appear in `ps`.
- Tiled uses your personal identity (via `tiled login`) instead of a shared `TILED_API_KEY`, so you get the right access scope and a real audit trail.

Always run `pixi install` **inside** the dev container, never on the host — that way the env's binaries link against the container's glibc, which is what they run against in production. If you previously ran `pixi install` on the host, delete `.pixi/` and re-install inside the container the first time so nothing is stale.

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

Each pipeline run produces a fresh container under `hxn/processed/holoptycho/{run_uid}/` (a per-run UUID; the catalog root is overrideable via `TILED_CATALOG_PATH`), tagged with the `synaps_project` spec. Container metadata records the raw scan it was reconstructed from (`raw_uid`, `scan_id`, `scan_num`, `started_at`, `recon_mode`, `xray_energy_kev`, `wavelength_m`, `distance_m`, plus a boolean `fine_tunable` flag that's true iff `recon_mode` is `iterative` or `both`).

Every run also writes a `<run>/diffraction/` subtree containing detector-frame amplitude (`dp`, `(nz, H, W) uint8`, i.e. `sqrt(intensity)` rounded to 8-bit) and meter-unit probe positions (`probe_position_x_m`, `probe_position_y_m`). uint8 storage cuts the on-the-wire write volume in half versus uint16 without measurable quality loss for ML (the 1-count quantization is below the Poisson noise floor). A run is usable as a ptycho-vit fine-tuning sample iff its metadata has `fine_tunable: true` — the iterative branch then also writes `final/probe` and `final/object` as supervised targets.

---

## Required environment variables

| Variable | Description |
|---|---|
| `SERVER_STREAM_SOURCE` | ZMQ endpoint of the Eiger detector, e.g. `tcp://<host>:5555` |
| `PANDA_STREAM_SOURCE` | ZMQ endpoint of the PandA box, e.g. `tcp://<host>:5556` |
| `TILED_BASE_URL` | URL of the Tiled server |

The pipeline will refuse to start if any of `SERVER_STREAM_SOURCE`, `PANDA_STREAM_SOURCE`, or `TILED_BASE_URL` are not set. `TILED_API_KEY` is optional — when unset, the writer uses the cached token from `tiled login` (run once: `tiled profile create <url> --name <name>` then `tiled login --profile <name>`).

## Optional environment variables

| Variable | Description |
|---|---|
| `TILED_CATALOG_PATH` | Tiled catalog path (default: `hxn/processed/holoptycho`) |
| `HOLOPTYCHO_LOG_LEVEL` | Root log level for the API and pipeline logs (default: `INFO`; set to `DEBUG` for per-write Tiled debug logs) |
| `SERVER_PUBLIC_KEY` | CurveZMQ public key of the [holoscan-proxy](https://github.com/NSLS2/holoscan-proxy). Required only if the proxy is configured with `encrypt: true`. |
| `CLIENT_PUBLIC_KEY` | CurveZMQ public key of this client. Required if `SERVER_PUBLIC_KEY` is set. |
| `CLIENT_SECRET_KEY` | CurveZMQ secret key of this client. Required if `SERVER_PUBLIC_KEY` is set. |

---

## Controlling the pipeline

Use the `hp` CLI to start, stop, and configure the pipeline.

### Selecting a remote

```bash
hp remote list          # show all remotes (* = active)
hp remote set           # interactive picker
hp remote set mars5     # direct
hp remote status        # show current
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
tiled profile create https://tiled.nsls2.bnl.gov --name nsls2  # once
tiled login --profile nsls2

# By scan id (most recent run with that id wins)
hp start "$(pixi run -e client config-from-tiled --scan-id 404611)"

# Or by run UID if you have it
hp start "$(pixi run -e client config-from-tiled --uid 67e77251-cbe4-444c-8a8c-36491b0b9100)"
```

Override reconstruction parameters as needed:

```bash
hp start "$(pixi run -e client config-from-tiled --scan-id 404611 --nx 256 --ny 256 --n-iterations 1000)"

# Run only the iterative solver or only the ViT branch (default is both):
hp start "$(pixi run -e client config-from-tiled --scan-id 404611 --mode iterative)"
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

pixi run api   # listens on 127.0.0.1:8000
```

`SERVER_STREAM_SOURCE` and `PANDA_STREAM_SOURCE` are required — the pipeline refuses to start without them. Use `tcp://localhost:5555` / `tcp://localhost:5556` when pairing with `scripts/replay_from_tiled.py` on the same host.

---

## Profiling

```bash
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 \
    pixi run api
```

Requires `perf_event_paranoid <= 2`:
```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```

---

## Deprecated (planned for removal)

The following are no longer used and remain in the repo for reference only. They will be removed in a future release:

- **`InitRecon`**, **`liverecon_utils.py`** — scan header file watcher for detecting new scans from a beamline-written text file. Scan parameters now come from the API config.
- **`--mode simulate`** CLI option — removed; `hp start` always runs the live ZMQ pipeline.
