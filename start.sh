#!/bin/bash
# Start the holoptycho production container.
#
# Prereqs (one-time):
#   - az login, with permissions to read genesisdemoskv and pull from
#     genesisdemosacr
#   - `pixi install -e client` on the host (one-time; no private deps), then
#     `pixi run -e client tiled profile create https://tiled.nsls2.bnl.gov --name nsls2`
#     and `pixi run -e client tiled login --profile nsls2` so personal Tiled
#     tokens are cached under ~/.config/tiled (unless using --api-key)
#   - On slurm / non-systemd hosts, configure ~/.config/containers/storage.conf
#     for a shared graphroot — see scripts/slurm_start_holoptycho.sh for the
#     storage.conf template.
#
# Tiled auth: by default the container uses your personal cached token from
# `tiled login` (mounted in from ~/.config/tiled) so Tiled writes are
# attributed to you. Pass `--api-key` to use the shared TILED_API_KEY from
# Key Vault instead — appropriate for unattended/production runs.
#
# By default the container runs in the foreground so its logs stream to this
# terminal — Ctrl-C to stop. Pass `-d` to run it detached; the script prints
# the logs/stop commands and exits.

set -euo pipefail

DETACH=0
USE_API_KEY=0
LIVE=0
EXPOSE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--detach) DETACH=1 ;;
    --api-key) USE_API_KEY=1 ;;
    --live) LIVE=1 ;;
    --expose) EXPOSE=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./start.sh [-d|--detach] [--api-key] [--live] [--expose]

  -d, --detach   Run the container detached. The script prints the
                 logs/stop commands and exits.
  --api-key      Use the shared TILED_API_KEY from Key Vault instead of
                 your personal cached token from `tiled login`.
  --live         Connect to the live beamline ZMQ streams (Eiger + PandA
                 on xf03idc-eiger2-ioc.nsls2.bnl.local). Fetches the
                 Eiger CurveZMQ server public key from Key Vault.
                 Without this flag the container listens on localhost
                 ports 5555/5556 for the replay script.
  --expose       Bind the API port to 0.0.0.0 instead of 127.0.0.1 so
                 other machines on the network can reach it directly at
                 http://<hostname>:8000. By default the port is only
                 accessible on localhost (via SSH tunnel).
USAGE
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

# --- Configuration ---------------------------------------------------------
ACR_NAME="genesisdemosacr"
IMAGE="${ACR_NAME}.azurecr.io/holoptycho:latest"
KEYVAULT="genesisdemoskv"
SP_DISPLAY_NAME="NSLS2-Genesis-Holoptycho"
RESOURCE_GROUP="rg-genesis-demos"
ML_WORKSPACE="genesis-mlw"
TILED_BASE_URL="https://tiled.nsls2.bnl.gov"
HOST_PORT=8000
CONTAINER_NAME="holoptycho"

# --- Podman runtime setup --------------------------------------------------
# On compute nodes `docker` is an alias for rootless podman, which needs a
# writable XDG_RUNTIME_DIR. Always point it at a private /tmp path we own.
#
# We deliberately do NOT trust an inherited XDG_RUNTIME_DIR=/run/user/<uid>.
# A Slurm allocation (salloc/sbatch) inherits that value from the login
# shell, where logind's PAM session created and mounted /run/user/<uid> — so
# the path *exists and passes the `-w` writable test on the compute node*
# even though logind never backed it there (no PAM session on the batch node
# unless `loginctl enable-linger` is set). A conditional `-w` fallback is
# therefore never triggered, and podman fails on a runtime dir that isn't
# properly backed. Unconditionally overriding avoids that (issue #36).
# Harmless on real-docker hosts and login nodes — the dir just sits unused.
XDG_RUNTIME_DIR="/tmp/xdg-$(id -u)"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# --- ACR login -------------------------------------------------------------
# az acr login normally hands a token to the Docker daemon, but the compute
# nodes use rootless podman (no daemon) via a `docker` alias. --expose-token
# prints the token instead so we can pass it directly to `docker login`. The
# 00000000-... username is ACR's documented sentinel value for token-based auth.
ACR_TOKEN="$(az acr login --name "$ACR_NAME" --expose-token --query accessToken -o tsv)"
docker login "${ACR_NAME}.azurecr.io" \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$ACR_TOKEN"

# --- Fetch secrets ---------------------------------------------------------
AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
AZURE_CLIENT_ID="$(az ad app list --display-name "$SP_DISPLAY_NAME" --query '[0].appId' -o tsv)"
AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-sp-cert --query value -o tsv)"
export AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID AZURE_CLIENT_ID AZURE_CERTIFICATE_B64

if [[ $USE_API_KEY -eq 1 ]]; then
  TILED_API_KEY="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-tiled-api-key --query value -o tsv)"
  export TILED_API_KEY
fi

# --- ZMQ stream sources ----------------------------------------------------
BEAMLINE_HOST="xf03idc-eiger2-ioc.nsls2.bnl.local"
if [[ $LIVE -eq 1 ]]; then
  SERVER_STREAM_SOURCE="tcp://${BEAMLINE_HOST}:5559"
  PANDA_STREAM_SOURCE="tcp://${BEAMLINE_HOST}:6666"
  SERVER_PUBLIC_KEY="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-eiger-server-public-key --query value -o tsv)"
  CLIENT_PUBLIC_KEY="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-client-public-key --query value -o tsv)"
  CLIENT_SECRET_KEY="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-client-secret-key --query value -o tsv)"
  export SERVER_PUBLIC_KEY CLIENT_PUBLIC_KEY CLIENT_SECRET_KEY
else
  SERVER_STREAM_SOURCE="tcp://host.docker.internal:5555"
  PANDA_STREAM_SOURCE="tcp://host.docker.internal:5556"
fi
export SERVER_STREAM_SOURCE PANDA_STREAM_SOURCE

# --- Run the container -----------------------------------------------------
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

BIND_ADDR="$([[ $EXPOSE -eq 1 ]] && echo "0.0.0.0" || echo "127.0.0.1")"

run_args=(
  --rm
  --name "$CONTAINER_NAME"
  --pull=always
  --gpus all
  --shm-size=32g
  -p "${BIND_ADDR}:${HOST_PORT}:8000"
  # host-gateway resolves to the host's gateway IP from inside the
  # container; works under both Docker Desktop (WSL2) and rootless podman
  # (slurm). Lets the container reach the replay script's ZMQ publishers
  # without --network host.
  --add-host=host.docker.internal:host-gateway
  -e AZURE_TENANT_ID
  -e AZURE_CLIENT_ID
  -e AZURE_SUBSCRIPTION_ID
  -e AZURE_CERTIFICATE_B64
  -e AZURE_RESOURCE_GROUP="$RESOURCE_GROUP"
  -e AZURE_ML_WORKSPACE="$ML_WORKSPACE"
  -e TILED_BASE_URL="$TILED_BASE_URL"
  -e SERVER_STREAM_SOURCE
  -e PANDA_STREAM_SOURCE
)

# Persist the TensorRT engine cache (ENGINE_CACHE_DIR=/models in the container)
# on the host so `hp model set` doesn't re-download the ONNX from Azure ML and
# re-compile via TensorRT on every --rm restart — a couple of minutes per model
# for a byte-identical result. Engine files are named {name}_v{version}.engine,
# so new versions add files alongside old ones (no manual versioning). (issue #33)
mkdir -p "$HOME/.cache/holoptycho/models"
run_args+=(-v "$HOME/.cache/holoptycho/models:/models")

if [[ $LIVE -eq 1 ]]; then
  run_args+=(-e SERVER_PUBLIC_KEY -e CLIENT_PUBLIC_KEY -e CLIENT_SECRET_KEY)
fi

if [[ $USE_API_KEY -eq 1 ]]; then
  run_args+=(-e TILED_API_KEY)
else
  # Mount the host's tiled token cache. tiled-client stores access/refresh
  # tokens under ~/.cache/tiled/tokens/<url-encoded-base>/ — the container's
  # HOME defaults to /root, so this is where it looks. Tiled writes are
  # attributed to the authenticated user.
  run_args+=(-v "$HOME/.cache/tiled:/root/.cache/tiled")
fi

if [[ $DETACH -eq 0 ]]; then
  # Foreground: logs stream here, Ctrl-C stops the container.
  exec docker run "${run_args[@]}" "$IMAGE"
fi

# Detached: hand the user the commands they'll need to inspect/stop it.
docker run -d "${run_args[@]}" "$IMAGE" >/dev/null
echo "Started ${CONTAINER_NAME} on http://127.0.0.1:${HOST_PORT}"
echo "Logs:  docker logs -f ${CONTAINER_NAME}"
echo "Stop:  docker stop ${CONTAINER_NAME}"
