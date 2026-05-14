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
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--detach) DETACH=1 ;;
    --api-key) USE_API_KEY=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./start.sh [-d|--detach] [--api-key]

  -d, --detach   Run the container detached. The script prints the
                 logs/stop commands and exits.
  --api-key      Use the shared TILED_API_KEY from Key Vault instead of
                 your personal cached token from `tiled login`.
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
# On compute nodes `docker` is an alias for rootless podman. sbatch and
# other non-systemd shells don't get XDG_RUNTIME_DIR from logind, so point
# it at a private /tmp path so podman has somewhere to write its runtime
# state. Harmless on real-docker hosts (the dir just sits unused).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$(id -u)}"
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

# --- Run the container -----------------------------------------------------
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

run_args=(
  --rm
  --name "$CONTAINER_NAME"
  --pull=always
  --gpus all
  --shm-size=32g
  --network host
  -e AZURE_TENANT_ID
  -e AZURE_CLIENT_ID
  -e AZURE_SUBSCRIPTION_ID
  -e AZURE_CERTIFICATE_B64
  -e AZURE_RESOURCE_GROUP="$RESOURCE_GROUP"
  -e AZURE_ML_WORKSPACE="$ML_WORKSPACE"
  -e TILED_BASE_URL="$TILED_BASE_URL"
  -e SERVER_STREAM_SOURCE="tcp://localhost:5555"
  -e PANDA_STREAM_SOURCE="tcp://localhost:5556"
)

if [[ $USE_API_KEY -eq 1 ]]; then
  run_args+=(-e TILED_API_KEY)
else
  # Mount the host's tiled cached tokens. Container HOME defaults to /root,
  # so this is where tiled-client looks for the per-server cached token
  # written by `tiled login`. Tiled writes are attributed to the
  # authenticated user.
  run_args+=(-v "$HOME/.config/tiled:/root/.config/tiled")
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
