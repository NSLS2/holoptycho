#!/bin/bash
#SBATCH --job-name=holoptycho
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Submit with:
#   mkdir -p $HOME/logs
#   sbatch --output=$HOME/logs/%x.%J.out --error=$HOME/logs/%x.%J.err scripts/slurm_start_holoptycho.sh
#
# Required setup (one-time):
#   - az login (with permissions to read genesisdemoskv and pull from genesisdemosacr)
#   - ~/.config/containers/storage.conf configured for shared graphroot (see README)

#[storage]
#driver = "overlay"
#graphroot = "/var/tmp/<username>/containers/storage"
#runroot = "/tmp/podman-run-<userid>"

set -euo pipefail

# --- Configuration ---------------------------------------------------------
ACR_NAME="genesisdemosacr"
IMAGE="${ACR_NAME}.azurecr.io/holoptycho:latest"
KEYVAULT="genesisdemoskv"
SP_DISPLAY_NAME="NSLS2-Genesis-Holoptycho"
RESOURCE_GROUP="rg-genesis-demos"
ML_WORKSPACE="genesis-mlw"
TILED_BASE_URL="https://tiled.nsls2.bnl.gov"
HOST_PORT=8000

# --- Podman runtime setup --------------------------------------------------
# sbatch jobs don't get a systemd user session, so XDG_RUNTIME_DIR isn't set
# by logind the way it is for interactive shells. Point it at a private /tmp
# path so podman has somewhere to write its runtime state.
export XDG_RUNTIME_DIR="/tmp/xdg-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# --- ACR login -------------------------------------------------------------
# Use --expose-token to get a refresh token that podman can consume directly.
# Username 00000000-0000-0000-0000-000000000000 is the documented sentinel
# value for token-based ACR auth.
ACR_TOKEN="$(az acr login --name "$ACR_NAME" --expose-token --query accessToken -o tsv)"
podman login "${ACR_NAME}.azurecr.io" \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$ACR_TOKEN"

# --- Fetch secrets ---------------------------------------------------------
AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
AZURE_CLIENT_ID="$(az ad app list --display-name "$SP_DISPLAY_NAME" --query '[0].appId' -o tsv)"
AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-sp-cert --query value -o tsv)"
TILED_API_KEY="$(az keyvault secret show --vault-name "$KEYVAULT" --name holoptycho-tiled-api-key --query value -o tsv)"

# --- Run the container -----------------------------------------------------
podman run --rm \
  --pull=always \
  --gpus all \
  --shm-size=32g \
  -p 127.0.0.1:${HOST_PORT}:8000 \
  -e AZURE_TENANT_ID \
  -e AZURE_CLIENT_ID \
  -e AZURE_SUBSCRIPTION_ID \
  -e AZURE_CERTIFICATE_B64 \
  -e AZURE_RESOURCE_GROUP="$RESOURCE_GROUP" \
  -e AZURE_ML_WORKSPACE="$ML_WORKSPACE" \
  -e TILED_BASE_URL="$TILED_BASE_URL" \
  -e TILED_API_KEY \
  -e SERVER_STREAM_SOURCE="tcp://localhost:5555" \
  -e PANDA_STREAM_SOURCE="tcp://localhost:5556" \
  "$IMAGE"
