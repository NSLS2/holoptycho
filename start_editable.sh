#!/bin/bash
# Drop into a dev shell with the host repo bind-mounted into a minimal
# CUDA+pixi container. Use on hosts with a glibc too old to run the pixi
# env directly (e.g. older RHEL).
#
# Edit, commit, and push from the host as normal — only code execution
# happens inside the container. Inside the shell:
#     pixi install                                          # first time
#     pixi run tiled profile create https://tiled.nsls2.bnl.gov --name nsls2  # once
#     pixi run tiled login --profile nsls2                   # once per shell
#     export ENGINE_CACHE_DIR=/tmp/models
#     pixi run api
#
# Prereqs (one-time):
#   - az login, with permissions to read genesisdemoskv

set -euo pipefail

DEV_IMAGE="cuda-dev"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- One-time build of the dev image --------------------------------------
# Layered on top of nvidia/cuda + pixi. Nothing holoptycho-specific lives
# in here, so it's reusable across any pixi/CUDA project.
if ! docker image inspect "$DEV_IMAGE" >/dev/null 2>&1; then
  echo "Building $DEV_IMAGE..."
  BUILDAH_ISOLATION=chroot docker build -t "$DEV_IMAGE" - <<'EOF'
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/no-sandbox && \
    apt-get update && apt-get install -y --no-install-recommends \
      libgl1 curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash
EOF
fi

# --- Run the dev shell ----------------------------------------------------
# --network host
#     The holoscan app reaches host services (Azure ML / MLflow, Tiled,
#     ZMQ streams) as if it were running on the host. With bridge
#     networking, localhost inside the container is the container itself.
# --user $(id -u):$(id -g)
#     Files created inside (e.g. into .pixi/) stay owned by the host user
#     in the mounted repo, not root.
# -v "$REPO_DIR":/app
#     The whole repo (incl. .pixi/) is mounted live. Host-side edits show
#     up inside the container immediately.
# -e HOME=/tmp
#     Keeps ~/.cache/, ~/.config/, tiled tokens, etc. out of the mounted
#     repo. Ephemeral — dies with --rm.
# --env-file <(cat <<EOF ... EOF)
#     Azure secrets are piped through a temp file (mode 600, deleted after
#     docker run) — re-pulled fresh from Azure each run. TILED_API_KEY is
#     fetched from Key Vault; use `pixi run tiled login` as a fallback if
#     az is not available.
# Build the Azure env block only when az is available and the user is logged in.
# Without it, the container still starts — Azure ML model management won't work,
# but pixi install and local development are unaffected.
#
# Note: podman does not support process substitution for --env-file, so we
# write to a temp file (mode 600) and delete it immediately after docker run.
AZURE_ENV_ARGS=()
ENV_TMPFILE=""
if command -v az > /dev/null 2>&1 && az account show > /dev/null 2>&1; then
  ENV_TMPFILE="$(mktemp)"
  chmod 600 "$ENV_TMPFILE"
  cat > "$ENV_TMPFILE" <<EOF
AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)
AZURE_CLIENT_ID=$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)
AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
AZURE_CERTIFICATE_B64=$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv)
TILED_API_KEY=$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)
AZURE_RESOURCE_GROUP=rg-genesis-demos
AZURE_ML_WORKSPACE=genesis-mlw
EOF
  AZURE_ENV_ARGS=(--env-file "$ENV_TMPFILE")
else
  echo "WARNING: az CLI not found or not logged in — starting without Azure credentials (hp model set will not work)" >&2
fi

docker run --rm -it --gpus all --shm-size=32g --network host \
  --userns=keep-id \
  -v "$REPO_DIR":/app \
  -v "$REPO_DIR/../ptycho":/ptycho \
  -v "$REPO_DIR/../ptychoml":/ptychoml \
  -e HOME=/tmp \
  -e TILED_BASE_URL=https://tiled.nsls2.bnl.gov \
  -e ENGINE_CACHE_DIR=/tmp/models \
  -e SERVER_STREAM_SOURCE=tcp://localhost:5555 \
  -e PANDA_STREAM_SOURCE=tcp://localhost:5556 \
  -w /app \
  "${AZURE_ENV_ARGS[@]}" \
  "$DEV_IMAGE" bash

# Clean up the temp env file (credentials never persist on disk longer than needed)
[[ -n "$ENV_TMPFILE" ]] && rm -f "$ENV_TMPFILE"
