# Source this (do not execute) to set up the environment for running the
# holoptycho API server natively (no container) against the replay script on the
# same host:
#
#   source scripts/env-replay.sh
#   pixi run api          # listens on 127.0.0.1:8000
#
# Then pair with the replay script in another shell:
#   pixi run -e replay replay --scan-id <N> --mode both
#
# Requires `az login` first. Pulls the Azure service-principal + Tiled creds from
# Key Vault, points the ZMQ sources at the local replay (localhost:5555/5556),
# and sets a writable engine cache. See AGENTS.md → "Starting the server
# natively (without container)".

if ! az account show >/dev/null 2>&1; then
  echo "env-replay: not logged in to Azure — run 'az login' first." >&2
  return 1 2>/dev/null || exit 1
fi

# --- Azure + Tiled credentials (read by the API server at startup) ---
export AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)"
export AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)"
export AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
export AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv)"
export AZURE_RESOURCE_GROUP=rg-genesis-demos
export AZURE_ML_WORKSPACE=genesis-mlw

export TILED_BASE_URL="https://tiled.nsls2.bnl.gov"
export TILED_API_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)"

# --- ZMQ sources for same-host replay (replay_from_tiled.py publishes here) ---
export SERVER_STREAM_SOURCE="tcp://localhost:5555"
export PANDA_STREAM_SOURCE="tcp://localhost:5556"

# --- Engine cache: /models (the container default) isn't writable outside it ---
export ENGINE_CACHE_DIR="${ENGINE_CACHE_DIR:-$HOME/.cache/holoptycho/models}"
mkdir -p "$ENGINE_CACHE_DIR"

echo "env-replay: ready."
echo "  SERVER_STREAM_SOURCE=$SERVER_STREAM_SOURCE"
echo "  PANDA_STREAM_SOURCE=$PANDA_STREAM_SOURCE"
echo "  ENGINE_CACHE_DIR=$ENGINE_CACHE_DIR"
echo "  TILED_BASE_URL=$TILED_BASE_URL"
echo "Now run: pixi run api"
