#!/bin/bash
#SBATCH --job-name=holoptycho
#SBATCH --error=/home/%u/logs/%x.%J.err
#SBATCH --output=/home/%u/logs/%x.%J.out
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --qos=long
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

mkdir -p "/home/$USER/logs"

export XDG_RUNTIME_DIR=/tmp/podman-run-$(id -u)
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

# Log in to ACR
podman login genesisdemosacr.azurecr.io \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$(az acr login --name genesisdemosacr --expose-token --query accessToken -o tsv)"

docker run --pull=always --gpus all -p 127.0.0.1:8000:8000 --shm-size=32g \
  -e AZURE_TENANT_ID="$(az account show --query tenantId -o tsv)" \
  -e AZURE_CLIENT_ID="$(az ad app list --display-name 'NSLS2-Genesis-Holoptycho' --query '[0].appId' -o tsv)" \
  -e AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)" \
  -e AZURE_CERTIFICATE_B64="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-sp-cert --query value -o tsv)" \
  -e AZURE_RESOURCE_GROUP=rg-genesis-demos \
  -e AZURE_ML_WORKSPACE=genesis-mlw \
  -e TILED_BASE_URL="https://tiled.nsls2.bnl.gov" \
  -e TILED_API_KEY="$(az keyvault secret show --vault-name genesisdemoskv --name holoptycho-tiled-api-key --query value -o tsv)" \
  -e SERVER_STREAM_SOURCE="tcp://localhost:5555" \
  -e PANDA_STREAM_SOURCE="tcp://localhost:5556" \
  genesisdemosacr.azurecr.io/holoptycho:latest
