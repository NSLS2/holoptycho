FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y curl vim
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash

RUN apt-get install -y wget
RUN cd /tmp && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/nsight-systems-2025.1.1_2025.1.1.103-1_amd64.deb && \
    apt-get install -y --fix-missing ./nsight-systems-2025.1.1_2025.1.1.103-1_amd64.deb && \
    rm -rf /tmp/*

# ---- Bake pixi environment into /pixi_env/ (NOT /podman_dir/) ----
# Why /pixi_env/? The old path /podman_dir/ gets clobbered by the
# run_container bind-mount `-v ./podman_dir:/podman_dir`, which wipes
# the pre-installed .pixi/ directory and forces pixi to re-resolve
# packages from the internet — breaking offline machines (axinite,
# amazonite, NSLS-II).  /pixi_env/ is never bind-mounted.
COPY pixi.toml /pixi_env/pixi.toml
COPY pixi.lock /pixi_env/pixi.lock
WORKDIR /pixi_env

# Install pixi deps (postinstall task runs at container start, not here)
RUN pixi install