FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y curl vim
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash

RUN apt-get install -y wget
RUN cd /tmp && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/nsight-systems-2025.1.1_2025.1.1.103-1_amd64.deb && \
    apt-get install -y --fix-missing ./nsight-systems-2025.1.1_2025.1.1.103-1_amd64.deb && \
    rm -rf /tmp/*      