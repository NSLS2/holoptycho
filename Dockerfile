# syntax=docker/dockerfile:1
# Multi-stage build: pixi deps in build stage, lean runtime image.
# CUDA 12.8.1 for native Blackwell (RTX 50xx) support; backward
# compatible with Volta, Turing, Ampere, Ada.

# ---------- Build stage ----------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates openssh-client git && \
    rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash

WORKDIR /app

# Copy dependency files and source (editable install needs the package dir)
COPY pixi.toml pixi.lock pyproject.toml set-ld-path.sh ./
COPY holoptycho/ ./holoptycho/

# SSH forwarding for private ptycho repo during pixi install.
# Requires: docker build --ssh default ...
RUN mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh pixi install --locked && \
    rm -rf ~/.cache/rattler

# Generate shell activation hook (no pixi binary needed at runtime)
RUN pixi shell-hook -s bash > /shell-hook.sh

# ---------- Runtime stage ----------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pixi environment from build stage (path must match)
COPY --from=build /app/.pixi /app/.pixi

# Copy app source and activation scripts
COPY --from=build /app/holoptycho ./holoptycho
COPY --from=build /app/pyproject.toml ./pyproject.toml
COPY --from=build /app/set-ld-path.sh ./set-ld-path.sh
COPY --from=build /shell-hook.sh /shell-hook.sh

# Build entrypoint that activates the pixi env
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    cat /shell-hook.sh >> /app/entrypoint.sh && \
    echo 'source /app/set-ld-path.sh' >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Holoscan runtime requirements
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "holoptycho"]
