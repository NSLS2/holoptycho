#!/bin/bash
# ---- Holoscan GUI launcher ----
#
# Invoked by: pixi run -e holoscan launch
#
# Automatically builds the container image if it does not exist, then
# launches the ptycho development GUI (run-ptycho-development) inside
# the hxn-ptycho-holoscan container.
#
# For an interactive shell use run_container directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

image_python_ok() {
    podman run --rm hxn-ptycho-holoscan \
        bash -lc 'env -u PYTHONHOME -u PYTHONUSERBASE PYTHONNOUSERSITE=1 /pixi_env/.pixi/envs/default/bin/python -c "import encodings"' \
        >/dev/null 2>&1
}

# Build the image only when it is not already present
if ! podman image exists hxn-ptycho-holoscan 2>/dev/null; then
    echo "[launch_holoscan_gui] Image 'hxn-ptycho-holoscan' not found — building..."
    "$SCRIPT_DIR/build_container"
    if [ $? -ne 0 ]; then
        echo "[launch_holoscan_gui] ERROR: build_container failed. Aborting." >&2
        exit 1
    fi
    echo "[launch_holoscan_gui] Build complete."
else
    if image_python_ok; then
        echo "[launch_holoscan_gui] Image 'hxn-ptycho-holoscan' found — skipping build."
    else
        echo "[launch_holoscan_gui] Image 'hxn-ptycho-holoscan' found but Python runtime check failed — rebuilding..."
        "$SCRIPT_DIR/build_container"
        if [ $? -ne 0 ]; then
            echo "[launch_holoscan_gui] ERROR: build_container failed. Aborting." >&2
            exit 1
        fi
        echo "[launch_holoscan_gui] Rebuild complete."
    fi
fi

echo "[launch_holoscan_gui] Starting container and launching GUI..."


"$SCRIPT_DIR/run_container"
