# Holoscan bundles its own UCX/GXF libs that must take precedence over
# conda-forge's versions. Find the holoscan lib directory without importing it.
HOLOSCAN_LIB=$(find "$CONDA_PREFIX/lib" -path "*/holoscan/lib" -type d 2>/dev/null | head -1)
if [ -n "$HOLOSCAN_LIB" ]; then
    export LD_LIBRARY_PATH="$HOLOSCAN_LIB:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
