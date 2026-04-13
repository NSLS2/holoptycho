# Holoscan's PyPI wheel bundles its own UCX/GXF/RMM libs that must be on
# LD_LIBRARY_PATH. The libs live in site-packages/holoscan/lib/.
HOLOSCAN_LIB="$CONDA_PREFIX/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/holoscan/lib"
if [ -d "$HOLOSCAN_LIB" ]; then
    export LD_LIBRARY_PATH="$HOLOSCAN_LIB:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
