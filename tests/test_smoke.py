"""Smoke tests for the holoptycho package.

Verify that every module can be imported with its declared dependencies.
Modules requiring GPU (cupy/numba) or the Holoscan SDK are gated with
``pytest.importorskip`` so the suite still passes in a plain CI
environment.
"""
import importlib

import pytest


# Pure Python modules (no GPU, no holoscan at import time)
PURE_MODULES = [
    "holoptycho",
    "holoptycho.liverecon_utils",
]

# Modules that import holoscan + cupy at module level
HOLOSCAN_MODULES = [
    "holoptycho.datasource",
    "holoptycho.preprocess",
    "holoptycho.ptycho_holo",
    "holoptycho.live_simulation",
    "holoptycho.vit_inference",
]

# Modules that only need cupy/numba (no holoscan at import time)
GPU_MODULES = [
    "holoptycho.streaming_recon",
]


@pytest.mark.parametrize("module_name", PURE_MODULES)
def test_pure_module_imports(module_name):
    """Each pure-Python module imports cleanly."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", HOLOSCAN_MODULES)
def test_holoscan_module_imports(module_name):
    """Each Holoscan-dependent module imports when holoscan + cupy are available."""
    pytest.importorskip("holoscan")
    pytest.importorskip("cupy")
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", GPU_MODULES)
def test_gpu_module_imports(module_name):
    """Each GPU-dependent module imports when cupy/numba are available."""
    pytest.importorskip("cupy")
    importlib.import_module(module_name)
