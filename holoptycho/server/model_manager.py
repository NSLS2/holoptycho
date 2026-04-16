from __future__ import annotations

"""Azure ML model pull + TensorRT compilation + sentinel file write.

Flow triggered by POST /model:
  1. Pull ONNX from Azure ML registry (azure-ai-ml + AzureCliCredential)
  2. Compile to .engine via trtexec (skip if .engine newer than .onnx)
  3. Write reload_engine.txt sentinel — PtychoViTInferenceOp picks it up
     on its next compute() tick without any pipeline restart.
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("holoptycho.model_manager")

# Directory where compiled .engine files are cached on this machine.
ENGINE_CACHE_DIR = os.environ.get("ENGINE_CACHE_DIR", "/models")

# trtexec binary (must be on PATH or set this env var).
TRTEXEC = os.environ.get("TRTEXEC", "trtexec")


def _pull_onnx(model_name: str, version: str, dest_dir: Path) -> Path:
    """Download the ONNX file from Azure ML and return its local path."""
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    workspace_name = os.environ["AZURE_ML_WORKSPACE"]

    logger.info(
        "Connecting to Azure ML workspace %s/%s/%s",
        subscription_id,
        resource_group,
        workspace_name,
    )
    client = MLClient(
        AzureCliCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    model = client.models.get(name=model_name, version=version)
    logger.info("Downloading model %s v%s from %s", model_name, version, model.path)

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = client.models.download(
        name=model_name, version=version, download_path=str(dest_dir)
    )
    # The SDK downloads into a subdirectory; find the .onnx file.
    onnx_files = list(Path(local_path).rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in downloaded model at {local_path}")
    return onnx_files[0]


def _compile_engine(onnx_path: Path, engine_path: Path) -> None:
    """Compile onnx_path → engine_path via trtexec (skip if up to date)."""
    if (
        engine_path.exists()
        and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime
    ):
        logger.info("Engine %s is up to date — skipping compilation", engine_path)
        return

    logger.info("Compiling %s → %s", onnx_path, engine_path)
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit {result.returncode}):\n{result.stderr}"
        )
    logger.info("Compilation complete: %s", engine_path)


def _write_sentinel(engine_path: Path) -> None:
    """Write the reload_engine.txt sentinel that PtychoViTInferenceOp watches."""
    sentinel = engine_path.parent / "reload_engine.txt"
    sentinel.write_text(str(engine_path))
    logger.info("Sentinel written: %s", sentinel)


def swap_model(model_name: str, version: str, state) -> None:
    """Full model swap: pull ONNX → compile → write sentinel.

    Intended to run in a background thread. Updates state.model_status
    throughout so callers can poll GET /model/status.
    """
    try:
        state.update(model_status="downloading", model_error=None)
        cache_dir = Path(ENGINE_CACHE_DIR)
        onnx_dir = cache_dir / "onnx" / model_name / version
        onnx_path = _pull_onnx(model_name, version, onnx_dir)

        state.update(model_status="compiling")
        engine_path = cache_dir / f"{model_name}_v{version}.engine"
        _compile_engine(onnx_path, engine_path)

        state.update(model_status="loading")
        _write_sentinel(engine_path)

        state.update(
            model_status="ready",
            current_engine_path=str(engine_path),
            current_model_name=model_name,
            current_model_version=version,
            model_error=None,
        )
        logger.info("Model swap complete: %s v%s", model_name, version)

    except Exception as exc:
        logger.exception("Model swap failed")
        state.update(model_status="error", model_error=str(exc))


def list_models() -> list[dict]:
    """Return available models from the Azure ML registry."""
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    client = MLClient(
        AzureCliCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )
    return [
        {"name": m.name, "version": m.version, "description": getattr(m, "description", None)}
        for m in client.models.list()
    ]
