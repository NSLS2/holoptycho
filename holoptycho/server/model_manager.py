from __future__ import annotations

"""Model management: local engine cache + optional Azure ML registry.

Local cache
-----------
Engine files (.engine) are placed in ENGINE_CACHE_DIR (default /models) either
manually or by a previous pull+compile. Naming convention for Azure-sourced
engines: ``{model_name}_v{version}.engine``.

Model selection flow (swap_model)
----------------------------------
1. Check whether the requested .engine already exists in ENGINE_CACHE_DIR.
2. If yes → update state.current_engine_path immediately (no network needed).
3. If no → pull ONNX from Azure ML, compile via trtexec, then update state.

The new engine takes effect the next time the Holoscan app is started via
POST /run or POST /restart.

list_models
-----------
Returns a combined list of:
- Local .engine files found in ENGINE_CACHE_DIR
- Models registered in Azure ML (if Azure env vars are configured)
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("holoptycho.model_manager")

ENGINE_CACHE_DIR = os.environ.get("ENGINE_CACHE_DIR", "/models")
TRTEXEC = os.environ.get("TRTEXEC", "trtexec")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_credential():
    """Return CertificateCredential if AZURE_CERTIFICATE_B64 is set, else AzureCliCredential.

    AZURE_CERTIFICATE_B64 must be a base64-encoded PEM containing both the
    private key and the certificate (as exported from Key Vault secrets).
    The private key is kept in memory only — never written to disk.
    """
    cert_b64 = os.environ.get("AZURE_CERTIFICATE_B64")
    if cert_b64:
        import base64
        from azure.identity import CertificateCredential
        return CertificateCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            certificate_data=base64.b64decode(cert_b64),
        )
    from azure.identity import AzureCliCredential
    return AzureCliCredential()

def _engine_filename(model_name: str, version: str) -> str:
    return f"{model_name}_v{version}.engine"


def _engine_path(model_name: str, version: str) -> Path:
    return Path(ENGINE_CACHE_DIR) / _engine_filename(model_name, version)


def _pull_onnx(model_name: str, version: str, dest_dir: Path) -> Path:
    """Download the ONNX file from Azure ML and return its local path."""
    from azure.ai.ml import MLClient

    client = MLClient(
        _get_credential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )
    model = client.models.get(name=model_name, version=version)
    logger.info("Downloading model %s v%s from %s", model_name, version, model.path)

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = client.models.download(
        name=model_name, version=version, download_path=str(dest_dir)
    )
    onnx_files = list(Path(local_path).rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file in downloaded model at {local_path}")
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
    cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit {result.returncode}):\n{result.stderr}"
        )
    logger.info("Compilation complete: %s", engine_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_local_engines() -> list[dict]:
    """Return all .engine files found in ENGINE_CACHE_DIR."""
    cache = Path(ENGINE_CACHE_DIR)
    if not cache.exists():
        return []
    return [
        {"filename": p.name, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 1)}
        for p in sorted(cache.glob("*.engine"))
    ]


def _azure_available() -> bool:
    base = all(
        os.environ.get(v)
        for v in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE")
    )
    if not base:
        return False
    # If cert auth is requested, also require tenant and client ID
    if os.environ.get("AZURE_CERTIFICATE_B64"):
        return all(os.environ.get(v) for v in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID"))
    return True


def list_azure_models() -> list[dict]:
    """Return models from Azure ML registry. Returns [] if Azure is not configured."""
    if not _azure_available():
        return []
    from azure.ai.ml import MLClient

    client = MLClient(
        _get_credential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )
    return [
        {
            "name": m.name,
            "version": m.version,
            "description": getattr(m, "description", None),
        }
        for m in client.models.list()
    ]


def list_models() -> dict:
    """Return combined local + Azure ML model inventory.

    Each Azure model entry gets a ``cached`` flag indicating whether the
    compiled .engine is already present locally.
    """
    local = list_local_engines()
    local_filenames = {e["filename"] for e in local}

    azure = list_azure_models()
    for m in azure:
        m["cached"] = _engine_filename(m["name"], m["version"]) in local_filenames

    return {"local": local, "azure": azure, "azure_available": _azure_available()}


def swap_model(model_name: str, version: str, state) -> None:
    """Select a model engine for the next pipeline run.

    If the compiled .engine is already in ENGINE_CACHE_DIR, state is updated
    immediately. Otherwise the ONNX is pulled from Azure ML and compiled first.
    The new engine takes effect the next time the Holoscan app is started via
    POST /run or POST /restart.
    """
    try:
        target = _engine_path(model_name, version)

        if target.exists():
            logger.info("Engine already cached: %s", target)
            state.update(model_status="loading", model_error=None)
        else:
            if not _azure_available():
                raise RuntimeError(
                    f"Engine {target.name} not found locally and Azure ML is not "
                    "configured (set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, "
                    "AZURE_ML_WORKSPACE)."
                )
            state.update(model_status="downloading", model_error=None)
            onnx_dir = Path(ENGINE_CACHE_DIR) / "onnx" / model_name / version
            onnx_path = _pull_onnx(model_name, version, onnx_dir)

            state.update(model_status="compiling")
            target.parent.mkdir(parents=True, exist_ok=True)
            _compile_engine(onnx_path, target)

            state.update(model_status="loading")

        state.update(
            model_status="ready",
            current_engine_path=str(target),
            current_model_name=model_name,
            current_model_version=version,
            model_error=None,
        )
        logger.info(
            "Model selected: %s v%s — will take effect on next start", model_name, version
        )

    except Exception as exc:
        logger.exception("Model swap failed")
        state.update(model_status="error", model_error=str(exc))
