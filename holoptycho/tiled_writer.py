"""Tiled writer for holoptycho reconstruction results.

Writes probe, object, ViT predictions, and associated metadata to a Tiled
catalog.  Initialized from environment variables:

    TILED_BASE_URL   — URL of the Tiled server (required)
    TILED_API_KEY    — API key for authentication (required)
    TILED_CATALOG_PATH — path within the catalog to write into
                         (default: hxn/processed/holoptycho)

If either TILED_BASE_URL or TILED_API_KEY is absent, all write methods fall
back to writing .npy files under /data/users/Holoscan/ (matching the prior
behaviour) and emit a warning.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger("holoptycho.tiled_writer")

_DEFAULT_CATALOG_PATH = "hxn/processed/holoptycho"
_SPECS = ["synaps_project"]


def _navigate(client, path: str):
    """Walk a slash-separated path in the tiled catalog, creating containers
    as needed at each level."""
    node = client
    for part in path.strip("/").split("/"):
        if part in node:
            node = node[part]
        else:
            node = node.create_container(key=part, specs=_SPECS)
    return node


class TiledWriter:
    """Writes holoptycho results to a Tiled catalog.

    Parameters
    ----------
    base_url:
        Tiled server URL.
    api_key:
        Tiled API key.
    catalog_path:
        Slash-separated path within the catalog (e.g. ``hxn/processed/holoptycho``).
        A sub-container named after the scan number is created beneath this path.
    """

    def __init__(self, base_url: str, api_key: str, catalog_path: str):
        from tiled.client import from_uri

        self._client = from_uri(base_url, api_key=api_key)
        self._root = _navigate(self._client, catalog_path)
        self._catalog_path = catalog_path
        logger.info("TiledWriter connected: %s / %s", base_url, catalog_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_container(self, scan_num) -> object:
        """Return (creating if necessary) the container for a given scan."""
        key = str(scan_num)
        if key in self._root:
            return self._root[key]
        return self._root.create_container(key=key, specs=_SPECS)

    def _write_or_overwrite_array(self, container, key: str, array: np.ndarray, metadata: dict | None = None):
        """Write an array into *container* under *key*, overwriting if it already exists."""
        arr = np.asarray(array)
        if key in container:
            container[key].write(arr)
        else:
            container.write_array(arr, key=key, metadata=metadata or {}, specs=_SPECS)

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def write_live(self, scan_num, iteration: int, probe: np.ndarray, obj: np.ndarray) -> None:
        """Overwrite the live probe/object snapshots for the current scan.

        Called every ``display_interval`` iterations.
        """
        try:
            scan = self._scan_container(scan_num)
            live = _navigate(scan, "live")
            meta = {"scan_num": str(scan_num), "iteration": iteration}
            self._write_or_overwrite_array(live, "probe", probe, metadata=meta)
            self._write_or_overwrite_array(live, "object", obj, metadata=meta)
            logger.debug("write_live scan=%s iter=%d", scan_num, iteration)
        except Exception:
            logger.exception("TiledWriter.write_live failed")

    def write_final(
        self,
        scan_num,
        probe: np.ndarray,
        obj: np.ndarray,
        timestamps: np.ndarray,
        num_points: np.ndarray,
    ) -> None:
        """Write final reconstruction results when a scan completes."""
        try:
            scan = self._scan_container(scan_num)
            final = _navigate(scan, "final")
            meta = {"scan_num": str(scan_num)}
            self._write_or_overwrite_array(final, "probe", probe, metadata=meta)
            self._write_or_overwrite_array(final, "object", obj, metadata=meta)
            self._write_or_overwrite_array(final, "timestamps", timestamps, metadata=meta)
            self._write_or_overwrite_array(final, "num_points", num_points, metadata=meta)
            logger.info("write_final scan=%s", scan_num)
        except Exception:
            logger.exception("TiledWriter.write_final failed")

    def write_vit(
        self,
        scan_num,
        batch_num: int,
        pred: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """Write a ViT inference batch result."""
        try:
            scan = self._scan_container(scan_num)
            vit = _navigate(scan, "vit")
            meta = {"scan_num": str(scan_num), "batch_num": batch_num}
            # Overwrite "latest" arrays for live viewing
            self._write_or_overwrite_array(vit, "pred_latest", pred, metadata=meta)
            self._write_or_overwrite_array(vit, "indices_latest", indices, metadata=meta)
            logger.debug("write_vit scan=%s batch=%d", scan_num, batch_num)
        except Exception:
            logger.exception("TiledWriter.write_vit failed")


def _npy_fallback_writer() -> "_NpyFallbackWriter":
    """Return a writer that replicates the old .npy behaviour."""
    return _NpyFallbackWriter()


class _NpyFallbackWriter:
    """Writes results to .npy files under /data/users/Holoscan/ — matches the
    behaviour before tiled integration was added."""

    _BASE = "/data/users/Holoscan"

    def write_live(self, scan_num, iteration: int, probe: np.ndarray, obj: np.ndarray) -> None:
        try:
            np.save(f"{self._BASE}/prb_live.npy", probe)
            np.save(f"{self._BASE}/obj_live.npy", obj)
            with open(f"{self._BASE}/iteration", "w") as f:
                f.write("%d\n" % iteration)
        except Exception:
            logger.exception("_NpyFallbackWriter.write_live failed")

    def write_final(self, scan_num, probe, obj, timestamps, num_points) -> None:
        try:
            save_dir = f"{self._BASE}/recon_{scan_num}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(f"{save_dir}/probe.npy", probe)
            np.save(f"{save_dir}/object.npy", obj)
            np.save(f"{save_dir}/timestamp_iter.npy", timestamps)
            np.save(f"{save_dir}/num_points_recv_iter.npy", num_points)
        except Exception:
            logger.exception("_NpyFallbackWriter.write_final failed")

    def write_vit(self, scan_num, batch_num: int, pred: np.ndarray, indices: np.ndarray) -> None:
        try:
            save_dir = self._BASE
            np.save(f"{save_dir}/vit_batch_{batch_num:06d}_pred.npy", pred)
            np.save(f"{save_dir}/vit_batch_{batch_num:06d}_indices.npy", indices)
            tmp = f"{save_dir}/_vit_pred_latest.tmp.npy"
            np.save(tmp, pred)
            os.replace(tmp, f"{save_dir}/vit_pred_latest.npy")
        except Exception:
            logger.exception("_NpyFallbackWriter.write_vit failed")


def get_writer() -> "TiledWriter | _NpyFallbackWriter":
    """Construct and return the appropriate writer based on environment variables.

    If TILED_BASE_URL and TILED_API_KEY are both set, returns a TiledWriter.
    Otherwise warns and returns an _NpyFallbackWriter.
    """
    base_url = os.environ.get("TILED_BASE_URL", "").strip()
    api_key = os.environ.get("TILED_API_KEY", "").strip()
    catalog_path = os.environ.get("TILED_CATALOG_PATH", _DEFAULT_CATALOG_PATH).strip()

    if not base_url or not api_key:
        missing = []
        if not base_url:
            missing.append("TILED_BASE_URL")
        if not api_key:
            missing.append("TILED_API_KEY")
        warnings.warn(
            f"Tiled not configured ({', '.join(missing)} not set). "
            "Falling back to writing .npy files under /data/users/Holoscan/.",
            stacklevel=2,
        )
        return _NpyFallbackWriter()

    return TiledWriter(base_url=base_url, api_key=api_key, catalog_path=catalog_path)
