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

A process-wide singleton is maintained so that multiple modules (ptycho_holo,
vit_inference) share a single Tiled connection.  Use :func:`get_writer` to
obtain it.
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
# Access tags gate which API keys can read/write this node. The holoptycho API
# key is scoped to {'synaps_project', 'hxn_beamline', 'public'}, so every
# container/array we create must carry one of these tags or Tiled returns 403.
_ACCESS_TAGS = ["synaps_project"]


def _get_or_create(container, key: str):
    """Return a sub-container by key, creating it if it doesn't exist."""
    if key in container:
        return container[key]
    return container.create_container(key=key, specs=_SPECS, access_tags=_ACCESS_TAGS)


class TiledWriter:
    """Writes holoptycho results to a Tiled catalog.

    Parameters
    ----------
    base_url:
        Tiled server URL.
    api_key:
        Tiled API key.
    catalog_path:
        Slash-separated path to an **existing** container in the catalog
        (e.g. ``hxn/processed/holoptycho``).  Per-scan sub-containers are
        created beneath it as needed.
    """

    def __init__(self, base_url: str, api_key: str, catalog_path: str):
        from tiled.client import from_uri

        client = from_uri(base_url, api_key=api_key)
        # Navigate to the existing root container using plain [] indexing.
        node = client
        for part in catalog_path.strip("/").split("/"):
            node = node[part]
        self._root = node
        self._catalog_path = catalog_path
        # Per-run container created lazily by start_run().
        self._run = None
        self._run_uid: str | None = None
        logger.info("TiledWriter connected: %s / %s", base_url, catalog_path)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_uid: str, metadata: dict | None = None) -> None:
        """Create a fresh per-run container under the catalog root.

        Each pipeline start gets its own container keyed by ``run_uid`` so
        repeated runs of the same scan don't collide. ``metadata`` is stored
        on the container and should include the raw run's uid and scan_id.
        """
        meta = dict(metadata or {})
        meta.setdefault("run_uid", run_uid)
        self._run = self._root.create_container(
            key=run_uid,
            metadata=meta,
            specs=_SPECS,
            access_tags=_ACCESS_TAGS,
        )
        self._run_uid = run_uid
        logger.info("TiledWriter.start_run uid=%s metadata=%s", run_uid, meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_or_overwrite_array(self, container, key: str, array: np.ndarray, metadata: dict | None = None):
        """Write an array into *container* under *key*, overwriting if it already exists."""
        arr = np.asarray(array)
        if key in container:
            node = container[key]
            node.write(arr)
            # `node.write()` overwrites bytes only; metadata is locked in at
            # array creation. Patch it so callers (e.g. write_live) can keep a
            # field like `iteration` in sync with each fresh snapshot.
            if metadata:
                node.update_metadata(metadata=metadata)
        else:
            container.write_array(
                arr,
                key=key,
                metadata=metadata or {},
                specs=_SPECS,
                access_tags=_ACCESS_TAGS,
            )

    # ------------------------------------------------------------------
    # Public write methods — all require start_run() to have been called.
    # ------------------------------------------------------------------

    def write_live(self, iteration: int, probe: np.ndarray, obj: np.ndarray) -> None:
        """Overwrite the live probe/object snapshots for the current run.

        Called every ``display_interval`` iterations.

        If either array contains non-finite values (NaN/Inf), the write is
        skipped so the previous finite snapshot stays visible to consumers
        (synaps-dash, etc.). The iterative engine can transiently diverge —
        writing the corrupted state would replace a useful image with a black
        one. The next finite iteration will overwrite as usual.
        """
        if self._run is None:
            logger.warning("write_live called before start_run; skipping")
            return
        if not np.isfinite(probe).all() or not np.isfinite(obj).all():
            logger.warning(
                "write_live skipped: non-finite values in probe/object at iter=%d "
                "(keeping last finite snapshot)",
                iteration,
            )
            return
        try:
            live = _get_or_create(self._run, "live")
            meta = {"iteration": iteration}
            self._write_or_overwrite_array(live, "probe", probe, metadata=meta)
            self._write_or_overwrite_array(live, "object", obj, metadata=meta)
            logger.info("write_live run=%s iter=%d", self._run_uid, iteration)
        except Exception:
            logger.exception("TiledWriter.write_live failed")

    def write_final(
        self,
        probe: np.ndarray,
        obj: np.ndarray,
        timestamps: np.ndarray,
        num_points: np.ndarray,
    ) -> None:
        """Write final reconstruction results when a scan completes."""
        if self._run is None:
            logger.warning("write_final called before start_run; skipping")
            return
        try:
            final = _get_or_create(self._run, "final")
            self._write_or_overwrite_array(final, "probe", probe)
            self._write_or_overwrite_array(final, "object", obj)
            self._write_or_overwrite_array(final, "timestamps", timestamps)
            self._write_or_overwrite_array(final, "num_points", num_points)
            logger.info("write_final run=%s", self._run_uid)
        except Exception:
            logger.exception("TiledWriter.write_final failed")

    def write_positions(self, positions_um: np.ndarray) -> None:
        """Overwrite the per-frame scan positions for the current run.

        ``positions_um`` is shape ``(nz, 2)`` in microns; rows for frames the
        PandA stream hasn't reached yet are NaN. The dashboard's mosaic
        stitcher reads this array to place each ViT patch at its real scan
        position rather than a deterministic raster (which was wrong for snake
        scans / motor jitter).
        """
        if self._run is None:
            logger.warning("write_positions called before start_run; skipping")
            return
        try:
            self._write_or_overwrite_array(self._run, "positions_um", positions_um)
        except Exception:
            logger.exception("TiledWriter.write_positions failed")

    def write_vit(
        self,
        batch_num: int,
        pred: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """Write a ViT inference batch result.

        Writes both:

        * ``vit/pred_latest`` and ``vit/indices_latest`` — overwritten each batch
          for cheap live polling (single small fetch).
        * ``vit/batches/{batch_num:06d}/{pred,indices}`` — append-only per-batch
          history. Required so downstream consumers (synaps-dash, offline
          analysis) can stitch the per-frame ViT predictions across all scan
          positions, mirroring how ``live_compare_viewer.py`` walks
          ``vit_batch_NNNNNN_pred.npy`` files in the npy fallback backend.
        """
        if self._run is None:
            logger.warning("write_vit called before start_run; skipping")
            return
        try:
            vit = _get_or_create(self._run, "vit")
            meta = {"batch_num": batch_num}
            # Live-polling mirrors (overwritten each batch).
            self._write_or_overwrite_array(vit, "pred_latest", pred, metadata=meta)
            self._write_or_overwrite_array(vit, "indices_latest", indices, metadata=meta)
            # Per-batch history (append-only, matches npy fallback layout).
            batches = _get_or_create(vit, "batches")
            batch_container = _get_or_create(batches, f"{batch_num:06d}")
            self._write_or_overwrite_array(batch_container, "pred", pred, metadata=meta)
            self._write_or_overwrite_array(batch_container, "indices", indices, metadata=meta)
            logger.info("write_vit run=%s batch=%d", self._run_uid, batch_num)
        except Exception:
            logger.exception("TiledWriter.write_vit failed")

    def write_vit_mosaic(
        self,
        mosaic: np.ndarray,
        *,
        batch_num: int,
        pixel_size_m: float,
        canvas_origin_um: tuple[float, float],
    ) -> None:
        """Overwrite the server-side ViT mosaic for the current run.

        Written under ``<run>/vit/mosaic`` so the dashboard can render it with
        the same TiledImageTile path used for the iterative live object,
        rather than re-stitching per-batch in the browser.
        """
        if self._run is None:
            logger.warning("write_vit_mosaic called before start_run; skipping")
            return
        try:
            vit = _get_or_create(self._run, "vit")
            meta = {
                "batch_num": batch_num,
                "pixel_size_m": float(pixel_size_m),
                "canvas_origin_um": [float(canvas_origin_um[0]), float(canvas_origin_um[1])],
            }
            self._write_or_overwrite_array(vit, "mosaic", mosaic, metadata=meta)
        except Exception:
            logger.exception("TiledWriter.write_vit_mosaic failed")

    def delete_vit_mosaic(self) -> None:
        """Delete the ``<run>/vit/mosaic`` array node, if present.

        Used by ``SaveViTResult`` when the canvas grows: tiled array nodes
        have a fixed shape, so a write with a different shape would 422.
        Recreate-on-next-write by deleting the existing node here.
        """
        if self._run is None:
            return
        try:
            vit = self._run.get("vit") if "vit" in self._run else None
            if vit is None or "mosaic" not in vit:
                return
            del vit["mosaic"]
            logger.info("delete_vit_mosaic: dropped existing mosaic node for shape change")
        except Exception:
            logger.exception("TiledWriter.delete_vit_mosaic failed")


class _NpyFallbackWriter:
    """Writes results to .npy files under /data/users/Holoscan/<run_uid>/ —
    matches the behaviour before tiled integration was added, but per-run."""

    _BASE = "/data/users/Holoscan"

    def __init__(self):
        self._run_dir: str | None = None

    def start_run(self, run_uid: str, metadata: dict | None = None) -> None:
        self._run_dir = f"{self._BASE}/{run_uid}"
        os.makedirs(self._run_dir, exist_ok=True)
        if metadata:
            try:
                import json
                with open(f"{self._run_dir}/metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
            except Exception:
                logger.exception("_NpyFallbackWriter: failed to write metadata.json")

    def write_live(self, iteration: int, probe: np.ndarray, obj: np.ndarray) -> None:
        if self._run_dir is None:
            logger.warning("write_live called before start_run; skipping")
            return
        if not np.isfinite(probe).all() or not np.isfinite(obj).all():
            logger.warning(
                "write_live skipped: non-finite values in probe/object at iter=%d "
                "(keeping last finite snapshot)",
                iteration,
            )
            return
        try:
            np.save(f"{self._run_dir}/prb_live.npy", probe)
            np.save(f"{self._run_dir}/obj_live.npy", obj)
            with open(f"{self._run_dir}/iteration", "w") as f:
                f.write("%d\n" % iteration)
        except Exception:
            logger.exception("_NpyFallbackWriter.write_live failed")

    def write_final(self, probe, obj, timestamps, num_points) -> None:
        if self._run_dir is None:
            logger.warning("write_final called before start_run; skipping")
            return
        try:
            np.save(f"{self._run_dir}/probe.npy", probe)
            np.save(f"{self._run_dir}/object.npy", obj)
            np.save(f"{self._run_dir}/timestamp_iter.npy", timestamps)
            np.save(f"{self._run_dir}/num_points_recv_iter.npy", num_points)
        except Exception:
            logger.exception("_NpyFallbackWriter.write_final failed")

    def write_vit(self, batch_num: int, pred: np.ndarray, indices: np.ndarray) -> None:
        if self._run_dir is None:
            logger.warning("write_vit called before start_run; skipping")
            return
        try:
            np.save(f"{self._run_dir}/vit_batch_{batch_num:06d}_pred.npy", pred)
            np.save(f"{self._run_dir}/vit_batch_{batch_num:06d}_indices.npy", indices)
            tmp = f"{self._run_dir}/_vit_pred_latest.tmp.npy"
            np.save(tmp, pred)
            os.replace(tmp, f"{self._run_dir}/vit_pred_latest.npy")
        except Exception:
            logger.exception("_NpyFallbackWriter.write_vit failed")

    def write_positions(self, positions_um: np.ndarray) -> None:
        if self._run_dir is None:
            logger.warning("write_positions called before start_run; skipping")
            return
        try:
            tmp = f"{self._run_dir}/_positions_um.tmp.npy"
            np.save(tmp, positions_um)
            os.replace(tmp, f"{self._run_dir}/positions_um.npy")
        except Exception:
            logger.exception("_NpyFallbackWriter.write_positions failed")

    def write_vit_mosaic(
        self,
        mosaic: np.ndarray,
        *,
        batch_num: int,
        pixel_size_m: float,
        canvas_origin_um: tuple[float, float],
    ) -> None:
        if self._run_dir is None:
            logger.warning("write_vit_mosaic called before start_run; skipping")
            return
        try:
            tmp = f"{self._run_dir}/_vit_mosaic.tmp.npy"
            np.save(tmp, mosaic)
            os.replace(tmp, f"{self._run_dir}/vit_mosaic.npy")
        except Exception:
            logger.exception("_NpyFallbackWriter.write_vit_mosaic failed")

    def delete_vit_mosaic(self) -> None:
        """Drop the .npy mosaic so the next write replaces it cleanly."""
        if self._run_dir is None:
            return
        try:
            path = f"{self._run_dir}/vit_mosaic.npy"
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.exception("_NpyFallbackWriter.delete_vit_mosaic failed")


# Module-level singleton — shared by all callers within the same process.
_writer_instance: "TiledWriter | _NpyFallbackWriter | None" = None


def get_writer() -> "TiledWriter | _NpyFallbackWriter":
    """Return the process-wide writer singleton.

    Constructed lazily on first call from environment variables:

    * ``TILED_BASE_URL`` and ``TILED_API_KEY`` both set → :class:`TiledWriter`
    * Otherwise → :class:`_NpyFallbackWriter` (with a warning)

    Subsequent calls return the same instance without re-reading env vars or
    re-connecting to Tiled.
    """
    global _writer_instance
    if _writer_instance is not None:
        return _writer_instance

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
        _writer_instance = _NpyFallbackWriter()
    else:
        _writer_instance = TiledWriter(base_url=base_url, api_key=api_key, catalog_path=catalog_path)

    return _writer_instance
