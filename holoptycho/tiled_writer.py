"""Tiled writer for holoptycho reconstruction results.

Writes probe, object, ViT predictions, and associated metadata to a Tiled
catalog.  Initialized from environment variables:

    TILED_BASE_URL   — URL of the Tiled server (required)
    TILED_API_KEY    — API key (optional; falls back to the cached token from
                       ``tiled login`` for the same server, then anonymous)
    TILED_CATALOG_PATH — path within the catalog to write into
                         (default: hxn/processed/holoptycho)

A process-wide singleton is maintained so that multiple modules (ptycho_holo,
vit_inference) share a single Tiled connection.  Use :func:`get_writer` to
obtain it.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("holoptycho.tiled_writer")

_DEFAULT_CATALOG_PATH = "hxn/processed/holoptycho"
_SPECS = ["synaps_project"]
# Access tags gate which API keys can read/write this node. The holoptycho API
# key is scoped to {'synaps_project', 'hxn_beamline', 'public'}, so every
# container/array we create must carry one of these tags or Tiled returns 403.
_ACCESS_TAGS = ["synaps_project"]


def _get_or_create(container, key: str):
    """Return a sub-container by key, creating it if it doesn't exist.

    The check-then-create is racy on its own — under the multi-threaded
    holoscan scheduler, both ``MosaicWriterOp`` and ``BatchWriterOp`` may
    hit a fresh run together and both try to create ``vit/``. The loser
    sees a 409 ``ClientError``. Catch that and re-fetch by key on the
    assumption another thread won the race.
    """
    if key in container:
        return container[key]
    try:
        return container.create_container(key=key, specs=_SPECS, access_tags=_ACCESS_TAGS)
    except Exception:
        if key in container:
            return container[key]
        raise


class TiledWriter:
    """Writes holoptycho results to a Tiled catalog.

    Parameters
    ----------
    base_url:
        Tiled server URL.
    api_key:
        Tiled API key.  If ``None`` (no ``TILED_API_KEY`` set),
        ``tiled.client.from_uri`` falls back to the cached token from
        ``tiled login`` for the same server, or anonymous access if no token
        is cached.
    catalog_path:
        Slash-separated path to an **existing** container in the catalog
        (e.g. ``hxn/processed/holoptycho``).  Per-scan sub-containers are
        created beneath it as needed.
    """

    def __init__(self, base_url: str, api_key: str | None, catalog_path: str):
        from tiled.client import from_uri

        client = from_uri(base_url, api_key=api_key) if api_key else from_uri(base_url)
        # Navigate to the existing root container using plain [] indexing.
        node = client
        for part in catalog_path.strip("/").split("/"):
            node = node[part]
        self._root = node
        self._catalog_path = catalog_path
        # Per-run container created lazily by start_run().
        self._run = None
        self._run_uid: str | None = None
        # Diffraction buffer node, set by start_diffraction_buffer when
        # fine_tune writes are enabled.
        self._dp_node = None
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

    # ------------------------------------------------------------------
    # Fine-tuning writes — only used when config.fine_tune=True
    # ------------------------------------------------------------------

    def start_diffraction_buffer(
        self,
        nz: int,
        frame_shape: tuple[int, int],
        dtype=np.uint16,
    ) -> None:
        """Pre-allocate the per-run diffraction buffer for fine-tuning data.

        Creates a zero-filled ``(nz, H, W)`` array under
        ``<run>/diffraction/dp``; subsequent ``write_diffraction_chunk`` calls
        slice into it. Pre-allocating keeps the schema knowable from the start
        so the dashboard / loader can size their views without waiting for the
        first chunk to land.
        """
        if self._run is None:
            logger.warning("start_diffraction_buffer before start_run; skipping")
            return
        try:
            diffraction = _get_or_create(self._run, "diffraction")
            empty = np.zeros((nz,) + tuple(frame_shape), dtype=dtype)
            self._write_or_overwrite_array(diffraction, "dp", empty)
            self._dp_node = diffraction["dp"]
            logger.info(
                "TiledWriter.start_diffraction_buffer run=%s shape=%s dtype=%s",
                self._run_uid, empty.shape, empty.dtype,
            )
        except Exception:
            logger.exception("TiledWriter.start_diffraction_buffer failed")
            self._dp_node = None

    def write_diffraction_chunk(
        self,
        indices: np.ndarray,
        frames: np.ndarray,
    ) -> None:
        """Write a chunk of detector-frame intensity into ``<run>/diffraction/dp``.

        ``indices`` is a ``(B,)`` array of global frame indices in scan order;
        ``frames`` is ``(B, H, W)`` matching the buffer's dtype. The upstream
        batching produces contiguous indices per chunk, so we use a single
        slice assignment; non-contiguous chunks fall back to per-row writes.
        """
        if self._dp_node is None:
            return
        try:
            indices = np.asarray(indices)
            if len(indices) == 0:
                return
            start = int(indices[0])
            end = int(indices[-1]) + 1
            if end - start == len(indices) and np.all(np.diff(indices) == 1):
                self._dp_node[start:end] = frames
            else:
                for i, idx in enumerate(indices):
                    self._dp_node[int(idx) : int(idx) + 1] = frames[i : i + 1]
        except Exception:
            logger.exception("TiledWriter.write_diffraction_chunk failed")

    def write_probe_positions_m(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
    ) -> None:
        """Overwrite the per-frame probe positions in meters.

        Sibling to ``write_positions`` (which writes microns under the run
        root); this writes the SI-unit form ptycho-vit's loader expects, under
        ``<run>/diffraction/probe_position_x_m`` and ``..._y_m``. Only called
        when fine-tuning writes are enabled.
        """
        if self._run is None:
            logger.warning("write_probe_positions_m before start_run; skipping")
            return
        try:
            diffraction = _get_or_create(self._run, "diffraction")
            self._write_or_overwrite_array(diffraction, "probe_position_x_m", x_m)
            self._write_or_overwrite_array(diffraction, "probe_position_y_m", y_m)
        except Exception:
            logger.exception("TiledWriter.write_probe_positions_m failed")

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
          positions.
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
            # Per-batch history (append-only).
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

# Module-level singleton — shared by all callers within the same process.
_writer_instance: "TiledWriter | None" = None


def get_writer() -> "TiledWriter":
    """Return the process-wide writer singleton.

    Constructed lazily on first call from environment variables.
    ``TILED_BASE_URL`` is required; raises :class:`RuntimeError` if unset.
    ``TILED_API_KEY`` is optional — when absent the client falls back to the
    cached token from ``tiled login`` (or anonymous access).

    Subsequent calls return the same instance without re-reading env vars or
    re-connecting to Tiled.
    """
    global _writer_instance
    if _writer_instance is not None:
        return _writer_instance

    base_url = os.environ.get("TILED_BASE_URL", "").strip()
    api_key = os.environ.get("TILED_API_KEY", "").strip() or None
    catalog_path = os.environ.get("TILED_CATALOG_PATH", _DEFAULT_CATALOG_PATH).strip()

    if not base_url:
        raise RuntimeError("Tiled writer requires TILED_BASE_URL to be set.")

    _writer_instance = TiledWriter(base_url=base_url, api_key=api_key, catalog_path=catalog_path)
    return _writer_instance
