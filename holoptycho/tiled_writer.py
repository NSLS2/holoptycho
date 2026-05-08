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
        self._dp_chunk_size = 0
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
        dtype=np.uint8,
        frames_per_chunk: int = 64,
    ) -> None:
        """Register the per-run diffraction buffer structure for fine-tuning data.

        Registers a ``(nz, H, W)`` array under ``<run>/diffraction/dp`` *without*
        uploading any data. Tiled records the array shape, dtype, and chunking
        immediately; chunks are populated lazily by ``write_diffraction_chunk``
        as frames arrive. The register-then-write-blocks split is essential —
        the alternative ``write_array(np.zeros(...))`` path would upload the
        whole zero buffer up front, which times out over WAN.

        ``dp`` stores **amplitude** (= ``sqrt(intensity)``, rounded to uint8)
        rather than raw uint16 intensity. The Eiger detector is 16-bit but
        ``sqrt(65535) ≈ 256``, so the full intensity range maps cleanly into
        uint8 — and the 1-count quantization that introduces is well below
        the Poisson noise floor for any pixel with non-trivial counts. This
        halves the on-the-wire write volume vs uint16 (640 MB → 320 MB for
        a 10K frame 256×256 scan) since Tiled's ``write_block`` does not
        accept compressed payloads. ptycho-vit's Tiled-backed loader expects
        this amplitude form and skips its own sqrt step on this data path.

        Chunking is set so each block holds ``frames_per_chunk`` whole frames
        (the natural batch boundary from ``ImageBatchOp``). Each
        ``write_diffraction_chunk`` call writes exactly one block.
        """
        if self._run is None:
            logger.warning("start_diffraction_buffer before start_run; skipping")
            return
        try:
            from tiled.structures.array import ArrayStructure, BuiltinDtype
            from tiled.structures.core import StructureFamily
            from tiled.structures.data_source import DataSource

            diffraction = _get_or_create(self._run, "diffraction")
            h, w = int(frame_shape[0]), int(frame_shape[1])
            dtype_np = np.dtype(dtype)

            # Frame chunking matches ImageBatchOp's batch size so each
            # FrameWriterOp invocation maps to exactly one block_id.
            n_full = nz // frames_per_chunk
            rem = nz - n_full * frames_per_chunk
            chunk_dim0 = (frames_per_chunk,) * n_full
            if rem:
                chunk_dim0 = chunk_dim0 + (rem,)
            structure = ArrayStructure(
                shape=(int(nz), h, w),
                chunks=(chunk_dim0, (h,), (w,)),
                data_type=BuiltinDtype.from_numpy_dtype(dtype_np),
            )
            data_source = DataSource(
                structure=structure,
                structure_family=StructureFamily.array,
            )
            # Replace any prior dp node so re-runs of the same run_uid
            # (shouldn't happen — fresh uuid each compose — but defensive)
            # don't fight an existing array.
            if "dp" in diffraction:
                del diffraction["dp"]
            self._dp_node = diffraction.new(
                StructureFamily.array,
                [data_source],
                key="dp",
                specs=_SPECS,
                access_tags=_ACCESS_TAGS,
            )
            self._dp_chunk_size = int(frames_per_chunk)
            logger.info(
                "TiledWriter.start_diffraction_buffer run=%s shape=(%d,%d,%d) "
                "dtype=%s chunks=%d frames",
                self._run_uid, nz, h, w, dtype_np, frames_per_chunk,
            )
        except Exception:
            logger.exception("TiledWriter.start_diffraction_buffer failed")
            self._dp_node = None
            self._dp_chunk_size = 0

    def write_diffraction_chunk(
        self,
        indices: np.ndarray,
        frames: np.ndarray,
    ) -> None:
        """Write a chunk of detector-frame intensity into ``<run>/diffraction/dp``.

        ``indices`` is a ``(B,)`` array of global frame indices in scan order;
        ``frames`` is ``(B, H, W)`` of intensity (any uint dtype is accepted
        — upstream ``ImageBatchOp`` allocates uint32 for arithmetic headroom).
        Frames are sqrt'd and cast to uint8 before write to halve the wire
        volume vs uint16; see ``start_diffraction_buffer`` for justification.

        We write one block per call. Misaligned or non-contiguous chunks are
        skipped with a warning — block writes require batch boundaries to
        match the chunk boundaries declared at registration.
        """
        if self._dp_node is None:
            return
        try:
            indices = np.asarray(indices)
            if len(indices) == 0:
                return
            start = int(indices[0])
            n = len(indices)
            chunk_size = getattr(self, "_dp_chunk_size", 0) or n
            if start % chunk_size != 0:
                logger.warning(
                    "write_diffraction_chunk: misaligned start=%d "
                    "(chunk_size=%d); skipping",
                    start, chunk_size,
                )
                return
            if not np.all(np.diff(indices) == 1):
                logger.warning(
                    "write_diffraction_chunk: non-contiguous indices "
                    "[%d..%d]; skipping",
                    start, int(indices[-1]),
                )
                return
            block_id = (start // chunk_size, 0, 0)
            # Convert intensity → amplitude → uint8 to match the declared
            # dtype at start_diffraction_buffer and halve the wire volume vs
            # uint16. Upstream emits uint32 with values in the uint16 detector
            # range; sqrt(65535) ≈ 256 fits cleanly in uint8 (we just clip
            # the very rare overflow). The 1-count quantization is well below
            # the Poisson noise floor for ML training.
            intensity = np.asarray(frames)
            amp = np.sqrt(intensity, dtype=np.float32)
            np.clip(amp, 0, 255, out=amp)
            amp_u8 = amp.astype(np.uint8, copy=False)
            self._dp_node.write_block(np.ascontiguousarray(amp_u8), block=block_id)
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

    def mark_run_complete(self) -> None:
        """Stamp ``complete: true`` into the per-run container metadata.

        Called when the holoscan pipeline naturally finishes processing the
        scan — either at iterative end-of-run (``SaveResult``) for
        iterative/both modes, or at clean subprocess exit for any mode.
        Lets downstream consumers query for finalised runs without having
        to inspect subcontainers.
        """
        if self._run is None:
            logger.warning("mark_run_complete before start_run; skipping")
            return
        try:
            self._run.update_metadata(metadata={"complete": True})
            logger.info("TiledWriter.mark_run_complete run=%s", self._run_uid)
        except Exception:
            logger.exception("TiledWriter.mark_run_complete failed")

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
