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
        # ViT inference output buffer (sibling to dp; same stride).
        self._inference_node = None
        # ViT mosaic node, set by the first write_vit_mosaic so subsequent
        # patch_vit_mosaic calls don't have to re-walk the container.
        self._vit_mosaic_node = None
        # Same caching for the parallel amp mosaic at <run>/vit/mosaic_amp.
        self._vit_amp_mosaic_node = None
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
        # Reset cached array nodes so they don't point at the previous run's
        # subcontainers.
        self._dp_node = None
        self._dp_chunk_size = 0
        self._inference_node = None
        self._vit_mosaic_node = None
        self._vit_amp_mosaic_node = None
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

    @staticmethod
    def _complex_to_amp_phase(arr: np.ndarray) -> np.ndarray:
        """Convert a complex reconstruction array to ``(2, H, W)`` float32.

        ``[0]`` is amplitude ``|arr|``, ``[1]`` is phase ``angle(arr)`` in
        radians. Multi-mode inputs (``(modes, H, W)``) collapse to mode 0
        — only the dominant mode is shipped to the dashboard.

        The dashboard's TiledImageTile renderer expects this two-channel
        layout (it picks an index to display); raw complex64 over the wire
        was not decodable.
        """
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a[0]
        return np.stack([
            np.abs(a).astype(np.float32),
            np.angle(a).astype(np.float32),
        ])

    def write_live(self, iteration: int, probe: np.ndarray, obj: np.ndarray) -> None:
        """Overwrite the live probe/object snapshots for the current run.

        Called every ``display_interval`` iterations. Stored as ``(2, H, W)``
        float32 (``[0]`` = amplitude, ``[1]`` = phase) so the dashboard can
        display either channel with a simple slice index.

        If either array contains non-finite values (NaN/Inf), the write is
        skipped so the previous finite snapshot stays visible to consumers
        (synaps-dash, etc.). The iterative engine can transiently diverge —
        writing the corrupted state would replace a useful image with a black
        one. The next finite iteration will overwrite as usual.
        """
        if self._run is None:
            logger.warning("write_live called before start_run; skipping")
            return
        probe_ap = self._complex_to_amp_phase(probe)
        obj_ap = self._complex_to_amp_phase(obj)
        if not np.isfinite(probe_ap).all() or not np.isfinite(obj_ap).all():
            logger.warning(
                "write_live skipped: non-finite values in probe/object at iter=%d "
                "(keeping last finite snapshot)",
                iteration,
            )
            return
        try:
            live = _get_or_create(self._run, "live")
            meta = {"iteration": iteration}
            self._write_or_overwrite_array(live, "probe", probe_ap, metadata=meta)
            self._write_or_overwrite_array(live, "object", obj_ap, metadata=meta)
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
        """Write final reconstruction results when a scan completes.

        Stores ``probe`` and ``obj`` as ``(2, H, W)`` float32 (amp, phase) —
        same layout as ``write_live`` — so offline consumers and the
        dashboard share a single decode path.
        """
        if self._run is None:
            logger.warning("write_final called before start_run; skipping")
            return
        try:
            final = _get_or_create(self._run, "final")
            self._write_or_overwrite_array(final, "probe", self._complex_to_amp_phase(probe))
            self._write_or_overwrite_array(final, "object", self._complex_to_amp_phase(obj))
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
        n_keep: int,
        frame_shape: tuple[int, int],
        dtype=np.uint8,
        frames_per_chunk: int = 64,
        stride: int = 1,
    ) -> None:
        """Register the per-run diffraction buffer structure for fine-tuning data.

        Registers a ``(n_keep, H, W)`` array under ``<run>/diffraction/dp``
        *without* uploading any data. Tiled records the array shape, dtype, and
        chunking immediately; chunks are populated lazily by
        ``write_diffraction_chunk`` as frames arrive. The register-then-write-
        blocks split is essential — the alternative
        ``write_array(np.zeros(...))`` path would upload the whole zero buffer
        up front, which times out over WAN.

        ``stride`` controls the row-to-scan-frame mapping: row ``r`` of dp
        holds scan frame ``r * stride``. ``n_keep`` is the count of kept
        frames (= ``(nz - 1) // stride + 1``). For ``stride=1`` (default) the
        buffer is dense, ``n_keep == nz``, and behaviour matches the original
        write-every-frame semantics. The stride is stamped into run metadata
        as ``dp_stride`` so consumers can recover the mapping.

        ``dp`` stores **amplitude** (= ``sqrt(intensity)``, rounded to uint8)
        rather than raw uint16 intensity. The Eiger detector is 16-bit but
        ``sqrt(65535) ≈ 256``, so the full intensity range maps cleanly into
        uint8 — and the 1-count quantization that introduces is well below
        the Poisson noise floor for any pixel with non-trivial counts. This
        halves the on-the-wire write volume vs uint16 (640 MB → 320 MB for
        a 10K frame 256×256 scan) since Tiled's ``write_block`` does not
        accept compressed payloads. ptycho-vit's Tiled-backed loader expects
        this amplitude form and skips its own sqrt step on this data path.
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

            # Chunking matches ImageBatchOp's batch size when stride=1 so each
            # FrameWriterOp invocation maps to exactly one block_id. For
            # large strides n_keep can be tiny (e.g. 40 rows for stride=1000
            # on a 40K-frame scan); clamp chunk_dim0 to n_keep so we don't
            # create a zero-row tail block.
            chunk0 = min(int(frames_per_chunk), int(n_keep))
            chunk0 = max(1, chunk0)
            n_full = n_keep // chunk0
            rem = n_keep - n_full * chunk0
            chunk_dim0 = (chunk0,) * n_full
            if rem:
                chunk_dim0 = chunk_dim0 + (rem,)
            structure = ArrayStructure(
                shape=(int(n_keep), h, w),
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
            self._dp_chunk_size = int(chunk0)
            self._dp_stride = int(stride)
            self._run.update_metadata(metadata={"dp_stride": int(stride)})
            logger.info(
                "TiledWriter.start_diffraction_buffer run=%s shape=(%d,%d,%d) "
                "dtype=%s chunks=%d frames stride=%d",
                self._run_uid, n_keep, h, w, dtype_np, chunk0, stride,
            )
        except Exception:
            logger.exception("TiledWriter.start_diffraction_buffer failed")
            self._dp_node = None
            self._dp_chunk_size = 0
            self._dp_stride = 1

    def write_diffraction_chunk(
        self,
        rows: np.ndarray,
        frames: np.ndarray,
    ) -> None:
        """Write a chunk of detector-frame intensity into ``<run>/diffraction/dp``.

        ``rows`` is a ``(B,)`` array of compact-dp row indices (not global
        scan-frame numbers); ``frames`` is ``(B, H, W)`` of intensity (any
        uint dtype is accepted — upstream ``ImageBatchOp`` allocates uint32).
        Frames are sqrt'd and cast to uint8 before write to halve the wire
        volume vs uint16; see ``start_diffraction_buffer`` for justification.

        With ``stride=1`` rows == global frame indices (dense case). With
        stride > 1 the caller has filtered batches down to kept frames and
        mapped each to its compact row (``row = frame // stride``); the
        writer is row-index agnostic.

        Uses ``ArrayClient.patch(data, offset=(row_start, 0, 0))`` rather than
        ``write_block``: the upstream batch boundary doesn't need to align to
        the on-disk chunk grid. This matters because **a single dropped frame
        mid-stream permanently desynchronises ImageBatchOp's counter** — the
        next batch starts at a non-aligned offset and stays non-aligned. With
        block writes that loses every subsequent batch (observed: 30000 of
        40000 frames lost from one ZMQ HWM-overflow drop). ``patch`` lands at
        the exact row offset; gaps stay as the buffer's zero-init.
        """
        if self._dp_node is None:
            return
        try:
            rows = np.asarray(rows)
            if len(rows) == 0:
                return
            intensity = np.asarray(frames)
            amp = np.sqrt(intensity, dtype=np.float32)
            np.clip(amp, 0, 255, out=amp)
            amp_u8 = np.ascontiguousarray(amp.astype(np.uint8, copy=False))

            start = int(rows[0])
            if np.all(np.diff(rows) == 1):
                # Contiguous rows — one patch covering [start, start+B).
                self._dp_node.patch(amp_u8, offset=(start, 0, 0))
            else:
                # Non-contiguous rows. Per-row patches so we don't corrupt
                # slots between the present rows. Common when stride > 1 and
                # a batch happens to span two stride boundaries.
                logger.warning(
                    "write_diffraction_chunk: non-contiguous rows "
                    "[%d..%d] (n=%d); writing per-row",
                    start, int(rows[-1]), len(rows),
                )
                for i, r in enumerate(rows):
                    self._dp_node.patch(amp_u8[i:i + 1], offset=(int(r), 0, 0))
        except Exception:
            logger.exception("TiledWriter.write_diffraction_chunk failed")

    def start_inference_buffer(
        self,
        n_keep: int,
        frame_shape: tuple[int, int],
        dtype=np.float32,
        n_channels: int = 2,
        frames_per_chunk: int = 64,
        stride: int = 1,
    ) -> None:
        """Register the per-run ViT inference buffer at ``<run>/diffraction/inference``.

        Sibling to ``start_diffraction_buffer``: same compact row layout
        keyed by ``stride``, but stores the model's full ``(n_channels, H, W)``
        prediction per kept frame (typically ``n_channels=2`` for amp+phase).
        Row r holds the inference output for scan frame ``r * stride`` — the
        same scan frame whose detector data lives at ``dp[r]``.

        Float32 because the model outputs are unbounded reals near 0; uint
        quantisation would clip useful dynamic range. For stride=1000 on a
        40K-frame scan the buffer is only ~21 MB, so the cost is negligible.
        """
        if self._run is None:
            logger.warning("start_inference_buffer before start_run; skipping")
            return
        try:
            from tiled.structures.array import ArrayStructure, BuiltinDtype
            from tiled.structures.core import StructureFamily
            from tiled.structures.data_source import DataSource

            diffraction = _get_or_create(self._run, "diffraction")
            h, w = int(frame_shape[0]), int(frame_shape[1])
            dtype_np = np.dtype(dtype)
            nc = int(n_channels)

            chunk0 = min(int(frames_per_chunk), int(n_keep))
            chunk0 = max(1, chunk0)
            n_full = n_keep // chunk0
            rem = n_keep - n_full * chunk0
            chunk_dim0 = (chunk0,) * n_full
            if rem:
                chunk_dim0 = chunk_dim0 + (rem,)
            structure = ArrayStructure(
                shape=(int(n_keep), nc, h, w),
                chunks=(chunk_dim0, (nc,), (h,), (w,)),
                data_type=BuiltinDtype.from_numpy_dtype(dtype_np),
            )
            data_source = DataSource(
                structure=structure,
                structure_family=StructureFamily.array,
            )
            if "inference" in diffraction:
                del diffraction["inference"]
            self._inference_node = diffraction.new(
                StructureFamily.array,
                [data_source],
                key="inference",
                specs=_SPECS,
                access_tags=_ACCESS_TAGS,
            )
            logger.info(
                "TiledWriter.start_inference_buffer run=%s shape=(%d,%d,%d,%d) "
                "dtype=%s stride=%d",
                self._run_uid, n_keep, nc, h, w, dtype_np, stride,
            )
        except Exception:
            logger.exception("TiledWriter.start_inference_buffer failed")
            self._inference_node = None

    def write_inference_chunk(
        self,
        rows: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        """Write ``(B, n_channels, H, W)`` predictions at compact rows in dp/inference.

        Mirrors ``write_diffraction_chunk`` shape-wise (contiguous → one
        patch, non-contiguous → per-row patches). Cast to float32 to match
        the buffer's declared dtype.
        """
        if getattr(self, "_inference_node", None) is None:
            return
        try:
            rows = np.asarray(rows)
            if len(rows) == 0:
                return
            arr = np.ascontiguousarray(np.asarray(preds, dtype=np.float32))
            start = int(rows[0])
            if np.all(np.diff(rows) == 1):
                self._inference_node.patch(arr, offset=(start, 0, 0, 0))
            else:
                logger.warning(
                    "write_inference_chunk: non-contiguous rows "
                    "[%d..%d] (n=%d); writing per-row",
                    start, int(rows[-1]), len(rows),
                )
                for i, r in enumerate(rows):
                    self._inference_node.patch(arr[i:i + 1], offset=(int(r), 0, 0, 0))
        except Exception:
            logger.exception("TiledWriter.write_inference_chunk failed")

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

    def update_dp_progress(self, n_written: int) -> None:
        """Stamp ``dp_frames_written: n_written`` into the run's metadata.

        Distinct from ``positions_um`` filled-count: ``PositionsWriterOp``
        writes the whole positions array per batch (cheap), while
        ``FrameWriterOp`` writes ~1 MB per 64-frame patch (slow over WAN),
        so dp can lag positions by tens of thousands of frames during a
        fast scan. The dashboard's frame slider needs to know the *dp*
        progress to clamp its max — otherwise users scroll past actual
        data into the buffer's zero-init.
        """
        if self._run is None:
            logger.warning("update_dp_progress before start_run; skipping")
            return
        try:
            self._run.update_metadata(metadata={"dp_frames_written": int(n_written)})
        except Exception:
            logger.exception("TiledWriter.update_dp_progress failed")

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
        """Overwrite the entire server-side ViT mosaic for the current run.

        Used for the first write of each scan to seed the canvas (including
        the fill colour applied to never-stitched regions). Incremental
        per-batch updates after that go through ``patch_vit_mosaic``, which
        only sends the bounding box of the newly-placed patches and is
        ~30-100× cheaper over WAN.
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
            # Cache the array node so subsequent patch() calls don't have to
            # walk the container path on every batch.
            self._vit_mosaic_node = vit["mosaic"]
        except Exception:
            logger.exception("TiledWriter.write_vit_mosaic failed")

    def patch_vit_mosaic(
        self,
        subregion: np.ndarray,
        *,
        offset_yx: tuple[int, int],
        batch_num: int,
    ) -> None:
        """Patch a sub-region of the mosaic in place.

        ``subregion`` is the already-normalised float32 patch (mosaic / counts
        for the affected bbox). ``offset_yx`` is the top-left corner of that
        region in canvas pixel coordinates. Sends only the bbox bytes, not
        the whole 36 MB canvas, so it's the path the live mosaic loop
        depends on staying ahead of WAN throughput.
        """
        if self._run is None:
            logger.warning("patch_vit_mosaic called before start_run; skipping")
            return
        node = getattr(self, "_vit_mosaic_node", None)
        if node is None:
            # Caller forgot to seed via write_vit_mosaic first — fall back to
            # re-fetching, which is still correct, just slower on first call.
            try:
                node = self._run["vit"]["mosaic"]
                self._vit_mosaic_node = node
            except Exception:
                logger.exception("patch_vit_mosaic: failed to resolve mosaic node")
                return
        try:
            node.patch(np.ascontiguousarray(subregion), offset=(int(offset_yx[0]), int(offset_yx[1])))
            node.update_metadata(metadata={"batch_num": int(batch_num)})
        except Exception:
            logger.exception("TiledWriter.patch_vit_mosaic failed")

    def write_vit_amp_mosaic(
        self,
        mosaic: np.ndarray,
        *,
        batch_num: int,
        pixel_size_m: float,
        canvas_origin_um: tuple[float, float],
    ) -> None:
        """Seed the server-side ViT amplitude mosaic at ``<run>/vit/mosaic_amp``.

        Parallel sibling to ``write_vit_mosaic`` (phase). First write of each
        run paints fill colour across the full canvas; subsequent batches
        go through ``patch_vit_amp_mosaic``.
        """
        if self._run is None:
            logger.warning("write_vit_amp_mosaic called before start_run; skipping")
            return
        try:
            vit = _get_or_create(self._run, "vit")
            meta = {
                "batch_num": batch_num,
                "pixel_size_m": float(pixel_size_m),
                "canvas_origin_um": [float(canvas_origin_um[0]), float(canvas_origin_um[1])],
            }
            self._write_or_overwrite_array(vit, "mosaic_amp", mosaic, metadata=meta)
            self._vit_amp_mosaic_node = vit["mosaic_amp"]
        except Exception:
            logger.exception("TiledWriter.write_vit_amp_mosaic failed")

    def patch_vit_amp_mosaic(
        self,
        subregion: np.ndarray,
        *,
        offset_yx: tuple[int, int],
        batch_num: int,
    ) -> None:
        """Patch a sub-region of the amp mosaic in place. Sibling of
        ``patch_vit_mosaic`` (phase); same offset semantics."""
        if self._run is None:
            logger.warning("patch_vit_amp_mosaic called before start_run; skipping")
            return
        node = getattr(self, "_vit_amp_mosaic_node", None)
        if node is None:
            try:
                node = self._run["vit"]["mosaic_amp"]
                self._vit_amp_mosaic_node = node
            except Exception:
                logger.exception("patch_vit_amp_mosaic: failed to resolve mosaic_amp node")
                return
        try:
            node.patch(np.ascontiguousarray(subregion), offset=(int(offset_yx[0]), int(offset_yx[1])))
            node.update_metadata(metadata={"batch_num": int(batch_num)})
        except Exception:
            logger.exception("TiledWriter.patch_vit_amp_mosaic failed")

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
