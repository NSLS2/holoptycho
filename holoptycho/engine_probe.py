"""Derive the ViT-patch ``inner_crop`` from a model's engine/ONNX artifacts.

Locates the ONNX that produced a compiled ``.engine`` and reads the probe
baked into it, handing the probe geometry to
:func:`ptychoml.inner_crop_from_probe`. Kept free of Holoscan / Tiled imports
so it can be unit tested without the full pipeline (``vit_inference`` performs
import-time I/O setup that needs a live Tiled endpoint).
"""
from __future__ import annotations

import re
from pathlib import Path

from ptychoml import inner_crop_from_probe


def find_onnx_for_engine(engine_path: str) -> Path | None:
    """Return the ONNX path corresponding to a compiled .engine file, or None.

    Convention (set by model_manager.py)::

        engine:  {ENGINE_CACHE_DIR}/{model_name}_v{version}.engine
        onnx:    {ENGINE_CACHE_DIR}/onnx/{model_name}/{version}/*.onnx
    """
    ep = Path(engine_path)
    m = re.fullmatch(r"(.+?)_v(\d+)\.engine", ep.name)
    if not m:
        return None
    model_name, version = m.group(1), m.group(2)
    onnx_dir = ep.parent / "onnx" / model_name / version
    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    return onnx_files[0] if onnx_files else None


def inner_crop_from_onnx(onnx_path, threshold: float = 0.50) -> int | None:
    """Derive ``inner_crop`` from the probe baked into an ONNX model.

    Loads the ONNX, extracts the complex probe (``probe_real`` / ``probe_imag``
    initializers, or two 2-D constant graph outputs as a fallback), and hands
    the probe geometry to :func:`ptychoml.inner_crop_from_probe`. Returns
    ``None`` if ``onnx`` isn't installed, the model can't be loaded, or the
    probe can't be extracted — so the caller can fall back to auto-deriving the
    crop.
    """
    try:
        import onnx
        import onnx.numpy_helper as nph
    except ImportError:
        return None

    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return None

    init_map = {i.name: nph.to_array(i) for i in model.graph.initializer}
    out_names = [o.name for o in model.graph.output]

    if "probe_real" in init_map and "probe_imag" in init_map:
        p_re = init_map["probe_real"]
        p_im = init_map["probe_imag"]
    else:
        # Fall back: any graph output (other than 'output') that is also a
        # constant initializer with 2-D spatial shape.
        probe_cands = [
            init_map[n] for n in out_names
            if n != "output" and n in init_map and init_map[n].ndim == 2
        ]
        if len(probe_cands) < 2:
            return None
        p_re, p_im = probe_cands[0], probe_cands[1]

    if p_re.shape != p_im.shape or p_re.ndim != 2:
        return None

    return inner_crop_from_probe(p_re + 1j * p_im, threshold)
