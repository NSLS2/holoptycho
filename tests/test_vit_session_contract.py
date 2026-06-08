"""Guard the ptychoml.PtychoViTInference call contract used by vit_inference.

``PtychoViTInferenceOp._init_session`` constructs the session with a
``fftshift`` keyword. ptychoml renamed this parameter (it was
``data_is_shifted``), and bumping the ptychoml pin across that rename without
updating the call site silently broke the ViT op at the first batch — not at
import, so the smoke tests missed it. This pins the contract from holoptycho's
side so a future rename fails loudly here instead of on the beamline.

Constructs ``PtychoViTInference`` directly (ptychoml only — no holoscan/TILED),
so it runs in the plain CI environment.
"""
import inspect

from ptychoml import PtychoViTInference


def test_session_accepts_fftshift_kwarg():
    params = inspect.signature(PtychoViTInference.__init__).parameters
    assert "fftshift" in params, (
        "ptychoml.PtychoViTInference no longer accepts 'fftshift'; "
        "PtychoViTInferenceOp._init_session passes it"
    )


def test_session_construction_matches_op_call():
    # Mirror PtychoViTInferenceOp._init_session's call exactly. __init__ only
    # stores config (the engine loads lazily in _init_engine), so this must
    # not raise on the keyword even with a non-existent engine path.
    session = PtychoViTInference(
        engine_path="/tmp/nonexistent.engine", gpu=0, fftshift=None
    )
    assert session is not None
