"""Tests for holoptycho.engine_probe (engine/ONNX -> inner_crop helpers).

find_onnx_for_engine is the holoptycho-specific filename-convention logic and
is fully covered here. The inscribed-square geometry lives in (and is tested
by) ptychoml.inner_crop_from_probe. inner_crop_from_onnx's ONNX *extraction*
path needs the `onnx` package, which isn't in the CI test env, so only its
graceful-degradation paths are checked here; the extraction is exercised by
real runs against actual model ONNX files.
"""
from holoptycho.engine_probe import find_onnx_for_engine, inner_crop_from_onnx


def test_find_onnx_locates_matching_onnx(tmp_path):
    # engine {dir}/run042901_v5.engine -> onnx {dir}/onnx/run042901/5/*.onnx
    engine = tmp_path / "run042901_v5.engine"
    engine.write_bytes(b"x")
    onnx_dir = tmp_path / "onnx" / "run042901" / "5"
    onnx_dir.mkdir(parents=True)
    onnx_file = onnx_dir / "model.onnx"
    onnx_file.write_bytes(b"x")

    assert find_onnx_for_engine(str(engine)) == onnx_file


def test_find_onnx_picks_first_sorted(tmp_path):
    engine = tmp_path / "m_v2.engine"
    engine.write_bytes(b"x")
    d = tmp_path / "onnx" / "m" / "2"
    d.mkdir(parents=True)
    (d / "b.onnx").write_bytes(b"x")
    (d / "a.onnx").write_bytes(b"x")

    assert find_onnx_for_engine(str(engine)).name == "a.onnx"


def test_find_onnx_parses_multidigit_version_and_underscored_name(tmp_path):
    engine = tmp_path / "run_042_901_v12.engine"
    engine.write_bytes(b"x")
    d = tmp_path / "onnx" / "run_042_901" / "12"
    d.mkdir(parents=True)
    (d / "m.onnx").write_bytes(b"x")

    assert find_onnx_for_engine(str(engine)) == d / "m.onnx"


def test_find_onnx_bad_name_returns_none(tmp_path):
    # no _v{N} version suffix -> doesn't match the convention
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"x")
    assert find_onnx_for_engine(str(engine)) is None


def test_find_onnx_missing_dir_returns_none(tmp_path):
    engine = tmp_path / "m_v1.engine"
    engine.write_bytes(b"x")
    assert find_onnx_for_engine(str(engine)) is None  # no onnx/ tree


def test_find_onnx_empty_dir_returns_none(tmp_path):
    engine = tmp_path / "m_v1.engine"
    engine.write_bytes(b"x")
    (tmp_path / "onnx" / "m" / "1").mkdir(parents=True)  # exists but no *.onnx
    assert find_onnx_for_engine(str(engine)) is None


def test_inner_crop_from_onnx_unloadable_returns_none(tmp_path):
    # Missing onnx package (CI) or unreadable file both fall back to None so
    # the caller can auto-derive instead of crashing.
    assert inner_crop_from_onnx(str(tmp_path / "does_not_exist.onnx")) is None
