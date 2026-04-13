"""Tests for holoptycho.liverecon_utils."""
from holoptycho.liverecon_utils import parse_scan_header


def test_parse_scan_header_valid(tmp_path):
    config_file = tmp_path / "header.txt"
    config_file.write_text(
        "[scan]\n"
        "scan_num = 12345\n"
        "x_num = 100\n"
        "y_num = 50\n"
        "nz = 10000\n"
        "det_roix_start = 0\n"
        "det_roiy_start = 0\n"
        "x_range = 5.0\n"
        "y_range = 3.0\n"
        "angle = 15.0\n"
        "xmotor = ssx\n"
        "ymotor = ssy\n"
    )
    p = parse_scan_header(str(config_file))
    assert p is not None
    assert p.scan_num == 12345
    assert p.x_num == 100
    assert p.y_num == 50
    assert p.nz == 10000
    assert p.x_range == 5.0
    assert p.y_range == 3.0
    assert p.angle == 15.0
    assert p.x_motor == "ssx"
    assert p.y_motor == "ssy"


def test_parse_scan_header_missing_file():
    result = parse_scan_header("/nonexistent/path/header.txt")
    assert result is None


def test_parse_scan_header_malformed(tmp_path):
    config_file = tmp_path / "bad.txt"
    config_file.write_text("this is not a valid config file\n")
    result = parse_scan_header(str(config_file))
    assert result is None


def test_parse_scan_header_angle_default(tmp_path):
    """Angle defaults to 0 when not specified."""
    config_file = tmp_path / "header.txt"
    config_file.write_text(
        "[scan]\n"
        "scan_num = 1\n"
        "x_num = 10\n"
        "y_num = 10\n"
        "nz = 100\n"
        "det_roix_start = 0\n"
        "det_roiy_start = 0\n"
        "x_range = 1.0\n"
        "y_range = 1.0\n"
        "xmotor = ssx\n"
        "ymotor = ssy\n"
    )
    p = parse_scan_header(str(config_file))
    assert p is not None
    assert p.angle == 0
