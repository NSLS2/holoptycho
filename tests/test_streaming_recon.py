"""Tests for holoptycho.streaming_recon.StreamingPtychoRecon."""
from types import SimpleNamespace

import numpy as np
import pytest


def _make_config(**overrides):
    """Build a minimal config SimpleNamespace for StreamingPtychoRecon."""
    defaults = dict(
        nx=64,
        ny=64,
        n_iterations=5,
        gpu_batch_size=16,
        init_prb_flag=True,
        gpus=[0],
        xray_energy_kev=12.0,
        z_m=0.5,
        ccd_pixel_um=55.0,
        live_num_points_max=256,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ------------------------------------------------------------------
# Pure-Python tests (no GPU required)
# ------------------------------------------------------------------


class TestInitConfig:
    """Test __init__ config parsing — pure Python, no GPU."""

    def test_basic_config(self):
        from holoptycho.streaming_recon import StreamingPtychoRecon

        config = _make_config()
        recon = StreamingPtychoRecon(config)
        assert recon.nx_prb == 64
        assert recon.ny_prb == 64
        assert recon.n_iterations == 5
        assert recon.gpu_batch_size == 16
        assert recon.gpu == 0

    def test_single_mode_enforced(self):
        from holoptycho.streaming_recon import StreamingPtychoRecon

        config = _make_config(prb_mode_num=2)
        with pytest.raises(NotImplementedError, match="single-mode"):
            StreamingPtychoRecon(config)

    def test_obj_multi_mode_rejected(self):
        from holoptycho.streaming_recon import StreamingPtychoRecon

        config = _make_config(obj_mode_num=3)
        with pytest.raises(NotImplementedError, match="single-mode"):
            StreamingPtychoRecon(config)

    def test_default_alpha_beta(self):
        from holoptycho.streaming_recon import StreamingPtychoRecon

        config = _make_config()
        recon = StreamingPtychoRecon(config)
        assert recon.alpha == pytest.approx(1e-3)
        assert recon.beta == pytest.approx(0.9)

    def test_snapshot_before_setup_raises(self):
        from holoptycho.streaming_recon import StreamingPtychoRecon

        config = _make_config()
        recon = StreamingPtychoRecon(config)
        with pytest.raises(RuntimeError, match="before gpu_setup"):
            recon.snapshot()


# ------------------------------------------------------------------
# GPU tests (require cupy + ptycho kernel libraries)
# ------------------------------------------------------------------


@pytest.fixture
def gpu_recon():
    """Create a StreamingPtychoRecon with GPU buffers allocated."""
    pytest.importorskip("cupy")
    pytest.importorskip("numba")
    pytest.importorskip("ptycho")

    from holoptycho.streaming_recon import StreamingPtychoRecon

    config = _make_config()
    recon = StreamingPtychoRecon(config)
    recon.gpu_setup(num_points_max=256)
    return recon


class TestGpuSetup:
    def test_buffer_shapes(self, gpu_recon):
        import cupy as cp

        recon = gpu_recon
        assert recon.prb_d.shape == (1, 64, 64)
        assert recon.diff_d.shape == (256, 64, 64)
        assert recon.point_info_d.shape == (256, 4)
        assert isinstance(recon.prb_d, cp.ndarray)

    def test_precision_single(self, gpu_recon):
        recon = gpu_recon
        assert recon.float_precision == np.float32
        assert recon.complex_precision == np.complex64

    def test_pixel_sizes_computed(self, gpu_recon):
        recon = gpu_recon
        assert recon.x_pixel_m > 0
        assert recon.y_pixel_m > 0

    def test_mmap_buffers_allocated(self, gpu_recon):
        recon = gpu_recon
        assert recon.mmap_prb is not None
        assert recon.mmap_obj is not None
        assert recon.mmap_prb.shape[0] == recon.n_iterations


class TestResetForScan:
    def test_resets_object_dimensions(self, gpu_recon):
        recon = gpu_recon
        recon.reset_for_scan(
            scan_num="12345",
            x_range_um=5.0,
            y_range_um=3.0,
            num_points_max=128,
        )
        assert recon.scan_num == "12345"
        assert recon.nx_obj > recon.nx_prb
        assert recon.ny_obj > recon.ny_prb
        assert recon.num_points == 128

    def test_object_reinitialized(self, gpu_recon):
        recon = gpu_recon
        recon.reset_for_scan(
            scan_num="1",
            x_range_um=2.0,
            y_range_um=2.0,
            num_points_max=64,
        )
        # Object should be ~0.99 * exp(-0.1j) everywhere
        expected = 0.99 * np.exp(-0.1j)
        assert np.allclose(recon.obj_mode[0, 0, 0], expected, atol=1e-6)


class TestInitialProbe:
    def test_probe_from_synthetic_diff(self, gpu_recon):
        import cupy as cp

        recon = gpu_recon
        recon.reset_for_scan("1", 2.0, 2.0, 64)

        # Fill diff_d with synthetic data (uniform intensity)
        recon.diff_d[:64] = cp.ones(
            (64, 64, 64), dtype=recon.float_precision
        )
        recon.num_points_recon = 64

        recon.initial_probe(64)

        # Probe should be non-zero after initialization
        assert np.any(recon.prb_mode[0] != 0)


class TestSaveFinal:
    def test_saves_probe_and_object(self, gpu_recon, tmp_path):
        recon = gpu_recon
        recon.reset_for_scan("1", 2.0, 2.0, 64)

        save_dir = str(tmp_path / "recon_output")
        result = recon.save_final(save_dir=save_dir)

        assert result == save_dir
        assert (tmp_path / "recon_output" / "probe.npy").exists()
        assert (tmp_path / "recon_output" / "object.npy").exists()

        probe = np.load(tmp_path / "recon_output" / "probe.npy")
        assert probe.shape == recon.prb_mode.shape
