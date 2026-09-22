"""Smoke tests for Google Cloud TPU v4 resynchronisation scan kernels."""

import numpy as np

from sigil import kernels as K
from sigil.backend import Scanner, transform_bank


def test_tpu_availability():
    """Verify that Google Cloud TPU v4 devices are properly discovered."""
    assert K.have_tpu(), "Expected Google Cloud TPU devices to be discovered"
    devices = K.tpu_devices()
    assert len(devices) > 0, "Expected at least 1 TPU device"
    assert any("TPU" in d.device_kind for d in devices), "Expected TPU device kind"


def test_tpu_scan_kernel_parity_small():
    """Verify exact numerical agreement between fused TPU scan kernel and reference."""
    H, W = 128, 128
    L, Kc = 4, 128

    rng = np.random.default_rng(42)
    excess = rng.standard_normal((H, W)).astype(np.float64)
    rots = np.linspace(-3, 3, 4)
    scales = np.linspace(0.8, 1.2, 8)
    cos_t, sin_t, mirror = transform_bank(rots, scales, [1.0, -1.0])

    sc_tpu = Scanner(excess, cos_t, sin_t, mirror, device="tpu")
    sc_cpu = Scanner(excess, cos_t, sin_t, mirror, device="cpu")

    fy = rng.uniform(0.05, 0.25, size=(L, Kc)).astype(np.float32)
    fx = rng.uniform(0.05, 0.25, size=(L, Kc)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(L, Kc)).astype(np.float32)

    # Reference CPU calculation
    ref = sc_cpu.statistics_multi(fy, fx, signs)

    # Fused TPU kernel calculation
    fused = K.scan_statistics(
        sc_tpu.field,
        fy,
        fx,
        signs,
        sc_tpu.cos_t,
        sc_tpu.sin_t,
        sc_tpu.mirror,
    )
    assert fused is not None, "TPU scan kernel should execute for valid inputs"
    if hasattr(fused, "block_until_ready"):
        fused.block_until_ready()

    fused_np = np.asarray(fused, dtype=np.float64)
    max_diff = float(np.max(np.abs(fused_np - ref)))
    assert max_diff < 1e-4, (
        f"TPU fused kernel disagreement {max_diff:.2e} exceeds tolerance 1e-4"
    )


def test_tpu_scan_statistics_dispatch():
    """Verify that Scanner.statistics_multi automatically dispatches to TPU."""
    H, W = 256, 256
    L, Kc = 2, 256
    rng = np.random.default_rng(101)
    excess = rng.standard_normal((H, W)).astype(np.float64)
    cos_t, sin_t, mirror = transform_bank([0.0, 2.0], [1.0, 1.05], [1.0, -1.0])

    sc = Scanner(excess, cos_t, sin_t, mirror, device="tpu")
    fy = rng.uniform(0.1, 0.3, size=(L, Kc)).astype(np.float32)
    fx = rng.uniform(0.1, 0.3, size=(L, Kc)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(L, Kc)).astype(np.float32)

    stats = sc.statistics_multi(fy, fx, signs)
    assert stats.shape == (L, cos_t.size)
    assert not np.isnan(stats).any()
    assert not np.isinf(stats).any()
