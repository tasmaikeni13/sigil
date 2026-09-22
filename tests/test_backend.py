"""Tests for sigil.backend Scanner on Google Cloud TPU v4 and CPU."""

import numpy as np

from sigil.backend import Scanner, tpu_available, transform_bank


def test_tpu_available_flag():
    """Verify tpu_available() returns True on TPU VM."""
    assert tpu_available() is True


def test_transform_bank_generation():
    """Verify transform bank decomposition and coordinates."""
    rots = [0.0, 45.0, 90.0]
    scales = [1.0, 1.2]
    refs = [1.0, -1.0]
    cos_t, sin_t, mirror = transform_bank(rots, scales, refs)

    expected_size = len(rots) * len(scales) * len(refs)
    assert cos_t.shape == (expected_size,)
    assert sin_t.shape == (expected_size,)
    assert mirror.shape == (expected_size,)

    # Identity transform at index 0 (0 deg, 1.0 scale, 1.0 mirror)
    np.testing.assert_allclose(cos_t[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(sin_t[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(mirror[0], 1.0, atol=1e-6)


def test_scanner_single_statistics_tpu_vs_cpu():
    """Verify Scanner.statistics() agreement between TPU and CPU."""
    rng = np.random.default_rng(7)
    excess = rng.standard_normal((128, 128))
    cos_t, sin_t, mirror = transform_bank([0.0, 15.0], [1.0, 1.1], [1.0, -1.0])

    sc_cpu = Scanner(excess, cos_t, sin_t, mirror, device="cpu")
    sc_tpu = Scanner(excess, cos_t, sin_t, mirror, device="tpu")

    fy = rng.uniform(0.1, 0.3, size=128)
    fx = rng.uniform(0.1, 0.3, size=128)
    signs = rng.choice([-1.0, 1.0], size=128)

    stat_cpu = sc_cpu.statistics(fy, fx, signs)
    stat_tpu = sc_tpu.statistics(fy, fx, signs)

    np.testing.assert_allclose(stat_tpu, stat_cpu, atol=1e-4)


def test_scanner_evidence_vector():
    """Verify Scanner.evidence() extraction on TPU."""
    rng = np.random.default_rng(9)
    excess = rng.standard_normal((128, 128))
    cos_t, sin_t, mirror = transform_bank([0.0], [1.0], [1.0])

    sc_cpu = Scanner(excess, cos_t, sin_t, mirror, device="cpu")
    sc_tpu = Scanner(excess, cos_t, sin_t, mirror, device="tpu")

    fy = rng.uniform(0.1, 0.3, size=64)
    fx = rng.uniform(0.1, 0.3, size=64)

    ev_cpu = sc_cpu.evidence(fy, fx, 0)
    ev_tpu = sc_tpu.evidence(fy, fx, 0)

    np.testing.assert_allclose(ev_tpu, ev_cpu, atol=1e-4)
