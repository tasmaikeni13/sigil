"""Google Cloud TPU v4 fused resynchronisation scan kernels.

The scan in :mod:`sigil.backend` evaluates the detector statistic over all
hypotheses in a transform bank against candidate carrier sets:

    z        = bilinear(field, R_m · carrier_k)        # (L, M, K)
    num[l,m] = Σ_k z[l,m,k] · sign[l,k]
    den[l,m] = Σ_k z[l,m,k]²

On Google Cloud TPU v4 (including v4-32 pod slices), materialising ``z`` as an
intermediate global tensor for realistic evaluation grids (dozens of candidate
keys, thousands of transform hypotheses, thousands of carriers) would create
excessive memory traffic across high-bandwidth memory (HBM).

The TPU kernel fuses coordinate rotation, reflection, bilinear interpolation,
and dual accumulation directly into TPU Vector and Matrix Units (VPU/MXU) using
tiled carrier accumulation. Intermediate carrier slices are processed in SRAM
registers while keeping the partial sums resident, streaming only the excess
field and writing the final (L, M) statistics.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

# Configure Google Cloud TPU v4 topology defaults if running in single-host or local evaluation mode
os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")

_HAVE_TPU = False
_JAX = None
_JNP = None
_LAX = None
_TPU_DEVICES: List[Any] = []

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    _JAX = jax
    _JNP = jnp
    _LAX = lax
    devices = jax.devices()
    _TPU_DEVICES = [d for d in devices if "TPU" in d.device_kind]
    _HAVE_TPU = len(_TPU_DEVICES) > 0
except Exception:  # pragma: no cover
    _HAVE_TPU = False


def have_tpu() -> bool:
    """Return True if Google Cloud TPU devices are available."""
    return _HAVE_TPU


def tpu_devices() -> List[Any]:
    """Return the list of available TPU devices."""
    return list(_TPU_DEVICES)


def tpu_synchronize() -> None:
    """Synchronize pending TPU computations."""
    if _HAVE_TPU and _JAX is not None:
        # Block until all TPU computations are complete
        _JAX.effects_barrier()


#: Minimum number of gathered samples to justify dispatching to TPU kernels
MIN_WORK = 1 << 12


if _HAVE_TPU and _JAX is not None:

    @_JAX.jit
    def _tpu_scan_kernel_tiled(
        field: _JAX.Array,
        fy: _JAX.Array,
        fx: _JAX.Array,
        signs: _JAX.Array,
        cos_t: _JAX.Array,
        sin_t: _JAX.Array,
        mirror: _JAX.Array,
    ) -> _JAX.Array:
        """Fused TPU scan with carrier tiling using lax.scan."""
        h, w = field.shape
        cy = float(h // 2)
        cx = float(w // 2)
        ymax = float(h) - 1.001
        xmax = float(w) - 1.001

        L_dim, K_dim = fy.shape
        M_dim = cos_t.shape[0]

        # Use 128-element carrier tiles matching TPU vector registers
        block_k = 128
        num_blocks = K_dim // block_k

        fy_blk = fy.reshape(L_dim, num_blocks, block_k)
        fx_blk = fx.reshape(L_dim, num_blocks, block_k)
        sg_blk = signs.reshape(L_dim, num_blocks, block_k)

        def step_fn(acc: Tuple[_JAX.Array, _JAX.Array], i: _JAX.Array):
            num_acc, den_acc = acc
            fy_k = fy_blk[:, i, :]
            fx_k = fx_blk[:, i, :]
            sg_k = sg_blk[:, i, :]

            uy = (fy_k * float(h))[:, None, :]
            ux = (fx_k * float(w))[:, None, :] * mirror[None, :, None]

            ry = cos_t[None, :, None] * uy - sin_t[None, :, None] * ux
            rx = sin_t[None, :, None] * uy + cos_t[None, :, None] * ux

            gy = _JNP.clip(ry + cy, 0.0, ymax)
            gx = _JNP.clip(rx + cx, 0.0, xmax)

            y0 = _JNP.floor(gy).astype(_JNP.int32)
            x0 = _JNP.floor(gx).astype(_JNP.int32)
            ty = gy - y0
            tx = gx - x0
            y1 = _JNP.minimum(y0 + 1, h - 1)
            x1 = _JNP.minimum(x0 + 1, w - 1)

            p00 = field[y0, x0]
            p01 = field[y0, x1]
            p10 = field[y1, x0]
            p11 = field[y1, x1]

            z = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p01) + ty * (
                (1.0 - tx) * p10 + tx * p11
            )
            num_step = _JNP.sum(z * sg_k[:, None, :], axis=-1)
            den_step = _JNP.sum(z * z, axis=-1)
            return (num_acc + num_step, den_acc + den_step), None

        init_acc = (
            _JNP.zeros((L_dim, M_dim), dtype=_JNP.float32),
            _JNP.zeros((L_dim, M_dim), dtype=_JNP.float32),
        )
        (num, den), _ = _LAX.scan(step_fn, init_acc, _JNP.arange(num_blocks))
        return num / (_JNP.sqrt(den) + 1e-12)


def scan_statistics(
    field: Any,
    fy: Any,
    fx: Any,
    signs: Any,
    cos_t: Any,
    sin_t: Any,
    mirror: Any,
    block_k: int = 128,
) -> Optional[Any]:
    """Fused scan statistic for (key, hypothesis) pairs on Google Cloud TPU.

    ``field`` is (H, W); ``fy``, ``fx``, ``signs`` are (L, K); the transform
    vectors are (M,). Returns an (L, M) array on TPU or None if falling back.
    """
    if not _HAVE_TPU:
        return None

    L, K = fy.shape
    M = cos_t.size if hasattr(cos_t, "size") else len(cos_t)
    if L * M * K < MIN_WORK:
        return None

    # Carrier dimension must be divisible by block_k for tiled kernel
    if K % block_k != 0:
        return None

    # Convert to JAX device arrays if needed
    if not isinstance(field, _JAX.Array):
        field = _JNP.asarray(field, dtype=_JNP.float32)
    if not isinstance(fy, _JAX.Array):
        fy = _JNP.asarray(fy, dtype=_JNP.float32)
    if not isinstance(fx, _JAX.Array):
        fx = _JNP.asarray(fx, dtype=_JNP.float32)
    if not isinstance(signs, _JAX.Array):
        signs = _JNP.asarray(signs, dtype=_JNP.float32)
    if not isinstance(cos_t, _JAX.Array):
        cos_t = _JNP.asarray(cos_t, dtype=_JNP.float32).reshape(-1)
    if not isinstance(sin_t, _JAX.Array):
        sin_t = _JNP.asarray(sin_t, dtype=_JNP.float32).reshape(-1)
    if not isinstance(mirror, _JAX.Array):
        mirror = _JNP.asarray(mirror, dtype=_JNP.float32).reshape(-1)

    return _tpu_scan_kernel_tiled(field, fy, fx, signs, cos_t, sin_t, mirror)


__all__ = [
    "scan_statistics",
    "have_tpu",
    "tpu_devices",
    "tpu_synchronize",
    "MIN_WORK",
]
