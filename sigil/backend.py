"""Compute backend for the analytic stratum's resynchronisation scan.

The scan is the detector's whole cost: for every (content-seed, rotation, scale,
reflection) hypothesis it resamples one normalised spectrum at a few thousand
carrier coordinates and correlates the result with that seed's chip signs. That
is a gather followed by a matrix-vector product, which Google Cloud TPU v4 does about two
orders of magnitude faster than NumPy fancy indexing.

:class:`Scanner` keeps the spectrum and the transform bank resident on the
TPU device and returns only the per-hypothesis statistic — a few hundred floats —
per seed candidate. The full evidence matrix is fetched once, for the winning
hypothesis only; materialising it for every candidate would move hundreds of
megabytes across the bus and dominate the runtime.

The TPU and NumPy paths compute the same quantity, so a TPU is an optimisation
and never a requirement.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from . import kernels as _kernels

Array = np.ndarray
_EPS = 1e-12

_JAX = None
_JNP = None
_TPU_OK: Optional[bool] = None


def _get_jax():
    global _JAX, _JNP, _TPU_OK
    if _TPU_OK is None:
        try:
            import jax
            import jax.numpy as jnp

            _JAX = jax
            _JNP = jnp
            _TPU_OK = _kernels.have_tpu()
        except Exception:  # pragma: no cover
            _JAX, _JNP, _TPU_OK = None, None, False
    return (_JAX, _JNP) if _TPU_OK else (None, None)


def tpu_available() -> bool:
    """Return True if Google Cloud TPU hardware is available for execution."""
    _get_jax()
    return bool(_TPU_OK)


def transform_bank(
    rotations_deg,
    scales,
    reflections,
) -> Tuple[Array, Array, Array]:
    """Flatten a (rotation, scale, reflection) grid into three coefficient vectors.

    Ordering is rotation-major, then scale, then reflection, so a flat index
    decomposes back into its three components by integer division.
    """
    th = np.radians(np.asarray(rotations_deg, dtype=np.float64))[:, None, None]
    sc = np.asarray(scales, dtype=np.float64)[None, :, None]
    rf = np.asarray(reflections, dtype=np.float64)[None, None, :]
    ones = np.ones_like(th) * np.ones_like(sc) * np.ones_like(rf)
    cos_t = (np.cos(th) * sc * np.ones_like(rf)).reshape(-1)
    sin_t = (np.sin(th) * sc * np.ones_like(rf)).reshape(-1)
    mirror = (ones * rf).reshape(-1)
    return cos_t, sin_t, mirror


class Scanner:
    """Evaluate the detector statistic over a transform bank, for many seeds."""

    def __init__(
        self,
        excess: Array,
        cos_t: Array,
        sin_t: Array,
        mirror: Array,
        device: Optional[str] = "tpu",
    ):
        self.h, self.w = excess.shape
        self.n_tf = int(cos_t.size)
        jax, jnp = _get_jax() if device != "cpu" else (None, None)
        self.jax = jax
        self.jnp = jnp

        if jax is None:
            self.excess = np.ascontiguousarray(excess, dtype=np.float64)
            self.cos_t = np.asarray(cos_t, dtype=np.float64)
            self.sin_t = np.asarray(sin_t, dtype=np.float64)
            self.mirror = np.asarray(mirror, dtype=np.float64)
            self.dev = None
        else:
            self.dev = "tpu"
            self.field = jnp.asarray(excess, dtype=jnp.float32)
            self.cos_t = jnp.asarray(cos_t, dtype=jnp.float32)
            self.sin_t = jnp.asarray(sin_t, dtype=jnp.float32)
            self.mirror = jnp.asarray(mirror, dtype=jnp.float32)

    # -- internals ---------------------------------------------------------

    def _sample_np(self, fy: Array, fx: Array) -> Array:
        h, w = self.h, self.w
        cy, cx = h // 2, w // 2
        uy = (np.asarray(fy) * h)[None, :]
        ux = (np.asarray(fx) * w)[None, :] * self.mirror[:, None]
        ry = self.cos_t[:, None] * uy - self.sin_t[:, None] * ux
        rx = self.sin_t[:, None] * uy + self.cos_t[:, None] * ux
        gy = np.clip(ry + cy, 0.0, h - 1.001)
        gx = np.clip(rx + cx, 0.0, w - 1.001)
        y0 = gy.astype(np.int64)
        x0 = gx.astype(np.int64)
        ty, tx = gy - y0, gx - x0
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        flat = self.excess.ravel()
        return (
            (1 - ty) * (1 - tx) * flat[y0 * w + x0]
            + (1 - ty) * tx * flat[y0 * w + x1]
            + ty * (1 - tx) * flat[y1 * w + x0]
            + ty * tx * flat[y1 * w + x1]
        )

    def _sample_tpu(self, fy: Array, fx: Array, rows=None) -> Any:
        jnp = self.jnp
        h, w = self.h, self.w
        cy, cx = float(h // 2), float(w // 2)
        ymax, xmax = float(h) - 1.001, float(w) - 1.001

        cos_t, sin_t, mirror = self.cos_t, self.sin_t, self.mirror
        if rows is not None:
            cos_t = cos_t[rows]
            sin_t = sin_t[rows]
            mirror = mirror[rows]

        fy_t = jnp.asarray(fy, dtype=jnp.float32)[None, :] * float(h)
        fx_t = jnp.asarray(fx, dtype=jnp.float32)[None, :] * float(w)

        ux = fx_t * mirror[:, None]
        ry = cos_t[:, None] * fy_t - sin_t[:, None] * ux
        rx = sin_t[:, None] * fy_t + cos_t[:, None] * ux

        gy = jnp.clip(ry + cy, 0.0, ymax)
        gx = jnp.clip(rx + cx, 0.0, xmax)

        y0 = jnp.floor(gy).astype(jnp.int32)
        x0 = jnp.floor(gx).astype(jnp.int32)
        ty = gy - y0
        tx = gx - x0
        y1 = jnp.minimum(y0 + 1, h - 1)
        x1 = jnp.minimum(x0 + 1, w - 1)

        p00 = self.field[y0, x0]
        p01 = self.field[y0, x1]
        p10 = self.field[y1, x0]
        p11 = self.field[y1, x1]

        return (1.0 - ty) * ((1.0 - tx) * p00 + tx * p01) + ty * (
            (1.0 - tx) * p10 + tx * p11
        )

    # -- public ------------------------------------------------------------

    def statistics_multi(self, fy: Array, fx: Array, signs: Array) -> Array:
        """Statistics for many carrier sets at once: inputs are (L, K), output (L, M).

        The detector evaluates dozens of candidate anchor keys against the same
        transform bank. Issuing one kernel per key leaves the hardware underutilised;
        batching them turns the whole proposal stage into a single gather and
        fused matrix product.
        """
        fy = np.atleast_2d(np.asarray(fy))
        fx = np.atleast_2d(np.asarray(fx))
        signs = np.atleast_2d(np.asarray(signs))

        if self.jax is None:
            return np.stack(
                [self.statistics(fy[i], fx[i], signs[i]) for i in range(fy.shape[0])]
            )

        # Fused TPU kernel path
        fused = _kernels.scan_statistics(
            self.field,
            fy,
            fx,
            signs,
            self.cos_t,
            self.sin_t,
            self.mirror,
        )
        if fused is not None:
            return np.asarray(fused, dtype=np.float64)

        # Vectorized JAX path on TPU
        jnp = self.jnp
        h, w = self.h, self.w
        cy, cx = float(h // 2), float(w // 2)
        ymax, xmax = float(h) - 1.001, float(w) - 1.001

        fy_t = jnp.asarray(fy, dtype=jnp.float32)[:, None, :] * float(h)
        fx_t = jnp.asarray(fx, dtype=jnp.float32)[:, None, :] * float(w)
        sg_t = jnp.asarray(signs, dtype=jnp.float32)[:, None, :]

        ux = fx_t * self.mirror[None, :, None]
        ry = self.cos_t[None, :, None] * fy_t - self.sin_t[None, :, None] * ux
        rx = self.sin_t[None, :, None] * fy_t + self.cos_t[None, :, None] * ux

        gy = jnp.clip(ry + cy, 0.0, ymax)
        gx = jnp.clip(rx + cx, 0.0, xmax)

        y0 = jnp.floor(gy).astype(jnp.int32)
        x0 = jnp.floor(gx).astype(jnp.int32)
        ty = gy - y0
        tx = gx - x0
        y1 = jnp.minimum(y0 + 1, h - 1)
        x1 = jnp.minimum(x0 + 1, w - 1)

        p00 = self.field[y0, x0]
        p01 = self.field[y0, x1]
        p10 = self.field[y1, x0]
        p11 = self.field[y1, x1]

        z = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p01) + ty * (
            (1.0 - tx) * p10 + tx * p11
        )
        num = jnp.sum(z * sg_t, axis=-1)
        den = jnp.sqrt(jnp.sum(z * z, axis=-1)) + _EPS
        return np.asarray(num / den, dtype=np.float64)

    def statistics(self, fy: Array, fx: Array, signs: Array) -> Array:
        """Normalised Rademacher statistic for every hypothesis in the bank."""
        if self.jax is None:
            z = self._sample_np(fy, fx)
            num = z @ np.asarray(signs, dtype=np.float64)
            den = np.sqrt(np.einsum("ij,ij->i", z, z)) + _EPS
            return num / den

        jnp = self.jnp
        z = self._sample_tpu(fy, fx)
        sg = jnp.asarray(signs, dtype=jnp.float32)
        num = jnp.matmul(z, sg)
        den = jnp.sqrt(jnp.sum(z * z, axis=-1)) + _EPS
        return np.asarray(num / den, dtype=np.float64)

    def evidence(self, fy: Array, fx: Array, index: int) -> Array:
        """Per-carrier evidence vector for one hypothesis."""
        if self.jax is None:
            return self._sample_np(fy, fx)[index]
        rows = np.array([int(index)])
        return np.asarray(self._sample_tpu(fy, fx, rows=rows)[0], dtype=np.float64)


__all__ = ["Scanner", "tpu_available", "transform_bank"]
