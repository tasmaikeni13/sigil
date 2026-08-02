"""Compute backend for the analytic stratum's resynchronisation scan.

The scan is the detector's whole cost: for every (content-seed, rotation, scale,
reflection) hypothesis it resamples one normalised spectrum at a few thousand
carrier coordinates and correlates the result with that seed's chip signs.  That
is a gather followed by a matrix-vector product, which a GPU does about two
orders of magnitude faster than NumPy fancy indexing.

:class:`Scanner` keeps the spectrum and the transform bank resident on the
device and returns only the per-hypothesis statistic — a few hundred floats —
per seed candidate.  The full evidence matrix is fetched once, for the winning
hypothesis only; materialising it for every candidate would move hundreds of
megabytes across the bus and dominate the runtime.

The CUDA and NumPy paths compute the same quantity, so a GPU is an optimisation
and never a requirement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import kernels as _kernels

Array = np.ndarray
_EPS = 1e-12

_TORCH = None
_TORCH_OK: Optional[bool] = None


def _torch():
    global _TORCH, _TORCH_OK
    if _TORCH_OK is None:
        try:
            import torch  # noqa: F401

            _TORCH = torch
            _TORCH_OK = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            _TORCH, _TORCH_OK = None, False
    return _TORCH if _TORCH_OK else None


def cuda_available() -> bool:
    return _torch() is not None


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
        device: Optional[str] = None,
    ):
        self.h, self.w = excess.shape
        self.n_tf = int(cos_t.size)
        torch = _torch() if device != "cpu" else None
        self.torch = torch
        if torch is None:
            self.excess = np.ascontiguousarray(excess, dtype=np.float64)
            self.cos_t = np.asarray(cos_t, dtype=np.float64)
            self.sin_t = np.asarray(sin_t, dtype=np.float64)
            self.mirror = np.asarray(mirror, dtype=np.float64)
            self.dev = None
        else:
            self.dev = torch.device(device or "cuda:0")
            self.field = torch.as_tensor(
                np.ascontiguousarray(excess), dtype=torch.float32, device=self.dev
            )[None, None]
            self.cos_t = torch.as_tensor(cos_t, dtype=torch.float32, device=self.dev)[
                :, None
            ]
            self.sin_t = torch.as_tensor(sin_t, dtype=torch.float32, device=self.dev)[
                :, None
            ]
            self.mirror = torch.as_tensor(mirror, dtype=torch.float32, device=self.dev)[
                :, None
            ]

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

    def _sample_torch(self, fy: Array, fx: Array, rows=None):
        torch = self.torch
        h, w = self.h, self.w
        cy, cx = h // 2, w // 2
        fy_t = (
            torch.as_tensor(np.asarray(fy, dtype=np.float32), device=self.dev)[None, :]
            * h
        )
        fx_t = (
            torch.as_tensor(np.asarray(fx, dtype=np.float32), device=self.dev)[None, :]
            * w
        )
        cos_t, sin_t, mirror = self.cos_t, self.sin_t, self.mirror
        if rows is not None:
            cos_t, sin_t, mirror = cos_t[rows], sin_t[rows], mirror[rows]
        ux = fx_t * mirror
        ry = cos_t * fy_t - sin_t * ux
        rx = sin_t * fy_t + cos_t * ux
        gy = (ry + cy).clamp(0, h - 1.001) / max(h - 1, 1) * 2.0 - 1.0
        gx = (rx + cx).clamp(0, w - 1.001) / max(w - 1, 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None]
        return torch.nn.functional.grid_sample(
            self.field, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]

    # -- public ------------------------------------------------------------

    def statistics_multi(self, fy: Array, fx: Array, signs: Array) -> Array:
        """Statistics for many carrier sets at once: inputs are (L, K), output (L, M).

        The detector evaluates dozens of candidate anchor keys against the same
        transform bank.  Issuing one kernel per key leaves the GPU idle between
        launches; batching them turns the whole proposal stage into a single
        gather and a single batched matrix product.
        """
        fy = np.atleast_2d(np.asarray(fy))
        fx = np.atleast_2d(np.asarray(fx))
        signs = np.atleast_2d(np.asarray(signs))
        if self.torch is None:
            return np.stack(
                [self.statistics(fy[i], fx[i], signs[i]) for i in range(fy.shape[0])]
            )
        torch = self.torch
        L, K = fy.shape
        h, w = self.h, self.w
        cy, cx = h // 2, w // 2

        # The fused kernel computes exactly this function without ever
        # materialising the gathered tensor; it declines when the problem is
        # too small to repay a launch, and then the tensor path runs.
        sg_t = torch.as_tensor(signs.astype(np.float32), device=self.dev)
        fused = _kernels.scan_statistics(
            torch,
            self.field[0, 0],
            torch.as_tensor(fy.astype(np.float32), device=self.dev),
            torch.as_tensor(fx.astype(np.float32), device=self.dev),
            sg_t,
            self.cos_t,
            self.sin_t,
            self.mirror,
        )
        if fused is not None:
            return fused.detach().cpu().numpy().astype(np.float64)

        fy_t = torch.as_tensor(fy.astype(np.float32), device=self.dev) * h  # (L,K)
        fx_t = torch.as_tensor(fx.astype(np.float32), device=self.dev) * w
        # (L, M, K) coordinates: transform bank on the middle axis.
        ux = fx_t[:, None, :] * self.mirror[None, :, :]
        uy = fy_t[:, None, :]
        ry = self.cos_t[None, :, :] * uy - self.sin_t[None, :, :] * ux
        rx = self.sin_t[None, :, :] * uy + self.cos_t[None, :, :] * ux
        gy = (ry + cy).clamp(0, h - 1.001) / max(h - 1, 1) * 2.0 - 1.0
        gx = (rx + cx).clamp(0, w - 1.001) / max(w - 1, 1) * 2.0 - 1.0
        grid = torch.stack([gx.reshape(1, -1, K), gy.reshape(1, -1, K)], dim=-1)
        z = torch.nn.functional.grid_sample(
            self.field, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0].view(L, -1, K)
        sg = torch.as_tensor(signs.astype(np.float32), device=self.dev)
        num = torch.einsum("lmk,lk->lm", z, sg)
        den = z.pow(2).sum(dim=2).sqrt() + _EPS
        return (num / den).detach().cpu().numpy().astype(np.float64)

    def statistics(self, fy: Array, fx: Array, signs: Array) -> Array:
        """Normalised Rademacher statistic for every hypothesis in the bank."""
        if self.torch is None:
            z = self._sample_np(fy, fx)
            num = z @ np.asarray(signs, dtype=np.float64)
            den = np.sqrt(np.einsum("ij,ij->i", z, z)) + _EPS
            return num / den
        torch = self.torch
        z = self._sample_torch(fy, fx)
        sg = torch.as_tensor(np.asarray(signs, dtype=np.float32), device=self.dev)
        num = z @ sg
        den = z.pow(2).sum(dim=1).sqrt() + _EPS
        return (num / den).detach().cpu().numpy().astype(np.float64)

    def evidence(self, fy: Array, fx: Array, index: int) -> Array:
        """Per-carrier evidence vector for one hypothesis."""
        if self.torch is None:
            return self._sample_np(fy, fx)[index]
        rows = self.torch.as_tensor([int(index)], device=self.dev)
        return (
            self._sample_torch(fy, fx, rows=rows)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )


__all__ = ["Scanner", "cuda_available", "transform_bank"]
