"""A fused resynchronisation kernel.

The scan in :mod:`sigil.backend` is a gather immediately followed by two
reductions over the gathered axis::

    z        = bilinear(field, R_m · carrier_k)        # (L, M, K)
    num[l,m] = Σ_k z[l,m,k] · sign[l,k]
    den[l,m] = Σ_k z[l,m,k]²

Written with ``grid_sample`` plus ``einsum``, ``z`` is a real tensor: for the
sizes the detector actually uses — a few dozen candidate keys, a few thousand
transform hypotheses, a few hundred carriers — it is hundreds of megabytes,
written once and read twice, purely to be summed away.  The arithmetic is a
handful of flops per element, so the whole stage runs at the speed of that
traffic and nothing else.

Fusing removes it.  Each program owns one ``(key, hypothesis)`` pair, walks the
carriers in blocks, and keeps both partial sums in registers, so the only global
traffic is the field itself — about a megabyte, resident in L2 — and the small
output.

The kernel is an optimisation, never a requirement: :func:`scan_statistics`
falls back to the tensor path whenever Triton is unavailable, the problem is too
small to be worth a launch, or the device is not CUDA.  Both paths compute the
same quantity, which :mod:`SIGIL.scripts.bench_kernels` checks numerically.
"""

from __future__ import annotations

from typing import Optional

_EPS = 1e-12

try:  # pragma: no cover - depends on the installed stack
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _scan_kernel(
        FIELD,
        FY,
        FX,
        SG,
        COS,
        SIN,
        MIR,
        OUT,
        H,
        W,
        K,
        M,
        Hf,
        Wf,
        stride_l,
        stride_k,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        key_index = pid // M
        m = pid % M

        c = tl.load(COS + m)
        s = tl.load(SIN + m)
        mir = tl.load(MIR + m)
        cy = (H // 2).to(tl.float32)
        cx = (W // 2).to(tl.float32)
        # ``grid_sample`` with align_corners=True inverts the caller's own
        # normalisation exactly, so sampling at these pixel coordinates is the
        # same operation and not an approximation of it.
        ymax = Hf - 1.001
        xmax = Wf - 1.001

        num = tl.zeros((), dtype=tl.float32)
        den = tl.zeros((), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            base = key_index * stride_l + offs * stride_k
            uy = tl.load(FY + base, mask=mask, other=0.0) * Hf
            ux = tl.load(FX + base, mask=mask, other=0.0) * Wf * mir
            sg = tl.load(SG + base, mask=mask, other=0.0)

            ry = c * uy - s * ux
            rx = s * uy + c * ux
            gy = tl.minimum(tl.maximum(ry + cy, 0.0), ymax)
            gx = tl.minimum(tl.maximum(rx + cx, 0.0), xmax)

            y0 = tl.floor(gy).to(tl.int32)
            x0 = tl.floor(gx).to(tl.int32)
            ty = gy - y0.to(tl.float32)
            tx = gx - x0.to(tl.float32)
            y1 = tl.minimum(y0 + 1, H - 1)
            x1 = tl.minimum(x0 + 1, W - 1)

            r0 = y0 * W
            r1 = y1 * W
            p00 = tl.load(FIELD + r0 + x0, mask=mask, other=0.0)
            p01 = tl.load(FIELD + r0 + x1, mask=mask, other=0.0)
            p10 = tl.load(FIELD + r1 + x0, mask=mask, other=0.0)
            p11 = tl.load(FIELD + r1 + x1, mask=mask, other=0.0)

            z = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p01) + ty * (
                (1.0 - tx) * p10 + tx * p11
            )
            z = tl.where(mask, z, 0.0)

            num += tl.sum(z * sg, axis=0)
            den += tl.sum(z * z, axis=0)

        tl.store(OUT + key_index * M + m, num / (tl.sqrt(den) + 1e-12))


def have_triton() -> bool:
    return _HAVE_TRITON


#: Below this many gathered samples the launch overhead outweighs the saving.
MIN_WORK = 1 << 21


def scan_statistics(
    torch,
    field,
    fy,
    fx,
    signs,
    cos_t,
    sin_t,
    mirror,
    block_k: int = 128,
) -> Optional["object"]:
    """Fused statistic for every (key, hypothesis) pair, or ``None`` to fall back.

    ``field`` is (H, W); ``fy``, ``fx``, ``signs`` are (L, K); the transform
    vectors are (M,).  Returns an (L, M) tensor on the same device.
    """
    if not _HAVE_TRITON or not field.is_cuda:
        return None
    L, K = fy.shape
    M = cos_t.numel()
    if L * M * K < MIN_WORK:
        return None

    field = field.contiguous()
    fy = fy.contiguous()
    fx = fx.contiguous()
    signs = signs.contiguous()
    cos_t = cos_t.reshape(-1).contiguous()
    sin_t = sin_t.reshape(-1).contiguous()
    mirror = mirror.reshape(-1).contiguous()

    H, W = int(field.shape[0]), int(field.shape[1])
    out = torch.empty((L, M), dtype=torch.float32, device=field.device)
    _scan_kernel[(L * M,)](
        field,
        fy,
        fx,
        signs,
        cos_t,
        sin_t,
        mirror,
        out,
        H,
        W,
        int(K),
        int(M),
        float(H),
        float(W),
        fy.stride(0),
        fy.stride(1),
        BLOCK_K=int(block_k),
        num_warps=4,
    )
    return out


__all__ = ["scan_statistics", "have_triton", "MIN_WORK"]
