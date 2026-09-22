#!/usr/bin/env python3
"""Check the Google Cloud TPU v4 scan kernel against the reference path, then benchmark.

The statistic computed by the scan kernel is the foundational quantity for the
false-positive guarantees. It must agree with the reference calculation within
numerical precision before performance metrics are evaluated. This script
establishes numerical agreement and benchmarks execution time on Google Cloud
TPU v4 hardware across detector evaluation shapes.

Agreement is judged on the statistic itself (normalised correlation in roughly
[-1, 1]); accumulation order differs between paths, so agreement is expected
within floating-point tolerance, well below the margin between detection and null.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from sigil import kernels as K
from sigil.backend import Scanner, transform_bank

#: (name, keys, carriers, rotations, scales) — small, proposal, refine.
SHAPES = [
    ("small", 4, 256, 5, 40),
    ("proposal", 16, 512, 13, 260),
    ("refine", 8, 2048, 13, 260),
]


@jax.jit
def _reference_single(field, fy, fx, signs, cos_t, sin_t, mirror):
    """Reference tensor sampling path for a key batch."""
    h, w = field.shape
    cy, cx = float(h // 2), float(w // 2)
    ymax, xmax = float(h) - 1.001, float(w) - 1.001

    fy_t = jnp.asarray(fy, dtype=jnp.float32)[:, None, :] * float(h)
    fx_t = jnp.asarray(fx, dtype=jnp.float32)[:, None, :] * float(w)
    sg_t = jnp.asarray(signs, dtype=jnp.float32)[:, None, :]

    ux = fx_t * mirror[None, :, None]
    ry = cos_t[None, :, None] * fy_t - sin_t[None, :, None] * ux
    rx = sin_t[None, :, None] * fy_t + cos_t[None, :, None] * ux

    gy = jnp.clip(ry + cy, 0.0, ymax)
    gx = jnp.clip(rx + cx, 0.0, xmax)

    y0 = jnp.floor(gy).astype(jnp.int32)
    x0 = jnp.floor(gx).astype(jnp.int32)
    ty = gy - y0
    tx = gx - x0
    y1 = jnp.minimum(y0 + 1, h - 1)
    x1 = jnp.minimum(x0 + 1, w - 1)

    p00 = field[y0, x0]
    p01 = field[y0, x1]
    p10 = field[y1, x0]
    p11 = field[y1, x1]

    z = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p01) + ty * ((1.0 - tx) * p10 + tx * p11)
    num = jnp.sum(z * sg_t, axis=-1)
    den = jnp.sqrt(jnp.sum(z * z, axis=-1)) + 1e-12
    return num / den


def reference_tensor(field, fy, fx, signs, cos_t, sin_t, mirror, batch_keys=2):
    """Compute reference by processing keys in small batches."""
    L = fy.shape[0]
    outs = []
    for i in range(0, L, batch_keys):
        o = _reference_single(
            field,
            fy[i : i + batch_keys],
            fx[i : i + batch_keys],
            signs[i : i + batch_keys],
            cos_t,
            sin_t,
            mirror,
        )
        outs.append(o)
    return jnp.concatenate(outs, axis=0)


def timeit(fn, warmup=1, iters=2):
    for _ in range(warmup):
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
    K.tpu_synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
    K.tpu_synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", type=int, default=512)
    ap.add_argument("--device", default="tpu")
    ap.add_argument(
        "--tol",
        type=float,
        default=2e-3,
        help="max allowed absolute difference in the statistic",
    )
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()

    if not K.have_tpu() and args.device == "tpu":
        print("TPU unavailable on this host, nothing to check")
        return 0

    rng = np.random.default_rng(0)
    n = args.canon
    excess = rng.standard_normal((n, n)).astype(np.float64)

    print(f"Resynchronisation field {n}x{n} on Google Cloud TPU v4 ({args.device})\n")
    print(
        f"{'shape':10s} {'L':>4s} {'K':>5s} {'M':>6s} "
        f"{'maxdiff':>9s} {'ref ms':>10s} {'fused ms':>9s} {'speedup':>8s}"
    )
    worst = 0.0
    for name, L, Kc, nrot, nsc in SHAPES:
        rots = np.linspace(-3, 3, nrot)
        scales = np.linspace(0.7, 1.45, nsc)
        cos_t, sin_t, mirror = transform_bank(rots, scales, [1.0, -1.0])
        sc_tpu = Scanner(excess, cos_t, sin_t, mirror, device=args.device)
        M = cos_t.size

        rad = rng.uniform(0.055, 0.240, size=(L, Kc))
        ang = rng.uniform(0, 2 * np.pi, size=(L, Kc))
        fy = (rad * np.sin(ang)).astype(np.float32)
        fx = (rad * np.cos(ang)).astype(np.float32)
        signs = rng.choice([-1.0, 1.0], size=(L, Kc)).astype(np.float32)

        ref = reference_tensor(
            sc_tpu.field,
            fy,
            fx,
            signs,
            sc_tpu.cos_t,
            sc_tpu.sin_t,
            sc_tpu.mirror,
        ).block_until_ready()
        ref_np = np.asarray(ref, dtype=np.float64)

        fused = K.scan_statistics(
            sc_tpu.field,
            fy,
            fx,
            signs,
            sc_tpu.cos_t,
            sc_tpu.sin_t,
            sc_tpu.mirror,
        )
        if fused is None:
            print(f"{name:10s} fused path declined (below MIN_WORK)")
            continue
        if hasattr(fused, "block_until_ready"):
            fused.block_until_ready()
        fused_np = np.asarray(fused, dtype=np.float64)
        diff = float(np.max(np.abs(fused_np - ref_np)))
        worst = max(worst, diff)

        t_fus = timeit(
            lambda: K.scan_statistics(
                sc_tpu.field,
                fy,
                fx,
                signs,
                sc_tpu.cos_t,
                sc_tpu.sin_t,
                sc_tpu.mirror,
            ),
            iters=args.iters,
        )
        t_ref = timeit(
            lambda: reference_tensor(
                sc_tpu.field,
                fy,
                fx,
                signs,
                sc_tpu.cos_t,
                sc_tpu.sin_t,
                sc_tpu.mirror,
            ),
            iters=args.iters,
        )

        print(
            f"{name:10s} {L:4d} {Kc:5d} {M:6d} {diff:9.2e} "
            f"{t_ref * 1e3:10.2f} {t_fus * 1e3:9.2f} {t_ref / max(t_fus, 1e-9):7.2f}x"
        )

    print()
    if worst > args.tol:
        print(f"FAIL: worst disagreement {worst:.2e} exceeds tolerance {args.tol:.0e}")
        return 1
    print(
        f"PASS: fused TPU and reference paths agree to {worst:.2e} (tolerance {args.tol:.0e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
