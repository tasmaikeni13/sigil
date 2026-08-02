#!/usr/bin/env python3
"""Check the fused scan kernel against the tensor path, then time both.

A hand-written kernel that is merely fast is worthless: the statistic it
computes is the number the whole false-positive argument is about, so it has to
agree with the reference to floating-point noise before any speed figure means
anything.  This script establishes that first and reports timings second, at the
shapes the detector actually issues.

Agreement is judged on the statistic itself, which is a normalised correlation
in roughly [-1, 1]; both paths accumulate in float32 but in a different order,
so exact equality is not expected and not required.  What matters is that the
difference is far below the gap between a detection and a miss.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import kernels as K
from sigil.backend import Scanner, transform_bank

#: (name, keys, carriers, rotations, scales) — the first two rows are the
#: proposal and refine stages of a real detection.
SHAPES = [
    ("proposal", 64, 512, 13, 260),
    ("refine", 8, 4096, 13, 260),
    ("small", 4, 256, 5, 40),
]


def reference(sc: Scanner, fy, fx, signs):
    """The tensor path, with the fused route disabled."""
    torch = sc.torch
    h, w = sc.h, sc.w
    cy, cx = h // 2, w // 2
    L, Kc = fy.shape
    fy_t = torch.as_tensor(fy.astype(np.float32), device=sc.dev) * h
    fx_t = torch.as_tensor(fx.astype(np.float32), device=sc.dev) * w
    ux = fx_t[:, None, :] * sc.mirror[None, :, :]
    uy = fy_t[:, None, :]
    ry = sc.cos_t[None, :, :] * uy - sc.sin_t[None, :, :] * ux
    rx = sc.sin_t[None, :, :] * uy + sc.cos_t[None, :, :] * ux
    gy = (ry + cy).clamp(0, h - 1.001) / max(h - 1, 1) * 2.0 - 1.0
    gx = (rx + cx).clamp(0, w - 1.001) / max(w - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx.reshape(1, -1, Kc), gy.reshape(1, -1, Kc)], dim=-1)
    z = torch.nn.functional.grid_sample(
        sc.field, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, 0].view(L, -1, Kc)
    sg = torch.as_tensor(signs.astype(np.float32), device=sc.dev)
    num = torch.einsum("lmk,lk->lm", z, sg)
    den = z.pow(2).sum(dim=2).sqrt() + 1e-12
    return num / den


def timeit(fn, torch, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--tol",
        type=float,
        default=2e-3,
        help="max allowed absolute difference in the statistic",
    )
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    import torch

    torch.set_grad_enabled(False)
    if not K.have_triton():
        print("triton unavailable — fused path disabled, nothing to check")
        return 0

    rng = np.random.default_rng(0)
    n = args.canon
    excess = rng.standard_normal((n, n)).astype(np.float64)

    print(f"field {n}x{n} on {args.device}\n")
    print(
        f"{'shape':10s} {'L':>4s} {'K':>5s} {'M':>6s} "
        f"{'maxdiff':>9s} {'tensor ms':>10s} {'fused ms':>9s} {'speedup':>8s}"
    )
    worst = 0.0
    for name, L, Kc, nrot, nsc in SHAPES:
        rots = np.linspace(-3, 3, nrot)
        scales = np.linspace(0.7, 1.45, nsc)
        cos_t, sin_t, mirror = transform_bank(rots, scales, [1.0, -1.0])
        sc = Scanner(excess, cos_t, sin_t, mirror, device=args.device)
        M = cos_t.size

        rad = rng.uniform(0.055, 0.240, size=(L, Kc))
        ang = rng.uniform(0, 2 * np.pi, size=(L, Kc))
        fy = (rad * np.sin(ang)).astype(np.float32)
        fx = (rad * np.cos(ang)).astype(np.float32)
        signs = rng.choice([-1.0, 1.0], size=(L, Kc)).astype(np.float32)

        fy_t = torch.as_tensor(fy, device=sc.dev)
        fx_t = torch.as_tensor(fx, device=sc.dev)
        sg_t = torch.as_tensor(signs, device=sc.dev)

        ref = reference(sc, fy, fx, signs)
        fused = K.scan_statistics(
            torch, sc.field[0, 0], fy_t, fx_t, sg_t, sc.cos_t, sc.sin_t, sc.mirror
        )
        if fused is None:
            print(f"{name:10s} fused path declined (below MIN_WORK)")
            continue
        diff = float((fused - ref).abs().max().item())
        worst = max(worst, diff)

        scanner = sc
        t_ref = timeit(
            lambda: reference(scanner, fy, fx, signs), torch, iters=args.iters
        )
        t_fus = timeit(
            lambda: K.scan_statistics(
                torch,
                scanner.field[0, 0],
                fy_t,
                fx_t,
                sg_t,
                scanner.cos_t,
                scanner.sin_t,
                scanner.mirror,
            ),
            torch,
            iters=args.iters,
        )
        print(
            f"{name:10s} {L:4d} {Kc:5d} {M:6d} {diff:9.2e} "
            f"{t_ref * 1e3:10.2f} {t_fus * 1e3:9.2f} {t_ref / max(t_fus, 1e-9):7.2f}x"
        )
        torch.cuda.empty_cache()

    print()
    if worst > args.tol:
        print(f"FAIL: worst disagreement {worst:.2e} exceeds tolerance {args.tol:.0e}")
        return 1
    print(
        f"PASS: fused and tensor paths agree to {worst:.2e} (tolerance {args.tol:.0e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
