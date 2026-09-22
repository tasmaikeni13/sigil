#!/usr/bin/env python3
"""End-to-end detector timing, with the fused scan on and off.

A kernel microbenchmark answers whether one stage got faster.  The number that
matters is whether a *detection* got faster, since the scan shares the budget
with a spectrum, network passes, and hypothesis searches.  This runs real
images through the real detector both ways and reports the split, so the speed
claim in the paper is about the thing the user waits for.

Toggling is by raising the fused path's work threshold above anything the
detector will ask for, which leaves every other line of code identical.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil import kernels as K
from sigil.common import list_images, load_image
from sigil.invariant import InvariantConfig
from sigil.system import Sigil, SigilConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/photo")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--checkpoint", default="checkpoints/latent.pt")
    ap.add_argument("--latent-strength", type=float, default=0.045)
    ap.add_argument("--analytic-alpha", type=float, default=0.46)
    args = ap.parse_args()

    paths = list_images(args.corpus)[: args.limit]
    if not paths:
        print(f"no images under {args.corpus}")
        return 0
    images = [load_image(p, max_size=args.max_size) for p in paths]

    sig = Sigil(
        SigilConfig(
            invariant=InvariantConfig(alpha=args.analytic_alpha),
            latent_checkpoint=args.checkpoint,
            device=args.device,
        )
    )
    if sig.latent is not None:
        sig.latent.cfg = type(sig.latent.cfg)(
            **{**sig.latent.cfg.__dict__, "strength": args.latent_strength}
        )

    marked = [(sig.embed(im), im) for im in images]

    def run():
        stats = []
        for e, _ in marked:
            d = sig.detect(e.image, expected_nonce=e.nonce)
            stats.append(
                (d.analytic.statistic, d.latent.statistic if d.latent else 0.0)
            )
        return stats

    original = K.MIN_WORK
    results = {}
    for label, thresh in (("fused", original), ("tensor", 1 << 62)):
        K.MIN_WORK = thresh
        run()  # warm caches and JIT
        K.tpu_synchronize()
        t0 = time.perf_counter()
        stats = run()
        K.tpu_synchronize()
        results[label] = ((time.perf_counter() - t0) / len(marked), stats)
    K.MIN_WORK = original

    tf, sf = results["fused"]
    tt, st = results["tensor"]
    dA = max(abs(a[0] - b[0]) for a, b in zip(sf, st))
    dL = max(abs(a[1] - b[1]) for a, b in zip(sf, st))

    print(f"\n{len(marked)} images at max-size {args.max_size}")
    print(f"  tensor path : {tt * 1e3:8.1f} ms / detection")
    print(f"  fused  path : {tf * 1e3:8.1f} ms / detection")
    print(f"  speedup     : {tt / max(tf, 1e-9):8.2f}x")
    print(f"  largest statistic disagreement: analytic {dA:.2e}, learned {dL:.2e}")
    ok = dA < 1e-2 and dL < 1e-2
    print(
        "  PASS: the two paths detect identically"
        if ok
        else "  FAIL: the fused path changes the statistic"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
